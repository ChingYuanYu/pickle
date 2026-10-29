import numpy as np
import pytest

from pbcoach import court, synth
from pbcoach.calibrate import calibrate_from_points, calibrate_image
from pbcoach.camera import focal_consistency, pose_from_homography, project
from pbcoach.players import CourtTracker, Detection, OccupancyDetector
from tests.eval_calibration import court_error_m


def _view(name):
    return next(v for v in synth.VIEWS if v.name == name)


def test_court_dimensions():
    assert court.WIDTH == pytest.approx(6.096)
    assert court.LENGTH == pytest.approx(13.4112)
    assert court.NVZ_FAR_Y - court.NVZ_NEAR_Y == pytest.approx(2 * 2.1336)


def test_pose_recovered_from_true_homography():
    img, H, K, R, t = synth.render(_view("corner_high"), synth.Scene(), seed=0)
    pose = pose_from_homography(H, (img.shape[1], img.shape[0]))
    assert pose.focal_px == pytest.approx(K[0, 0], rel=0.02)
    assert np.allclose(pose.position, -R.T @ t, atol=0.05)
    assert pose.view == "corner"


def test_wrong_court_match_is_camera_inconsistent():
    img, H, *_ = synth.render(_view("corner_high"), synth.Scene(), seed=0)
    size = (img.shape[1], img.shape[0])
    # Stretch the court 50% along its length: no real camera produces that from a 20x44 ft court.
    S = np.diag([1.0, 1.5, 1.0])
    assert focal_consistency(H, size) < 0.02
    assert focal_consistency(H @ S, size) > 0.1


@pytest.mark.parametrize("name", ["baseline_high", "corner_low", "sideline_low", "overhead"])
def test_auto_calibration_across_angles(name):
    img, H_true, *_ = synth.render(_view(name), synth.Scene(), seed=0)
    cal = calibrate_image(img)
    assert cal is not None
    mean_err, max_err, _ = court_error_m(cal.H, H_true, (img.shape[1], img.shape[0]))
    assert mean_err < 0.15 and max_err < 0.5


def test_manual_calibration():
    img, H_true, *_ = synth.render(_view("baseline_high"), synth.Scene(), seed=0)
    names = ["near_left", "near_right", "far_right", "far_left"]
    pts = project(H_true, [court.KEYPOINTS[n] for n in names])
    cal = calibrate_from_points(pts, names, (img.shape[1], img.shape[0]))
    assert np.allclose(cal.to_court(pts), [court.KEYPOINTS[n] for n in names], atol=1e-6)


def test_occupancy_finds_near_players_from_corner():
    v = _view("corner_high")
    bg, *_ = synth.render(v, synth.Scene(), seed=0)
    pos = np.array([[1.7, 4.0], [4.2, 4.2]])
    scene = synth.Scene(players=[synth.Player(x, y, synth.SHIRTS[i]) for i, (x, y) in enumerate(pos)],
                        noise=2.0, blur=0, jpeg=0)
    img, *_ = synth.render(v, scene, seed=0, background=bg)
    det = OccupancyDetector(calibrate_image(bg), bg)
    found = np.array([d.court_xy for d in det(img)])
    assert len(found) == 2
    for p in pos:
        assert np.min(np.linalg.norm(found - p, axis=1)) < 0.3


def test_tracker_keeps_ids():
    tr = CourtTracker(fps=30)
    for f in range(30):
        dets = [Detection((0, 0, 1, 1), 1.0, court_xy=np.array([1.0 + 0.02 * f, 3.0])),
                Detection((0, 0, 1, 1), 1.0, court_xy=np.array([4.0, 10.0 - 0.02 * f]))]
        tracks = tr.update(dets, f)
    assert sorted(t.id for t in tracks) == [1, 2]
    assert {t.team for t in tracks} == {"near", "far"}


def _rows(positions, fps=10):
    """Tracker-style rows from {player_id: (team, [(x, y), ...])}."""
    return [{"frame": f, "t": f / fps, "player_id": pid, "team": team, "x": x, "y": y}
            for pid, (team, path) in positions.items() for f, (x, y) in enumerate(path)]


def test_coach_flags_split_partners_and_praises_spacing():
    from pbcoach import coach
    n = 60
    up = [(1.5, court.NVZ_NEAR_Y - 0.3)] * n          # at the kitchen line
    back = [(4.5, 0.5)] * n                            # stays on the baseline
    ghost = [(3.0, 3.0)] * 3                           # too short to be a player
    rows = _rows({1: ("near", up), 2: ("near", back), 9: ("near", ghost)})
    assert coach.player_ids(rows) == {1, 2}
    titles = {(i.who.split(" (")[0], i.title) for i in coach.analyze(rows, {"calibration": {"score": 1.0}}, 10)}
    assert ("Near team", "Partners split up and back") in titles
    assert ("Player 2", "Never reached the kitchen line") in titles
    assert ("Player 1", "Gets to the kitchen line") in titles


def test_far_positions_are_trusted_less_from_a_low_camera():
    from pbcoach.players import ground_covariance
    img, *_ = synth.render(_view("baseline_low"), synth.Scene(), seed=0)
    cal = calibrate_from_points(project(synth_H("baseline_low"), [court.KEYPOINTS[n] for n in CORNERS]), CORNERS,
                                (img.shape[1], img.shape[0]))
    near = ground_covariance(cal, cal.to_image([(3.0, 2.0)])[0], 2.0)
    far = ground_covariance(cal, cal.to_image([(3.0, 11.0)])[0], 2.0)
    assert np.trace(far) > 10 * np.trace(near)


def test_tracker_keeps_sides_and_reacquires_lost_player():
    tr = CourtTracker(fps=10)
    d = lambda x, y, f=None: Detection((0, 0, 1, 1), 0.9, court_xy=np.array([x, y]), cov=np.eye(2) * 0.01, feat=f)
    red, blue = np.r_[1.0, 0, 0], np.r_[0, 0, 1.0]
    for f in range(10):
        tr.update([d(1.0, 2.0, red), d(4.0, 2.0, blue), d(2.0, 11.0), d(4.0, 11.0)], f)
    ids = {round(t.pos[0]): t.id for t in tr.tracks if t.side == "near"}
    for f in range(10, 40):   # the red player is unseen for 3 s, then reappears elsewhere on the near side
        tr.update([d(4.0, 2.0, blue), d(2.0, 11.0), d(4.0, 11.0)], f)
    tracks = tr.update([d(2.5, 1.0, red), d(4.0, 2.0, blue), d(2.0, 11.0), d(4.0, 11.0), d(3.0, 3.0, red * 0 + 0.3)], 40)
    near = {t.id: t for t in tracks if t.side == "near"}
    assert set(near) == {ids[1], ids[4]}                      # no third near-side identity
    assert np.allclose(near[ids[1]].pos, (2.5, 1.0), atol=0.3)  # re-acquired by appearance
    assert len(tr.tracks) == 4


def test_lifted_pose_stands_upright_on_the_player():
    from pbcoach.players import lift_keypoints
    img, H, K, R, t = synth.render(_view("corner_high"), synth.Scene(), seed=0)
    cal = calibrate_from_points(project(H, [court.KEYPOINTS[n] for n in CORNERS]), CORNERS,
                                (img.shape[1], img.shape[0]))
    g = np.array([2.0, 4.0])
    heights = np.array([0.05, 0.5, 1.0, 1.5, 1.7])
    kp = np.c_[cal.pose.project3d(np.c_[np.tile(g, (5, 1)), heights]), np.ones(5)]
    J = lift_keypoints(cal, kp, g)
    assert np.allclose(J[:, 2], heights, atol=0.08)
    assert np.allclose(J[:, :2], g, atol=0.1)


def test_pose_smoothing_steadies_joints_without_lagging_the_player():
    from pbcoach.pose import smooth_track
    rng = np.random.default_rng(0)
    fps, n, h = 30.0, 60, 80.0
    x0 = np.linspace(100, 400, n)                      # running fast across the image: 10 px per frame
    rel = rng.uniform(-0.3, 0.3, (17, 2)) * h            # a fixed pose relative to the box
    truth = np.stack([np.c_[x0[i] + rel[:, 0], 300 + rel[:, 1]] for i in range(n)])
    kps = [np.c_[truth[i] + rng.normal(0, 2.0, (17, 2)), np.full(17, 0.9)] for i in range(n)]
    kps[30][9, :2] += 40                                  # one wrist jumps to a wrong place
    kps[40][9, 2] = 0.05                                  # and drops out for a frame
    boxes = [(x0[i] - 25, 300 - h / 2, x0[i] + 25, 300 + h / 2) for i in range(n)]
    out = smooth_track(kps, boxes, fps)
    err_raw = np.median([np.linalg.norm(k[:, :2] - t, axis=1).mean() for k, t in zip(kps, truth)])
    err = np.median([np.linalg.norm(k[:, :2] - t, axis=1).mean() for k, t in zip(out, truth)])
    assert err < 0.6 * err_raw
    assert np.linalg.norm(out[30][9, :2] - truth[30][9]) < 6     # outlier removed
    assert out[40][9, 2] >= 0.3 and np.linalg.norm(out[40][9, :2] - truth[40][9]) < 6   # dropout bridged


CORNERS = ["near_left", "near_right", "far_right", "far_left"]


def synth_H(name):
    return synth.render(_view(name), synth.Scene(), seed=0)[1]


# ---------------------------------------------------------------- tennis

def _tview(name):
    return next(v for v in synth.TENNIS_VIEWS if v.name == name)


def test_tennis_court_dimensions():
    from pbcoach.court import TENNIS
    assert TENNIS.width == pytest.approx(10.9728)
    assert TENNIS.length == pytest.approx(23.7744)
    assert TENNIS.net_y - TENNIS.keypoints["service_near_center"][1] == pytest.approx(6.4008)
    assert TENNIS.keypoints["near_singles_right"][0] - TENNIS.keypoints["near_singles_left"][0] == pytest.approx(8.2296)
    assert TENNIS.net_tape_height(TENNIS.center_x) == pytest.approx(0.9144)
    assert TENNIS.net_tape_height(-TENNIS.net_overhang) == pytest.approx(1.0668)


@pytest.mark.parametrize("name", ["broadcast_high", "corner_high", "sideline_high"])
def test_tennis_auto_calibration(name):
    from pbcoach.court import TENNIS
    img, H_true, *_ = synth.render(_tview(name), synth.Scene(court=TENNIS), seed=0)
    cal = calibrate_image(img, court=TENNIS)
    assert cal is not None and cal.court is TENNIS
    mean_err, max_err, _ = court_error_m(cal.H, H_true, (img.shape[1], img.shape[0]), TENNIS)
    assert mean_err < 0.2 and max_err < 0.6   # the court is 1.8x as long as pickleball's


def test_tennis_manual_calibration():
    from pbcoach.court import TENNIS
    img, H_true, *_ = synth.render(_tview("broadcast_high"), synth.Scene(court=TENNIS), seed=0)
    names = ["near_left", "near_right", "service_far_right", "service_far_left"]
    pts = project(H_true, [TENNIS.keypoints[n] for n in names])
    cal = calibrate_from_points(pts, names, (img.shape[1], img.shape[0]), court="tennis")
    assert np.allclose(cal.to_court([(640, 360)]), project(np.linalg.inv(H_true), [(640, 360)]), atol=1e-3)
    assert cal.summary()["sport"] == "tennis"


def test_singles_tracker_keeps_one_player_per_side():
    from pbcoach.court import TENNIS
    tr = CourtTracker(fps=10, max_players=2, net_y=TENNIS.net_y)
    d = lambda x, y: Detection((0, 0, 1, 1), 0.9, court_xy=np.array([x, y]), cov=np.eye(2) * 0.01)
    for f in range(20):   # a ball kid standing near the far baseline must not become a third player
        tracks = tr.update([d(5.0 + 0.1 * f, -0.5), d(6.0, 24.5), d(1.0, 26.0)], f)
    assert len(tr.tracks) == 2
    assert {t.team for t in tracks} == {"near", "far"}
    assert np.allclose(next(t for t in tracks if t.team == "far").pos, (6.0, 24.5), atol=0.2)


def test_tennis_coach_flags_deep_player_and_slow_recovery():
    from pbcoach import coach
    n, fps = 200, 10
    # Player 1 camps 4 m behind the baseline; player 2 is pulled wide and drifts back slowly.
    deep = [(5.5, -4.0)] * n
    wide = []
    for k in range(4):
        wide += [(9.5, 24.0)] * 10 + [(9.5 - 0.1 * i, 24.0) for i in range(40)]
    rows = _rows({1: ("near", deep), 2: ("far", wide)}, fps)
    titles = {(i.who.split(" (")[0], i.title) for i in coach.analyze(rows, {"sport": "tennis"}, fps)}
    assert ("Player 1", "Playing far behind the baseline") in titles
    assert ("Player 2", "Slow to recover to the middle") in titles
    assert ("Player 2", "Good baseline position") in titles


def test_tennis_doubles_formation_metrics():
    from pbcoach.court import TENNIS
    from pbcoach.metrics import team_metrics
    n = 20
    rows = _rows({1: ("near", [(3.0, 0.2)] * n), 2: ("near", [(8.0, 9.0)] * n),   # one up, one back
                  3: ("far", [(3.0, 23.5)] * n), 4: ("far", [(8.0, 23.5)] * n)})   # both back
    m = team_metrics(rows, TENNIS)
    assert m["near"]["share_one_up_one_back"] == 1.0
    assert m["far"]["share_both_back"] == 1.0
