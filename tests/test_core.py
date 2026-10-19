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


CORNERS = ["near_left", "near_right", "far_right", "far_left"]


def synth_H(name):
    return synth.render(_view(name), synth.Scene(), seed=0)[1]
