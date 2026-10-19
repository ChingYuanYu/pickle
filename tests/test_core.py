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
