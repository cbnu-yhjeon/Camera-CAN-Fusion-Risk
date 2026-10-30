import numpy as np
import pytest

from src.data.geometry import box_corners, global_to_cam, project, quat_to_rot, quat_yaw
from src.vision.distance import Camera

# nuScenes CAM_FRONT 와 같은 꼴: 광축이 ego +x, 카메라 x = ego −y, 카메라 y = ego −z
CAM_Q = [0.5, -0.5, 0.5, -0.5]
K = [[1266.0, 0, 800.0], [0, 1266.0, 450.0], [0, 0, 1]]


def cam():
    return Camera(K, [1.7, 0.0, 1.5], CAM_Q, 1600, 900)


def test_cam_axes():
    R = quat_to_rot(CAM_Q)                      # camera → ego
    np.testing.assert_allclose(R @ [0, 0, 1], [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(R @ [1, 0, 0], [0, -1, 0], atol=1e-9)
    np.testing.assert_allclose(R @ [0, 1, 0], [0, 0, -1], atol=1e-9)


def test_quat_yaw():
    q = [np.cos(0.3), 0, 0, np.sin(0.3)]       # z 축 0.6 rad
    assert quat_yaw(q) == pytest.approx(0.6)


def test_ground_distance_roundtrip():
    c = cam()
    # 카메라 앞 20 m, 지면 위 점 (ego 좌표 x = 21.7, z = 0)
    p_cam = global_to_cam(np.array([[21.7, 0.0, 0.0]]), np.zeros(3), [1, 0, 0, 0], c.t, CAM_Q)
    u, v = project(p_cam, c.K)[0]
    assert c.ground_distance(u, v) == pytest.approx(20.0, rel=1e-6)
    assert np.isnan(c.ground_distance(800.0, 300.0))        # 지평선 위


def test_height_distance():
    c = cam()
    assert c.height_distance(1266.0 * 1.7 / 25.0, 1.7) == pytest.approx(25.0)


def test_lateral():
    c = cam()
    assert c.lateral(800 + 1266.0 * 2 / 10, 10.0) == pytest.approx(2.0)


def test_box_corners_extent():
    corners = box_corners([0, 0, 0], [2.0, 4.0, 1.5], [1, 0, 0, 0])
    np.testing.assert_allclose(corners.max(0), [2.0, 1.0, 0.75])
