import numpy as np
import pytest

from src.data.can import CanSeries, align, estimate_lag, smooth_derivative
from src.fusion.motion_state import classify


def make_can():
    t = np.arange(0, 10, 0.02)
    v = 10 - 0.5 * t                                         # 0.5 m/s² 감속
    tm = np.arange(0, 10, 0.5)
    brake = np.where(tm >= 5, 30.0, 0.0)
    us = lambda x: (x * 1e6).astype(np.int64)               # noqa: E731
    return CanSeries(us(t), v, smooth_derivative(t, v, 0.5), np.zeros_like(t),
                     us(t), np.zeros_like(t), us(tm), brake)


def test_align_interp_and_hold():
    can = make_can()
    a = align(can, np.array([2_010_000, 5_100_000, 5_490_000, 4_990_000]))
    assert a.v_ego.iloc[0] == pytest.approx(10 - 0.5 * 2.01, abs=1e-6)
    assert a.a_ego.iloc[0] == pytest.approx(-0.5, abs=1e-3)
    # 브레이크는 직전 메시지 유지 — 4.99 s 는 아직 OFF (5.0 s 메시지는 미래)
    assert list(a.brake_on) == [False, True, True, False]


def test_align_outside_range_is_missing():
    a = align(make_can(), np.array([20_000_000]))
    assert not a.can_ok.iloc[0]
    assert align(None, np.array([0, 1])).can_ok.sum() == 0


def test_estimate_lag():
    t = np.arange(0, 10, 0.05)
    v = np.sin(t)
    lag, rmse = estimate_lag(t, v, t + 0.2, v)              # CAN 이 0.2 s 늦게 찍힘
    assert lag == pytest.approx(0.2, abs=0.011)
    assert rmse < 1e-2


def test_motion_state_rules():
    v_obj = np.array([0.2, 10.0, 10.0, 10.0, np.nan])
    zdot = np.array([-10.0, 0.1, -3.0, 2.0, 0.0])
    assert list(classify(v_obj, zdot)) == ["stationary", "following", "approaching", "receding", None]
