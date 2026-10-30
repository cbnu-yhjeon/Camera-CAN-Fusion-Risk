import math

import numpy as np
import pytest

from src.fusion.estimators import run_imm, run_kf, ttc_area, zdot_difference
from src.fusion.kalman import DepthKF, ttc_from_state

IMM = {"p_stay": 0.97, "mu_stationary0": 0.6, "pseudo_rel": 0.05, "pseudo_floor": 0.3,
       "sigma_a_stationary": 0.3}


def synth(z0=40.0, zdot=-8.0, n=60, dt=1 / 12, noise=0.08, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    z = z0 + zdot * t
    zm = z * (1 + noise * rng.standard_normal(n))
    return t, z, zm, np.maximum(noise * zm, 0.3)


def test_ttc_from_state():
    assert ttc_from_state(20.0, -5.0) == pytest.approx(4.0)
    assert math.isinf(ttc_from_state(20.0, 1.0))            # 멀어짐
    assert math.isinf(ttc_from_state(20.0, -0.05))          # eps 이하
    assert math.isinf(ttc_from_state(200.0, -5.0, max_ttc=20))


def test_kf_converges_to_true_rate():
    t, z, zm, s = synth()
    nan = np.full_like(zm, np.nan)
    Z, V, _ = run_kf(t, zm, s, nan, nan, np.zeros_like(t), 1.0, True)
    assert V[-1] == pytest.approx(-8.0, abs=1.0)
    assert abs(Z[-1] - z[-1]) < 2.0


def test_kf_accel_input_sign():
    # 자차가 2 m/s² 로 가속 → 정지 물체까지 Ż 는 점점 더 음수가 된다
    kf = DepthKF(50.0, 0.5, 0.1, zdot0=-10.0)
    kf.predict(1.0, u=2.0, sigma_a=0.0)
    assert kf.v == pytest.approx(-12.0)
    assert kf.z == pytest.approx(50 - 10 - 1.0)


def test_kf_gate_rejects_outlier_then_resets():
    kf = DepthKF(30.0, 0.5, 0.5)
    assert kf.update(30.2, 0.5)
    assert not kf.update(60.0, 0.5)
    assert not kf.update(60.0, 0.5)
    kf.update(60.0, 0.5)                                    # 3번째 → 재초기화
    assert kf.z == pytest.approx(60.0)


def test_imm_prefers_stationary_for_parked_object():
    v_ego = 8.0
    t, z, zm, s = synth(zdot=-v_ego, n=36)
    nan = np.full_like(zm, np.nan)
    Z, V, _, P = run_imm(t, zm, s, nan, nan, np.zeros_like(t), np.full_like(t, v_ego), 2.0, IMM)
    # 수렴한 등속 모델도 정지 객체를 설명하므로 P(정지)는 1 로 가지 않는다 — 사전값 위에 머물면 된다
    assert P[-1] > IMM["mu_stationary0"]
    assert V[3] == pytest.approx(-v_ego, abs=1.0)           # 시작 직후부터 맞는 Ż


def test_imm_moving_object_switches_to_cv():
    t, z, zm, s = synth(zdot=-2.0, n=72, noise=0.03)        # 앞차가 6 m/s 로 달림, 자차 8 m/s
    nan = np.full_like(zm, np.nan)
    Z, V, _, P = run_imm(t, zm, s, nan, nan, np.zeros_like(t), np.full_like(t, 8.0), 2.0, IMM)
    assert P[-1] < 0.2
    assert V[-1] == pytest.approx(-2.0, abs=1.0)


def test_ttc_area_v2_factor_two():
    # 실제 TTC 4 s 인 접근: 면적 ∝ 1/Z² 이라 v2 식은 약 2 s 를 낸다
    t = np.array([0.0, 0.1])
    z = np.array([20.0, 20.0 - 0.5])
    area = 1e4 / z ** 2
    out = ttc_area(t, area, b0_min=0.0)
    assert out[1] == pytest.approx(4.0 / 2, rel=0.05)


def test_zdot_difference_window():
    t = np.arange(20) / 12
    z = 30 - 6 * t
    d = zdot_difference(t, z, 0.5)
    assert np.isnan(d[:5]).all()
    assert d[-1] == pytest.approx(-6.0)


def test_coasting_without_measurements_goes_invalid():
    t = np.arange(24) / 12
    zm = 10 - 5 * t
    zm[6:] = np.nan                                         # 0.5 s 뒤 박스가 잘려 측정 끊김
    s = np.full_like(zm, 0.5)
    nan = np.full_like(zm, np.nan)
    Z, V, _ = run_kf(t, zm, s, nan, nan, np.zeros_like(t), 1.0, True, max_coast_s=0.5)
    assert np.isfinite(Z[11]) and np.isnan(Z[12:]).all()
    Z, V, _, _ = run_imm(t, zm, s, nan, nan, np.zeros_like(t), np.full_like(t, 5.0), 2.0, IMM, max_coast_s=0.5)
    assert np.isnan(Z[12:]).all()
