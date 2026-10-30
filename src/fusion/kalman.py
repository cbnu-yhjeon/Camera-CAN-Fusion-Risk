"""객체별 1차원 칼만 필터 — 상태 x = [Z, Ż] (카메라 깊이와 그 변화율).

운동 모델 (등가속 백색잡음):

    Z̈ = a_obj − a_ego
    x_{k+1} = F x_k + B u_k + w,   F = [[1, dt], [0, 1]],  B = [−dt²/2, −dt],  u = a_ego (CAN)
    Q = σ_a² [[dt⁴/4, dt³/2], [dt³/2, dt²]]

a_obj 는 모르므로 σ_a 로 흡수한다. CAN 이 없으면(F0) a_ego 도 모르므로 u = 0 으로 두고
그 몫을 σ_a 에 더한다 — 같은 필터에서 CAN 입력만 빼는 비교가 된다.

측정은 영상 거리 Z (핀홀 높이·지면 접촉 중 쓸 수 있는 것, 순차 갱신). σ_Z = rel·Z 로
거리에 비례한다. 혁신 제곱(NIS)이 gate 를 넘으면 그 측정을 버리고, 연속 3번 버리면
트랙이 다른 물체로 바뀐 것으로 보고 재초기화한다.
"""

from __future__ import annotations

import math


class DepthKF:
    def __init__(self, z0: float, sigma_z0: float, sigma_v0: float = 5.0, zdot0: float = 0.0,
                 gate: float = 9.0):
        self.z, self.v = z0, zdot0
        self.p00, self.p01, self.p11 = sigma_z0 ** 2, 0.0, sigma_v0 ** 2
        self.sigma_v0, self.gate = sigma_v0, gate
        self.rejects = 0

    def predict(self, dt: float, u: float, sigma_a: float):
        if dt <= 0:
            return
        self.z += self.v * dt - 0.5 * dt * dt * u
        self.v += -dt * u
        q = sigma_a * sigma_a
        p00 = self.p00 + 2 * dt * self.p01 + dt * dt * self.p11 + q * dt ** 4 / 4
        p01 = self.p01 + dt * self.p11 + q * dt ** 3 / 2
        p11 = self.p11 + q * dt * dt
        self.p00, self.p01, self.p11 = p00, p01, p11

    def update(self, zm: float, sigma: float) -> bool:
        """측정 갱신. 게이트에서 버렸으면 False."""
        r = sigma * sigma
        s = self.p00 + r
        innov = zm - self.z
        if innov * innov / s > self.gate:
            self.rejects += 1
            if self.rejects >= 3:
                self.reset(zm, sigma)
            return False
        self.rejects = 0
        k0, k1 = self.p00 / s, self.p01 / s
        self.z += k0 * innov
        self.v += k1 * innov
        p00, p01, p11 = self.p00, self.p01, self.p11
        self.p00 = (1 - k0) * p00
        self.p01 = (1 - k0) * p01
        self.p11 = p11 - k1 * p01
        return True

    def reset(self, zm: float, sigma: float):
        self.z, self.v = zm, 0.0
        self.p00, self.p01, self.p11 = sigma * sigma, 0.0, self.sigma_v0 ** 2
        self.rejects = 0

    @property
    def sigma_v(self) -> float:
        return math.sqrt(max(self.p11, 0.0))


def ttc_from_state(z: float, zdot: float, eps: float = 0.1, max_ttc: float = 20.0) -> float:
    """TTC = Z / v_rel,  v_rel = −Ż.  접근하지 않거나 max 이상이면 ∞."""
    v_rel = -zdot
    if not (z == z) or v_rel <= eps:
        return math.inf
    t = max(z, 0.0) / v_rel
    return t if t < max_ttc else math.inf


class IMMDepth:
    """2-모델 IMM — 등속(CV) 모델 + 정지 객체 모델. CAN 자차 속도를 쓰는 F1 의 핵심.

    정지 모델: 객체 속도 0 이면 Ż = −v_ego 다. 이를 매 스텝 Ż 에 대한 의사 측정
    (σ = rel·v_ego + floor) 으로 넣는다. 화면 속 객체의 약 60% 가 주차·정지 객체라
    (mini 기준) 트랙 시작 직후 Ż 를 0 에서 출발시키는 대신 곧바로 맞는 값에서 시작한다.

    모델 확률은 실제 거리 측정의 우도로만 갱신한다 (의사 측정은 모델 정의의 일부).
    μ_stationary 는 객체 상태 분류(정지 여부)에 그대로 쓸 수 있다.
    """

    def __init__(self, z0: float, sigma_z0: float, v_ego: float, sigma_v0: float = 5.0,
                 p_stay: float = 0.97, mu_stationary0: float = 0.6,
                 pseudo_rel: float = 0.05, pseudo_floor: float = 0.3, sigma_a_stationary: float = 0.3,
                 gate: float = 9.0):
        import numpy as np
        self.np = np
        # 모르는 객체의 사전: 절대 속도 0 ± σ_v0 → Ż₀ = −v_ego. CV 모델도 여기서 출발한다
        x_cv = np.array([z0, -v_ego])
        x_st = np.array([z0, -v_ego])
        P0 = np.diag([sigma_z0 ** 2, sigma_v0 ** 2])
        Pst = np.diag([sigma_z0 ** 2, (pseudo_rel * abs(v_ego) + pseudo_floor) ** 2])
        self.x = [x_cv, x_st]
        self.P = [P0, Pst]
        self.mu = np.array([1 - mu_stationary0, mu_stationary0])
        self.T = np.array([[p_stay, 1 - p_stay], [1 - p_stay, p_stay]])
        self.pseudo_rel, self.pseudo_floor = pseudo_rel, pseudo_floor
        self.sa_st, self.gate, self.sigma_v0 = sigma_a_stationary, gate, sigma_v0
        self.rejects = 0

    def step(self, dt: float, u: float, v_ego: float, sigma_a: float, meas: list[tuple[float, float]]) -> bool:
        """한 스텝. 측정을 하나라도 받아들였으면 True."""
        np = self.np
        # 1. 혼합
        c = self.T.T @ self.mu
        mix = (self.T * self.mu[:, None]) / np.maximum(c[None, :], 1e-12)     # mix[i, j] = μ_{i|j}
        xs, Ps = [], []
        for j in range(2):
            x0 = mix[0, j] * self.x[0] + mix[1, j] * self.x[1]
            P0 = sum(mix[i, j] * (self.P[i] + np.outer(self.x[i] - x0, self.x[i] - x0)) for i in range(2))
            xs.append(x0)
            Ps.append(P0)
        # 2. 모델별 예측·갱신
        F = np.array([[1.0, dt], [0.0, 1.0]])
        B = np.array([-0.5 * dt * dt, -dt])
        G = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt * dt]])
        lik = np.ones(2)
        accepted = False
        for j, sa in enumerate([sigma_a, self.sa_st]):
            x = F @ xs[j] + B * u
            P = F @ Ps[j] @ F.T + sa * sa * G
            if j == 1 and np.isfinite(v_ego):                 # 정지 가설: Ż = −v_ego
                r = (self.pseudo_rel * abs(v_ego) + self.pseudo_floor) ** 2
                s = P[1, 1] + r
                k = P[:, 1] / s
                x = x + k * (-v_ego - x[1])
                P = P - np.outer(k, P[1, :])
            for zm, sig in meas:
                s = P[0, 0] + sig * sig
                innov = zm - x[0]
                if innov * innov / s > self.gate:
                    lik[j] *= np.exp(-0.5 * self.gate) / np.sqrt(2 * np.pi * s)
                    continue
                accepted = True
                k = P[:, 0] / s
                x = x + k * innov
                P = P - np.outer(k, P[0, :])
                lik[j] *= np.exp(-0.5 * innov * innov / s) / np.sqrt(2 * np.pi * s)
            xs[j], Ps[j] = x, P
        self.x, self.P = xs, Ps
        mu = lik * c
        self.mu = mu / mu.sum() if mu.sum() > 0 else np.array([0.5, 0.5])
        # 3. 두 모델 모두 측정에서 크게 벗어나면 다른 물체로 바뀐 것 — 재초기화
        if meas:
            if not accepted:
                self.rejects += 1
                if self.rejects >= 3:
                    self.__init__(meas[0][0], meas[0][1], v_ego if np.isfinite(v_ego) else 0.0,
                                  self.sigma_v0, self.T[0, 0], 0.5, self.pseudo_rel, self.pseudo_floor,
                                  self.sa_st, self.gate)
            else:
                self.rejects = 0
        return accepted

    @property
    def estimate(self):
        return self.mu[0] * self.x[0] + self.mu[1] * self.x[1]

    @property
    def P_comb(self):
        xc = self.estimate
        return sum(self.mu[i] * (self.P[i] + self.np.outer(self.x[i] - xc, self.x[i] - xc)) for i in range(2))

    @property
    def p_stationary(self) -> float:
        return float(self.mu[1])
