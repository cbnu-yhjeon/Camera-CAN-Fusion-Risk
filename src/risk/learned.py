"""학습 기반 위험도 — XGBoost 3-클래스 분류 + SHAP 해석.

특징 (README 설계):
    z, v_rel, ttc, x_lat              — 영상 + 융합 추정
    v_ego, yaw_rate, brake_on         — CAN
TTC 는 ∞ 가 있으므로 ttc_cap 으로 자른다. 희소한 DANGER 를 놓치지 않도록 클래스 빈도의
역수로 표본 가중을 준다. 조기 종료는 val split, 최종 수치는 test split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VISION_FEATURES = ["z", "v_rel", "ttc", "x_lat"]
CAN_FEATURES = ["v_ego", "yaw_rate", "brake_on"]
TTC_CAP = 20.0


def make_features(df: pd.DataFrame, src: str, with_can: bool = True) -> pd.DataFrame:
    X = pd.DataFrame({
        "z": df[f"z_{src}"],
        "v_rel": -df[f"zdot_{src}"],
        "ttc": np.minimum(df[f"ttc_{src}"], TTC_CAP),
        "x_lat": df["x_lat"],
    }, index=df.index)
    if with_can:
        X["v_ego"] = df["v_ego"]
        X["yaw_rate"] = df["yaw_rate"]
        X["brake_on"] = df["brake_on"].astype(float)
    return X


def balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=3).astype(float)
    w = len(y) / (3 * np.maximum(counts, 1))
    return w[y]


def train_xgb(X_tr, y_tr, X_va, y_va, params: dict, seed: int = 0):
    import xgboost as xgb
    model = xgb.XGBClassifier(objective="multi:softprob", num_class=3, tree_method="hist",
                              eval_metric="mlogloss", early_stopping_rounds=40, random_state=seed,
                              n_jobs=8, **params)
    model.fit(X_tr, y_tr, sample_weight=balanced_weights(y_tr),
              eval_set=[(X_va, y_va)], sample_weight_eval_set=[balanced_weights(y_va)], verbose=False)
    return model


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """TreeSHAP — (n, features, classes).

    xgboost 의 pred_contribs 로 계산한다 (shap.TreeExplainer 와 같은 알고리즘). shap 0.49 는
    xgboost 3.x 다중 클래스 모델의 클래스별 base_score 를 읽지 못한다.
    """
    import xgboost as xgb
    c = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)   # (n, classes, f+1)
    return np.transpose(c[:, :, :-1], (0, 2, 1))
