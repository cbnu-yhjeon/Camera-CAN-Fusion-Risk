import numpy as np
import pandas as pd
import pytest

from src.events.near_miss import brake_trigger, match_events, min_ttc_in_path, near_trigger, to_events
from src.risk.labels import gt_ttc, risk_label
from src.risk.rule_based import CAUTION, DANGER, SAFE, v2_level, v2_score


def test_brake_trigger_onset_only_while_moving():
    f = pd.DataFrame({"brake_on": [0, 1, 1, 0, 1, 1, 0, 0], "v_ego": [5, 5, 4, 4, 0.5, 0, 6, 6],
                      "a_ego": [0, -1, -1, 0, -1, 0, 0, -3.5]})
    assert list(brake_trigger(f, -3.0)) == [False, True, False, False, False, False, False, True]


def test_events_merge_and_match():
    t = np.arange(0, 10, 0.5)
    flags = np.isin(t, [1.0, 1.5, 2.0, 6.0])
    ev = to_events(t, flags, merge_gap_s=1.0)
    assert ev == [(1.0, 2.0), (6.0, 6.0)]
    c, r = match_events(ev, [(2.8, 3.0), (9.0, 9.5)], tol_s=1.0)
    assert list(c) == [True, False] and list(r) == [True, False]


def test_near_trigger_window():
    assert list(near_trigger(np.array([0.0, 2.0, 5.0]), np.array([1.5]), 1.0)) == [False, True, False]


def test_min_ttc_in_path_ignores_out_of_corridor_and_young_tracks():
    rows = pd.DataFrame({"frame_idx": [0, 0, 0, 1], "tid": [1, 2, 3, 1], "ttc": [3.0, 0.5, 1.0, 2.5],
                         "x": [0.5, 3.0, 0.2, 0.5], "track_age": [5, 5, 0, 6]})
    s = min_ttc_in_path(rows, "ttc", "x", corridor=2.0)
    assert list(s.ttc_min) == [3.0, 2.5]


def test_gt_ttc_and_labels():
    t = gt_ttc([10.0, 10.0, 10.0], [-5.0, 1.0, np.nan])
    assert t[0] == pytest.approx(2.0) and np.isinf(t[1]) and np.isnan(t[2])
    lab = risk_label([1.0, 3.0, 6.0, 1.0], [0.0, 0.5, 0.0, 5.0], 2.0, 2.0, 4.0)
    assert list(lab) == [DANGER, CAUTION, SAFE, SAFE]


def test_v2_rule_ttc_weight_raises_score():
    kw = dict(distance_m=15.0, cx=800, cy=450, img_w=1600, img_h=900, visibility=1.0, in_lane=True)
    s_far, s_near = v2_score(ttc=np.inf, **kw), v2_score(ttc=0.5, **kw)
    assert s_near > s_far
    assert float(s_far) == pytest.approx((0.45 * (1 - 10 / 45) + 0.45) * 1.1)
    assert list(v2_level([0.2, 0.4, 0.7])) == [SAFE, CAUTION, DANGER]
