# -*- coding: utf-8 -*-
"""§4.8 #10 fail-closed robust-minimum selection (production `acoustic_liner.fem.placement`). These are pure and
always run (no FEM/oracle): the selection is the load-bearing gate Step 8 inherits."""
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner.fem import placement as RP  # noqa: E402

EXPECTED = (("kappa_lo", "low"), ("kappa_lo", "high"), ("kappa_hi", "low"), ("kappa_hi", "high"))


def _rec(cid, value, ok=True):
    return dict(corner_id=cid, value=value, ok=ok)


def _full(values):
    """Four well-formed corner records with the given values (in EXPECTED order)."""
    return [_rec(c, v) for c, v in zip(EXPECTED, values)]


def test_selects_independent_worst_case_minimum():
    res = RP.select_robust_minimum(_full([6.0, 5.0, 7.0, 5.5]), EXPECTED)
    assert res["robust_value"] == 5.0
    assert res["robust_corner_id"] == ("kappa_lo", "high")
    assert res["n_corners"] == 4 and res["tie_broken"] is False
    assert set(res["per_corner"]) == {"kappa_lo|low", "kappa_lo|high", "kappa_hi|low", "kappa_hi|high"}


def test_order_invariance():
    base = RP.select_robust_minimum(_full([6.0, 5.0, 7.0, 5.5]), EXPECTED)
    recs = _full([6.0, 5.0, 7.0, 5.5])
    for seed in range(5):
        random.Random(seed).shuffle(recs)
        res = RP.select_robust_minimum(recs, EXPECTED)
        assert res["robust_value"] == base["robust_value"]
        assert res["robust_corner_id"] == base["robust_corner_id"]
        assert res["per_corner"] == base["per_corner"]


def test_deterministic_tie_break_by_canonical_id():
    # two corners tie at the minimum -> pick the smaller canonical id, flag tie_broken
    res = RP.select_robust_minimum(_full([5.0, 6.0, 5.0, 7.0]), EXPECTED)
    assert res["robust_value"] == 5.0
    # ties at 5.0: ("kappa_lo","low") and ("kappa_hi","low"); "kappa_hi|low" < "kappa_lo|low" ('h' < 'l')
    assert res["robust_corner_id"] == ("kappa_hi", "low")
    assert res["tie_broken"] is True


def test_missing_corner_fails_closed():
    with pytest.raises(RP.RobustSelectionError):
        RP.select_robust_minimum(_full([6.0, 5.0, 7.0, 5.5])[:3], EXPECTED)   # only 3 of 4


def test_duplicate_corner_fails_closed():
    recs = _full([6.0, 5.0, 7.0, 5.5]) + [_rec(("kappa_lo", "low"), 4.0)]     # duplicate id
    with pytest.raises(RP.RobustSelectionError):
        RP.select_robust_minimum(recs, EXPECTED)


def test_unexpected_corner_fails_closed():
    recs = _full([6.0, 5.0, 7.0, 5.5]) + [_rec(("kappa_mid", "low"), 4.0)]    # not pre-registered
    with pytest.raises(RP.RobustSelectionError):
        RP.select_robust_minimum(recs, EXPECTED)


def test_nonfinite_value_fails_closed():
    for bad in (float("nan"), float("inf"), True, "5.0"):
        recs = _full([6.0, 5.0, 7.0, 5.5]); recs[1] = _rec(EXPECTED[1], bad)
        with pytest.raises(RP.RobustSelectionError):
            RP.select_robust_minimum(recs, EXPECTED)


def test_failed_corner_fails_closed():
    recs = _full([6.0, 5.0, 7.0, 5.5]); recs[2] = _rec(EXPECTED[2], 7.0, ok=False)
    with pytest.raises(RP.RobustSelectionError):
        RP.select_robust_minimum(recs, EXPECTED)


def test_illformed_preregistration_and_records():
    with pytest.raises(RP.RobustSelectionError):
        RP.select_robust_minimum(_full([1, 2, 3, 4]), EXPECTED + (("kappa_lo", "low"),))   # dup in expected
    with pytest.raises(RP.RobustSelectionError):
        RP.select_robust_minimum([{"corner_id": ("kappa_lo", "low"), "value": 5.0}], (("kappa_lo", "low"),))  # no 'ok'
