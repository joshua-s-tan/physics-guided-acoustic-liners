# -*- coding: utf-8 -*-
"""Unified statistical follow-up: overlap check, stitching without duplicates, U* substitution, wording."""
import numpy as np
import pytest

from acoustic_liner.vehicles import statistical_followup as F


def _sig(t):
    return np.sin(2 * np.pi * 100 * t)


def test_overlap_check_passes_identical_and_fails_jump():
    tp = np.linspace(0, 0.12, 1201); tc = np.linspace(0.117, 0.15, 331)
    ok, info = F.overlap_ok(tp, _sig(tp), tc, _sig(tc), 0.117, 0.12, 0.01)
    assert ok and info["n"] > 0 and info["max_diff"] < 1e-9
    ok, _ = F.overlap_ok(tp, _sig(tp), tc, _sig(tc) + 0.05, 0.117, 0.12, 0.01)
    assert not ok
    assert not F.overlap_ok(tp, _sig(tp), tc[tc > 0.13], _sig(tc[tc > 0.13]), 0.117, 0.12, 0.01)[0]


def test_stitch_removes_overlap_duplicates():
    tp = np.array([0.0, 0.1, 0.2, 0.3]); tc = np.array([0.2, 0.25, 0.3, 0.4])
    t, v = F.stitch(tp, tp * 10, tc, tc * 100, 0.2)
    assert t.tolist() == [0.0, 0.1, 0.2, 0.25, 0.3, 0.4] and np.all(np.diff(t) > 0)
    assert v.tolist() == [0.0, 1.0, 2.0, 25.0, 30.0, 40.0]


def test_shifted_row_applies_measured_change():
    assert F.shifted_row((12.0, 10.0, 40.0), (12.1, 9.0, 41.0), (12.0, 9.5, 40.0)) == pytest.approx((11.9, 10.5, 39.0))


def test_render_claim_discipline():
    v = dict(vehicle="atlas", outcome=F.UNRESOLVED, target_db=0.9213, interval=None, reasons=["x"],
             evidence=dict(targeted=None), reported_not_verdict={}, wording=F.WORDING, protocol=F.PROTOCOL)
    md = F.render(v)
    assert "post-result" in md and "not preregistered" in md and "normalised by |Z|" in md
    low = md.lower().replace("not experimentally validated", "").replace("not flight-qualified", "")
    assert "experimentally validated" not in low and "flight-qualified" not in low


def test_frozen_records_listed():
    assert len(F.FROZEN) == 3 and all(len(h) == 64 for h in F.FROZEN.values())
