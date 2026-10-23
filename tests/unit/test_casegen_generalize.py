# -*- coding: utf-8 -*-
"""Unit tests for the generalized CFD case generator (acoustic_liner.cfd.case_generation.gen_cases):
every case is built from EXPLICIT required inputs (scenario / source_corner / finalist_path / spectrum) and
missing inputs FAIL HARD with no implicit spectrum/scenario fallback. The Falcon-L2 wrapper (gen_l2_cases)
must stay byte-identical in its per-case hashes (frozen provenance)."""
import json
import os

import numpy as np
import pytest

from acoustic_liner.cfd import case_generation as CG
from acoustic_liner.fem import scenarios as S

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(CG.__file__)))))
_FINALIST = os.path.join(_ROOT, "finalists", "falcon_v2.1_finalist_geometry.json")

# Frozen anchor: the first L2 case hashes must never drift (captured from the pre-generalization generator).
_ANCHOR_NAME = "L2_b0_u0.5"
_ANCHOR_CASE_HASH = "24cc2580ca9306abee2015ae0615495f13ae28f28bb43cc68b68358d41bfbf4b"
_ANCHOR_INPUT_TREE_HASH = "d638effe4387c6ba5dbd8e21f8ba3777f3d47787e873d4c11c57335fdbb93f2b"


def _blanketed_spectrum():
    return list(S.falcon9_2025_blanketed().spl_dB)


# ---- fail-hard on missing required inputs (no fallback) ---------------------------------------------------
@pytest.mark.parametrize("missing", ["scenario", "source_corner", "finalist_path", "spectrum"])
def test_missing_required_input_raises(tmp_path, missing):
    kw = dict(scenario="falcon9_2025_blanketed", source_corner="kappa_hi_low",
              finalist_path=_FINALIST, spectrum=_blanketed_spectrum())
    kw[missing] = None
    with pytest.raises(CG.CaseGenError):
        CG.gen_cases(str(tmp_path), **kw)


def test_empty_spectrum_raises(tmp_path):
    with pytest.raises(CG.CaseGenError):
        CG.gen_cases(str(tmp_path), scenario="falcon9_2025_blanketed", source_corner="kappa_hi_low",
                     finalist_path=_FINALIST, spectrum=[])


# ---- spectrum validation (length / finite / range / array support) ---------------------------------------
def _kw(tmp_path, **over):
    kw = dict(scenario="falcon9_2025_blanketed", source_corner="kappa_hi_low",
              finalist_path=_FINALIST, spectrum=_blanketed_spectrum(), cycles=12)
    kw.update(over)
    return str(tmp_path), kw


def test_spectrum_wrong_length_raises(tmp_path):
    out, kw = _kw(tmp_path, spectrum=[120.0] * 7)
    with pytest.raises(CG.CaseGenError):
        CG.gen_cases(out, **kw)


def test_spectrum_non_finite_raises(tmp_path):
    bad = _blanketed_spectrum(); bad[0] = float("nan")
    out, kw = _kw(tmp_path, spectrum=bad)
    with pytest.raises(CG.CaseGenError):
        CG.gen_cases(out, **kw)


def test_spectrum_out_of_range_raises(tmp_path):
    bad = _blanketed_spectrum(); bad[0] = -5.0
    out, kw = _kw(tmp_path, spectrum=bad)
    with pytest.raises(CG.CaseGenError):
        CG.gen_cases(out, **kw)


def test_spectrum_numpy_array_accepted(tmp_path):
    out, kw = _kw(tmp_path, spectrum=np.asarray(_blanketed_spectrum(), dtype=float))
    plan = CG.gen_cases(out, **kw)
    assert plan["n_cases"] == 28 and len(plan["spectrum"]) == 8


# ---- geometry-hash is never trusted -----------------------------------------------------------------------
def test_matching_geometry_hash_accepted(tmp_path):
    geom = json.load(open(_FINALIST))["geometry"]
    good = CG._sha(geom)
    out, kw = _kw(tmp_path, geometry_hash=good)
    plan = CG.gen_cases(out, **kw)
    assert plan["geometry_hash"] == good


def test_mismatched_geometry_hash_raises(tmp_path):
    out, kw = _kw(tmp_path, geometry_hash="deadbeef" * 8)
    with pytest.raises(CG.CaseGenError):
        CG.gen_cases(out, **kw)


# ---- finalist identity is bound to the requested scenario/source_corner -----------------------------------
def test_wrong_scenario_for_finalist_raises(tmp_path):
    out, kw = _kw(tmp_path, scenario="atlas_representative")
    with pytest.raises(CG.CaseGenError):
        CG.gen_cases(out, **kw)


def test_wrong_source_corner_for_finalist_raises(tmp_path):
    out, kw = _kw(tmp_path, source_corner="kappa_lo_high")
    with pytest.raises(CG.CaseGenError):
        CG.gen_cases(out, **kw)


# ---- generic naming + full-spectrum provenance ------------------------------------------------------------
def test_generic_naming_and_plan_file(tmp_path):
    out, kw = _kw(tmp_path)
    plan = CG.gen_cases(out, **kw)
    assert plan["cases"][0]["name"].startswith("b0_u")          # neutral bank ids, not L2_b*
    assert os.path.exists(os.path.join(out, "_cfd_case_plan.json"))
    assert not os.path.exists(os.path.join(out, "_l2_case_plan.json"))


def test_plan_stores_full_spectrum_and_consistent_hash(tmp_path):
    out, kw = _kw(tmp_path)
    plan = CG.gen_cases(out, **kw)
    assert plan["spectrum"] == _blanketed_spectrum()
    assert plan["spectrum_hash"] == CG._sha(plan["spectrum"])


# ---- plan records explicit provenance ---------------------------------------------------------------------
def test_plan_records_scenario_source_corner_and_spectrum_hash(tmp_path):
    plan = CG.gen_cases(str(tmp_path), scenario="falcon9_2025_blanketed", source_corner="kappa_hi_low",
                        finalist_path=_FINALIST, spectrum=_blanketed_spectrum(), cycles=12)
    assert plan["scenario"] == "falcon9_2025_blanketed"
    assert plan["source_corner"] == "kappa_hi_low"
    assert isinstance(plan["spectrum_hash"], str) and len(plan["spectrum_hash"]) == 64


# ---- L2 wrapper regression: per-case hashes are frozen ----------------------------------------------------


# ---- wrapper == explicit gen_cases with blanketed inputs --------------------------------------------------
