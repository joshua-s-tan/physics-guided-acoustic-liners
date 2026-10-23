# -*- coding: utf-8 -*-
"""Geometric feasibility (§4.7): full constraint set, inactive-bank masking (the Electron L_neck>D case),
and an explicit truncated-cell rule. A well-formed but infeasible design is REPORTED, not raised."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner.design import constraints as F  # noqa: E402


def _cell_side_to_Vcav(s, D):
    # invert footprint a = s^2 = V_cav / D  ->  V_cav = s^2 * D
    return s * s * D


def _good_group(area_frac, D=0.05, r_neck=0.003, s=0.03):
    # a comfortably feasible cell: neck fits (2r+2*ligament < s), sigma in range
    return dict(f0=120.0, r_neck=r_neck, L_neck=0.02, area_frac=area_frac, cavity_volume=_cell_side_to_Vcav(s, D))


def _good_design():
    return dict(D=0.05, face_t=0.0008, wall_t=0.0006,
                groups=[_good_group(0.5), _good_group(0.5)])


def test_feasible_design_passes():
    r = F.check_feasibility(_good_design(), coverage=0.6)
    assert r["feasible"] is True and r["failures"] == []
    assert r["area_frac_sum"] == pytest.approx(1.0)
    assert r["truncated_cell_rule"] == "count_fractional_area"


def test_neck_longer_than_depth_always_infeasible():
    d = _good_design()
    d["groups"][0]["L_neck"] = 0.06                      # > D=0.05
    r = F.check_feasibility(d)
    assert r["feasible"] is False
    assert any("L_neck" in f and "> D" in f for f in r["failures"])
    # folded/protruding neck is NOT supported yet -> still infeasible, with an "unsupported" reason
    r2 = F.check_feasibility(d, F.FeasibilityLimits(allow_folded_neck=True))
    assert r2["feasible"] is False
    assert any("unsupported" in f.lower() for f in r2["failures"])


def test_negative_area_fraction_is_infeasible_not_inactive():
    d = dict(D=0.05, face_t=0.0008, wall_t=0.0006, groups=[_good_group(1.0), _good_group(-0.2)])
    r = F.check_feasibility(d)
    assert r["feasible"] is False
    assert any("area_frac" in f and "[0,1]" in f for f in r["failures"])
    assert r["inactive_banks"] == []                     # a negative fraction is NOT classified inactive


def test_nonpositive_depth_returns_structured_infeasible_not_crash():
    d = _good_design(); d["D"] = 0.0                     # must NOT raise ZeroDivisionError
    r = F.check_feasibility(d)
    assert r["feasible"] is False
    assert any("backing depth" in f for f in r["failures"])


def test_neck_does_not_fit_in_cell_rejected():
    d = _good_design()
    d["groups"][0] = _good_group(0.5, r_neck=0.02, s=0.01)   # 2r=0.04 >> cell side 0.01
    r = F.check_feasibility(d)
    assert r["feasible"] is False
    assert any("cell side" in f for f in r["failures"])


def test_area_fractions_must_sum_to_one():
    d = dict(D=0.05, face_t=0.0008, wall_t=0.0006, groups=[_good_group(0.5), _good_group(0.3)])
    r = F.check_feasibility(d)
    assert r["feasible"] is False
    assert any("area fractions sum" in f for f in r["failures"])


def test_inactive_bank_is_masked_and_canonicalized():
    """An inactive bank (area_frac~0) with a physically-impossible dormant L_neck>D must NOT fail the design;
    its geometry is masked from checks and canonicalized into a valid placeholder (the Electron CMA case)."""
    d = dict(D=0.05, face_t=0.0008, wall_t=0.0006,
             groups=[_good_group(1.0), dict(f0=250.0, r_neck=0.003, L_neck=0.9,   # L_neck 0.9 >> D
                                             area_frac=0.0, cavity_volume=-1.0)])  # also impossible V_cav
    r = F.check_feasibility(d)
    assert r["feasible"] is True                          # active bank sums to 1; inactive one is masked
    assert r["inactive_banks"] == [1]
    canon = r["canonicalized"]["groups"][1]
    assert canon["_inactive"] is True and canon["area_frac"] == 0.0
    assert 0.0 < canon["L_neck"] <= d["D"]                # canonicalized into (0, D]
    assert canon["cavity_volume"] > 0                     # impossible -1.0 fully replaced, not retained
    assert F.FeasibilityLimits().min_neck_r_m <= canon["r_neck"] <= F.FeasibilityLimits().max_neck_r_m


def test_only_implemented_truncated_rule_is_offered():
    assert F._TRUNCATED_CELL_RULES == ("count_fractional_area",)
    with pytest.raises(F.FeasibilityInputError):
        F.FeasibilityLimits(truncated_cell_rule="exclude_boundary_cells")


def test_feasibility_limits_self_validated():
    with pytest.raises(F.FeasibilityInputError):
        F.FeasibilityLimits(min_neck_r_m=0.06, max_neck_r_m=0.05)   # min !< max
    with pytest.raises(F.FeasibilityInputError):
        F.FeasibilityLimits(sigma_min=0.3, sigma_max=0.2)           # contradictory
    with pytest.raises(F.FeasibilityInputError):
        F.FeasibilityLimits(max_depth_m=float("nan"))               # NaN
    with pytest.raises(F.FeasibilityInputError):
        F.FeasibilityLimits(ligament_min_m=-0.001)                  # negative
    with pytest.raises(F.FeasibilityInputError):
        F.FeasibilityLimits(area_frac_sum_tol=1.5)                  # tol out of [0,1)


def test_require_coverage_makes_missing_coverage_infeasible():
    d = _good_design()
    assert F.check_feasibility(d)["feasible"] is True               # local-cell feasibility, no coverage
    r = F.check_feasibility(d, require_coverage=True)               # installed-design path
    assert r["feasible"] is False
    assert any("coverage is required" in f for f in r["failures"])
    assert F.check_feasibility(d, coverage=0.6, require_coverage=True)["feasible"] is True


def test_inactive_placeholder_f0_is_deterministic():
    d = dict(D=0.05, face_t=0.0008, wall_t=0.0006,
             groups=[_good_group(1.0), dict(f0=999.0, r_neck=0.003, L_neck=0.9, area_frac=0.0, cavity_volume=-1.0)])
    canon = F.check_feasibility(d)["canonicalized"]["groups"][1]
    assert canon["f0"] == 0.0                                       # NOT the dormant 999.0


def test_depth_and_thickness_limits():
    d = _good_design(); d["D"] = 0.08                     # > 50 mm
    assert F.check_feasibility(d)["feasible"] is False
    d2 = _good_design(); d2["wall_t"] = 0.02              # > max wall thickness
    assert any("wall thickness" in f for f in F.check_feasibility(d2)["failures"])


def test_coverage_bounds_and_malformed_raises():
    assert F.check_feasibility(_good_design(), coverage=1.5)["feasible"] is False
    with pytest.raises(F.FeasibilityInputError):
        F.check_feasibility({"groups": []})               # no groups
    with pytest.raises(F.FeasibilityInputError):
        F.check_feasibility({"D": 0.05, "groups": [dict(area_frac=float("nan"))]})
