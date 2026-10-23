# -*- coding: utf-8 -*-
"""Step 3 / gate G3 GLOBAL sweep: prove the entire migrated-vs-reference LABEL delta is explained SOLELY by the
mass-formula change (corrected shared-wall/BOM vs legacy v1 4*sqrt(n)+mu_A), across a broad design space --
one four-group candidate is not enough.

Sweep dimensions: coverage {0, partial, 1} x active-bank counts {1,2,4} x area-fraction {equal, skewed} x
low/high feasible cavity volumes/cell sizes x BOM allowance structure x {nominal, boundary-feasible}. For every
case: (1) candidate valid under the intended feasibility classification; (2) acoustic fields + attenuation labels
unchanged (mass-independent); (3) old (legacy, test-only) and corrected (production) mass independently
recomputed; (4) new_label_mass == corrected_mass_model_output; (5) observed label delta == corrected - legacy;
(6) coverage applied exactly once incl. exact zero-mass at zero coverage; (7) group-order permutations change
neither mass nor acoustics; (8) inactive banks contribute no phantom mass.

The legacy formula lives ONLY in tests (_legacy_mass); it is never in a production module.
"""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from acoustic_liner.physics import oracle as O2  # noqa: E402
from acoustic_liner.physics import mass as MM
from acoustic_liner.design import constraints as F
from _legacy_mass import legacy_areal_mass, legacy_total_mass           # noqa: E402

A_WALL = 165.878
KAPPA = 0.23270570707355803
SPEC = O2.falcon9_design_band()


def _enrich(groups):
    return [dict(g, cavity_volume=O2._group_cavity_volume(g)) for g in groups]


def _g(f0, r, af, L=0.03, Q=15.0):
    return dict(f0=f0, r_neck=r, L_neck=L, area_frac=af, Q=Q)


# Deterministic sweep designs (all release-feasible), covering the required dimensions.
SWEEP = {
    "one_group_nominal": dict(D=0.05, face_t=0.0008, wall_t=0.0006, groups=[_g(196.0, 0.008, 1.0)]),
    "two_group_equal": dict(D=0.05, face_t=0.0008, wall_t=0.0006, groups=[_g(196.0, 0.008, 0.5), _g(251.0, 0.006, 0.5)]),
    "two_group_skewed": dict(D=0.05, face_t=0.0008, wall_t=0.0006, groups=[_g(196.0, 0.008, 0.7), _g(251.0, 0.006, 0.3)]),
    "four_group_equal": dict(D=0.05, face_t=0.0008, wall_t=0.0006,
                             groups=[_g(126.0, 0.012, 0.25), _g(160.0, 0.010, 0.25),
                                     _g(196.0, 0.008, 0.25), _g(251.0, 0.006, 0.25)]),
    "four_group_skewed": dict(D=0.05, face_t=0.0008, wall_t=0.0006,
                              groups=[_g(126.0, 0.012, 0.4), _g(160.0, 0.010, 0.3),
                                      _g(196.0, 0.008, 0.2), _g(251.0, 0.006, 0.1)]),
    "low_freq_large_cavity": dict(D=0.05, face_t=0.0008, wall_t=0.0006,
                                  groups=[_g(126.0, 0.012, 0.5), _g(160.0, 0.010, 0.5)]),   # big cavity, low density
    "high_freq_small_cavity": dict(D=0.03, face_t=0.0008, wall_t=0.0006,
                                   groups=[_g(251.0, 0.006, 0.5), _g(251.0, 0.005, 0.5)]),  # small cavity, high density
    "boundary_feasible_sigma": dict(D=0.05, face_t=0.0008, wall_t=0.0006,
                                    groups=[_g(126.0, 0.005, 1.0)]),   # sigma ~0.0103, just above sigma_min=0.01
}
COVERAGES = [0.0, 0.6, 1.0]
CASES = [(n, c) for n in SWEEP for c in COVERAGES]


@pytest.mark.parametrize("name,cov", CASES, ids=lambda x: x if isinstance(x, str) else str(x))
def test_g3_case(name, cov):
    design = SWEEP[name]
    groups = _enrich(design["groups"])
    D, face_t, wall_t = design["D"], design["face_t"], design["wall_t"]

    # (1) valid under the INTENDED feasibility classification
    feas = F.check_feasibility(dict(design, groups=groups), coverage=cov, require_coverage=True)
    assert feas["feasible"] is True, f"{name}: intended-feasible design rejected: {feas['failures']}"

    # (2) acoustics + attenuation labels are mass-independent (unchanged by the mass rewire)
    lab = O2.label_v2(dict(design, groups=groups, f_cov=cov, A_wall=A_WALL, spectrum=SPEC, kappa_nominal=KAPPA))
    a_direct = O2.liner_diffuse_alpha_band(groups, D)
    d_direct = O2.robust_delta_oaspl(a_direct, SPEC, cov, A_WALL, KAPPA)
    assert np.allclose(list(lab["diffuse_alpha_band"].values()), a_direct, atol=1e-12)
    assert lab["dOASPL_worst_dB"] == pytest.approx(d_direct["worst"], abs=1e-12)

    # (3) old + corrected mass, INDEPENDENTLY recomputed
    legacy_areal = legacy_areal_mass(design["groups"], D, face_t, wall_t)
    legacy_total = legacy_total_mass(legacy_areal, cov, A_WALL)
    corrected_areal = MM.liner_areal_mass(groups, D, face_t, wall_t)
    corrected_rec = MM.total_installed_mass(corrected_areal, cov, A_WALL)
    corrected_total = corrected_rec["m_total"]

    # (4) the migrated label's mass IS the corrected mass-model output
    assert lab["total_installed_mass_kg"] == pytest.approx(corrected_total, rel=1e-12, abs=1e-12)
    assert lab["areal_mass_kg_m2"] == pytest.approx(corrected_areal, rel=1e-12)

    # (5) the observed label delta is EXACTLY corrected - legacy (nothing else moved)
    observed_delta = lab["total_installed_mass_kg"] - legacy_total
    assert observed_delta == pytest.approx(corrected_total - legacy_total, rel=1e-12, abs=1e-12)

    # (6) coverage applied EXACTLY once, incl. exact zero-mass at zero coverage
    A_t = MM.treated_area(cov, A_WALL)
    assert A_t == pytest.approx(cov * A_WALL, rel=1e-12)
    assert corrected_rec["m_liner"] == pytest.approx(corrected_areal * A_t, rel=1e-12)
    assert corrected_rec["m_install"] == pytest.approx(sum(corrected_rec["install_breakdown"].values()), rel=1e-12)
    assert corrected_total == pytest.approx(corrected_rec["m_liner"] + corrected_rec["m_install"]
                                            + corrected_rec["m_fixed"], rel=1e-12)
    if cov == 0.0:
        assert corrected_total == pytest.approx(0.0, abs=1e-12)   # zero coverage -> zero installed mass
        assert legacy_total == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("name", list(SWEEP))
def test_g3_group_order_permutation_invariant(name):
    design = SWEEP[name]
    groups = _enrich(design["groups"])
    D, face_t, wall_t = design["D"], design["face_t"], design["wall_t"]
    rev = list(reversed(groups))
    # mass (corrected + legacy) unchanged under permutation
    assert MM.liner_areal_mass(rev, D, face_t, wall_t) == pytest.approx(
        MM.liner_areal_mass(groups, D, face_t, wall_t), rel=1e-12)
    assert legacy_areal_mass(list(reversed(design["groups"])), D, face_t, wall_t) == pytest.approx(
        legacy_areal_mass(design["groups"], D, face_t, wall_t), rel=1e-12)
    # acoustics unchanged under permutation (admittance sum reorders -> agreement to float roundoff)
    f = np.linspace(50, 250, 40)
    assert np.allclose(O2.z_surface(f, rev, D), O2.z_surface(f, groups, D), rtol=1e-9, atol=1e-6)


@pytest.mark.parametrize("name", list(SWEEP))
def test_g3_inactive_banks_add_no_phantom_mass(name):
    design = SWEEP[name]
    groups = _enrich(design["groups"])
    D, face_t, wall_t = design["D"], design["face_t"], design["wall_t"]
    inactive = dict(_g(300.0, 0.006, 0.0), cavity_volume=O2._group_cavity_volume(_g(300.0, 0.006, 0.0)))
    with_inactive = groups + [inactive]
    assert MM.liner_areal_mass(with_inactive, D, face_t, wall_t) == pytest.approx(
        MM.liner_areal_mass(groups, D, face_t, wall_t), rel=1e-12)
    assert legacy_areal_mass(design["groups"] + [_g(300.0, 0.006, 0.0)], D, face_t, wall_t) == pytest.approx(
        legacy_areal_mass(design["groups"], D, face_t, wall_t), rel=1e-12)


def test_corrected_and_legacy_actually_differ():
    """Guard against a no-op: the corrected model must genuinely differ from the legacy one (else the sweep is
    vacuous). Keep the 1.4017 dOASPL a MIGRATION FIXTURE, not a Falcon verdict."""
    design = SWEEP["four_group_equal"]; groups = _enrich(design["groups"])
    corrected = MM.total_installed_mass(MM.liner_areal_mass(groups, design["D"], design["face_t"],
                                                            design["wall_t"]), 0.6, A_WALL)["m_total"]
    legacy = legacy_total_mass(legacy_areal_mass(design["groups"], design["D"], design["face_t"],
                                                 design["wall_t"]), 0.6, A_WALL)
    assert abs(corrected - legacy) > 1.0            # a real, non-trivial mass difference
