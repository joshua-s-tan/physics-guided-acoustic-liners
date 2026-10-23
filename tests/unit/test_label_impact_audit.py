# -*- coding: utf-8 -*-
"""Step 5 label-impact audit (a historical audit document that is not distributed), as a REPRODUCIBLE gate.

Proves the audit's four load-bearing claims, self-contained (no reference tree):
  1. acoustic labels are EXACTLY mass-independent (migration preserved them);
  2. the corrected-vs-legacy mass change is DESIGN-DEPENDENT (not a constant offset/scale) -> no rescaling of the
     mass condition can convert an old-mass model to a corrected-mass one;
  3. the min-mass ORDERING inverts between the two models -> the training targets themselves moved (not fixable
     by any monotone label transform);
  4. the corrected model is systematically lighter (sanity, ~30%).
Together => the historical dataset/checkpoint are INVALID and retraining is required (not null).
"""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner.physics import oracle as O2  # noqa: E402
from acoustic_liner.physics import mass as MM  # noqa: E402
from acoustic_liner.design import schema as DS  # noqa: E402
from acoustic_liner.design import constraints as FEAS  # noqa: E402
from _legacy_mass import legacy_areal_mass, legacy_total_mass   # noqa: E402  (the SUPERSEDED v1 mass, test-only)

A_WALL = 165.878
COV = 0.6
KAPPA = 0.2327
SPEC = O2.falcon9_design_band()


def _feasible_designs(n=60, seed=7):
    lr = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        d = DS.vector_to_design(DS.sample_random(lr))
        if FEAS.is_feasible(d, 1.0)[0]:
            out.append(d)
    return out


@pytest.fixture(scope="module")
def sweep():
    designs = _feasible_designs()
    recs = []
    for d in designs:
        a_band = O2.liner_diffuse_alpha_band(d["groups"], d["D"])
        lab = O2.label_v2(dict(d, f_cov=COV, A_wall=A_WALL, spectrum=SPEC, kappa_nominal=KAPPA))
        direct = O2.robust_delta_oaspl(a_band, SPEC, COV, A_WALL, KAPPA)
        eg = O2._enrich_groups(d)
        corr_areal = MM.liner_areal_mass(eg, d["D"], d["face_t"], d["wall_t"])
        corr_total = MM.total_installed_mass(corr_areal, COV, A_WALL)["m_total"]
        leg_areal = legacy_areal_mass(d["groups"], d["D"], d["face_t"], d["wall_t"])
        leg_total = legacy_total_mass(leg_areal, COV, A_WALL)
        recs.append(dict(design=d, lab=lab, direct=direct, a_band=a_band,
                         corr_areal=corr_areal, corr_total=corr_total,
                         leg_areal=leg_areal, leg_total=leg_total, face=MM.RHO_WALL * d["face_t"]))
    return recs


def test_acoustic_labels_mass_independent(sweep):
    """Claim 1: the acoustic labels equal the mass-free oracle outputs EXACTLY."""
    worst = 0.0
    for r in sweep:
        worst = max(worst, abs(r["lab"]["dOASPL_worst_dB"] - r["direct"]["worst"]),
                    abs(r["lab"]["dOASPL_nominal_dB"] - r["direct"]["nominal"]),
                    abs(r["lab"]["dOASPL_best_dB"] - r["direct"]["best"]))
        ab = np.array([r["lab"]["diffuse_alpha_band"][f] for f in O2.DESIGN_BAND_HZ])
        assert np.allclose(ab, r["a_band"], atol=1e-12)
    assert worst == pytest.approx(0.0, abs=1e-12)


def test_mass_change_is_design_dependent(sweep):
    """Claim 2: at a FIXED condition the corrected/legacy ratio is NOT constant -> not an offset/scale."""
    ratio = np.array([r["corr_total"] / r["leg_total"] for r in sweep])
    assert ratio.max() - ratio.min() > 0.1        # spans a wide band (constant would give ~0)
    assert ratio.std() > 0.02
    # the driver is the (design-varying) facesheet fraction of the corrected areal mass
    face_frac = np.array([r["face"] / r["corr_areal"] for r in sweep])
    assert face_frac.max() - face_frac.min() > 0.3


def test_min_mass_ordering_inverts(sweep):
    """Claim 3 (narrowed): there exist designs whose mass RANKING flips between legacy and corrected (same
    condition). A ranking inversion cannot be undone by any monotone transform of the mass label -> post-hoc
    rescaling of m_total_target is provably invalid. (This is the claim the audit relies on; it does NOT assert
    that the global minimum-mass SELECTION changes -- see the scope note of that audit.)"""
    found = False
    for a in sweep:
        for b in sweep:
            if (a["leg_total"] < b["leg_total"] - 1e-6) and (a["corr_total"] > b["corr_total"] + 1e-6):
                found = True
                break
        if found:
            break
    assert found, "no min-mass ordering inversion found (audit claim 3 would not hold)"


def test_corrected_is_systematically_lighter(sweep):
    """Claim 4: corrected total mass is systematically lighter (objective moved toward lighter designs)."""
    corr = np.mean([r["corr_total"] for r in sweep])
    leg = np.mean([r["leg_total"] for r in sweep])
    assert corr < leg
    assert 0.15 < (1.0 - corr / leg) < 0.5        # ~a third lighter on this sweep


def test_retrain_required_conclusion(sweep):
    """The audit verdict: acoustic-null AND (design-dependent mass change OR ordering inversion) => the historical
    labels are NOT output-neutral => retraining is required. Guards the doc's conclusion against silent drift."""
    acoustic_null = max(abs(r["lab"]["dOASPL_worst_dB"] - r["direct"]["worst"]) for r in sweep) < 1e-12
    ratio = np.array([r["corr_total"] / r["leg_total"] for r in sweep])
    design_dependent = (ratio.max() - ratio.min()) > 0.1
    assert acoustic_null and design_dependent   # not output-neutral -> retrain (never a null-impact reuse)
