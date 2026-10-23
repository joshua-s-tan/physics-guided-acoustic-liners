# -*- coding: utf-8 -*-
"""Step 3 numerical-equivalence: migrated `acoustic_liner` oracle/spectra vs COMMITTED FROZEN FIXTURES.

SELF-CONTAINED: reads `tests/fixtures/migration_fixtures.json` (which carries reference SHA-256, inputs, seeds,
units, expected outputs, generation command, dependency versions, and the tolerance policy). It NEVER imports
the historical tree at runtime -- regeneration is a separate optional audit (tests/fixtures/regenerate_fixtures.py).

Acoustic outputs are expected byte-identical (migration preserves the math). The legacy (v1) mass formula is
reproduced test-only (_legacy_mass) and shown to match the fixture's reference masses, so the G3 sweep can rely
on it WITHOUT the reference tree.
"""
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))     # for _legacy_mass
from acoustic_liner.physics import oracle as O2  # noqa: E402
from acoustic_liner.physics import spectra as SP  # noqa: E402
from _legacy_mass import legacy_areal_mass, legacy_total_mass   # noqa: E402

FIX = json.loads((ROOT / "tests" / "fixtures" / "migration_fixtures.json").read_text())
TOL = FIX["tolerance"]
SPEC = np.array(FIX["inputs"]["spectrum_falcon9_design_band"], dtype=float)
DESIGNS = FIX["inputs"]["designs"]
A_WALL = FIX["inputs"]["A_wall"]
KAPPA = FIX["inputs"]["kappa_nominal"]


def test_fixture_carries_required_provenance():
    p = FIX["provenance"]
    assert set(p["reference_source_sha256"]) >= {"oracle_v2.py", "spectra.py", "resonator_model.py",
                                                 "fairing_oracle.py", "launch_environments.py", "mass_model_v1.py"}
    for h in p["reference_source_sha256"].values():
        assert len(h) == 64
    assert p["dependency_versions"]["numpy"] and p["dependency_versions"]["python"]
    assert p["units"]["z_surface"] == "Pa*s/m" and p["units"]["mass"] == "kg"
    assert "acoustics_abs" in TOL and "legacy_mass_rel" in TOL


@pytest.mark.parametrize("case", FIX["oracle_cases"], ids=lambda c: f"{c['design_name']}@cov{c['coverage']}")
def test_oracle_matches_frozen_reference(case):
    design = DESIGNS[case["design_name"]]
    groups, D = design["groups"], design["D"]
    # acoustics: diffuse alpha band + z_surface (byte-identical migration)
    a_band = O2.liner_diffuse_alpha_band(groups, D)
    assert np.allclose(a_band, case["expected_diffuse_alpha_band"], atol=TOL["acoustics_abs"], rtol=TOL["acoustics_rel"])
    zf = np.array([z["f"] for z in case["expected_z_surface"]], dtype=float)
    zs = O2.z_surface(zf, groups, D)
    for z_got, z_exp in zip(zs, case["expected_z_surface"]):
        assert abs(z_got.real - z_exp["real"]) <= TOL["acoustics_abs"] + TOL["acoustics_rel"] * abs(z_exp["real"])
        assert abs(z_got.imag - z_exp["imag"]) <= TOL["acoustics_abs"] + TOL["acoustics_rel"] * abs(z_exp["imag"])
    # spectrum-weighted robust labels (acoustic part of label_v2)
    lab = O2.label_v2(dict(design, f_cov=case["coverage"], A_wall=A_WALL, spectrum=SPEC, kappa_nominal=KAPPA))
    assert abs(lab["dOASPL_worst_dB"] - case["expected_dOASPL_worst_dB"]) <= TOL["acoustics_abs"]
    assert abs(lab["dOASPL_nominal_dB"] - case["expected_dOASPL_nominal_dB"]) <= TOL["acoustics_abs"]
    assert lab["robust_ngrid"] == case["expected_robust_ngrid"]


@pytest.mark.parametrize("case", FIX["oracle_cases"], ids=lambda c: f"{c['design_name']}@cov{c['coverage']}")
def test_legacy_helper_reproduces_reference_mass(case):
    """The test-only legacy formula must reproduce the fixture's REFERENCE (v1) masses exactly -- this is what
    lets the G3 sweep recompute the old mass without importing the reference tree."""
    design = DESIGNS[case["design_name"]]
    areal = legacy_areal_mass(design["groups"], design["D"], design["face_t"], design["wall_t"])
    total = legacy_total_mass(areal, case["coverage"], A_WALL)
    assert areal == pytest.approx(case["reference_areal_mass_kg_m2"], rel=TOL["legacy_mass_rel"])
    assert total == pytest.approx(case["reference_total_mass_kg"], rel=TOL["legacy_mass_rel"], abs=1e-9)


@pytest.mark.parametrize("seed", [int(s) for s in FIX["spectra_family"]])
def test_spectra_family_matches_frozen(seed):
    rows = SP.sample_family(FIX["inputs"]["spectra_rows"], seed=seed)
    exp = FIX["spectra_family"][str(seed)]
    assert len(rows) == len(exp)
    for got, e in zip(rows, exp):
        assert np.allclose(got["spl_dB"], e["spl_dB"], atol=TOL["spectra_abs"], rtol=0.0)
        assert np.allclose(got["weights"], e["weights"], atol=TOL["spectra_abs"], rtol=0.0)
