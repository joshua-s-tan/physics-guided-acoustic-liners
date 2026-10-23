# -*- coding: utf-8 -*-
"""Step 3 STANDALONE physics/passivity/unit/invariant tests for the migrated modules.

FRAMING (owner): exact agreement with the historical modules proves faithful migration, NOT physical
correctness. These tests stand on their own -- they assert physical properties directly, independent of any
reference fixture: finite/passive impedance, 0<=alpha<=1, energy-domain SPL aggregation, permutation invariance,
deterministic seeded spectra, and rejection of malformed/non-finite inputs.
"""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner.physics import oracle as O2  # noqa: E402
from acoustic_liner.physics import liner as FP  # noqa: E402
from acoustic_liner.physics import spectra as SP  # noqa: E402
from acoustic_liner.physics import resonator as RM  # noqa: E402

Z_AIR = O2.Z_AIR
DESIGN = [dict(f0=126.0, r_neck=0.012, L_neck=0.03, area_frac=0.25, Q=15.0),
          dict(f0=160.0, r_neck=0.010, L_neck=0.03, area_frac=0.25, Q=15.0),
          dict(f0=196.0, r_neck=0.008, L_neck=0.03, area_frac=0.25, Q=15.0),
          dict(f0=251.0, r_neck=0.006, L_neck=0.03, area_frac=0.25, Q=15.0)]
D = 0.05
BAND = np.array(O2.DESIGN_BAND_HZ, dtype=float)


# ------------------------------- passivity / impedance ------------------------------- #
def test_surface_impedance_is_finite_and_passive():
    f = np.linspace(40.0, 300.0, 120)
    z = O2.z_surface(f, DESIGN, D)
    assert np.all(np.isfinite(z.real)) and np.all(np.isfinite(z.imag))
    assert np.all(z.real >= -1e-9)                        # passive: Re(z_s) >= 0
    beta = Z_AIR / z                                      # normalized admittance
    assert np.all(beta.real >= -1e-9)                     # passive admittance (Re(beta) >= 0)


def test_diffuse_absorption_in_unit_interval():
    a = O2.liner_diffuse_alpha_band(DESIGN, D)
    assert np.all(a >= 0.0) and np.all(a <= 1.0)
    # a single passive impedance too
    z = O2.z_surface(BAND, DESIGN, D)
    assert np.all(O2.diffuse_alpha(z) >= 0.0) and np.all(O2.diffuse_alpha(z) <= 1.0)


def test_diffuse_alpha_rejects_nonpassive_impedance():
    active = np.array([-100.0 + 0j, -50.0 + 10j])         # Re(z_s) < 0 -> active (|R|>1 -> alpha<0)
    with pytest.raises(ValueError):
        O2.diffuse_alpha(active)


# ------------------------------- energy-domain SPL aggregation ------------------------------- #
def test_delta_oaspl_is_energy_weighted_and_level_invariant():
    spl = np.array([120.0, 120.0, 119.8, 120.5, 121.5, 122.0, 121.5, 120.5])
    # uniform NR across all bands -> dOASPL equals that NR exactly (energy weights sum to 1)
    for nr in (0.0, 3.0, 7.5):
        assert O2.delta_oaspl(np.full(8, nr), spl) == pytest.approx(nr, abs=1e-12)
    # level-invariance: a uniform dB offset of the spectrum leaves dOASPL unchanged
    nr_band = np.array([1.0, 2.0, 5.0, 3.0, 8.0, 2.0, 6.0, 4.0])
    assert O2.delta_oaspl(nr_band, spl) == pytest.approx(O2.delta_oaspl(nr_band, spl + 12.3), abs=1e-12)
    # more attenuation never reduces the OASPL reduction (monotone in a uniform NR bump)
    assert O2.delta_oaspl(nr_band + 1.0, spl) > O2.delta_oaspl(nr_band, spl)


def test_partial_nr_zero_coverage_is_zero_and_full_positive():
    a = O2.liner_diffuse_alpha_band(DESIGN, D)
    nr0 = O2.partial_nr_band(a, 0.05, 0.0, 165.878, 0.2327 * 165.878)
    assert np.allclose(nr0, 0.0, atol=1e-12)              # no treated area -> no NR
    nr1 = O2.partial_nr_band(a, 0.05, 1.0, 165.878, 0.2327 * 165.878)
    assert np.all(np.isfinite(nr1))
    assert nr1.max() > 0.0        # the liner helps at its resonances
    # per-band NR is +ve where the liner out-absorbs the bare wall and -ve at off-resonance bands where
    # alpha_liner < alpha_bare=0.05 (physically correct, not a bug). With NO fixed A_other, full-coverage NR
    # is exactly the Sabine alpha ratio 10*log10(alpha_liner/alpha_bare).
    nr_full_noother = O2.partial_nr_band(a, 0.05, 1.0, 165.878, 0.0)
    assert np.allclose(nr_full_noother, 10.0 * np.log10(np.maximum(a, 1e-30) / 0.05), atol=1e-9)
    assert nr_full_noother.min() < 0.0      # confirms off-resonance under-absorption gives negative NR


# ------------------------------- invariance / determinism ------------------------------- #
def test_acoustics_permutation_invariant():
    # admittances add in a different order under permutation -> agreement to float roundoff, not bit-exact
    f = np.linspace(50, 250, 50)
    assert np.allclose(O2.z_surface(f, list(reversed(DESIGN)), D), O2.z_surface(f, DESIGN, D),
                       rtol=1e-9, atol=1e-6)


def test_seeded_spectra_are_deterministic_and_seed_dependent():
    a = SP.sample_family(16, seed=7)
    b = SP.sample_family(16, seed=7)
    assert all(np.array_equal(x["spl_dB"], y["spl_dB"]) for x, y in zip(a, b))   # same seed -> identical
    c = SP.sample_family(16, seed=8)
    assert any(not np.allclose(x["spl_dB"], y["spl_dB"]) for x, y in zip(a, c))  # different seed -> differs


# ------------------------------- malformed / non-finite rejection ------------------------------- #
def test_rejects_malformed_and_nonfinite_inputs():
    spl = np.full(8, 120.0)
    with pytest.raises(ValueError):
        O2.delta_oaspl(np.array([np.nan] * 8), spl)                # non-finite NR
    with pytest.raises(ValueError):
        O2.delta_oaspl(np.zeros(7), spl)                           # shape mismatch
    with pytest.raises(ValueError):
        O2._validate_fcov(1.5)                                     # coverage out of [0,1]
    with pytest.raises(ValueError):
        O2.label_v2(dict(D=D, groups=DESIGN, face_t=0.0008, wall_t=0.0006, f_cov=0.6, A_wall=165.878))  # no spectrum
    with pytest.raises(ValueError):
        O2.label_v2(dict(D=D, groups=DESIGN, face_t=0.0008, wall_t=0.0006, f_cov=0.6, A_wall=165.878,
                         spectrum=np.zeros(5)))                     # wrong spectrum length
    with pytest.raises(ValueError):
        O2.label_v2(dict(D=D, groups=DESIGN, face_t=0.0008, wall_t=0.0006, f_cov=0.6, A_wall=165.878,
                         spectrum=O2.falcon9_design_band(), kappa_nominal=-0.1))   # negative kappa


def test_resonator_physics_basic_identities():
    # resonant frequency inverts cavity_volume_for_target_f0 exactly (frozen physics sanity)
    S = RM.neck_area(0.008); Le = RM.effective_neck_length(0.03, 0.008)
    V = RM.cavity_volume_for_target_f0(180.0, S, Le)
    assert RM.resonant_frequency(S, Le, V) == pytest.approx(180.0, rel=1e-12)
    assert FP.Z_AIR == pytest.approx(RM.RHO * RM.C, rel=1e-12)
