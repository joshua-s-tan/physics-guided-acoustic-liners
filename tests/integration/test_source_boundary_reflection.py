# -*- coding: utf-8 -*-
r"""§4.8 #8 -- source/impedance FEM boundary reflection: MAGNITUDE and PHASE vs an INDEPENDENT analytic value.

The production source boundary (§4.3) is a radiation Robin (normalized admittance beta_base = 1, anechoic) plus
an incident source; the wall/impedance boundary is the same Robin kernel `1j*k*beta`. `acoustic_liner.fem.verification.duct_reflection`
drives a plane wave down a 1-D duct assembled with those PRODUCTION coefficients and extracts the COMPLEX
incident (A) and reflected (B) amplitudes, so the reflection coefficient R = (B/A) e^{+2ikL} can be compared to
the hand-derived analytic value R = (1 - beta)/(1 + beta) in BOTH magnitude and wrapped phase.

Independence (no circular test): the analytic R is derived by hand in the module docstring for the stated
e^{+iwt} convention; the FEM path uses the production boundary coefficients. A sign error in those coefficients
makes the measured R disagree with the analytic R -- in phase for a complex beta, and (for beta = 1) also in
magnitude. This file therefore FAILS on a sign-convention error even where |R| alone would not reveal it.
"""
import cmath
import math

import pytest

np = pytest.importorskip("numpy")

import sys, os                                                       # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from acoustic_liner.fem.verification import duct_reflection as DR  # noqa: E402
from acoustic_liner.fem import boundary as FB  # noqa: E402
from acoustic_liner.physics import constants as C  # noqa: E402

C_AIR = C.C_AIR
L_DUCT = 3.0
FREQS = (80.0, 150.0, 220.0)
# passive test impedances: matched/radiating, nearly rigid, real passive, complex passive
BETAS = {"matched": 1.0 + 0j, "near_rigid": 0.01 + 0j, "real_passive": 1.0 / 3.0 + 0j,
         "complex_passive": 0.3 + 0.2j}


def _k(f):
    return 2.0 * math.pi * f / C_AIR


def _wrapped_phase_diff(a: complex, b: complex) -> float:
    """Smallest signed angle between two complex numbers, robust to +-2pi wrapping."""
    return abs(cmath.phase(a / b))


def test_matched_boundary_is_anechoic():
    """beta = 1 (the source boundary's OWN radiation condition) must be anechoic: |R| ~ 0 at every frequency.
    A Robin sign flip would instead give R = (1+beta)/(1-beta) -> infinite at beta=1, so this is a strong,
    magnitude-level sign check on the production coefficient."""
    for f in FREQS:
        k = _k(f)
        z, p = DR.solve_duct(k, L_DUCT, n_elem=800, beta_term=1.0 + 0j)
        R, A, B, resid = DR.measured_reflection(z, p, k, L_DUCT)
        assert resid < 1e-4, f"field is not a clean two-wave superposition (resid {resid})"
        assert abs(A) > 1e-6, "source boundary launched no incident wave"
        assert abs(R) < 2e-3, f"matched boundary not anechoic at {f} Hz: |R|={abs(R)}"


def test_reflection_magnitude_and_phase_across_impedances():
    """|R| AND wrapped phase match the analytic (1-beta)/(1+beta) for nearly-rigid, real-passive and
    complex-passive impedances, at multiple frequencies. Complex A and B are both extracted (not just |p|)."""
    for name, beta in BETAS.items():
        if name == "matched":
            continue                                                # covered by the anechoic test
        R_an = DR.analytic_reflection(beta)
        for f in FREQS:
            k = _k(f)
            z, p = DR.solve_duct(k, L_DUCT, n_elem=800, beta_term=beta)
            R, A, B, resid = DR.measured_reflection(z, p, k, L_DUCT)
            assert resid < 1e-4 and abs(A) > 1e-6 and abs(B) > 1e-9
            assert abs(R) == pytest.approx(abs(R_an), abs=5e-3), f"|R| off ({name}, {f} Hz): {abs(R)} vs {abs(R_an)}"
            assert _wrapped_phase_diff(R, R_an) < 5e-3, (
                f"phase off ({name}, {f} Hz): {cmath.phase(R)} vs {cmath.phase(R_an)}")


def test_reflection_converges_with_mesh():
    """Refining the mesh drives the measured R monotonically toward the analytic value (linear-FEM dispersion
    error shrinks); the finest mesh is within a tight tolerance."""
    beta = BETAS["complex_passive"]
    R_an = DR.analytic_reflection(beta)
    k = _k(220.0)                                                   # worst case: shortest wavelength
    errs = []
    for n_elem in (100, 200, 400, 800):
        z, p = DR.solve_duct(k, L_DUCT, n_elem=n_elem, beta_term=beta)
        R, *_ = DR.measured_reflection(z, p, k, L_DUCT)
        errs.append(abs(R - R_an))
    for i in range(1, len(errs)):
        assert errs[i] <= errs[i - 1] + 1e-9, f"error not decreasing under refinement: {errs}"
    assert errs[-1] < 2e-3, f"finest-mesh reflection error {errs[-1]} too large"


def test_phase_comparison_pins_the_sign_convention():
    """A sign-convention (time-convention / conjugation) error preserves |R| but flips its phase, so a
    magnitude-only check cannot catch it -- only the wrapped-PHASE comparison can. For a complex passive beta:
      * |conj(R_an)| == |R_an|            (magnitude alone is blind to the sign),
      * phase(conj(R_an)) != phase(R_an)  (a conjugation error is a real, large phase error),
      * the MEASURED R matches R_an, NOT conj(R_an), in phase (production uses the stated sign)."""
    beta = BETAS["complex_passive"]
    R_an = DR.analytic_reflection(beta)
    R_conj = R_an.conjugate()                                       # what a time-convention error would yield
    assert abs(abs(R_conj) - abs(R_an)) < 1e-12                     # identical magnitude
    assert _wrapped_phase_diff(R_conj, R_an) > 0.5                  # but a large phase error (~0.86 rad here)
    k = _k(150.0)
    z, p = DR.solve_duct(k, L_DUCT, n_elem=800, beta_term=beta)
    R, *_ = DR.measured_reflection(z, p, k, L_DUCT)
    assert _wrapped_phase_diff(R, R_an) < 5e-3                      # measured agrees with the correctly-signed R
    assert _wrapped_phase_diff(R, R_conj) > 0.5                     # and is NOT the conjugate -> sign is caught

    # a genuine FEM Robin sign flip at the matched point is also caught by magnitude (R -> infinite, not 0)
    z2, p2 = DR.solve_duct(k, L_DUCT, n_elem=800, beta_term=1.0 + 0j, robin_sign=-1.0)
    R_bad, *_ = DR.measured_reflection(z2, p2, k, L_DUCT)
    assert abs(R_bad) > 0.5, f"sign-flipped Robin should NOT look anechoic; |R|={abs(R_bad)}"


def test_analytic_agrees_with_independent_absorption_helper():
    """Cross-check the hand-derived R against the SEPARATE production analytic helper `normal_incidence_alpha`
    (also independent of the FEM assembly): 1 - |R|^2 == normal_incidence_alpha(zeta), zeta = 1/beta."""
    for beta in (0.01 + 0j, 1.0 / 3.0 + 0j, 0.3 + 0.2j, 1.0 + 0j):
        R = DR.analytic_reflection(beta)
        zeta = 1.0 / beta
        assert (1.0 - abs(R) ** 2) == pytest.approx(FB.normal_incidence_alpha(zeta), abs=1e-12)
