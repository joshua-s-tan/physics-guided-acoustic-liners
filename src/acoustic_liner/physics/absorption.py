# -*- coding: utf-8 -*-
"""Diffuse (random-incidence) absorption mapping for a locally-reacting surface (remediation §4.2, #3).

This is the PRODUCTION diffuse-absorption mapping used for the generic-benchmark explicit boundary and its
mandatory round-trip -- NOT the normal-incidence stand-in used only in unit stubs. For a locally-reacting
surface of normalized specific impedance `zeta` (= z_s / (rho c)), the angle-dependent absorption is

    alpha(theta) = 1 - |R(theta)|^2,   R(theta) = (zeta*cos theta - 1) / (zeta*cos theta + 1)

and the statistical (Paris) random-incidence absorption is the sin(2theta)-weighted average over the
hemisphere:  alpha_diffuse = integral_0^{pi/2} alpha(theta) sin(2 theta) d theta   (the weight integrates to 1).

The forward map `diffuse_alpha_from_zeta` and its monotone inverse `zeta_from_diffuse_alpha` (bisection on the
rigid branch zeta >= 1) give a validated round-trip. Physical specific admittance is `Y = 1/(zeta * Z_air)`;
the FEM Robin coefficient uses beta = Z_air * Y (see `fem.boundary.to_normalized_beta`).
"""
from __future__ import annotations
import math

import numpy as np

from .constants import Z_AIR  # single authoritative source

DIFFUSE_MAPPING_ID = "paris_random_incidence_locally_reacting_rigid_branch"
DIFFUSE_QUADRATURE_N = 2000              # quadrature resolution for the Paris integral
_NTHETA = DIFFUSE_QUADRATURE_N


def diffuse_alpha_max() -> float:
    """Maximum achievable diffuse absorption on the rigid branch (zeta=1)."""
    return diffuse_alpha_from_zeta(1.0)


class AcousticsError(ValueError):
    """Invalid absorption/impedance input."""


def diffuse_alpha_from_zeta(zeta: float) -> float:
    """Random-incidence (Paris) absorption for a real normalized impedance zeta >= 1 (rigid branch)."""
    z = float(zeta)
    if not (math.isfinite(z) and z >= 1.0):
        raise AcousticsError(f"zeta must be finite and >= 1 (rigid branch); got {zeta}")
    theta = np.linspace(0.0, math.pi / 2.0, _NTHETA)
    ct = np.cos(theta)
    R = (z * ct - 1.0) / (z * ct + 1.0)
    alpha = 1.0 - R ** 2
    w = np.sin(2.0 * theta)
    return float(np.trapezoid(alpha * w, theta))


def zeta_from_diffuse_alpha(alpha: float, tol: float = 1e-10) -> float:
    """Invert the diffuse-absorption map on the rigid branch: find zeta >= 1 with diffuse_alpha == alpha.
    diffuse_alpha is monotone decreasing in zeta (alpha->1 as zeta->1, alpha->0 as zeta->inf)."""
    a = float(alpha)
    if not (math.isfinite(a) and 0.0 < a < 1.0):
        raise AcousticsError(f"diffuse alpha must be in (0,1); got {alpha}")
    a_max = diffuse_alpha_from_zeta(1.0)                 # maximum achievable (zeta=1)
    if a > a_max + 1e-9:
        raise AcousticsError(f"diffuse alpha {a} exceeds the max achievable {a_max:.4f} (zeta>=1)")
    lo, hi = 1.0, 1.0
    while diffuse_alpha_from_zeta(hi) > a:               # grow hi until alpha(hi) <= a
        hi *= 2.0
        if hi > 1e12:
            raise AcousticsError("failed to bracket zeta")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if diffuse_alpha_from_zeta(mid) > a:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def admittance_from_diffuse_alpha(alpha: float, Z_air: float = Z_AIR) -> complex:
    """PHYSICAL specific admittance Y = 1/(zeta * Z_air) reproducing diffuse absorption `alpha`."""
    if not (math.isfinite(Z_air) and Z_air > 0):
        raise AcousticsError(f"Z_air must be finite and > 0; got {Z_air}")
    zeta = zeta_from_diffuse_alpha(alpha)
    return complex(1.0 / (zeta * Z_air))


def diffuse_alpha_from_admittance(Y, Z_air: float = Z_AIR) -> float:
    """Inverse: diffuse absorption implied by a physical specific admittance Y (real rigid branch)."""
    if not (math.isfinite(Z_air) and Z_air > 0):
        raise AcousticsError(f"Z_air must be finite and > 0; got {Z_air}")
    y = complex(Y)
    if abs(y.imag) > 1e-9 or y.real <= 0:
        raise AcousticsError(f"admittance must be real and > 0 on the rigid branch; got {Y}")
    zeta = 1.0 / (y.real * Z_air)
    return diffuse_alpha_from_zeta(zeta)
