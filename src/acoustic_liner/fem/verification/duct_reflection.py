# -*- coding: utf-8 -*-
r"""Independent plane-wave duct reflection harness for the FEM source/impedance boundary (§4.8 #8).

WHY: the production source boundary (`fem.boundary`, §4.3) is a radiation Robin (normalized admittance
beta_base = 1) plus an incident source; the wall/impedance boundary is the SAME Robin kernel `1j*k*beta*u*v`
with beta = Z_air*Y. This module drives a plane wave down a straight 1-D duct and measures the COMPLEX
reflection coefficient at an impedance boundary, so both its MAGNITUDE and PHASE (i.e. the sign convention)
can be checked against a hand-derived analytic value.

CONVENTION (stated here; it matches the production passivity note in `fem.boundary.assert_passive_admittance`
and the impedance/absorption in `fem.boundary.normal_incidence_alpha`, but is DERIVED independently below, so
the comparison is NOT circular):

  * time factor            e^{+i w t}   =>   a +z-travelling wave is  e^{-ikz},  a -z-travelling wave is e^{+ikz};
  * outward normal on the z=L termination is  +z^;  on the z=0 source boundary it is  -z^;
  * impedance Robin strong form (outward normal n^):   dp/dn = -i k beta p ,   beta = Z_air * Y  (dimensionless);
      the weak form  int(grad p . grad v) - k^2 int(p v) = int_Gamma (dp/dn) v  then contributes the bilinear
      term  +i k beta int(p v)  -- exactly the production coefficient `1j*k*beta`.
  * DERIVE R at a z=L impedance termination.  Write  p(z) = A e^{-ikz} + B e^{+ikz}  (A = incident/+z,
      B = reflected/-z).  dp/dz = -ikA e^{-ikz} + ikB e^{+ikz};  at z=L with n^=+z^, dp/dn = dp/dz.  Impose
      dp/dn = -ik beta p:
          -A e^{-ikL} + B e^{+ikL} = -beta (A e^{-ikL} + B e^{+ikL})
      =>  B e^{+ikL} (1 + beta) = A e^{-ikL} (1 - beta)
      =>  R := (B e^{+ikL}) / (A e^{-ikL}) = (1 - beta) / (1 + beta) = (zeta - 1)/(zeta + 1),  zeta = 1/beta.
    So the MEASURED reflection at z=L is (B/A) e^{+2ikL}, and the ANALYTIC value is (1-beta)/(1+beta).

The 1-D Helmholtz VOLUME assembly (stiffness - k^2 mass) is standard and unambiguous; the BOUNDARY terms use
the PRODUCTION coefficients from `fem.boundary`, so a sign error in those production coefficients makes the
measured R disagree with the independent analytic R above (in phase, and for beta=1 also in magnitude).
"""
from __future__ import annotations

import numpy as np

from .. import boundary as FB


def analytic_reflection(beta) -> complex:
    """Hand-derived plane-wave reflection at an impedance boundary, R = (1 - beta)/(1 + beta) (see module
    docstring). Independent of the FEM assembly -- this is the reference the FEM must reproduce."""
    b = complex(beta)
    return (1.0 - b) / (1.0 + b)


def solve_duct(k: float, L: float, n_elem: int, beta_term, p0: float = 1.0,
               robin_sign: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Solve the 1-D Helmholtz duct on [0, L] with `n_elem` linear elements.

      z=0   : PRODUCTION source boundary -- radiation Robin `base_radiation_robin_coeff(k)` (beta_base=1,
              anechoic) + incident source `incident_source_coeff(k, p0)`; launches a +z plane wave.
      z=L   : impedance termination -- production Robin `robin_sign * 1j*k*beta_term`.

    `robin_sign` is +1 for the production sign; a test may pass -1 to inject a deliberate sign error.
    Returns node coordinates `z` and the COMPLEX nodal pressure `p`.
    """
    if not (isinstance(n_elem, int) and n_elem >= 2):
        raise ValueError(f"n_elem must be an int >= 2; got {n_elem}")
    if not (k > 0 and L > 0):
        raise ValueError("k and L must be > 0")
    n = n_elem + 1
    z = np.linspace(0.0, L, n)
    h = L / n_elem
    Ke = np.array([[1.0, -1.0], [-1.0, 1.0]]) / h          # element stiffness
    Me = np.array([[2.0, 1.0], [1.0, 2.0]]) * (h / 6.0)    # element mass
    A = np.zeros((n, n), dtype=complex)
    for e in range(n_elem):
        A[e:e + 2, e:e + 2] += Ke - (k ** 2) * Me          # int grad p.grad v - k^2 int p v
    rhs = np.zeros(n, dtype=complex)
    # z=0 source boundary (production coefficients)
    A[0, 0] += FB.base_radiation_robin_coeff(k)            # 1j*k  (anechoic radiation, beta_base=1)
    rhs[0] += FB.incident_source_coeff(k, p0)             # 2j*k*p0  incident source
    # z=L impedance termination (production Robin sign 1j*k*beta; robin_sign lets a test flip it)
    A[n - 1, n - 1] += robin_sign * 1j * k * complex(beta_term)
    p = np.linalg.solve(A, rhs)
    return z, p


def extract_amplitudes(z: np.ndarray, p: np.ndarray, k: float,
                       convention: int = 1) -> tuple[complex, complex, float]:
    """Least-squares fit  p(z) ~= A e^{-ikz} + B e^{+ikz}  (production convention e^{+iwt}, `convention=+1`),
    returning (A incident/+z, B reflected/-z, relative_residual). `convention=-1` reads the SAME field with the
    opposite time convention (swapped exponentials) -- used only to show |R| alone cannot pin the sign."""
    s = -1j * k * float(convention)
    Phi = np.column_stack([np.exp(s * z), np.exp(-s * z)])
    coeff, *_ = np.linalg.lstsq(Phi, p, rcond=None)
    resid = float(np.linalg.norm(Phi @ coeff - p) / max(np.linalg.norm(p), 1e-300))
    return complex(coeff[0]), complex(coeff[1]), resid


def measured_reflection(z: np.ndarray, p: np.ndarray, k: float, L: float,
                        convention: int = 1) -> tuple[complex, complex, complex, float]:
    """Return (R at z=L, A incident, B reflected, fit residual). For the production convention,
    R = (B/A) e^{+2ikL} (see module docstring); the opposite convention negates the exponent's sign."""
    A, B, resid = extract_amplitudes(z, p, k, convention)
    R = (B / A) * np.exp(2j * k * L * float(convention))
    return complex(R), A, B, resid
