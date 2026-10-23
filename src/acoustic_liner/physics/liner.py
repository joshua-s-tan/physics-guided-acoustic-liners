# -*- coding: utf-8 -*-
"""Case-F reuse-only physics bridge for the fairing liner (the acoustic layer the oracle builds on).

MIGRATED from the historical `fairing_case/fairing_oracle.py` (frozen SHA-256
ae75c5a428713af0711c08ffbbe617e8f9425a8988aad40ba5eee8e9c894a393) for the §14 step-8 self-containment migration
(the historical migration record, which is not distributed). Changes vs the reference:
  * the `sys.path` hack + `import resonator_model` is replaced by the intra-package `from . import resonator`;
  * the v1 `areal_mass` (4*wall_t*D*sqrt(n)) is DELIBERATELY NOT carried here -- the release mass model is the
    CORRECTED shared-wall `physics.mass` (2*sqrt(n)). Carrying the v1 areal mass would reintroduce the
    superseded objective. Mass belongs to `physics.mass`; this module is acoustics only.
  * the v1 `label(...)` and `__main__` self-test (which used the v1 areal mass) are dropped for the same reason.
Every retained physics function is byte-identical to the reference, so `_resonator_Za`, `_group_density`, and
`surface_absorption` reproduce the reference exactly (verified against frozen fixtures).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from . import resonator as RM  # frozen physics (intra-package; no reference tree)

RHO = RM.RHO            # 1.204 kg/m^3
C = RM.C                # 343 m/s
Z_AIR = RHO * C         # ~413 Pa*s/m  (specific characteristic impedance of air)

# Benchmark constants (historical benchmark definition, not distributed); Falcon 9 max-predicted 1/3-octave SPL (dB re 20 uPa).
FALCON9_SPL = {
    25: 113.5, 31.5: 118.0, 40: 119.5, 50: 120.0, 63: 120.0, 80: 119.8,
    100: 120.5, 125: 121.5, 160: 122.0, 200: 121.5, 250: 120.5,
}
FALCON9_OASPL = 131.4
DESIGN_BAND_HZ = [50, 63, 80, 100, 125, 160, 200, 250]
ALPHA_BARE = 0.05                        # bare composite-fairing wall interior absorption (low, ~rigid)

# Envelope / manufacturability (historical benchmark definition sec 4)
MAX_DEPTH_M = 0.050
MIN_NECK_R_M = 0.0005
MAX_AREAL_MASS = 5.0
RHO_WALL = 1550.0


def _resonator_Za(freqs: np.ndarray, r_neck: float, L_neck: float, V_cav: float, Q: float) -> np.ndarray:
    """Acoustic impedance Z_r(f) [Pa*s/m^3] of ONE Helmholtz resonator, built entirely from physics.resonator."""
    S = RM.neck_area(r_neck)
    L_eff = RM.effective_neck_length(L_neck, r_neck)
    M_a = RM.acoustic_mass(L_eff, S)
    C_a = RM.acoustic_compliance(V_cav)
    omega = 2 * np.pi * freqs
    return RM.resonator_impedance(omega, M_a, C_a, Q)


def _group_density(group: Dict, D: float) -> float:
    """Cells/m^2 of TOTAL wall for a tuned group, derived from the backing depth D so every cavity fits within
    D by construction: cells/m^2 = area_frac * D / V_cav."""
    S = RM.neck_area(group["r_neck"])
    L_eff = RM.effective_neck_length(group["L_neck"], group["r_neck"])
    V_cav = RM.cavity_volume_for_target_f0(group["f0"], S, L_eff)
    return group["area_frac"] * D / max(V_cav, 1e-9)


def surface_absorption(freqs: np.ndarray, groups: Sequence[Dict], D: float) -> np.ndarray:
    """Normal-incidence absorption alpha(f) of the parallel resonator-group liner packed into a backing of
    depth D. Each group: {r_neck, L_neck, f0, area_frac, Q}."""
    beta = np.zeros_like(freqs, dtype=complex)
    for g in groups:
        S = RM.neck_area(g["r_neck"])
        L_eff = RM.effective_neck_length(g["L_neck"], g["r_neck"])
        V_cav = RM.cavity_volume_for_target_f0(g["f0"], S, L_eff)
        Za = _resonator_Za(freqs, g["r_neck"], g["L_neck"], V_cav, g["Q"])
        beta += _group_density(g, D) / Za          # admittances add in parallel
    with np.errstate(divide="ignore", invalid="ignore"):
        z_s = np.where(np.abs(beta) > 0, 1.0 / beta, 1e30 + 0j)
    R = (z_s - Z_AIR) / (z_s + Z_AIR)
    alpha = 1.0 - np.abs(R) ** 2
    return np.clip(alpha.real, 0.0, 1.0)
