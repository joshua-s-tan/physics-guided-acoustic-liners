# -*- coding: utf-8 -*-
"""TEST-ONLY legacy (v1) mass formulas, reproduced verbatim from `fairing_oracle.areal_mass`
(4*wall_t*D*sqrt(n_local)) + v1 `mass_model.total_mass` (m_areal*A_treated + m_fixed + mu_A*A_treated).

This is DELIBERATELY test-only and MUST NEVER be imported by any `acoustic_liner` production module -- the corrected
shared-wall model (`acoustic_liner.physics.mass`, 2*sqrt(n) + BOM) supersedes it. It exists purely so the G3 sweep can
independently recompute the OLD mass and show the entire migrated-vs-reference label delta is `corrected-legacy`.
Physics (V_cav) is taken from `acoustic_liner.physics.resonator` (the same frozen physics), so no reference tree is used.
"""
import math

from acoustic_liner.physics import resonator as RM

LEGACY_RHO_WALL = 1550.0        # kg/m^3 (v1 fairing_oracle)
LEGACY_MU_A = 0.30              # kg/m^2 lumped install allowance (v1 mass_model.MU_A_DEFAULT)


def _cavity_volume(g) -> float:
    S = RM.neck_area(g["r_neck"])
    L_eff = RM.effective_neck_length(g["L_neck"], g["r_neck"])
    return float(RM.cavity_volume_for_target_f0(g["f0"], S, L_eff))


def _group_density(g, D) -> float:
    return g["area_frac"] * D / max(_cavity_volume(g), 1e-9)


def legacy_areal_mass(groups, D, face_t, wall_t) -> float:
    """v1 areal mass [kg/m^2]: facesheet + 4*wall_t*D*sqrt(n_local) partitions, area-fraction weighted."""
    m = LEGACY_RHO_WALL * face_t
    for g in groups:
        n = _group_density(g, D)
        n_local = n / max(g["area_frac"], 1e-6)
        wall_vol = 4.0 * wall_t * D * math.sqrt(max(n_local, 0.0))
        m += g["area_frac"] * LEGACY_RHO_WALL * wall_vol
    return m


def legacy_total_mass(m_areal, f_cov, A_wall, m_fixed=0.0, mu_A=LEGACY_MU_A) -> float:
    """v1 total installed mass [kg] = m_areal*A_treated + m_fixed + mu_A*A_treated. f_cov=0 -> 0 (m_fixed=0)."""
    A_t = min(1.0, max(0.0, float(f_cov))) * float(A_wall)
    return m_areal * A_t + (m_fixed + mu_A * A_t)
