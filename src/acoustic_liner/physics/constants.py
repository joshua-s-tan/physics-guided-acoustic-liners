# -*- coding: utf-8 -*-
"""Authoritative physical constants for the release package -- defined ONCE and consumed everywhere (no
per-module duplication). Air properties match the historical solver (fem_full_duct_radiation)."""
from __future__ import annotations

C_AIR = 343.0            # m/s, speed of sound
RHO_AIR = 1.204          # kg/m^3, air density
Z_AIR = RHO_AIR * C_AIR  # ~= 413.0 Pa.s/m, characteristic specific impedance

AIR_PROPERTIES = dict(c_air=C_AIR, rho_air=RHO_AIR, z_air=Z_AIR)
