# -*- coding: utf-8 -*-
"""Analytical ventilated-Helmholtz-resonator physics (FROZEN, reuse-only).

MIGRATED VERBATIM (functions + constants) from the historical reference implementation
`resonator_updated_6/resonator_model.py` (frozen SHA-256
5a96549852a55d6bcb235bf04eca9f9c343e65cfcc65595329b6300b1b495a1e) as part of the §14 step-8 self-containment
migration (the historical migration record, which is not distributed). Only the module's `__main__` demo/reporting harness was dropped; every
physics function is byte-for-byte identical, so numerical equivalence to the reference is exact (verified by
tests against frozen fixtures). No reference-tree / sys.path imports.

Physics (standard closed-form): Helmholtz resonance f0 = (c/2pi) sqrt(S/(V L_eff)); Rayleigh end correction;
lumped mass-spring-damper acoustic impedance with an assumed quality factor Q; shunt-impedance transmission
loss. Resonator damping is parameterized by Q (the model's largest documented uncertainty), not derived from
first-principles viscous/thermal loss.
"""
import numpy as np

# Physical constants (air, ~20 C)
RHO = 1.204      # kg/m^3, air density
C = 343.0        # m/s, speed of sound
MU = 1.81e-5     # Pa*s, dynamic viscosity of air

# Benchmark operating point (locked in Phase 0; retained for reference/continuity)
DUCT_DIAMETER = 0.120
DUCT_AREA = np.pi * (DUCT_DIAMETER / 2) ** 2
VELOCITY = 3.0
Q_FLOW = DUCT_AREA * VELOCITY
DUCT_TEST_LENGTH = 1.3
TARGET_F0 = 120.0
TARGET_BAND = (110.0, 130.0)


def neck_area(r_neck):
    return np.pi * r_neck ** 2


def end_correction(r_neck, both_sides=True):
    """Rayleigh end correction. both_sides=True for a through-flow neck."""
    factor = 2 if both_sides else 1
    return factor * 0.85 * r_neck


def effective_neck_length(L_neck, r_neck):
    return L_neck + end_correction(r_neck)


def resonant_frequency(S_neck, L_eff, V_cav):
    return (C / (2 * np.pi)) * np.sqrt(S_neck / (V_cav * L_eff))


def cavity_volume_for_target_f0(f0_target, S_neck, L_eff):
    """Invert the resonance formula for the cavity volume that tunes to f0_target."""
    k = 2 * np.pi * f0_target / C
    return S_neck / (L_eff * k ** 2)


def acoustic_mass(L_eff, S_neck):
    return RHO * L_eff / S_neck


def acoustic_compliance(V_cav):
    return V_cav / (RHO * C ** 2)


def resonator_impedance(omega, M_a, C_a, Q_factor):
    """Lumped mass-spring-damper acoustic impedance; R_r from the assumed quality factor at resonance."""
    omega0 = 1 / np.sqrt(M_a * C_a)
    R_r = omega0 * M_a / Q_factor
    reactance = omega * M_a - 1 / (omega * C_a)
    return R_r + 1j * reactance


def transmission_loss(freqs, N, M_a, C_a, Q_factor, duct_area):
    omega = 2 * np.pi * freqs
    Z_r = resonator_impedance(omega, M_a, C_a, Q_factor)
    Z0 = RHO * C / duct_area          # duct characteristic impedance
    Z_shunt = Z_r / N                 # N identical resonators in parallel
    return 20 * np.log10(np.abs(1 + Z0 / (2 * Z_shunt)))


def orifice_pressure_drop(Q_flow, A_open_total, duct_area, Cd=0.6):
    beta = A_open_total / duct_area
    v_orifice = Q_flow / (Cd * A_open_total)
    return 0.5 * RHO * v_orifice ** 2 * (1 - beta ** 2)


def duct_friction_pressure_drop(Q_flow, duct_area, diameter, length):
    v = Q_flow / duct_area
    Re = RHO * v * diameter / MU
    f = 0.316 / Re ** 0.25            # Blasius correlation, smooth duct, turbulent flow
    return f * (length / diameter) * 0.5 * RHO * v ** 2
