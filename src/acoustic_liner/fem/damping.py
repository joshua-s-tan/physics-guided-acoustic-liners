# -*- coding: utf-8 -*-
"""Equivalent distributed (volumetric) non-wall damping for `A_other` (remediation §4.2, permitted-alternative
clause; owner decision 2026-09-26).

OWNER DECISION: the Falcon scenario represents `A_other` as EQUIVALENT DISTRIBUTED non-wall damping, NOT an
explicit payload boundary. Real Falcon payload/structure geometry and absorbing area are mission-dependent and
unavailable, so no `S_other` is invented. The volumetric loss enters the interior Helmholtz operator as

    k^2  ->  k^2 (1 - i*eta(f)),      eta(f) = COEF * A_other / (V * f)

This module DERIVES `eta` from the Sabine reverberation relationship and lets the tests verify it BOTH
analytically and numerically (the modal decay of the damped operator must reproduce the Sabine T60), which is
the precondition for retaining the volumetric conversion.

Derivation (why COEF ~= 13.66):
  * Sabine:        T60 = k_S * V / A,   k_S = 0.161 s/m  (= 24 ln10 / c, metric, c ~ 343 m/s).
  * Modal loss:    a uniform loss factor eta gives modal energy decay  E(t) = E0 exp(-omega * eta * t)
                   (for k^2(1 - i*eta), a Neumann mode has k = sqrt(mu)(1 - i*eta)^(-1/2), so
                   Im(omega)/Re(omega) = eta/2 and energy ~ exp(-2 Im(omega) t) = exp(-omega*eta*t)).
  * 60 dB energy decay:  omega * eta * T60 = ln(10^6) = 6 ln10 =: DB60.
  * Equate the two T60:   eta = DB60 / (2*pi*f * k_S) * A/(V) = COEF * A/(V*f),
                   COEF = DB60 / (2*pi * k_S) = 13.8155 / (2*pi*0.161) = 13.657 ~= 13.66.
"""
from __future__ import annotations
import math

SABINE_CONST = 0.161                       # s/m (metric Sabine, c ~ 343 m/s)
DB60 = 6.0 * math.log(10.0)                # ln(10^6) = 13.815510557964274 (60 dB energy decay)
VOLUMETRIC_COEF = DB60 / (2.0 * math.pi * SABINE_CONST)   # ~= 13.657 (the historical 13.66)
A_OTHER_MODEL_ID = "equivalent_distributed_volumetric_damping"


class DampingError(ValueError):
    """A damping input is invalid (nonpositive volume/frequency, negative absorption, non-dissipative eta)."""


def _pos(name, v):
    x = float(v)
    if not (math.isfinite(x) and x > 0):
        raise DampingError(f"{name} must be finite and > 0; got {v}")
    return x


def sabine_t60(volume: float, A_sabine: float) -> float:
    """Sabine reverberation time T60 = 0.161 * V / A [s]."""
    return SABINE_CONST * _pos("volume", volume) / _pos("A_sabine", A_sabine)


def exact_decay_ratio(eta: float) -> float:
    """EXACT modal decay-rate / Re(omega) for the operator k^2(1 - i*eta): 2*tan(atan(eta)/2). Written in the
    NUMERICALLY STABLE form  2*eta/(sqrt(1+eta^2) + 1)  (algebraically equal to 2*(sqrt(1+eta^2)-1)/eta but
    free of catastrophic cancellation as eta -> 0). The small-loss form uses `eta`; they differ at O(eta^2)."""
    e = float(eta)
    if not (math.isfinite(e) and e >= 0):
        raise DampingError(f"eta must be finite and >= 0; got {eta}")
    if e == 0.0:
        return 0.0
    return 2.0 * e / (math.sqrt(1.0 + e * e) + 1.0)


def small_loss_relative_error(eta: float) -> float:
    """Relative error of the small-loss approximation (ratio ~= eta) vs the exact decay ratio. At the worst
    current Falcon point eta ~= 0.0635 this is ~0.10% -- recorded so the approximation is bounded, not hidden."""
    e = float(eta)
    if not (math.isfinite(e) and e >= 0):
        raise DampingError(f"eta must be finite and >= 0; got {eta}")
    if e == 0.0:
        return 0.0
    exact = exact_decay_ratio(e)
    return abs(e - exact) / exact


def volumetric_loss_factor(A_other: float, volume: float, freq: float) -> float:
    """SMALL-LOSS loss factor eta(f) = COEF * A_other / (V * f), applied as k^2 -> k^2(1 - i*eta). A_other = 0
    -> eta = 0. Derived so the small-loss modal T60 equals the Sabine T60 (module docstring). This is the
    small-loss form: its exact modal-decay discrepancy is `small_loss_relative_error(eta)` (~0.10% at the
    worst current Falcon eta ~= 0.0635); use the exact ratio if a tighter bound is ever required."""
    A = float(A_other)
    if not (math.isfinite(A) and A >= 0):
        raise DampingError(f"A_other must be finite and >= 0; got {A_other}")
    V = _pos("volume", volume)                 # validate volume/freq REGARDLESS of A_other (fail-closed)
    f = _pos("freq", freq)
    if A == 0.0:
        return 0.0
    return VOLUMETRIC_COEF * A / (V * f)


def t60_from_loss_factor(eta: float, freq: float) -> float:
    """Invert the modal-decay relation: T60 = DB60 / (2*pi*f*eta) [s]. eta must be > 0."""
    return DB60 / (2.0 * math.pi * _pos("freq", freq) * _pos("eta", eta))


def assert_sabine_consistency(A_other: float, volume: float, freq: float, rtol: float = 1e-6) -> float:
    """Assert the loss factor reproduces the Sabine T60 (the analytic verification). Returns the T60 [s]."""
    if not (isinstance(rtol, (int, float)) and math.isfinite(rtol) and rtol > 0):
        raise DampingError(f"rtol must be finite and > 0; got {rtol}")
    eta = volumetric_loss_factor(A_other, volume, freq)
    if eta <= 0:
        raise DampingError("assert_sabine_consistency requires A_other > 0")
    t_modal = t60_from_loss_factor(eta, freq)
    t_sabine = sabine_t60(volume, A_other)
    if abs(t_modal - t_sabine) / t_sabine > rtol:
        raise DampingError(f"modal T60 {t_modal} != Sabine T60 {t_sabine} (rtol {rtol})")
    return t_sabine


def modal_energy_decay_rate(eta: float, freq: float) -> float:
    """Modal energy decay rate for a loss factor eta at frequency f: dE/dt = -rate*E, rate = 2*pi*f*eta [1/s].
    Rejects negative/non-finite eta (a decay rate can never be negative)."""
    e = float(eta)
    if not (math.isfinite(e) and e >= 0):
        raise DampingError(f"eta must be finite and >= 0; got {eta}")
    return 2.0 * math.pi * _pos("freq", freq) * e


def loss_factor_record(A_other: float, volume: float, kappa: float, freqs) -> dict:
    """The distributed-damping block to embed in EVERY FEM artifact (§4.2 item 3): model id, formula, cavity
    volume, A_other, kappa, and the per-frequency loss factor."""
    k = float(kappa)
    if not (math.isfinite(k) and 0.0 <= k <= 1.0):
        raise DampingError(f"kappa must be finite in [0,1]; got {kappa}")
    fs = list(freqs)
    if not fs:
        raise DampingError("freqs must be a non-empty sequence")
    for f in fs:
        _pos("freq", f)                        # each frequency finite and > 0
    fl = {float(f): volumetric_loss_factor(A_other, volume, f) for f in fs}
    return dict(a_other_model=A_OTHER_MODEL_ID, formula="eta(f) = COEF * A_other/(V*f); k^2 -> k^2(1 - i*eta)",
                coef=VOLUMETRIC_COEF, sabine_const=SABINE_CONST, cavity_volume=float(volume),
                A_other=float(A_other), kappa=k, loss_factor_by_freq=fl)
