# -*- coding: utf-8 -*-
"""Equivalent distributed (volumetric) A_other damping (§4.2 permitted alternative). Verifies the Sabine <->
loss-factor relationship BOTH analytically and NUMERICALLY (the damped operator's modal energy decay must
reproduce eta = COEF*A/(V*f)), which is the precondition for retaining the volumetric conversion."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner.fem import damping as D  # noqa: E402


def test_coef_matches_historical_1366():
    assert D.VOLUMETRIC_COEF == pytest.approx(13.66, abs=0.02)     # 13.657
    assert D.DB60 == pytest.approx(math.log(1e6))


def test_zero_A_other_zero_loss():
    assert D.volumetric_loss_factor(0.0, 187.6, 100.0) == 0.0


def test_analytic_sabine_consistency():
    # modal T60 (from eta) must equal Sabine T60 across V, A, f
    for V in (100.0, 187.595, 300.0):
        for A in (20.0, 43.6):
            for f in (50.0, 125.0, 250.0):
                t = D.assert_sabine_consistency(A, V, f)
                assert t == pytest.approx(D.SABINE_CONST * V / A, rel=1e-9)


def _fe_1d(L=1.0, n=200):
    """Assemble 1-D P1 stiffness K and mass M on [0,L] (Neumann/rigid ends)."""
    h = L / n
    N = n + 1
    K = np.zeros((N, N)); M = np.zeros((N, N))
    ke = np.array([[1.0, -1.0], [-1.0, 1.0]]) / h
    me = np.array([[2.0, 1.0], [1.0, 2.0]]) * (h / 6.0)
    for e in range(n):
        K[e:e+2, e:e+2] += ke
        M[e:e+2, e:e+2] += me
    return K, M


def test_numeric_complex_damped_eigenproblem_decay():
    """Solve the ACTUAL complex damped generalized eigenproblem directly:
        K x = k^2 (1 - i*eta) M x   ->   eig(K, (1 - i*eta) M) gives COMPLEX k^2.
    This is not the analytic sqrt() reapplication: the complex operator is solved, and the recovered complex
    wavenumber must satisfy Im(k)/Re(k) ~= eta/2 (energy decay rate 2*pi*f*eta)."""
    from scipy.linalg import eig
    K, M = _fe_1d()
    eta = 0.05
    lam = eig(K, ((1.0 - 1j * eta) * M), right=False)      # COMPLEX generalized eigenproblem, solved directly
    lam = lam[np.isfinite(lam)]
    mu_eff = lam[np.argsort(np.abs(lam.real))][1]          # lowest nonzero mode, complex k^2
    k = np.sqrt(mu_eff)
    assert k.real > 0
    assert abs(abs(k.imag / k.real) - eta / 2.0) / (eta / 2.0) < 5e-3   # Im/Re == eta/2, from the damped solve
    f = 100.0
    assert D.modal_energy_decay_rate(eta, f) == pytest.approx(2 * math.pi * f * eta)


def test_numeric_ringdown_integration_matches_decay_rate():
    """GENUINE ring-down: NUMERICALLY integrate the damped modal ODE  q'' + (eta*omega0) q' + omega0^2 q = 0
    with a symplectic-ish leapfrog (no analytic solution used), extract the energy-envelope decay from the
    integrated trajectory's successive peaks, and compare the fitted rate to 2*pi*f*eta."""
    from scipy.linalg import eig
    K, M = _fe_1d()
    mu = np.sort(eig(K, M, right=False).real)[1]           # (pi/L)^2
    c, eta = 343.0, 0.06
    omega0 = c * math.sqrt(mu)
    f = omega0 / (2 * math.pi)
    gamma = eta * omega0                                   # velocity-damping coefficient

    # explicit numerical integration (velocity-Verlet with damping) of q'' = -omega0^2 q - gamma q'
    dt = (2 * math.pi / omega0) / 200.0                    # 200 steps per period
    nsteps = int(round(60.0 / (gamma) / dt))               # integrate ~ several e-foldings
    q, v = 1.0, 0.0
    ts, qs = [], []
    for i in range(nsteps):
        a = -omega0**2 * q - gamma * v
        q = q + v * dt + 0.5 * a * dt * dt
        a_new = -omega0**2 * q - gamma * (v + a * dt)      # semi-implicit damping update
        v = v + 0.5 * (a + a_new) * dt
        ts.append(i * dt); qs.append(q)
    ts = np.array(ts); qs = np.array(qs)

    # find positive peaks of the INTEGRATED trajectory and fit ln(peak^2) vs t -> energy decay rate
    pk = [i for i in range(1, len(qs) - 1) if qs[i] > qs[i-1] and qs[i] > qs[i+1] and qs[i] > 0]
    pk = pk[2:]                                            # drop the first couple (transient)
    tp = ts[pk]; ep = 2.0 * np.log(np.abs(qs[pk]))         # ln(energy) ~ ln(peak^2)
    slope = np.polyfit(tp, ep, 1)[0]                       # d ln(E)/dt = -rate
    rate_fit = -slope
    assert rate_fit == pytest.approx(D.modal_energy_decay_rate(eta, f), rel=0.02)


def test_exact_decay_ratio_stable_and_rejects_bad_eta():
    # matches the small-loss form to O(eta^2); stable at tiny eta (no cancellation)
    assert D.exact_decay_ratio(1e-8) == pytest.approx(1e-8, rel=1e-6)
    assert D.exact_decay_ratio(0.0) == 0.0
    with pytest.raises(D.DampingError):
        D.exact_decay_ratio(-0.01)
    with pytest.raises(D.DampingError):
        D.small_loss_relative_error(float("nan"))


def test_small_loss_error_bound_at_worst_falcon_point():
    # worst Falcon eta: high-kappa A_other ~43.58, V=187.595, lowest band f=50 -> eta ~0.0635
    A_hi = 0.262705707073558 * 165.878
    eta = D.volumetric_loss_factor(A_hi, 187.595, 50.0)
    assert eta == pytest.approx(0.0635, abs=0.001)
    err = D.small_loss_relative_error(eta)
    assert err == pytest.approx(0.0010, abs=0.0003)        # ~0.10% -- bounded, not material


def test_bad_inputs_rejected():
    with pytest.raises(D.DampingError):
        D.sabine_t60(0.0, 10.0)
    with pytest.raises(D.DampingError):
        D.volumetric_loss_factor(-1.0, 100.0, 50.0)
    with pytest.raises(D.DampingError):
        D.t60_from_loss_factor(0.0, 50.0)


def test_fail_closed_even_when_A_other_zero():
    # volume/frequency must be validated REGARDLESS of the zero-absorption early return
    with pytest.raises(D.DampingError):
        D.volumetric_loss_factor(0.0, -1.0, -2.0)
    with pytest.raises(D.DampingError):
        D.volumetric_loss_factor(0.0, 100.0, 0.0)


def test_modal_decay_rate_rejects_negative_eta():
    with pytest.raises(D.DampingError):
        D.modal_energy_decay_rate(-0.1, 100.0)
    with pytest.raises(D.DampingError):
        D.modal_energy_decay_rate(float("nan"), 100.0)


def test_consistency_and_record_validation():
    with pytest.raises(D.DampingError):
        D.assert_sabine_consistency(20.0, 187.6, 50.0, rtol=0.0)          # bad rtol
    with pytest.raises(D.DampingError):
        D.loss_factor_record(43.6, 187.6, kappa=1.5, freqs=(50.0,))       # kappa out of [0,1]
    with pytest.raises(D.DampingError):
        D.loss_factor_record(43.6, 187.6, kappa=0.23, freqs=())           # empty freqs
    with pytest.raises(D.DampingError):
        D.loss_factor_record(43.6, 187.6, kappa=0.23, freqs=(50.0, -1.0))  # non-positive freq


def test_loss_factor_record_for_artifact():
    rec = D.loss_factor_record(A_other=43.6, volume=187.595, kappa=0.2327, freqs=(50.0, 250.0))
    assert rec["a_other_model"] == "equivalent_distributed_volumetric_damping"
    assert set(rec["loss_factor_by_freq"]) == {50.0, 250.0}
    assert rec["loss_factor_by_freq"][50.0] > rec["loss_factor_by_freq"][250.0]   # eta ~ 1/f
