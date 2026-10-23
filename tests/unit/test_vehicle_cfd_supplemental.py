# -*- coding: utf-8 -*-
"""Falcon supplemental (post-hoc) block estimator: transient discard, effective sample size, gates."""
import numpy as np
import pytest

from acoustic_liner.vehicles import cfd_supplemental as S


def test_discard_first_third():
    assert (S.n_discard(12), S.n_discard(24)) == (4, 8)


def test_ess_iid_matches_student_t():
    x = np.array([1.0, -1.0] * 8)                       # strongly anti-correlated -> rho clipped to 0
    h, n_eff, rho = S.ess_halfwidth(x)
    from scipy import stats
    assert rho == 0 and n_eff == 16
    assert h == pytest.approx(stats.t.ppf(0.975, 15) * np.std(x, ddof=1) / 4)


def test_ess_positive_autocorrelation_widens_interval():
    rng = np.random.default_rng(0)
    e = rng.normal(size=400)
    ar = np.zeros(400)
    for i in range(1, 400):
        ar[i] = 0.8 * ar[i - 1] + e[i]
    h_ar, ne, rho = S.ess_halfwidth(ar)
    assert 0.6 < rho < 0.95 and ne < 400 / 4
    assert h_ar > S.ess_halfwidth(rng.permutation(ar))[0]


def _Z(R, X):
    return np.asarray(R) + 1j * np.asarray(X)


def test_stationary_noisy_R_passes_trend_and_stability():
    rng = np.random.default_rng(1)
    R = 10 + 0.8 * rng.standard_normal(16)
    st = S.block_stats(_Z(R, np.full(16, 43.0)), np.full(16, 12.0 + 0j))
    g = S.gates(st, 0.05, 0.05, dict(a=True))
    assert st["z_mag"] == pytest.approx(abs(complex(R.mean(), 43.0)))
    assert g["uncertainty"] and g["distortion"] and g["other_validator_checks"]


def test_trend_and_shift_fail():
    R = np.linspace(8, 12, 16)                          # drifting resistance
    st = S.block_stats(_Z(R, np.full(16, 20.0)), np.ones(16))
    g = S.gates(st, 0.05, 0.05, dict(a=True))
    assert not g["trend"] and not g["stable_means"]


def test_thd_and_other_checks_fail_closed():
    st = S.block_stats(_Z(np.full(8, 10.0), np.full(8, 40.0)), np.ones(8))
    assert not S.gates(st, 0.2, 0.05, dict(a=True))["distortion"]
    assert not S.gates(st, 0.05, 0.05, dict(a=True, b=False))["other_validator_checks"]


def test_perturb_shifts_rows_by_own_halfwidth_keeps_anchor():
    rx = [[(0.4, 5.0, 30.0), (6.0, 6.0, 34.0)]]
    hw = [[(0.4, 0.1, 0.2), (6.0, 0.3, 0.5)]]
    out = S.perturb(rx, hw, [(0.4, 5.0, 30.0)], 1, -1)
    assert out[0][0] == (0.4, 5.0, 30.0) and out[0][1] == pytest.approx((6.0, 6.3, 33.5))
    with pytest.raises(S.SupplementalError):
        S.perturb(rx, [[(0.4, 0.1, 0.2), (7.0, 0.3, 0.5)]], [None], 1, 1)


def _fem(base):
    return {t: {f: {0.2: base + 0.02, 0.15: base + 0.01, 0.1: base + i * 0.01} for f in S.FORCINGS}
            for i, t in enumerate(S.TABLES)}


def test_interval_takes_worst_table_and_outcomes():
    iv = S.supp_interval(_fem(3.0), 0.1, 0.05)
    assert iv["F01"] == 3.0 and iv["F01_hi"] == pytest.approx(3.04) and iv["M"] == pytest.approx(0.03)
    assert iv["L"] == pytest.approx(3.0 - 0.03 - 0.1)
    assert S.outcome(iv, []) == S.PASS and S.outcome(iv, ["x"]) == S.UNRESOLVED
    assert S.outcome(S.supp_interval(_fem(0.5), 0.0, 0.0), []) == S.FAIL
    f = _fem(3.0); del f["R-X-"]
    assert S.supp_interval(f, 0, 0) is None and S.outcome(None, []) == S.UNRESOLVED


def test_render_keeps_primary_and_claim_discipline():
    v = dict(outcome=S.PASS, primary_outcome="UNRESOLVED", target_db=0.9213, interval=S.supp_interval(_fem(3.0), 0.1, 0.05),
             reasons=[], evidence={}, wording=S.WORDING, protocol=S.PROTOCOL, reported_not_verdict={})
    md = S.render(v)
    assert "SUPPLEMENTAL ROBUST PASS" in md and "**UNRESOLVED** (unchanged)" in md and "not preregistered" in md
    assert "remained above target across the resulting impedance interval" in md and "normalised by |Z|" in md
    low = md.lower().replace("not experimentally validated", "").replace("not flight-qualified", "")
    assert "experimentally validated" not in low and "flight-qualified" not in low
    assert "no deterministic periodic convergence is claimed" in low
