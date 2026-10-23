"""Reproduces the frozen study splits: the synthetic request family (independent spectrum and
environment streams, target = fraction of the achievable maximum) that every evaluation split and
the historical splits are built from. It is not part of the LinerGen architecture; the rest of the
historical module (the cVAE selection-rule evaluation) stays private.
"""
from __future__ import annotations

import numpy as np
from ...physics import oracle as O2
from ...physics import spectra as SP


def _weights_from_spl(spl_dB):
    spl = np.asarray(spl_dB, dtype=float)
    w = 10.0 ** ((spl - spl.max()) / 10.0)
    return w / w.sum()


def build_fresh_test_requests(n=120, seed=31415, target_frac=0.5, group="fresh_test"):
    """A FRESH, independent held-out set of REQUESTS (amendment 2): synthetic spectra and the A_wall/kappa DOE
    draws come from SEPARATELY SPAWNED RNG streams (SeedSequence.spawn) so the spectrum shape and the
    environment variables are not correlated through a shared stream. dOASPL target = target_frac x the
    oracle-estimated achievable max for that condition (a meaningful positive target). NEVER used for selection;
    a distinct `seed`/`group` gives the confirmatory set, independent of the dev fresh-test set."""
    spec_ss, env_ss = np.random.SeedSequence(seed).spawn(2)      # independent streams: spectrum vs environment
    specs = SP.sample_family(n, seed=int(spec_ss.generate_state(1)[0]))
    rng_env = np.random.default_rng(env_ss)
    from ..requests import oracle_achievable_max_fn
    amax_fn = oracle_achievable_max_fn(O2)
    reqs = []
    for i, s in enumerate(specs):
        spl = np.asarray(s["spl_dB"], float)
        A_wall = float(rng_env.uniform(50.0, 200.0)); kappa = float(rng_env.uniform(0.0, 0.30))
        amax = float(amax_fn(spl_dB=spl, kappa=kappa, A_wall=A_wall))
        dO_tgt = max(0.25, target_frac * amax)                   # meaningful positive target
        reqs.append(dict(id=f"{group}_{i}", group=group, weights=_weights_from_spl(spl), spl_dB=spl.tolist(),
                         dOASPL_target=float(dO_tgt), m_total_target=200.0, A_wall=A_wall, kappa=kappa))
    return reqs
