"""CMA-ES building blocks shared by the public methods: the locked population and step sizes, the
random-screen size, the restart budget split, the feasible restart start, and the budgeted
restart/reinitialization loop under the constrained-quality objective.
"""
from __future__ import annotations

from ..design import constraints as FEAS
from ..design import schema as DS
import numpy as np
from ..physics.quality import quality_objective


N_RESTARTS = 4
HYBRID_K0_MAX = 128
CMA_POPSIZE = 14                                  # = released closure-search setting
CMA_SIGMA0_SEEDED = 0.10                          # approved default: ~half of cold
FEASIBLE_START_DRAWS = 4000                       # = released closure-search feasible-start bound


def _feasible_start(rng):
    for _ in range(FEASIBLE_START_DRAWS):
        xr = DS.sample_random(rng)
        if FEAS.is_feasible(DS.vector_to_design(xr), 1.0)[0]:
            return DS.normalize01(xr)
    return DS.normalize01(DS.sample_random(rng))


def _split(total, n):
    q, r = divmod(int(total), int(n))
    return [q + (1 if i < r else 0) for i in range(n)]


def hybrid_k0(budget):
    return min(HYBRID_K0_MAX, int(budget) // 4)


def _quality_cma_restart(L, x0, sigma0, sub_budget, rng_start, cma_seed, source):
    """The historical `study_core._cma_restart`, verbatim restart/reinit rule, with `quality_objective` in place of
    `physics.oracle_evaluation.objective`."""
    import cma
    stop_at = len(L.trace) + sub_budget
    n_reinit = 0
    while len(L.trace) < stop_at and not L.exhausted:
        es = cma.CMAEvolutionStrategy(list(np.clip(x0, 0, 1)), sigma0,
                                      {"bounds": [0, 1], "popsize": CMA_POPSIZE, "verbose": -9,
                                       "seed": int(cma_seed) + n_reinit, "maxiter": 10 ** 6})
        while not es.stop() and len(L.trace) < stop_at and not L.exhausted:
            sols = es.ask(); vals = []
            for u in sols:
                if len(L.trace) >= stop_at or L.exhausted:
                    break
                r = L.propose(DS.vector_to_design(DS.denormalize01(np.clip(u, 0, 1))), source)
                vals.append(quality_objective(r))
            if len(vals) == len(sols):
                es.tell(sols, vals)
        if len(L.trace) < stop_at and not L.exhausted:
            n_reinit += 1
            x0 = _feasible_start(rng_start)
    return n_reinit
