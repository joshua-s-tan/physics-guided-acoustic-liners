# -*- coding: utf-8 -*-
r"""PHYS-GEN-V1 authoritative production comparator: EXACT-GAUGE plain quality-CMA (Arm A).

Pre-seal comparator correction (owner decision 2026-10-02, before any pilot/development/final/Falcon/Atlas result was
opened). The locked historical `v26_objective.run_quality_cma` builds its CMA restart starts with `DS.design_to_vector`, which
clips log(area_frac) to the logit box [-3, 3]; on random feasible designs that changes the area fractions of ~80% of
the screened seeds (up to 0.115), so CMA does not start from the design its screen selected. A generator win against
that comparator would be vulnerable.

`run_quality_cma_exact` is `run_quality_cma(seed_source="random")` VERBATIM -- same K0 random screen, same registered
streams (`evaluation.random_streams.stream` random / cma_start / cma), same restart split, same `_quality_cma_restart` loop and reinit
rule, same ledger and ranking -- with ONE change: restart starts are encoded with the exact centered-logit gauge
(`optimization.seeded_cma.to_cma01(..., sort_banks=False)`), so each CMA run starts at exactly the selected design. Because
the streams are shared, the corrected and legacy arms see identical random screens; they differ only from the first
CMA restart onward. The locked function is untouched and kept as a continuity/legacy diagnostic.

Scoring: the ledger is `physics.scorer.ExactQualityLedger` (authoritative exact constrained-coverage scorer, owner
decision 2026-10-02); the legacy `quality_eval` remains only in the legacy diagnostic.
"""
from __future__ import annotations

from ..design import schema as DS
from ..evaluation import random_streams as RS
from . import cma as CMA
from ..physics import quality as Q
from . import seeded_cma as HO
from ..physics import scorer as PS


def run_quality_cma_exact(cond, cond_index, seed, budget, base_seed, sigma0=CMA.CMA_SIGMA0_SEEDED,
                          tag="quality_cma_exact"):
    """Authoritative Arm A: plain quality-CMA with exact restart-start encoding (see module docstring)."""
    k0 = CMA.hybrid_k0(budget)
    rng_start = RS.stream(base_seed, cond_index, seed, "cma_start")
    cma_rng = RS.stream(base_seed, cond_index, seed, "cma")
    L = PS.ExactQualityLedger(cond, budget)                       # authoritative exact scorer
    rng = RS.stream(base_seed, cond_index, seed, "random")
    while len(L.trace) < k0 and not L.exhausted:
        L.propose(DS.vector_to_design(DS.sample_random(rng)), f"{tag}_random")
    seeds_ev = sorted(L.trace, key=Q.quality_rank_key)[:CMA.N_RESTARTS]
    starts = [HO.to_cma01(e["design"], sort_banks=False) for e in seeds_ev]          # <-- the only change
    n_random_starts = CMA.N_RESTARTS - len(starts)
    starts += [CMA._feasible_start(rng_start) for _ in range(n_random_starts)]
    reinit = 0
    for r, (x0, sub) in enumerate(zip(starts, CMA._split(budget - len(L.trace), CMA.N_RESTARTS))):
        reinit += CMA._quality_cma_restart(L, x0, sigma0, sub, rng_start, int(cma_rng.integers(1, 2 ** 31 - 1)),
                                           f"{tag}_cma_r{r}")
    L.n_reinit = reinit
    L.k0 = k0
    L.n_random_seed_starts = n_random_starts
    L.start_encoding = "exact_centered_logit"
    return L
