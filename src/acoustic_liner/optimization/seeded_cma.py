# -*- coding: utf-8 -*-
r"""PHYS-GEN-V1 population -> CMA handoff (pre-registration sec. 12), Phase I.

Locks three interfaces before any neural training:

1. REPRESENTATION. The locked plain quality-CMA searches the box-normalized `design.schema` vector
   [D, face, wall, (f0, r, L, a) x 4] with area logits a in [-3, 3] and decodes it with
   `DS.vector_to_design(DS.denormalize01(clip(y)))`. `to_cma01` maps a PHYS-GEN design into that vector EXACTLY, using
   the CENTERED logit gauge a = log frac - (max + min)/2, so the locked decode path reproduces the same geometry, area
   fractions and oracle score. It never clips: a design outside the CMA-representable set raises. (The locked
   `DS.design_to_vector` is NOT used: it clips log(frac) to [-3, 3] and alters any bank below ~5% area.)

2. ACCOUNTING. Everything runs on `physics.scorer.ExactQualityLedger` (the locked QualityLedger's dedup/charging
   rule, scored by the authoritative exact constrained-coverage scorer): a proposal is charged iff it is a
   physically feasible, previously unseen geometry, exactly as for Arm A. The first N_PROPOSALS charged evaluations
   are the proposal population; any shortfall (duplicate / uncharged proposals) is filled from the Arm-B random
   sampler on its own stream, every fill charged and counted. CMA spends the remaining B - N_PROPOSALS.

3. COMPARATOR. Arm B (initialization control) draws its 32 proposals as u ~ Logistic(0, 1)^18 through the SAME
   feasibility transform (every conditional interval sampled uniformly), so any Arm-D advantage cannot come from the
   representation alone -- only from learned conditioning. Arms B and D share this module's population -> CMA
   initialization verbatim; only the proposal source differs.

Initialization constants below were PINNED by the owner on 2026-10-02 (prereg sec. 12 / 18 item 8) from interface
and smoke reasoning only, never from pilot/development/final/Falcon/Atlas outcomes.
"""
from __future__ import annotations

import numpy as np
import torch

from ..design import schema as DS
from . import cma as CMA
from ..physics import quality as Q
from ..design import coordinates as PC
from ..physics import scorer as PS

# ---- PINNED initialization constants (owner, 2026-10-02; recorded in the Phase I notes, not distributed) -----------
N_PROPOSALS = 32                 # proposals charged before CMA (prereg sec. 12 arms B/C/D)
K_RETAIN = 8                     # best proposals (by the locked quality_rank_key) that shape the CMA start
SIGMA0 = CMA.CMA_SIGMA0_SEEDED    # 0.10 in the normalized box: the locked seeded-CMA step size
STD_FLOOR = 0.02                 # effective initial marginal std floor (normalized box); CMA_stds = std / SIGMA0
STD_CAP = 0.30                   # effective initial marginal std cap
POPSIZE = CMA.CMA_POPSIZE         # 14, locked CMA population size
# Stream tags: disjoint from evaluation.random_streams.stream's {random:1, cvae:2, cma_start:3, cma:4}.
_TAGS = {"pg_proposal": 101, "pg_fill": 102, "pg_cma": 103, "pg_cma_start": 104}

_LO, _HI = DS.bounds()
_SPAN = _HI - _LO


class HandoffError(ValueError):
    """A design cannot be represented exactly in the locked CMA coordinates, or a handoff input is malformed."""


def stream(base_seed, cond_index, seed, tag):
    return np.random.default_rng(np.random.SeedSequence([int(base_seed), int(cond_index), int(seed), _TAGS[tag]]))


# =================================================================================================================
#  1. representation
# =================================================================================================================
def cma_vector(design, sort_banks=True):
    """Design -> locked design.schema vector (physical units), CENTERED area logits; banks sorted by f0 unless
    sort_banks=False (then the design's own bank order is kept, ties allowed). Raises HandoffError unless every
    coordinate lies in the box (no clipping)."""
    if sort_banks:
        try:
            d = PC.canonicalize(design)
        except PC.CoordinateDomainError as e:
            raise HandoffError(str(e)) from e
    else:
        d = design
    frac = np.array([float(g["area_frac"]) for g in d["groups"]])
    if not np.all(np.isfinite(frac)) or not np.all(frac > 0.0):
        raise HandoffError("area fractions must be finite and > 0")
    lf = np.log(frac)
    a = lf - 0.5 * (lf.max() + lf.min())
    x = [float(d["D"]), float(d["face_t"]), float(d["wall_t"])]
    for g, ag in zip(d["groups"], a):
        x += [float(g["f0"]), float(g["r_neck"]), float(g["L_neck"]), float(ag)]
    x = np.asarray(x, float)
    if not np.all(np.isfinite(x)) or np.any(x < _LO) or np.any(x > _HI):
        bad = np.flatnonzero(~((x >= _LO) & (x <= _HI)))
        raise HandoffError(f"design not representable in the locked CMA box at coordinates {bad.tolist()} "
                           f"(area log-span {float(lf.max() - lf.min()):.6g} vs {PC.SPAN_MAX})")
    return x


def to_cma01(design, sort_banks=True):
    """Design -> the locked CMA's normalized coordinates in [0, 1]^19 (exact; raises instead of clipping)."""
    return (cma_vector(design, sort_banks=sort_banks) - _LO) / _SPAN


def from_cma01(y):
    """Normalized CMA coordinates -> design, through EXACTLY the locked CMA's decode path."""
    return DS.vector_to_design(DS.denormalize01(np.clip(np.asarray(y, float), 0.0, 1.0)))


# =================================================================================================================
#  3. comparator: Arm-B random proposals through the same feasibility transform
# =================================================================================================================
def sample_noise(rng, n, extra_dim=0):
    """THE shared noise sampler (Arm B and the generator): [n, 18 + extra_dim]. Columns 0..17 are u ~ Logistic(0, 1)
    (sigmoid(u) ~ Uniform(0, 1): every conditional interval sampled uniformly), drawn FIRST as one block; columns 18..
    are standard normal, drawn after. So the first 18 columns are bit-identical to Arm B's draws from the same rng state."""
    u = rng.logistic(0.0, 1.0, size=(int(n), PC.DIM))
    if int(extra_dim) == 0:
        return u
    return np.concatenate([u, rng.standard_normal((int(n), int(extra_dim)))], axis=1)


def random_u(rng, n):
    """Arm B: u ~ Logistic(0, 1)^18 via the shared sampler."""
    return sample_noise(rng, n)


def designs_from_u(u):
    out = PC.decode_t(torch.as_tensor(np.asarray(u, float)))
    return [from_cma01((row - _LO) / _SPAN) for row in out["cma_vec"].detach().numpy()]


def random_proposals(rng, n):
    return designs_from_u(random_u(rng, n))


# =================================================================================================================
#  2. accounting + population -> CMA initialization
# =================================================================================================================
def _cma_weights(k):
    w = np.log(k + 0.5) - np.log(np.arange(1, k + 1))          # standard CMA positive recombination weights
    return w / w.sum()


def init_from_population(evs):
    """Retained evaluations (best first) -> (x0, CMA_stds) in normalized CMA coordinates."""
    Y = np.array([to_cma01(e["design"]) for e in evs])
    w = _cma_weights(len(Y))
    x0 = w @ Y
    std = np.sqrt(np.maximum(w @ (Y - x0) ** 2, 0.0))
    std = np.clip(std, STD_FLOOR, STD_CAP)
    return x0, std / SIGMA0


def _cma_phase(L, x0, cma_stds, stop_at, cma_seed, rng_start, source):
    """Locked restart/reinit loop (`optimization.cma._quality_cma_restart`), first run initialized from the population
    (mean x0, per-coordinate CMA_stds); a converged run reinitializes exactly as the locked rule does."""
    import cma
    n_reinit = 0
    opts = {"bounds": [0, 1], "popsize": POPSIZE, "verbose": -9, "maxiter": 10 ** 6, "CMA_stds": list(cma_stds)}
    while len(L.trace) < stop_at and not L.exhausted:
        es = cma.CMAEvolutionStrategy(list(np.clip(x0, 0, 1)), SIGMA0, dict(opts, seed=int(cma_seed) + n_reinit))
        while not es.stop() and len(L.trace) < stop_at and not L.exhausted:
            sols = es.ask(); vals = []
            for y in sols:
                if len(L.trace) >= stop_at or L.exhausted:
                    break
                vals.append(Q.quality_objective(L.propose(from_cma01(y), source)))
            if len(vals) == len(sols):
                es.tell(sols, vals)
        if len(L.trace) < stop_at and not L.exhausted:
            n_reinit += 1
            x0 = CMA._feasible_start(rng_start)
            opts = {k: v for k, v in opts.items() if k != "CMA_stds"}   # locked cold reinit: isotropic
    return n_reinit


def run_seeded_cma(cond, cond_index, seed, budget, base_seed, proposal_fn, tag, *, cma=True):
    """One (condition, seed, budget) run of a population-seeded arm.
    proposal_fn(rng, n) -> list of designs (Arm D: generator samples; Arm B: `random_proposals`).
    cma=False gives the generator-only diagnostic (Arm C): proposals only, no CMA.
    Returns the ExactQualityLedger with handoff provenance attached."""
    L = PS.ExactQualityLedger(cond, budget)                       # authoritative exact scorer
    n_target = min(N_PROPOSALS, int(budget))
    for d in proposal_fn(stream(base_seed, cond_index, seed, "pg_proposal"), N_PROPOSALS):
        if len(L.trace) >= n_target or L.exhausted:
            break
        L.propose(d, f"{tag}_proposal")
    n_from_source = len(L.trace)
    fill_rng = stream(base_seed, cond_index, seed, "pg_fill")
    while len(L.trace) < n_target and not L.exhausted:
        L.propose(random_proposals(fill_rng, 1)[0], f"{tag}_fill")
    L.n_proposal_charged, L.n_fill_charged = n_from_source, len(L.trace) - n_from_source
    L.n_reinit = 0
    if cma and len(L.trace) < int(budget) and not L.exhausted:
        retained = sorted(L.trace, key=Q.quality_rank_key)[:K_RETAIN]
        x0, cma_stds = init_from_population(retained)
        L.init_x0, L.init_cma_stds, L.k_retained = x0.tolist(), cma_stds.tolist(), len(retained)
        cma_seed = int(stream(base_seed, cond_index, seed, "pg_cma").integers(1, 2 ** 31 - 1))
        L.n_reinit = _cma_phase(L, x0, cma_stds, int(budget), cma_seed,
                                stream(base_seed, cond_index, seed, "pg_cma_start"), f"{tag}_cma")
    return L
