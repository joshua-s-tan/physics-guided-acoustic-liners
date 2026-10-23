# -*- coding: utf-8 -*-
r"""PHYS-GEN-V1 conditional generator (pre-registration sec. 5), Phase I.

    u = z[:, :18] + head( ResFiLM( z, c ) ),   z = [Logistic(0,1)^18, N(0,1)^14],   c = normalized canonical condition

`u` is consumed by `design.coordinates.decode_t` (18-D exact feasibility-by-construction transform). Architecture:
a residual MLP with FiLM condition injection (per-block scale and shift computed from the condition) in EVERY residual
block; noise dimension 32 >= the 18-D effective coordinate dimension (no compression bottleneck); one architecture, no
sweep. The output head is zero-initialized and the first 18 noise dimensions pass straight through (fixed, not learned),
so AT INITIALIZATION the generator IS Arm B: the first 18 noise dimensions come from the shared sampler
`optimization.seeded_cma.sample_noise` (Logistic(0,1), uniform within every conditional interval) and the zero-head output
is bit-identical to Arm B's u and designs from the same rng state (owner 2026-10-02: the prior must not differ from
Arm B, so any C/D-vs-B difference is learned). The remaining 14 standard-normal dimensions carry learned residual
variation and are inert until the head becomes nonzero. The model outputs a population (one design per noise draw).

Condition features (fixed, deterministic, hash-bound; no data-fitted statistics): the canonical condition vector
`design.conditions.condition_vector` = [8 spectrum energy weights, dOASPL target, m_total target, A_wall, kappa], mapped to
    [8 * w (8), dOASPL_target, log(m_total_target), log(A_wall), kappa / 0.3, log(m_total_target / A_wall)].
Architecture PINNED for r1 (owner 2026-10-02, after the Arm-B prior correction and the rerun gradient suite): fixed
after shape / numerical / gradient conformance only, never chosen between alternatives on any outcome.
`public_architecture_sha256` identifies the architecture by its specification: the configuration and the tensor
schema (names, shapes, dtypes), not by a module path or source text.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from ..design import coordinates as PC
from ..optimization import seeded_cma as HO
from ..design import conditions as CND

DTYPE = torch.float64
# ---- PINNED architecture (r1) ---------------------------------------------------------------------------------------
NOISE_DIM = 32
WIDTH = 256
N_BLOCKS = 4
COND_DIM = 13
CONFIG = dict(noise_dim=NOISE_DIM, width=WIDTH, n_blocks=N_BLOCKS, cond_dim=COND_DIM, out_dim=PC.DIM,
              activation="silu", norm="layernorm", film="per_block_scale_shift", passthrough="z[:18]",
              noise="logistic18_then_normal14",
              head_init="zeros", dtype="float64")


class GeneratorInputError(ValueError):
    """Malformed condition / noise input to the generator."""


def condition_features(conds):
    """List of condition dicts -> [B, 13] fixed-normalization features (see module docstring)."""
    rows = []
    for c in conds:
        v = np.asarray(CND.condition_vector(c), float)                 # [w(8), target, m_total, A_wall, kappa]
        w, tgt, m_t, A, k = v[:8], v[8], v[9], v[10], v[11]
        if not (np.all(np.isfinite(v)) and m_t > 0 and A > 0 and k >= 0):
            raise GeneratorInputError(f"invalid condition vector {v.tolist()}")
        rows.append(np.concatenate([8.0 * w, [tgt, math.log(m_t), math.log(A), k / 0.3, math.log(m_t / A)]]))
    return torch.as_tensor(np.array(rows), dtype=DTYPE)


class FiLMBlock(nn.Module):
    def __init__(self, width, cond_width):
        super().__init__()
        self.norm = nn.LayerNorm(width, dtype=DTYPE)
        self.lin1 = nn.Linear(width, width, dtype=DTYPE)
        self.lin2 = nn.Linear(width, width, dtype=DTYPE)
        self.film = nn.Linear(cond_width, 2 * width, dtype=DTYPE)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)   # starts as identity modulation
        self.act = nn.SiLU()

    def forward(self, h, ce):
        gamma, beta = self.film(ce).chunk(2, dim=-1)
        y = self.lin1(self.norm(h))
        y = y * (1.0 + gamma) + beta
        return h + self.lin2(self.act(y))


class LinerGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(NOISE_DIM + COND_DIM, WIDTH, dtype=DTYPE)
        self.cond = nn.Sequential(nn.Linear(COND_DIM, WIDTH, dtype=DTYPE), nn.SiLU(),
                                  nn.Linear(WIDTH, WIDTH, dtype=DTYPE))
        self.blocks = nn.ModuleList(FiLMBlock(WIDTH, WIDTH) for _ in range(N_BLOCKS))
        self.out_norm = nn.LayerNorm(WIDTH, dtype=DTYPE)
        self.head = nn.Linear(WIDTH, PC.DIM, dtype=DTYPE)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, z, c):
        if not (torch.is_tensor(z) and torch.is_tensor(c)) or z.dtype != DTYPE or c.dtype != DTYPE:
            raise GeneratorInputError("z and c must be float64 tensors")
        if z.dim() != 2 or z.shape[1] != NOISE_DIM or c.dim() != 2 or c.shape != (z.shape[0], COND_DIM):
            raise GeneratorInputError(f"shapes z {tuple(z.shape)} / c {tuple(c.shape)} invalid")
        if not (bool(torch.isfinite(z).all()) and bool(torch.isfinite(c).all())):
            raise GeneratorInputError("non-finite generator input")
        ce = self.cond(c)
        h = self.inp(torch.cat([z, c], dim=1))
        for blk in self.blocks:
            h = blk(h, ce)
        return z[:, :PC.DIM] + self.head(nn.functional.silu(self.out_norm(h)))


def build(seed):
    """Deterministically initialized generator."""
    g = torch.Generator().manual_seed(int(seed))
    with torch.random.fork_rng():
        torch.manual_seed(int(g.initial_seed()))
        return LinerGenerator()


def sample_noise(rng, n):
    """z [n, 32]: first 18 Logistic(0, 1) (= Arm B's u, bit-identical from the same rng state), last 14 N(0, 1)."""
    return HO.sample_noise(rng, n, NOISE_DIM - PC.DIM)


def sample_u(model, conds, n_per_cond, rng):
    """Population of n_per_cond designs per condition: returns u [len(conds) * n_per_cond, 18] (condition-major) and the
    repeated condition list. Noise comes from the caller's numpy rng (registered stream) for exact reproducibility."""
    z = torch.as_tensor(sample_noise(rng, len(conds) * int(n_per_cond)), dtype=DTYPE)
    rep = [c for c in conds for _ in range(int(n_per_cond))]
    return model(z, condition_features(rep)), rep


def proposal_fn(model, cond):
    """Adapter for `optimization.seeded_cma.run_seeded_cma`: (rng, n) -> list of designs for ONE condition (Arm C / D)."""
    def fn(rng, n):
        with torch.no_grad():
            u, _ = sample_u(model, [cond], n, rng)
        return HO.designs_from_u(u.numpy())
    return fn


def architecture_specification():
    """The canonical architecture specification: the pinned configuration and the tensor schema of the model's state
    dictionary (tensor names in order, shapes, dtypes). It contains no module path, class name or source text."""
    tensors = [[name, list(t.shape), str(t.dtype).replace("torch.", "")] for name, t in build(0).state_dict().items()]
    return dict(config=CONFIG, tensors=tensors)


def public_architecture_sha256():
    """The public architecture identity: SHA-256 of the canonical JSON of `architecture_specification()`."""
    import hashlib
    import json
    spec = json.dumps(architecture_specification(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(spec.encode("ascii")).hexdigest()