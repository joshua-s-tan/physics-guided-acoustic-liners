# -*- coding: utf-8 -*-
r"""Historical cVAE v2.6 study: its seeds and the development_v26 split.

Kept only so that the leakage barrier (`evaluation.splits`) can rebuild the conditions this study consumed.
The seed-domain labels are frozen inputs and are kept byte for byte. The rest of the
historical study, including its pre-registration, is not distributed.
"""

from __future__ import annotations

import hashlib
import numpy as np

from .. import conditions as COND
from . import v24 as SV
from . import v25 as SV25


class V26Error(RuntimeError):
    """Fail-closed v2.6 study error (seeds, leakage, or an incomplete/undefined result)."""


# --------------------------------------------------------------------------- #
#  sec.8 -- seeds (no discretion). Same fixed stream names/order as v2.4/v2.5.
# --------------------------------------------------------------------------- #
ROOT_LABEL = b"casef-v2.6-development-v1"
STREAM_NAMES = SV.STREAM_NAMES     # split_construction, model_training, latent_banks, cma, bootstrap,
derived_seed = SV.derived_seed


def v26_root_seed() -> int:
    return int.from_bytes(hashlib.sha256(ROOT_LABEL).digest()[:8], "big")


def v26_streams():
    """The 6 named independent streams, in the fixed registered order (sec.8)."""
    children = np.random.SeedSequence(v26_root_seed()).spawn(6)
    return dict(zip(STREAM_NAMES, children))


def v26_named_seeds():
    return {name: derived_seed(ss) for name, ss in v26_streams().items()}


def check_seed_collisions(named_seeds):
    """sec.8: collision-checked against CONSUMED_SEEDS (via `evaluation.historical_splits.v24`, reused unmodified) PLUS the v2.4 AND
    v2.5 derived seeds themselves -- a v2.6 seed silently equal to either would let a v2.6 run share common
    random numbers with a run it must be independent of."""
    SV.check_seed_collisions(named_seeds)
    v24_seeds = SV.v24_named_seeds()
    v25_seeds = SV25.v25_named_seeds()
    bad_24 = {name: s for name, s in named_seeds.items() if s in set(v24_seeds.values())}
    if bad_24:
        raise V26Error(f"v2.6 derived seed(s) collide with v2.4 derived seeds: {bad_24} (v2.4: {v24_seeds})")
    bad_25 = {name: s for name, s in named_seeds.items() if s in set(v25_seeds.values())}
    if bad_25:
        raise V26Error(f"v2.6 derived seed(s) collide with v2.5 derived seeds: {bad_25} (v2.5: {v25_seeds})")


# --------------------------------------------------------------------------- #
#  sec.8 -- development_v26 split. DEFINED, NOT INVOKED in the gate phase (see module docstring).
# --------------------------------------------------------------------------- #
DEVELOPMENT_V26_N = 30


def build_development_v26(seed):
    """Generated ONLY by a future manifest builder, in a LATER phase of this study -- never called by the gate
    phase (this commit does not generate, inspect or consume `development_v26`)."""
    return COND.synthetic_conditions(DEVELOPMENT_V26_N, seed, "development_v26")
