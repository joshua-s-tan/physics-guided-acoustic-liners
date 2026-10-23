# -*- coding: utf-8 -*-
r"""Historical cVAE v2.7 study: its seeds and the development and pilot split builders, which `v28` also uses.

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
from . import v26 as SV26

ROOT_LABEL = b"casef-v2.7-development-v1"
PILOT_LABEL = b"casef-v2.7-pilot-v1"          # amendment r2: pilot runs on a SEPARATE split, never development_v27
STREAM_NAMES = SV.STREAM_NAMES
derived_seed = SV.derived_seed

DEVELOPMENT_V27_N = 30


class V27Error(RuntimeError):
    pass


def v27_root_seed() -> int:
    return int.from_bytes(hashlib.sha256(ROOT_LABEL).digest()[:8], "big")


def v27_streams():
    return dict(zip(STREAM_NAMES, np.random.SeedSequence(v27_root_seed()).spawn(6)))


def v27_named_seeds():
    return {name: derived_seed(ss) for name, ss in v27_streams().items()}


def check_seed_collisions(named_seeds):
    """Collision-checked vs CONSUMED_SEEDS + v2.4 + v2.5 + v2.6 derived seeds (prereg sec.8)."""
    SV.check_seed_collisions(named_seeds)
    for tag, seeds in (("v2.4", SV.v24_named_seeds()), ("v2.5", SV25.v25_named_seeds()),
                       ("v2.6", SV26.v26_named_seeds())):
        bad = {n: s for n, s in named_seeds.items() if s in set(seeds.values())}
        if bad:
            raise V27Error(f"v2.7 derived seed(s) collide with {tag} seeds: {bad}")


def build_development_v27(seed):
    return COND.synthetic_conditions(DEVELOPMENT_V27_N, seed, "development_v27")


def pilot_split_seed() -> int:
    return int.from_bytes(hashlib.sha256(PILOT_LABEL).digest()[:8], "big")


def build_pilot_v27(n):
    """Separate pilot conditions (distinct root label -> disjoint from development_v27). Go/no-go only; never
    consumed by the confirmatory run (prereg amendment r2 point 1)."""
    return COND.synthetic_conditions(n, pilot_split_seed(), "pilot_v27")
