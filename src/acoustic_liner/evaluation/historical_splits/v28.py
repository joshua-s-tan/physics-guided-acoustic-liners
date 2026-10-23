# -*- coding: utf-8 -*-
r"""Historical cVAE v2.8 study: its seeds, its splits, the expert conditions and
`eval_split_condition_hashes`.

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
from . import v27 as SV27
ROOT_LABEL = b"casef-v2.8-development-v1"
PILOT_LABEL = b"casef-v2.8-pilot-v1"
STREAM_NAMES = SV.STREAM_NAMES                    # ('split_construction','model_training','latent_banks','cma','bootstrap','expert_data_generation')
derived_seed = SV.derived_seed
DEVELOPMENT_V28_N = 30
PILOT_V28_N = 10


class V28Error(RuntimeError):
    pass


def v28_root_seed() -> int:
    return int.from_bytes(hashlib.sha256(ROOT_LABEL).digest()[:8], "big")


def pilot_split_seed() -> int:
    return int.from_bytes(hashlib.sha256(PILOT_LABEL).digest()[:8], "big")


def v28_streams():
    return dict(zip(STREAM_NAMES, np.random.SeedSequence(v28_root_seed()).spawn(len(STREAM_NAMES))))


def v28_named_seeds():
    return {name: derived_seed(ss) for name, ss in v28_streams().items()}


def check_seed_collisions(named_seeds):
    """Collision-checked vs CONSUMED_SEEDS + v2.4/2.5/2.6/2.7 derived seeds (prereg sec. 12)."""
    SV.check_seed_collisions(named_seeds)
    for tag, seeds in (("v2.4", SV.v24_named_seeds()), ("v2.5", SV25.v25_named_seeds()),
                       ("v2.6", SV26.v26_named_seeds()), ("v2.7", SV27.v27_named_seeds())):
        bad = {n: s for n, s in named_seeds.items() if s in set(seeds.values())}
        if bad:
            raise V28Error(f"v2.8 derived seed(s) collide with {tag} seeds: {bad}")
    return named_seeds


def build_development_v28(seed):
    """Fresh headline split (prereg sec. 8/12), disjoint from every prior split (hash-checked by callers)."""
    return COND.synthetic_conditions(DEVELOPMENT_V28_N, seed, "development_v28")


def build_pilot_v28(n=PILOT_V28_N):
    """Separate pilot split (distinct label -> disjoint from development_v28 and all priors; prereg sec. 11-12)."""
    return COND.synthetic_conditions(n, pilot_split_seed(), "pilot_v28")


def expert_conditions_v28(n, seed):
    """Training-only expert conditions (prereg sec. 7): synthetic family, group 'training', the expert-data stream
    seed. NEVER evaluated; leakage-gated disjoint from all eval splits by v28_expertgen."""
    return COND.synthetic_conditions(n, seed, "training")


# --- leakage: hashes of every split the expert corpus / archive must stay disjoint from (prereg sec. 7) ------- #
def eval_split_condition_hashes():
    """SHA-256 condition hashes of every split the expert corpus must NOT overlap (prereg sec. 7)."""
    hashes = set()
    sources = []
    try:
        sources.append(build_development_v28(v28_named_seeds()["split_construction"]))
    except Exception:
        pass
    sources.append(build_pilot_v28())
    sources.append(SV27.build_development_v27(SV27.v27_named_seeds()["split_construction"]))
    try:
        sources.append(SV27.build_pilot_v27(10))
    except Exception:
        pass
    for conds in sources:
        for c in conds:
            hashes.add(COND.condition_hash(c))
    # sealed/eval splits (final_test, falcon) via `evaluation.conditions` committed seeds
    try:
        fin = COND.synthetic_conditions(100, 70207, "final_test")
        for c in fin:
            hashes.add(COND.condition_hash(c))
    except Exception:
        pass
    try:
        hashes.add(COND.condition_hash(COND.falcon_condition()))
    except Exception:
        pass
    return hashes
