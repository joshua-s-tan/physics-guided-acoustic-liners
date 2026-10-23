# -*- coding: utf-8 -*-
r"""Historical cVAE v2.4 study: its seeds and the development_v24 split.

Kept only so that the leakage barrier (`evaluation.splits`) can rebuild the conditions this study consumed.
The seed-domain labels are frozen inputs and are kept byte for byte. The rest of the
historical study, including its pre-registration, is not distributed.
`CONSUMED_SEEDS` lists the seeds used before v2.4.
"""

from __future__ import annotations

import hashlib
import numpy as np

from .. import conditions as COND


class V24Error(RuntimeError):
    """Fail-closed v2.4 study error (seeds, leakage, accounting, or an incomplete/undefined result)."""


# Seeds already consumed by earlier evaluations (v2.1 dev/confirmatory, v2.2 confirmatory, harness defaults).
# Moved here unchanged from cvae_study.CONSUMED_SEEDS.
CONSUMED_SEEDS = frozenset({27182, 90271, 13513, 51413, 77291, 60623, 31415, 20260928, 777})


# --------------------------------------------------------------------------- #
#  sec.2 -- seeds (no discretion)
# --------------------------------------------------------------------------- #
ROOT_LABEL = b"casef-v2.4-development-v1"
STREAM_NAMES = ("split_construction", "model_training", "latent_banks", "cma", "bootstrap",
                "expert_data_generation")


def v24_root_seed() -> int:
    return int.from_bytes(hashlib.sha256(ROOT_LABEL).digest()[:8], "big")


def v24_streams():
    """The 6 named independent streams, in the fixed registered order (sec.2)."""
    children = np.random.SeedSequence(v24_root_seed()).spawn(6)
    return dict(zip(STREAM_NAMES, children))


def derived_seed(stream_ss) -> int:
    """One deterministic plain-integer representative of a SeedSequence stream, for use wherever a single
    32-bit seed integer is required (manifest fields, `evaluation.random_streams.stream` base seeds)."""
    return int(stream_ss.generate_state(1, dtype=np.uint32)[0])


def v24_named_seeds():
    return {name: derived_seed(ss) for name, ss in v24_streams().items()}


def check_seed_collisions(named_seeds):
    """sec.2: 'The only check is literal collision with CONSUMED_SEEDS; a collision is a hard failure, not a
    redraw.' Checked against v2.3's registered CONSUMED_SEEDS (the study's own hard-coded consumed-seed set)."""
    bad = {name: s for name, s in named_seeds.items() if s in CONSUMED_SEEDS}
    if bad:
        raise V24Error(f"v2.4 derived seed(s) collide with CONSUMED_SEEDS: {bad}")
    vals = list(named_seeds.values())
    if len(set(vals)) != len(vals):
        raise V24Error(f"v2.4 derived seeds are not pairwise distinct: {named_seeds}")


# --------------------------------------------------------------------------- #
#  sec.3 -- development_v24 split + its own extra leakage check vs the OLD development split
# --------------------------------------------------------------------------- #
DEVELOPMENT_V24_N = 30


def build_development_v24(seed):
    """Generated ONLY by the manifest builder (sec.2/sec.3); never called at run time."""
    return COND.synthetic_conditions(DEVELOPMENT_V24_N, seed, "development_v24")
