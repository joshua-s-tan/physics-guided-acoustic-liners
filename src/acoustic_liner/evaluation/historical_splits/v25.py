# -*- coding: utf-8 -*-
r"""Historical cVAE v2.5 study: its seeds and the development_v25 split.

Kept only so that the leakage barrier (`evaluation.splits`) can rebuild the conditions this study consumed.
The seed-domain labels are frozen inputs and are kept byte for byte. The rest of the
historical study, including its pre-registration, is not distributed.
"""

from __future__ import annotations

import hashlib
import numpy as np

from .. import conditions as COND
from . import v24 as SV


class V25Error(RuntimeError):
    """Fail-closed v2.5 study error (seeds, leakage, accounting, or an incomplete/undefined result)."""


# --------------------------------------------------------------------------- #
#  sec.2 -- seeds (no discretion)
# --------------------------------------------------------------------------- #
ROOT_LABEL = b"casef-v2.5-development-v1"
STREAM_NAMES = SV.STREAM_NAMES          # identical fixed order: split_construction, model_training, latent_banks,
derived_seed = SV.derived_seed          # one deterministic plain-integer representative of a SeedSequence stream


def v25_root_seed() -> int:
    return int.from_bytes(hashlib.sha256(ROOT_LABEL).digest()[:8], "big")


def v25_streams():
    """The 6 named independent streams, in the fixed registered order (sec.2)."""
    children = np.random.SeedSequence(v25_root_seed()).spawn(6)
    return dict(zip(STREAM_NAMES, children))


def v25_named_seeds():
    return {name: derived_seed(ss) for name, ss in v25_streams().items()}


def check_seed_collisions(named_seeds):
    """sec.2: 'a collision is a hard failure, not a redraw.' Checked against v2.3's CONSUMED_SEEDS (via
    `evaluation.historical_splits.v24.check_seed_collisions`, reused unmodified) PLUS the v2.4 derived seeds themselves
    (`evaluation.historical_splits.v24.v24_named_seeds()`), since a v2.5 seed silently equal to a v2.4 one would let a v2.5 run share
    common-random-number draws with a v2.4 run it is meant to be independent of -- `evaluation.historical_splits.v24`'s own collision
    check has no way to know about v2.5 seeds, so this is genuinely new here, not a duplicate."""
    SV.check_seed_collisions(named_seeds)
    v24_seeds = SV.v24_named_seeds()
    bad = {name: s for name, s in named_seeds.items() if s in set(v24_seeds.values())}
    if bad:
        raise V25Error(f"v2.5 derived seed(s) collide with v2.4 derived seeds: {bad} (v2.4: {v24_seeds})")


# --------------------------------------------------------------------------- #
#  sec.3 -- development_v25 split + its own extra leakage checks vs BOTH `development` and `development_v24`
# --------------------------------------------------------------------------- #
DEVELOPMENT_V25_N = 30


def build_development_v25(seed):
    """Generated ONLY by the manifest builder (sec.2/sec.3); never called at run time. Committed AFTER this
    document (prereg preamble: 'generated and inspected only AFTER this document is committed')."""
    return COND.synthetic_conditions(DEVELOPMENT_V25_N, seed, "development_v25")
