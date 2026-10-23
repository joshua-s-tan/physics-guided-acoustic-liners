"""The registered random streams: one independent generator per (base seed, condition, run seed,
purpose).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
#  Deterministic streams
# --------------------------------------------------------------------------- #
def stream(base_seed, cond_index, seed, tag):
    """Independent, registered RNG stream per (base seed, condition, run seed, purpose)."""
    tags = {"random": 1, "cvae": 2, "cma_start": 3, "cma": 4}
    return np.random.default_rng(np.random.SeedSequence([int(base_seed), int(cond_index), int(seed), tags[tag]]))
