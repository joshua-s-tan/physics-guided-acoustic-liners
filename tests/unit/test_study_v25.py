# -*- coding: utf-8 -*-
"""cVAE v2.5 STUDY CORE (`acoustic_liner.evaluation.historical_splits.v25`; its pre-registration is not distributed): seeds (incl. collision with
v2.4's OWN derived seeds, not just CONSUMED_SEEDS) and the development_v25 split. (phys-gen: the leakage-report,
comparator and gate tests were removed with that code.)
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner.evaluation.historical_splits import v24 as SV  # noqa: E402
from acoustic_liner.evaluation.historical_splits import v25 as SV25  # noqa: E402


# --------------------------------------------------------------------------- #
#  sec.2 -- seeds
# --------------------------------------------------------------------------- #
def test_v25_named_seeds_are_registered_and_collision_free():
    named = SV25.v25_named_seeds()
    assert set(named) == set(SV25.STREAM_NAMES) == set(SV.STREAM_NAMES)
    SV25.check_seed_collisions(named)          # must not raise
    assert len(set(named.values())) == len(named)


def test_v25_seed_collision_with_v24_seed_is_a_hard_failure():
    named = dict(SV25.v25_named_seeds())
    v24_seed = next(iter(SV.v24_named_seeds().values()))
    named["model_training"] = v24_seed          # force a collision with a REAL v2.4 derived seed
    with pytest.raises(SV25.V25Error):
        SV25.check_seed_collisions(named)


def test_v25_seed_collision_with_consumed_seed_is_a_hard_failure():
    named = dict(SV25.v25_named_seeds())
    named["bootstrap"] = next(iter(SV.CONSUMED_SEEDS))
    with pytest.raises(SV.V24Error):           # study_v24.check_seed_collisions raises its own error type
        SV25.check_seed_collisions(named)


def test_v25_root_seed_is_deterministic_and_distinct_from_v24():
    assert SV25.v25_root_seed() == SV25.v25_root_seed()
    assert SV25.v25_root_seed() != SV.v24_root_seed()


# --------------------------------------------------------------------------- #
#  sec.3 -- development_v25 split
# --------------------------------------------------------------------------- #
def test_build_development_v25_is_deterministic_and_sized():
    a = SV25.build_development_v25(12345)
    b = SV25.build_development_v25(12345)
    c = SV25.build_development_v25(54321)
    assert len(a) == SV25.DEVELOPMENT_V25_N == 30
    from acoustic_liner.evaluation import conditions as COND
    assert COND.split_hash(a) == COND.split_hash(b)
    assert COND.split_hash(a) != COND.split_hash(c)
