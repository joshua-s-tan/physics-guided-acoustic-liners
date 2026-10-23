# -*- coding: utf-8 -*-
"""cVAE v2.6 STUDY CORE (`acoustic_liner.evaluation.historical_splits.v26`; sec.8 of its pre-registration, which is not distributed): seeds, collision
checks against v2.4 AND v2.5's own derived seeds (not just CONSUMED_SEEDS), and that `build_development_v26`
is defined but this module never calls it itself (gate-phase scope)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner.evaluation.historical_splits import v24 as SV  # noqa: E402
from acoustic_liner.evaluation.historical_splits import v25 as SV25  # noqa: E402
from acoustic_liner.evaluation.historical_splits import v26 as SV26  # noqa: E402


def test_v26_named_seeds_are_registered_and_collision_free():
    named = SV26.v26_named_seeds()
    assert set(named) == set(SV26.STREAM_NAMES) == set(SV.STREAM_NAMES)
    SV26.check_seed_collisions(named)          # must not raise
    assert len(set(named.values())) == len(named)


def test_v26_seed_collision_with_v24_seed_is_a_hard_failure():
    named = dict(SV26.v26_named_seeds())
    v24_seed = next(iter(SV.v24_named_seeds().values()))
    named["model_training"] = v24_seed
    with pytest.raises(SV26.V26Error):
        SV26.check_seed_collisions(named)


def test_v26_seed_collision_with_v25_seed_is_a_hard_failure():
    named = dict(SV26.v26_named_seeds())
    v25_seed = next(iter(SV25.v25_named_seeds().values()))
    named["cma"] = v25_seed
    with pytest.raises(SV26.V26Error):
        SV26.check_seed_collisions(named)


def test_v26_seed_collision_with_consumed_seed_is_a_hard_failure():
    named = dict(SV26.v26_named_seeds())
    named["bootstrap"] = next(iter(SV.CONSUMED_SEEDS))
    with pytest.raises(SV.V24Error):           # study_v24.check_seed_collisions raises its own error type
        SV26.check_seed_collisions(named)


def test_v26_root_seed_is_deterministic_and_distinct_from_v24_and_v25():
    assert SV26.v26_root_seed() == SV26.v26_root_seed()
    assert SV26.v26_root_seed() != SV.v24_root_seed()
    assert SV26.v26_root_seed() != SV25.v25_root_seed()


def test_build_development_v26_is_defined_and_produces_the_registered_count():
    # DEFINED for a later phase of this study; this test only checks it works, never calling it from any
    # gate-phase code path (study_v26/v26_objective/v26_invert/v26_audit never import or call it).
    conds = SV26.build_development_v26(SV26.v26_named_seeds()["split_construction"])
    assert len(conds) == SV26.DEVELOPMENT_V26_N
    assert all(c["group"] == "development_v26" for c in conds)
