# -*- coding: utf-8 -*-
"""cVAE v2.8 implementation-conformance tests (historical cVAE v2.8 pre-registration, sealed r4; not distributed).

phys-gen: only the seed and split tests remain (study_v28 was slimmed to those)."""
import pytest

pytest.importorskip("torch")

from acoustic_liner.evaluation.historical_splits import v28 as S


def test_seeds_collision_free_and_streams():
    named = S.check_seed_collisions(S.v28_named_seeds())
    assert "model_training" in named and "expert_data_generation" in named


def test_expert_conditions_leakage_disjoint():
    """Expert (training) conditions must be disjoint from every eval split (sec. 7)."""
    conds = S.expert_conditions_v28(30, S.v28_named_seeds()["expert_data_generation"])
    from acoustic_liner.evaluation import conditions as COND
    evalh = S.eval_split_condition_hashes()
    assert not any(COND.condition_hash(c) in evalh for c in conds)

