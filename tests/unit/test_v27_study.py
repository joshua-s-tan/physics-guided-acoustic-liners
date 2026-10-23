# -*- coding: utf-8 -*-
"""cVAE v2.7 study wiring: seeds collision-free vs v2.4/2.5/2.6, dev split shape.

phys-gen: the gate test was removed with that code."""
import pytest

pytest.importorskip("torch")

from acoustic_liner.evaluation.historical_splits import v27 as S27


def test_seeds_collision_free():
    named = S27.v27_named_seeds()
    assert set(named) == set(S27.STREAM_NAMES)
    S27.check_seed_collisions(named)                # raises on any collision vs CONSUMED/v2.4/v2.5/v2.6


def test_root_seed_is_label_derived():
    import hashlib
    assert S27.v27_root_seed() == int.from_bytes(hashlib.sha256(b"casef-v2.7-development-v1").digest()[:8], "big")


def test_development_v27_shape():
    conds = S27.build_development_v27(S27.v27_named_seeds()["split_construction"])
    assert len(conds) == S27.DEVELOPMENT_V27_N == 30
    for c in conds[:3]:
        assert len(c["spl"]) == 8
        assert c["m_total_target"] > 0 and c["A_wall"] > 0
