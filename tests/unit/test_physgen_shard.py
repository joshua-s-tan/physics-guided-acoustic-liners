# -*- coding: utf-8 -*-
"""PHYS-GEN-V1 sharded evaluation on TRAINING-ONLY conditions: sharded == single-machine (cells and aggregate),
resume never rerolls, inputs must hash identically, merge fails closed. No evaluation split is opened."""
import json
import os
import shutil

import pytest

from acoustic_liner.evaluation import splits as PCN
from acoustic_liner.evaluation import shards as SH
from acoustic_liner.models import training as TRAIN
from acoustic_liner.evaluation import study as ST

SEEDS, BUDGETS = (0, 1), (40,)


@pytest.fixture(autouse=True)
def _clean_tree(monkeypatch):
    monkeypatch.setattr(SH, "git_src_tree", lambda repo=".": "test-tree")       # the dev tree is not committed here


@pytest.fixture(scope="module")
def conds():
    tr, va = PCN.training_splits(workers=4, n_train=TRAIN.SMOKE_CONFIG["n_train"], n_val=TRAIN.SMOKE_CONFIG["n_val"])
    return va[:3]


@pytest.fixture(scope="module")
def checkpoint(conds, tmp_path_factory):
    tr, va = PCN.training_splits(workers=4, n_train=TRAIN.SMOKE_CONFIG["n_train"], n_val=TRAIN.SMOKE_CONFIG["n_val"])
    out = str(tmp_path_factory.mktemp("ck"))
    TRAIN.Trainer(dict(TRAIN.SMOKE_CONFIG, epochs=1, val_every=1, val_pop=8), tr, va, frozenset(), out).run()
    return os.path.join(out, "selected.pt")


@pytest.fixture
def manifest(conds, checkpoint, tmp_path):
    m = SH.build_manifest("training_smoke", conds, SEEDS, BUDGETS, checkpoint, weights=(1, 1),
                          canary_conds=conds[:1], canary_budget=40)
    p = str(tmp_path / "manifest.json")
    SH.write_manifest(m, p)
    return p


def test_assign_shards_contiguous_and_weighted():
    assert SH.assign_shards(100, (1, 1)) == [0] * 50 + [1] * 50
    assert SH.assign_shards(100, (0.6, 0.4)) == [0] * 60 + [1] * 40
    assert SH.assign_shards(3, (1, 1)) == [0, 0, 1]
    with pytest.raises(SH.ShardError):
        SH.assign_shards(10, (1, 0))


def test_manifest_is_full_grid_and_tamper_evident(manifest):
    m = SH.load_manifest(manifest)
    assert len(m["cells"]) == 3 * len(SEEDS) * len(BUDGETS)
    assert all(len({c["shard"] for c in m["cells"] if c["cond_index"] == i}) == 1 for i in range(3))
    with pytest.raises(SH.ShardError):
        SH.write_manifest(m, manifest)                                            # written once
    bad = dict(m, seeds=[0, 1, 2]); json.dump(bad, open(manifest, "w"))
    with pytest.raises(SH.ShardError):
        SH.load_manifest(manifest)


def test_sharded_equals_single_machine(conds, checkpoint, manifest, tmp_path):
    single = ST.evaluate("training_smoke", conds, checkpoint, BUDGETS, SEEDS, workers=2)   # 1-thread workers
    d0, d1 = str(tmp_path / "m0"), str(tmp_path / "m1")
    SH.run_shard(manifest, 0, d0, checkpoint, conds=conds, progress=lambda s: None, canary_conds=conds[:1])
    SH.run_shard(manifest, 1, d1, checkpoint, workers=2, conds=conds, progress=lambda s: None, canary_conds=conds[:1])   # pool path too
    merged = SH.merge(manifest, [d0, d1])
    assert merged["cells"] == SH._canon(single["cells"])
    assert merged["aggregate"] == SH._canon(single["aggregate"])
    assert merged["split_hash"] == single["split_hash"]


def test_resume_runs_only_missing_and_never_replaces(conds, checkpoint, manifest, tmp_path):
    d0 = str(tmp_path / "m0")
    assert SH.run_shard(manifest, 0, d0, checkpoint, conds=conds, progress=lambda s: None, canary_conds=conds[:1]) == 4
    files = sorted(os.listdir(d0))
    stamp = {f: os.stat(os.path.join(d0, f)).st_mtime_ns for f in files}
    os.remove(os.path.join(d0, files[0]))
    assert SH.run_shard(manifest, 0, d0, checkpoint, conds=conds, progress=lambda s: None, canary_conds=conds[:1]) == 1
    assert all(os.stat(os.path.join(d0, f)).st_mtime_ns == stamp[f] for f in files[1:])
    assert SH.run_shard(manifest, 0, d0, checkpoint, conds=conds, progress=lambda s: None, canary_conds=conds[:1]) == 0


def test_shard_refuses_foreign_cells_inputs_and_conditions(conds, checkpoint, manifest, tmp_path, monkeypatch):
    with pytest.raises(SH.ShardError):
        SH.run_shard(manifest, 2, str(tmp_path / "x"), checkpoint, conds=conds)               # no such shard
    d1 = str(tmp_path / "m1")
    SH.run_shard(manifest, 1, d1, checkpoint, conds=conds, progress=lambda s: None, canary_conds=conds[:1])
    with pytest.raises(SH.ShardError):                                                        # shard-1 records in a
        SH.run_shard(manifest, 0, d1, checkpoint, conds=conds, progress=lambda s: None, canary_conds=conds[:1])       # shard-0 directory
    with pytest.raises(SH.ShardError):
        SH.run_shard(manifest, 0, str(tmp_path / "y"), checkpoint, conds=conds[::-1])         # different conditions
    monkeypatch.setattr(SH, "dependency_sha256", lambda: "other machine")
    with pytest.raises(SH.ShardError):
        SH.run_shard(manifest, 0, str(tmp_path / "z"), checkpoint, conds=conds)               # input hash differs
    with pytest.raises(SH.ShardError):                                                        # non-training conditions
        SH.run_shard(manifest, 0, str(tmp_path / "w"), checkpoint,
                     conds=[dict(c, group="final_physgen_v1") for c in conds])


def test_canary_must_reproduce_bit_for_bit(conds, checkpoint, manifest, tmp_path):
    m = SH.load_manifest(manifest)
    SH.check_canary(m, conds[:1], checkpoint)                                                 # this machine: exact
    with pytest.raises(SH.ShardError):
        SH.check_canary(m, conds[1:2], checkpoint)                                            # other conditions
    body = {k: v for k, v in m.items() if k != "manifest_sha256"}
    body["canary"]["records"][0]["D"]["best_val"] += 1e-15                                    # one ulp-scale change
    bad = dict(body, manifest_sha256=SH._sha_obj(body))
    p = str(tmp_path / "bad.json"); json.dump(bad, open(p, "w"))
    with pytest.raises(SH.ShardError, match="bit-for-bit"):
        SH.run_shard(p, 0, str(tmp_path / "r"), checkpoint, conds=conds, progress=lambda s: None,
                     canary_conds=conds[:1])
    assert not os.listdir(str(tmp_path / "r"))                                                # nothing scored


def test_merge_fails_closed(conds, checkpoint, manifest, tmp_path):
    d0, d1 = str(tmp_path / "m0"), str(tmp_path / "m1")
    SH.run_shard(manifest, 0, d0, checkpoint, conds=conds, progress=lambda s: None, canary_conds=conds[:1])
    SH.run_shard(manifest, 1, d1, checkpoint, conds=conds, progress=lambda s: None, canary_conds=conds[:1])
    with pytest.raises(SH.ShardError, match="missing"):
        SH.merge(manifest, [d0])
    dup = str(tmp_path / "dup"); shutil.copytree(d1, dup)
    with pytest.raises(SH.ShardError, match="duplicate"):
        SH.merge(manifest, [d0, d1, dup])
    f = sorted(os.listdir(d1))[0]
    shutil.copy(os.path.join(d1, f), os.path.join(d1, "c999-s0-b40.json"))
    with pytest.raises(SH.ShardError, match="unexpected"):
        SH.merge(manifest, [d0, d1])
    os.remove(os.path.join(d1, "c999-s0-b40.json"))
    r = json.load(open(os.path.join(d1, f))); r["manifest_sha256"] = "0" * 64
    json.dump(r, open(os.path.join(d1, f), "w"))
    with pytest.raises(SH.ShardError):
        SH.merge(manifest, [d0, d1])
    r = json.load(open(os.path.join(d0, sorted(os.listdir(d0))[0])))
    del r["cell"]["D"]; json.dump(r, open(os.path.join(d0, sorted(os.listdir(d0))[0]), "w"))
    with pytest.raises(SH.ShardError, match="missing arms"):
        SH.merge(manifest, [d0])
