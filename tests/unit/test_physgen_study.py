# -*- coding: utf-8 -*-
"""PHYS-GEN-V1 study driver: arms A/B/C/D with matched exact-charged budgets, the legacy diagnostic, the clustered
bootstrap, the frozen-split and evaluation-split checks and checkpoint compatibility. TRAINING-ONLY conditions are
evaluated; no evaluation split is run."""
import argparse
import json
import os

import numpy as np
import pytest

from acoustic_liner.evaluation import splits as PCN
from acoustic_liner.models import linergen as PM
from acoustic_liner.models import checkpoint as CKPT
from acoustic_liner.models import training as TRAIN
from acoustic_liner.evaluation import statistics as STAT
from acoustic_liner.evaluation import study as ST

BUDGET = 40


@pytest.fixture(scope="module")
def splits():
    return PCN.training_splits(workers=4, n_train=TRAIN.SMOKE_CONFIG["n_train"], n_val=TRAIN.SMOKE_CONFIG["n_val"])


@pytest.fixture(scope="module")
def checkpoint(splits, tmp_path_factory):
    tr, va = splits
    out = str(tmp_path_factory.mktemp("ck"))
    TRAIN.Trainer(dict(TRAIN.SMOKE_CONFIG, epochs=1, val_every=1, val_pop=8), tr, va, frozenset(), out).run()
    return os.path.join(out, "selected.pt")


def test_untrained_generator_arm_d_equals_arm_b(splits):
    tr, va = splits
    rec_m = PM.build(PCN.named_seeds()["model_init"])
    from acoustic_liner.optimization import seeded_cma as HO
    base = PCN.named_seeds()["evaluation"]
    B = HO.run_seeded_cma(va[0], 0, 0, BUDGET, base, HO.random_proposals, "armB")
    D = HO.run_seeded_cma(va[0], 0, 0, BUDGET, base, PM.proposal_fn(rec_m, va[0]), "armD")
    C = HO.run_seeded_cma(va[0], 0, 0, BUDGET, base, PM.proposal_fn(rec_m, va[0]), "armC", cma=False)
    assert [e["geometry"] for e in D.trace] == [e["geometry"] for e in B.trace]
    assert [e["best_val"] for e in D.trace] == [e["best_val"] for e in B.trace]
    assert [e["geometry"] for e in C.trace] == [e["geometry"] for e in D.trace][:len(C.trace)]


def test_cell_matched_budgets(splits):
    tr, va = splits
    rec = ST.run_cell(va[1], 1, 0, BUDGET, PCN.named_seeds()["evaluation"], PM.build(0))
    assert rec["A"]["charged"] == rec["B"]["charged"] == rec["D"]["charged"] == BUDGET
    assert rec["C"]["charged"] == 32 and rec["C"]["n_proposal_charged"] + rec["C"]["n_fill_charged"] == 32
    for a in "ABCD":
        assert rec[a]["feasible"] and rec[a]["mass_ok"]
    assert rec["proposals"]["raw_feasibility"] == 1.0 and rec["proposals"]["unique_fraction"] >= 0.95
    agg = STAT.aggregate([rec], [BUDGET])[str(BUDGET)]
    assert "legacy" not in str(sorted(agg))                                      # never compared or gated


def test_evaluate_training_smoke_end_to_end(splits, checkpoint):
    tr, va = splits
    ood = PCN.OODScorer(tr, va)
    seen = []
    res = ST.evaluate("training_smoke", va[:1], checkpoint, (BUDGET,), (0,), ood=ood,
                      progress=lambda k, cell: seen.append((k, cell["cond_index"])))
    assert seen == [(1, 0)]
    c = res["cells"][0]
    assert c["A"]["charged"] == c["B"]["charged"] == c["D"]["charged"] == BUDGET
    assert res["checkpoint"]["model_sha256"] and len(res["ood"]["per_condition"]) == 1
    assert set(res["aggregate"][str(BUDGET)]) >= {"D_minus_A", "D_minus_B", "C32_minus_B32"}
    assert c["C"]["initial32_best_val"] == c["C"]["best_val"] and c["B"]["initial32_best_val"] <= c["B"]["best_val"]


def test_evaluation_splits_open_only_on_the_released_conditions(splits, checkpoint):
    """An evaluation split is evaluated only on the conditions the released study defined (same hash); training
    conditions under an evaluation split's name, and non-training conditions under the smoke name, are refused."""
    tr, va = splits
    for split in ST.EVAL_SPLITS:
        with pytest.raises(ST.StudyError, match="do not hash to the released split"):
            ST.evaluate(split, va[:1], checkpoint, (BUDGET,), (0,))
    with pytest.raises(ST.StudyError, match="unknown evaluation split"):
        ST.check_evaluation_split("other_split", va[:1])
    with pytest.raises(ST.StudyError):                                           # non-training conditions refused
        ST.evaluate("training_smoke", [dict(va[0], group="development_physgen_v1")], checkpoint, (BUDGET,), (0,))


@pytest.mark.parametrize("split", ST.EVAL_SPLITS)
def test_the_split_generator_reproduces_the_released_evaluation_split(split):
    """Rebuilt from the generator alone, with no file of the study: each evaluation split hashes to the value the
    released study recorded."""
    conds = PCN.evaluation_split(split)
    ST.check_evaluation_split(split, conds)
    assert PCN.split_hash(conds) == ST.EVALUATION_SPLIT_HASHES[split]


def test_the_frozen_split_file_must_be_the_released_one(splits, tmp_path, monkeypatch):
    """`train` and evaluation read the training / validation split only from a file that hashes to FROZEN_SPLITS."""
    tr, va = splits
    split_dir = str(tmp_path / "splits")
    PCN.training_splits(cache_dir=split_dir, n_train=len(tr), n_val=len(va), workers=4)
    monkeypatch.setattr(ST, "SPLIT_DIR", split_dir)
    small = PCN.frozen_split_path(split_dir, len(tr), len(va))
    os.replace(small, PCN.frozen_split_path(split_dir))            # a valid split file, not the released one
    with pytest.raises(PCN.PhysgenConditionError, match="expected hashes"):
        ST.frozen_splits()
    assert set(ST.FROZEN_SPLITS) == {"training_hash", "validation_hash", "file_sha256"}


def test_no_study_command_reads_or_writes_a_document():
    """The seal and lock commands are gone; the study driver names no file under docs/."""
    import inspect
    src = inspect.getsource(ST)
    assert "docs" not in src.replace("docstring", "") and "SEAL" not in src and "LOCK" not in src
    with pytest.raises(SystemExit):
        ST.main(["seal"])
    with pytest.raises(SystemExit):
        ST.main(["lock"])


def test_an_incompatible_checkpoint_is_refused(checkpoint, tmp_path):
    """The loader refuses what this code cannot reproduce: another architecture or coordinate transform, a
    configuration that does not match its hash, other seeds, or edited weights."""
    import torch
    ck = torch.load(checkpoint, weights_only=False)
    assert ck["binding"]["provenance_schema_version"] == CKPT.BINDING_VERSION
    assert "package_source_sha256" in ck["binding"] and "public_model_source_sha256" not in ck["binding"]
    CKPT.load_checkpoint(checkpoint)
    edits = {
        "arch": lambda c: c["binding"].update(public_architecture_sha256="0" * 64),
        "coordinate_transform": lambda c: c["binding"].update(public_coordinate_transform_sha256="0" * 64),
        "config": lambda c: c["binding"]["config"].update(lr=1.0),
        "seeds": lambda c: c["binding"].update(seeds=dict(c["binding"]["seeds"], evaluation=-1)),
        "weights": lambda c: c["model"].__setitem__(next(iter(c["model"])), c["model"][next(iter(c["model"]))] + 1.0),
    }
    for name, change in edits.items():
        c = torch.load(checkpoint, weights_only=False)
        change(c)
        p = tmp_path / f"{name}.pt"
        torch.save(c, p)
        with pytest.raises(CKPT.CheckpointError):
            CKPT.load_checkpoint(str(p))


def test_an_edited_source_file_does_not_lock_the_checkpoint_out(checkpoint, monkeypatch):
    """The source fingerprint the checkpoint recorded is information only: a different fingerprint now loads."""
    from acoustic_liner.provenance import hashing as H
    monkeypatch.setattr(H, "package_source_sha256", lambda: "f" * 64)
    CKPT.load_checkpoint(checkpoint)


def test_one_split_at_a_time(tmp_path, monkeypatch):
    """Pilot opens first; development only on the committed passing pilot record; final stays closed."""
    import json
    ST.require_previous_stage("pilot_physgen_v1")
    p = tmp_path / "pilot.json"
    with pytest.raises(ST.StudyError):
        ST.require_previous_stage("development_physgen_v1", pilot_path=str(p))                       # missing
    p.write_text(json.dumps(dict(advance=True)))
    with pytest.raises(ST.StudyError):
        ST.require_previous_stage("development_physgen_v1", pilot_path=str(p))                       # not the committed record
    monkeypatch.setattr(ST, "PILOT_RECORD_SHA256", ST._sha_file(str(p)))
    ST.require_previous_stage("development_physgen_v1", pilot_path=str(p))
    p.write_text(json.dumps(dict(advance=False)))
    monkeypatch.setattr(ST, "PILOT_RECORD_SHA256", ST._sha_file(str(p)))
    with pytest.raises(ST.StudyError):
        ST.require_previous_stage("development_physgen_v1", pilot_path=str(p))                       # failed pilot terminates
    with pytest.raises(ST.StudyError):
        ST.require_previous_stage("final_physgen_v1", dev_path=str(tmp_path / "missing.json"))
    d = tmp_path / "dev.json"
    cells = [dict(cond_index=i, seed=s, budget=b, proposals=dict(raw_feasibility=1.0, unique_fraction=1.0),
                  **{x: dict(best_val=1.0 + (0.5 if x == "D" else 0) + 0.01 * i, initial32_best_val=1.0,
                             feasible=True, mass_ok=True) for x in "ABCD"})
             for i in range(4) for s in (0, 1) for b in (128, 512)]
    rec = dict(cells=cells, aggregate=STAT.aggregate(cells, [128, 512]),
               ood=dict(per_condition=[dict(cond_index=i, cls="in") for i in range(4)]))
    d.write_text(json.dumps(rec, default=str))
    monkeypatch.setattr(ST, "DEV_RECORD_SHA256", ST._sha_file(str(d)))
    ST.require_previous_stage("final_physgen_v1", dev_path=str(d))                        # passing development
    rec["cells"][0]["A"]["feasible"] = False                                                  # one gate fails
    d.write_text(json.dumps(rec, default=str))
    monkeypatch.setattr(ST, "DEV_RECORD_SHA256", ST._sha_file(str(d)))
    with pytest.raises(ST.StudyError, match="promotion gate"):
        ST.require_previous_stage("final_physgen_v1", dev_path=str(d))
    with pytest.raises(ST.StudyError, match="acoustic-liner evaluation shards"):                                # final never single-run
        ST._evaluate(argparse.Namespace(split="final_physgen_v1"))


def test_generator_only_comparison_is_budget_matched():
    cells = [dict(cond_index=i, seed=0, budget=128,
                  C=dict(best_val=0.9, initial32_best_val=0.9), B=dict(best_val=1.5, initial32_best_val=0.8))
             for i in range(4)]
    agg = STAT.aggregate([dict(c, proposals=dict(raw_feasibility=1.0, unique_fraction=1.0),
                             A=dict(best_val=1.4, feasible=True, mass_ok=True), D=dict(best_val=1.6, feasible=True,
                             mass_ok=True), **{k: dict(c[k], feasible=True, mass_ok=True) for k in "BC"})
                        for c in cells], [128])["128"]
    assert agg["C32_minus_B32"]["mean"] == pytest.approx(0.1)
    assert agg["diagnostic_C32_minus_Bfinal"]["mean"] == pytest.approx(-0.6)
    assert "C_minus_B" not in agg


def test_clustered_bootstrap_contrast():
    cells = []
    for ci in range(6):
        for s in range(3):
            cells.append(dict(cond_index=ci, seed=s, budget=128, D=dict(best_val=1.0 + 0.1 * ci + 0.01 * s),
                              A=dict(best_val=1.0)))
    r = STAT.contrast(cells, 128, "D", "A", n_boot=2000)
    assert r["n_conditions"] == 6 and r["mean"] == pytest.approx(0.26)
    assert r["ci95"][0] <= r["mean"] <= r["ci95"][1]
    assert r["seed_means"][0] == pytest.approx(0.25) and r["min_seed_mean"] == pytest.approx(0.25)


def test_ood_scorer_threshold_from_validation(splits):
    tr, va = splits
    ood = PCN.OODScorer(tr, va)
    d = ood.score(va)
    assert ood.near <= ood.threshold and np.isfinite(d).all()
    assert sum(c == "beyond" for c in ood.classify(va)) <= max(1, int(np.ceil(0.01 * len(va))))


# ---- stage records of new runs -------------------------------------------------------------------------------------
@pytest.fixture
def new_runs(monkeypatch, tmp_path):
    """A stand-in frozen training/validation split (the released file is generated by `freeze-splits`, which takes
    minutes) and empty RUN_DIR / REPRO_DIR folders under tmp_path. Everything else is the real code and the real
    evaluation splits."""
    tr, va = PCN.evaluation_split("development_physgen_v1"), PCN.evaluation_split("pilot_physgen_v1")
    monkeypatch.setattr(ST, "frozen_splits", lambda: (tr, va))
    monkeypatch.setattr(ST, "SPLIT_DIR", str(tmp_path / "splits"))
    monkeypatch.setattr(ST, "RUN_DIR", str(tmp_path / "dev"))
    monkeypatch.setattr(ST, "REPRO_DIR", str(tmp_path / "repro"))
    return tmp_path


def stage_record(split, d_gain=0.5, **edit):
    """A complete record of `split` as a new run writes it: real conditions, seeds, budgets, released checkpoint
    and run identity; synthetic cell values (D better than A and B by `d_gain` dB); aggregate, OOD and gates
    computed by the code under test."""
    conds = PCN.evaluation_split(split)
    seeds, budgets = ST.SPLIT_SEEDS[split], ST.SPLIT_BUDGETS[split]
    cells = [dict(cond_index=i, condition_hash=PCN.condition_hash(c), seed=s, budget=b,
                  proposals=dict(raw_feasibility=1.0, unique_fraction=1.0),
                  **{x: dict(best_val=1.0 + (d_gain if x == "D" else 0) + 0.01 * i + 0.001 * s,
                             initial32_best_val=1.0, feasible=True, mass_ok=True) for x in "ABCD"})
             for i, c in enumerate(conds) for s in seeds for b in budgets]
    tr, va = ST.frozen_splits()
    ood = PCN.OODScorer(tr, va)
    per = [dict(cond_index=i, score=float(v), cls=k) for i, (v, k) in enumerate(zip(ood.score(conds),
                                                                                       ood.classify(conds)))]
    rec = dict(evaluation_schema_version=ST.EVALUATION_SCHEMA_VERSION, split=split, split_hash=PCN.split_hash(conds),
               checkpoint=dict(model_sha256=CKPT.released_weights_sha256()), base_seed=PCN.named_seeds()["evaluation"],
               budgets=list(budgets), seeds=list(seeds), run=ST.run_identity(), cells=cells,
               aggregate=STAT.aggregate(cells, budgets), ood=dict(ood.record(), per_condition=per), wall_s=1234.5)
    rec = json.loads(json.dumps(rec, default=str))
    rec.update(ST.pilot_gates(rec) if split == "pilot_physgen_v1" else dict(gates=None, advance=None))
    rec.update(edit)
    return json.loads(json.dumps(rec, default=str))


def write(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))
    return str(path)


def test_a_new_run_opens_the_next_stage_whatever_its_timestamps_and_paths(new_runs):
    """A pilot record of a new run, with its own wall time, timestamp and location, opens development; a new
    development record opens final. Nothing is compared byte for byte."""
    released = CKPT.released_weights_sha256()
    rec = stage_record("pilot_physgen_v1")
    p = write(new_runs / "repro" / "pilot_physgen_v1.json", rec)
    ST.require_previous_stage("development_physgen_v1", released)                  # found in REPRO_DIR
    other = write(new_runs / "elsewhere" / "p.json", dict(rec, wall_s=99999.0, written_at="2031-01-01T00:00:00Z"))
    ST.require_previous_stage("development_physgen_v1", released, pilot_path=other)
    assert ST.verify_stage_record(json.load(open(p)), "pilot_physgen_v1")["advance"]
    write(new_runs / "repro" / "development_physgen_v1.json", stage_record("development_physgen_v1"))
    ST.require_previous_stage("final_physgen_v1", released)


def _drop(r):
    r["cells"].pop()


def _dup(r):
    r["cells"].append(dict(r["cells"][0]))


def _extra(r):
    r["cells"].append(dict(r["cells"][0], seed=9))


def _other_condition(r):
    r["cells"][0]["condition_hash"] = "0" * 64


def _no_arm(r):
    del r["cells"][0]["C"]


REFUSED = {
    "a missing cell": (_drop, "missing cells"),
    "a duplicate cell": (_dup, "duplicate cell"),
    "an unexpected cell": (_extra, "unexpected cell"),
    "a cell of another condition": (_other_condition, "condition.s hash"),
    "a cell without an arm": (_no_arm, "lacks arm C"),
    "another checkpoint": (lambda r: r.update(checkpoint=dict(model_sha256="0" * 64)), "not the released"),
    "another base seed": (lambda r: r.update(base_seed=r["base_seed"] + 1), "base_seed"),
    "other seeds": (lambda r: r.update(seeds=[0, 1]), "seeds or budgets"),
    "another split": (lambda r: r.update(split="development_physgen_v1"), "record of"),
    "another split hash": (lambda r: r.update(split_hash="0" * 64), "split_hash"),
    "the older format": (lambda r: r.update(evaluation_schema_version=2), "evaluation_schema_version"),
    "another package source": (lambda r: r["run"].update(package_source_sha256="0" * 64), "package_source"),
    "another frozen split file": (lambda r: r["run"].update(frozen_split_file_sha256="0" * 64), "frozen_split_file"),
    "an edited aggregate": (lambda r: r["aggregate"]["128"]["D_minus_A"].update(mean=9.0), "stored aggregate"),
    "an edited OOD class": (lambda r: r["ood"]["per_condition"][0].update(cls="beyond"), "OOD"),
    "a stored pass the cells do not support": (lambda r: r.update(gates={}, advance=True), "stored gates"),
}


@pytest.mark.parametrize("name", list(REFUSED))
def test_an_incomplete_inconsistent_or_incompatible_record_is_refused(new_runs, name):
    change, match = REFUSED[name]
    rec = stage_record("pilot_physgen_v1")
    change(rec)
    p = write(new_runs / "repro" / "pilot_physgen_v1.json", rec)
    with pytest.raises(ST.StudyError, match=match):
        ST.require_previous_stage("development_physgen_v1", CKPT.released_weights_sha256(), pilot_path=p)


def test_the_next_stage_must_use_the_same_checkpoint(new_runs):
    p = write(new_runs / "repro" / "pilot_physgen_v1.json", stage_record("pilot_physgen_v1"))
    with pytest.raises(ST.StudyError, match="another checkpoint"):
        ST.require_previous_stage("development_physgen_v1", "f" * 64, pilot_path=p)


def test_gates_are_recomputed_and_a_failing_stage_stays_closed(new_runs):
    """D worse than A by 0.1 dB: the recomputed pilot and development gates fail and the next stage stays closed."""
    released = CKPT.released_weights_sha256()
    rec = stage_record("pilot_physgen_v1", d_gain=-0.1)
    assert rec["advance"] is False
    write(new_runs / "repro" / "pilot_physgen_v1.json", rec)
    with pytest.raises(ST.StudyError, match="pilot_physgen_v1 failed its gates.*D_minus_A_128_nonnegative"):
        ST.require_previous_stage("development_physgen_v1", released)
    write(new_runs / "repro" / "development_physgen_v1.json", stage_record("development_physgen_v1", d_gain=-0.1))
    with pytest.raises(ST.StudyError, match="development_physgen_v1 failed its gates"):
        ST.require_previous_stage("final_physgen_v1", released)


def test_new_records_never_overwrite_and_never_enter_the_released_folder(new_runs):
    ST.new_record_path(str(new_runs / "repro" / "x.json"))
    with pytest.raises(ST.StudyError, match="released study"):
        ST.new_record_path(os.path.join(ST.RUN_DIR, "pilot_physgen_v1.json"))
    with pytest.raises(ST.StudyError, match="released study"):
        ST.new_record_path(os.path.join(ST.RUN_DIR, "shards", "m.json"))
    p = write(new_runs / "repro" / "pilot_physgen_v1.json", {})
    with pytest.raises(ST.StudyError, match="written once"):
        ST.new_record_path(p)


def test_the_historical_records_keep_their_rule(new_runs, monkeypatch):
    """A record with the released study's SHA-256 is accepted by that hash alone (its older format included)."""
    p = write(new_runs / "dev" / "pilot_physgen_v1.json", dict(advance=True, evaluation_schema_version=1))
    monkeypatch.setattr(ST, "PILOT_RECORD_SHA256", ST._sha_file(p))
    ST.require_previous_stage("development_physgen_v1", CKPT.released_weights_sha256())
