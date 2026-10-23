# -*- coding: utf-8 -*-
"""PHYS-GEN-V1 trainer gates (owner 2026-10-02). TRAINING-ONLY smoke conditions from the registered training streams;
no pilot / development / final / Falcon / Atlas condition is scored, trained on or selected with."""
import ast
import inspect
import json
import os

import numpy as np
import pytest
import torch

from acoustic_liner.evaluation import splits as PCN
from acoustic_liner.models import objective as PO
from acoustic_liner.models import checkpoint as CKPT
from acoustic_liner.models import training as TRAIN
from acoustic_liner.models import training as PT

WORKERS = 4


@pytest.fixture(scope="module")
def splits():
    return PCN.training_splits(workers=WORKERS, n_train=TRAIN.SMOKE_CONFIG["n_train"], n_val=TRAIN.SMOKE_CONFIG["n_val"])


@pytest.fixture(scope="module")
def smoke(splits, tmp_path_factory):
    tr, va = splits
    out = str(tmp_path_factory.mktemp("smoke"))
    T = TRAIN.Trainer(dict(TRAIN.SMOKE_CONFIG), tr, va, frozenset(), out, workers=WORKERS).run()
    return T, out


def _short_cfg():
    return dict(TRAIN.SMOKE_CONFIG, epochs=2, val_every=1, val_pop=8)


# ---- determinism / resume ------------------------------------------------------------------------------------------
def test_interrupted_resume_is_bit_identical(splits, tmp_path):
    tr, va = splits
    a = TRAIN.Trainer(_short_cfg(), tr, va, frozenset(), str(tmp_path / "a")).run()
    TRAIN.Trainer(_short_cfg(), tr, va, frozenset(), str(tmp_path / "b")).run(stop_at_step=3)   # mid-epoch interrupt
    b = TRAIN.Trainer.resume(str(tmp_path / "b"), tr, va, frozenset()).run(stop_at_step=6)    # interrupt again
    b = TRAIN.Trainer.resume(str(tmp_path / "b"), tr, va, frozenset()).run()
    sa, sb = a.model.state_dict(), b.model.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert a.history == b.history and a.validations == b.validations and a.selected == b.selected
    oa, ob = a.opt.state_dict()["state"], b.opt.state_dict()["state"]
    assert all(torch.equal(oa[k][n], ob[k][n]) for k in oa for n in ("exp_avg", "exp_avg_sq"))
    assert a.noise_rng.bit_generator.state == b.noise_rng.bit_generator.state
    assert a.sched.state_dict() == b.sched.state_dict()


# ---- splits / leakage ----------------------------------------------------------------------------------------------
def test_frozen_split_is_deterministic_and_stratified(splits):
    tr, va = splits
    tr2, va2 = PCN.training_splits(workers=WORKERS, n_train=len(tr), n_val=len(va))
    assert PCN.split_hash(tr) == PCN.split_hash(tr2) and PCN.split_hash(va) == PCN.split_hash(va2)
    for k, (lo, hi) in zip(PCN.LHS_NAMES, PCN.LHS_BOX):                  # LHS: exactly one draw per marginal stratum
        x = np.array([c["strata"][k] for c in tr])
        bins = np.floor((x - lo) / (hi - lo) * len(tr)).astype(int)
        assert sorted(bins.tolist()) == list(range(len(tr)))
    assert all(c["group"] == "training" for c in tr) and all(c["group"] == "training_validation" for c in va)


def test_seeds_named_and_collision_checked():
    s = PCN.named_seeds()
    assert set(s) == set(PCN.STREAM_NAMES) and len(set(s.values())) == len(s)
    with pytest.raises(PCN.PhysgenConditionError):
        PCN.check_seed_collisions(dict(s, extra=s["model_init"]))


def test_leakage_barrier_fails_closed(splits):
    tr, va = splits
    rep = PCN.leakage_report(tr, va, frozenset())
    assert rep["ok"] and rep["training_hash"] == PCN.split_hash(tr)
    with pytest.raises(PCN.PhysgenConditionError):                       # forbidden-hash intersection
        PCN.leakage_report(tr, va, frozenset({PCN.condition_hash(va[0])}))
    with pytest.raises(PCN.PhysgenConditionError):                       # train/validation overlap
        PCN.leakage_report(tr, va[:-1] + [dict(tr[0], group="training_validation")], frozenset())
    near = dict(tr[1], group="training_validation", A_wall=tr[1]["A_wall"] * (1 + 1e-9))
    with pytest.raises(PCN.PhysgenConditionError):                       # near-duplicate (standardized features)
        PCN.leakage_report(tr, va[:-1] + [near], frozenset())
    with pytest.raises(PCN.PhysgenConditionError):                       # a non-training group never enters
        PCN.leakage_report(tr, va[:-1] + [dict(va[-1], group="pilot_physgen_v1")], frozenset())


def test_registered_splits_disjoint_from_every_forbidden_split(splits):
    """Full barrier on the registered streams: v2.4-v2.8 evaluation splits, PHYS-GEN pilot/development/final,
    Falcon blanketed + unblanketed West/East and the reference vehicle spectra are only HASHED / compared."""
    tr, va = splits
    b = PCN.full_barrier()
    assert len(b["forbidden_hashes"]) >= 160 + 100
    assert {"falcon9_2025_blanketed", "falcon9_2025_unblanketed_west", "falcon9_2025_unblanketed_east",
            "reference:Atlas V"} <= set(b["vehicle_spectra"])
    assert {PCN.condition_hash(c) for c in PCN.falcon_conditions()} <= b["forbidden_hashes"]
    rep = PCN.leakage_report(tr, va, **b)
    print(f"\nvehicle min shape-weight distance: {min(rep['vehicle_min_weight_distance'].values()):.3e}")
    for f in PCN.falcon_conditions():                                    # a vehicle spectrum in training is refused
        with pytest.raises(PCN.PhysgenConditionError):
            PCN.leakage_report(tr, va[:-1] + [dict(va[-1], weights=f["weights"], spl=f["spl"])], **b)


def test_frozen_split_artifact_is_atomic_immutable_and_verified(tmp_path):
    d = str(tmp_path)
    tr, va = PCN.training_splits(cache_dir=d, n_train=8, n_val=4, workers=WORKERS)
    path = PCN.frozen_split_path(d, 8, 4)
    assert not os.access(path, os.W_OK) and not [f for f in os.listdir(d) if ".tmp" in f]
    sha = PCN.split_file_sha256(path)
    tr2, va2 = PCN.training_splits(cache_dir=d, n_train=8, n_val=4)      # loads, never regenerates / overwrites
    assert (tr2, va2) == (tr, va) and PCN.split_file_sha256(path) == sha
    with pytest.raises(PCN.PhysgenConditionError):
        PCN.load_frozen_splits(path, expected=dict(file_sha256="0" * 64))
    os.chmod(path, 0o666)
    doc = json.load(open(path)); doc["training"][0]["A_wall"] += 1.0; json.dump(doc, open(path, "w"))
    with pytest.raises(PCN.PhysgenConditionError):                       # edited content no longer matches
        PCN.load_frozen_splits(path)


def test_progress_reports_every_epoch_without_changing_results(splits, tmp_path):
    tr, va = splits
    events = []
    a = TRAIN.Trainer(_short_cfg(), tr, va, frozenset(), str(tmp_path / "p"), progress=events.append).run()
    b = TRAIN.Trainer(_short_cfg(), tr, va, frozenset(), str(tmp_path / "q")).run()
    ep = [e for e in events if e["kind"] == "epoch"]
    assert [e["epoch"] for e in ep] == [1, 2] and all(e["seconds"] > 0 and e["eta"] >= 0 for e in ep)
    assert [e["epoch"] for e in events if e["kind"] == "validation"] == [0, 1, 2]
    assert a.history == b.history and a.validations == b.validations


# ---- binding / loader ----------------------------------------------------------------------------------------------
def test_checkpoint_binding_complete_and_loader_enforces_policy(smoke, splits, tmp_path):
    T, out = smoke
    tr, va = splits
    path = os.path.join(out, "selected.pt")
    model, ck = CKPT.load_checkpoint(path, tr, va)
    b = ck["binding"]
    for k in ("public_architecture_sha256", "config_sha256", "package_source_sha256", "training_hash", "validation_hash", "public_coordinate_transform_sha256"):
        assert isinstance(b[k], str) and len(b[k]) == 64
    assert b["scorer"] == "quality_eval_exact" and b["node_grad"] == "cap_only" and b["tau_schedule"]["kind"] == "geometric"

    def tampered(fn):
        c = torch.load(path, weights_only=False); fn(c)
        p = str(tmp_path / f"t{np.random.randint(1e9)}.pt"); torch.save(c, p)
        with pytest.raises(CKPT.CheckpointError):
            CKPT.load_checkpoint(p, tr, va)
    tampered(lambda c: c["binding"].__setitem__("node_grad", "all"))
    tampered(lambda c: c["binding"].__setitem__("scorer", "quality_eval"))
    tampered(lambda c: c["binding"].__setitem__("public_architecture_sha256", "0" * 64))
    tampered(lambda c: c["binding"].__setitem__("public_coordinate_transform_sha256", "0" * 64))
    tampered(lambda c: c["binding"]["config"].__setitem__("lr", 1.0))           # config hash no longer matches
    tampered(lambda c: c["binding"]["tau_schedule"].__setitem__("kind", "linear"))
    tampered(lambda c: c["history"][-1].__setitem__("tau_cov", 0.02))            # history departs from schedule
    tampered(lambda c: c["binding"].pop("seeds"))
    tampered(lambda c: c["model"]["head.bias"].add_(1e-3))                      # weights vs content hash
    with pytest.raises(CKPT.CheckpointError):                                      # different split
        CKPT.load_checkpoint(path, tr[::-1], va)
    c = torch.load(path, weights_only=False)                                       # the source fingerprint is
    c["binding"]["package_source_sha256"] = "0" * 64                               # recorded, not compared
    p = str(tmp_path / "source_edit.pt"); torch.save(c, p)
    CKPT.load_checkpoint(p, tr, va)


def test_trainer_refuses_unregistered_policy(splits, tmp_path):
    tr, va = splits
    for bad in (dict(node_grad="all"), dict(scorer="quality_eval"), dict(tau_corner=0.01)):
        with pytest.raises(TRAIN.TrainingError):
            TRAIN.Trainer(dict(TRAIN.SMOKE_CONFIG, **bad), tr, va, frozenset(), str(tmp_path))


def test_curriculum_and_geometric_tau():
    cfg = TRAIN.CONFIG
    ps = np.linspace(0, 1, 401)
    sched = [PO.curriculum(p, cfg) for p in ps]
    assert [s[0] for s in sched] == sorted(s[0] for s in sched)                # phases 1 -> 4 in order
    assert all(np.diff([s[1] for s in sched]) >= 0) and sched[-1][1] == 1.0
    taus = np.array([s[2] for s in sched])
    assert all(np.diff(taus) <= 0) and taus[0] == PO.TAU_COV_START and taus[-1] == pytest.approx(PO.TAU_COV_END)
    for p, (ph, w, t) in zip(ps, sched):
        if ph == 3:
            q = (p - cfg["ramp_end"]) / (cfg["sharpen_end"] - cfg["ramp_end"])
            assert t == pytest.approx(PO.TAU_COV_START * (PO.TAU_COV_END / PO.TAU_COV_START) ** q, rel=1e-12)


# ---- smoke gates ---------------------------------------------------------------------------------------------------
def test_smoke_improves_exact_score_and_reduces_dead_zone(smoke):
    T, _ = smoke
    v0, vN = T.validations[0], T.validations[-1]
    print(f"\nsmoke exact: best-of-pop {v0['perf']:.4f} -> {vN['perf']:.4f} dB; mean proposal {v0['mean_value']:.4f} -> "
          f"{vN['mean_value']:.4f} dB; dead fraction {v0['dead_fraction']:.4f} -> {vN['dead_fraction']:.4f}")
    assert vN["perf"] > v0["perf"] and vN["mean_value"] > v0["mean_value"]
    assert vN["dead_fraction"] < v0["dead_fraction"]


def test_smoke_feasibility_mass_uniqueness_diversity(smoke):
    T, _ = smoke
    for m in T.validations:
        assert m["feasibility"] == 1.0 and m["mass_compliance"] == 1.0 and m["winner_compliance"] == 1.0
        assert m["unique_fraction"] >= 0.95 and m["rank_gate"]
    assert T.validations[-1]["noise_sensitivity_min"] > 0.0                     # every noise dimension became live


def test_selection_is_lexicographic_on_validation_only(smoke):
    T, _ = smoke
    best = max(T.validations, key=lambda m: TRAIN.selection_key(m, m["epoch"]))
    assert T.selected["epoch"] == best["epoch"]
    mk = lambda g1, g2, perf: dict(gate1=g1, gate2=g2, perf=perf)
    ms = [(mk(True, True, 0.5), 0), (mk(False, True, 9.0), 5), (mk(True, False, 9.0), 10), (mk(True, True, 0.7), 15),
          (mk(True, True, 0.7), 20)]
    assert max(ms, key=lambda t: TRAIN.selection_key(*t))[1] == 15                # gates first, perf, earliest tie


def test_trainer_never_touches_heldout_sources():
    """No identifier in the trainer can build, load or score a held-out / vehicle split."""
    names = {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(ast.parse(inspect.getsource(PT)))
             if isinstance(n, (ast.Attribute, ast.Name))}
    forbidden = {"evaluation_split", "forbidden_condition_hashes", "falcon_condition", "synthetic_conditions",
                 "eval_split_condition_hashes", "build_pilot_v28", "build_development_v28", "spl_array",
                 "reference_environments", "launch_environments"}
    assert not (names & forbidden)
