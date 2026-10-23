r"""PHYS-GEN-V1 study driver, Phase I.

Per (condition, seed, budget), with matched exact-charged budgets and the authoritative exact scorer throughout:
  A  corrected exact-gauge plain quality-CMA            `optimization.baseline.run_quality_cma_exact`     B calls
  B  32 same-transform random proposals + pinned CMA    `optimization.seeded_cma.run_seeded_cma(random)`  B calls
  C  32 generator proposals, no CMA (diagnostic)        `run_seeded_cma(generator, cma=False)`       32 calls
  D  32 generator proposals + pinned CMA (primary)      `run_seeded_cma(generator)`                  B calls
Arms B/C/D share the registered proposal streams, so C's proposals are exactly D's proposal phase, and the untrained
generator's D equals B bit for bit (tested).

GENERATOR-ONLY CLAIM is budget-matched: best of C's 32 proposals vs best of B's first 32 (proposal-phase) calls;
C vs B's full-budget winner is kept only as a labelled diagnostic.

FROZEN SPLITS: the training/validation split file and the three evaluation splits must hash to the values the
released study used (`FROZEN_SPLITS`, `EVALUATION_SPLIT_HASHES`); anything else is refused. The evaluation splits open
one at a time: development only on a passing pilot record, final only on a development record that passes every
promotion gate (`require_previous_stage`). Falcon / Atlas evaluation is not implemented here. `train` prints one progress line per epoch (seconds per epoch, elapsed, ETA) and per validation.

TWO KINDS OF STAGE RECORD. The historical records of the released study (under RUN_DIR, not distributed) are accepted
by their SHA-256. A record written by a new run (under REPRO_DIR, never RUN_DIR) is accepted only after
`verify_stage_record`: same split, conditions, seeds, budgets and base seed; every cell present once; the released
checkpoint, the same checkpoint as the next stage, and the same package source and frozen split file; and the
aggregate, OOD classes and gates recomputed from the cells and conditions. Stored `aggregate`, `ood`, `gates` and
`advance` are never trusted: a record whose stored values differ from the recomputed ones is refused. Timestamps,
wall time and paths play no part.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import time
import numpy as np
from ..optimization import baseline as PB
from . import splits as PCN
from ..models import linergen as PM
from ..physics import scorer as PS
from ..optimization import seeded_cma as HO
from ..models import checkpoint as CKPT
from ..models import training as TRAIN
from . import leakage as LEAK
from ..design import canonical as CAN
from ..physics import quality as Q
from .statistics import aggregate
from ..provenance import hashing as H


EVAL_SPLITS = ("pilot_physgen_v1", "development_physgen_v1", "final_physgen_v1")
SPLIT_SEEDS = {"pilot_physgen_v1": (0, 1, 2), "development_physgen_v1": (0, 1, 2, 3, 4),
               "final_physgen_v1": (0, 1, 2, 3, 4)}
SPLIT_BUDGETS = {"pilot_physgen_v1": (128, 512), "development_physgen_v1": (128, 512, 2048),
                 "final_physgen_v1": (128, 512, 2048)}
PROMOTION_FLOOR_DB = 0.03
# 3 adds `run` (package source fingerprint, frozen split file, environment); 2: methods A-D only; historical
# schema 1 adds legacy_A_diagnostic. Only version 3 records are accepted as new stage records.
EVALUATION_SCHEMA_VERSION = 3
# the frozen split file of the released study (training / validation) and the hashes of the three evaluation splits
FROZEN_SPLITS = dict(training_hash="96a5a90b15c08c3d361709ec59597012466b577c9708e1f2cac14817884d5f9d",
                     validation_hash="aed622c73284f1ffd7ce65fb8b196b5e90720f46254820e9e98e0bcd9e751da0",
                     file_sha256="8bd66fd51af38811c388c0f81aa435a69bc581d12b3e8db15c69433cb7628ef7")
EVALUATION_SPLIT_HASHES = {
    "pilot_physgen_v1": "1e545bec4aade1935878a87ab5226808db85bc3f8a577b662f92906b2770a13a",
    "development_physgen_v1": "32f670585a2384660297502e5f563807b7889813270fd5876a27a540004afd9b",
    "final_physgen_v1": "74a60fba9af5fc8817c093e059b6c2ff9015c8ccea87fe57d95c6cf6c3eb4c9c",
}
RUN_DIR = os.path.join("artifacts", "dev", "physgen_v1")          # the released study's records (historical)
REPRO_DIR = os.path.join("artifacts", "repro", "physgen_v1")      # records of new runs of the evaluation splits


class StudyError(RuntimeError):
    """Split / budget / evidence violation (fail closed)."""


# =================================================================================================================
#  one cell
# =================================================================================================================
def _winner(L, cond):
    ok = [e for e in L.trace if e["feasible_budget"]]
    if not ok:
        return dict(best_val=None, feasible=False, mass_ok=False)
    w = min(ok, key=Q.quality_rank_key)
    return dict(best_val=float(w["best_val"]), feasible=bool(w["physically_feasible"]),
                mass_ok=bool(w["mass"] <= cond["m_total_target"]), mass=float(w["mass"]), cov=float(w["best_cov"]),
                geometry=w["geometry"])


def _initial_best(L, n=HO.N_PROPOSALS):
    """Best exact value among the first n charged evaluations (the proposal phase of a seeded arm)."""
    ok = [e["best_val"] for e in L.trace[:n] if e["feasible_budget"]]
    return float(max(ok)) if ok else None


def _arm(L, cond, arm):
    r = dict(arm=arm, charged=len(L.trace), exhausted=bool(L.exhausted), n_infeasible=int(L.n_infeasible),
             n_duplicate=int(L.n_duplicate), attempts=int(L.attempts), initial32_best_val=_initial_best(L),
             **_winner(L, cond))
    if hasattr(L, "n_proposal_charged"):
        r.update(n_proposal_charged=int(L.n_proposal_charged), n_fill_charged=int(L.n_fill_charged),
                 n_reinit=int(L.n_reinit))
    return r


def proposal_diagnostics(designs, cond, tol=TRAIN.CONFIG["unique_tol"]):
    """Raw proposal feasibility (canonical) and unique fraction (geometry key + normalized L2 tolerance)."""
    X = np.array([HO.to_cma01(d, sort_banks=False) for d in designs])
    P = len(designs)
    D2 = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1) / TRAIN.NORM_DIM) + np.eye(P) * 1e9
    keys = [CAN.geometry_key(d) for d in designs]
    feas = [PS.quality_eval_exact(d, cond)["physically_feasible"] for d in designs]
    return dict(raw_feasibility=float(np.mean(feas)),
                unique_fraction=float(np.mean([D2[i].min() > tol and keys.count(keys[i]) == 1 for i in range(P)])))


def run_cell(cond, ci, seed, budget, base_seed, model):
    gen = PM.proposal_fn(model, cond)
    A = PB.run_quality_cma_exact(cond, ci, seed, budget, base_seed)
    B = HO.run_seeded_cma(cond, ci, seed, budget, base_seed, HO.random_proposals, "armB")
    C = HO.run_seeded_cma(cond, ci, seed, budget, base_seed, gen, "armC", cma=False)
    D = HO.run_seeded_cma(cond, ci, seed, budget, base_seed, gen, "armD")
    for L, nm in ((A, "A"), (B, "B"), (D, "D")):
        if len(L.trace) != int(budget) and not L.exhausted:
            raise StudyError(f"arm {nm} charged {len(L.trace)} != budget {budget}")
    if len(C.trace) != min(HO.N_PROPOSALS, int(budget)) and not C.exhausted:
        raise StudyError(f"arm C charged {len(C.trace)}")
    props = gen(HO.stream(base_seed, ci, seed, "pg_proposal"), HO.N_PROPOSALS)
    rec = dict(cond_index=int(ci), condition_hash=PCN.condition_hash(cond), seed=int(seed), budget=int(budget),
               A=_arm(A, cond, "A"), B=_arm(B, cond, "B"), C=_arm(C, cond, "C"), D=_arm(D, cond, "D"),
               proposals=proposal_diagnostics(props, cond))
    return rec


# =================================================================================================================
#  frozen splits + evaluation
# =================================================================================================================
def _sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _environment():
    import platform
    import torch
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                torch_threads=TRAIN.CONFIG["torch_threads"])


def frozen_splits():
    """(training, validation) conditions from the frozen split file; refused unless it hashes to FROZEN_SPLITS."""
    return PCN.load_frozen_splits(PCN.frozen_split_path(SPLIT_DIR), expected=FROZEN_SPLITS)


def check_evaluation_split(split, conds):
    """An evaluation split is evaluated only as the released study defined it (same conditions, same hash)."""
    if split not in EVAL_SPLITS:
        raise StudyError(f"unknown evaluation split {split!r}")
    if PCN.split_hash(conds) != EVALUATION_SPLIT_HASHES[split]:
        raise StudyError(f"{split}: the conditions do not hash to the released split")


@functools.lru_cache(maxsize=2)
def _worker_model(path):
    return CKPT.load_checkpoint(path)[0]


def _cell_worker(args):
    import torch
    torch.set_num_threads(1)
    cond, ci, seed, budget, base_seed, path = args
    with LEAK.sealed_guard():
        return run_cell(cond, ci, seed, budget, base_seed, _worker_model(path))


def cell_printer(n):
    """One flushed line per finished cell: count, elapsed, ETA (from the mean cell wall time so far)."""
    t0 = time.perf_counter()

    def emit(k, cell):
        el = time.perf_counter() - t0
        d, a = cell["D"]["best_val"], cell["A"]["best_val"]
        da = "  n/a " if d is None or a is None else f"{d - a:+.3f}"
        print(f"cell {k:>4}/{n}  cond {cell['cond_index']:>2} seed {cell['seed']} B {cell['budget']:>4}  "
              f"D-A {da} dB  elapsed {_hms(el)}  "
              f"ETA {_hms(el / k * (n - k))}", flush=True)
    return emit


def evaluate(split, conds, checkpoint_path, budgets, seeds, workers=0, ood=None, progress=None):
    model, ck = CKPT.load_checkpoint(checkpoint_path)
    if split in EVAL_SPLITS:
        check_evaluation_split(split, conds)
    elif split != "training_smoke" or any(c.get("group") not in ("training", "training_validation") for c in conds):
        raise StudyError("only the evaluation splits or training-only conditions may be evaluated")
    base_seed = PCN.named_seeds()["evaluation"]
    jobs = [(c, ci, s, b, base_seed, checkpoint_path) for ci, c in enumerate(conds) for s in seeds
            for b in budgets]
    t0 = time.perf_counter()
    cells = [None] * len(jobs)                        # filled by original job index: deterministic ordering
    if workers:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_cell_worker, j): i for i, j in enumerate(jobs)}
            for k, f in enumerate(as_completed(futs), 1):
                cells[futs[f]] = f.result()
                if progress:
                    progress(k, cells[futs[f]])
    else:
        with LEAK.sealed_guard():
            for i, (c, ci, s, b, bs, _) in enumerate(jobs):
                cells[i] = run_cell(c, ci, s, b, bs, model)
                if progress:
                    progress(i + 1, cells[i])
    ood_rec = None
    if ood is not None:
        sc = ood.score(conds); cl = ood.classify(conds)
        ood_rec = dict(ood.record(), per_condition=[dict(cond_index=i, score=float(sc[i]), cls=cl[i])
                                                     for i in range(len(conds))])
    return dict(evaluation_schema_version=EVALUATION_SCHEMA_VERSION,
                split=split, split_hash=PCN.split_hash(conds), checkpoint=CKPT.checkpoint_identity(ck),
                base_seed=base_seed, budgets=list(budgets), seeds=list(seeds), run=run_identity(), cells=cells,
                aggregate=aggregate(cells, budgets), ood=ood_rec, wall_s=time.perf_counter() - t0)


def run_identity(package_source_sha256=None, environment=None):
    """What a stage record must share with the next stage besides the checkpoint: the package source and the frozen
    split file it ran with (the numerical environment is recorded for information)."""
    path = PCN.frozen_split_path(SPLIT_DIR)
    return dict(package_source_sha256=package_source_sha256 or H.package_source_sha256(),
                frozen_split_file_sha256=PCN.split_file_sha256(path) if os.path.exists(path) else None,
                environment=environment or _environment())


# =================================================================================================================
#  CLI
# =================================================================================================================
SPLIT_DIR = os.path.join(RUN_DIR, "splits")
TRAIN_DIR = os.path.join(RUN_DIR, "train")


def _hms(s):
    s = int(round(s)); return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def progress_printer(log_path=None):
    """Per-epoch / per-validation progress: one flushed console line plus one JSON line in `log_path`.
    Called once per epoch (~128 steps), so it costs nothing measurable."""
    def emit(r):
        if r["kind"] == "epoch":
            line = (f"epoch {r['epoch']:>3}/{r['epochs']}  {r['seconds']:6.1f} s/epoch  elapsed {_hms(r['elapsed'])}  "
                    f"ETA {_hms(r['eta'])}  phase {r['phase']}  tau_cov {r['tau_cov']:.4f}  w_perf {r['w_perf']:.2f}  "
                    f"lr {r['lr']:.2e}  loss {r['loss']:+.4f}  J_train {r['perf']:.4f} dB")
        else:
            line = (f"  validation @ epoch {r['epoch']}  ({r['seconds']:.1f} s)  best-of-32 {r['perf']:.4f} dB  "
                    f"mean {r['mean_value']:.4f} dB  dead {r['dead_fraction']:.3f}  feas {r['feasibility']:.2f}  "
                    f"mass {r['mass_compliance']:.2f}  unique {r['unique_fraction']:.2f}  erank {r['erank_median']:.1f}  "
                    f"gates {'pass' if r['gate1'] and r['gate2'] else 'FAIL'}")
        print(line, flush=True)
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps(dict(r, t=time.time())) + "\n")
    return emit


def _freeze(a):
    path = PCN.frozen_split_path(SPLIT_DIR)
    if os.path.exists(path):
        raise StudyError(f"{path} is frozen already (immutable)")
    tr, va = PCN.training_splits(workers=a.workers, cache_dir=SPLIT_DIR)
    print(json.dumps(dict(path=path, file_sha256=PCN.split_file_sha256(path), training_hash=PCN.split_hash(tr),
                          validation_hash=PCN.split_hash(va)), indent=1))


def _train(a):
    tr, va = frozen_splits()
    os.makedirs(TRAIN_DIR, exist_ok=True)
    prog = progress_printer(os.path.join(TRAIN_DIR, "progress.jsonl"))
    print("building leakage barrier (hashes only) ...", flush=True)
    barrier = PCN.full_barrier()
    if a.resume:
        T = TRAIN.Trainer.resume(TRAIN_DIR, tr, va, barrier, workers=a.workers, progress=prog)
    else:
        if os.path.exists(os.path.join(TRAIN_DIR, "state.pt")):
            raise StudyError(f"{TRAIN_DIR} already holds a run; use --resume (one definitive training run)")
        T = TRAIN.Trainer(dict(TRAIN.CONFIG), tr, va, barrier, TRAIN_DIR, workers=a.workers, progress=prog)
    print(f"training: {T.total_steps} steps ({T.steps_per_epoch}/epoch x {TRAIN.CONFIG['epochs']} epochs), "
          f"resuming at step {T.step}", flush=True)
    T.run()
    print(json.dumps(dict(selected={k: v for k, v in T.selected.items() if k != "metrics"}, leakage_ok=T.leakage["ok"]),
                     default=str, indent=1))


def _smoke(a):
    tr, va = PCN.load_frozen_splits(PCN.frozen_split_path(SPLIT_DIR))
    res = evaluate("training_smoke", va[:a.n], a.checkpoint, (a.budget,), (0,), workers=a.workers)
    print(json.dumps(res["aggregate"], indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="acoustic-liner evaluation study")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--workers", type=int, default=16); t.add_argument("--resume", action="store_true")
    fz = sub.add_parser("freeze-splits"); fz.add_argument("--workers", type=int, default=16)
    s = sub.add_parser("smoke"); s.add_argument("--checkpoint", required=True); s.add_argument("--n", type=int, default=4)
    s.add_argument("--budget", type=int, default=128); s.add_argument("--workers", type=int, default=8)
    e = sub.add_parser("evaluate"); e.add_argument("--split", required=True, choices=EVAL_SPLITS)
    e.add_argument("--checkpoint", required=True); e.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    if a.cmd in ("train", "freeze-splits"):
        return dict(train=_train, **{"freeze-splits": _freeze})[a.cmd](a)
    if a.cmd == "smoke":
        return _smoke(a)
    CKPT.load_checkpoint(a.checkpoint)        # refuse an incompatible checkpoint before any condition is built
    return _evaluate(a)


def pilot_gates(res):
    """Prereg section 13 pilot gates (B = 128 for the D - A point estimate; all budgets for the rest)."""
    agg, cells = res["aggregate"], res["cells"]
    da = agg["128"]["D_minus_A"]
    seed_effects = [v for b in agg.values() for k in ("D_minus_A", "D_minus_B") if b[k] for v in b[k]["seed_means"].values()]
    ood = res["ood"]["per_condition"]
    g = dict(D_minus_A_128_nonnegative=da["mean"] >= 0,
             raw_feasibility_ge_098=min(b["raw_proposal_feasibility"] for b in agg.values()) >= 0.98,
             unique_ge_095=min(b["unique_fraction"] for b in agg.values()) >= 0.95,
             winners_feasible_in_budget=all(c[a]["feasible"] and c[a]["mass_ok"] for c in cells for a in "ABCD"),
             no_seed_mean_below_minus_003=min(seed_effects) >= -PROMOTION_FLOOR_DB,
             ood_complete=len(ood) == len({c["cond_index"] for c in cells}) and all(o["cls"] for o in ood))
    return dict(gates=g, advance=all(g.values()))
PILOT_RECORD = os.path.join(RUN_DIR, "pilot_physgen_v1.json")
PILOT_RECORD_SHA256 = "cd6f34c53bc187eb87181bfc1853dd93d574498e1ade8b07ed5e001e2b1bf4fc"   # historical pilot outcome record
DEV_RECORD = os.path.join(RUN_DIR, "development_physgen_v1.json")
DEV_RECORD_SHA256 = "1fe1fd0743cf4fda7cbedb8ea5931ae79efe2001852416d4f2aa6cf0b565137f"   # historical development outcome record


PREVIOUS = {"development_physgen_v1": "pilot_physgen_v1", "final_physgen_v1": "development_physgen_v1"}
def historical_sha256(split):
    """SHA-256 of the released study's record of `split`."""
    return {"pilot_physgen_v1": PILOT_RECORD_SHA256, "development_physgen_v1": DEV_RECORD_SHA256}[split]


def stage_record_path(split, record_dir=None):
    """The record a stage is opened on: the given folder, else a new run's record (REPRO_DIR), else the historical."""
    if record_dir:
        return os.path.join(record_dir, f"{split}.json")
    new = os.path.join(REPRO_DIR, f"{split}.json")
    return new if os.path.exists(new) else os.path.join(RUN_DIR, f"{split}.json")


def require_previous_stage(split, model_sha256=None, pilot_path=None, dev_path=None):
    """One split at a time: development opens only on a passing pilot record; final only on a development record
    passing every section-13 promotion gate. A historical record is accepted by its SHA-256 (unchanged rule); any
    other record must pass `verify_stage_record` against `model_sha256`, the checkpoint the next stage will use."""
    if split == "pilot_physgen_v1":
        return
    if split not in PREVIOUS:
        raise StudyError(f"unknown split {split!r}")
    prev = PREVIOUS[split]
    given = pilot_path if prev == "pilot_physgen_v1" else dev_path
    p = given or stage_record_path(prev)
    if not os.path.exists(p):
        where = p if given else (f"{os.path.join(REPRO_DIR, prev + '.json')} (a new run) or "
                                 f"{os.path.join(RUN_DIR, prev + '.json')} (the released study)")
        raise StudyError(f"{split} opens only on a {prev} record; none at {where}")
    if _sha_file(p) == historical_sha256(prev):          # the released study's record: the historical rule
        rec = json.load(open(p))
        if prev == "pilot_physgen_v1" and not rec.get("advance"):
            raise StudyError("pilot did not pass its gates; PHYS-GEN-V1 terminates (no tuning)")
        if prev == "development_physgen_v1" and not development_gates(rec)["advance"]:
            raise StudyError("development failed a promotion gate; final_physgen_v1 stays closed")
        return
    gates = verify_stage_record(json.load(open(p)), prev, model_sha256)
    if not gates["advance"]:
        failed = sorted(k for k, v in gates["gates"].items() if not v)
        raise StudyError(f"{prev} failed its gates (recomputed from the cells: {failed}); {split} stays closed")


STAGE_GATES = {"pilot_physgen_v1": lambda r: pilot_gates(r), "development_physgen_v1": lambda r: development_gates(r)}
ARMS = ("A", "B", "C", "D")
ARM_KEYS = ("best_val", "feasible", "mass_ok", "initial32_best_val")


def _json(obj):
    return json.loads(json.dumps(obj, default=str))


def verify_stage_record(rec, split, model_sha256=None):
    """Check a new run's record of `split` against the released study's definition and recompute its gates.

    Refused unless: schema version 3; this split, its frozen conditions (hash), seeds, budgets and base seed; every
    (condition, seed, budget) cell exactly once, each with its condition's hash and all four arms; the released
    checkpoint (and `model_sha256`, the next stage's checkpoint, when given); the package source and frozen split file
    in use now. The aggregate (from the cells), the OOD classes (from the frozen splits and the conditions) and the
    gates are recomputed; a stored value that differs from its recomputation is refused. Returns the recomputed gates."""
    def bad(msg):
        raise StudyError(f"{split} record refused: {msg}")

    if rec.get("evaluation_schema_version") != EVALUATION_SCHEMA_VERSION:
        bad(f"evaluation_schema_version {rec.get('evaluation_schema_version')!r} is not {EVALUATION_SCHEMA_VERSION} "
            "(only the released study's own records are accepted in an older format)")
    if rec.get("split") != split:
        bad(f"it is a record of {rec.get('split')!r}")
    conds = PCN.evaluation_split(split)
    check_evaluation_split(split, conds)
    if rec.get("split_hash") != EVALUATION_SPLIT_HASHES[split]:
        bad("split_hash is not the released split's")
    if rec.get("base_seed") != PCN.named_seeds()["evaluation"]:
        bad("base_seed differs from the evaluation seed")
    if list(rec.get("seeds", [])) != list(SPLIT_SEEDS[split]) or list(rec.get("budgets", [])) != list(SPLIT_BUDGETS[split]):
        bad("seeds or budgets differ from the split's")
    released = CKPT.released_weights_sha256()
    got = (rec.get("checkpoint") or {}).get("model_sha256")
    if got != released:
        bad(f"checkpoint weights {got} are not the released checkpoint's {released}")
    if model_sha256 is not None and model_sha256 != got:
        bad("it was made with another checkpoint than the one given for the next stage")
    run, now = rec.get("run") or {}, run_identity()
    for k in ("package_source_sha256", "frozen_split_file_sha256"):
        if run.get(k) != now[k]:
            bad(f"run.{k} differs from this checkout (rerun {split} with this code and split file)")
    want = {(ci, s, b) for ci in range(len(conds)) for s in SPLIT_SEEDS[split] for b in SPLIT_BUDGETS[split]}
    seen = set()
    for c in rec.get("cells") or []:
        key = (c.get("cond_index"), c.get("seed"), c.get("budget"))
        if key in seen:
            bad(f"duplicate cell {key}")
        if key not in want:
            bad(f"unexpected cell {key}")
        seen.add(key)
        if c.get("condition_hash") != PCN.condition_hash(conds[key[0]]):
            bad(f"cell {key} carries another condition's hash")
        for a in ARMS:
            if not isinstance(c.get(a), dict) or any(k not in c[a] for k in ARM_KEYS):
                bad(f"cell {key} lacks arm {a} or one of its fields")
        if not isinstance(c.get("proposals"), dict):
            bad(f"cell {key} lacks its proposal diagnostics")
    if seen != want:
        bad(f"{len(want - seen)} missing cells, e.g. {sorted(want - seen)[:3]}")
    agg = _json(aggregate(rec["cells"], SPLIT_BUDGETS[split]))
    if _json(rec.get("aggregate")) != agg:
        bad("the stored aggregate is not the one recomputed from its cells")
    tr, va = frozen_splits()
    ood = PCN.OODScorer(tr, va)
    sc, cl = ood.score(conds), ood.classify(conds)
    ood_rec = _json(dict(ood.record(), per_condition=[dict(cond_index=i, score=float(sc[i]), cls=cl[i])
                                                      for i in range(len(conds))]))
    if _json(rec.get("ood")) != ood_rec:
        bad("the stored OOD classification is not the one recomputed from the frozen splits")
    gates = STAGE_GATES[split](dict(rec, aggregate=agg, ood=ood_rec))
    for k in ("gates", "advance"):
        if rec.get(k) is not None and _json(rec[k]) != _json(gates[k]):
            bad(f"the stored {k} is not the one recomputed from its cells")
    return gates


def new_record_path(path):
    """A new run writes only where nothing exists, and never into the released study's folder."""
    if os.path.exists(path):
        raise StudyError(f"{path} exists; a record is written once")
    hist = os.path.abspath(RUN_DIR)
    if os.path.commonpath([os.path.abspath(path), hist]) == hist:
        raise StudyError(f"{path} is inside {RUN_DIR}, the released study's records; write new runs elsewhere")
    return path


def development_gates(res):
    """Prereg section 13 primary promotion gates (also the final split's gates, section 14)."""
    agg, cells = res["aggregate"], res["cells"]
    a, b5 = agg["128"], agg["512"]
    cb = [c for c in cells if c["budget"] == 128]
    ood = res["ood"]["per_condition"]
    g = dict(D_minus_A_128=a["D_minus_A"]["mean"] >= PROMOTION_FLOOR_DB and a["D_minus_A"]["ci95"][0] > 0,
             D_minus_B_128=a["D_minus_B"]["mean"] >= PROMOTION_FLOOR_DB and a["D_minus_B"]["ci95"][0] > 0,
             no_negative_seed_mean_128=min(a["D_minus_A"]["min_seed_mean"], a["D_minus_B"]["min_seed_mean"]) >= 0,
             raw_feasibility_ge_098=a["raw_proposal_feasibility"] >= 0.98,
             unique_ge_095=a["unique_fraction"] >= 0.95,
             winner_feasibility_eq_1=all(c[x]["feasible"] for c in cb for x in "ABCD"),
             winner_mass_eq_1=all(c[x]["mass_ok"] for c in cb for x in "ABCD"),
             D_not_worse_than_A_512=b5["D_minus_A"]["mean"] >= -PROMOTION_FLOOR_DB,
             ood_complete=len(ood) == len({c["cond_index"] for c in cells}) and all(o["cls"] for o in ood))
    return dict(gates=g, advance=all(g.values()))


def _evaluate(a):
    if a.split == "final_physgen_v1":
        raise StudyError("final_physgen_v1 runs only through `acoustic-liner evaluation shards` (manifest -> run -> merge)")
    model_sha = CKPT.checkpoint_identity(CKPT.load_checkpoint(a.checkpoint)[1])["model_sha256"]
    require_previous_stage(a.split, model_sha)
    out = new_record_path(os.path.join(REPRO_DIR, f"{a.split}.json"))
    os.makedirs(REPRO_DIR, exist_ok=True)
    tr, va = frozen_splits()
    conds = PCN.evaluation_split(a.split)
    print(f"opening {a.split}: {len(conds)} conditions x seeds {SPLIT_SEEDS[a.split]} x budgets "
          f"{SPLIT_BUDGETS[a.split]}", flush=True)
    n = len(conds) * len(SPLIT_SEEDS[a.split]) * len(SPLIT_BUDGETS[a.split])
    res = evaluate(a.split, conds, a.checkpoint, SPLIT_BUDGETS[a.split], SPLIT_SEEDS[a.split], workers=a.workers,
                   ood=PCN.OODScorer(tr, va), progress=cell_printer(n))
    if a.split == "pilot_physgen_v1":
        res.update(pilot_gates(res))
    else:   # development: gates are assessed from the record against prereg section 13 (not computed here)
        res.update(gates=None, advance=None)
    tmp = out + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, out)
    print(json.dumps(dict(gates=res["gates"], advance=res["advance"], wall_s=res["wall_s"],
                          aggregate={b: {k: v for k, v in r.items() if k in ("D_minus_A", "D_minus_B", "C32_minus_B32")}
                                     for b, r in res["aggregate"].items()}), indent=1, default=str))
    print(f"wrote {out}")
if __name__ == "__main__":
    main()
