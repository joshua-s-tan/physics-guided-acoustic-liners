# -*- coding: utf-8 -*-
"""PHYS-GEN-V1 sharded evaluation: ONE evaluation split across machines as deterministic condition shards.

    manifest  one immutable JSON listing every expected cell (condition index + hash, seed, budget, fixed shard) and the
              input hashes every machine must reproduce (checkpoint identity, package source fingerprint, committed
              `src/` git tree, dependency set, environment). A machine with uncommitted `src/` changes is refused.
    run       one machine runs ONLY its shard; each finished cell is written atomically to <out>/<cell_id>.json; a rerun
              resumes the missing cell ids and never replaces a completed record.
    merge     fail-closed: exactly the manifest's cells, no duplicate / unexpected / foreign-manifest record, every arm
              present; then the clustered-bootstrap aggregate is computed ONCE on the complete set.

All of a condition's seeds and budgets live on one shard (the condition is the bootstrap unit). Randomness comes from
run_cell(cond, global condition index, seed, budget, base seed) only, never worker order or machine identity, so a
sharded run reproduces the single-machine cell records exactly. Neither machine decides pass/fail.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import time

from . import splits as PCN
from ..models import checkpoint as CKPT
from . import statistics as STAT
from . import study as ST
from ..provenance import hashing as H
from ..provenance import schemas as SC

ARMS = ("A", "B", "C", "D")
MANIFEST_DIR = os.path.join(ST.REPRO_DIR, "shards")      # new runs only; never the released study's folder


class ShardError(ST.StudyError):
    """Manifest / input-hash / completeness violation (fail closed)."""


def _canon(obj):
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def _sha_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def cell_id(ci, seed, budget):
    return f"c{int(ci):03d}-s{int(seed)}-b{int(budget)}"


# =================================================================================================================
#  input hashes every machine must reproduce
# =================================================================================================================
def dependency_sha256():
    from importlib import metadata
    return _sha_obj(sorted(f"{(d.metadata['Name'] or '').lower()}=={d.version}" for d in metadata.distributions()))


def git_src_tree(repo="."):
    """Git tree hash of committed `src/`; refuses a dirty `src/` (uncommitted code cannot be reproduced elsewhere)."""
    try:
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=repo, capture_output=True,
                               text=True, check=True).stdout.strip()
        tree = subprocess.run(["git", "rev-parse", "HEAD:src"], cwd=repo, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        raise ShardError(f"cannot read the git tree of src/: {e}")
    if dirty:
        raise ShardError(f"src/ has uncommitted changes:\n{dirty}")
    return tree


def input_hashes(checkpoint_path, repo="."):
    model, ck = CKPT.load_checkpoint(checkpoint_path)
    ident = CKPT.checkpoint_identity(ck)          # the checkpoint's own identity (historical or public binding)
    return dict(**ident, package_source_sha256=H.package_source_sha256(), src_git_tree=git_src_tree(repo),
                dependency_sha256=dependency_sha256(), environment=ST._environment(),
                base_seed=PCN.named_seeds()["evaluation"])


# =================================================================================================================
#  manifest
# =================================================================================================================
def assign_shards(n_conditions, weights):
    """Contiguous condition blocks sized by the predeclared weights (largest remainder; deterministic)."""
    w = [float(x) for x in weights]
    if not w or min(w) <= 0:
        raise ShardError("shard weights must be positive")
    raw = [n_conditions * x / sum(w) for x in w]
    size = [int(r) for r in raw]
    for i in sorted(range(len(w)), key=lambda i: (-(raw[i] - size[i]), i))[:n_conditions - sum(size)]:
        size[i] += 1
    return [s for s, k in enumerate(size) for _ in range(k)]


def _jobs(specs, conds, base, checkpoint_path):
    return [(conds[s["cond_index"]], s["cond_index"], s["seed"], s["budget"], base, checkpoint_path)
            for s in specs]


def run_cells(jobs, workers, on_done):
    """Every cell runs in a 1-thread torch context (ST._cell_worker in a pool, or in-process with threads pinned to
    1): the multi-thread kernels round differently in the last place, and CMA can amplify that."""
    if workers:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(ST._cell_worker, j): i for i, j in enumerate(jobs)}
            for k, f in enumerate(as_completed(futs), 1):
                on_done(k, futs[f], f.result())
        return
    import torch
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for k, j in enumerate(jobs, 1):
            on_done(k, k - 1, ST._cell_worker(j))
    finally:
        torch.set_num_threads(n)


def canary_cells(canary_conds, checkpoint_path, budget, workers=0):
    """Reference records for a few TRAINING-VALIDATION cells; every machine must reproduce them bit-for-bit."""
    specs = [dict(cond_index=i, seed=0, budget=int(budget)) for i in range(len(canary_conds))]
    out = [None] * len(specs)
    run_cells(_jobs(specs, canary_conds, PCN.named_seeds()["evaluation"], checkpoint_path), workers,
              lambda k, i, cell: out.__setitem__(i, cell))
    return dict(conditions_hash=PCN.split_hash(canary_conds), budget=int(budget), records=_canon(out))


def check_canary(m, canary_conds, checkpoint_path, workers=0):
    ref = m["canary"]
    if PCN.split_hash(canary_conds) != ref["conditions_hash"]:
        raise ShardError("canary conditions differ from the manifest")
    got = canary_cells(canary_conds, checkpoint_path, ref["budget"], workers)
    if got["records"] != ref["records"]:
        raise ShardError("this machine does not reproduce the manifest's canary cells bit-for-bit; do not run here")


def build_manifest(split, conds, seeds, budgets, checkpoint_path, weights=(1, 1), repo=".",
                   canary_conds=(), canary_budget=128, workers=0):
    shard = assign_shards(len(conds), weights)
    cells = [dict(cell_id=cell_id(ci, s, b), cond_index=ci, condition_hash=PCN.condition_hash(c), seed=int(s),
                  budget=int(b), shard=shard[ci])
             for ci, c in enumerate(conds) for s in seeds for b in budgets]
    head = dict(schema=SC.schema("shard-manifest"), evaluation_schema_version=ST.EVALUATION_SCHEMA_VERSION,
                split=split, split_hash=PCN.split_hash(conds), n_conditions=len(conds), seeds=[int(s) for s in seeds],
                budgets=[int(b) for b in budgets], shard_weights=[float(x) for x in weights],
                n_shards=len(weights), inputs=input_hashes(checkpoint_path, repo),
                canary=canary_cells(list(canary_conds), checkpoint_path, canary_budget, workers))
    m = _canon(dict(head, cells=cells))
    return dict(m, manifest_sha256=_sha_obj(m))


def write_manifest(m, path):
    if os.path.exists(path):
        raise ShardError(f"{path} exists; the manifest is written once")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def load_manifest(path):
    m = json.load(open(path))
    body = {k: v for k, v in m.items() if k != "manifest_sha256"}
    if _sha_obj(body) != m.get("manifest_sha256"):
        raise ShardError(f"{path}: manifest content does not match its manifest_sha256")
    ids = [c["cell_id"] for c in m["cells"]]
    if len(set(ids)) != len(ids) or len(ids) != m["n_conditions"] * len(m["seeds"]) * len(m["budgets"]):
        raise ShardError("manifest cell list is not the full condition x seed x budget grid")
    return m


def verify_inputs(m, checkpoint_path, conds, repo="."):
    """This machine's inputs must hash identically to the manifest (same checkpoint, package source, committed `src/`
    tree, dependencies and environment); the split's conditions must match cell by cell."""
    if SC.record_version(m, "shard-manifest") != 2:
        raise ShardError("this manifest was written by the sealed study (its inputs include the seal and the lock, "
                         "which are not distributed); write a new manifest to run shards")
    here = _canon(input_hashes(checkpoint_path, repo))
    bad = sorted(k for k in set(m["inputs"]) | set(here) if m["inputs"].get(k) != here.get(k))
    if PCN.split_hash(conds) != m["split_hash"] or len(conds) != m["n_conditions"]:
        bad.append("split_hash")
    else:
        bad += [c["cell_id"] for c in m["cells"] if PCN.condition_hash(conds[c["cond_index"]]) != c["condition_hash"]][:5]
    if bad:
        raise ShardError(f"inputs differ from the manifest: {bad}")
    return here


def open_split(m, checkpoint_path, conds=None):
    """Evaluation splits: previous-stage gate, conditions rebuilt by the split generator and checked against the
    released split hash. Anything else: training-only conditions passed in explicitly (tests / smoke)."""
    split = m["split"]
    if split in ST.EVAL_SPLITS:
        ck = CKPT.load_checkpoint(checkpoint_path)[1]
        ST.require_previous_stage(split, CKPT.checkpoint_identity(ck)["model_sha256"])
        conds = PCN.evaluation_split(split)
        ST.check_evaluation_split(split, conds)
        return conds
    if conds is None or any(c.get("group") not in ("training", "training_validation") for c in conds):
        raise ShardError("non-evaluation manifests run on explicitly passed training-only conditions")
    return conds


# =================================================================================================================
#  run one shard (atomic per-cell records, resume, never reroll)
# =================================================================================================================
def _record_path(out_dir, cid):
    return os.path.join(out_dir, f"{cid}.json")


def read_record(path, m, spec):
    r = json.load(open(path))
    if r.get("manifest_sha256") != m["manifest_sha256"] or r.get("cell_id") != spec["cell_id"]:
        raise ShardError(f"{path}: record belongs to another manifest / cell")
    c = r["cell"]
    if (c["cond_index"], c["seed"], c["budget"], c["condition_hash"]) != (
            spec["cond_index"], spec["seed"], spec["budget"], spec["condition_hash"]):
        raise ShardError(f"{path}: cell identity differs from the manifest")
    missing = [a for a in ARMS + ("proposals",) if a not in c]
    if missing:
        raise ShardError(f"{path}: missing arms {missing}")
    return r


def run_shard(manifest_path, shard, out_dir, checkpoint_path, workers=0, conds=None, repo=".", progress=print,
              canary_conds=()):
    m = load_manifest(manifest_path)
    if not 0 <= int(shard) < m["n_shards"]:
        raise ShardError(f"shard {shard} not in manifest (0..{m['n_shards'] - 1})")
    conds = open_split(m, checkpoint_path, conds)
    here = verify_inputs(m, checkpoint_path, conds, repo)
    os.makedirs(out_dir, exist_ok=True)
    mine = [c for c in m["cells"] if c["shard"] == int(shard)]
    foreign = [f for f in os.listdir(out_dir) if f.endswith(".json")
               and f[:-5] not in {c["cell_id"] for c in mine}]
    if foreign:
        raise ShardError(f"{out_dir} holds records outside shard {shard}: {foreign[:5]}")
    todo = []
    for spec in mine:
        p = _record_path(out_dir, spec["cell_id"])
        if os.path.exists(p):
            read_record(p, m, spec)                  # completed: validated, never replaced
        else:
            todo.append(spec)
    if todo:
        check_canary(m, list(canary_conds), checkpoint_path, workers)   # before any cell of the split is scored
    progress(f"shard {shard}: {len(mine)} cells, {len(mine) - len(todo)} done, {len(todo)} to run "
             f"(manifest {m['manifest_sha256'][:12]})")
    jobs = _jobs(todo, conds, m["inputs"]["base_seed"], checkpoint_path)
    t0 = time.perf_counter()

    def done(k, i, cell):
        spec = todo[i]
        rec = _canon(dict(cell_id=spec["cell_id"], shard=int(shard), manifest_sha256=m["manifest_sha256"],
                          inputs_sha256=_sha_obj(here), cell=cell))
        p = _record_path(out_dir, spec["cell_id"])
        if os.path.exists(p):
            raise ShardError(f"{p} appeared while running; refusing to replace it")
        with open(p + ".tmp", "w") as f:
            json.dump(rec, f, sort_keys=True)
        os.replace(p + ".tmp", p)
        el = time.perf_counter() - t0
        progress(f"cell {k:>4}/{len(todo)}  {spec['cell_id']}  elapsed {ST._hms(el)}  "
                 f"ETA {ST._hms(el / k * (len(todo) - k))}")

    run_cells(jobs, workers, done)
    return len(todo)


# =================================================================================================================
#  merge (fail closed) + one aggregate
# =================================================================================================================
def merge(manifest_path, record_dirs, ood=None):
    m = load_manifest(manifest_path)
    spec = {c["cell_id"]: c for c in m["cells"]}
    found, problems = {}, []
    for d in record_dirs:
        for f in sorted(os.listdir(d)):
            if f.endswith(".tmp"):
                problems.append(f"incomplete write {os.path.join(d, f)}")
                continue
            if not f.endswith(".json"):
                continue
            cid = f[:-5]
            if cid not in spec:
                problems.append(f"unexpected record {os.path.join(d, f)}")
            elif cid in found:
                problems.append(f"duplicate record {cid}")
            else:
                r = read_record(os.path.join(d, f), m, spec[cid])
                if r["shard"] != spec[cid]["shard"]:
                    problems.append(f"{cid} ran on shard {r['shard']}, manifest says {spec[cid]['shard']}")
                found[cid] = r
    missing = [c for c in spec if c not in found]
    if missing:
        problems.append(f"{len(missing)} missing cells, e.g. {missing[:5]}")
    hashes = {r["inputs_sha256"] for r in found.values()}
    if len(hashes) > 1:
        problems.append(f"records were produced under {len(hashes)} different input sets")
    if problems:
        raise ShardError("merge refused:\n  " + "\n  ".join(problems))
    cells = [found[c["cell_id"]]["cell"] for c in m["cells"]]          # manifest order = single-machine job order
    per_cond = {}
    for c in cells:
        per_cond.setdefault(c["cond_index"], set()).add((c["seed"], c["budget"]))
    want = {(s, b) for s in m["seeds"] for b in m["budgets"]}
    if len(per_cond) != m["n_conditions"] or any(v != want for v in per_cond.values()):
        raise ShardError("merge refused: a condition lacks a seed x budget cell")
    return dict(evaluation_schema_version=ST.EVALUATION_SCHEMA_VERSION,
                split=m["split"], split_hash=m["split_hash"], manifest_sha256=m["manifest_sha256"],
                checkpoint={k: m["inputs"][k] for k in CKPT.CHECKPOINT_IDENTITY_KEYS if k in m["inputs"]},
                base_seed=m["inputs"]["base_seed"], budgets=m["budgets"], seeds=m["seeds"],
                run=ST.run_identity(m["inputs"]["package_source_sha256"], m["inputs"]["environment"]), cells=cells,
                aggregate=_canon(STAT.aggregate(cells, m["budgets"])), ood=ood, n_shards=m["n_shards"])   # JSON form


# =================================================================================================================
#  CLI
# =================================================================================================================
N_CANARY, CANARY_BUDGET = 2, 128


def _frozen():
    return ST.frozen_splits()


def _canary_conds():
    return _frozen()[1][:N_CANARY]          # training-validation conditions, never an evaluation split


def _ood_record(conds):
    tr, va = _frozen()
    ood = PCN.OODScorer(tr, va)
    sc, cl = ood.score(conds), ood.classify(conds)
    return dict(ood.record(), per_condition=[dict(cond_index=i, score=float(sc[i]), cls=cl[i])
                                             for i in range(len(conds))])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="acoustic-liner evaluation shards")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("manifest"); mk.add_argument("--split", required=True, choices=ST.EVAL_SPLITS)
    mk.add_argument("--checkpoint", required=True); mk.add_argument("--weights", default="1,1")
    hs = sub.add_parser("hashes"); hs.add_argument("--checkpoint", required=True)
    rn = sub.add_parser("run"); rn.add_argument("--manifest", required=True); rn.add_argument("--shard", type=int,
                                                                                            required=True)
    rn.add_argument("--checkpoint", required=True); rn.add_argument("--out", required=True)
    rn.add_argument("--workers", type=int, default=16)
    mg = sub.add_parser("merge"); mg.add_argument("--manifest", required=True)
    mg.add_argument("--records", nargs="+", required=True); mg.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "hashes":                    # compare this output between machines before running
        h = input_hashes(a.checkpoint)
        can = canary_cells(_canary_conds(), a.checkpoint, CANARY_BUDGET, workers=N_CANARY)
        print(json.dumps(dict(h, canary_sha256=_sha_obj(can)), indent=1, sort_keys=True, default=str))
    elif a.cmd == "manifest":
        ck = CKPT.load_checkpoint(a.checkpoint)[1]
        ST.require_previous_stage(a.split, CKPT.checkpoint_identity(ck)["model_sha256"])
        conds = PCN.evaluation_split(a.split)
        ST.check_evaluation_split(a.split, conds)
        m = build_manifest(a.split, conds, ST.SPLIT_SEEDS[a.split], ST.SPLIT_BUDGETS[a.split],
                           a.checkpoint, [float(x) for x in a.weights.split(",")], canary_conds=_canary_conds(),
                           canary_budget=CANARY_BUDGET, workers=N_CANARY)
        os.makedirs(MANIFEST_DIR, exist_ok=True)
        path = ST.new_record_path(os.path.join(MANIFEST_DIR, f"{a.split}_manifest.json"))
        write_manifest(m, path)
        print(f"wrote {path}  sha256 {m['manifest_sha256']}  "
              f"cells per shard {[sum(c['shard'] == s for c in m['cells']) for s in range(m['n_shards'])]}")
    elif a.cmd == "run":
        run_shard(a.manifest, a.shard, a.out, a.checkpoint, workers=a.workers,
                  progress=lambda s: print(s, flush=True), canary_conds=_canary_conds())
    else:
        ST.new_record_path(a.out)
        m = load_manifest(a.manifest)
        res = merge(a.manifest, a.records, ood=_ood_record(PCN.evaluation_split(m["split"])))
        # pilot: its own gates; development and final (section 14: final gates == section-13 primary gates)
        res.update(ST.pilot_gates(res) if m["split"] == "pilot_physgen_v1" else ST.development_gates(res))
        with open(a.out + ".tmp", "w") as f:
            json.dump(res, f, indent=1, default=str)
        os.replace(a.out + ".tmp", a.out)
        print(f"wrote {a.out}: {len(res['cells'])} cells merged from {len(a.records)} shard directories")


if __name__ == "__main__":
    main()
