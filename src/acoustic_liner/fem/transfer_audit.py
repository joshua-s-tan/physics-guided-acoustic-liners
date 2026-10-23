# -*- coding: utf-8 -*-
"""PHYS-GEN-V1 synthetic FEM-transfer audit. Descriptive; it does not alter promotion.

20 new synthetic conditions from the `fem_audit` stream (evaluation family). For each: the Arm-D winner at B = 128,
seed 0 (released selected checkpoint, unchanged `run_seeded_cma`), then the UNCHANGED four-corner partial-coverage FEM
(`fem.robust.evaluate_design_robust_fem`: kappa lo/hi x anchor low/high, bare alpha 0.05) at the winner's coverage.
Reports oracle-minus-FEM by performance, coverage, placement and corner; median |oracle - FEM| > 1 dB is a transfer
failure of the oracle-level claim.

GEOMETRY RULE (declared here, before any audit FEM ran; synthetic conditions carry A_wall but no shape): the corrected
Falcon representative interior (R 2.6, H_barrel 6.7, H_total 13.2 m) scaled isotropically by s = sqrt(A_wall / A_ref)
so the lined wall area equals the condition's A_wall. kappa range = nominal -+ oracle DELTA_KAPPA_DEFAULT (lo clipped
at 0). No payload (synthetic conditions have none). Vehicle scenarios are never touched.

MESH: 0.15 m. The first attempt used the run_scenario_fem default (0.35 m) and stopped at the fail-closed coverage
gate on a small scaled cavity (requested 0.674 vs achieved 0.701); the gate prescribes refinement, so the audit uses
0.15 m (FEM mechanics unchanged). Search winners are cached in search_winners.json so a FEM-stage failure does not
rerun the searches. The cache is reused only when its key is equal: split hash, checkpoint weights and configuration
hash, base seed, budget, seed, the package source fingerprint and the numerical environment (Python, Torch, NumPy,
thread count). The fingerprint makes any code edit invalidate the cache; it does not cover changes outside the package
(for example a different BLAS build under the same NumPy version).

    acoustic-liner fem transfer-audit run --workers 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from . import robust as DRUN
from ..physics import oracle as O2
from ..evaluation import splits as PCN
from ..models import linergen as PM
from ..optimization import seeded_cma as HO
from ..models import checkpoint as CKPT
from . import scenarios as S
from ..evaluation import conditions as COND
from ..evaluation import leakage as LEAK
from ..evaluation import study as ST
from ..physics import quality as Q
from ..provenance import hashing as H

N_CONDITIONS = 20
BUDGET = 128
SEED = 0
SPLIT_NAME = "fem_audit_physgen_v1"
REF_GEOMETRY = dict(R_fair=2.6, H_barrel=6.7, H_total=13.2)
TRANSFER_FAILURE_MEDIAN_DB = 1.0
MESH_SIZE = 0.15
OUT_DIR = Path(ST.RUN_DIR) / "fem_audit"
RECORD = OUT_DIR / "fem_transfer_audit.json"
CHECKPOINT = Path(ST.TRAIN_DIR) / "selected.pt"


class AuditError(RuntimeError):
    pass


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float).encode()).hexdigest()


def audit_conditions():
    conds = COND.synthetic_conditions(N_CONDITIONS, PCN.named_seeds()["fem_audit"], SPLIT_NAME)
    clash = {PCN.condition_hash(c) for c in conds} & PCN.forbidden_condition_hashes()
    if clash:
        raise AuditError(f"{len(clash)} audit conditions collide with a sealed / evaluated split")
    return conds


def synthetic_scenario(cond, ci):
    """Declared geometry rule (module docstring) -> a PhysicalScenario for the unchanged FEM path."""
    ref = S.FairingGeometry(**REF_GEOMETRY)
    s = math.sqrt(float(cond["A_wall"]) / ref.lined_wall_area)
    g = S.FairingGeometry(R_fair=ref.R_fair * s, H_barrel=ref.H_barrel * s, H_total=ref.H_total * s,
                          provenance="fem-audit rule: corrected Falcon representative interior scaled to A_wall")
    k = float(cond["kappa"])
    dk = float(O2.DELTA_KAPPA_DEFAULT)
    spl = tuple(float(x) for x in cond["spl"])
    sem = S.ClaimSemantics(id="fem_transfer_audit_synthetic", metric="relative_dOASPL_only",
                           absolute_post_treatment_spl_allowed=False, validated_retrofit_claim_allowed=False,
                           blankets_modelled=False, blanket_replacement_claim_allowed=False,
                           alpha_bare_source="modelling_assumption", kappa_source="modelling_assumption")
    src = S.SourceConfig(name=SPLIT_NAME, revision="synthetic", url="synthetic", blankets="n/a", fill_factor=None)
    oaspl = 10.0 * math.log10(sum(10.0 ** (x / 10.0) for x in spl))
    return S.PhysicalScenario(
        name=f"{SPLIT_NAME}:{ci}", geometry=g, source=src, semantics=sem, spl_dB=spl, oaspl_dB=oaspl,
        alpha_bare_nominal=DRUN.ALPHA_BARE_NOMINAL, alpha_bare_range=(0.02, 0.10), kappa_nominal=k,
        kappa_eval_range=(max(0.0, k - dk), k + dk), is_vehicle=False)


def _search_worker(args):
    import torch
    torch.set_num_threads(1)
    cond, ci, base_seed, path = args
    with LEAK.sealed_guard():
        model = CKPT.load_checkpoint(path)[0]
        L = HO.run_seeded_cma(cond, ci, SEED, BUDGET, base_seed, PM.proposal_fn(model, cond), "armD")
    ok = [e for e in L.trace if e["feasible_budget"]]
    if not ok:
        return dict(cond_index=ci, feasible=False)
    w = min(ok, key=Q.quality_rank_key)
    return dict(cond_index=ci, feasible=True, charged=len(L.trace), best_val=float(w["best_val"]),
                best_cov=float(w["best_cov"]), mass=float(w["mass"]), design=w["design"], geometry=w["geometry"])


def _fem_worker(args):
    cond, ci, win, out = args
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scen = synthetic_scenario(cond, ci)
    design = dict(win["design"])
    design["groups"] = [dict(g, cavity_volume=O2._group_cavity_volume(g)) for g in design["groups"]]
    cfg = out / "scenario.json"
    cfg.write_text(json.dumps(scen.to_dict(), sort_keys=True, default=str, indent=1))
    fin = out / "winner.json"
    fin.write_text(json.dumps(dict(win, condition_hash=PCN.condition_hash(cond)), sort_keys=True, indent=1))
    from ..evaluation import requests as R
    sha = R.canonical_scenario_sha256(scen)
    corners, robust = DRUN.evaluate_design_robust_fem(
        scen, O2, design, coverage=win["best_cov"], alpha_bare=DRUN.ALPHA_BARE_NOMINAL, out_dir=out,
        config_path=cfg, finalists_path=fin, scenario_sha256=sha, mesh_size=MESH_SIZE)
    return dict(cond_index=ci, scenario_sha256=sha, geometry=dict(R_fair=scen.geometry.R_fair,
                H_barrel=scen.geometry.H_barrel, H_total=scen.geometry.H_total), robust_fem=robust["robust_value"],
                robust_corner=list(robust["robust_corner_id"]),
                corners=[{k: c[k] for k in ("corner_id", "kappa", "anchor", "coverage", "value", "artifact_sha256")}
                         for c in corners])


def _pool(fn, jobs, workers, label):
    t0, out = time.perf_counter(), [None] * len(jobs)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, j): i for i, j in enumerate(jobs)}
        for k, f in enumerate(as_completed(futs), 1):
            out[futs[f]] = f.result()
            print(f"{label} {k}/{len(jobs)}  {time.perf_counter() - t0:.0f}s", flush=True)
    return out


def cache_key(conds, ck, base):
    """What the cached search winners depend on (see the module docstring)."""
    return dict(split_hash=PCN.split_hash(conds), model_sha256=ck["model_sha256"],
                config_sha256=ck["binding"]["config_sha256"], base_seed=base, budget=BUDGET, seed=SEED, arm="D",
                package_source_sha256=H.package_source_sha256(), environment=ST._environment())


def summarize(rows):
    d = [r for r in rows if r.get("diff") is not None]
    absd = [abs(r["diff"]) for r in d]
    by_corner = {}
    for r in d:
        for c in r["corners"]:
            by_corner.setdefault("_".join(map(str, c["corner_id"])), []).append(r["oracle"] - c["value"])
    med = float(np.median(absd)) if absd else None
    return dict(n=len(rows), n_evaluated=len(d), median_abs_diff_dB=med,
                mean_diff_dB=float(np.mean([r["diff"] for r in d])) if d else None,
                max_abs_diff_dB=float(max(absd)) if absd else None,
                by_corner_median_diff_dB={k: float(np.median(v)) for k, v in sorted(by_corner.items())},
                corr_oracle_fem=float(np.corrcoef([r["oracle"] for r in d], [r["fem"] for r in d])[0, 1])
                if len(d) > 2 else None,
                transfer_failure=None if med is None else bool(med > TRANSFER_FAILURE_MEDIAN_DB),
                rule=f"median |oracle - FEM| > {TRANSFER_FAILURE_MEDIAN_DB} dB = transfer failure (descriptive)")


def run(workers):
    if RECORD.exists():
        raise AuditError(f"{RECORD} exists; the audit runs once")
    model, ck = CKPT.load_checkpoint(str(CHECKPOINT))
    try:
        CKPT.require_released(ck)
    except CKPT.CheckpointError as e:
        raise AuditError(str(e))
    conds = audit_conditions()
    base = PCN.named_seeds()["evaluation"]
    cache = OUT_DIR / "search_winners.json"
    key = cache_key(conds, ck, base)
    if cache.exists() and json.load(open(cache))["key"] == key:
        wins = json.load(open(cache))["winners"]
        print(f"reusing {len(wins)} cached search winners", flush=True)
    else:
        wins = _pool(_search_worker, [(c, i, base, str(CHECKPOINT)) for i, c in enumerate(conds)], workers, "search")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(dict(key=key, winners=wins), default=float))
    jobs = [(conds[w["cond_index"]], w["cond_index"], w, str(OUT_DIR / f"cond_{w['cond_index']:02d}"))
            for w in wins if w["feasible"]]
    fems = {f["cond_index"]: f for f in _pool(_fem_worker, jobs, workers, "fem")}
    rows = []
    for w in wins:
        ci, f = w["cond_index"], fems.get(w["cond_index"])
        rows.append(dict(cond_index=ci, condition_hash=PCN.condition_hash(conds[ci]),
                         A_wall=conds[ci]["A_wall"], kappa=conds[ci]["kappa"], target=conds[ci]["dOASPL_target"],
                         feasible=w["feasible"], oracle=w.get("best_val"), coverage=w.get("best_cov"),
                         mass=w.get("mass"), design_sha256=_sha(w["design"]) if w["feasible"] else None,
                         geometry_key_sha256=_sha(w["geometry"]) if w["feasible"] else None,
                         fem=None if f is None else f["robust_fem"],
                         diff=None if f is None else w["best_val"] - f["robust_fem"],
                         **({} if f is None else {k: f[k] for k in ("scenario_sha256", "geometry", "robust_corner",
                                                                       "corners")})))
    rec = dict(study="physgen_v1_fem_transfer_audit", split=SPLIT_NAME, split_hash=PCN.split_hash(conds),
               n_conditions=N_CONDITIONS, budget=BUDGET, seed=SEED, base_seed=base, arm="D",
               checkpoint_model_sha256=ck["model_sha256"], geometry_rule=REF_GEOMETRY,
               fem=dict(corners=[list(c) for c in DRUN.PREREGISTERED_CORNERS], alpha_bare=DRUN.ALPHA_BARE_NOMINAL,
                        mesh_size=MESH_SIZE),
               rows=rows, summary=summarize(rows))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RECORD.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True, default=float))
    os.replace(tmp, RECORD)
    print(json.dumps(rec["summary"], indent=1), flush=True)
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(prog="acoustic-liner fem transfer-audit")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    a = p.parse_args(argv)
    run(a.workers)


if __name__ == "__main__":
    main()
