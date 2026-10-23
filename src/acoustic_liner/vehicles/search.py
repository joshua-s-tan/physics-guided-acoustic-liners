# -*- coding: utf-8 -*-
"""Vehicle PHYS-GEN search, finalist freeze and initial payload-resolving four-corner FEM
(Phases 2-3 of the vehicle FEM/CFD plan, a historical development document that is not distributed).
Consumes ONLY `vehicles.scenarios` (approved hashes pinned below).

    search  OOD-route the PRIMARY condition with the locked scorer, then five independent B = 2048 runs (seeds 0-4) of
            the routed authoritative method on the primary corner only; keep the best canonically feasible,
            mass-compliant winner (robust exact dOASPL desc, then mass, coverage, geometry hash asc); check the frozen
            geometry + coverage on the secondary corner (never reselects). FEM never chooses.
    fem     unchanged four-corner partial-coverage FEM (kappa lo/hi x anchor low/high, bare alpha 0.05) on the
            payload-resolving vehicle mesh, for every source corner x payload case x mesh size (0.2 / 0.15 / 0.1 m).

Every record is stamped with the vehicle identity; FEM records descend from the frozen finalist record.
Vehicle outcomes never feed back into training, checkpoint, OOD threshold or CMA initialization.

    acoustic-liner vehicles search search --vehicle falcon --workers 5
    acoustic-liner vehicles search fem --vehicle falcon --workers 8
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

from ..physics import oracle as O2
from ..fem import robust as DRUN
from ..evaluation.historical_splits import selection_rule as SEL
from ..optimization import baseline as PB
from ..evaluation import splits as PCN
from ..models import linergen as PM
from ..optimization import seeded_cma as HO
from ..models import checkpoint as CKPT
from ..fem import placement as RP
from ..evaluation import conditions as COND
from ..evaluation import leakage as LEAK
from ..design import canonical as CAN
from ..evaluation import study as ST
from ..physics import quality as Q
from ..provenance import hashing as H
from ..provenance import schemas as SC
from . import fem as VF
from . import scenarios as V

APPROVED_SCENARIO_HASHES = {
    "falcon": "b019791a1e68ff7391892e2a4279055e9e80e3ee51bc93eb679439edbe48cdb7",
    "atlas": "6c0ace5e181c12f0230d1b62d3322cbe146c6d361fdfc88014b79c8a53ae4272",
}
TARGET_DB = 0.9213
FALCON_MASS_KG = 400.0
BUDGET = 2048
SEEDS = (0, 1, 2, 3, 4)
COND_INDEX = {"falcon": 1000, "atlas": 1001}          # CMA stream indices (disjoint from every evaluation split)
MESH_SIZES = (0.2, 0.15, 0.1)
UNMET_LABEL = "best oracle candidate — target unmet"
RUN_DIR = Path("artifacts") / "dev" / "vehicle_v1"
CHECKPOINT = Path(ST.TRAIN_DIR) / "selected.pt"
A_OTHER_NOTE = ("A_other is an assumed effective background-loss term that may include payload-associated losses; "
                "it is not measured payload absorption")


class VehiclePipelineError(RuntimeError):
    pass


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float).encode()).hexdigest()


def scenario(vehicle):
    vs = {"falcon": V.falcon9_unblanketed, "atlas": V.atlas_i_14ft}[vehicle]()
    if vs.scenario_hash != APPROVED_SCENARIO_HASHES[vehicle]:
        raise VehiclePipelineError(f"{vehicle} scenario hash {vs.scenario_hash[:12]} is not the approved "
                                   f"{APPROVED_SCENARIO_HASHES[vehicle][:12]}")
    return vs


def mass_limit(vehicle, vs):
    if vehicle == "falcon":
        return dict(m_total_kg=FALCON_MASS_KG, rule="Falcon mass limit 400.0 kg")
    f = scenario("falcon")
    m = FALCON_MASS_KG * vs.geometry.lined_wall_area / f.geometry.lined_wall_area
    return dict(m_total_kg=m, rule="400.0 kg x A_wall_Atlas / A_wall_Falcon", A_wall_atlas=vs.geometry.lined_wall_area,
                A_wall_falcon=f.geometry.lined_wall_area, falcon_scenario_hash=f.scenario_hash)


def condition(vs, corner_id, m_total):
    spl = np.asarray(vs.model_levels(corner_id), float)
    return dict(id=f"{vs.name}:{corner_id}", group="vehicle", weights=list(map(float, SEL._weights_from_spl(spl))),
                spl=spl.tolist(), dOASPL_target=TARGET_DB, m_total_target=float(m_total),
                A_wall=float(vs.geometry.lined_wall_area), kappa=float(vs.acoustics["kappa_nominal"]))


def ood_route(cond):
    from ..evaluation import shards as PSH
    tr, va = PSH._frozen()
    sc = PCN.OODScorer(tr, va)
    score, cls = float(sc.score([cond])[0]), sc.classify([cond])[0]
    method = {"in": "physgen_seeded_cma", "near": "physgen_random_mixture_cma", "beyond": "plain_quality_cma"}[cls]
    return dict(sc.record(), score=score, cls=cls, authoritative_method=method)


def mixture_fn(gen, frac=PCN.NEAR_MIX_FRACTION):
    def f(rng, n):
        k = int(round(n * frac))
        return list(gen(rng, k)) + list(HO.random_proposals(rng, n - k))
    return f


def _seed_worker(args):
    import torch
    torch.set_num_threads(1)
    cond, ci, seed, method, path = args
    base = PCN.named_seeds()["evaluation"]
    with LEAK.sealed_guard():
        if method == "plain_quality_cma":
            L = PB.run_quality_cma_exact(cond, ci, seed, BUDGET, base)
        else:
            gen = PM.proposal_fn(CKPT.load_checkpoint(path)[0], cond)
            fn = gen if method == "physgen_seeded_cma" else mixture_fn(gen)
            L = HO.run_seeded_cma(cond, ci, seed, BUDGET, base, fn, "armD")
    ok = [e for e in L.trace if e["feasible_budget"] and e["mass"] <= cond["m_total_target"]]
    res = dict(seed=seed, charged=len(L.trace), exhausted=bool(L.exhausted), n_feasible=len(ok))
    if ok:
        w = min(ok, key=Q.quality_rank_key)
        res.update(best_val=float(w["best_val"]), coverage=float(w["best_cov"]), mass=float(w["mass"]),
                   design=w["design"], geometry_hash=_sha(w["geometry"]))
    return res


def select(runs):
    ok = [r for r in runs if r.get("best_val") is not None]
    if not ok:
        raise VehiclePipelineError("no canonically feasible, mass-compliant result in any seed")
    return min(ok, key=lambda r: (-r["best_val"], r["mass"], r["coverage"], r["geometry_hash"]))


def exact_at(design, cond, coverage):
    """Robust exact dOASPL of a FROZEN geometry at a FROZEN coverage (no re-optimization)."""
    d = CAN.canonical_design(design)
    return float(Q._worst(O2._a_band(d), cond, float(coverage)))


def is_historical_audit_seal(seal):
    """True for the archived seal written by the historical implementation (no version field of either kind)."""
    return "provenance_schema_version" not in seal and "schema" not in seal


def _quoted_audit_seal():
    """The audit seal as a record quotes it: under `historical_seal` when it is the archived seal of the historical
    implementation (its source hash is historical metadata), under `seal` when this package wrote it."""
    seal = json.load(open(_audit_seal_path()))
    return {"historical_seal" if is_historical_audit_seal(seal) else "seal": seal}


def _audit_seal_path():
    from ..fem import transfer_audit as FTA
    return FTA.RECORD.with_name("fem_transfer_audit.seal.json")


def seal_audit(run_commit):
    """After the audit finishes: bind the record bytes, so validation rejects any later edit of the record. The seal
    names the commit that produced the record and the package source fingerprint at sealing, for information."""
    from ..fem import transfer_audit as FTA
    import subprocess
    from .. import paths
    rec = require_phase1(check_seal=False)
    seal = dict(schema=SC.schema("fem-audit-seal"), record_sha256=_sha_file_bytes(FTA.RECORD),
                package_source_sha256=H.package_source_sha256(),
                commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.ROOT, capture_output=True,
                                      text=True).stdout.strip(), run_commit=run_commit, summary=rec["summary"])
    p = _audit_seal_path()
    if p.exists():
        raise VehiclePipelineError(f"{p} exists; the audit is sealed once")
    p.write_text(json.dumps(seal, indent=1, sort_keys=True))
    return seal


def _sha_file_bytes(p):
    """Line-ending-normalized file hash (git stores LF; a Windows checkout may restore CRLF)."""
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def require_phase1(check_seal=True):
    """Phase 1 unlocks ONLY on a complete, valid, unedited, current audit: 20/20 feasible rows with finite oracle and
    FEM values, transfer_failure exactly False, summary reproducible from the rows, split / checkpoint / mesh /
    budget / seed / arm matching the pinned inputs, and (sealed) record bytes unchanged."""
    from ..fem import transfer_audit as FTA
    if not FTA.RECORD.exists():
        raise VehiclePipelineError("Phase 1 FEM-transfer audit record missing; vehicle search stays closed")
    rec = json.load(open(FTA.RECORD))
    bad = []
    conds = COND.synthetic_conditions(FTA.N_CONDITIONS, PCN.named_seeds()["fem_audit"], FTA.SPLIT_NAME)
    pins = dict(study="physgen_v1_fem_transfer_audit", split=FTA.SPLIT_NAME, split_hash=PCN.split_hash(conds),
                n_conditions=FTA.N_CONDITIONS, budget=FTA.BUDGET, seed=FTA.SEED, arm="D",
                base_seed=PCN.named_seeds()["evaluation"], checkpoint_model_sha256=CKPT.released_weights_sha256())
    bad += [f"{k} {rec.get(k)!r} != {v!r}" for k, v in pins.items() if rec.get(k) != v]
    if rec.get("fem", {}).get("mesh_size") != FTA.MESH_SIZE:
        bad.append(f"mesh {rec.get('fem', {}).get('mesh_size')!r} != {FTA.MESH_SIZE}")
    rows = rec.get("rows", [])
    if sorted(r.get("cond_index") for r in rows) != list(range(FTA.N_CONDITIONS)):
        bad.append("rows do not cover conditions 0-19 exactly once")
    for r in rows:
        ci = r.get("cond_index")
        fin = all(isinstance(r.get(k), (int, float)) and math.isfinite(r[k]) for k in ("oracle", "fem", "diff"))
        if not (r.get("feasible") is True and fin and len(r.get("corners", [])) == 4):
            bad.append(f"row {ci}: infeasible / non-finite / incomplete corners")
        elif isinstance(ci, int) and 0 <= ci < len(conds) and r.get("condition_hash") != PCN.condition_hash(conds[ci]):
            bad.append(f"row {ci}: condition hash mismatch")
        elif abs(r["diff"] - (r["oracle"] - r["fem"])) > 1e-12:
            bad.append(f"row {ci}: diff != oracle - fem")
    s = rec.get("summary", {})
    if s.get("n_evaluated") != FTA.N_CONDITIONS or s.get("transfer_failure") is not False:
        bad.append(f"n_evaluated {s.get('n_evaluated')} / transfer_failure {s.get('transfer_failure')!r}")
    if not bad and json.loads(json.dumps(FTA.summarize(rows), default=float)) != s:
        bad.append("summary is not reproducible from the rows (edited)")
    if check_seal:
        p = _audit_seal_path()
        if not p.exists():
            bad.append("audit is not sealed (acoustic-liner vehicles search seal-audit)")
        else:
            seal = json.load(open(p))
            if not is_historical_audit_seal(seal) and "schema" in seal:
                try:
                    SC.record_version(seal, "fem-audit-seal")
                except SC.SchemaError as e:
                    bad.append(str(e))
            if seal.get("record_sha256") != _sha_file_bytes(FTA.RECORD):
                bad.append("audit record bytes changed after sealing")
    if bad:
        raise VehiclePipelineError("Phase 1 audit does not unlock vehicle search: " + "; ".join(bad[:8]))
    return rec


def search(vehicle, workers):
    vs = scenario(vehicle)
    out = RUN_DIR / vs.name
    fin_path = out / "finalist.json"
    if fin_path.exists():
        raise VehiclePipelineError(f"{fin_path} exists; the finalist is frozen")
    audit = require_phase1()
    model, ck = CKPT.load_checkpoint(str(CHECKPOINT))
    if ck.get("model_sha256") != CKPT.released_weights_sha256():
        raise VehiclePipelineError("checkpoint is not the released selected checkpoint")
    ml = mass_limit(vehicle, vs)
    prim, sec = vs.corners["primary"]["id"], vs.corners["secondary"]["id"]
    cp, cs = condition(vs, prim, ml["m_total_kg"]), condition(vs, sec, ml["m_total_kg"])
    route = ood_route(cp)
    route["secondary_diagnostic"] = ood_route(cs)["score"]
    print(f"{vs.name}: OOD {route['cls']} (score {route['score']:.3f}, near {route['near']:.3f}, threshold "
          f"{route['threshold']:.3f}) -> {route['authoritative_method']}", flush=True)
    jobs = [(cp, COND_INDEX[vehicle], s, route["authoritative_method"], str(CHECKPOINT)) for s in SEEDS]
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0, runs = time.perf_counter(), [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        futs = {ex.submit(_seed_worker, j): i for i, j in enumerate(jobs)}
        for k, f in enumerate(as_completed(futs), 1):
            runs[futs[f]] = r = f.result()
            print(f"seed {r['seed']} done ({k}/{len(jobs)}): best {r.get('best_val')} cov {r.get('coverage')} "
                  f"mass {r.get('mass')}  {time.perf_counter() - t0:.0f}s", flush=True)
    w = select(runs)
    p_val, s_val = exact_at(w["design"], cp, w["coverage"]), exact_at(w["design"], cs, w["coverage"])
    if abs(p_val - w["best_val"]) > 1e-9:
        raise VehiclePipelineError(f"re-scored primary {p_val} != search value {w['best_val']}")
    meets = p_val >= TARGET_DB and s_val >= TARGET_DB and w["mass"] <= ml["m_total_kg"]
    rec = dict(
        study="vehicle_physgen_v1_search", vehicle=vehicle, status="finalist" if meets else UNMET_LABEL,
        target_dB=TARGET_DB, mass_limit=ml, budget=BUDGET, seeds=list(SEEDS),
        budget_note=f"{len(SEEDS)} independent {BUDGET}-call runs (not one {BUDGET * len(SEEDS)}-call search)",
        ood=route, cond_index=COND_INDEX[vehicle], base_seed=PCN.named_seeds()["evaluation"],
        checkpoint_model_sha256=ck["model_sha256"],
        phase1_audit=dict(_quoted_audit_seal(), summary=audit["summary"]),
        conditions=dict(primary=dict(cp, condition_hash=PCN.condition_hash(cp)),
                        secondary=dict(cs, condition_hash=PCN.condition_hash(cs))),
        runs=[{k: v for k, v in r.items() if k != "design"} for r in runs],
        selection_rule="max robust exact dOASPL; ties: lower mass, lower coverage, smaller geometry hash; FEM not used",
        selected_seed=w["seed"], design=w["design"], design_sha256=_sha(w["design"]), geometry_hash=w["geometry_hash"],
        coverage=w["coverage"], mass_kg=w["mass"], primary_dOASPL=p_val, secondary_dOASPL=s_val,
        semantics="relative dOASPL on published MPE / test envelopes; not measured flight spectra",
        **V.finalist_coverage(vs))
    rec = V.stamp(rec, vs.identity(prim), "finalist")
    out.mkdir(parents=True, exist_ok=True)
    tmp = fin_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True, default=float))
    os.replace(tmp, fin_path)
    print(f"{vs.name}: {rec['status']}  primary {p_val:.4f}  secondary {s_val:.4f} dB  cov {w['coverage']:.4f}  "
          f"mass {w['mass']:.1f}/{ml['m_total_kg']:.1f} kg  -> {fin_path}", flush=True)
    return rec


def load_finalist(vs, path=None):
    path = Path(path) if path else RUN_DIR / vs.name / "finalist.json"
    fin = json.load(open(path))
    vs.require_identity(fin)
    if fin.get("stage") != "finalist" or fin.get("study") != "vehicle_physgen_v1_search":
        raise VehiclePipelineError(f"{path} is not a vehicle search finalist record (placeholders are refused)")
    if fin.get("design_sha256") != _sha(fin["design"]):
        raise VehiclePipelineError("finalist design hash mismatch (tampered)")
    return fin


def four_corner_fem(vs, fin, corner_id, payload_case, mesh_size, wall_beta_fn=None, stage="fem", parent=None):
    """Unchanged four-corner robust mechanics (kappa lo/hi x anchor low/high) on the payload-resolving mesh."""
    parent = fin if parent is None else parent
    design = dict(CAN.canonical_design(fin["design"]))
    design["groups"] = [dict(g, cavity_volume=O2._group_cavity_volume(g)) for g in design["groups"]]
    beta_fn = wall_beta_fn or DRUN._wall_beta_fn(O2, design["groups"], design["D"])
    from ..physics import absorption as acoustics
    from ..physics import constants
    bare = constants.Z_AIR * acoustics.admittance_from_diffuse_alpha(DRUN.ALPHA_BARE_NOMINAL, constants.Z_AIR)
    ps = vs.to_physical_scenario(corner_id)
    band, spl = list(ps.band_hz), np.asarray(ps.spl_dB, float)
    klev = {"kappa_lo": min(ps.kappa_eval_range), "kappa_hi": max(ps.kappa_eval_range)}
    kw = dict(corner_id=corner_id, payload_case=payload_case, mesh_size=mesh_size)
    bare_ms = {kl: VF.run_vehicle_fem(vs, band, bare, kappa=k, coverage=1.0, **kw)["ms_by_freq"]
               for kl, k in klev.items()}
    corners = []
    for kl, an in DRUN.PREREGISTERED_CORNERS:
        t = VF.run_vehicle_fem(vs, band, beta_fn, kappa=klev[kl], coverage=fin["coverage"], bare_beta=bare,
                               anchor=an, **kw)
        nr = [10.0 * math.log10(bare_ms[kl][float(f)] / t["ms_by_freq"][float(f)]) for f in band]
        d = float(O2.delta_oaspl(np.asarray(nr), spl))
        if not math.isfinite(d):
            raise VehiclePipelineError(f"non-finite dOASPL at corner ({kl}, {an})")
        corners.append(dict(corner_id=(kl, an), value=d, ok=True, kappa=klev[kl], anchor=an, nr_band=nr,
                            mesh_sha256=t["mesh_sha256"], achieved_coverage=t["achieved_coverage"],
                            consistency=t["consistency"]))
    robust = RP.select_robust_minimum(corners, DRUN.PREREGISTERED_CORNERS)
    rec = dict(study="vehicle_initial_fem" if stage == "fem" else "vehicle_corrected_fem", mesh_size=mesh_size,
               coverage=fin["coverage"], design_sha256=fin["design_sha256"], geometry_hash=fin["geometry_hash"],
               payload=vs.payload_cases[payload_case], corners=corners, robust=robust,
               nominal_mean_dOASPL=float(np.mean([c["value"] for c in corners])), a_other_note=A_OTHER_NOTE,
               wall_impedance="cfd_corrected" if wall_beta_fn else "linear_oracle_z_surface")
    return V.stamp(rec, vs.identity(corner_id, payload_case), stage, parent)


def _fem_worker(args):
    vehicle, corner, case, h = args
    vs = scenario(vehicle)
    return four_corner_fem(vs, load_finalist(vs), corner, case, h)


def fem(vehicle, workers):
    vs = scenario(vehicle)
    fin = load_finalist(vs)
    out = RUN_DIR / vs.name / "initial_fem.json"
    if out.exists():
        raise VehiclePipelineError(f"{out} exists; initial FEM runs once (placement/coverage are frozen)")
    jobs = [(vehicle, c["id"], case, h) for c in vs.corners.values() for case in vs.payload_cases for h in MESH_SIZES]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        recs = list(ex.map(_fem_worker, jobs))
    for r in recs:
        V.verify_lineage(vs, [fin, r])
    conv = {}
    for r in recs:
        conv.setdefault(f"{r['source_corner']}|{r['payload_case']}", {})[r["mesh_size"]] = r["robust"]["robust_value"]
    unc = {k: dict(by_mesh=v, abs_0p2_vs_0p1=abs(v[0.2] - v[0.1]), abs_0p15_vs_0p1=abs(v[0.15] - v[0.1]))
           for k, v in conv.items()}
    doc = dict(vehicle=vehicle, scenario_hash=vs.scenario_hash, finalist_sha256=V.record_sha256(fin),
               finalist_status=fin["status"], mesh_sizes=list(MESH_SIZES), records=recs, convergence=unc,
               authoritative=f"{vs.corners['primary']['id']}|{vs.payload_primary} at mesh 0.1 m",
               a_other_note=A_OTHER_NOTE)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=float))
    os.replace(tmp, out)
    for k, v in unc.items():
        print(f"{k}: " + "  ".join(f"h={h}: {x:+.4f}" for h, x in v["by_mesh"].items())
              + f"  |0.2-0.1| {v['abs_0p2_vs_0p1']:.4f} dB", flush=True)
    return doc


def main(argv=None):
    p = argparse.ArgumentParser(prog="acoustic-liner vehicles search")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("search", "fem"):
        s = sub.add_parser(name)
        s.add_argument("--vehicle", choices=sorted(APPROVED_SCENARIO_HASHES), required=True)
        s.add_argument("--workers", type=int, default=5)
    sa = sub.add_parser("seal-audit")
    sa.add_argument("--run-commit", required=True)
    sub.add_parser("check-audit")
    a = p.parse_args(argv)
    if a.cmd == "seal-audit":
        print(json.dumps(seal_audit(a.run_commit), indent=1))
    elif a.cmd == "check-audit":
        print(json.dumps(require_phase1()["summary"], indent=1))
    else:
        (search if a.cmd == "search" else fem)(a.vehicle, a.workers)


if __name__ == "__main__":
    main()
