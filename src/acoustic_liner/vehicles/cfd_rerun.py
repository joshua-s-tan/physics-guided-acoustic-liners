# -*- coding: utf-8 -*-
"""Operating-point reruns for a returned vehicle CFD bundle: fresh longer-cycle cases only where the nominal sweep
lacks valid support around the self-consistent operating point U*. The original returned evidence is never edited.

    acoustic-liner vehicles cfd-bundle returned --bundle DIR --groups nominal        (hash the returned nominal results)
    acoustic-liner vehicles cfd-rerun select --bundle DIR [--cycles 24]        (bracket table -> rerun bundle)
    bash run_cases.sh nominal_24c [N]                                            (WSL, inside the rerun bundle)
    acoustic-liner vehicles cfd-rerun merge  --bundle DIR --rerun RDIR         (gates unchanged -> _cfd_result.json)

Support rule (fixed before any rerun result): with U* taken over both forcing bounds as [U_lo, U_hi], the nearest
planned case at or below U_lo, the nearest at or above U_hi, and every planned case inside (U_lo, U_hi) must be
valid. A non-valid case is not needed when a valid case on the same side lies within REDUNDANT_TOL of its velocity.
Every other non-valid case is a documented exclusion. A needed case that is still not valid after its rerun leaves
the bank unresolved: merge stops before targeted sensitivity or corrected FEM.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..physics import oracle as O2
from ..design import canonical as CAN
from . import cfd_bundle as VC
from . import search as VP
from . import scenarios as V
from ..cfd import case_generation as CG
from ..cfd import feedback as FB
from ..cfd import runner as RUN

GROUP = "nominal_24c"
REDUNDANT_TOL = 0.05
SUPPORT_RULE = __doc__.split("Support rule ")[1].strip()


class RerunError(RuntimeError):
    pass


def bracket(planned, u_lo, u_hi):
    """planned: [(name, u_target, state)] non-reference cases of one bank. Returns the support table for one bank."""
    rows = sorted(planned, key=lambda r: (r[1], r[0]))
    valid = [r for r in rows if r[2] == RUN.VALID]
    below = [r for r in rows if r[1] <= u_lo]
    above = [r for r in rows if r[1] >= u_hi]
    inside = [r for r in rows if u_lo < r[1] < u_hi]
    nb, na = (below[-1] if below else None), (above[0] if above else None)
    vb = [r for r in valid if r[1] <= u_lo]
    va = [r for r in valid if r[1] >= u_hi]
    needed, redundant = [], []
    for r, side in [(nb, vb), (na, va)] + [(r, valid) for r in inside]:
        if r is None or r[2] == RUN.VALID:
            continue
        near = [v for v in side if abs(v[1] - r[1]) <= REDUNDANT_TOL * r[1]]
        (redundant if near else needed).append(dict(name=r[0], u_target=r[1], state=r[2],
                                                     covered_by=(near[0][0] if near else None)))
    missing_side = [s for s, v in (("below", vb), ("above", va)) if not v]
    return dict(U_star_lo=u_lo, U_star_hi=u_hi, valid=[r[1] for r in valid],
                valid_below=(vb[-1][1] if vb else None), valid_above=(va[0][1] if va else None),
                nearest_planned_below=nb, nearest_planned_above=na, inside=inside, needed=needed,
                redundant=redundant, missing_side=missing_side,
                status=("adequate" if not needed and not missing_side else "needs_rerun"))


def _design_cond(vs, fin, rec):
    design = dict(CAN.canonical_design(fin["design"]))
    design["groups"] = [dict(g, cavity_volume=O2._group_cavity_volume(g)) for g in design["groups"]]
    ps = vs.to_physical_scenario(rec["velocity_source_corner"])
    cond = dict(spl_dB=list(ps.spl_dB), f_cov=float(fin["coverage"]), A_wall=float(ps.A_wall),
                kappa_nominal=float(ps.kappa_nominal), dOASPL_target=VP.TARGET_DB)
    return design, cond


def _verify_returned(bdir, rec):
    rm_p = Path(bdir) / "_returned_manifest.json"
    if not rm_p.exists():
        raise RerunError(f"{rm_p} missing: run `acoustic-liner vehicles cfd-bundle returned --bundle <bundle> --groups nominal` first")
    rm = json.load(open(rm_p))
    if rm["bundle_sha256"] != V.record_sha256(rec):
        raise RerunError("returned manifest belongs to another bundle")
    for rel, h in rm["files"].items():
        if VC._sha_file(Path(bdir) / rel) != h:
            raise RerunError(f"returned file changed after its manifest: {rel}")
    return VC._sha_file(rm_p)


def table(manifest, base, design, cond):
    """Status + valid-row relabel + per-bank support table for a (possibly merged) nominal manifest."""
    st = RUN.scan_status(manifest, base)
    states = {c["name"]: c for c in st["cases"]}
    a = RUN.aggregate(manifest, base)
    iv = FB.relabel_design_interval(design, cond, a["per_bank_RX"], a["per_bank_anchor"])
    banks = {}
    for j in range(a["n_banks"]):
        us = [iv[k]["per_bank"][j].get("U_star") for k in ("broadband", "resonant_tone")]
        if any(u is None for u in us):
            banks[f"b{j}"] = dict(status="unresolved", detail=[iv[k]["per_bank"][j] for k in ("broadband", "resonant_tone")])
            continue
        planned = [(c["name"], float(c["u_target"]), states[c["name"]]["state"]) for c in manifest["cases"]
                   if c["bank_id"] == f"b{j}" and abs(float(c["u_target"]) - 0.5) > 1e-9]
        banks[f"b{j}"] = bracket(planned, min(us), max(us))
    return dict(status=st, aggregate=a, relabel=iv, banks=banks)


def select(bdir, cycles=24, out=None, allow_dirty=False):
    bdir = Path(bdir)
    pre = VC.preflight(allow_dirty)
    vs, fin, fem, rec = VC.load_bundle(bdir)
    rm_sha = _verify_returned(bdir, rec)
    design, cond = _design_cond(vs, fin, rec)
    man = RUN.load_manifest(str(bdir / "nominal"))
    t = table(man, str(bdir / "nominal"), design, cond)
    if any(b["status"] == "unresolved" for b in t["banks"].values()):
        raise RerunError(f"U* not resolvable from the valid rows: {t['banks']}")
    needed = [n for b in t["banks"].values() for n in b["needed"]]
    rerun_names = {n["name"] for n in needed}
    exclusions = []
    for c in t["status"]["cases"]:
        if c["state"] == RUN.VALID or c["name"] in rerun_names:
            continue
        red = [r for b in t["banks"].values() for r in b["redundant"] if r["name"] == c["name"]]
        exclusions.append(dict(name=c["name"], state=c["state"], failed=c.get("failed"),
                               reason=(f"redundant: valid {red[0]['covered_by']} within {REDUNDANT_TOL:.0%}" if red
                                       else "outside every bank's operating-point bracket")))
    root = Path(out) if out else VP.RUN_DIR / vs.name / f"cfd_rerun_{cycles}c"
    if (root / "_rerun.json").exists():
        raise RerunError(f"{root} already holds a rerun bundle; it is immutable evidence")
    entries = {c["name"]: c for c in man["cases"]}
    made = []
    for nm in sorted(rerun_names):
        meta = json.load(open(bdir / "nominal" / nm / "_case_meta.json"))
        _, m = CG.generate(str(root / GROUP / nm), bank_id=meta["bank_id"], d_mm=meta["d_mm"], L_mm=meta["L_mm"],
                           f0=meta["f0_Hz"], uneck=meta["uneck_m_s"], cycles=cycles,
                           geometry_hash=meta["geometry_hash"], refine=meta["refine"], maxco=meta["maxco"])
        if m["case_hash"] != entries[nm]["case_hash"]:
            raise RerunError(f"{nm}: rerun case identity differs from its parent")
        made.append(dict(name=nm, bank_id=m["bank_id"], u_target=m["uneck_m_s"], case_hash=m["case_hash"],
                         input_tree_hash=m["input_tree_hash"], cycles=cycles,
                         parent_input_tree_hash=entries[nm]["input_tree_hash"],
                         parent_status_sha256=VC._sha_file(bdir / "nominal" / nm / "_status.json")))
    plan = dict(geometry_hash=man["geometry_hash"], cycles=cycles, n_cases=len(made), cases=made)
    (root / GROUP).mkdir(parents=True, exist_ok=True)
    (root / GROUP / "_cfd_case_plan.json").write_text(json.dumps(plan, indent=2, default=float))
    VC._write_scripts(root, {GROUP: plan})
    doc = dict(study="vehicle_cfd_operating_point_rerun", vehicle=rec["vehicle"], preflight=pre, support_rule=SUPPORT_RULE,
               parent_bundle_sha256=V.record_sha256(rec), parent_returned_manifest_sha256=rm_sha, cycles=cycles,
               provisional_interval=t["relabel"]["dOASPL_nonlinear_interval"], banks=t["banks"], rerun=made,
               exclusions=exclusions, plan_sha256=VC._sha_file(root / GROUP / "_cfd_case_plan.json"),
               commands=dict(wsl=f"bash run_cases.sh {GROUP} {max(1, len(made))}",
                             merge=f"acoustic-liner vehicles cfd-rerun merge --bundle {bdir} --rerun {root}"))
    (root / "_rerun.json").write_text(json.dumps(doc, sort_keys=True, indent=1, default=float))
    print(f"{vs.name}: rerun {[m['name'] for m in made]} at {cycles} cycles -> {root}; "
          f"{len(exclusions)} documented exclusions")
    return doc


def merged_manifest(bdir, rdir, doc):
    """The bundle's nominal manifest with each rerun case pointing at its rerun tree and its fresh input hash."""
    man = RUN.load_manifest(str(Path(bdir) / "nominal"))
    sub = {c["name"]: c for c in doc["rerun"]}
    return dict(man, cases=[dict(e, name=str((Path(rdir) / GROUP / e["name"]).resolve()),
                                 input_tree_hash=sub[e["name"]]["input_tree_hash"]) if e["name"] in sub else e
                            for e in man["cases"]])


def merge(bdir, rdir):
    """Substitute the returned rerun cases into the nominal manifest, apply the unchanged gates, and write the
    bundle's _cfd_result.json only when every bank has adequate support around U*."""
    bdir, rdir = Path(bdir), Path(rdir)
    vs, fin, fem, rec = VC.load_bundle(bdir)
    rm_sha = _verify_returned(bdir, rec)
    doc = json.load(open(rdir / "_rerun.json"))
    if doc["parent_bundle_sha256"] != V.record_sha256(rec) or doc["parent_returned_manifest_sha256"] != rm_sha:
        raise RerunError("rerun bundle belongs to another parent bundle / returned manifest")
    if VC._sha_file(rdir / GROUP / "_cfd_case_plan.json") != doc["plan_sha256"]:
        raise RerunError("rerun plan was modified")
    files = {}
    for c in doc["rerun"]:
        cd = rdir / GROUP / c["name"]
        if not cd.is_dir():
            raise RerunError(f"rerun case {c['name']} missing")
        for p in sorted(cd.rglob("*")):
            if p.is_file():
                files[p.relative_to(rdir).as_posix()] = VC._sha_file(p)
    rr_p = rdir / "_returned_manifest.json"
    if rr_p.exists():
        prev = json.load(open(rr_p))
        if prev["files"] != files:
            raise RerunError("rerun results changed after their first merge")
    else:
        rr_p.write_text(json.dumps(dict(rerun_sha256=VC._sha_file(rdir / "_rerun.json"), files=files),
                                   sort_keys=True, indent=1))
    sub = {c["name"]: c for c in doc["rerun"]}
    merged = merged_manifest(bdir, rdir, doc)
    design, cond = _design_cond(vs, fin, rec)
    t = table(merged, str(bdir / "nominal"), design, cond)
    unresolved = {b: v for b, v in t["banks"].items() if v["status"] != "adequate"}
    rerun_states = {Path(c["name"]).name: c["state"] for c in t["status"]["cases"] if Path(c["name"]).name in sub}
    summary = dict(rerun_states=rerun_states, banks=t["banks"], interval=t["relabel"]["dOASPL_nonlinear_interval"])
    (rdir / "_merge_report.json").write_text(json.dumps(summary, sort_keys=True, indent=1, default=float))
    if unresolved:
        print(f"{vs.name}: UNRESOLVED banks {sorted(unresolved)}; stop before targeted sensitivity / corrected FEM")
        raise RerunError(f"banks without adequate support around U*: {sorted(unresolved)}")
    out = dict(study="vehicle_cfd_result", groups=dict(nominal=dict(
                   per_bank_RX=t["aggregate"]["per_bank_RX"], per_bank_anchor=t["aggregate"]["per_bank_anchor"],
                   n_banks=t["aggregate"]["n_banks"])),
               groups_not_run=[g for g in rec["groups"] if g != "nominal"],
               cfd_numerical_uncertainty="targeted mesh/time-step sensitivity (docs/VEHICLE-CFD-TARGETED-SENSITIVITY-PREREG.md)",
               operating_point_rerun=dict(rerun_sha256=VC._sha_file(rdir / "_rerun.json"),
                                          rerun_returned_sha256=VC._sha_file(rr_p), cases=rerun_states,
                                          exclusions=doc["exclusions"], support=t["banks"],
                                          rerun_dir=os.path.relpath(rdir, bdir)),
               source_corners=rec["source_corners"], payload_cases=rec["payload_cases"],
               returned_manifest_sha256=rm_sha)
    out = V.stamp(out, {k: rec[k] for k in V.IDENTITY_KEYS}, "cfd_result", rec)
    (bdir / "_cfd_result.json").write_text(json.dumps(out, sort_keys=True, indent=1, default=float))
    print(f"{vs.name}: all banks supported; reruns {rerun_states}; interval {t['relabel']['dOASPL_nonlinear_interval']}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="acoustic-liner vehicles cfd-rerun")
    p.add_argument("cmd", choices=("select", "merge"))
    p.add_argument("--bundle", required=True)
    p.add_argument("--rerun")
    p.add_argument("--cycles", type=int, default=24)
    a = p.parse_args(argv)
    if a.cmd == "select":
        select(a.bundle, a.cycles)
    else:
        merge(a.bundle, a.rerun)


if __name__ == "__main__":
    main()
