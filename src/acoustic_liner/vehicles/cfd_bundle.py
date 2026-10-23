# -*- coding: utf-8 -*-
"""Portable two-machine CFD bundles for the vehicle finalists (Phase 4 of the vehicle FEM/CFD plan, a historical
development document that is not distributed).

A bundle is generated ONLY from real parent records: the frozen vehicle search finalist and its initial
payload-resolving FEM (`vehicles.search`). Placeholder lineage is refused. The neck CFD is local (no payload mesh);
the payload identity is still threaded through every record.

    acoustic-liner vehicles cfd-bundle bundle    --vehicle falcon [--sensitivity]   (Windows; writes artifacts/dev/vehicle_v1/<name>/cfd_bundle)
    bash run_cases.sh [nominal|sens_refine|sens_halfdt]                          (WSL, inside the copied bundle; resumable)
    acoustic-liner vehicles cfd-bundle status    --bundle DIR                        (any time; reads only)
    acoustic-liner vehicles cfd-bundle returned  --bundle DIR                        (hash every returned file -> _returned_manifest.json)
    acoustic-liner vehicles cfd-bundle aggregate --bundle DIR                        (valid rows -> per-bank R/X -> cfd_result record)
    acoustic-liner vehicles cfd-bundle corrected-fem --bundle DIR                    (CFD-impedance four-corner FEM, 0.2/0.15/0.1 m)

Velocity bounds come from the MORE DEMANDING registered source condition (higher 50-250 Hz OASPL); both source
identities are kept in the bundle manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from .. import paths
from ..provenance import schemas as SC
from . import search as VP
from . import scenarios as V
from ..cfd import case_generation as CG
from ..cfd import runner as RUN

BUNDLE_SCHEMA = SC.schema("vehicle-cfd-bundle")


def _refused_schema(value):
    """True unless `value` is the public bundle schema or its registered historical identifier."""
    try:
        return SC.accept_schema(value, "vehicle-cfd-bundle") != BUNDLE_SCHEMA
    except SC.SchemaError:
        return True
SENS_VARIANTS = ({"label": "refine", "refine": 2.0}, {"label": "halfdt", "maxco": 0.25})
OPENFOAM_CMD = "blockMesh > log.blockMesh 2>&1 && checkMesh > log.checkMesh 2>&1 && pimpleFoam > log.pimpleFoam 2>&1"


class BundleError(RuntimeError):
    pass


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float).encode()).hexdigest()


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _git(*a):
    return subprocess.run(["git", *a], cwd=paths.ROOT, capture_output=True, text=True).stdout.strip()


def preflight(allow_dirty=False):
    """Clean-tree + dependency preflight. Source / config changes since the commit would make the bundle unbound."""
    dirty = [l for l in _git("status", "--porcelain", "--", "src", "configs").splitlines() if l.strip()]
    if dirty and not allow_dirty:
        raise BundleError(f"uncommitted source/config changes: {dirty[:5]}")
    lock = paths.ROOT / "requirements.lock"
    return dict(commit=_git("rev-parse", "HEAD"), tag=_git("describe", "--tags", "--exact-match") or None,
                dirty=bool(dirty), requirements_lock_sha256=_sha_file(lock) if lock.exists() else None)


def demanding_corner(vs):
    lv = {c["id"]: V.oaspl(c["model_levels"]) for c in vs.corners.values()}
    return max(lv, key=lambda k: (lv[k], k)), lv


def _parent_fem(vs, fin):
    """The real initial FEM record that parents CFD: primary corner, primary payload case, finest mesh."""
    p = VP.RUN_DIR / vs.name / "initial_fem.json"
    if not p.exists():
        raise BundleError(f"{p} missing: generate CFD cases only after the initial payload-resolving FEM")
    doc = json.load(open(p))
    if doc.get("finalist_sha256") != V.record_sha256(fin):
        raise BundleError("initial FEM was not run on this finalist")
    want = (vs.corners["primary"]["id"], vs.payload_primary, min(VP.MESH_SIZES))
    for r in doc["records"]:
        if (r["source_corner"], r["payload_case"], r["mesh_size"]) == want:
            V.verify_lineage(vs, [fin, r])
            return r
    raise BundleError(f"no initial FEM record for {want}")


def _write_scripts(root, groups):
    one = ["#!/usr/bin/env bash", "# One case, resumable + atomic status: skipped when _status.json says done.",
           "set -u", 'c="${1%/}"', '[ -f "$c/_case_meta.json" ] || exit 0',
           'if grep -q \'"state": "done"\' "$c/_status.json" 2>/dev/null; then echo "skip $c"; exit 0; fi',
           'echo \'{"state": "running"}\' > "$c/_status.tmp" && mv "$c/_status.tmp" "$c/_status.json"',
           f'( cd "$c" && {OPENFOAM_CMD} ); rc=$?',
           'if [ $rc -eq 0 ]; then s=done; else s=failed; fi',
           'printf \'{"state": "%s", "rc": %d, "finished": "%s"}\\n\' "$s" "$rc" "$(date -Is)" > "$c/_status.tmp" '
           '&& mv "$c/_status.tmp" "$c/_status.json"', 'echo "$s $c (rc $rc)"']
    (root / "run_one.sh").write_text("\n".join(one) + "\n", newline="\n")
    sh = ["#!/usr/bin/env bash", "# bash run_cases.sh <group> [parallel jobs, default 1]   e.g. bash run_cases.sh nominal 4",
          'cd "$(dirname "$0")"', 'GROUP="${1:-nominal}"; N="${2:-1}"',
          'ls -d "$GROUP"/*/ | xargs -P "$N" -n 1 bash run_one.sh', 'echo "group $GROUP finished; rerun to resume"']
    (root / "run_cases.sh").write_text("\n".join(sh) + "\n", newline="\n")


def bundle(vehicle, sensitivity=False, allow_dirty=False, out=None, cycles=12):
    pre = preflight(allow_dirty)
    vs = VP.scenario(vehicle)
    fin = VP.load_finalist(vs)
    if fin["status"] != "finalist":
        print(f"WARNING: {vs.name} finalist status is {fin['status']!r}; CFD documents it but it is not a target pass")
    fem = _parent_fem(vs, fin)
    root = Path(out) if out else VP.RUN_DIR / vs.name / "cfd_bundle"
    if (root / "_bundle.json").exists():
        raise BundleError(f"{root} already holds a bundle; bundles are immutable evidence")
    root.mkdir(parents=True, exist_ok=True)
    corner, lv = demanding_corner(vs)
    spectrum = vs.model_levels(corner)
    fin_cfd = root / "cfd_finalist.json"
    fin_cfd.write_text(json.dumps(dict(geometry=fin["design"], coverage=fin["coverage"], scenario=vs.name,
                                       finalist_sha256=V.record_sha256(fin)), sort_keys=True, indent=1, default=float))
    groups = {"nominal": CG.gen_cases(str(root / "nominal"), scenario=vs.name, source_corner=corner,
                                      finalist_path=str(fin_cfd), spectrum=spectrum, cycles=cycles)}
    if sensitivity:
        for v in SENS_VARIANTS:
            groups[f"sens_{v['label']}"] = CG.gen_cases(
                str(root / f"sens_{v['label']}"), scenario=vs.name, source_corner=corner, finalist_path=str(fin_cfd),
                spectrum=spectrum, cycles=cycles, refine=float(v.get("refine", 1.0)), maxco=float(v.get("maxco", 0.5)))
    _write_scripts(root, groups)
    ident = vs.identity(vs.corners["primary"]["id"], vs.payload_primary)
    cases = {g: [dict(name=c["name"], bank_id=c["bank_id"], u_target=c["u_target"], case_hash=c["case_hash"],
                      input_tree_hash=c["input_tree_hash"]) for c in p["cases"]] for g, p in groups.items()}
    rec = dict(schema=BUNDLE_SCHEMA, vehicle=vehicle, preflight=pre, source_corners=[c["id"] for c in vs.corners.values()], finalist_sha256=V.record_sha256(fin),
               finalist_status=fin["status"], design_sha256=fin["design_sha256"], coverage=fin["coverage"],
               velocity_source_corner=corner, source_inband_oaspl_dB=lv, both_source_corners=list(lv),
               payload_cases=V.finalist_coverage(vs)["payload_cases"], geometry_hash=groups["nominal"]["geometry_hash"],
               spectrum=spectrum, cycles=cycles, groups={g: dict(n_cases=p["n_cases"], plan_sha256=_sha_file(
                   root / g / "_cfd_case_plan.json")) for g, p in groups.items()},
               expected_cases=cases, n_expected=sum(len(v) for v in cases.values()),
               commands=dict(wsl_nominal="bash run_cases.sh nominal 4",
                             wsl_sensitivity=[f"bash run_cases.sh {g} 4" for g in groups if g != "nominal"],
                             status=f"acoustic-liner vehicles cfd-bundle status --bundle {root}",
                             returned=f"acoustic-liner vehicles cfd-bundle returned --bundle {root}",
                             aggregate=f"acoustic-liner vehicles cfd-bundle aggregate --bundle {root}",
                             corrected_fem=f"acoustic-liner vehicles cfd-bundle corrected-fem --bundle {root}"))
    rec = V.stamp(rec, ident, "cfd_case", fem)
    (root / "_bundle.json").write_text(json.dumps(rec, sort_keys=True, indent=1, default=float))
    (root / "_parent_fem.json").write_text(json.dumps(fem, sort_keys=True, indent=1, default=float))
    return rec


def load_bundle(bdir):
    bdir = Path(bdir)
    rec = json.load(open(bdir / "_bundle.json"))
    if _refused_schema(rec.get("schema")):
        raise BundleError("not a vehicle CFD bundle")
    vs = VP.scenario(rec["vehicle"])
    fin = VP.load_finalist(vs)
    fem = json.load(open(bdir / "_parent_fem.json"))
    V.verify_lineage(vs, [fin, fem, rec])
    for g, info in rec["groups"].items():
        if _sha_file(bdir / g / "_cfd_case_plan.json") != info["plan_sha256"]:
            raise BundleError(f"{g} case plan was modified")
    return vs, fin, fem, rec


def status(bdir):
    vs, fin, fem, rec = load_bundle(bdir)
    out = {}
    for g in rec["groups"]:
        man = RUN.load_manifest(str(Path(bdir) / g))
        out[g] = RUN.scan_status(man, str(Path(bdir) / g))
        out[g]["resume"] = RUN.resume_manifest(out[g], str(Path(bdir) / g))
        print(f"{g}: {out[g]['summary']}  all_valid={out[g]['all_valid']}")
    return out


def _returned_groups(rec, groups):
    groups = list(rec["groups"]) if groups is None else list(groups)
    if "nominal" not in groups or not set(groups) <= set(rec["groups"]):
        raise BundleError(f"returned groups must include 'nominal' and be a subset of {sorted(rec['groups'])}")
    return [g for g in rec["groups"] if g in groups]


def returned_manifest(bdir, groups=None):
    """Hash every returned case file (0/, constant/, system/, logs, postProcessing/, time dirs, metadata) of the
    returned groups (default: all; `nominal` alone is enough for the corrected FEM)."""
    vs, fin, fem, rec = load_bundle(bdir)
    groups = _returned_groups(rec, groups)
    files = {}
    for g, cases in rec["expected_cases"].items():
        if g not in groups:
            continue
        for c in cases:
            cd = Path(bdir) / g / c["name"]
            if not cd.is_dir():
                raise BundleError(f"expected case {g}/{c['name']} missing")
            for p in sorted(cd.rglob("*")):
                if p.is_file():
                    files[p.relative_to(bdir).as_posix()] = _sha_file(p)
    doc = dict(bundle_sha256=V.record_sha256(rec), groups=groups, n_files=len(files), files=files)
    tmp = Path(bdir) / "_returned_manifest.tmp"
    tmp.write_text(json.dumps(doc, sort_keys=True, indent=1))
    os.replace(tmp, Path(bdir) / "_returned_manifest.json")
    return doc


def aggregate(bdir):
    vs, fin, fem, rec = load_bundle(bdir)
    rm = json.load(open(Path(bdir) / "_returned_manifest.json"))
    if rm["bundle_sha256"] != V.record_sha256(rec):
        raise BundleError("returned-results manifest belongs to another bundle")
    for rel, h in rm["files"].items():
        if _sha_file(Path(bdir) / rel) != h:
            raise BundleError(f"returned file changed after its manifest: {rel}")
    returned = _returned_groups(rec, rm.get("groups"))
    res = {}
    for g in returned:
        man = RUN.load_manifest(str(Path(bdir) / g))
        a = RUN.aggregate(man, str(Path(bdir) / g))
        res[g] = dict(per_bank_RX=a["per_bank_RX"], per_bank_anchor=a["per_bank_anchor"], n_banks=a["n_banks"])
    not_run = [g for g in rec["groups"] if g not in returned]
    out = dict(study="vehicle_cfd_result", groups=res, groups_not_run=not_run,
               cfd_numerical_uncertainty=("not quantified: sensitivity groups not run" if not_run else "see sens_* groups"),
               source_corners=rec["source_corners"],
               payload_cases=rec["payload_cases"], returned_manifest_sha256=_sha_file(
        Path(bdir) / "_returned_manifest.json"))
    out = V.stamp(out, {k: rec[k] for k in V.IDENTITY_KEYS}, "cfd_result", rec)
    (Path(bdir) / "_cfd_result.json").write_text(json.dumps(out, sort_keys=True, indent=1, default=float))
    return out


def corrected_fem(bdir):
    """Relabel (both forcings, fixed point) with the nominal R/X, then the CFD-impedance four-corner FEM on the
    payload-resolving mesh at 0.2 / 0.15 / 0.1 m for every source corner x payload case."""
    import numpy as np
    from ..physics import oracle as O2
    from ..design import canonical as CAN
    from ..cfd import feedback as FB
    from ..cfd import impedance_correction as RL
    vs, fin, fem, rec = load_bundle(bdir)
    cr = json.load(open(Path(bdir) / "_cfd_result.json"))
    V.verify_lineage(vs, [fin, fem, rec, cr])
    design = dict(CAN.canonical_design(fin["design"]))
    design["groups"] = [dict(g, cavity_volume=O2._group_cavity_volume(g)) for g in design["groups"]]
    nom = cr["groups"]["nominal"]
    ps = vs.to_physical_scenario(rec["velocity_source_corner"])
    cond = dict(spl_dB=list(ps.spl_dB), f_cov=float(fin["coverage"]), A_wall=float(ps.A_wall),
                kappa_nominal=float(ps.kappa_nominal), dOASPL_target=VP.TARGET_DB)
    interval = FB.relabel_design_interval(design, cond, nom["per_bank_RX"], nom["per_bank_anchor"])
    out = dict(relabel=interval, records={})
    for key in ("broadband", "resonant_tone"):
        leg = interval[key]
        if leg["status"] != "complete":
            out["records"][key] = dict(status="incomplete", banks_not_ok=leg.get("banks_not_ok"))
            continue
        bfn = RL._nl_beta_fn(design["groups"], design["D"], np.array(leg["dR_excess_Rayl"], float),
                             np.array(leg["dM_inertance_Rayl_s"], float))
        recs = []
        for c in vs.corners.values():
            for case in vs.payload_cases:
                for h in VP.MESH_SIZES:
                    recs.append(VP.four_corner_fem(vs, fin, c["id"], case, h, wall_beta_fn=bfn, stage="corrected_fem",
                                                   parent=cr))
        out["records"][key] = recs
    (Path(bdir) / "_corrected_fem.json").write_text(json.dumps(out, sort_keys=True, indent=1, default=float))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="acoustic-liner vehicles cfd-bundle")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--vehicle", choices=sorted(VP.APPROVED_SCENARIO_HASHES), required=True)
    b.add_argument("--sensitivity", action="store_true")
    b.add_argument("--cycles", type=int, default=12)
    for n in ("status", "returned", "aggregate", "corrected-fem"):
        s = sub.add_parser(n)
        s.add_argument("--bundle", required=True)
        if n == "returned":
            s.add_argument("--groups", nargs="+", default=None, help="returned groups (default all; e.g. nominal)")
    a = p.parse_args(argv)
    if a.cmd == "returned":
        returned_manifest(a.bundle, a.groups)
        return
    if a.cmd == "bundle":
        r = bundle(a.vehicle, a.sensitivity, cycles=a.cycles)
        print(f"{r['vehicle']}: {r['n_expected']} cases, velocity corner {r['velocity_source_corner']}, "
              f"bundle identity {r['scenario_hash'][:12]} / payload {r['payload_geometry_sha256'][:12]}")
    else:
        {"status": status, "returned": returned_manifest, "aggregate": aggregate,
         "corrected-fem": corrected_fem}[a.cmd](a.bundle)


if __name__ == "__main__":
    main()
