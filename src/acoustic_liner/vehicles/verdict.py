# -*- coding: utf-8 -*-
"""Final simulation-based vehicle verdict.

    acoustic-liner vehicles verdict --vehicle falcon --results DIR --targeted TDIR

Collects and verifies the full evidence chain, applies the preregistered conservative interval
L = F01 - M - S_down, U = F01_hi + M + S_up against T = 0.9213 dB, and publishes verdict.json plus VERDICT.md
(rendered only from the JSON) under artifacts/dev/vehicle_v1/<scenario>/verdict/. Any verification failure is
recorded as a reason and makes the outcome UNRESOLVED; incomplete evidence never becomes a PASS.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import stat
from pathlib import Path

from ..provenance import schemas as SC
from . import cfd_bundle as VC
from . import cfd_rerun as VR
from . import cfd_targeted as VT
from . import search as VP
from . import scenarios as V
from ..cfd import runner as RUN

SCHEMA = SC.schema("vehicle-verdict")      # version 2: no pre-registration file or hash (version 1 named both)
FORCINGS = ("broadband", "resonant_tone")
PASS, FAIL, UNRESOLVED = "PASS", "FAIL", "UNRESOLVED"
WORDING = dict(
    basis="simulation-based (oracle search, payload-resolving FEM, OpenFOAM neck CFD); not experimentally validated "
          "and not flight-qualified",
    a_other="A_other is an assumed effective background-loss term that may include payload-associated losses; it is "
            "not measured payload absorption",
    audit="FEM-transfer audit limitation: transfer error is larger and less predictable for some partial-coverage "
          "designs",
    cfd="targeted operating-region mesh/time-step sensitivity: numerical sensitivity in the operating region only, "
        "not formal grid convergence and not full-range CFD convergence; " + VT.ENVELOPE_WORDING,
    cfd_limits="; ".join(VT.LIMITATIONS).capitalize())


class VerdictError(RuntimeError):
    pass


# ---- pure rule (unit-tested) -----------------------------------------------------------------------------------
def interval(fem, s_down, s_up):
    """fem: {forcing: {mesh: value}} for the two forcings x three meshes. Returns the sec. 2 interval or None when
    any value is missing."""
    try:
        vals = {f: {h: float(fem[f][h]) for h in VP.MESH_SIZES} for f in FORCINGS}
    except (KeyError, TypeError, ValueError):
        return None
    fine = min(VP.MESH_SIZES)
    f01 = min(v[fine] for v in vals.values())
    f01_hi = max(v[fine] for v in vals.values())
    m = max(abs(v[h] - v[fine]) for v in vals.values() for h in VP.MESH_SIZES if h != fine)
    return dict(F01=f01, F01_hi=f01_hi, M=m, S_down=s_down, S_up=s_up, L=f01 - m - s_down, U=f01_hi + m + s_up)


def sensitivity_allowance(targeted):
    vals = [targeted["fem_nominal_0p1"]] + [c["fem_robust_db"] for c in targeted["fem_envelope_combos_0p1"]
                                            if c["fem_robust_db"] is not None]
    f = targeted["fem_nominal_0p1"]
    return max(0.0, f - min(vals)), max(0.0, max(vals) - f)


def decide(iv, reasons, target=VP.TARGET_DB):
    """Outcome per sec. 3. `reasons`: every completeness / verification failure found (empty when complete)."""
    if reasons or iv is None:
        return UNRESOLVED
    if iv["L"] > target:
        return PASS
    if iv["U"] < target:
        return FAIL
    return UNRESOLVED


# ---- evidence collection (fail closed: failures become reasons) ----------------------------------------------
def _sha_file(p):
    return VC._sha_file(p)


def _openfoam_build(case_dir):
    p = Path(case_dir) / "log.pimpleFoam"
    if not p.exists():
        return None
    with open(p, errors="replace") as f:
        for _ in range(40):
            m = re.match(r"Build\s*:\s*(.+)", f.readline().strip())
            if m:
                return m.group(1).strip()
    return None


def _fem_tables(vs, records):
    """{(corner, payload): {forcing: {mesh: robust value}}} from the corrected-FEM file."""
    out = {}
    for f in FORCINGS:
        recs = records.get(f)
        if not isinstance(recs, list):
            continue
        for r in recs:
            out.setdefault((r["source_corner"], r["payload_case"]), {}).setdefault(f, {})[float(r["mesh_size"])] = \
                r["robust"]["robust_value"]
    return out


def collect(vehicle, results, targeted):
    """Gather and verify all evidence. Returns (evidence, reasons)."""
    reasons, ev = [], dict(vehicle=vehicle)

    def check(cond, msg):
        if not cond:
            reasons.append(msg)
        return cond

    results, targeted = Path(results), Path(targeted)
    try:
        vs = VP.scenario(vehicle)
        fin = VP.load_finalist(vs)
        vs2, fin2, fem0, rec = VC.load_bundle(results)
    except Exception as e:                                   # noqa: BLE001 - any failure is a reason, never a pass
        return ev, [f"bundle / finalist / scenario failed verification: {e}"]
    check(vs2.name == vs.name and rec.get("vehicle") == vehicle, f"bundle belongs to {rec.get('vehicle')!r}, not {vehicle!r}")
    check(VP.APPROVED_SCENARIO_HASHES.get(vehicle) == vs.scenario_hash, "scenario hash is not the approved one")
    ev.update(scenario_name=vs.name, scenario_hash=vs.scenario_hash, payload_cases=V.finalist_coverage(vs)["payload_cases"],
              finalist_sha256=V.record_sha256(fin), design_sha256=fin.get("design_sha256"),
              geometry_hash=fin.get("geometry_hash"), finalist_status=fin.get("status"), coverage=fin.get("coverage"),
              mass_kg=fin.get("mass_kg"), initial_fem_sha256=V.record_sha256(fem0), bundle_sha256=V.record_sha256(rec),
              bundle_preflight=rec.get("preflight"))
    try:
        ev["returned_manifest_sha256"] = VR._verify_returned(results, rec)
    except Exception as e:                                   # noqa: BLE001
        reasons.append(f"returned CFD results failed verification: {e}")
    cr_p = results / "_cfd_result.json"
    if not check(cr_p.exists(), "CFD result missing"):
        for mr in sorted(results.parent.glob("cfd_rerun_*/_merge_report.json")):     # why the chain stopped
            rep = json.load(open(mr))
            bad = sorted(k for k, b in rep.get("banks", {}).items() if b.get("status") != "adequate")
            ev.setdefault("rerun_merge_reports", {})[mr.parent.name] = dict(
                sha256=_sha_file(mr), rerun_states=rep.get("rerun_states"), unresolved_banks=bad)
            check(not bad, f"operating-point rerun merge stopped ({mr.parent.name}): unresolved banks {bad}, "
                           f"rerun states {rep.get('rerun_states')}")
        return ev, reasons
    cr = json.load(open(cr_p))
    ev["cfd_result_sha256"] = V.record_sha256(cr)
    try:
        V.verify_lineage(vs, [fin, fem0, rec, cr])
    except Exception as e:                                   # noqa: BLE001
        reasons.append(f"lineage finalist -> FEM -> CFD case -> CFD result broken: {e}")
    # accepted / excluded CFD cases, re-derived from the evidence (must reproduce the recorded tables)
    man = RUN.load_manifest(str(results / "nominal"))
    op = cr.get("operating_point_rerun")
    if op:
        rdir = (results / op["rerun_dir"]).resolve()
        doc = json.load(open(rdir / "_rerun.json"))
        check(_sha_file(rdir / "_rerun.json") == op["rerun_sha256"], "rerun selection manifest changed after merge")
        rr = json.load(open(rdir / "_returned_manifest.json")) if (rdir / "_returned_manifest.json").exists() else None
        if check(rr is not None and _sha_file(rdir / "_returned_manifest.json") == op["rerun_returned_sha256"],
                 "rerun returned manifest missing or changed"):
            bad = [k for k, h in rr["files"].items() if not (rdir / k).exists() or _sha_file(rdir / k) != h]
            check(not bad, f"rerun result files changed after merge: {bad[:5]}")
        man = VR.merged_manifest(results, rdir, doc)
        ev["operating_point_rerun"] = dict(rerun_sha256=op["rerun_sha256"], cases=op["cases"],
                                           exclusions=op["exclusions"])
    st = RUN.scan_status(man, str(results / "nominal"))
    accepted = [Path(c["name"]).name for c in st["cases"] if c["state"] == RUN.VALID]
    ev["cfd_cases"] = dict(accepted=accepted, not_accepted=[dict(name=Path(c["name"]).name, state=c["state"],
                                                                 failed=c.get("failed")) for c in st["cases"]
                                                            if c["state"] != RUN.VALID],
                           exclusions=(op or {}).get("exclusions", []))
    try:
        agg = RUN.aggregate(man, str(results / "nominal"))
        check(json.loads(json.dumps(agg["per_bank_RX"])) == json.loads(json.dumps(cr["groups"]["nominal"]["per_bank_RX"])),
              "CFD result R/X tables do not reproduce from the accepted cases")
    except Exception as e:                                   # noqa: BLE001
        reasons.append(f"CFD aggregation failed: {e}")
    if op:
        check(all(b.get("status") == "adequate" for b in op["support"].values()),
              f"unresolved banks: {[k for k, b in op['support'].items() if b.get('status') != 'adequate']}")
    builds = {}
    for c in st["cases"]:
        if c["state"] == RUN.VALID:
            cd = Path(c["name"]) if Path(c["name"]).is_absolute() else results / "nominal" / c["name"]
            builds[Path(c["name"]).name] = _openfoam_build(cd)
    # targeted sensitivity
    sel_p, tr_p = targeted / "_targeted_selection.json", targeted / "_targeted_result.json"
    tr = None
    fail_p = targeted / VT.FAILURE
    if not tr_p.exists() and fail_p.exists():
        fl = json.load(open(fail_p))
        ev["targeted_failure"] = dict(sha256=_sha_file(fail_p), invalid_cases=fl["invalid_cases"])
        reasons.append(f"targeted sensitivity failed closed: invalid cases {fl['invalid_cases']}")
    if check(sel_p.exists() and tr_p.exists(), "targeted sensitivity selection / result missing"):
        sel, tr = json.load(open(sel_p)), json.load(open(tr_p))
        try:
            SC.record_version(sel, "cfd-targeted-selection"), SC.record_version(tr, "cfd-targeted-result")
        except SC.SchemaError as e:
            check(False, f"targeted record format: {e}")
        check(sel["cfd_result_sha256"] == ev["cfd_result_sha256"], "targeted selection was made on another CFD result")
        check(tr["selection_sha256"] == _sha_file(sel_p), "targeted result does not match its selection")
        check(not tr.get("escalate"), f"targeted sensitivity escalation triggered: {tr.get('escalation_triggers')}")
        for g, h in sel["plan_sha256"].items():
            check(_sha_file(targeted / g / "_cfd_case_plan.json") == h, f"targeted plan {g} changed")
        try:
            cases, am_sha = VT.retained_cases(targeted)
        except (VT.TargetedError, OSError, KeyError, ValueError) as e:
            cases, am_sha = [], None
            check(False, f"targeted five-case protocol not verified: {e}")
        check(tr.get("protocol") == VT.PROTOCOL and tr.get("amendment_sha256") == am_sha
              and tr.get("cases_used") == [f"{c['group']}/{c['name']}" for c in cases],
              "targeted result was not aggregated under the five-case protocol")
        for c in cases:
            builds[f"{c['group']}/{c['name']}"] = _openfoam_build(targeted / c["group"] / c["name"])
        ev["targeted"] = dict(selection_sha256=_sha_file(sel_p), result_sha256=_sha_file(tr_p), bank=tr["bank"],
                              protocol=VT.PROTOCOL, amendment_sha256=am_sha, cases_used=tr.get("cases_used"),
                              controlling_forcing=tr["controlling_forcing"], picks=sel["picks"],
                              envelope=tr["envelope"], fem_range_db=tr["fem_range_db"],
                              escalation_triggers=tr.get("escalation_triggers"))
    ev["openfoam_builds"] = sorted(set(builds.values()), key=str)
    check(len(ev["openfoam_builds"]) == 1 and None not in ev["openfoam_builds"],
          f"OpenFOAM build missing or not uniform across accepted cases: {ev['openfoam_builds']}")
    # corrected FEM
    cf_p = results / "_corrected_fem.json"
    tables = {}
    if check(cf_p.exists(), "corrected FEM missing"):
        cf = json.load(open(cf_p))
        ev["corrected_fem_sha256"] = _sha_file(cf_p)
        for f in FORCINGS:
            recs = cf["records"].get(f)
            if not check(isinstance(recs, list), f"corrected FEM incomplete for forcing {f}: {recs}"):
                continue
            for r in recs:
                try:
                    V.verify_lineage(vs, [fin, fem0, rec, cr, r])
                except Exception as e:                       # noqa: BLE001
                    reasons.append(f"corrected FEM record lineage broken: {e}")
                    break
        tables = _fem_tables(vs, cf["records"])
        for c in vs.corners.values():
            for case in vs.payload_cases:
                for f in FORCINGS:
                    got = tables.get((c["id"], case), {}).get(f, {})
                    check(all(h in got for h in VP.MESH_SIZES),
                          f"corrected FEM missing meshes for {c['id']}/{case}/{f}")
        if tr is not None:
            ctrl = tables.get((vs.corners["primary"]["id"], vs.payload_primary), {}).get(tr["controlling_forcing"], {})
            fine = min(VP.MESH_SIZES)
            check(fine in ctrl and abs(ctrl[fine] - tr["fem_nominal_0p1"]) < 1e-6,
                  "targeted nominal FEM does not reproduce the corrected FEM at 0.1 m")
    ev["fem_tables"] = {f"{k[0]}|{k[1]}": v for k, v in tables.items()}
    pre = VC.preflight(allow_dirty=True)
    ev["source"] = pre
    check(not pre["dirty"], "uncommitted src/ or configs/ changes")
    return ev, reasons


def build(vehicle, results, targeted):
    ev, reasons = collect(vehicle, results, targeted)
    s_down, s_up = (sensitivity_allowance(json.load(open(Path(targeted) / "_targeted_result.json")))
                    if "targeted" in ev else (None, None))
    tables = {tuple(k.split("|")): {f: {float(h): v for h, v in m.items()} for f, m in t.items()}
              for k, t in ev.get("fem_tables", {}).items()}
    vs = VP.scenario(vehicle)
    prim = (vs.corners["primary"]["id"], vs.payload_primary)
    iv = interval(tables.get(prim), s_down, s_up) if s_down is not None else None
    outcome = decide(iv, reasons)
    other = {}
    for k, t in tables.items():
        if k != prim and s_down is not None:
            o = interval(t, s_down, s_up)
            if o:
                other["|".join(k)] = dict(o, below_target=o["L"] <= VP.TARGET_DB)
    v = dict(schema=SCHEMA, vehicle=vehicle, scenario_name=vs.name, outcome=outcome, target_db=VP.TARGET_DB,
             primary=dict(source_corner=prim[0], payload_case=prim[1], interval=iv), reported_not_verdict=other,
             reasons=reasons, evidence=ev, wording=WORDING)
    v["record_sha256"] = V.record_sha256(json.loads(json.dumps(v, default=float)))
    return v


def _f(x, n=3):
    return "n/a" if x is None else f"{x:.{n}f}"


def render(v):
    """VERDICT.md, generated only from the verdict JSON."""
    iv = v["primary"]["interval"] or {}
    ev = v["evidence"]
    t = ev.get("targeted", {})
    lines = [f"# {v['scenario_name']} — vehicle verdict: **{v['outcome']}**", "",
             f"Simulation-based verdict against the {v['target_db']} dB target"
             + (f" ({v['prereg']})" if v.get("prereg") else "") + ". "
             f"Record hash `{v['record_sha256']}`.", "",
             "| quantity | value (dB) |", "|---|---|",
             f"| corrected FEM, 0.1 m, worst forcing (F01) | {_f(iv.get('F01'))} |",
             f"| FEM mesh allowance (M) | {_f(iv.get('M'))} |",
             f"| targeted CFD sensitivity, downward (S_down) | {_f(iv.get('S_down'))} |",
             f"| **conservative lower bound L = F01 - M - S_down** | **{_f(iv.get('L'))}** |",
             f"| conservative upper bound U | {_f(iv.get('U'))} |", "",
             f"Primary case: source corner `{v['primary']['source_corner']}`, payload `{v['primary']['payload_case']}`.", ""]
    if v["reasons"]:
        lines += ["## Why the verdict is not final", ""] + [f"- {r}" for r in v["reasons"]] + [""]
    if v["reported_not_verdict"]:
        lines += ["## Reported cases (not part of the verdict)", "", "| corner / payload | L | U | below target |",
                  "|---|---|---|---|"]
        lines += [f"| `{k}` | {_f(o['L'])} | {_f(o['U'])} | {'**yes**' if o['below_target'] else 'no'} |"
                  for k, o in sorted(v["reported_not_verdict"].items())] + [""]
    lines += ["## Evidence", "",
              f"- Scenario `{ev.get('scenario_hash')}`; finalist `{ev.get('finalist_sha256')}` (design "
              f"`{ev.get('design_sha256')}`, coverage {ev.get('coverage')}, mass {_f(ev.get('mass_kg'), 1)} kg).",
              f"- Initial FEM `{ev.get('initial_fem_sha256')}`; CFD bundle `{ev.get('bundle_sha256')}`; CFD result "
              f"`{ev.get('cfd_result_sha256')}`; corrected FEM `{ev.get('corrected_fem_sha256')}`.",
              f"- Source commit `{(ev.get('source') or {}).get('commit')}`; OpenFOAM build {ev.get('openfoam_builds')}.",
              f"- CFD cases accepted: {len((ev.get('cfd_cases') or {}).get('accepted', []))}; excluded: "
              f"{len((ev.get('cfd_cases') or {}).get('exclusions', []))} (listed in verdict.json).",
              f"- Targeted operating-region sensitivity ({t.get('protocol')}): bank b{t.get('bank')}, forcing "
              f"{t.get('controlling_forcing')}, cases {t.get('cases_used')}, FEM range {t.get('fem_range_db')}"
              + (f"; amendment `{t.get('amendment_sha256')}`" if t.get("amendment_sha256") else "") + ".", "",
              "## Limitations", ""] + [f"- {w}" for w in v["wording"].values()]
    return "\n".join(lines) + "\n"


def publish(v, out_dir, renderer=None):
    """Atomic write of verdict.json + VERDICT.md; an identical existing verdict is a no-op, a different one refused."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    js = json.dumps(v, sort_keys=True, indent=1, default=float) + "\n"
    md = (renderer or render)(json.loads(js))
    files = {"verdict.json": js, "VERDICT.md": md}
    existing = {n: (out_dir / n).read_text() for n in files if (out_dir / n).exists()}
    if existing:
        if existing == files:
            return out_dir
        raise VerdictError(f"{out_dir} already holds a different verdict; verdicts are immutable")
    for n, text in files.items():
        tmp = out_dir / (n + ".tmp")
        tmp.write_text(text, newline="\n")
        os.replace(tmp, out_dir / n)
        os.chmod(out_dir / n, stat.S_IREAD)
    return out_dir


def main(argv=None):
    p = argparse.ArgumentParser(prog="acoustic-liner vehicles verdict")
    p.add_argument("--vehicle", choices=sorted(VP.APPROVED_SCENARIO_HASHES), required=True)
    p.add_argument("--results", required=True, help="returned CFD bundle holding _cfd_result.json and _corrected_fem.json")
    p.add_argument("--targeted", required=True, help="cfd_targeted directory with returned results")
    a = p.parse_args(argv)
    v = build(a.vehicle, a.results, a.targeted)
    out = publish(v, VP.RUN_DIR / v["scenario_name"] / "verdict")
    iv = v["primary"]["interval"] or {}
    print(f"{v['scenario_name']}: {v['outcome']}  L {_f(iv.get('L'))}  U {_f(iv.get('U'))} dB  -> {out}")
    for r in v["reasons"]:
        print("  reason:", r)


if __name__ == "__main__":
    main()
