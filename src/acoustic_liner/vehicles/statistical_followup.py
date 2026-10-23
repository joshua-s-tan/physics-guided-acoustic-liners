# -*- coding: utf-8 -*-
"""Unified statistical-impedance follow-up for Atlas and Falcon (protocol `VEHICLE-STATISTICAL-FOLLOWUP-PROTOCOL`,
kept in the private archive).
POST-RESULT and separately declared; the frozen UNRESOLVED records are never touched.

    acoustic-liner vehicles statistical-followup table    --vehicle V   (block table + bank support + selection check)
    acoustic-liner vehicles statistical-followup pairs    --vehicle V   (gate the five sensitivity pairs)
    acoustic-liner vehicles statistical-followup continue --vehicle V   (copies + run_continue.sh for failing pairs)
    bash run_continue.sh [N]                                                (WSL; serial per case)
    acoustic-liner vehicles statistical-followup finish   --vehicle V   (stitch, gates, envelope, FEM, verdict)
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import shutil
import socket
from pathlib import Path

import numpy as np

from ..provenance import schemas as SC
from . import cfd_bundle as VC
from . import cfd_supplemental as S
from . import cfd_targeted as VT
from . import search as VP
from . import scenarios as V
from ..cfd import case_generation as CG
from ..cfd import feedback as FB

# the protocol's name, as records carry it (the document itself is kept privately)
PROTOCOL = "docs/VEHICLE-STATISTICAL-FOLLOWUP-PROTOCOL.md"
OUT = "statistical_followup"
SETTLING_CYCLES, ANALYSIS_CYCLES = 1, 12
OVERLAP_TOL = 0.01
PLANES = ("pUp", "pDown", "fluxMid")
PASS, FAIL, UNRESOLVED = "STATISTICAL FOLLOW-UP PASS", "STATISTICAL FOLLOW-UP FAIL", "STATISTICAL FOLLOW-UP UNRESOLVED"
_V = VP.RUN_DIR
VEHICLES = dict(
    falcon=dict(dir="falcon9_2025_unblanketed_v2", bundle="nominal_completed_cfd_bundle", rerun="cfd_rerun_24c",
                targeted="supplemental_statistical/cfd_targeted"),
    atlas=dict(dir="atlas_i_14ft_representative_unblanketed", bundle="nominal_comleted_cfd_bundle",
               rerun="cfd_rerun_24c", targeted="cfd_targeted"))
FROZEN = {  # protocol sec. 0
    "falcon9_2025_unblanketed_v2/verdict/verdict.json": "0dc0dc01357d010d41a15a704019ce500bb759453da03e251e0a16c4bb63c419",
    "falcon9_2025_unblanketed_v2/supplemental_statistical/verdict/verdict.json":
        "ceaac17efd0e04b24c5ef5bf3cd5e40fdf8ab93b123efda0a556088caf97109c",
    "atlas_i_14ft_representative_unblanketed/verdict/verdict.json":
        "77e059185d92ee4c68fe56bdb0fdcd29fd47eef95e1845d5bb1a7c0d5e8393b2"}


class FollowupError(RuntimeError):
    pass


def _paths(vehicle):
    c = VEHICLES[vehicle]
    base = _V / c["dir"]
    return dict(bdir=base / c["bundle"], rdir=base / c["rerun"], tdir=base / c["targeted"], root=base / OUT)


def _wj(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, sort_keys=True, indent=1, default=float))


def _immutable(p, obj):
    p = Path(p)
    if p.exists() and json.load(open(p)) != json.loads(json.dumps(obj, default=float)):
        raise FollowupError(f"{p} exists with different content; it is immutable evidence")
    _wj(p, obj)


# ---- pure helpers (unit-tested) ----------------------------------------------------------------------------------
def overlap_ok(t_par, v_par, t_con, v_con, t_r, t_end, period):
    """Protocol sec. 4: max |cont - parent| over re-covered samples <= OVERLAP_TOL * max |parent| (last cycle)."""
    t_par, v_par, t_con, v_con = map(np.asarray, (t_par, v_par, t_con, v_con))
    m = (t_con > t_r) & (t_con <= t_end)
    if not m.any():
        return False, dict(error="no re-covered samples")
    diff = float(np.max(np.abs(v_con[m] - np.interp(t_con[m], t_par, v_par))))
    scale = float(np.max(np.abs(v_par[t_par >= t_end - period])))
    return diff <= OVERLAP_TOL * scale, dict(max_diff=diff, scale=scale, n=int(m.sum()), ratio=diff / scale if scale else None)


def stitch(t_par, v_par, t_con, v_con, t_r):
    """Parent samples t <= t_r, then continuation samples t > t_r (duplicated overlap removed)."""
    t_par, v_par, t_con, v_con = map(np.asarray, (t_par, v_par, t_con, v_con))
    a, b = t_par <= t_r, t_con > t_r
    return np.concatenate([t_par[a], t_con[b]]), np.concatenate([v_par[a], v_con[b]])


def shifted_row(table_row, src_est, var_est):
    """U* substitution (sec. 5): the case's table row shifted by the measured change (variant - source)."""
    return tuple(t + (v - s) for t, s, v in zip(table_row, src_est, var_est))


# ---- step 1: statistical table, selection consistency --------------------------------------------------------
def table(vehicle):
    P = _paths(vehicle)
    res = S.estimate(P["bdir"], P["rdir"], out=P["root"], vehicles=tuple(VEHICLES), protocol=PROTOCOL,
                     name="_stat_cfd_result.json")
    chk = dict(protocol=PROTOCOL)
    if res["unresolved_banks"]:
        chk.update(match=False, reason=f"banks without support {res['unresolved_banks']}")
    else:
        sdir = P["root"] / "_selection_check"
        if (sdir / "_targeted_selection.json").exists():
            sel = json.load(open(sdir / "_targeted_selection.json"))
        else:
            sel = VT.select(P["bdir"], out=sdir, cr_path=P["root"] / "_stat_cfd_result.json", estimator="supplemental_block")
        existing, _ = VT.retained_cases(P["tdir"])
        old_sel = json.load(open(P["tdir"] / "_targeted_selection.json"))
        a = sorted((c["group"], c["name"]) for c in sel["cases"])
        b = sorted((c["group"], c["name"]) for c in existing)
        chk.update(match=(sel["bank"] == old_sel["bank"] and a == b), bank_followup=sel["bank"],
                   bank_existing=old_sel["bank"], cases_followup=a, cases_existing=b,
                   forcing_followup=sel["controlling_forcing"], u_star_followup=sel["u_star"])
    _immutable(P["root"] / "_selection_check.json", chk)
    print(f"{vehicle}: table banks {res['unresolved_banks'] or 'all supported'}; selection consistent: {chk['match']}")
    return res, chk


# ---- step 2: gate the five pairs ------------------------------------------------------------------------------
def _ctx(vehicle):
    P = _paths(vehicle)
    return P, VT._ctx(P["bdir"], P["root"] / "_stat_cfd_result.json")


def pairs(vehicle):
    P, ctx = _ctx(vehicle)
    chk = json.load(open(P["root"] / "_selection_check.json"))
    if not chk["match"]:
        raise FollowupError(f"{UNRESOLVED}: {chk.get('reason') or 'controlling bank / five-case set changed'}")
    cases, _ = VT.retained_cases(P["tdir"])
    src = VT._sources(ctx)
    out = []
    for c in cases:
        e, gh = VT._plan_entry(P["tdir"], c["group"], c["name"])
        vrow, vok, vrep = S.bound_supp_row(P["tdir"] / c["group"] / c["name"], e, gh)
        scd, se, sgh = src[c["name"]]
        if se["input_tree_hash"] != e["source_input_tree_hash"]:
            raise FollowupError(f"{c['name']}: nominal source differs from the one the variant was built from")
        srow, sok, srep = S.bound_supp_row(scd, se, sgh)
        out.append(dict(case=f"{c['group']}/{c['name']}", group=c["group"], name=c["name"], slot=c["slot"],
                        variant_dir=str(P["tdir"] / c["group"] / c["name"]), source_dir=str(scd),
                        variant=dict(ok=vok, failed=vrep.get("failed"), row=vrow),
                        source=dict(ok=sok, failed=srep.get("failed"), row=srow),
                        continue_pair=not (vok and sok)))
    _immutable(P["root"] / "_pairs.json", dict(protocol=PROTOCOL, pairs=out))
    print(f"{vehicle}: pairs needing continuation {[p['case'] for p in out if p['continue_pair']] or 'none'}")
    return out


# ---- step 3: continuation copies -------------------------------------------------------------------------------
def _latest(cd):
    ts = [d.name for d in Path(cd).iterdir() if re.fullmatch(r"[0-9.eE+-]+", d.name) and d.name != "0"]
    return max(ts, key=float)


def _sha_tree(d):
    return VC._sha(sorted((p.relative_to(d).as_posix(), VC._sha_file(p)) for p in Path(d).rglob("*") if p.is_file()))


def _build(cd, log="log.pimpleFoam"):
    from . import verdict as VV
    p = Path(cd) / log
    if log != "log.pimpleFoam":
        tmp = Path(cd) / "_build_probe"
        tmp.mkdir(exist_ok=True); shutil.copy(p, tmp / "log.pimpleFoam")
        b = VV._openfoam_build(tmp); shutil.rmtree(tmp); return b
    return VV._openfoam_build(cd)


def prepare_continuation(vehicle):
    P = _paths(vehicle)
    pr = json.load(open(P["root"] / "_pairs.json"))["pairs"]
    croot = P["root"] / "continuation"
    if (croot / "_continuation_plan.json").exists():
        raise FollowupError("continuation already prepared; it is immutable evidence")
    members = []
    for p in pr:
        if not p["continue_pair"]:
            continue
        for role in ("variant", "source"):
            par = Path(p[f"{role}_dir"])
            meta = json.load(open(par / "_case_meta.json"))
            N, f0 = int(meta["cycles"]), float(meta["f0_Hz"])
            tr = _latest(par)
            dst = croot / f"{p['group']}__{p['name']}" / role
            shutil.copytree(par, dst)
            (dst / "postProcessing").rename(dst / "_parent_postProcessing")
            (dst / "log.pimpleFoam").rename(dst / "log.pimpleFoam.parent")
            (dst / "_status.json").rename(dst / "_status.parent.json")
            cdp = dst / "system" / "controlDict"
            txt = cdp.read_text()
            new_end = (N + SETTLING_CYCLES + ANALYSIS_CYCLES) / f0
            txt2 = re.sub(r"^startFrom\s+\S+;", "startFrom       latestTime;", txt, flags=re.M)
            txt2 = re.sub(r"^endTime\s+\S+;", f"endTime         {new_end!r};", txt2, flags=re.M)
            if txt2.count("latestTime") != 1 or f"{new_end!r}" not in txt2:
                raise FollowupError(f"{dst}: controlDict edit failed")
            cdp.write_text(txt2, newline="\n")
            members.append(dict(pair=p["case"], role=role, dir=dst.relative_to(croot).as_posix(), parent_dir=str(par),
                                parent_case_hash=meta["case_hash"], parent_input_tree_hash=CG.input_tree_hash(str(par)),
                                restart_time=tr, restart_state_sha256=_sha_tree(par / tr), cycles_parent=N, f0=f0,
                                old_end=N / f0, new_end=new_end, parent_build=_build(par),
                                controlDict_sha256=VC._sha_file(cdp)))
    lines = ["#!/usr/bin/env bash", "# bash run_continue.sh [parallel jobs, default 6]  (serial solver per case)",
             'cd "$(dirname "$0")"', 'N="${1:-6}"',
             "run() { c=\"$1\"; [ -f \"$c/_cont_status.json\" ] && grep -q done \"$c/_cont_status.json\" && exit 0;"
             " ( cd \"$c\" && pimpleFoam > log.pimpleFoam 2>&1 ); rc=$?; s=failed; [ $rc -eq 0 ] && s=done;"
             " printf '{\"state\": \"%s\", \"rc\": %d, \"host\": \"%s\", \"finished\": \"%s\"}\\n' \"$s\" $rc \"$(hostname)\" \"$(date -Is)\""
             " > \"$c/_cont_status.json\"; echo \"$s $c\"; }", "export -f run",
             "printf '%s\\n' " + " ".join(m["dir"] for m in members) + ' | xargs -P "$N" -I{} bash -c \'run "$@"\' _ {}',
             'echo "continuations finished"']
    croot.mkdir(parents=True, exist_ok=True)
    (croot / "run_continue.sh").write_text("\n".join(lines) + "\n", newline="\n")
    _immutable(croot / "_continuation_plan.json", dict(protocol=PROTOCOL, members=members,
                                                       pairs_sha256=VC._sha_file(P["root"] / "_pairs.json"),
                                                       settling_cycles=SETTLING_CYCLES, analysis_cycles=ANALYSIS_CYCLES,
                                                       overlap_tol=OVERLAP_TOL, prepared_on=socket.gethostname()))
    print(f"{vehicle}: {len(members)} continuation copies -> {croot}")
    return members


# ---- step 4: stitch + evaluate continued members ---------------------------------------------------------------
def _plane(dirp):
    d = Path(dirp)
    tds = [x for x in d.iterdir() if x.is_dir()]
    if len(tds) != 1:
        raise FollowupError(f"{d}: expected one time directory, got {[x.name for x in tds]}")
    t, v, raw = [], [], []
    for line in open(tds[0] / "surfaceFieldValue.dat"):
        if line.startswith("#") or not line.strip():
            continue
        q = line.replace("(", " ").replace(")", " ").split()
        t.append(float(q[0])); v.append(float(q[-1])); raw.append(line.rstrip("\n"))
    return np.array(t), np.array(v), raw


def evaluate_member(croot, m, entry, geometry_hash):
    """Verify binding, overlap, stitch the raw histories, then block-estimate the last ANALYSIS_CYCLES cycles."""
    d = croot / m["dir"]
    st = d / "_cont_status.json"
    rep = dict(member=m["dir"])
    if not (st.exists() and json.load(open(st)).get("state") == "done"):
        return None, False, dict(rep, error="continuation not finished")
    if VC._sha_file(d / "system" / "controlDict") != m["controlDict_sha256"] or \
            _sha_tree(Path(m["parent_dir"]) / m["restart_time"]) != m["restart_state_sha256"]:
        return None, False, dict(rep, error="binding changed (controlDict / restart state)")
    b = _build(d)
    if b != m["parent_build"]:
        return None, False, dict(rep, error=f"OpenFOAM build {b} != parent {m['parent_build']}")
    sdir = d.parent / (d.name + "_stitched")
    if sdir.exists():
        shutil.rmtree(sdir)
    tr, period = float(m["restart_time"]), 1.0 / m["f0"]
    ov = {}
    for pl in sorted(x.name for x in (d / "_parent_postProcessing").iterdir()):
        tp, vp, rawp = _plane(d / "_parent_postProcessing" / pl)
        tc, vc, rawc = _plane(d / "postProcessing" / pl)
        if pl in PLANES:
            ok, info = overlap_ok(tp, vp, tc, vc, tr, m["old_end"], period)
            ov[pl] = dict(info, ok=ok)
        keep = [r for r, t in zip(rawp, tp) if t <= tr] + [r for r, t in zip(rawc, tc) if t > tr]
        (sdir / "postProcessing" / pl / "0").mkdir(parents=True)
        (sdir / "postProcessing" / pl / "0" / "surfaceFieldValue.dat").write_text("\n".join(keep) + "\n")
    rep["overlap"] = ov
    if not all(o["ok"] for o in ov.values()) or set(ov) != set(PLANES):
        return None, False, dict(rep, error="overlap check failed")
    meta = json.load(open(d / "_case_meta.json"))
    meta["cycles"] = m["cycles_parent"] + SETTLING_CYCLES + ANALYSIS_CYCLES
    if CG.recompute_case_hash(meta) != m["parent_case_hash"]:
        raise FollowupError("case hash must not depend on the cycle count")
    (sdir / "_case_meta.json").write_text(json.dumps(meta, indent=2))
    shutil.copy(d / "log.checkMesh", sdir / "log.checkMesh")
    row, ok, r = S.supp_row(sdir, entry, geometry_hash, analysis_cycles=ANALYSIS_CYCLES)
    rep.update(r, stitched_sha256=_sha_tree(sdir / "postProcessing"), build=b,
               host=json.load(open(st)).get("host"))
    return row, ok, rep


# ---- step 5: envelope, FEM, decision ---------------------------------------------------------------------------
def finish(vehicle):
    P, ctx = _ctx(vehicle)
    reasons = []
    pr = json.load(open(P["root"] / "_pairs.json"))["pairs"]
    croot = P["root"] / "continuation"
    plan = json.load(open(croot / "_continuation_plan.json")) if (croot / "_continuation_plan.json").exists() else None
    src = VT._sources(ctx)
    used, member_reports = [], []
    for p in pr:
        if not p["continue_pair"]:
            used.append(dict(p, est_variant=p["variant"]["row"], est_source=p["source"]["row"], window="existing"))
            continue
        if plan is None:
            reasons.append(f"{p['case']}: continuation required but not prepared"); continue
        e, gh = VT._plan_entry(P["tdir"], p["group"], p["name"])
        _, se, sgh = src[p["name"]]
        rows = {}
        for m in plan["members"]:
            if m["pair"] != p["case"]:
                continue
            row, ok, rep = evaluate_member(croot, m, e if m["role"] == "variant" else se, gh if m["role"] == "variant" else sgh)
            member_reports.append(dict(rep, ok=ok))
            if not ok:
                reasons.append(f"{p['case']} {m['role']}: {rep.get('error') or rep.get('failed')}")
            rows[m["role"]] = row
        if all(rows.get(k) for k in ("variant", "source")) and not any(
                r.startswith(p["case"]) for r in reasons):
            used.append(dict(p, est_variant=rows["variant"], est_source=rows["source"], window="last 12 cycles"))
    out = dict(protocol=PROTOCOL, vehicle=vehicle, member_reports=member_reports, reasons=list(reasons))
    if not reasons:
        per_case = []
        for u in used:
            sv, vv = u["est_source"], u["est_variant"]
            r0, x0, r1, x1 = sv["R_phasor"], sv["X_phasor"], vv["R_phasor"], vv["X_phasor"]
            z = math.hypot(r0, x0)
            per_case.append(dict(case=u["case"], slot=u["slot"], window=u["window"], R_source=r0, X_source=x0,
                                 R_variant=r1, X_variant=x1, norm_dR=abs(r1 - r0) / z, norm_dX=abs(x1 - x0) / z,
                                 src=(sv["u_amp"], r0, x0), var=(vv["u_amp"], r1, x1)))
        sel = json.load(open(P["tdir"] / "_targeted_selection.json"))
        bank, key = sel["bank"], sel["controlling_forcing"]
        forcing = VT.FORCINGS[key]
        eR, eX = max(p["norm_dR"] for p in per_case), max(p["norm_dX"] for p in per_case)
        nom_leg = FB.relabel_design(ctx["design"], ctx["cond"], ctx["rx"], ctx["anc"], forcing=forcing)
        u0 = float(nom_leg["per_bank"][bank]["U_star"])
        tab_rows = {Path(c["name"]).name: tuple(c["row"][k] for k in ("u_amp", "R_phasor", "X_phasor"))
                    for c in json.load(open(P["root"] / "_stat_cfd_result.json"))["cases"] if c["state"] == S.SUPPORTED}
        shifts = []
        for pc, u in zip(per_case, used):
            if pc["slot"] != "ustar":
                continue
            old = tab_rows[u["name"]]
            new = shifted_row(old, pc["src"], pc["var"])
            rx = [[new if t == old else t for t in tab] for tab in ctx["rx"]]
            if rx == ctx["rx"]:
                raise FollowupError(f"{u['name']}: table row not found for the U* substitution")
            leg = FB.relabel_design(ctx["design"], ctx["cond"], rx, ctx["anc"], forcing=forcing)
            us = leg["per_bank"][bank].get("U_star")
            shifts.append(dict(case=pc["case"], U_star=us, rel_shift=abs(us - u0) / u0 if us is not None else None))
        fem_nom = VT._fem(ctx, nom_leg["dR_excess_Rayl"], nom_leg["dM_inertance_Rayl_s"], VT.RESULT_MESH)
        combos = []
        for sR, sX in itertools.product((1, -1), (1, -1)):
            rx = VT.perturb_tables(ctx["rx"], ctx["anc"], VT.z_mag, eR, eX, sR, sX)
            leg = FB.relabel_design(ctx["design"], ctx["cond"], rx, ctx["anc"], forcing=forcing)
            val = (VT._fem(ctx, leg["dR_excess_Rayl"], leg["dM_inertance_Rayl_s"], VT.RESULT_MESH)
                   if leg["status"] == "complete" else None)
            combos.append(dict(sign_R=sR, sign_X=sX, relabel_status=leg["status"], fem_robust_db=val))
        vals = [fem_nom] + [c["fem_robust_db"] for c in combos if c["fem_robust_db"] is not None]
        ust = [s["rel_shift"] for s in shifts if s["rel_shift"] is not None]
        trig = VT.triggers(fem_range_db=max(vals) - min(vals), max_norm_change=max(eR, eX),
                           ustar_shift=max(ust) if ust else None, verdicts=[v >= VP.TARGET_DB for v in vals])
        if any(c["fem_robust_db"] is None for c in combos) or any(s["U_star"] is None for s in shifts):
            trig.append("relabel incomplete under a perturbation (fixed point lost)")
        for pc in per_case:
            pc.pop("src"); pc.pop("var")
        targeted = dict(bank=bank, controlling_forcing=key, per_case=per_case, envelope=dict(norm_dR=eR, norm_dX=eX),
                        u_star_nominal=u0, u_star_shifts=shifts, fem_nominal_0p1=fem_nom, fem_envelope_combos_0p1=combos,
                        fem_range_db=[min(vals), max(vals)], escalation_triggers=trig, escalate=bool(trig))
        out["targeted"] = targeted
        if trig:
            reasons.append(f"escalation triggered and not completed: {trig}")
        res = json.load(open(P["root"] / "_stat_cfd_result.json"))
        if not (P["root"] / "_stat_fem.json").exists():
            S.propagate(P["bdir"], root=P["root"], res=res, protocol=PROTOCOL, out_name="_stat_fem.json")
        sf = json.load(open(P["root"] / "_stat_fem.json"))
        if sf["supp_cfd_result_sha256"] != V.record_sha256(res):
            reasons.append("FEM belongs to another statistical table")
        if sf["lost_fixed_point"]:
            reasons.append(f"fixed point lost under the impedance interval: {sf['lost_fixed_point']}")
        from . import verdict as VV
        s_down, s_up = VV.sensitivity_allowance(targeted)
        fem = {t: {f: {float(h): v for h, v in m.items()} for f, m in d.items()} for t, d in sf["fem"].items()}
        out["interval"] = S.supp_interval(fem, s_down, s_up)
        out["reported_central"] = sf["reported_central"]
    _wj(P["root"] / "_followup_result.json", out)
    return publish(vehicle, out, reasons)


WORDING = dict(
    followup=("The preregistered verdicts remained unresolved. This post-result, unified statistical follow-up treated "
              "the neck resistance as statistically unsteady and estimated a block-averaged complex impedance with "
              "uncertainty for both vehicles with one method"),
    not_prereg="Post-result and separately declared; not preregistered; no deterministic periodic convergence is claimed",
    normalisation=S.WORDING["normalisation"], targeted=S.WORDING["targeted"], basis=S.WORDING["basis"],
    a_other=S.WORDING["a_other"], audit=S.WORDING["audit"])


def _frozen_ok():
    return {k: (_V / k).exists() and VC._sha_file(_V / k) == h for k, h in FROZEN.items()}


def publish(vehicle, out, reasons):
    from . import verdict as VV
    P = _paths(vehicle)
    try:
        src = VC.preflight()
    except Exception as e:                                   # noqa: BLE001
        src = None
        reasons.append(f"source tree not clean: {e}")
    fz = _frozen_ok()
    if not all(fz.values()):
        reasons.append(f"frozen records changed or missing: {[k for k, v in fz.items() if not v]}")
    res = json.load(open(P["root"] / "_stat_cfd_result.json"))
    try:
        SC.record_version(res, "statistical-impedance-estimate")
    except SC.SchemaError as e:
        reasons.append(f"statistical table format: {e}")
    chk = json.load(open(P["root"] / "_selection_check.json"))
    if not chk["match"]:
        reasons.append(f"controlling bank / five-case set changed: {chk}")
    iv = out.get("interval")
    outcome = S.outcome(iv, reasons)
    outcome = {S.PASS: PASS, S.FAIL: FAIL, S.UNRESOLVED: UNRESOLVED}[outcome]
    vs = VP.scenario(vehicle)
    v = dict(schema=SC.schema("vehicle-statistical-followup"), vehicle=vehicle, scenario_name=vs.name, outcome=outcome,
             target_db=VP.TARGET_DB, interval=iv, reasons=reasons, frozen_records=FROZEN, source=src,
             evidence=dict(stat_cfd_result_sha256=V.record_sha256(res),
                           followup_result_sha256=VC._sha_file(P["root"] / "_followup_result.json"),
                           selection_check=chk, targeted=out.get("targeted"), member_reports=out["member_reports"]),
             reported_not_verdict=out.get("reported_central"), wording=WORDING, protocol=PROTOCOL, post_result=True)
    v["record_sha256"] = V.record_sha256(json.loads(json.dumps(v, default=float)))
    d = VV.publish(v, P["root"] / "verdict", renderer=render)
    print(f"{vehicle}: {outcome}  L {S._f((iv or {}).get('L'))}  U {S._f((iv or {}).get('U'))} dB -> {d}")
    for r in reasons:
        print("  reason:", r)
    return v


def render(v):
    iv = v.get("interval") or {}
    t = (v.get("evidence") or {}).get("targeted") or {}
    lines = [f"# {v['vehicle'].capitalize()} statistical follow-up (post-result): **{v['outcome']}**", "",
             f"{v['wording']['followup']} ({v['protocol']}). The preregistered result and earlier records stay "
             "unchanged (hashes in verdict.json).", "",
             "| quantity | dB |", "|---|---|",
             f"| F01 (min over 5 impedance tables x 2 forcings, 0.1 m) | {S._f(iv.get('F01'))} |",
             f"| F01_hi | {S._f(iv.get('F01_hi'))} |", f"| mesh allowance M | {S._f(iv.get('M'))} |",
             f"| targeted sensitivity S_down / S_up | {S._f(iv.get('S_down'))} / {S._f(iv.get('S_up'))} |",
             f"| **L** | **{S._f(iv.get('L'))}** |", f"| **U** | **{S._f(iv.get('U'))}** |",
             f"| target | {v['target_db']} |", ""]
    if v["reasons"]:
        lines += ["## Why not resolved", ""] + [f"- {r}" for r in v["reasons"]] + [""]
    lines += ["## Sensitivity pairs", "", "| pair | window | dR/abs(Z) | dX/abs(Z) |", "|---|---|---|---|"]
    for p in t.get("per_case", []):
        lines.append(f"| `{p['case']}` | {p['window']} | {S._f(p['norm_dR'])} | {S._f(p['norm_dX'])} |")
    lines += ["", f"Escalation triggers: {t.get('escalation_triggers')}.", "",
              "## Reported, not part of the decision (central table)", "", "| corner / payload | forcing | 0.1 m |",
              "|---|---|---|"]
    for k, d in sorted((v.get("reported_not_verdict") or {}).items()):
        for f, m in sorted(d.items()):
            lines.append(f"| `{k}` | {f} | {S._f(m.get('0.1'))} |")
    lines += ["", "## Limitations", ""] + [f"- {w}" for k, w in v["wording"].items() if k != "followup"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(prog="acoustic-liner vehicles statistical-followup")
    p.add_argument("cmd", choices=("table", "pairs", "continue", "finish"))
    p.add_argument("--vehicle", choices=sorted(VEHICLES), required=True)
    a = p.parse_args(argv)
    dict(table=table, pairs=pairs, finish=finish)[a.cmd](a.vehicle) if a.cmd != "continue" else \
        prepare_continuation(a.vehicle)


if __name__ == "__main__":
    main()
