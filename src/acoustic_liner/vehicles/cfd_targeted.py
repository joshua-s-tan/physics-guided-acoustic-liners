# -*- coding: utf-8 -*-
"""Targeted CFD mesh/time-step sensitivity for a vehicle bundle.

Runs after the nominal result exists (`acoustic-liner vehicles cfd-bundle aggregate`, or
`acoustic-liner vehicles cfd-rerun merge` when operating-point
reruns replaced unsettled cases). `select` writes a small portable bundle (`<vehicle>/cfd_targeted/`) holding the 5
variant cases of the operating-region protocol (prereg amendment, sec. 7), each generated from its nominal source case with the SAME imposed velocity and the SAME cycle count
(24 for a rerun source, 12 otherwise); a 12-cycle variant must reproduce the parent bundle's own sens_* case exactly.

    acoustic-liner vehicles cfd-targeted select    --bundle DIR                 (bank + 3 velocities -> cfd_targeted/)
    bash run_targeted.sh [N]                                                       (WSL, inside cfd_targeted; resumable)
    acoustic-liner vehicles cfd-targeted amend     --targeted TDIR              (seven-case selection -> five-case)
    acoustic-liner vehicles cfd-targeted aggregate --bundle DIR --targeted TDIR (envelope -> FEM dB range + triggers)

The result is a targeted operating-region mesh/time-step sensitivity: numerical sensitivity in the operating region
only, not formal grid convergence, full-range CFD convergence or a fully quantified CFD uncertainty.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..physics import oracle as O2
from ..provenance import schemas as SC
from ..design import canonical as CAN
from . import cfd_bundle as VC
from . import search as VP
from . import scenarios as V
from ..cfd import case_generation as CG
from ..cfd import feedback as FB
from ..cfd import postprocess as POST
from ..cfd import impedance_correction as RL
from ..cfd import runner as RUN
from ..cfd import validation as VAL

VARIANTS = dict(sens_refine=dict(refine=2.0, maxco=0.5), sens_halfdt=dict(refine=1.0, maxco=0.25),
                sens_refine_halfdt=dict(refine=2.0, maxco=0.25))   # combined: seven-case protocol only (dropped)
VARIANT_GROUPS = ("sens_refine", "sens_halfdt")
SELECTION_MESH, RESULT_MESH = 0.2, 0.1
TIE_DB = 1e-3
ENVELOPE_DB_TOL, Z_TOL, USTAR_TOL = 0.10, 0.10, 0.10
FORCINGS = dict(broadband="broadband_equivalent", resonant_tone="resonant_tone_upper_bound")
ENVELOPE_WORDING = ("a conservative empirical sensitivity envelope, assuming untested banks and velocities are no "
                    "more sensitive than the tested controlling cases")
# prereg amendment 1 (sec. 7): five-case operating-region protocol, identical for both vehicles
PROTOCOL = "operating_region_5case"
LEGACY_PROTOCOL = "seven_case"
FIVE_CASE = (("sens_refine", "low"), ("sens_refine", "ustar"), ("sens_halfdt", "low"), ("sens_halfdt", "ustar"),
             ("sens_halfdt", "high"))
LABEL = ("targeted operating-region mesh/time-step sensitivity (numerical sensitivity in the operating region only; "
         "not formal grid convergence, full-range CFD convergence or full CFD uncertainty)")
LIMITATIONS = ("high-speed mesh sensitivity is not quantified",
               "mesh-time-step interaction is not quantified",
               "the measured envelope is not a proven global bound over the entire velocity range")
AMENDMENT = "_targeted_amendment.json"
FAILURE = "_targeted_failure.json"           # written when aggregate fails closed (names the invalid cases)
AMENDMENT_REASON = ("compute time and relevance to the self-consistent operating region; not the expected verdict "
                    "or margin")


class TargetedError(RuntimeError):
    pass


# ---- pure rules (unit-tested) -------------------------------------------------------------------------------
def pick_velocities(rows, u_star):
    """rows: [(name, u_target, u_amp)] validated, non-reference. Returns [(slot, name)] per prereg sec. 3."""
    rows = sorted(rows, key=lambda r: (r[1], r[0]))
    if len(rows) <= 3:
        mid = min(rows, key=lambda r: (abs(r[2] - u_star), r[1])) if rows else None
        return [("ustar" if r is mid else ("low" if r[1] < mid[1] else "high"), r[0]) for r in rows]
    k = min(range(len(rows)), key=lambda i: (abs(rows[i][2] - u_star), rows[i][1]))
    lo = 1 if k == 0 else 0
    hi = len(rows) - 2 if k == len(rows) - 1 else len(rows) - 1
    return [("low", rows[lo][0]), ("ustar", rows[k][0]), ("high", rows[hi][0])]


def pick_bank(deltas):
    """deltas: per-bank |FEM_j - FEM_uncorrected|. Largest; ties within TIE_DB go to the lowest index."""
    top = max(deltas)
    return min(j for j, d in enumerate(deltas) if d >= top - TIE_DB)


def controlling_forcing(fem_by_forcing):
    """Lower robust corrected FEM controls; ties within TIE_DB go to resonant_tone."""
    bb, tn = fem_by_forcing["broadband"], fem_by_forcing["resonant_tone"]
    return "broadband" if bb < tn - TIE_DB else "resonant_tone"


def perturb_tables(per_bank_RX, per_bank_anchor, z_mag, eR, eX, sR, sX):
    """Shift every NON-anchor row by sign*e*|Z| (anchor kept nominal so the excess carries the shift)."""
    out = []
    for j, tab in enumerate(per_bank_RX):
        anc = per_bank_anchor[j]
        rows = []
        for u, r, x in tab:
            if anc is not None and abs(u - anc[0]) < 1e-12 and abs(r - anc[1]) < 1e-12:
                rows.append((u, r, x))
            else:
                z = z_mag(j, u, r, x)
                rows.append((u, r + sR * eR * z, x + sX * eX * z))
        out.append(rows)
    return out


def protocol_cases(picks):
    """[(slot, name)] -> the five-case protocol's cases, in pick order (prereg sec. 7)."""
    return [dict(group=g, name=nm, slot=s) for s, nm in picks for g in VARIANT_GROUPS if (g, s) in FIVE_CASE]


def split_legacy(cases):
    """Seven-case selection cases -> (retained, dropped) under the five-case protocol."""
    keep = [c for c in cases if (c["group"], c["slot"]) in FIVE_CASE]
    return keep, [c for c in cases if c not in keep]


def triggers(*, fem_range_db, max_norm_change, ustar_shift, verdicts):
    t = []
    if fem_range_db > ENVELOPE_DB_TOL:
        t.append(f"corrected-FEM envelope range {fem_range_db:.3f} dB > {ENVELOPE_DB_TOL} dB")
    if max_norm_change > Z_TOL:
        t.append(f"impedance change {max_norm_change:.3f} of |Z| > {Z_TOL}")
    if ustar_shift is not None and ustar_shift > USTAR_TOL:
        t.append(f"U* shift {ustar_shift:.3f} > {USTAR_TOL}")
    if len(set(verdicts)) > 1:
        t.append("target verdict differs across the envelope")
    return t


# ---- bundle context --------------------------------------------------------------------------------------------
ESTIMATORS = ("deterministic", "supplemental_block")


def _rowfn(estimator):
    """Row function for an estimator: the preregistered deterministic gates, or the Falcon post-hoc block estimator
    (the Falcon supplemental statistical-impedance protocol)."""
    if estimator == "deterministic":
        return _bound_row
    if estimator == "supplemental_block":
        from . import cfd_supplemental as S
        return S.bound_supp_row
    raise TargetedError(f"unknown estimator {estimator!r}")


def _ctx(bdir, cr_path=None):
    bdir = Path(bdir)
    vs, fin, fem, rec = VC.load_bundle(bdir)
    p = Path(cr_path) if cr_path else bdir / "_cfd_result.json"
    if not p.exists():
        raise TargetedError("run `acoustic-liner vehicles cfd-bundle aggregate --bundle <bundle>` on the nominal results first")
    cr = json.load(open(p))
    V.verify_lineage(vs, [fin, fem, rec, cr])
    design = dict(CAN.canonical_design(fin["design"]))
    design["groups"] = [dict(g, cavity_volume=O2._group_cavity_volume(g)) for g in design["groups"]]
    ps = vs.to_physical_scenario(rec["velocity_source_corner"])
    cond = dict(spl_dB=list(ps.spl_dB), f_cov=float(fin["coverage"]), A_wall=float(ps.A_wall),
                kappa_nominal=float(ps.kappa_nominal), dOASPL_target=VP.TARGET_DB)
    nom = cr["groups"]["nominal"]
    rx = [[tuple(map(float, r)) for r in t] for t in nom["per_bank_RX"]]
    anc = [tuple(map(float, a)) for a in nom["per_bank_anchor"]]
    return dict(bdir=bdir, vs=vs, fin=fin, rec=rec, cr=cr, design=design, cond=cond, rx=rx, anc=anc)


def _fem(ctx, dR, dM, h):
    bfn = RL._nl_beta_fn(ctx["design"]["groups"], ctx["design"]["D"], np.asarray(dR, float), np.asarray(dM, float))
    r = VP.four_corner_fem(ctx["vs"], ctx["fin"], ctx["vs"].corners["primary"]["id"], ctx["vs"].payload_primary, h,
                           wall_beta_fn=bfn, stage="corrected_fem", parent=ctx["cr"])
    return float(r["robust"]["robust_value"])


def z_mag(j, u, r, x):
    """Total impedance magnitude |Z| = sqrt(R^2 + X^2) of the nominal row (prereg sec. 5)."""
    return math.hypot(r, x)


def _bound_row(case_dir, entry, geometry_hash):
    """Bind one returned case to its plan entry (case + input-tree hashes), post-process and validate it."""
    meta = json.load(open(case_dir / "_case_meta.json"))
    if CG.recompute_case_hash(meta) != entry["case_hash"] or CG.input_tree_hash(str(case_dir)) != entry["input_tree_hash"]:
        raise TargetedError(f"{case_dir}: case does not match its plan (edited)")
    row = POST.post_case(str(case_dir), n_periods=3)
    ok, rep = VAL.validate_cfd_row(row, expected_bank_id=entry["bank_id"], expected_case_hash=entry["case_hash"],
                                   expected_geometry_hash=geometry_hash)
    return row, ok, rep


def _plan_entry(root, group, name):
    man = RUN.load_manifest(str(Path(root) / group))
    for e in man["cases"]:
        if e["name"] == name:
            return e, man["geometry_hash"]
    raise TargetedError(f"{group}/{name} not in its plan")


def _sources(ctx):
    """name -> (case dir, plan entry, geometry hash) of the nominal evidence the cfd_result was built from: the
    bundle's nominal cases, with operating-point rerun cases substituted when the result came from a rerun merge."""
    bdir = ctx["bdir"]
    man = RUN.load_manifest(str(bdir / "nominal"))
    src = {e["name"]: (bdir / "nominal" / e["name"], e, man["geometry_hash"]) for e in man["cases"]}
    op = ctx["cr"].get("operating_point_rerun")
    if op:
        rdir = (bdir / op["rerun_dir"]).resolve()
        if VC._sha_file(rdir / "_rerun.json") != op["rerun_sha256"]:
            raise TargetedError("operating-point rerun bundle differs from the one merged into the cfd_result")
        rman = RUN.load_manifest(str(rdir / "nominal_24c"))
        for e in rman["cases"]:
            src[e["name"]] = (rdir / "nominal_24c" / e["name"], e, rman["geometry_hash"])
    return src


# ---- select ----------------------------------------------------------------------------------------------------
def select(bdir, out=None, cr_path=None, estimator="deterministic"):
    ctx = _ctx(bdir, cr_path)
    rowfn = _rowfn(estimator)
    bdir = ctx["bdir"]
    root = Path(out) if out else VP.RUN_DIR / ctx["vs"].name / "cfd_targeted"
    if (root / "_targeted_selection.json").exists():
        raise TargetedError("selection already made; it is immutable evidence")
    src = _sources(ctx)
    interval = FB.relabel_design_interval(ctx["design"], ctx["cond"], ctx["rx"], ctx["anc"])
    if interval["status"] != "complete":
        raise TargetedError(f"nominal relabel incomplete (banks {interval['banks_not_ok']})")
    fem_f = {k: _fem(ctx, interval[k]["dR_excess_Rayl"], interval[k]["dM_inertance_Rayl_s"], SELECTION_MESH)
             for k in FORCINGS}
    key = controlling_forcing(fem_f)
    leg = interval[key]
    n = len(ctx["rx"])
    base = _fem(ctx, np.zeros(n), np.zeros(n), SELECTION_MESH)
    deltas = []
    for j in range(n):
        dR, dM = np.zeros(n), np.zeros(n)
        dR[j], dM[j] = leg["dR_excess_Rayl"][j], leg["dM_inertance_Rayl_s"][j]
        deltas.append(abs(_fem(ctx, dR, dM, SELECTION_MESH) - base))
    j = pick_bank(deltas)
    u_star = float(leg["per_bank"][j]["U_star"])
    bank_id = f"b{j}"
    rows = []
    for nm, (cd, e, gh) in src.items():
        if e["bank_id"] != bank_id or abs(float(e["u_target"]) - RL.ANCHOR_U_TARGET) < 1e-9:
            continue
        row, ok, _ = rowfn(cd, e, gh)
        if ok:
            rows.append((nm, float(e["u_target"]), float(row["u_amp"])))
    if not rows:
        raise TargetedError(f"bank b{j} has no validated non-reference nominal case")
    picks = pick_velocities(rows, u_star)
    cases = protocol_cases(picks)
    plans = {}
    for c in cases:
        cd, e, gh = src[c["name"]]
        meta = json.load(open(cd / "_case_meta.json"))
        v = VARIANTS[c["group"]]
        _, m = CG.generate(str(root / c["group"] / c["name"]), bank_id=meta["bank_id"], d_mm=meta["d_mm"],
                           L_mm=meta["L_mm"], f0=meta["f0_Hz"], uneck=meta["uneck_m_s"], cycles=meta["cycles"],
                           geometry_hash=meta["geometry_hash"], refine=v["refine"], maxco=v["maxco"])
        entry = dict(name=c["name"], bank_id=m["bank_id"], u_target=m["uneck_m_s"], case_hash=m["case_hash"],
                     input_tree_hash=m["input_tree_hash"], refine=v["refine"], maxco=v["maxco"], cycles=meta["cycles"],
                     source_input_tree_hash=e["input_tree_hash"])
        if meta["cycles"] == ctx["rec"]["cycles"] and c["group"] in ctx["rec"]["groups"]:
            pe, _ = _plan_entry(bdir, c["group"], c["name"])           # must reproduce the bundle's own sens case
            if (pe["case_hash"], pe["input_tree_hash"]) != (m["case_hash"], m["input_tree_hash"]):
                raise TargetedError(f"{c['group']}/{c['name']}: regenerated case differs from the parent bundle's")
        c["cycles"] = meta["cycles"]
        plans.setdefault(c["group"], dict(geometry_hash=gh, cases=[]))["cases"].append(entry)
    for g, plan in plans.items():
        plan["n_cases"] = len(plan["cases"])
        (root / g / "_cfd_case_plan.json").write_text(json.dumps(plan, indent=2, default=float))
    VC._write_scripts(root, plans)
    _run_script(root, cases)
    sel = dict(schema=SC.schema("cfd-targeted-selection"), study="vehicle_cfd_targeted_selection", protocol=PROTOCOL,
               estimator=estimator, cfd_result_sha256=V.record_sha256(ctx["cr"]),
               bundle_sha256=V.record_sha256(ctx["rec"]), fem_by_forcing_0p2=fem_f, controlling_forcing=key,
               fem_uncorrected_0p2=base, bank_deltas_db=deltas, bank=j, u_star=u_star, candidates=rows, picks=picks,
               cases=cases, plan_sha256={g: VC._sha_file(root / g / "_cfd_case_plan.json") for g in plans},
               neck_resolution="identical block topology/cell counts for all banks; no extra bank case")
    (root / "_targeted_selection.json").write_text(json.dumps(sel, sort_keys=True, indent=1, default=float))
    print(f"{ctx['vs'].name}: forcing {key}, bank b{j} (|dFEM| {[round(d, 4) for d in deltas]}), U* {u_star:.3f}, "
          f"cases {[c['group'] + '/' + c['name'] for c in cases]}")
    return sel


def _run_script(root, cases):
    lines = ["#!/usr/bin/env bash", "# bash run_targeted.sh [parallel jobs, default 4]   (resumable)",
             'cd "$(dirname "$0")"', 'N="${1:-4}"',
             "printf '%s\\n' " + " ".join(f"{c['group']}/{c['name']}" for c in cases) + ' | xargs -P "$N" -n 1 bash run_one.sh',
             'echo "targeted cases finished; rerun to resume"']
    (Path(root) / "run_targeted.sh").write_text("\n".join(lines) + "\n", newline="\n")


# ---- amendment (seven-case selection made before prereg sec. 7) ------------------------------------------------
def _state(case_dir):
    """Run state only (done / failed / running / not started); never reads any numerical output."""
    p = Path(case_dir) / "_status.json"
    return json.load(open(p)).get("state", "unknown") if p.exists() else "not started"


def amend(tdir):
    tdir = Path(tdir)
    sel_p = tdir / "_targeted_selection.json"
    sel = json.load(open(sel_p))
    if sel.get("protocol", LEGACY_PROTOCOL) == PROTOCOL:
        raise TargetedError("selection already follows the five-case protocol")
    if (tdir / AMENDMENT).exists():
        raise TargetedError("amendment already recorded; it is immutable evidence")
    keep, drop = split_legacy(sel["cases"])
    st = {f"{c['group']}/{c['name']}": _state(tdir / c["group"] / c["name"]) for c in sel["cases"]}
    am = dict(schema=SC.schema("cfd-targeted-amendment"), study="vehicle_cfd_targeted_amendment", protocol=PROTOCOL,
              original_protocol=LEGACY_PROTOCOL, prereg_section=7,
              selection_sha256=VC._sha_file(sel_p), amended_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
              reason=AMENDMENT_REASON, retained=keep, dropped=drop, state_at_amendment=st,
              completed_before_amendment=sorted(k for k, v in st.items() if v == "done"),
              numerical_outputs_inspected=False,
              note="only run status and solver progress were read before the amendment; dropped cases are never "
                   "post-processed or used", label=LABEL, limitations=list(LIMITATIONS))
    (tdir / AMENDMENT).write_text(json.dumps(am, sort_keys=True, indent=1))
    _run_script(tdir, keep)
    print(f"amended: retained {[k['group'] + '/' + k['name'] for k in keep]}; "
          f"dropped {[d['group'] + '/' + d['name'] for d in drop]}")
    return am


def retained_cases(tdir):
    """(cases, amendment sha or None) of the five-case protocol; a seven-case selection needs its bound amendment."""
    tdir = Path(tdir)
    sel_p = tdir / "_targeted_selection.json"
    sel = json.load(open(sel_p))
    if sel.get("protocol", LEGACY_PROTOCOL) == PROTOCOL:
        return sel["cases"], None
    am_p = tdir / AMENDMENT
    if not am_p.exists():
        raise TargetedError("seven-case selection without the five-case amendment (prereg sec. 7)")
    am = json.load(open(am_p))
    SC.record_version(am, "cfd-targeted-amendment")
    if am["selection_sha256"] != VC._sha_file(sel_p) or am["protocol"] != PROTOCOL:
        raise TargetedError("amendment does not bind this selection")
    if am["retained"] != split_legacy(sel["cases"])[0]:
        raise TargetedError("amendment retained cases differ from the protocol")
    return am["retained"], VC._sha_file(am_p)


# ---- aggregate ---------------------------------------------------------------------------------------------------
def aggregate(bdir, tdir, cr_path=None):
    ctx = _ctx(bdir, cr_path)
    tdir = Path(tdir)
    sel = json.load(open(tdir / "_targeted_selection.json"))
    rowfn = _rowfn(sel.get("estimator", "deterministic"))
    if sel["cfd_result_sha256"] != V.record_sha256(ctx["cr"]) or sel["bundle_sha256"] != V.record_sha256(ctx["rec"]):
        raise TargetedError("selection belongs to another bundle / nominal result")
    for g, h in sel["plan_sha256"].items():
        if VC._sha_file(tdir / g / "_cfd_case_plan.json") != h:
            raise TargetedError(f"{g} plan was modified")
    cases, am_sha = retained_cases(tdir)
    src = _sources(ctx)
    j, key = sel["bank"], sel["controlling_forcing"]
    per_case, bad = [], []
    for c in cases:
        e, gh = _plan_entry(tdir, c["group"], c["name"])
        row, ok, rep = rowfn(tdir / c["group"] / c["name"], e, gh)
        if not ok:
            bad.append(dict(case=f"{c['group']}/{c['name']}", report=rep)); continue
        ncd, ne, ngh = src[c["name"]]
        if ne["input_tree_hash"] != e["source_input_tree_hash"]:
            raise TargetedError(f"{c['name']}: nominal source differs from the one the variant was built from")
        nrow, nok, _ = rowfn(ncd, ne, ngh)
        if not nok:
            raise TargetedError(f"nominal/{c['name']} is not valid")
        u, r0, x0 = float(nrow["u_amp"]), float(nrow["R_phasor"]), float(nrow["X_phasor"])
        r1, x1 = float(row["R_phasor"]), float(row["X_phasor"])
        z = z_mag(j, u, r0, x0)
        per_case.append(dict(case=f"{c['group']}/{c['name']}", slot=c["slot"], u_amp_nominal=u,
                             u_amp_variant=float(row["u_amp"]), R_nominal=r0, X_nominal=x0, R_variant=r1, X_variant=x1,
                             dR=r1 - r0, dX=x1 - x0, rel_dR=(r1 - r0) / abs(r0) if r0 else None,
                             rel_dX=(x1 - x0) / abs(x0) if x0 else None, z_total_mag=z,
                             norm_dR=abs(r1 - r0) / z, norm_dX=abs(x1 - x0) / z, _row=(float(row["u_amp"]), r1, x1),
                             _nom=(u, r0, x0)))
    if bad:
        (tdir / FAILURE).write_text(json.dumps(dict(selection_sha256=VC._sha_file(tdir / "_targeted_selection.json"),
                                                    invalid_cases=[dict(case=x["case"], failed=x["report"].get("failed"))
                                                                   for x in bad]), sort_keys=True, indent=1))
        raise TargetedError(f"invalid targeted cases (fail closed): {bad}")
    eR = max(p["norm_dR"] for p in per_case)
    eX = max(p["norm_dX"] for p in per_case)
    forcing = FORCINGS[key]
    nom_leg = FB.relabel_design(ctx["design"], ctx["cond"], ctx["rx"], ctx["anc"], forcing=forcing)
    u0 = float(nom_leg["per_bank"][j]["U_star"])
    shifts = []
    for p in per_case:
        if p["slot"] != "ustar":
            continue
        rx = [list(t) for t in ctx["rx"]]
        rx[j] = [p["_row"] if abs(t[0] - p["_nom"][0]) < 1e-12 and abs(t[1] - p["_nom"][1]) < 1e-12 else t for t in rx[j]]
        leg = FB.relabel_design(ctx["design"], ctx["cond"], rx, ctx["anc"], forcing=forcing)
        us = leg["per_bank"][j].get("U_star")
        shifts.append(dict(case=p["case"], U_star=us, rel_shift=(abs(us - u0) / u0 if us is not None else None)))
    fem_nom = _fem(ctx, nom_leg["dR_excess_Rayl"], nom_leg["dM_inertance_Rayl_s"], RESULT_MESH)
    combos = []
    for sR, sX in itertools.product((1, -1), (1, -1)):
        rx = perturb_tables(ctx["rx"], ctx["anc"], z_mag, eR, eX, sR, sX)
        leg = FB.relabel_design(ctx["design"], ctx["cond"], rx, ctx["anc"], forcing=forcing)
        val = (_fem(ctx, leg["dR_excess_Rayl"], leg["dM_inertance_Rayl_s"], RESULT_MESH)
               if leg["status"] == "complete" else None)
        combos.append(dict(sign_R=sR, sign_X=sX, relabel_status=leg["status"], fem_robust_db=val))
    vals = [fem_nom] + [c["fem_robust_db"] for c in combos if c["fem_robust_db"] is not None]
    incomplete = [c for c in combos if c["fem_robust_db"] is None]
    rng = max(vals) - min(vals)
    ust = [s["rel_shift"] for s in shifts if s["rel_shift"] is not None]
    trig = triggers(fem_range_db=rng, max_norm_change=max(eR, eX), ustar_shift=max(ust) if ust else None,
                    verdicts=[v >= VP.TARGET_DB for v in vals])
    if incomplete or any(s["U_star"] is None for s in shifts):
        trig.append("relabel incomplete under a perturbation (fixed point lost)")
    for p in per_case:
        p.pop("_row"); p.pop("_nom")
    # a version-1 selection named the pre-registration and its hash; the result carries them through unchanged
    v1 = SC.record_version(sel, "cfd-targeted-selection") == 1
    out = dict(**(dict(prereg=sel["prereg"], prereg_sha256=sel["prereg_sha256"]) if v1
                  else dict(schema=SC.schema("cfd-targeted-result"))),
               study="vehicle_cfd_targeted_sensitivity", selection_sha256=VC._sha_file(tdir / "_targeted_selection.json"), wording=ENVELOPE_WORDING,
               protocol=PROTOCOL, amendment_sha256=am_sha, estimator=sel.get("estimator", "deterministic"), cases_used=[f"{c['group']}/{c['name']}" for c in cases],
               label=LABEL, limitations=list(LIMITATIONS),
               controlling_forcing=key, bank=j, per_case=per_case, envelope=dict(norm_dR=eR, norm_dX=eX),
               u_star_nominal=u0, u_star_shifts=shifts, fem_nominal_0p1=fem_nom, fem_envelope_combos_0p1=combos,
               fem_range_db=[min(vals), max(vals)], fem_range_width_db=rng, target_db=VP.TARGET_DB,
               escalation_triggers=trig, escalate=bool(trig))
    (tdir / "_targeted_result.json").write_text(json.dumps(out, sort_keys=True, indent=1, default=float))
    print(f"{ctx['vs'].name}: envelope R {eR:.3f} X {eX:.3f} of |z|; corrected FEM {min(vals):.3f}..{max(vals):.3f} dB "
          f"(width {rng:.3f}); triggers: {trig or 'none'}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="acoustic-liner vehicles cfd-targeted")
    p.add_argument("cmd", choices=("select", "amend", "aggregate"))
    p.add_argument("--bundle")
    p.add_argument("--targeted", help="cfd_targeted directory (amend, aggregate)")
    a = p.parse_args(argv)
    if a.cmd != "amend" and not a.bundle:
        p.error(f"{a.cmd} needs --bundle")
    if a.cmd == "amend":
        if not a.targeted:
            p.error("amend needs --targeted")
        amend(a.targeted)
    elif a.cmd == "select":
        select(a.bundle)
    else:
        if not a.targeted:
            p.error("aggregate needs --targeted")
        aggregate(a.bundle, a.targeted)


if __name__ == "__main__":
    main()
