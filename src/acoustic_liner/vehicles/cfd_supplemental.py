# -*- coding: utf-8 -*-
"""Falcon supplemental analysis: statistically unsteady neck resistance
(protocol `FALCON-SUPPLEMENTAL-STATISTICAL-IMPEDANCE-PROTOCOL`, kept in the private archive). POST-HOC and separately declared; the preregistered
Falcon verdict stays UNRESOLVED and is never edited.

    acoustic-liner vehicles cfd-supplemental estimate --bundle DIR --rerun RDIR   (block estimates -> table)
    acoustic-liner vehicles cfd-supplemental select   --bundle DIR                (five-case targeted set)
    bash run_targeted.sh [N]                                                         (WSL; resumable)
    acoustic-liner vehicles cfd-supplemental aggregate --bundle DIR               (targeted, block estimator)
    acoustic-liner vehicles cfd-supplemental decide   --bundle DIR                (interval -> FEM -> outcome)
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import stats

from ..provenance import schemas as SC
from . import cfd_bundle as VC
from . import cfd_rerun as VR
from . import search as VP
from . import scenarios as V
from ..cfd import case_generation as CG
from ..cfd import postprocess as POST
from ..cfd import impedance_correction as RL
from ..cfd import runner as RUN
from ..cfd import validation as VAL

# the protocol's name, as records carry it (the document itself is kept privately)
PROTOCOL = "docs/FALCON-SUPPLEMENTAL-STATISTICAL-IMPEDANCE-PROTOCOL.md"
VEHICLE = "falcon"
DISCARD_FRACTION = 1.0 / 3.0
RHO_MAX = 0.95
CONF = 0.95
TREND_MAX, SHIFT_MAX, HALFWIDTH_MAX = 0.05, 0.05, 0.10
SUPPORTED = RUN.VALID                 # "passes the supplemental gates" (protocol sec. 4)
NOT_SUPPORTED = "supplemental_failed"
DRIFT_CHECKS = {"R_converged", "X_converged"}
PASS, FAIL, UNRESOLVED = "SUPPLEMENTAL ROBUST PASS", "SUPPLEMENTAL FAIL", "SUPPLEMENTAL UNRESOLVED"
OUT = "supplemental_statistical"


class SupplementalError(RuntimeError):
    pass


# ---- pure statistics (protocol sec. 2-3; unit-tested) -------------------------------------------------------
def n_discard(n_cycles):
    return math.ceil(n_cycles * DISCARD_FRACTION - 1e-9)


def ess_halfwidth(x):
    """(half-width of the 95 % interval on the mean, n_eff, rho) with a lag-1 autocorrelation-adjusted n_eff."""
    x = np.asarray(x, float)
    n = x.size
    d = x - x.mean()
    den = float(np.sum(d * d))
    rho = float(np.sum(d[:-1] * d[1:]) / den) if den > 0 else 0.0
    rho = min(max(rho, 0.0), RHO_MAX)
    n_eff = max(2.0, n * (1 - rho) / (1 + rho))
    s = float(np.std(x, ddof=1)) if n > 1 else float("inf")
    return float(stats.t.ppf(0.5 + CONF / 2, n_eff - 1) * s / math.sqrt(n_eff)), n_eff, rho


def block_stats(Zk, Uk):
    """Zk: complex block impedances (one per post-transient cycle); Uk: complex velocity phasors."""
    Zk, Uk = np.asarray(Zk, complex), np.asarray(Uk, complex)
    n = Zk.size
    zbar = complex(Zk.mean())
    zmag = abs(zbar)
    k = np.arange(n)
    trend_R = abs(np.polyfit(k, Zk.real, 1)[0]) * (n - 1) / zmag
    trend_X = abs(np.polyfit(k, Zk.imag, 1)[0]) * (n - 1) / zmag
    shift = abs(Zk[: n // 2].mean() - Zk[n // 2:].mean()) / zmag
    hR, neR, rhoR = ess_halfwidth(Zk.real)
    hX, neX, rhoX = ess_halfwidth(Zk.imag)
    return dict(n_blocks=n, R=zbar.real, X=zbar.imag, z_mag=zmag, u_amp=float(np.abs(Uk).mean()),
                h_R=hR, h_X=hX, n_eff_R=neR, n_eff_X=neX, rho_R=rhoR, rho_X=rhoX, trend_R=float(trend_R),
                trend_X=float(trend_X), half_shift=float(shift), R_blocks=Zk.real.tolist(), X_blocks=Zk.imag.tolist())


def gates(st, thd_p, thd_u, other_checks):
    g = dict(trend=st["trend_R"] < TREND_MAX and st["trend_X"] < TREND_MAX,
             stable_means=st["half_shift"] < SHIFT_MAX,
             uncertainty=st["h_R"] <= HALFWIDTH_MAX * st["z_mag"] and st["h_X"] <= HALFWIDTH_MAX * st["z_mag"],
             distortion=thd_p <= VAL.THD_MAX and thd_u <= VAL.THD_MAX,
             other_validator_checks=all(other_checks.values()))
    return g


# ---- per case ---------------------------------------------------------------------------------------------------
def block_estimate(case_dir, analysis_cycles=None):
    """Block estimate over the post-transient window; `analysis_cycles` pins the window to the last n cycles
    (follow-up protocol sec. 4: equal windows for a continued pair)."""
    p = POST.load_case_params(case_dir)
    f0, N = p["f0"], int(p["meta"]["cycles"])
    n = int(analysis_cycles) if analysis_cycles else N - n_discard(N)
    tu, dp, u = POST._uniform(case_dir, p["a_neck"], f0, n)
    m = tu.size // n
    Zk, Uk = [], []
    for b in range(n):
        s = slice(b * m, (b + 1) * m)
        t, d, v = tu[s], dp[s] - dp[s].mean(), u[s] - u[s].mean()
        P, U = POST._phasor(t, d, f0), POST._phasor(t, v, f0)
        Zk.append(P / U); Uk.append(U)
    st = block_stats(Zk, Uk)
    return dict(st, cycles=N, discarded=N - n, thd_p=float(POST._thd(tu, dp, f0)), thd_u=float(POST._thd(tu, u, f0)))


def supp_row(case_dir, entry, geometry_hash, analysis_cycles=None):
    """(row, ok, report) under the supplemental gates; row carries u_amp / R_phasor / X_phasor = block means."""
    det = POST.post_case(str(case_dir), n_periods=3)
    _, rep = VAL.validate_cfd_row(det, expected_bank_id=entry["bank_id"], expected_case_hash=entry["case_hash"],
                                  expected_geometry_hash=geometry_hash)
    other = {k: bool(v) for k, v in rep["checks"].items() if k not in DRIFT_CHECKS}
    try:
        est = block_estimate(str(case_dir), analysis_cycles)
    except (POST.ProbeError, ValueError, ZeroDivisionError) as e:
        return dict(det, supplemental_error=str(e)), False, dict(checks=other, gates=None, error=str(e))
    g = gates(est, est["thd_p"], est["thd_u"], other)
    ok = all(g.values()) and all(math.isfinite(est[k]) for k in ("R", "X", "u_amp", "h_R", "h_X"))
    row = dict(bank_id=det["bank_id"], case_hash=det["case_hash"], geometry_hash=det["geometry_hash"],
               u_target=det["u_target"], u_amp=est["u_amp"], R_phasor=est["R"], X_phasor=est["X"],
               h_R=est["h_R"], h_X=est["h_X"], deterministic=dict(R_drift=det["R_drift"], X_drift=det["X_drift"],
                                                                  R_phasor=det["R_phasor"], X_phasor=det["X_phasor"]))
    return row, ok, dict(checks=other, gates=g, estimate=est, failed=sorted(k for k, v in g.items() if not v))


def _bind(case_dir, entry):
    meta = json.load(open(Path(case_dir) / "_case_meta.json"))
    if CG.recompute_case_hash(meta) != entry["case_hash"] or CG.input_tree_hash(str(case_dir)) != entry["input_tree_hash"]:
        raise SupplementalError(f"{case_dir}: case does not match its plan (edited)")


def bound_supp_row(case_dir, entry, geometry_hash):
    """Plan-bound variant of supp_row (the targeted tool's row function)."""
    _bind(case_dir, entry)
    return supp_row(case_dir, entry, geometry_hash)


# ---- nominal table + bank support (protocol sec. 4) ----------------------------------------------------------
def _root(vs):
    return VP.RUN_DIR / vs.name / OUT


def estimate(bdir, rdir, out=None, vehicles=(VEHICLE,), protocol=PROTOCOL, name="_supp_cfd_result.json"):
    bdir, rdir = Path(bdir), Path(rdir)
    vs, fin, fem, rec = VC.load_bundle(bdir)
    if rec.get("vehicle") not in vehicles:
        raise SupplementalError(f"{protocol} is declared for {vehicles} only")
    rm_sha = VR._verify_returned(bdir, rec)
    doc = json.load(open(rdir / "_rerun.json"))
    rr_p = rdir / "_returned_manifest.json"
    if doc["parent_bundle_sha256"] != V.record_sha256(rec) or not rr_p.exists():
        raise SupplementalError("rerun bundle does not belong to this bundle or was never merged")
    for k, h in json.load(open(rr_p))["files"].items():
        if VC._sha_file(rdir / k) != h:
            raise SupplementalError(f"rerun result changed after merge: {k}")
    man = VR.merged_manifest(bdir, rdir, doc)
    base = bdir / "nominal"
    n_banks = len(RUN._manifest_bank_indices(man))
    per_bank, anchors, halfw = [[] for _ in range(n_banks)], [None] * n_banks, [[] for _ in range(n_banks)]
    cases, states = [], {}
    for e in man["cases"]:
        cd = Path(e["name"]) if Path(e["name"]).is_absolute() else base / e["name"]
        if not RUN._case_ran(str(cd)):
            states[e["name"]] = RUN.MISSING
            cases.append(dict(name=Path(e["name"]).name, state=RUN.MISSING)); continue
        _bind(cd, e)
        row, ok, rep = supp_row(cd, e, man["geometry_hash"])
        states[e["name"]] = SUPPORTED if ok else NOT_SUPPORTED
        cases.append(dict(name=Path(e["name"]).name, bank_id=e["bank_id"], u_target=float(e["u_target"]),
                          state=states[e["name"]], row=row, report=rep))
        if ok:
            j = int(str(e["bank_id"]).lstrip("b"))
            rx = (float(row["u_amp"]), float(row["R_phasor"]), float(row["X_phasor"]))
            per_bank[j].append(rx); halfw[j].append((rx[0], float(row["h_R"]), float(row["h_X"])))
            if abs(float(e["u_target"]) - RL.ANCHOR_U_TARGET) < 1e-9:
                anchors[j] = rx
    design, cond = VR._design_cond(vs, fin, rec)
    banks, iv = {}, None
    if all(per_bank) and all(a is not None for a in anchors):
        from ..cfd import feedback as FB
        iv = FB.relabel_design_interval(design, cond, per_bank, anchors)
        for j in range(n_banks):
            us = [iv[k]["per_bank"][j].get("U_star") for k in ("broadband", "resonant_tone")]
            if any(u is None for u in us):
                banks[f"b{j}"] = dict(status="unresolved"); continue
            planned = [(c["name"], float(c["u_target"]), states[c["name"]]) for c in man["cases"]
                       if c["bank_id"] == f"b{j}" and abs(float(c["u_target"]) - RL.ANCHOR_U_TARGET) > 1e-9]
            banks[f"b{j}"] = VR.bracket(planned, min(us), max(us))
    else:
        banks = {f"b{j}": dict(status="no supported row or anchor") for j in range(n_banks)
                 if not per_bank[j] or anchors[j] is None}
    unresolved = sorted(b for b, v in banks.items() if v.get("status") != "adequate") if banks else ["all"]
    root = Path(out) if out else _root(vs)
    root.mkdir(parents=True, exist_ok=True)
    res = dict(schema=SC.schema("statistical-impedance-estimate"),
               study=("falcon_supplemental_statistical_impedance" if protocol == PROTOCOL else "statistical_impedance_table"),
               protocol=protocol, post_hoc=True,
               groups=dict(nominal=dict(per_bank_RX=per_bank, per_bank_anchor=anchors, n_banks=n_banks)),
               per_bank_halfwidth=halfw, cases=cases, banks=banks, unresolved_banks=unresolved,
               central_interval=(iv or {}).get("dOASPL_nonlinear_interval"),
               operating_point_rerun=dict(rerun_sha256=VC._sha_file(rdir / "_rerun.json"),
                                          rerun_returned_sha256=VC._sha_file(rr_p),
                                          rerun_dir=os.path.relpath(rdir, bdir)),
               groups_not_run=[g for g in rec["groups"] if g != "nominal"], source_corners=rec["source_corners"],
               payload_cases=rec["payload_cases"], returned_manifest_sha256=rm_sha,
               bundle_dir=os.path.relpath(bdir, root))
    res = V.stamp(res, {k: rec[k] for k in V.IDENTITY_KEYS}, "cfd_result", rec)
    p = root / name
    if p.exists() and json.load(open(p)) != json.loads(json.dumps(res, default=float)):
        raise SupplementalError(f"{p} exists with different content; it is immutable evidence")
    p.write_text(json.dumps(res, sort_keys=True, indent=1, default=float))
    n_ok = sum(c["state"] == SUPPORTED for c in cases)
    print(f"{vs.name}: supplemental estimator {n_ok}/{len(cases)} cases supported; banks "
          f"{ {b: v.get('status') for b, v in banks.items()} }; central interval {res['central_interval']}")
    return res


# ---- supplemental targeted sensitivity (protocol sec. 6) -------------------------------------------------------
def _supp(bdir):
    vs, _, _, _ = VC.load_bundle(Path(bdir))
    root = _root(vs)
    res = json.load(open(root / "_supp_cfd_result.json"))
    if res["unresolved_banks"]:
        raise SupplementalError(f"{UNRESOLVED}: banks without supplemental support {res['unresolved_banks']}")
    return root, res


def select_targeted(bdir):
    from . import cfd_targeted as VT
    root, _ = _supp(bdir)
    return VT.select(bdir, out=root / "cfd_targeted", cr_path=root / "_supp_cfd_result.json",
                     estimator="supplemental_block")


def aggregate_targeted(bdir):
    from . import cfd_targeted as VT
    root, _ = _supp(bdir)
    return VT.aggregate(bdir, root / "cfd_targeted", cr_path=root / "_supp_cfd_result.json")


# ---- interval propagation (protocol sec. 5) ---------------------------------------------------------------------
TABLES = ("central", "R+X+", "R+X-", "R-X+", "R-X-")
FORCINGS = ("broadband", "resonant_tone")


def perturb(per_bank, halfw, anchors, sR, sX):
    """Shift every non-anchor row by sign * its own 95 % half-width; the anchor keeps its estimate."""
    out = []
    for tab, hw, anc in zip(per_bank, halfw, anchors):
        rows = []
        for (u, r, x), (u2, hR, hX) in zip(tab, hw):
            if abs(u - u2) > 1e-12:
                raise SupplementalError("half-width table out of step with the impedance table")
            is_anchor = anc is not None and abs(u - anc[0]) < 1e-12 and abs(r - anc[1]) < 1e-12
            rows.append((u, r, x) if is_anchor else (u, r + sR * hR, x + sX * hX))
        out.append(rows)
    return out


def table_set(res):
    nom = res["groups"]["nominal"]
    rx = [[tuple(map(float, r)) for r in t] for t in nom["per_bank_RX"]]
    anc = [tuple(map(float, a)) for a in nom["per_bank_anchor"]]
    hw = [[tuple(map(float, r)) for r in t] for t in res["per_bank_halfwidth"]]
    signs = {"R+X+": (1, 1), "R+X-": (1, -1), "R-X+": (-1, 1), "R-X-": (-1, -1)}
    return anc, {"central": rx, **{k: perturb(rx, hw, anc, *s) for k, s in signs.items()}}


def supp_interval(fem, s_down, s_up):
    """fem: {table: {forcing: {mesh: value}}} over the five tables (protocol sec. 7); None when any value missing."""
    try:
        vals = [{h: float(fem[t][f][h]) for h in VP.MESH_SIZES} for t in TABLES for f in FORCINGS]
    except (KeyError, TypeError, ValueError):
        return None
    fine = min(VP.MESH_SIZES)
    f01, f01_hi = min(v[fine] for v in vals), max(v[fine] for v in vals)
    m = max(abs(v[h] - v[fine]) for v in vals for h in VP.MESH_SIZES if h != fine)
    return dict(F01=f01, F01_hi=f01_hi, M=m, S_down=s_down, S_up=s_up, L=f01 - m - s_down, U=f01_hi + m + s_up)


def outcome(iv, reasons, target=VP.TARGET_DB):
    if reasons or iv is None:
        return UNRESOLVED
    return PASS if iv["L"] > target else (FAIL if iv["U"] < target else UNRESOLVED)


def propagate(bdir, root=None, res=None, protocol=PROTOCOL, out_name="_supp_fem.json"):
    """Fixed point + corrected FEM for the five tables (primary corner / payload, three meshes, both forcings), and
    the central table at every corner x payload (reported only)."""
    from ..cfd import feedback as FB
    bdir = Path(bdir)
    vs, fin, fem, rec = VC.load_bundle(bdir)
    if root is None:
        root, res = _supp(bdir)
    V.verify_lineage(vs, [fin, fem, rec, res])
    design, cond = VR._design_cond(vs, fin, rec)
    anc, tables = table_set(res)
    prim = (vs.corners["primary"]["id"], vs.payload_primary)
    out = dict(study=("falcon_supplemental_fem" if protocol == PROTOCOL else "statistical_impedance_fem"), protocol=protocol,
               supp_cfd_result_sha256=V.record_sha256(res),
               relabel={}, fem={}, reported_central={}, lost_fixed_point=[])
    for t, rx in tables.items():
        iv = FB.relabel_design_interval(design, cond, rx, anc)
        out["relabel"][t] = dict(interval=iv.get("dOASPL_nonlinear_interval"),
                                 status={f: iv[f]["status"] for f in FORCINGS},
                                 U_star={f: [b.get("U_star") for b in iv[f]["per_bank"]] for f in FORCINGS})
        for f in FORCINGS:
            leg = iv[f]
            if leg["status"] != "complete":
                out["lost_fixed_point"].append(f"{t}/{f}")
                continue
            bfn = RL._nl_beta_fn(design["groups"], design["D"], np.array(leg["dR_excess_Rayl"], float),
                                 np.array(leg["dM_inertance_Rayl_s"], float))
            pairs = [prim] if t != "central" else [(c["id"], p) for c in vs.corners.values() for p in vs.payload_cases]
            for corner, payload in pairs:
                for h in VP.MESH_SIZES:
                    v = float(VP.four_corner_fem(vs, fin, corner, payload, h, wall_beta_fn=bfn, stage="corrected_fem",
                                                 parent=res)["robust"]["robust_value"])
                    if (corner, payload) == prim:
                        out["fem"].setdefault(t, {}).setdefault(f, {})[str(h)] = v
                    else:
                        out["reported_central"].setdefault(f"{corner}|{payload}", {}).setdefault(f, {})[str(h)] = v
    (Path(root) / out_name).write_text(json.dumps(out, sort_keys=True, indent=1, default=float))
    print(f"{vs.name}: supplemental FEM written; lost fixed points {out['lost_fixed_point'] or 'none'}")
    return out


# ---- decision + publication (protocol sec. 7-8) -----------------------------------------------------------------
WORDING = dict(
    supplemental=("The original deterministic cycle-convergence criterion was not met, so the preregistered Falcon "
                  "verdict remained unresolved. A separately declared post-hoc analysis treated the resistance as "
                  "statistically unsteady and estimated a block-averaged complex impedance with uncertainty."),
    not_prereg="Post-hoc and separately declared; not preregistered; no deterministic periodic convergence is claimed",
    normalisation="The trend and stability gates are normalised by |Z|, not by R; relative to R alone they are looser "
                  "than the deterministic 5 % drift gate",
    targeted="Targeted operating-region mesh/time-step sensitivity only; " + "; ".join(
        ("high-speed mesh sensitivity is not quantified", "mesh-time-step interaction is not quantified",
         "the measured envelope is not a proven global bound over the entire velocity range")),
    basis="Simulation-based (oracle search, payload-resolving FEM, OpenFOAM neck CFD); not experimentally validated "
          "and not flight-qualified",
    a_other="A_other is an assumed effective background-loss term that may include payload-associated losses; it is "
            "not measured payload absorption",
    audit="FEM-transfer audit limitation: transfer error is larger and less predictable for some partial-coverage "
          "designs")


def collect(bdir):
    from . import cfd_targeted as VT
    from . import verdict as VV
    bdir = Path(bdir)
    reasons, ev = [], {}

    def check(c, msg):
        if not c:
            reasons.append(msg)
        return c

    vs, fin, fem, rec = VC.load_bundle(bdir)
    root = _root(vs)
    try:
        ev["source"] = VC.preflight()
    except Exception as e:                                   # noqa: BLE001
        reasons.append(f"source tree not clean: {e}")
    prim_v = VP.RUN_DIR / vs.name / "verdict" / "verdict.json"
    if check(prim_v.exists(), "primary (preregistered) verdict not published"):
        pv = json.load(open(prim_v))
        ev["primary_verdict"] = dict(outcome=pv["outcome"], sha256=VC._sha_file(prim_v))
        check(pv["outcome"] == "UNRESOLVED", "primary verdict is not the frozen UNRESOLVED one")
    res_p, fem_p = root / "_supp_cfd_result.json", root / "_supp_fem.json"
    if not check(res_p.exists() and fem_p.exists(), "supplemental CFD result / FEM missing"):
        return ev, reasons, None, None
    res, sf = json.load(open(res_p)), json.load(open(fem_p))
    try:
        SC.record_version(res, "statistical-impedance-estimate")
    except SC.SchemaError as e:
        check(False, f"supplemental estimate format: {e}")
    try:
        V.verify_lineage(vs, [fin, fem, rec, res])
    except Exception as e:                                   # noqa: BLE001
        reasons.append(f"lineage broken: {e}")
    check(not res["unresolved_banks"], f"banks without supplemental support: {res['unresolved_banks']}")
    check(sf["supp_cfd_result_sha256"] == V.record_sha256(res), "supplemental FEM belongs to another result")
    check(not sf["lost_fixed_point"], f"fixed point lost under the impedance interval: {sf['lost_fixed_point']}")
    ev.update(supp_cfd_result_sha256=V.record_sha256(res), supp_fem_sha256=VC._sha_file(fem_p),
              protocol=PROTOCOL, **({"protocol_sha256": res["protocol_sha256"]} if "protocol_sha256" in res else {}),
              cases=dict(supported=[c["name"] for c in res["cases"] if c["state"] == SUPPORTED],
                         not_supported=[dict(name=c["name"], failed=(c.get("report") or {}).get("failed"))
                                        for c in res["cases"] if c["state"] != SUPPORTED]),
              banks={b: {k: v.get(k) for k in ("U_star_lo", "U_star_hi", "valid", "status")}
                     for b, v in res["banks"].items()})
    tdir = root / "cfd_targeted"
    sel_p, tr_p = tdir / "_targeted_selection.json", tdir / "_targeted_result.json"
    s = None
    if not tr_p.exists() and (tdir / VT.FAILURE).exists():
        fl = json.load(open(tdir / VT.FAILURE))
        ev["targeted_failure"] = dict(sha256=VC._sha_file(tdir / VT.FAILURE), invalid_cases=fl["invalid_cases"])
        reasons.append(f"supplemental targeted sensitivity failed the block gates: {fl['invalid_cases']}")
    if check(sel_p.exists() and tr_p.exists(), "supplemental targeted sensitivity selection / result missing"):
        sel, tr = json.load(open(sel_p)), json.load(open(tr_p))
        try:
            SC.record_version(sel, "cfd-targeted-selection"), SC.record_version(tr, "cfd-targeted-result")
        except SC.SchemaError as e:
            check(False, f"targeted record format: {e}")
        check(sel["cfd_result_sha256"] == V.record_sha256(res), "targeted selection made on another result")
        check(sel.get("estimator") == "supplemental_block" and tr.get("estimator") == "supplemental_block",
              "targeted sensitivity not analysed with the block estimator")
        check(tr["selection_sha256"] == VC._sha_file(sel_p), "targeted result does not match its selection")
        check(tr.get("protocol") == VT.PROTOCOL, "targeted result not under the five-case protocol")
        check(not tr.get("escalate"), f"targeted escalation triggered and not completed: {tr.get('escalation_triggers')}")
        for g, h in sel["plan_sha256"].items():
            check(VC._sha_file(tdir / g / "_cfd_case_plan.json") == h, f"targeted plan {g} changed")
        builds = {VV._openfoam_build(tdir / c["group"] / c["name"]) for c in sel["cases"]}
        check(len(builds) == 1 and None not in builds, f"OpenFOAM builds disagree or missing: {builds}")
        ev["targeted"] = dict(selection_sha256=VC._sha_file(sel_p), result_sha256=VC._sha_file(tr_p),
                              bank=tr["bank"], controlling_forcing=tr["controlling_forcing"],
                              cases_used=tr.get("cases_used"), envelope=tr["envelope"],
                              fem_range_db=tr["fem_range_db"], openfoam_builds=sorted(map(str, builds)))
        s = VV.sensitivity_allowance(tr)
    return ev, reasons, sf, s


def build(bdir):
    ev, reasons, sf, s = collect(bdir)
    vs, *_ = VC.load_bundle(Path(bdir))
    fem = {t: {f: {float(h): v for h, v in m.items()} for f, m in d.items()} for t, d in (sf or {}).get("fem", {}).items()}
    iv = supp_interval(fem, *s) if s else None
    v = dict(schema=SC.schema("falcon-supplemental-verdict"), vehicle=VEHICLE, scenario_name=vs.name,
             outcome=outcome(iv, reasons), primary_outcome=(ev.get("primary_verdict") or {}).get("outcome"),
             target_db=VP.TARGET_DB, interval=iv, fem_primary=fem,
             reported_not_verdict=(sf or {}).get("reported_central"), reasons=reasons, evidence=ev, wording=WORDING,
             protocol=PROTOCOL, post_hoc=True)
    v["record_sha256"] = V.record_sha256(json.loads(json.dumps(v, default=float)))
    return v


def _f(x, n=3):
    return "n/a" if x is None else f"{x:.{n}f}"


def render(v):
    iv = v.get("interval") or {}
    res = {PASS: "found that the CFD-corrected FEM remained above target across the resulting impedance interval",
           FAIL: "found that the CFD-corrected FEM stayed below target across the resulting impedance interval",
           UNRESOLVED: "could not establish a robust result"}[v["outcome"]]
    ev, t = v["evidence"], v["evidence"].get("targeted", {})
    lines = [f"# Falcon supplemental verdict (post-hoc): **{v['outcome']}**", "",
             f"Primary preregistered verdict: **{v['primary_outcome']}** (unchanged). This supplemental result is "
             f"post-hoc and separately declared ({v['protocol']}); it is not preregistered.", "",
             v["wording"]["supplemental"][:-1] + f", and {res}.", "",
             "| quantity | dB |", "|---|---|",
             f"| F01 (min over 5 impedance tables x 2 forcings, 0.1 m) | {_f(iv.get('F01'))} |",
             f"| F01_hi | {_f(iv.get('F01_hi'))} |", f"| mesh allowance M | {_f(iv.get('M'))} |",
             f"| targeted sensitivity S_down / S_up | {_f(iv.get('S_down'))} / {_f(iv.get('S_up'))} |",
             f"| **L** | **{_f(iv.get('L'))}** |", f"| **U** | **{_f(iv.get('U'))}** |",
             f"| target | {v['target_db']} |", ""]
    if v["reasons"]:
        lines += ["## Why not resolved", ""] + [f"- {r}" for r in v["reasons"]] + [""]
    lines += ["## Evidence", "",
              f"- Supplemental CFD result `{ev.get('supp_cfd_result_sha256')}`; FEM `{ev.get('supp_fem_sha256')}`; "
              f"protocol `{ev.get('protocol_sha256')}`; primary verdict `{(ev.get('primary_verdict') or {}).get('sha256')}`.",
              f"- Cases supported: {len((ev.get('cases') or {}).get('supported', []))}; not supported: "
              f"{[c['name'] for c in (ev.get('cases') or {}).get('not_supported', [])]}.",
              f"- Targeted: bank b{t.get('bank')}, forcing {t.get('controlling_forcing')}, cases {t.get('cases_used')}, "
              f"FEM range {t.get('fem_range_db')}, OpenFOAM {t.get('openfoam_builds')}.",
              f"- Source commit `{(ev.get('source') or {}).get('commit')}`.", "",
              "## Reported, not part of the decision (central table)", "",
              "| corner / payload | forcing | 0.1 m |", "|---|---|---|"]
    for k, d in sorted((v.get("reported_not_verdict") or {}).items()):
        for f, m in sorted(d.items()):
            lines.append(f"| `{k}` | {f} | {_f(m.get('0.1'))} |")
    lines += ["", "## Limitations", ""] + [f"- {w}" for k, w in v["wording"].items() if k != "supplemental"]
    return "\n".join(lines) + "\n"


def decide(bdir):
    from . import verdict as VV
    v = build(bdir)
    vs, *_ = VC.load_bundle(Path(bdir))
    out = VV.publish(v, _root(vs) / "verdict", renderer=render)
    iv = v["interval"] or {}
    print(f"{vs.name}: {v['outcome']}  L {_f(iv.get('L'))}  U {_f(iv.get('U'))} dB  -> {out}")
    for r in v["reasons"]:
        print("  reason:", r)
    return v


def main(argv=None):
    p = argparse.ArgumentParser(prog="acoustic-liner vehicles cfd-supplemental")
    p.add_argument("cmd", choices=("estimate", "select", "aggregate", "propagate", "decide"))
    p.add_argument("--bundle", required=True)
    p.add_argument("--rerun")
    a = p.parse_args(argv)
    if a.cmd == "estimate":
        if not a.rerun:
            p.error("estimate needs --rerun")
        estimate(a.bundle, a.rerun)
    elif a.cmd == "select":
        select_targeted(a.bundle)
    elif a.cmd == "aggregate":
        aggregate_targeted(a.bundle)
    elif a.cmd == "propagate":
        propagate(a.bundle)
    else:
        decide(a.bundle)


if __name__ == "__main__":
    main()
