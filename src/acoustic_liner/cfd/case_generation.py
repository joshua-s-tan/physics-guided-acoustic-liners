"""Per-bank OpenFOAM case generation for a finalist design: each bank's velocity bracket from the scenario
spectrum, one case tree per (bank, velocity) with a `_case_meta.json` carrying the identity hashes
(bank, geometry, case, input tree), and the case plan every later stage binds to. The solver is not run
here.
"""
from __future__ import annotations

import os, sys, json, math, hashlib, argparse
import numpy as np
from ..physics import oracle as O2
from ..physics import resonator as RM
from ..physics import liner as OR
from . import case as MI


PREF, NU = 20e-6, 1.5e-5
BANDS = np.array(O2.DESIGN_BAND_HZ, float)
THIRD_OCT_BW = 2 ** (1 / 6.0) - 2 ** (-1 / 6.0)     # ~0.2316


class CaseGenError(Exception):
    """A required case-generation input (scenario/source_corner/finalist_path/spectrum) was missing or
    inconsistent. Fails hard -- there is no implicit spectrum or scenario fallback, and no silent coercion."""
SPL_MIN_DB = 20.0     # plausible per-band SPL floor (dB); rejects garbage (negatives, un-physical lows)
SPL_MAX_DB = 220.0    # plausible per-band SPL ceiling (dB)


def _validate_spectrum(spectrum):
    """Return the spectrum as a list of exactly len(BANDS) finite floats in a plausible SPL range, or raise
    CaseGenError. Accepts list/tuple/ndarray; rejects None, wrong length, non-numeric, non-finite, out-of-range."""
    if spectrum is None:
        raise CaseGenError("spectrum is required -- no implicit spectrum fallback")
    try:
        arr = np.asarray(spectrum, dtype=float).ravel()
    except (TypeError, ValueError) as e:
        raise CaseGenError(f"spectrum must be numeric SPL values: {e}")
    if arr.size != len(BANDS):
        raise CaseGenError(f"spectrum must have exactly {len(BANDS)} band values, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise CaseGenError("spectrum must be all-finite numeric SPL values")
    if np.any(arr < SPL_MIN_DB) or np.any(arr > SPL_MAX_DB):
        raise CaseGenError(f"spectrum SPL outside plausible range [{SPL_MIN_DB}, {SPL_MAX_DB}] dB")
    return [float(v) for v in arr]


def bank_velocity(g, spl_dB):
    """Two bracketing PEAK neck-velocity estimates for one bank (neither a predicted launch velocity):
    resonant-tone upper bound (all 1/3-oct band energy as a tone at f0) and broadband-equivalent (only the
    resonator-bandwidth energy). p_pk = sqrt(2)*p_rms, NO rigid-wall doubling (interior/reverberant levels)."""
    f0, S = g["f0"], RM.neck_area(g["r_neck"])
    Le = RM.effective_neck_length(g["L_neck"], g["r_neck"])
    V = RM.cavity_volume_for_target_f0(f0, S, Le)
    Za = OR._resonator_Za(np.array([f0]), g["r_neck"], g["L_neck"], V, g["Q"])[0]
    spl = spl_dB[int(np.argmin(np.abs(BANDS - f0)))]
    p_rms = PREF * 10 ** (spl / 20.0)
    R_spec = float(abs(Za) * S)
    frac_bb = 1.0 / (g["Q"] * THIRD_OCT_BW)
    u_tone = float(np.sqrt(2) * p_rms / R_spec)
    u_bb = float(np.sqrt(2) * p_rms * np.sqrt(min(1.0, frac_bb)) / R_spec)
    d = 2 * g["r_neck"]
    return dict(f0_Hz=round(f0, 3), d_mm=round(d * 1000, 4), r_neck_m=g["r_neck"], L_neck_m=g["L_neck"],
                S_neck_m2=float(S), V_cav_m3=float(V), spl_dB=round(float(spl), 2),
                R_linear_loaded_Rayl=round(float(abs(Za.real) * S), 5),
                u_tone_upperbound=round(u_tone, 3), u_broadband_equiv=round(u_bb, 3),
                Re_tone=round(u_tone * d / NU, 0), Sr_tone=round(f0 * d / max(u_tone, 1e-9), 3))


def velocity_bracket(u_tone_ub, u_bb):
    """Pre-registered adaptive sweep (historical nonlinear-CFD protocol sec.3, step 3), deduplicated per bank:
    [0.5, 2, 6, 12, 20, u_bb, mid(u_bb,u_tone), max(20, 1.2*u_tone)] m/s."""
    hi = max(20.0, 1.2 * u_tone_ub)
    grid = sorted(set([0.5, 2.0, 6.0, 12.0, 20.0,
                       round(u_bb, 1), round(0.5 * (u_bb + u_tone_ub), 1), round(hi, 1)]))
    return [v for v in grid if v > 0]


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()
_CASE_HASH_KEYS = ("bank_id", "d_mm", "L_mm", "f0_Hz", "uneck_m_s", "refine", "maxco", "geometry_hash")


def _hash_keys(meta):
    return _CASE_HASH_KEYS + (("H_ext_mm",) if "H_ext_mm" in meta else ())


def recompute_case_hash(meta):
    """Re-derive a case_hash from a case's _case_meta.json params (the consumer recomputes, never trusts the
    stored value). Must match cfd.case_generation.generate's computation exactly. H_ext_mm enters only for an
    extended-domain case, so every existing case hash is unchanged."""
    return _sha({k: meta[k] for k in _hash_keys(meta)})


def input_tree_hash(outdir):
    """Deterministic hash over the generated OpenFOAM INPUT dictionaries under system/, constant/, 0/ (sorted
    relpath -> content sha256; excludes _case_meta.json and any solver output). Lets the consumer verify the
    case tree was not edited after generation, independent of the editable _case_meta.json.
    constant/polyMesh/ is EXCLUDED: blockMesh writes it (solver output, absent at generation), so a run case
    must hash the same as the freshly generated one; the mesh itself is bound via blockMeshDict + checkMesh."""
    h = hashlib.sha256()
    for sub in ("0", "constant", "system"):
        d = os.path.join(outdir, sub)
        if not os.path.isdir(d):
            continue
        for rel in sorted(os.path.relpath(os.path.join(dp, fn), outdir).replace(os.sep, "/")
                          for dp, _, fns in os.walk(d) for fn in fns):
            if os.path.basename(rel) == "_case_meta.json" or rel.startswith("constant/polyMesh/"):
                continue
            h.update(rel.encode()); h.update(b"\0")
            h.update(hashlib.sha256(open(os.path.join(outdir, rel), "rb").read()).digest())
    return h.hexdigest()


def generate(outdir, *, bank_id, d_mm, L_mm, f0, uneck, cycles, geometry_hash, refine=1.0, maxco=0.5, H_ext_mm=0.0):
    """Generate ONE OpenFOAM case into outdir by overriding cfd.case globals (never mutates other cases).
    Writes _case_meta.json with the per-bank f0/neck area + identity hashes for post/validate.
    H_ext_mm > 0 extends both reservoirs by a uniform block (cfd.case.build_neck_mesh); 0 gives the original case."""
    r = d_mm / 2000.0; L = L_mm / 1000.0
    MI.R_NECK, MI.L_NECK, MI.R_TUBE = r, L, 3.0 * r
    MI.A_NECK = math.pi * r ** 2
    MI.A_TUBE = math.pi * (3.0 * r) ** 2
    MI.AREA_RATIO = MI.A_NECK / MI.A_TUBE
    MI.F0 = float(f0)
    MI.HERE = outdir
    for d in ("0", "system", "constant"):
        os.makedirs(os.path.join(outdir, d), exist_ok=True)
    V = MI.Verts(MI.WEDGE_DEG)
    H_res = MI.H_RES
    u0 = uneck * MI.AREA_RATIO
    open(os.path.join(outdir, "system", "fvSchemes"), "w").write(MI.build_fvschemes())
    open(os.path.join(outdir, "system", "fvSolution"), "w").write(MI.build_fvsolution())
    open(os.path.join(outdir, "constant", "transportProperties"), "w").write(MI.build_transport())
    open(os.path.join(outdir, "constant", "turbulenceProperties"), "w").write(MI.build_turbulence())
    open(os.path.join(outdir, "system", "blockMeshDict"), "w").write(MI.build_neck_mesh(V, H_res, refine, H_ext_mm / 1000.0))
    open(os.path.join(outdir, "system", "controlDict"), "w").write(
        MI.build_controldict_neck(endTime=cycles / MI.F0, H_res=H_res, maxCo=maxco))
    for nm, txt in MI.build_fields(u0).items():
        open(os.path.join(outdir, "0", nm), "w").write(txt)
    meta = dict(bank_id=bank_id, d_mm=d_mm, L_mm=L_mm, f0_Hz=f0, uneck_m_s=uneck, cycles=cycles,
                inlet_u0_m_s=u0, R_tube_mm=3 * d_mm / 2, refine=refine, maxco=maxco,
                geometry_hash=geometry_hash)
    if H_ext_mm > 0:
        meta.update(H_ext_mm=float(H_ext_mm), H_res_mm=H_res * 1000.0,
                    n_ext=MI.n_ext(H_ext_mm / 1000.0, H_res, refine))
    meta["case_hash"] = _sha({k: meta[k] for k in _hash_keys(meta)})
    meta["input_tree_hash"] = input_tree_hash(outdir)      # over the just-written OpenFOAM dictionaries
    json.dump(meta, open(os.path.join(outdir, "_case_meta.json"), "w"), indent=1, default=float)
    return outdir, meta


def gen_cases(outbase, *, scenario, source_corner, finalist_path, spectrum, geometry_hash=None,
              bank_prefix="b", plan_filename="_cfd_case_plan.json", cycles=12, refine=1.0, maxco=0.5):
    """Generate ALL bank x each bank's own pre-registered velocity bracket cases from EXPLICIT required
    inputs: scenario id, source_corner id, finalist geometry path, and spectrum (len(BANDS)-band SPL dB).
    Every one is REQUIRED -- raises CaseGenError on any missing/invalid input, with NO implicit
    spectrum/scenario fallback and NO silent coercion. Hardening:
      * spectrum is fully validated (length/numeric/finite/plausible-range; accepts list/tuple/ndarray);
      * geometry_hash is ALWAYS recomputed from the finalist geometry -- a supplied value is rejected unless
        it matches exactly (callers cannot pass a hash that disagrees with the loaded geometry);
      * if the finalist record carries scenario / fem_corner identity, it MUST match the requested
        `scenario` / `source_corner` (a Falcon finalist cannot be relabelled Atlas by passing other strings);
      * the plan stores the COMPLETE spectrum plus its recomputed hash (self-reproducing provenance).
    Bank ids / plan filename are generic by default (`b0..`, `_cfd_case_plan.json`); the Falcon-L2 wrapper
    overrides them to preserve byte-identical frozen hashes."""
    if not scenario:
        raise CaseGenError("scenario is required (no implicit default)")
    if not source_corner:
        raise CaseGenError("source_corner is required (no implicit default)")
    if not finalist_path:
        raise CaseGenError("finalist_path is required (no implicit default)")
    scen_spl = _validate_spectrum(spectrum)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    fin = json.load(open(finalist_path))
    geom = fin["geometry"]; groups = geom["groups"]; D = geom["D"]
    # (2) never trust a supplied geometry hash -- recompute and require an exact match if one was given.
    computed_hash = _sha(geom)
    if geometry_hash is not None and geometry_hash != computed_hash:
        raise CaseGenError(f"supplied geometry_hash {geometry_hash!r} != recomputed {computed_hash!r}")
    geometry_hash = computed_hash
    # (3) bind finalist identity to the requested scenario/source_corner when the finalist declares them.
    fin_scen = fin.get("scenario")
    if fin_scen is not None and fin_scen != scenario:
        raise CaseGenError(f"finalist scenario {fin_scen!r} != requested scenario {scenario!r}")
    fin_corner = fin.get("fem_corner")
    if fin_corner is not None:
        fin_corner_id = "_".join(str(c) for c in fin_corner)
        if fin_corner_id != source_corner:
            raise CaseGenError(f"finalist source corner {fin_corner_id!r} != requested source_corner {source_corner!r}")
    spectrum_hash = _sha(scen_spl)
    made = []
    plan = dict(geometry_path=os.path.relpath(finalist_path, root), geometry_hash=geometry_hash,
                scenario=scenario, source_corner=source_corner,
                spectrum=scen_spl, spectrum_hash=spectrum_hash,
                coverage=fin.get("coverage"), banks=[], outbase=outbase)
    for j, g in enumerate(groups):
        bv = bank_velocity(g, scen_spl)
        vels = velocity_bracket(bv["u_tone_upperbound"], bv["u_broadband_equiv"])
        bank_id = f"{bank_prefix}{j}"
        plan["banks"].append(dict(bank_id=bank_id, **bv, velocities_m_s=vels))
        for u in vels:
            name = f"{bank_id}_u{u:g}"
            outdir = os.path.join(outbase, name)
            _, meta = generate(outdir, bank_id=bank_id, d_mm=bv["d_mm"], L_mm=bv["L_neck_m"] * 1000.0,
                               f0=bv["f0_Hz"], uneck=u, cycles=cycles, geometry_hash=geometry_hash,
                               refine=refine, maxco=maxco)
            made.append(dict(name=name, bank_id=bank_id, u_target=u, case_hash=meta["case_hash"],
                             input_tree_hash=meta["input_tree_hash"], refine=refine, maxco=maxco))
    plan["n_cases"] = len(made); plan["cases"] = made
    plan["case_names"] = [c["name"] for c in made]
    os.makedirs(outbase, exist_ok=True)
    json.dump(plan, open(os.path.join(outbase, plan_filename), "w"), indent=2, default=float)
    return plan
