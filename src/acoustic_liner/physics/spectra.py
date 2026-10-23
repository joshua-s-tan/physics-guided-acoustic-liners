# -*- coding: utf-8 -*-
r"""Case-F v2.0 training-spectrum FAMILY + held-out reference-environment splits.

MIGRATED from `fairing_case/v2/spectra.py` (frozen SHA-256
119e861689e3a84902a8e9f930336b7669cf65e368860dceee389c7d44a4e08c) for the §14 step-8 self-containment migration
(the historical migration record, which is not distributed). Changes vs the reference: `sys.path` hacks +
`import oracle_v2/launch_environments` replaced
by intra-package imports; `__main__` manifest writer dropped. All synthetic-family math is byte-identical, so
`sample_family` / `make_splits` reproduce the reference EXACTLY for a given seed (fixture-verified). The only
behavioural change is inherited from `physics.environments`: the reference set now omits the DORMANT Electron
environment (4 held-out references, not 5).

SHAPE MODEL: L(f) = A*exp[-1/2 (log2(f/fc)/W)^2] + T*log2(f/Fg)  (4 independent params: centre, width, amp, tilt).
Weights w_i = 10^(L_i/10)/sum after subtracting max (level cancels -> v2.0 label is level-invariant).

LEAKAGE / METHODOLOGY: the family parameter ranges are fixed constants; there is NO runtime data path from the
reference spectra into the generator (enforced by test). The ranges were chosen knowing whether the reference
spectra sat inside them, so those references influenced FAMILY SELECTION -> they are DEVELOPMENT/REFERENCE
environments, not an untouched test set; "family-approximable" is NOT a generalization claim.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from . import oracle as O2
from . import environments as LE

BAND = np.array(O2.DESIGN_BAND_HZ, dtype=float)             # [50..250], 8 bands
F_GEOMEAN = float(np.exp(np.mean(np.log(BAND))))            # tilt reference (band geo-mean)

CENTER_HZ_RANGE = (40.0, 315.0)
WIDTH_OCT_RANGE = (0.30, 1.50)
AMP_DB_RANGE = (0.0, 15.0)
TILT_DB_PER_OCT_RANGE = (-6.0, 6.0)
LEVEL_DB_RANGE = (120.0, 145.0)        # documentation / v2.1 ONLY -- NOT sampled in v2.0

PARAM_NAMES = ("center_hz", "width_oct", "amp_db", "tilt_db_per_oct")
PARAM_BOX = (CENTER_HZ_RANGE, WIDTH_OCT_RANGE, AMP_DB_RANGE, TILT_DB_PER_OCT_RANGE)

MAX_BAND_SPAN_DB = 45.0
MAX_ADJ_JUMP_DB = 18.0


def _bump(x, width_oct):
    return np.exp(-0.5 * (x / width_oct) ** 2)


def shape_spl(center_hz, width_oct, amp_db, tilt_db_per_oct, band=BAND):
    """Relative SPL (dB, arbitrary offset) at the design bands for one parameter set."""
    x = np.log2(band / center_hz)
    return amp_db * _bump(x, width_oct) + tilt_db_per_oct * np.log2(band / F_GEOMEAN)


def repr_from_spl(spl):
    """Level-invariant representation: energy weights (sum 1) + relative dB + band."""
    spl = np.asarray(spl, float)
    rel = spl - spl.max()
    e = 10.0 ** (rel / 10.0)
    w = e / e.sum()
    return dict(weights=w.tolist(), rel_dB=rel.tolist(), band_hz=BAND.tolist())


def spectrum_from_params(p, level_db=130.0):
    """Full spectrum dict for one parameter set (label uses only the level-invariant repr)."""
    spl = shape_spl(p["center_hz"], p["width_oct"], p["amp_db"], p["tilt_db_per_oct"])
    spl = spl - spl.max() + level_db
    return dict(params=dict(p), level_db=float(level_db), spl_dB=spl.tolist(), **repr_from_spl(spl))


def band_span_dB(spl):
    spl = np.asarray(spl, float)
    return float(spl.max() - spl.min())


def _lhs(n, box, rng):
    """Standard LHS: stratify each dim into n bins, one uniform draw per bin, permute per dim."""
    d = len(box)
    out = np.empty((n, d))
    for j, (lo, hi) in enumerate(box):
        edges = np.linspace(lo, hi, n + 1)
        u = edges[:-1] + rng.random(n) * (edges[1:] - edges[:-1])
        out[:, j] = rng.permutation(u)
    return out


def sample_family(n, seed=0, level_db=130.0, enforce_bounds=True):
    """n synthetic training spectra via LHS over the documented parameter box, with a runtime span guard."""
    X = _lhs(n, PARAM_BOX, np.random.default_rng(seed))
    out = [spectrum_from_params(dict(zip(PARAM_NAMES, row)), level_db) for row in X]
    if enforce_bounds and out:
        span = max(band_span_dB(s["spl_dB"]) for s in out)
        jump = max(float(np.max(np.abs(np.diff(s["spl_dB"])))) for s in out)
        if span > MAX_BAND_SPAN_DB or jump > MAX_ADJ_JUMP_DB:
            raise ValueError(f"family exceeds validated bounds: span {span:.1f}/{MAX_BAND_SPAN_DB}, "
                             f"jump {jump:.1f}/{MAX_ADJ_JUMP_DB} -- re-check the parameter box")
    return out


def _ood_structural(n, seed):
    """Shapes a single-bump+tilt family CANNOT represent: bimodal (two separated peaks), notched, or rough."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        kind = ("bimodal", "notched", "rough")[i % 3]
        if kind == "bimodal":
            c1, c2 = rng.uniform(50, 60), rng.uniform(210, 250)
            spl = (12 * _bump(np.log2(BAND / c1), 0.35) + 12 * _bump(np.log2(BAND / c2), 0.35))
        elif kind == "notched":
            cn = rng.uniform(100, 140)
            spl = 5 * np.log2(BAND / BAND[0]) - 18 * _bump(np.log2(BAND / cn), 0.30)
        else:
            saw = 6.0 * (np.arange(BAND.size) % 2) * 2 - 6.0
            spl = 3 * np.log2(BAND / F_GEOMEAN) + saw + rng.normal(0, 1.5, size=BAND.size)
        out.append(dict(kind=kind, spl_dB=(spl - spl.max()).tolist(), **repr_from_spl(spl)))
    return out


def _ood_parametric(n, seed):
    """Exactly n draws from boxes OUTSIDE the training box, spread across several directions."""
    dirs = [((18.0, 36.0), WIDTH_OCT_RANGE, AMP_DB_RANGE, TILT_DB_PER_OCT_RANGE),
            ((340.0, 520.0), WIDTH_OCT_RANGE, AMP_DB_RANGE, TILT_DB_PER_OCT_RANGE),
            (CENTER_HZ_RANGE, (0.12, 0.26), AMP_DB_RANGE, TILT_DB_PER_OCT_RANGE),
            (CENTER_HZ_RANGE, WIDTH_OCT_RANGE, (18.0, 30.0), TILT_DB_PER_OCT_RANGE),
            (CENTER_HZ_RANGE, WIDTH_OCT_RANGE, AMP_DB_RANGE, (-12.0, -8.0)),
            (CENTER_HZ_RANGE, WIDTH_OCT_RANGE, AMP_DB_RANGE, (8.0, 12.0))]
    counts = np.full(len(dirs), n // len(dirs), dtype=int)
    counts[: n % len(dirs)] += 1
    out = []
    for k, (box, cnt) in enumerate(zip(dirs, counts)):
        if cnt == 0:
            continue
        for row in _lhs(int(cnt), box, np.random.default_rng(seed + 100 + k)):
            s = spectrum_from_params(dict(zip(PARAM_NAMES, row)))
            s["kind"] = f"param_ood_dir{k}"
            out.append(s)
    np.random.default_rng(seed).shuffle(out)
    return out


def make_splits(n_train=2000, n_interp=400, n_ood_param=180, n_ood_struct=60, seed=0):
    """Disjoint synthetic draws: train + in-distribution interpolation + OOD (parametric + structural)."""
    train = sample_family(n_train, seed=seed)
    interp = sample_family(n_interp, seed=seed + 10_000)
    ood = _ood_parametric(n_ood_param, seed + 20_000) + _ood_structural(n_ood_struct, seed + 30_000)
    np.random.default_rng(seed + 40_000).shuffle(ood)
    return dict(train=train, interpolation=interp, ood=ood)


def reference_environments():
    """The held-out REFERENCE environments as level-invariant spectra (never trained on). Electron is DORMANT
    and excluded (see physics.environments). NASA GEVS is a qualification envelope; several are reconstructions."""
    out = {}
    for name in LE.ORDER:
        spl = LE.spl_array(name)
        out[name] = dict(name=name, oaspl=LE.ENVIRONMENTS[name]["oaspl"],
                         provenance=LE.ENVIRONMENTS[name]["provenance"],
                         spl_dB=spl.tolist(), **repr_from_spl(spl))
    return out


def fit_to_family(spl, restrict_to_box=False, peak_grid=None, width_grid=None):
    """Best (center,width,amp,tilt) fit of the bump+tilt form to a target rel-dB shape (true bounded LS in
    (amp,tilt,offset) for fixed (center,width), gridded over (center,width))."""
    spl = np.asarray(spl, float)
    rel = spl - spl.max()
    if peak_grid is None:
        box_c = np.linspace(*CENTER_HZ_RANGE, 50)
        peak_grid = box_c if restrict_to_box else np.union1d(np.linspace(30.0, 400.0, 50), box_c)
    if width_grid is None:
        box_w = np.linspace(*WIDTH_OCT_RANGE, 35)
        width_grid = box_w if restrict_to_box else np.union1d(np.linspace(0.15, 2.0, 35), box_w)
    logf = np.log2(BAND / F_GEOMEAN)
    if restrict_to_box:
        lb = [AMP_DB_RANGE[0], TILT_DB_PER_OCT_RANGE[0], -np.inf]
        ub = [AMP_DB_RANGE[1], TILT_DB_PER_OCT_RANGE[1], np.inf]
    else:
        lb = [0.0, -np.inf, -np.inf]; ub = [np.inf, np.inf, np.inf]
    best = None
    for c in peak_grid:
        x = np.log2(BAND / c)
        for w in width_grid:
            A = np.column_stack([_bump(x, w), logf, np.ones_like(logf)])
            sol = lsq_linear(A, rel, bounds=(lb, ub), method="bvls")
            resid = float(np.sqrt(np.mean((A @ sol.x - rel) ** 2)))
            if best is None or resid < best[0]:
                best = (resid, c, w, float(sol.x[0]), float(sol.x[1]))
    resid, c, w, amp, tilt = best
    p = dict(center_hz=float(c), width_oct=float(w), amp_db=amp, tilt_db_per_oct=tilt)
    in_box, why = _in_train_box(p)
    return dict(params=p, rms_resid_dB=resid, in_train_box=in_box, out_of_box_reason=why,
                restricted=restrict_to_box)


def _in_train_box(p):
    checks = [("center_hz", p["center_hz"], CENTER_HZ_RANGE),
              ("width_oct", p["width_oct"], WIDTH_OCT_RANGE),
              ("amp_db", p["amp_db"], AMP_DB_RANGE),
              ("tilt_db_per_oct", p["tilt_db_per_oct"], TILT_DB_PER_OCT_RANGE)]
    reasons = [f"{nm}={v:.3g} outside {rng}" for nm, v, rng in checks if not (rng[0] <= v <= rng[1])]
    return (len(reasons) == 0), ("; ".join(reasons) if reasons else "")


def classify_references(resid_tol_dB=1.5):
    """Assign each reference env to family-approximable vs OOD by a true bounded in-box fit. NOT a
    generalization claim -- only whether the family can APPROXIMATE the shape."""
    out = {}
    for name, r in reference_environments().items():
        fit_box = fit_to_family(r["spl_dB"], restrict_to_box=True)
        fit_free = fit_to_family(r["spl_dB"], restrict_to_box=False)
        reachable = bool(fit_box["rms_resid_dB"] <= resid_tol_dB)
        out[name] = dict(fit=fit_box, fit_unrestricted=fit_free, reachable=reachable,
                         verdict=("family-approximable" if reachable else "OOD"),
                         reason=(f"in-box residual {fit_box['rms_resid_dB']:.2f} dB <= {resid_tol_dB}"
                                 if reachable else
                                 f"no in-box member within {resid_tol_dB} dB (best {fit_box['rms_resid_dB']:.2f} dB)"))
    return out


def weight_space_stats(train, refs=None):
    refs = refs or reference_environments()
    Wtr = np.array([s["weights"] for s in train])
    out = {}
    for name, r in refs.items():
        d = np.abs(Wtr - np.array(r["weights"])).sum(axis=1)     # L1 in weight space
        out[name] = dict(min_L1=float(d.min()), tv_min=float(d.min() / 2.0),
                         n_within_0p1=int((d < 0.1).sum()))
    return out
