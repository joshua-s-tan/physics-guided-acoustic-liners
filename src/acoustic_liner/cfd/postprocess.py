"""Post-processing of one solved case: probe parsing, the energy resistance and fundamental-phasor
impedance (R, X) with cross-checks, harmonic distortion, inter-window drift, and the mesh check.
R_eff = <dp*u>/<u^2>; phasor Z = dp_hat/u_hat (fundamental); X_accel cross-check.
"""
from __future__ import annotations

import os, re, json, math, hashlib
import numpy as np


RHO = 1.2
WEDGE_FACTOR = 72.0           # 360/5deg wedge
FLOW_SIGN = -1.0              # downward flow positive


def load_case_params(case):
    """Read per-case f0 [Hz] and neck area [m^2] (+ identity) from `_case_meta.json` written by cfd.case_generation."""
    meta = json.load(open(os.path.join(case, "_case_meta.json")))
    r = float(meta["d_mm"]) / 2000.0
    return dict(f0=float(meta["f0_Hz"]), a_neck=math.pi * r * r, u_target=float(meta.get("uneck_m_s", float("nan"))),
                bank_id=meta.get("bank_id"), geometry_hash=meta.get("geometry_hash"),
                case_hash=meta.get("case_hash"), meta=meta)


class ProbeError(ValueError):
    """A probe history failed sanitation (fail-closed; never silently repaired/held)."""


def read_plane(case, name):
    """Read one probe plane with sanitation. FAIL CLOSED (ProbeError) on: missing / MULTIPLE time directories
    (a restart split -> not one coherent history), <2 samples, non-monotone or duplicate times, non-finite."""
    base = os.path.join(case, "postProcessing", name)
    if not os.path.isdir(base):
        raise ProbeError(f"{name}: no postProcessing directory")
    tdirs = sorted([d for d in os.listdir(base) if re.fullmatch(r"[0-9.eE+-]+", d)], key=float)
    if len(tdirs) == 0:
        raise ProbeError(f"{name}: no time directory")
    if len(tdirs) > 1:
        raise ProbeError(f"{name}: {len(tdirs)} time directories {tdirs} (restart split; not one coherent history)")
    t, v = [], []
    for line in open(os.path.join(base, tdirs[0], "surfaceFieldValue.dat")):
        if line.startswith("#") or not line.strip():
            continue
        p = line.replace("(", " ").replace(")", " ").split()
        t.append(float(p[0])); v.append(float(p[-1]))
    t = np.asarray(t); v = np.asarray(v)
    if t.size < 2:
        raise ProbeError(f"{name}: <2 samples")
    if not np.all(np.diff(t) > 0):
        raise ProbeError(f"{name}: times not strictly increasing / duplicated")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(v))):
        raise ProbeError(f"{name}: non-finite samples")
    return t, v


# a valid case must START near zero (< START_TOL_CYCLES of a period) and carry >= MIN_HISTORY_CYCLES of history
START_TOL_CYCLES = 0.5
MIN_HISTORY_CYCLES = 3.0


def _pair_signals(case, a_neck, f0, up="pUp", dn="pDown"):
    """Common-bounds t, dp_phys, u_neck. Interpolation is restricted to the OVERLAP of all three series (never
    holds an endpoint). FAIL CLOSED (ProbeError) on late start, mismatched spans, or insufficient history."""
    tu, pu = read_plane(case, up); td, pd = read_plane(case, dn); tf, qf = read_plane(case, "fluxMid")
    T = 1.0 / f0
    starts = [tu[0], td[0], tf[0]]; ends = [tu[-1], td[-1], tf[-1]]
    t0 = max(starts); t1 = min(ends)
    if min(starts) > START_TOL_CYCLES * T:
        raise ProbeError(f"probe start {min(starts):.4g}s > {START_TOL_CYCLES} cycle(s) (late start)")
    if (max(starts) - min(starts)) > START_TOL_CYCLES * T or (max(ends) - min(ends)) > START_TOL_CYCLES * T:
        raise ProbeError(f"probe spans mismatch across planes (starts {starts}, ends {ends})")
    if (t1 - t0) < MIN_HISTORY_CYCLES * T:
        raise ProbeError(f"insufficient history: common span {t1 - t0:.4g}s < {MIN_HISTORY_CYCLES} cycles")
    t = tu[(tu >= t0) & (tu <= t1)]
    dp = RHO * (np.interp(t, tu, pu) - np.interp(t, td, pd))
    u = FLOW_SIGN * np.interp(t, tf, qf) * WEDGE_FACTOR / a_neck
    return t, dp, u, dict(common_t0=float(t0), common_t1=float(t1), span_s=float(t1 - t0))


def _phasor(t, x, f0, k=1):
    A = np.column_stack([np.cos(k * 2 * math.pi * f0 * t), np.sin(k * 2 * math.pi * f0 * t)])
    a, b = np.linalg.lstsq(A, x, rcond=None)[0]
    return a - 1j * b


def _thd(t, x, f0):
    h1 = abs(_phasor(t, x, f0, 1))
    hh = math.sqrt(sum(abs(_phasor(t, x, f0, k)) ** 2 for k in range(2, 8)))
    return hh / h1 if h1 else float("nan")


def _uniform(case, a_neck, f0, n_periods, skip_last=0):
    """dp,u resampled onto a UNIFORM grid over an integer number of cycles (adaptive-dt -> biased mean bug).
    FAIL CLOSED if the requested window would fall outside the measured samples (no endpoint holding)."""
    t, dp, u, _b = _pair_signals(case, a_neck, f0)
    T = 1.0 / f0; t1 = t[-1] - skip_last * T; t0 = t1 - n_periods * T
    if t0 < t[0] - 1e-12:
        raise ProbeError(f"requested {n_periods}-cycle window starts at {t0:.4g}s < first sample {t[0]:.4g}s "
                         f"(would extrapolate/hold endpoint)")
    tu = np.linspace(t0, t1, 4000 * n_periods, endpoint=False)
    dpu = np.interp(tu, t, dp); uu = np.interp(tu, t, u)
    return tu, dpu - dpu.mean(), uu - uu.mean()


def extract(case, n_periods=3, params=None):
    """R_eff/R_phasor/X_phasor/X_accel/THD/u_amp for one case (per-bank f0 + neck area from meta)."""
    p = params or load_case_params(case)
    f0, a_neck = p["f0"], p["a_neck"]
    tu, dpu, uu = _uniform(case, a_neck, f0, n_periods)
    R_eff = float(np.mean(dpu * uu) / np.mean(uu * uu))
    Z = _phasor(tu, dpu, f0) / _phasor(tu, uu, f0)
    R_phasor, X_phasor = float(Z.real), float(Z.imag)
    a = np.gradient(uu, tu)
    X_accel = float(2 * math.pi * f0 * np.mean(dpu * a) / np.mean(a * a))
    eps_R = abs(R_eff - R_phasor) / abs(R_eff) if R_eff else float("nan")
    finite = all(np.isfinite(v) for v in (R_eff, R_phasor, X_phasor, X_accel))
    return dict(u_amp=float(abs(_phasor(tu, uu, f0))), R_eff=R_eff, R_phasor=R_phasor, X_phasor=X_phasor,
                X_accel_xcheck=X_accel, identity_residual=eps_R, thd_u=float(_thd(tu, uu, f0)),
                thd_p=float(_thd(tu, dpu, f0)), n_periods=n_periods, probe_finite=bool(finite))


def convergence(case, params=None):
    """Inter-window drift on BOTH R and X: last cycle vs the previous one (non-overlapping single-cycle windows)."""
    p = params or load_case_params(case)
    f0, a_neck = p["f0"], p["a_neck"]
    tu, dpu, uu = _uniform(case, a_neck, f0, 2)
    tsplit = tu[-1] - 1.0 / f0

    def RX_of(mask):
        d = dpu[mask] - dpu[mask].mean(); v = uu[mask] - uu[mask].mean()
        R = float(np.mean(d * v) / np.mean(v * v))
        Z = _phasor(tu[mask], d, f0) / _phasor(tu[mask], v, f0)
        return R, float(Z.imag)
    Rl, Xl = RX_of(tu >= tsplit); Rp, Xp = RX_of(tu < tsplit)
    return dict(R_drift=abs(Rl - Rp) / abs(Rl) if Rl else float("nan"),
                X_drift=abs(Xl - Xp) / abs(Xl) if Xl else float("nan"))


def _mesh_ok(case):
    """Parse a checkMesh log if present (mesh OK <=> 'Mesh OK.'); absent log -> False (fail-closed)."""
    for name in ("log.checkMesh", "checkMesh.log"):
        p = os.path.join(case, name)
        if os.path.isfile(p):
            return "Mesh OK." in open(p, errors="ignore").read()
    return False


def post_case(case, n_periods=3):
    """Full validate-ready row for one CFD case: signal extraction + R/X drift + mesh + identity/hashes.
    A ProbeError (restart split / non-monotone times / late start / span mismatch / insufficient history /
    out-of-bounds window) FAILS CLOSED: probe_clean=False and the numeric fields are NaN so validate_cfd_row
    rejects the row (never a silent hold/repair)."""
    p = load_case_params(case)
    nan = float("nan")
    base = dict(bank_id=p["bank_id"], case_hash=p["case_hash"], geometry_hash=p["geometry_hash"],
                u_target=p["u_target"], mesh_ok=_mesh_ok(case), n_periods=n_periods,
                case_dir=os.path.abspath(case))
    try:
        ex = extract(case, n_periods, params=p)
        cv = convergence(case, params=p)
    except ProbeError as e:
        return dict(base, u_amp=nan, R_phasor=nan, X_phasor=nan, R_eff=nan, X_accel_xcheck=nan,
                    R_drift=nan, X_drift=nan, thd_p=nan, thd_u=nan, probe_clean=False,
                    identity_residual=nan, probe_error=str(e))
    return dict(base, u_amp=ex["u_amp"], R_phasor=ex["R_phasor"], X_phasor=ex["X_phasor"], R_eff=ex["R_eff"],
                X_accel_xcheck=ex["X_accel_xcheck"], R_drift=cv["R_drift"], X_drift=cv["X_drift"],
                thd_p=ex["thd_p"], thd_u=ex["thd_u"], probe_clean=bool(ex["probe_finite"]),
                identity_residual=ex["identity_residual"])
