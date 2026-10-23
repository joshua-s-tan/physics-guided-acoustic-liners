"""Self-consistent SPL -> U -> z -> U fixed point per bank and the complete-design nonlinear relabel with
the excess-inertance law: dR(U) = max(0, R1(U) - R1(U_ref)), dM(U) = (X1(U) - X1(U_ref)) / (2 pi f0),
z_total(f, U) = z_linear(f) + dR(U) + i 2 pi f dM(U).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from ..physics import oracle as O2
from ..physics import resonator as RM
from ..physics import liner as OR


PREF = 20e-6
THIRD_OCT_BW = 2 ** (1 / 6.0) - 2 ** (-1 / 6.0)     # 1/3-octave fractional bandwidth ~0.2316
U_REF_MAX = 1.0                                      # a valid anchor's MEASURED u_amp must be <= this (low-amp)
R_CLAMP_TOL = 0.05                                   # Rayl: max R decrease treated as NUMERICAL (running-max);


def _prep_RX(rx_table):
    """From validated CFD rows [(u_amp, R_phasor, X_phasor), ...] build monotone-clamped R(u), signed X(u), and
    the clamp record. Returns (Rfun, Xfun, Umin, Umax, clamp). Raises ValueError('nonmonotonic_R') on a MATERIAL
    resistive decrease (> R_CLAMP_TOL). The reference (R,X) is NOT taken here -- it is the measured anchor row."""
    t = sorted(((float(u), float(r), float(x)) for u, r, x in rx_table), key=lambda z: z[0])
    U = np.array([p[0] for p in t]); R = np.array([p[1] for p in t]); X = np.array([p[2] for p in t])
    clamp_points = []
    Rc = R.copy()
    for i in range(1, len(Rc)):
        if Rc[i] < Rc[i - 1]:
            drop = float(Rc[i - 1] - Rc[i])
            if drop > R_CLAMP_TOL:
                raise ValueError(f"nonmonotonic_R: R dropped {drop:.4f} Rayl (> {R_CLAMP_TOL}) between "
                                 f"U={U[i-1]:.3f} and U={U[i]:.3f} m/s (material nonmonotonicity; fail closed)")
            clamp_points.append(dict(u=float(U[i]), raw=float(Rc[i]), clamped_to=float(Rc[i - 1]), drop=drop))
            Rc[i] = Rc[i - 1]
    Rfun = lambda u: float(np.interp(u, U, Rc))
    Xfun = lambda u: float(np.interp(u, U, X))
    clamp = dict(n_clamped=len(clamp_points), points=clamp_points)
    return Rfun, Xfun, float(U[0]), float(U[-1]), clamp


def _validate_anchor(anchor):
    """The reference must be the MEASURED (u_amp, R, X) of the validated low-amplitude (target-0.5) case, NOT an
    interpolation at a fixed 0.5. Reject a missing/non-finite/non-low-amplitude anchor (fail closed)."""
    if anchor is None:
        return None, "no_anchor (validated target-0.5 case missing)"
    try:
        u0, R0, X0 = (float(anchor[0]), float(anchor[1]), float(anchor[2]))
    except (TypeError, ValueError, IndexError):
        return None, f"no_anchor (malformed anchor {anchor!r})"
    if not all(np.isfinite(v) for v in (u0, R0, X0)):
        return None, "no_anchor (non-finite anchor)"
    if not (0.0 < u0 <= U_REF_MAX):
        return None, f"no_anchor (anchor u_amp {u0:.3f} not in (0, {U_REF_MAX}] -- not a low-amplitude anchor)"
    return (u0, R0, X0), None


def _p_peak(spl_dB, p_scale):
    return p_scale * np.sqrt(2) * PREF * 10 ** (spl_dB / 20.0)


def self_consistent_U(bank, spl_dB_at_f0, rx_table, anchor, *, p_scale=1.0):
    """Bracketed self-consistent neck velocity for one bank under the excess-inertance law. `bank` carries
    f0_Hz and z_linear_complex_Rayl. `anchor` = the validated low-amplitude (target-0.5) row's MEASURED
    (u_amp, R_phasor, X_phasor); the excess is referenced to it (NO endpoint-holding interp at a fixed 0.5).
    status: 'ok' | 'extend_sweep' | 'below_range' | 'ambiguous_root' | 'nonmonotonic_R' | 'no_anchor'."""
    f0 = float(bank["f0_Hz"])
    z_lin = complex(bank["z_linear_complex_Rayl"])
    anc, aerr = _validate_anchor(anchor)
    if anc is None:
        return dict(status="no_anchor", U_star=None, detail=aerr)
    u_ref, Rref, Xref = anc
    try:
        Rfun, Xfun, Umin, Umax, clamp = _prep_RX(rx_table)
    except ValueError as e:
        return dict(status="nonmonotonic_R", U_star=None, detail=str(e))
    clamp = dict(clamp, U_ref_measured=round(u_ref, 5), R_ref_Rayl=round(Rref, 5), X_ref_Rayl=round(Xref, 5))
    p_pk = _p_peak(spl_dB_at_f0, p_scale)
    w0 = 2.0 * np.pi * f0

    def dz(u):
        dR = max(0.0, Rfun(u) - Rref)
        dM = (Xfun(u) - Xref) / w0                    # inertance change (signed, unclamped)
        return dR + 1j * w0 * dM                      # excess at f0

    def h(u):
        return u * abs(z_lin + dz(u)) - p_pk

    us = np.linspace(Umin, Umax, 200); hs = np.array([h(u) for u in us])
    sign_changes = int(np.sum(np.diff(np.sign(hs)) != 0))
    if hs[0] > 0:
        return dict(status="below_range", U_star=None, U_min=Umin, U_max=Umax, clamp=clamp)
    if hs[-1] < 0:
        return dict(status="extend_sweep", U_star=None, U_needed_gt=round(Umax, 2),
                    U_linear_unlimited=round(float(p_pk / abs(z_lin)), 2), U_min=Umin, U_max=Umax, clamp=clamp)
    u = brentq(h, Umin, Umax, xtol=1e-4, rtol=1e-8)
    dR = max(0.0, Rfun(u) - Rref); dM = (Xfun(u) - Xref) / w0
    status = "ok" if sign_changes == 1 else "ambiguous_root"   # fail closed on non-unique root
    return dict(status=status, U_star=round(float(u), 3), f0_Hz=f0,
                dR_excess_Rayl=round(float(dR), 5), dM_inertance_Rayl_s=round(float(dM), 8),
                dX_excess_at_f0_Rayl=round(float(w0 * dM), 5),
                z_total_mag_at_f0_Rayl=round(float(abs(z_lin + (dR + 1j * w0 * dM))), 5),
                z_linear_mag_Rayl=round(float(abs(z_lin)), 5),
                U_linear_unlimited=round(float(p_pk / abs(z_lin)), 2),
                n_sign_changes=sign_changes, unique=bool(sign_changes == 1),
                U_min=Umin, U_max=Umax, clamp=clamp)


def _z_surface_NL(freqs, groups, D, dR_specific, dM_specific):
    """oracle.z_surface with a per-bank SPECIFIC nonlinear excess added FREQUENCY-DEPENDENTLY:
    Za_j(f) -> Za_j(f) + (dR_j + i*2*pi*f*dM_j)/S_j  (specific excess -> acoustic impedance via /S)."""
    freqs = np.asarray(freqs, float)
    beta = np.zeros_like(freqs, dtype=complex)
    w = 2.0 * np.pi * freqs
    for j, g in enumerate(groups):
        S = RM.neck_area(g["r_neck"]); Le = RM.effective_neck_length(g["L_neck"], g["r_neck"])
        V = RM.cavity_volume_for_target_f0(g["f0"], S, Le)
        excess_specific = dR_specific[j] + 1j * w * dM_specific[j]      # frequency-dependent reactive term
        Za = OR._resonator_Za(freqs, g["r_neck"], g["L_neck"], V, g["Q"]) + (excess_specific / S)
        beta += OR._group_density(g, D) / Za
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(beta) > 0, 1.0 / beta, 1e30 + 0j)


def _alpha_band_NL(groups, D, dR_specific, dM_specific, n_angle=48):
    centers = np.array(O2.DESIGN_BAND_HZ, float)
    fine = np.linspace(O2.DESIGN_BAND_HZ[0] * 0.9, O2.DESIGN_BAND_HZ[-1] * 1.05, 900)
    a_fine = O2.diffuse_alpha(_z_surface_NL(fine, groups, D, dR_specific, dM_specific), n_angle=n_angle)
    return np.interp(centers, fine, a_fine)


def _scalar(x):
    return float(x["worst"]) if isinstance(x, dict) else float(x)


def relabel_design(design, cond, per_bank_RX, per_bank_anchor, *, forcing="resonant_tone_upper_bound"):
    """COMPLETE-DESIGN nonlinear relabel under ONE forcing bound. per_bank_RX: list (len=#banks) of validated
    (u_amp, R_phasor, X_phasor) tables or None (uncorrected). per_bank_anchor: list of each bank's validated
    low-amplitude (target-0.5) MEASURED (u_amp, R, X) reference, or None. FAILS CLOSED: if ANY bank is not
    'ok', the design-level nonlinear dOASPL/target verdict are None (per-bank diagnostics retained),
    status='incomplete'."""
    if forcing not in ("resonant_tone_upper_bound", "broadband_equivalent"):
        raise ValueError(f"forcing must be 'resonant_tone_upper_bound' or 'broadband_equivalent'; got {forcing!r}")
    groups = design["groups"]
    spl = np.array(cond["spl_dB"], float); bands = np.array(O2.DESIGN_BAND_HZ, float)
    dR = np.zeros(len(groups)); dM = np.zeros(len(groups)); per_bank = []
    for j, g in enumerate(groups):
        S = RM.neck_area(g["r_neck"]); Le = RM.effective_neck_length(g["L_neck"], g["r_neck"])
        V = RM.cavity_volume_for_target_f0(g["f0"], S, Le)
        Za0 = OR._resonator_Za(np.array([g["f0"]]), g["r_neck"], g["L_neck"], V, g["Q"])[0]
        bank = dict(f0_Hz=g["f0"], S_neck_m2=float(S), z_linear_complex_Rayl=complex(Za0 * S))
        if per_bank_RX[j] is None:
            per_bank.append(dict(bank=j, status="uncorrected")); continue
        anchor = per_bank_anchor[j] if per_bank_anchor is not None else None
        p_scale = 1.0 if forcing == "resonant_tone_upper_bound" else float(np.sqrt(1.0 / (g["Q"] * THIRD_OCT_BW)))
        sc = self_consistent_U(bank, float(spl[int(np.argmin(np.abs(bands - g["f0"])))]), per_bank_RX[j],
                               anchor, p_scale=p_scale)
        sc["bank"] = j
        if sc["status"] == "ok":
            dR[j] = sc["dR_excess_Rayl"]; dM[j] = sc["dM_inertance_Rayl_s"]
        per_bank.append(sc)
    complete = all(p.get("status") == "ok" for p in per_bank)
    a_lin = O2.liner_diffuse_alpha_band(groups, design["D"])
    a_nl = _alpha_band_NL(groups, design["D"], dR, dM)
    f_cov = float(cond["f_cov"] if "f_cov" in cond else design.get("f_cov", 1.0))
    args = (cond["spl_dB"], f_cov, cond["A_wall"], cond["kappa_nominal"])
    dz_lin = _scalar(O2.robust_delta_oaspl(a_lin, *args))
    dz_nl = _scalar(O2.robust_delta_oaspl(a_nl, *args))
    tgt = cond.get("dOASPL_target")
    return dict(forcing=forcing, status=("complete" if complete else "incomplete"), per_bank=per_bank,
                dR_excess_Rayl=[round(float(x), 5) for x in dR],
                dM_inertance_Rayl_s=[round(float(x), 8) for x in dM],
                dOASPL_linear=round(dz_lin, 4),
                dOASPL_nonlinear=(round(dz_nl, 4) if complete else None),
                dOASPL_nonlinear_provisional=round(dz_nl, 4),
                dOASPL_drop=(round(dz_lin - dz_nl, 4) if complete else None), target=tgt,
                target_met_nonlinear=((bool(dz_nl >= tgt) if complete else None) if tgt is not None else None),
                banks_not_ok=[p["bank"] for p in per_bank if p.get("status") != "ok"])


def relabel_design_interval(design, cond, per_bank_RX, per_bank_anchor):
    """Run BOTH forcing bounds -> nonlinear attenuation INTERVAL [broadband_equiv .. resonant_tone_upper_bound]."""
    bb = relabel_design(design, cond, per_bank_RX, per_bank_anchor, forcing="broadband_equivalent")
    tn = relabel_design(design, cond, per_bank_RX, per_bank_anchor, forcing="resonant_tone_upper_bound")
    tgt = cond.get("dOASPL_target")
    complete = bb["status"] == "complete" and tn["status"] == "complete"
    if complete:
        lo, hi = sorted([bb["dOASPL_nonlinear"], tn["dOASPL_nonlinear"]])
        interval, met = [round(lo, 4), round(hi, 4)], (bool(lo >= tgt) if tgt is not None else None)
    else:
        interval, met = None, None
    return dict(status=("complete" if complete else "incomplete"), dOASPL_nonlinear_interval=interval,
                dOASPL_linear=bb["dOASPL_linear"], target=tgt, target_met_worst_bound=met,
                banks_not_ok=sorted(set(bb["banks_not_ok"]) | set(tn["banks_not_ok"])),
                broadband=bb, resonant_tone=tn)
