"""Cycle-resolved impedance histories and pressure-flow relationship of the visualization cases (read-only).

Uses the saved probe histories of the AUTHORITATIVE selected cases (no new CFD); a case continued by the statistical
follow-up uses its continuation record and the follow-up's stitched block estimate. Nothing is written next to the
source cases. Output: artifacts/dev/vehicle_v1/figures/cfd_viz/ (no titles or footnotes on the images; they go to
the bookkeeping notes merged into TITLES.md / FOOTNOTES.md by acoustic-liner figures cfd-fields).

    acoustic-liner figures cfd-histories
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .. import paths
ROOT = paths.ROOT
from ..vehicles import cfd_supplemental as S  # noqa: E402
from . import cfd_fields as VF  # noqa: E402

VDIR = ROOT / "artifacts/dev/vehicle_v1"
OUT = VF.OUT
RHO = VF.RHO
NOTES = {}
COLS = ("low", "ustar", "high")


def history(c):
    """(case dir for probe series, block estimate)."""
    m = c["final_status"]["member"]
    if m is None:
        return ROOT / c["source_dir"], S.block_estimate(str(ROOT / c["source_dir"]))
    fol = VDIR / VF.VEH[c["vehicle"]] / "statistical_followup"
    rep = next(r for r in json.loads((fol / "_followup_result.json").read_text(encoding="utf-8"))["member_reports"]
               if r["member"] == m)
    return fol / "continuation" / m, rep["estimate"]


def waveforms(src, f0):
    """Last two complete saved cycles: phase (deg), delta p (Pa), downward neck mean velocity (m/s)."""
    g = VF.geometry(src)
    a_w = 0.5 * math.radians(g["wedge_deg"]) * g["a"] ** 2
    t, pu = VF.series(src, "pUp")
    _, pdn = VF.series(src, "pDown")
    tq, q = VF.series(src, "fluxMid")
    n_end = int(np.floor(t[-1] * f0 + 1e-3))
    m = t >= (n_end - 2) / f0
    mq = tq >= (n_end - 2) / f0
    return (360.0 * (f0 * t[m] - (n_end - 2)), RHO * (pu[m] - pdn[m]),
            360.0 * (f0 * tq[mq] - (n_end - 2)), -q[mq] / a_w, (n_end - 1, n_end))


def label(c):
    st = c["final_status"]
    s = VF.short(c) + (" (continued)" if st["member"] else "")
    return s + (f"  [failed: {', '.join(st['failed'])}; NOT CONVERGED]" if not st["converged"] else "")


def fail_box(ax, c):
    st = c["final_status"]
    if not st["converged"]:
        ax.text(0.5, 0.04, "CFD CASE NOT CONVERGED\nfailed " + ", ".join(st["failed"]) +
                ("\nafter its single permitted continuation" if st["member"] else ""), transform=ax.transAxes,
                ha="center", va="bottom", fontsize=6.4, color="w", fontweight="bold", multialignment="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="#c62828", ec="none"))


def tags(fig):
    fig.text(0.125, 0.895, "VEHICLE VERDICT: UNRESOLVED   |   CFD case status per legend   |   impedance R, X from "
             "the authoritative records", fontsize=8, va="bottom", fontweight="bold", color="#263238")


def ordered(cases, slot):
    return sorted([c for c in cases if c["slot"] == slot],
                  key=lambda c: ["nominal", "sens_refine", "sens_halfdt"].index(c["case"].split("/")[0]))


def cycle_histories(veh, cases):
    fig, ax = plt.subplots(4, 3, figsize=(15, 13), sharex="row")
    titles = []
    for j, slot in enumerate(COLS):
        members = ordered(cases, slot)
        for c in members:
            src, est = history(c)
            f0 = c["f0_Hz"]
            k = np.arange(est["discarded"] + 1, est["discarded"] + 1 + len(est["R_blocks"]))
            ls = "-" if c["final_status"]["converged"] else "--"
            for i, (key, h) in enumerate((("R", "h_R"), ("X", "h_X"))):
                ln, = ax[i, j].plot(k, est[key + "_blocks"], "o" + ls, ms=3.5, label=label(c))
                ax[i, j].axhline(est[key], color=ln.get_color(), lw=0.8, ls=":")
                ax[i, j].axhspan(est[key] - est[h], est[key] + est[h], color=ln.get_color(), alpha=0.12)
            ph, dp, phq, u, _ = waveforms(src, f0)
            ax[2, j].plot(ph, dp, ls, lw=1, label=label(c))
            ax[3, j].plot(phq, u, ls, lw=1, label=label(c))
            if slot == "high":
                for i in range(4):
                    fail_box(ax[i, j], c)
        titles.append(f"column {j + 1}: {VF.SLOT_LABEL[slot]} (" + ", ".join(c["case"] for c in members) + ")")
        for i, yl in enumerate(["block resistance R_k (Rayl)", "block reactance X_k (Rayl)",
                                "Δp = ρ (p_up − p_down) (Pa)", "neck mean velocity u, downward (m/s)"]):
            ax[i, j].grid(alpha=0.3)
            if j == 0:
                ax[i, j].set_ylabel(yl)
        ax[0, j].set_xlabel("cycle")
        ax[1, j].set_xlabel("cycle")
        for i in (2, 3):
            ax[i, j].set_xlabel("phase θ over the last two saved cycles (deg)")
            ax[i, j].set_xticks(range(0, 721, 180))
        ax[0, j].legend(fontsize=6.3, loc="upper left", framealpha=0.85)
    tags(fig)
    name = f"{veh}_cycle_histories.png"
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    NOTES[name] = dict(titles=[f"{VF.VNAME[veh]}: cycle-by-cycle block R and X with the block mean (dotted) and its "
                               "uncertainty half-width (band), pressure difference and neck mean velocity versus "
                               "phase; low / nearest-U* / high speed (columns); the U* column overlays the nominal, "
                               "refined-mesh and half-time-step cases"] + titles,
                       notes=[VF.SCOPE, VF.VERDICT_NOTE[veh], NOTE_SRC, NOTE_U])


def pressure_flow(veh, cases):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.6))
    titles = []
    for j, slot in enumerate(COLS):
        for c in ordered(cases, slot):
            src, est = history(c)
            ph, dp, phq, u, _ = waveforms(src, c["f0_Hz"])
            ls = "-" if c["final_status"]["converged"] else "--"
            ln, = ax[0, j].plot(ph, dp / np.max(np.abs(dp)), ls, lw=1.1,
                                label="Δp / max|Δp|, " + VF.short(c))
            ax[0, j].plot(phq, u / np.max(np.abs(u)), ls, lw=1.1, color=ln.get_color(), alpha=0.45,
                          label="u / max|u|, " + VF.short(c))
            ui = np.interp(ph, phq, u)
            argz = math.degrees(math.atan2(est["X"], est["R"]))
            ax[1, j].plot(ui, dp, ls, lw=1.0, color=ln.get_color(),
                          label=label(c) + f"\nR = {est['R']:.2f} ± {est['h_R']:.2f}, X = {est['X']:.2f} ± "
                          f"{est['h_X']:.2f} Rayl, arg Z = {argz:.1f}°")
            if slot == "high":
                fail_box(ax[0, j], c)
                fail_box(ax[1, j], c)
        ax[0, j].set_xlabel("phase θ over the last two saved cycles (deg)")
        ax[0, j].set_xticks(range(0, 721, 90))
        ax[0, j].set_ylabel("normalized waveform" if j == 0 else "")
        ax[0, j].legend(fontsize=6, loc="upper right", framealpha=0.85)
        ax[1, j].set_xlabel("neck mean velocity u, downward (m/s)")
        ax[1, j].set_ylabel("Δp (Pa)" if j == 0 else "")
        lo, hi = ax[1, j].get_ylim()
        ax[1, j].set_ylim(lo, hi + 0.5 * (hi - lo))
        ax[1, j].legend(fontsize=5.8, loc="upper left", framealpha=0.85)
        for a in ax[:, j]:
            a.grid(alpha=0.3)
        titles.append(f"column {j + 1}: {VF.SLOT_LABEL[slot]}")
    tags(fig)
    name = f"{veh}_pressure_flow.png"
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    NOTES[name] = dict(titles=[f"{VF.VNAME[veh]}: pressure-flow relationship. Top: normalized pressure difference "
                               "(solid colour) and neck mean velocity (faded) versus phase. Bottom: Δp versus u "
                               "loops; the loop opening reflects the phase lead arg Z = atan(X/R) of the complex "
                               "impedance Z = R + iX"] + titles,
                       notes=[VF.SCOPE, VF.VERDICT_NOTE[veh], NOTE_SRC, NOTE_U,
                              "R, X and their half-widths are the block estimates of the authoritative records "
                              "(analysis cycles 14-25, or the follow-up continuation record); arg Z > 0 means the "
                              "pressure difference leads the velocity."])


NOTE_SRC = ("Histories are the authoritative probe records of the selected cases, read-only; cases continued by the "
            "statistical follow-up ('continued') show the continuation record and the follow-up's stitched block "
            "estimate. Dashed lines and red boxes: cases whose final recorded status failed a stability gate; they are "
            "not presented as converged. Phase reference: imposed inlet velocity u_z = -U_in sin(2 pi f0 t). p in the "
            "CFD is kinematic; Δp multiplied by rho = 1.204 kg/m^3.")
NOTE_U = ("Neck mean velocity u = -(volume flow through the 5-degree wedge sector at the neck mid-plane) / (wedge "
          "sector of the neck cross-section, 0.5 * wedge angle * a^2), positive downward (from the inlet side to the "
          "outlet side). The earlier version of this figure plotted the wedge-sector volume flow (1/72 of the full "
          "ring) under the label 'neck volume flow'; corrected here.")


def main():
    from .. import cli
    if cli.show_help("figures", "cfd-histories"):
        return
    pl = VF.plan()
    for veh in VF.VEH:
        cases = [c for c in pl["cases"] if c["vehicle"] == veh]
        cycle_histories(veh, cases)
        pressure_flow(veh, cases)
    nb = OUT / "_notes.json"
    alln = json.loads(nb.read_text(encoding="utf-8")) if nb.exists() else {}
    alln.update(NOTES)
    nb.write_text(json.dumps(alln, indent=1), encoding="utf-8")
    print("written", sorted(NOTES))


if __name__ == "__main__":
    main()
