"""Global FEM acoustic-field figure and harmonic animation (VISUALIZATION ONLY; reads no evidence it could change).

    acoustic-liner figures fem falcon

Computes the bare and lined complex fields with `visualization.fem_fields` (verified against the production solver and
the frozen initial-FEM record), exports VTU + manifest, and renders a 4-panel still and a synchronized 36-frame
bare/lined animation into artifacts/dev/vehicle_v1/figures/fem_viz/<vehicle>/. No titles or footnotes on the images.
"""
import io
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from .. import paths
ROOT = paths.ROOT
from . import fem_fields as FV  # noqa: E402
from . import cfd_fields as VZ  # noqa: E402

OUT = ROOT / "artifacts/dev/vehicle_v1/figures/fem_viz"
NFRAMES = 36
LEVEL_RANGE_DB = 60.0      # shared level colour range of panels (a)-(c), below the rounded maximum
ANIM_PCT = 99.0           # animation colour limit: this percentile of |p| over bare + lined nodes
REL_CLIP_DB = 40.0         # panel (d) colour clipped at +/- this (fraction clipped is recorded)
DPI = 130
SOLID, LINER, BARE, SRC = "0.62", "tab:purple", "0.25", "#d32f2f"
VNAME = {"falcon": "Falcon 9", "atlas": "Atlas I representative"}


class View:
    """Mirrored (display-only) triangulation of the axisymmetric (r, z) mesh."""

    def __init__(self, R):
        m = R["mesh"]
        r, z = m.p
        n = len(r)
        self.x = np.concatenate([r, -r])
        self.y = np.concatenate([z, z])
        t = m.t.T
        self.tri = mtri.Triangulation(self.x, self.y, np.vstack([t, t[:, ::-1] + n]))
        self.R = R
        self.g = R["vs"].geometry
        self.pc = R["payload"]

    def both(self, v):
        return np.concatenate([v, v])


def outline(ax, V, lined, labels=False):
    g, pc, R = V.g, V.pc, V.R
    m, sets = R["mesh"], R["tagged"]["sets"]
    for side in (1, -1):
        for fid in sets["wall"]:
            a, b = m.facets[:, fid]
            ax.plot(side * m.p[0, [a, b]], m.p[1, [a, b]], color=LINER if lined else BARE,
                    lw=3.2 if lined else 2.0, solid_capstyle="round", zorder=6)
        x0, x1 = pc["r_in_m"], pc["r_out_m"]
        ax.add_patch(Rectangle((side * x0 if side > 0 else -x1, pc["z_base_m"]), x1 - x0,
                               pc["z_top_m"] - pc["z_base_m"], fc=SOLID, ec="k", lw=0.8, hatch="////", zorder=5))
    ax.plot([-g.R_fair, g.R_fair], [0, 0], color=SRC, lw=3.4, zorder=7, solid_capstyle="butt")
    ax.plot([0, 0], [-0.2, g.H_total + 0.2], color="k", lw=0.9, ls=(0, (6, 3, 1, 3)), zorder=4)
    ax.set_xlim(-g.R_fair - 0.35, g.R_fair + 0.35)
    ax.set_ylim(-0.75, g.H_total + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    if labels:
        box = dict(fc="w", ec="none", alpha=0.8, pad=1.0)
        zc = 0.5 * (pc["z_base_m"] + pc["z_top_m"])
        ax.text(0, zc, "payload\n(rigid)", ha="center", va="center", fontsize=7, zorder=8, bbox=box)
        ax.text(0, -0.42, "source plane z = 0 (incident wave)", ha="center", va="center", fontsize=6.6,
                color=SRC, zorder=8)
        ax.text(0.12, g.H_total - 1.1, "axis of\nsymmetry", fontsize=6, zorder=8, bbox=box)


def scalebar(ax, V):
    x = V.g.R_fair - 1.0
    ax.plot([x, x + 1.0], [V.g.H_total - 0.2] * 2, color="k", lw=2, zorder=8)
    ax.text(x + 0.5, V.g.H_total - 0.05, "1 m", ha="center", va="bottom", fontsize=6.5)


def tags(fig, extra):
    x = 0.01
    for txt, col in (("ILLUSTRATIVE VISUALIZATION ONLY", "#6d4c41"), ("VEHICLE VERDICT: UNRESOLVED", "#455a64"),
                     extra):
        t = fig.text(x, 0.992, txt, fontsize=7.5, color="w", fontweight="bold", va="top",
                     bbox=dict(boxstyle="round,pad=0.3", fc=col, ec="none"))
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(fig.transFigure.inverted())
        x = bb.x1 + 0.012


def still(R, out):
    V = View(R)
    D = FV.derived(R)
    lev = lambda v: 20.0 * np.log10(np.maximum(np.abs(v), 1e-12))
    Lb, Lt, Ld = lev(R["p_bare"]), lev(R["p_treated"]), lev(D["diff"])
    top = float(np.ceil(max(Lb.max(), Lt.max(), Ld.max()) / 5.0) * 5.0)
    lo = top - LEVEL_RANGE_DB
    f = R["sel"]["freq"]
    fig, axs = plt.subplots(1, 4, figsize=(12.5, 8.6), gridspec_kw=dict(wspace=0.04))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.955, bottom=0.06)
    lvl = "level re p_inc, 20 log₁₀(|p| / p_inc) (dB)"
    panels = [
        ("(a) bare wall", Lb, "viridis", Normalize(lo, top), lvl, False, "min"),
        ("(b) lined wall", Lt, "viridis", Normalize(lo, top), lvl, True, "min"),
        ("(c) lined − bare", Ld, "magma", Normalize(lo, top), "20 log₁₀(|p̂_lined − p̂_bare| / p_inc) (dB)", True,
         "min"),
        ("(d) relative level", D["rel_dB"], "RdBu_r", TwoSlopeNorm(0.0, -REL_CLIP_DB, REL_CLIP_DB),
         "20 log₁₀(|p_lined| / |p_bare|) (dB)", True, "both")]
    for ax, (lab, v, cmap, norm, cl, lined, ext) in zip(axs, panels):
        im = ax.tripcolor(V.tri, V.both(v), cmap=cmap, norm=norm, shading="gouraud", rasterized=True)
        outline(ax, V, lined, labels=(ax is axs[0]))
        ax.text(-V.g.R_fair - 0.3, V.g.H_total + 0.95, lab, fontsize=9, fontweight="bold", va="top")
        if lined:
            ax.text(V.g.R_fair - 0.05, V.g.H_barrel * 0.5, "lined\nwall", color=LINER, fontsize=6.6, ha="right",
                    va="center", zorder=8, bbox=dict(fc="w", ec="none", alpha=0.8, pad=1.0))
        else:
            ax.text(V.g.R_fair - 0.05, V.g.H_barrel * 0.5, "bare\nwall", color=BARE, fontsize=6.6, ha="right",
                    va="center", zorder=8, bbox=dict(fc="w", ec="none", alpha=0.8, pad=1.0))
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.035, pad=0.015, aspect=22, extend=ext)
        cb.set_label(cl, fontsize=7.2)
        cb.ax.tick_params(labelsize=6.5)
    scalebar(axs[0], V)
    tags(fig, (f"FEM HARMONIC FIELD, {f:.0f} Hz (frozen mesh)", "#1565c0"))
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return dict(level_limits_dB=[lo, top], level_shared_abc=True, rel_dB_clip=REL_CLIP_DB,
                rel_dB_clipped_fraction=float(np.mean(np.abs(D["rel_dB"]) > REL_CLIP_DB)))


def animation(R, out):
    V = View(R)
    amp = np.abs(np.concatenate([R["p_bare"], R["p_treated"]]))
    vmax = float(np.percentile(amp, ANIM_PCT))
    f = R["sel"]["freq"]
    frames = []
    for k in range(NFRAMES):
        wt = 2.0 * math.pi * k / NFRAMES
        fig, axs = plt.subplots(1, 2, figsize=(7.4, 8.4), gridspec_kw=dict(wspace=0.03))
        fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.11)
        for ax, (lab, p, lined) in zip(axs, (("bare wall", R["p_bare"], False), ("lined wall", R["p_treated"], True))):
            v = p.real * math.cos(wt) - p.imag * math.sin(wt)
            im = ax.tripcolor(V.tri, V.both(v), cmap="RdBu_r", norm=Normalize(-vmax, vmax), shading="gouraud")
            outline(ax, V, lined, labels=(ax is axs[0]))
            ax.text(-V.g.R_fair - 0.3, V.g.H_total + 0.95, lab, fontsize=9, fontweight="bold", va="top")
        cb = fig.colorbar(im, ax=list(axs), orientation="horizontal", fraction=0.03, pad=0.015, aspect=40, extend="both")
        cb.set_label("normalized instantaneous pressure p(t) / p_inc  (same scale both panels)", fontsize=7.2)
        cb.ax.tick_params(labelsize=6.5)
        x = 0.01
        for txt, col in (("ILLUSTRATIVE VISUALIZATION ONLY", "#6d4c41"), ("VEHICLE VERDICT: UNRESOLVED", "#455a64")):
            t = fig.text(x, 0.992, txt, fontsize=7, color="w", fontweight="bold", va="top",
                         bbox=dict(boxstyle="round,pad=0.3", fc=col, ec="none"))
            fig.canvas.draw()
            x = t.get_window_extent().transformed(fig.transFigure.inverted()).x1 + 0.012
        fig.text(0.01, 0.945, f"harmonic FEM field, {f:.0f} Hz   ωt = {math.degrees(wt):5.1f}°   "
                 f"(reconstructed from the complex solution, not a transient simulation)", fontsize=7.4, va="top")
        buf = io.BytesIO()
        fig.savefig(buf, dpi=95)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
        if k in (0, 9):
            frames[-1].save(out.with_name(out.stem + f"_frame{k:02d}.png"))
    VZ.write_gif(frames, out)
    return dict(n_frames=NFRAMES, limits=[-vmax, vmax], limit_rule=f"{ANIM_PCT}th percentile of |p| over both states; clipped beyond",
                clipped_node_fraction=float(np.mean(amp > vmax)), fps=VZ.FPS, frames_per_cycle=NFRAMES)


PUBLIC = ROOT / "docs/figures/fem"


def captions(veh, man):
    """Caption and footnotes (bookkeeping; never drawn on the images), numbers taken from the manifest."""
    st, an, sel = man["render"]["still"], man["render"]["animation"], man["selection"]
    f = man["frequency_hz"]
    still_cap = (
        f"{VNAME[veh]}: global harmonic acoustic response of the payload-resolving fairing model at {f:.0f} Hz, on "
        f"the frozen {sel['mesh_size']:g} m P1 axisymmetric FEM mesh. The bare and lined cases use the same source, "
        "mesh, payload geometry and colour scale. The lined wall uses the frozen finalist's linear resonator-model "
        "impedance (not CFD-corrected). (a) and (b): normalized pressure level re the incident source amplitude. "
        "(c): magnitude of the complex-field change |p̂_lined − p̂_bare|; it shows where the field changed, not "
        "whether it became quieter or louder. (d): 20 log₁₀(|p_lined| / |p_bare|); only this panel indicates quieter "
        "(blue) versus louder (red). Colours show normalized acoustic pressure, not calibrated flight SPL. These "
        "fields visualize the modeled mechanism and do not change the formally UNRESOLVED vehicle-level verdict.")
    anim_cap = (
        f"{VNAME[veh]}: one harmonic cycle at {f:.0f} Hz, bare (left) and lined (right) on the same phase and colour "
        "scale, reconstructed from the complex FEM solution as p(t) = Re p̂ cos ωt − Im p̂ sin ωt (36 frames); not a "
        "transient simulation.")
    notes = [
        f"Colour clipping. Still, panel (d): clipped at ±{st['rel_dB_clip']:.0f} dB "
        f"({100 * st['rel_dB_clipped_fraction']:.1f}% of nodes). Panels (a)-(c): shared range "
        f"{st['level_limits_dB'][0]:.0f} to {st['level_limits_dB'][1]:.0f} dB re p_inc, lower values shown at the "
        f"bottom colour. Animation: limit = 99th percentile of |p| over both states (±{an['limits'][1]:.2f} p_inc); "
        f"{100 * an['clipped_node_fraction']:.1f}% of nodes exceed it at peak.",
        f"Mesh: the frozen {sel['mesh_size']:g} m P1 (linear triangle) axisymmetric mesh of the authoritative "
        f"initial-FEM record ({sel['source_corner']} | {sel['payload_case']}), mesh sha256 "
        f"{man['mesh']['mesh_sha256'][:16]}..., {man['mesh']['n_nodes']} nodes. The left half of each panel is a "
        "mirror image for display.",
        f"Frequency selection (fixed before rendering): the band with the largest share of the treated-versus-bare "
        f"broadband reduction at the record's robust corner {'|'.join(sel['robust_corner'])}; {f:.0f} Hz carries "
        f"{100 * sel['share'][sel['index']]:.0f}%. Ties go to the lower frequency.",
        f"Verification: the exported field reproduces the production solver's mean-square pressure and the frozen "
        f"record's {man['verification']['nr_record_dB']:.3f} dB band reduction (difference "
        f"{man['verification']['nr_abs_err_dB']:.1e} dB).",
        "CFD-corrected state: " + man["cfd_corrected_state"] + ".",
        "Visualization only: not read by any verdict, sensitivity or evidence record; fem_solver.py and "
        "vehicle_fem.py are unmodified."]
    return still_cap, anim_cap, notes


def write_books(veh, man, od):
    still_cap, anim_cap, notes = captions(veh, man)
    (od / "TITLES.md").write_text(f"# Titles\n\n- `{veh}_global_fem_acoustic_field.png`: {still_cap}\n"
                                  f"- `{veh}_global_fem_acoustic_dynamics.gif`: {anim_cap}\n", encoding="utf-8")
    (od / "FOOTNOTES.md").write_text("# Footnotes\n\n" + "".join(f"{i + 1}. {n}\n" for i, n in enumerate(notes)),
                                     encoding="utf-8")


def publish():
    import shutil
    PUBLIC.mkdir(parents=True, exist_ok=True)
    lines = ["# Global FEM acoustic-field figures", "",
             "Visualization only. Both vehicle verdicts remain **UNRESOLVED**. Full exports (ParaView VTU) and "
             "manifests: `artifacts/dev/vehicle_v1/figures/fem_viz/` (not in git).", ""]
    for veh in ("falcon", "atlas"):
        od = OUT / veh
        mp = od / f"{veh}_fem_field_manifest.json"
        if not mp.exists():
            continue
        man = json.loads(mp.read_text())
        still_cap, anim_cap, notes = captions(veh, man)
        for n in (f"{veh}_global_fem_acoustic_field.png", f"{veh}_global_fem_acoustic_dynamics.gif"):
            shutil.copy2(od / n, PUBLIC / n)
            if FV.sha_file(PUBLIC / n) != man["outputs"][n]:
                raise SystemExit(f"{n}: published copy does not match the manifest")
        lines += [f"## {VNAME[veh]}" + ("" if veh == "falcon" else " (supplementary)"), "",
                  f"![{veh} field]({veh}_global_fem_acoustic_field.png)", "", still_cap, "",
                  f"![{veh} animation]({veh}_global_fem_acoustic_dynamics.gif)", "", anim_cap, "",
                  *[f"{i + 1}. {n}" for i, n in enumerate(notes)], "",
                  f"Manifest sha256: `{FV.sha_file(mp)}`", ""]
    (PUBLIC / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("published", sorted(p.name for p in PUBLIC.iterdir()))


def main(veh):
    od = OUT / veh
    R = FV.compute(veh)
    man = FV.export(R, od)
    s = still(R, od / f"{veh}_global_fem_acoustic_field.png")
    a = animation(R, od / f"{veh}_global_fem_acoustic_dynamics.gif")
    man["render"] = dict(still=s, animation=a, dpi=DPI, mirror="display only",
                         script_sha256=FV.sha_file(__file__))
    for p in (od / f"{veh}_global_fem_acoustic_field.png", od / f"{veh}_global_fem_acoustic_dynamics.gif"):
        man["outputs"][p.name] = FV.sha_file(p)
    (od / f"{veh}_fem_field_manifest.json").write_text(json.dumps(man, indent=1, default=float))
    write_books(veh, man, od)
    print(veh, R["sel"]["freq"], R["verification"], s, a)


if __name__ == "__main__":
    from .. import cli
    if cli.show_help("figures", "fem"):
        raise SystemExit(0)
    arg = sys.argv[1] if len(sys.argv) > 1 else "falcon"
    publish() if arg == "publish" else main(arg)
