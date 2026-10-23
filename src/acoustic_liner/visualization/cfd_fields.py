"""Visualization layer for the visualization-only CFD continuations (NON-AUTHORITATIVE, descriptive only).

Reads the OpenFOAM fields of the visualization copies (~/visualization_cases, prepared by acoustic-liner figures cfd-cases and
run by run_viz.sh) directly, and draws every product with matplotlib on the cell-centre data. Reads frozen records
(follow-up member reports, block estimates) only to label cases. Writes only under
artifacts/dev/vehicle_v1/figures/cfd_viz/ (gallery) and, with `publish`, docs/figures/cfd/ (README selection).
Nothing here is read by impedance extraction, sensitivity, corrected FEM or verdict code.

    acoustic-liner figures cfd-fields extract      # cell data of the 2-cycle window -> _cache/*.npz
    acoustic-liner figures cfd-fields render       # stills, montages, animations, geometry, histories
    acoustic-liner figures cfd-qa qa               # checks; writes visualization_manifest.json if all pass
    acoustic-liner figures cfd-qa publish          # copy the README selection to docs/figures/cfd/

Phase: theta = 360 f0 t mod 360 deg (f0 = imposed frequency of the case). The imposed inlet velocity is
u_z = -U_in sin(theta) (z up; inlet at the top), so the downward neck velocity is u_down = +U sin(theta); the
measured fundamental phase of the neck flow in each window is recorded to confirm the reference.
"""
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle, FancyArrowPatch  # noqa: E402
from PIL import Image  # noqa: E402

from .. import paths
ROOT = paths.ROOT
from ..physics import resonator as RM  # noqa: E402

VDIR = ROOT / "artifacts/dev/vehicle_v1"
PLAN = VDIR / "visualization_cases/_viz_plan.json"
OUT = VDIR / "figures/cfd_viz"
CACHE = OUT / "_cache"
PUBLIC = ROOT / "docs/figures/cfd"
WORK = Path(os.environ.get("ACOUSTIC_LINER_VISUALIZATION_CASES") or Path.home() / "visualization_cases")
VEH = {"falcon": "falcon9_2025_unblanketed_v2", "atlas": "atlas_i_14ft_representative_unblanketed"}
VNAME = {"falcon": "Falcon 9", "atlas": "Atlas I representative"}
RHO = 1.204                 # kg/m^3 (physics.resonator.RHO); the CFD itself uses kinematic p and nu only
EPS_W = 10.0                # 1/s, pinned epsilon for log10(|omega| + eps)
SNAP = 48                   # snapshots per cycle
FPS = 12                    # animation frame rate (2 cycles = 96 frames = 8 s)
PHASES = (0, 90, 180, 270)
DPI = 110
SCOPE = ("These visualization-only continuations illustrate local, phase-resolved flow in a representative resonator "
         "neck. They show the modeled mechanisms associated with nonlinear impedance\u2014neck acceleration, "
         "separation, vortex formation and viscous dissipation. They are not full-fairing acoustic CFD, do not "
         "simulate propagating sound through the fairing and are not used as numerical evidence in the vehicle "
         "verdict. Global attenuation is evaluated separately using the payload-resolving FEM with CFD-derived "
         "impedance corrections.")
VERDICT_NOTE = {
    "falcon": "The Falcon vehicle verdict remains UNRESOLVED because the high-speed 16.2 m/s impedance pair did not "
              "satisfy the prescribed statistical-stability checks after its single permitted continuation.",
    "atlas": "The Atlas vehicle verdict remains UNRESOLVED because a conservative resistance-bound uncertainty curve "
             "failed the prescribed global fixed-point admissibility check, despite favorable central CFD-corrected "
             "FEM results."}
COUPLED_CAPTION = ("The oscillating pressure difference drives flow through the resonator neck. Sharp-edge separation "
                   "generates vortical structures, while modeled viscous dissipation concentrates near the neck edges "
                   "and shed vortices. The amplitude and phase relationship between pressure and volume flow defines "
                   "the complex impedance R+iX transferred to the global FEM.")
COUPLED_LIMIT = ("This incompressible local CFD model imposes oscillatory neck flow. It does not simulate propagation "
                 "of an acoustic wave through the complete payload fairing.")
FIELDS = {  # key: (colorbar label, kind, cmap)
    "umag": ("|U| (m/s)", "linear", "viridis"),
    "pk": ("kinematic pressure p/\u03c1 (m\u00b2/s\u00b2)", "diverging", "RdBu_r"),
    "logw": ("log\u2081\u2080(|\u03c9| + \u03b5) , |\u03c9| in 1/s, \u03b5 = 10 1/s", "log", "inferno"),
    "wth": ("signed azimuthal vorticity \u03c9\u03b8 (1/s)", "diverging", "PuOr_r"),
    "phi": ("log\u2081\u2080 modeled viscous-dissipation density \u03a6 (W/m\u00b3)", "log", "plasma"),
}
SHORT_LABEL = {"umag": "|U| (m/s)", "pk": "p/ρ (m²/s²)", "logw": "log₁₀(|ω|+ε)",
               "wth": "ωθ (1/s)", "phi": "log₁₀ Φ (W/m³)"}
SLOT_LABEL = {"low": "low speed", "ustar": "nearest U*", "high": "high speed"}
NOTES = {}                  # output name -> {"titles": [...], "notes": [...]}


# ------------------------------------------------------------------ reading
def plan():
    return json.loads(PLAN.read_text(encoding="utf-8"))


def read_internal(path):
    txt = Path(path).read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", txt)
    if not m:
        raise ValueError(f"{path}: no nonuniform internalField")
    n = int(m.group(2))
    end = txt.index("\n)", m.end())
    a = np.array(txt[m.end():end].replace("(", " ").replace(")", " ").split(), dtype=float)
    return a.reshape(n, -1) if a.size != n else a


def geometry(case_dir):
    t = (Path(case_dir) / "system/blockMeshDict").read_text()
    v = [tuple(map(float, x)) for x in re.findall(r"\(([-\d.e]+) ([-\d.e]+) ([-\d.e]+)\)", t.split("blocks")[0])]
    zs = sorted({round(z, 9) for _, _, z in v})
    rs = sorted({round(math.hypot(x, y), 9) for x, y, _ in v})
    half = max(math.atan2(abs(y), x) for x, y, _ in v if x > 0)
    return dict(z0=zs[0], z1=zs[1], z2=zs[2], z3=zs[3], a=rs[1], R=rs[2], wedge_deg=math.degrees(2 * half))


def series(case_dir, name, col=1):
    d = Path(case_dir) / "postProcessing" / name
    subs = sorted((x for x in d.iterdir() if x.is_dir()), key=lambda x: float(x.name))
    a = np.vstack([np.loadtxt(s / "surfaceFieldValue.dat", comments="#", ndmin=2) for s in subs])
    a = a[np.argsort(a[:, 0], kind="stable")]
    _, k = np.unique(a[:, 0], return_index=True)
    return a[k, 0], a[k, col]


def times_of(case_dir):
    return sorted((float(p.name), p.name) for p in Path(case_dir).iterdir()
                  if p.is_dir() and re.match(r"^\d+(\.\d+)?(e-?\d+)?$", p.name) and p.name != "0")


def window(case_dir, f0):
    """The first two complete cycles (integer cycle boundaries) inside the saved snapshots."""
    ts = times_of(case_dir)
    c0 = math.ceil(ts[0][0] * f0 - 1e-6)
    sel = [(t, n) for t, n in ts if c0 - 1e-6 <= t * f0 < c0 + 2 - 1e-6]
    return (c0, c0 + 2), sel


def fundamental(t, y, f0):
    w = 2 * np.pi * f0
    A = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return c[0], c[1]                       # y ~ c0 cos(theta) + c1 sin(theta)


def status(c):
    st = c["final_status"]
    return dict(converged=st["converged"], tag="CFD CASE PASSED STATISTICAL CHECKS" if st["converged"]
                else "CFD CASE NOT CONVERGED", failed=st["failed"], basis=st["basis"])


def bank_of(c):
    return re.search(r"(b\d)_u", c["case"]).group(1)


def short(c):
    kind = c["case"].split("/")[0]
    return {"nominal": "nominal", "sens_refine": "refined mesh", "sens_halfdt": "half time step"}[kind] + \
        f", {c['U_target_m_s']:g} m/s"


# ------------------------------------------------------------------ extract
def extract(only=None):
    CACHE.mkdir(parents=True, exist_ok=True)
    for c in plan()["cases"]:
        if only and c["dir"] not in only:
            continue
        d = WORK / c["dir"]
        f0 = c["f0_Hz"]
        nu = float(re.search(r"nu\s+([\d.eE+-]+);", (d / "constant/transportProperties").read_text()).group(1))
        (c0, c1), sel = window(d, f0)
        first = times_of(d)[0][1]
        C = read_internal(d / first / "C")
        r = np.hypot(C[:, 0], C[:, 1])
        z = C[:, 2]
        n = len(sel)
        F = {k: np.empty((n, len(r)), np.float32) for k in ("umag", "ur", "uz", "pk", "wth", "phi", "q")}
        for i, (t, name) in enumerate(sel):
            U = read_internal(d / name / "U")
            G = read_internal(d / name / "grad(U)").reshape(-1, 3, 3)
            W = read_internal(d / name / "vorticity")
            S = 0.5 * (G + np.transpose(G, (0, 2, 1)))
            F["umag"][i] = np.linalg.norm(U, axis=1)
            F["ur"][i] = (C[:, 0] * U[:, 0] + C[:, 1] * U[:, 1]) / np.maximum(r, 1e-12)
            F["uz"][i] = U[:, 2]
            F["pk"][i] = read_internal(d / name / "p")
            F["wth"][i] = W[:, 1]           # omega_y = omega_theta on the y = 0 meridional plane
            F["phi"][i] = 2.0 * RHO * nu * np.einsum("nij,nij->n", S, S)
            F["q"][i] = read_internal(d / name / "Q")
        ts = np.array([t for t, _ in sel])
        geo = geometry(d)
        tp, pu = series(d, "pUp")
        _, pdn = series(d, "pDown")
        tq, q = series(d, "fluxMid")
        a_w = 0.5 * math.radians(geo["wedge_deg"]) * geo["a"] ** 2      # neck cross-section of the wedge sector
        u_down = -q / a_w
        m = (tq >= c0 / f0 - 1e-9) & (tq < c1 / f0)
        A, B = fundamental(tq[m], u_down[m], f0)
        np.savez_compressed(CACHE / (c["dir"].replace("/", "__") + ".npz"), r=r.astype(np.float32),
                            z=z.astype(np.float32), t=ts, names=np.array([nm for _, nm in sel]), **F,
                            tp=tp, dp_pa=RHO * (pu - pdn), tq=tq, u_down=u_down)
        meta = dict(dir=c["dir"], cycles=[c0, c1], n_frames=n, t_first=ts[0], t_last=ts[-1], nu=nu, geometry=geo,
                    n_cells=len(r), u_down_fundamental=dict(cos=A, sin=B, amp=math.hypot(A, B),
                                                            phase_offset_deg=math.degrees(math.atan2(-A, B))))
        (CACHE / (c["dir"].replace("/", "__") + ".json")).write_text(json.dumps(meta, indent=1))
        print("extracted", c["dir"], n, "frames", f"cycles {c0}-{c1}", flush=True)


def load(c):
    k = c["dir"].replace("/", "__")
    return dict(np.load(CACHE / (k + ".npz"))), json.loads((CACHE / (k + ".json")).read_text())


# ------------------------------------------------------------------ drawing primitives
class Panel:
    """Triangulation of one case (both halves of the meridional section) and the masks for solids/walls."""

    def __init__(self, c, D, meta):
        from scipy.spatial import cKDTree
        g = meta["geometry"]
        self.g, self.c, self.D = g, c, D
        r, z = D["r"].astype(float), D["z"].astype(float)
        # display-only boundary points (walls, facesheet faces, neck wall, domain ends) carrying the value of the
        # nearest cell centre, so the colour reaches the solid boundaries instead of stopping half a cell short
        zu, ru = np.unique(np.round(z, 9)), np.unique(np.round(r, 9))
        tube = zu[(zu < g["z1"]) | (zu > g["z2"])]
        neckz = zu[(zu > g["z1"]) & (zu < g["z2"])]
        ann = ru[ru > g["a"]]
        b = [np.column_stack([np.full_like(tube, g["R"]), tube]), np.column_stack([np.full_like(neckz, g["a"]), neckz]),
             np.column_stack([np.r_[ann, g["R"], g["a"]], np.full(len(ann) + 2, g["z1"])]),
             np.column_stack([np.r_[ann, g["R"], g["a"]], np.full(len(ann) + 2, g["z2"])]),
             np.column_stack([np.r_[ru, g["R"]], np.full(len(ru) + 1, g["z0"])]),
             np.column_stack([np.r_[ru, g["R"]], np.full(len(ru) + 1, g["z3"])])]
        B = np.unique(np.round(np.vstack(b), 9), axis=0)
        axis = np.unique(np.round(np.r_[zu, g["z0"], g["z3"]], 9))
        tree = cKDTree(np.column_stack([r, z]))
        self.gidx = tree.query(B)[1]
        self.aidx = tree.query(np.column_stack([np.zeros_like(axis), axis]))[1]
        xr = np.r_[r, B[:, 0]]
        yr = np.r_[z, B[:, 1]]
        self.tri = mtri.Triangulation(np.r_[xr, -xr, np.zeros_like(axis)], np.r_[yr, yr, axis])
        x = self.tri.x[self.tri.triangles]
        y = self.tri.y[self.tri.triangles]
        xc, yc = x.mean(1), y.mean(1)
        eps = 1e-9
        straddle = (y.min(1) < g["z1"] - eps) & (y.max(1) > g["z2"] + eps) & (np.abs(x).max(1) > g["a"] + eps)
        in_plate = (np.abs(xc) > g["a"] + eps) & (yc > g["z1"] + eps) & (yc < g["z2"] - eps)
        self.tri.set_mask(straddle | in_plate)
        self.n = len(r)

    def values(self, key, i):
        v = self.D[key][i].astype(float)
        half = np.r_[v, v[self.gidx]]
        odd = key in ("wth", "ur")          # mirror image: omega_theta and u_r change sign under x -> -x
        return np.r_[half, -half if odd else half, np.zeros(len(self.aidx)) if odd else v[self.aidx]]

    def u_bulk(self, i):
        """Area-mean downward neck velocity at frame i (from the neck-plane flux of the same run)."""
        return float(np.interp(self.D["t"][i], self.D["tq"], self.D["u_down"]))


def frame(veh_cases):
    """Common framing per vehicle: centred on the facesheet mid-plane, full tube width."""
    gs = [load(c)[1]["geometry"] for c in veh_cases]
    zc = 0.5 * (gs[0]["z1"] + gs[0]["z2"])
    hh = min(min(zc - g["z0"], g["z3"] - zc) for g in gs)
    R = max(g["R"] for g in gs)
    return dict(xlim=(-R * 1.003, R * 1.003), ylim=(zc - hh, zc + hh), zc=zc)


def field_values(P, key, i):
    if key == "umag":
        return P.values("umag", i)
    if key == "pk":
        return P.values("pk", i)
    if key == "wth":
        return P.values("wth", i)
    if key == "logw":
        return np.log10(np.abs(P.values("wth", i)) + EPS_W)
    if key == "phi":
        return np.log10(np.maximum(P.values("phi", i), 1e-30))
    raise KeyError(key)


def case_limits(D):
    """Fixed per-case limits over the whole 2-cycle window (cell data)."""
    u = float(np.percentile(D["umag"], 99.5))
    p = float(np.percentile(np.abs(D["pk"]), 99.5))
    w98 = float(np.percentile(np.abs(D["wth"]), 98.0))
    wmax = float(np.percentile(np.abs(D["wth"]), 99.5))
    ph = float(np.percentile(D["phi"], 99.5))
    lw_hi = math.log10(wmax + EPS_W)
    lp_hi = math.log10(max(ph, 1e-30))
    return dict(umag=(0.0, u), pk=(-p, p), wth=(-w98, w98), logw=(max(math.log10(EPS_W), lw_hi - 3.0), lw_hi),
                phi=(lp_hi - 4.0, lp_hi))


def union_limits(L):
    out = {}
    for k in FIELDS:
        lo = min(x[k][0] for x in L)
        hi = max(x[k][1] for x in L)
        if FIELDS[k][1] == "diverging":
            m = max(abs(lo), abs(hi))
            out[k] = (-m, m)
        elif FIELDS[k][1] == "log":
            out[k] = (hi - (L[0][k][1] - L[0][k][0]), hi)
        else:
            out[k] = (0.0, hi)
    return out


def draw_geometry_overlay(ax, g):
    a, R, z1, z2 = g["a"], g["R"], g["z1"], g["z2"]
    for s in (1, -1):
        ax.add_patch(Rectangle((a if s > 0 else -R, z1), R - a, z2 - z1, fc="0.62", ec="k", lw=0.6, hatch="////",
                               zorder=5))
    ax.plot([a, a], [z1, z2], color="tab:orange", lw=1.6, zorder=6)          # neck boundary
    ax.plot([-a, -a], [z1, z2], color="tab:orange", lw=1.6, zorder=6)
    ax.plot([R, R], [g["z0"], g["z3"]], color="k", lw=1.0, zorder=5)
    ax.plot([-R, -R], [g["z0"], g["z3"]], color="k", lw=1.0, zorder=5)
    ax.plot([0, 0], [g["z0"], g["z3"]], color="0.35", lw=0.6, ls=(0, (6, 3, 1, 3)), zorder=5)


def annotate_cfd(ax, P, i, fr, scale=True, regions=True, flow=True, small=False):
    """Scale bar, the two fluid regions as the CFD boundaries define them, and the instantaneous bulk flow."""
    g = P.g
    fs = 5.6 if small else 6.6
    x0, x1 = fr["xlim"]
    y0, y1 = fr["ylim"]
    if scale:
        L = 0.01
        xs, ys = x1 - 0.05 * (x1 - x0) - L, y0 + 0.075 * (y1 - y0)
        ax.plot([xs, xs + L], [ys, ys], color="w", lw=3.2, solid_capstyle="butt", zorder=9)
        ax.plot([xs, xs + L], [ys, ys], color="k", lw=1.6, solid_capstyle="butt", zorder=9)
        ax.text(xs + L / 2, ys + 0.012 * (y1 - y0), "10 mm", ha="center", va="bottom", fontsize=fs, zorder=9,
                bbox=dict(fc="w", ec="none", alpha=0.75, pad=0.8))
    if regions:
        box = dict(fc="w", ec="none", alpha=0.75, pad=0.8)
        ax.text(0.5, 0.80 if not small else 0.83, "fluid: imposed-velocity reservoir (inlet patch above)",
                transform=ax.transAxes, ha="center", fontsize=fs, zorder=9, bbox=box)
        ax.text(0.5, 0.13 if not small else 0.20, "fluid: pressure-reference reservoir (outlet patch below, p = 0)",
                transform=ax.transAxes, ha="center", fontsize=fs, zorder=9, bbox=box)
        ax.text(0.98, 0.5 + 0.5 * ((g["z1"] + g["z2"]) / 2 - (y0 + y1) / 2) / (y1 - y0) + 0.045,
                "solid facesheet", transform=ax.transAxes, ha="right", fontsize=fs, zorder=9, bbox=box)
    if flow:
        u = P.u_bulk(i)
        xa = -0.83 * g["R"]
        zc = 0.5 * (g["z1"] + g["z2"])
        L = 0.22 * (y1 - y0)
        d = -1 if u > 0 else 1
        ax.annotate("", (xa, zc + d * L / 2), (xa, zc - d * L / 2), zorder=9,
                    arrowprops=dict(arrowstyle="simple,head_width=0.9,head_length=0.9,tail_width=0.35",
                                    fc="w", ec="k", lw=0.6))
        ax.text(xa + 0.06 * (x1 - x0), zc - L / 2 - 0.02 * (y1 - y0), f"bulk flow\n{'down' if u > 0 else 'up'}, "
                f"|u| = {abs(u):.1f} m/s", fontsize=fs, va="top", ha="left", zorder=9,
                bbox=dict(fc="w", ec="none", alpha=0.75, pad=0.8))


def arrow_points(g, fr, n_across=17):
    """Regular interior sample points: fluid only, at least one spacing from walls, plate and frame edges."""
    xs = np.linspace(fr["xlim"][0], fr["xlim"][1], n_across + 2)[1:-1]
    dx = xs[1] - xs[0]
    ys = np.arange(fr["ylim"][0] + dx, fr["ylim"][1] - 0.5 * dx, dx)
    X, Y = np.meshgrid(xs, ys)
    X, Y = X.ravel(), Y.ravel()
    d = 0.6 * dx
    ok = (np.abs(X) < g["R"] - d) & (Y > g["z0"] + d) & (Y < g["z3"] - d)
    near_plate = (np.abs(X) > g["a"] - d) & (Y > g["z1"] - d) & (Y < g["z2"] + d)
    ok &= ~near_plate
    return X[ok], Y[ok], dx


def draw_field(ax, P, key, i, lim, fr, arrows=False, uref=None, n_across=17, aw=0.0045):
    v = field_values(P, key, i)
    kind, cmap = FIELDS[key][1], FIELDS[key][2]
    im = ax.tripcolor(P.tri, v, shading="gouraud", cmap=cmap, vmin=lim[0], vmax=lim[1], rasterized=True)
    draw_geometry_overlay(ax, P.g)
    pts = None
    if arrows:
        X, Y, dx = arrow_points(P.g, fr, n_across)
        ur = mtri.LinearTriInterpolator(P.tri, P.values("ur", i))(X, Y)
        uz = mtri.LinearTriInterpolator(P.tri, P.values("uz", i))(X, Y)
        good = ~(np.ma.getmaskarray(ur) | np.ma.getmaskarray(uz))
        X, Y, ur, uz = X[good], Y[good], np.asarray(ur)[good], np.asarray(uz)[good]
        big = np.hypot(ur, uz) > 0.06 * uref          # omit near-stagnant points (they draw as dots)
        X, Y, ur, uz = X[big], Y[big], ur[big], uz[big]
        ax.quiver(X, Y, ur, uz, color="w", edgecolor="k", linewidth=0.3, angles="xy", scale_units="xy",
                  scale=uref / (0.95 * dx), width=aw, headwidth=3.5, headlength=4, zorder=6)
        pts = np.column_stack([X, Y])
    ax.set_xlim(*fr["xlim"])
    ax.set_ylim(*fr["ylim"])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return im, pts


def tag(ax, text, color, loc=(0.015, 0.985), size=6.5):
    ax.text(*loc, text, transform=ax.transAxes, ha="left", va="top", fontsize=size, color="w", zorder=10,
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none", alpha=0.92))


def theta_of(t, f0):
    return (360.0 * f0 * t) % 360.0


def phase_index(D, f0, target, cycle):
    """Frame nearest the target phase in the given cycle of the window (0 = first)."""
    th = 360.0 * f0 * D["t"]
    want = 360.0 * (math.floor(f0 * D["t"][0] + 1e-6) + cycle) + target
    return int(np.argmin(np.abs(th - want)))


def save(fig, path, titles, notes, **kw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=kw.pop("dpi", DPI), **kw)
    plt.close(fig)
    NOTES[str(path.relative_to(OUT)).replace("\\", "/")] = dict(titles=titles, notes=notes)


# ------------------------------------------------------------------ products
def scale_label(shared):
    return "SHARED SCALE" if shared else "INDIVIDUAL SCALE"


def still(c, P, D, key, i, lim, fr, shared, out, uref):
    fig, ax = plt.subplots(figsize=(4.6, 5.6))
    im, pts = draw_field(ax, P, key, i, lim, fr, arrows=(key == "umag"), uref=uref)
    annotate_cfd(ax, P, i, fr)
    st = status(c)
    tag(ax, st["tag"], "#2e7d32" if st["converged"] else "#c62828")
    tag(ax, "VEHICLE VERDICT: UNRESOLVED", "#455a64", loc=(0.015, 0.935))
    tag(ax, "ILLUSTRATIVE VISUALIZATION ONLY", "#6d4c41", loc=(0.015, 0.885))
    tag(ax, scale_label(shared), "#1565c0", loc=(0.015, 0.835))
    ax.text(0.015, 0.012, f"{short(c)}   \u03b8 = {theta_of(D['t'][i], c['f0_Hz']):.1f}\u00b0   "
            f"t = {D['t'][i]:.6f} s", transform=ax.transAxes, fontsize=6.5, zorder=10,
            bbox=dict(fc="w", ec="none", alpha=0.8, pad=1.5))
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label(FIELDS[key][0], fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    save(fig, out, [f"{c['vehicle']} {c['case']} {key} theta={theta_of(D['t'][i], c['f0_Hz']):.1f} deg "
                    f"({scale_label(shared).lower()})"], [SCOPE, VERDICT_NOTE[c["vehicle"]]], bbox_inches="tight")
    return pts


def montage(veh, cases, key, lims, shared, fr, out, kind):
    n = len(cases)
    asp = (fr["xlim"][1] - fr["xlim"][0]) / (fr["ylim"][1] - fr["ylim"][0])
    fig, axs = plt.subplots(len(PHASES), n, figsize=(max(3.05 * asp, 1.9) * n + (0.9 if shared else 0.5 * n) + 0.4,
                                                      3.05 * len(PHASES)),
                            squeeze=False, gridspec_kw=dict(wspace=0.05 if shared else 0.32, hspace=0.04))
    info = []
    arrows_all = []
    for j, c in enumerate(cases):
        D, meta = load(c)
        P = Panel(c, D, meta)
        lim = lims["shared"] if shared else lims[c["dir"]]
        uref = lim["umag"][1]
        for i_ph, ph in enumerate(PHASES):
            ax = axs[i_ph, j]
            k = phase_index(D, c["f0_Hz"], ph, 0)
            im, pts = draw_field(ax, P, key, k, lim[key], fr, arrows=(key == "umag"), uref=uref, n_across=11,
                                 aw=0.011)
            annotate_cfd(ax, P, k, fr, scale=(j == 0), regions=False, flow=True, small=True)
            if pts is not None:
                arrows_all.append((c["dir"], pts))
            th = theta_of(D["t"][k], c["f0_Hz"])
            ax.text(0.02, 0.012, f"\u03b8 = {th:.1f}\u00b0, t = {D['t'][k]:.5f} s", transform=ax.transAxes,
                    fontsize=6, zorder=10, bbox=dict(fc="w", ec="none", alpha=0.8, pad=1.2))
            info.append(dict(case=c["dir"], phase_target=ph, theta=th, t=float(D["t"][k]), frame=k))
            if i_ph == 0:
                st = status(c)
                ax.set_title(("" if kind == "numerics" else SLOT_LABEL[c["slot"]] + ":\n") + short(c), fontsize=7.2,
                             pad=14)
                ax.text(0.5, 1.012, st["tag"], transform=ax.transAxes, ha="center", va="bottom", fontsize=5.4,
                        fontweight="bold", color="#2e7d32" if st["converged"] else "#c62828")
            if not shared and i_ph == len(PHASES) - 1:
                cb = fig.colorbar(im, ax=axs[:, j], orientation="horizontal", fraction=0.025, pad=0.01)
                cb.set_label(SHORT_LABEL[key], fontsize=6)
                cb.ax.tick_params(labelsize=5.5)
    if shared:
        cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.01)
        cb.set_label(FIELDS[key][0], fontsize=7.5)
    top = axs[0, 0].get_position().y1
    fig.text(axs[0, 0].get_position().x0, top + 0.045, f"{scale_label(shared)}   |   ILLUSTRATIVE VISUALIZATION ONLY"
             f"   |   VEHICLE VERDICT: UNRESOLVED", fontsize=7.5, va="bottom", fontweight="bold", color="#263238")
    titles = [f"{VNAME[veh]}: {'low / nearest-U* / high speed' if kind == 'speed' else 'nominal / refined mesh / half time step near U*'}"
              f", {key}, phases 0/90/180/270 deg (rows), {scale_label(shared).lower()}"]
    notes = [SCOPE, VERDICT_NOTE[veh],
             "Shared scale: identical colour limits in every panel (the quantitative comparison). Individual scale: "
             "each column has its own fixed limits (for viewing structure only)."]
    if kind == "numerics":
        notes.append("Qualitative companion to the numerical sensitivity analysis; visual similarity does not "
                     "establish numerical convergence.")
    save(fig, out, titles, notes, bbox_inches="tight")
    return info, arrows_all


def animation(c, key, lim, fr, out, shared=False):
    D, meta = load(c)
    P = Panel(c, D, meta)
    frames = []
    for i in range(len(D["t"])):
        fig, ax = plt.subplots(figsize=(4.2, 5.0))
        im, _ = draw_field(ax, P, key, i, lim[key], fr, arrows=(key == "umag"), uref=lim["umag"][1])
        annotate_cfd(ax, P, i, fr)
        st = status(c)
        tag(ax, st["tag"], "#2e7d32" if st["converged"] else "#c62828")
        tag(ax, "VEHICLE VERDICT: UNRESOLVED", "#455a64", loc=(0.015, 0.935))
        tag(ax, scale_label(shared) + " (fixed)", "#1565c0", loc=(0.015, 0.885))
        ax.text(0.015, 0.012, f"{short(c)}   \u03b8 = {theta_of(D['t'][i], c['f0_Hz']):5.1f}\u00b0",
                transform=ax.transAxes, fontsize=7, zorder=10, bbox=dict(fc="w", ec="none", alpha=0.8, pad=1.5))
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label(FIELDS[key][0], fontsize=6.5)
        cb.ax.tick_params(labelsize=6)
        buf = io.BytesIO()
        fig.savefig(buf, dpi=85, format="png")
        plt.close(fig)
        frames.append(Image.open(buf).convert("RGB"))
    write_gif(frames, out)
    NOTES[str(Path(out).relative_to(OUT)).replace("\\", "/")] = dict(
        titles=[f"{c['vehicle']} {c['case']} {key} animation, {len(frames)} frames (2 cycles), {FPS} fps, "
                f"fixed {scale_label(shared).lower()}"], notes=[SCOPE, VERDICT_NOTE[c["vehicle"]]])
    return dict(file=str(Path(out).relative_to(OUT)).replace("\\", "/"), frames=len(frames), fps=FPS,
                limits=list(lim[key]), theta=[float(theta_of(t, c["f0_Hz"])) for t in D["t"]])


def write_gif(frames, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pal = frames[len(frames) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(out, save_all=True, append_images=q[1:], duration=int(1000 / FPS), loop=0, optimize=False)


def coupled(veh, c, lim, fr, out, Zrec):
    """Primary synchronized pressure / velocity / vorticity / dissipation animation with waveforms and phasor."""
    D, meta = load(c)
    P = Panel(c, D, meta)
    f0 = c["f0_Hz"]
    c0, c1 = meta["cycles"]
    m = (D["tp"] >= c0 / f0 - 1e-9) & (D["tp"] < c1 / f0)
    mq = (D["tq"] >= c0 / f0 - 1e-9) & (D["tq"] < c1 / f0)
    ph_p = 360.0 * (f0 * D["tp"][m] - c0)
    ph_q = 360.0 * (f0 * D["tq"][mq] - c0)
    R, X, hR, hX = Zrec["R"], Zrec["X"], Zrec["h_R"], Zrec["h_X"]
    phi_deg = math.degrees(math.atan2(X, R))
    frames = []
    keys = ("pk", "umag", "logw", "phi")
    for i in range(len(D["t"])):
        asp = (fr["xlim"][1] - fr["xlim"][0]) / (fr["ylim"][1] - fr["ylim"][0])
        hf = 2.45 / asp - 0.45
        H = hf + 2.9
        fig = plt.figure(figsize=(11.0, H))
        gsp = fig.add_gridspec(2, 4, height_ratios=[hf, 2.0], hspace=0.30, wspace=0.28, top=1 - 0.55 / H,
                               bottom=0.07)
        for j, key in enumerate(keys):
            ax = fig.add_subplot(gsp[0, j])
            im, _ = draw_field(ax, P, key, i, lim[key], fr, arrows=(key == "umag"), uref=lim["umag"][1])
            annotate_cfd(ax, P, i, fr, scale=True, regions=(j == 0), flow=True, small=True)
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02)
            cb.set_label(FIELDS[key][0], fontsize=6.5)
            cb.ax.tick_params(labelsize=6)
        bot = gsp[1, :].subgridspec(1, 3, width_ratios=[2.6, 0.95, 1.05], wspace=0.22)
        draw_locator(fig.add_subplot(bot[2]), veh, c, meta)
        axw = fig.add_subplot(bot[0])
        axw.plot(ph_p, D["dp_pa"][m], color="tab:red", lw=1.3, label="\u0394p = \u03c1 (p_up \u2212 p_down) (Pa)")
        axw.set_ylabel("\u0394p (Pa)", color="tab:red", fontsize=8)
        axw.tick_params(labelsize=7)
        ax2 = axw.twinx()
        ax2.plot(ph_q, D["u_down"][mq], color="tab:blue", lw=1.3, label="neck mean velocity u (m/s, downward)")
        ax2.set_ylabel("neck mean velocity u (m/s)", color="tab:blue", fontsize=8)
        ax2.tick_params(labelsize=7)
        x = 360.0 * (f0 * D["t"][i] - c0)
        axw.axvline(x, color="k", lw=1.2)
        axw.set_xlim(0, 720)
        axw.set_xticks(range(0, 721, 90))
        axw.set_xlabel("phase \u03b8 over the two visualization cycles (deg)", fontsize=8)
        axw.grid(alpha=0.3)
        axw.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=6.5, frameon=False)
        ax2.legend(loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=6.5, frameon=False)
        axz = fig.add_subplot(bot[1])
        axz.set_aspect("equal")
        sc = max(abs(R), abs(X)) * 1.25
        axz.annotate("", (R, 0), (0, 0), arrowprops=dict(arrowstyle="->", color="tab:green", lw=1.5))
        axz.annotate("", (R, X), (R, 0), arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.5))
        axz.annotate("", (R, X), (0, 0), arrowprops=dict(arrowstyle="->", color="k", lw=1.8))
        axz.text(R / 2, -0.08 * sc, "R", color="tab:green", ha="center", va="top", fontsize=8)
        axz.text(R * 1.03, X / 2, "X", color="tab:purple", fontsize=8)
        axz.text(0.0, 1.0, f"Z = R + iX\nR = {R:.2f} \u00b1 {hR:.2f} Rayl\nX = {X:.2f} \u00b1 {hX:.2f} Rayl\n"
                 f"arg Z = {phi_deg:.1f}\u00b0:\n\u0394p leads u\n(authoritative\nblock estimate)",
                 transform=axz.transAxes, fontsize=5.8, va="top")
        axz.set_xlim(-1.7 * sc, 1.05 * sc)
        axz.set_ylim(-0.25 * sc, sc)
        axz.set_xticks([])
        axz.set_yticks([])
        st = status(c)
        fig.text(0.01, 0.995, f"{VNAME[veh]}, bank {bank_of(c)}, "
                 f"{short(c)}   \u03b8 = {theta_of(D['t'][i], f0):5.1f}\u00b0   t = {D['t'][i]:.6f} s", fontsize=9,
                 va="top")
        for k_, (txt, col) in enumerate(((st["tag"], "#2e7d32" if st["converged"] else "#c62828"),
                                         ("VEHICLE VERDICT: UNRESOLVED", "#455a64"),
                                         ("ILLUSTRATIVE VISUALIZATION ONLY", "#6d4c41"),
                                         ("INDIVIDUAL SCALE (fixed)", "#1565c0"))):
            fig.text(0.01 + 0.245 * k_, 1 - 0.3 / H, txt, fontsize=6.5, va="top", color="w", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.25", fc=col, ec="none"))
        buf = io.BytesIO()
        fig.savefig(buf, dpi=80, format="png")
        if i == phase_index(D, f0, 90, 0):
            fig.savefig(Path(out).with_suffix(".png"), dpi=130)
        plt.close(fig)
        frames.append(Image.open(buf).convert("RGB"))
    write_gif(frames, out)
    A = meta["u_down_fundamental"]
    for p in (Path(out), Path(out).with_suffix(".png")):
        NOTES[str(p.relative_to(OUT)).replace("\\", "/")] = dict(
            titles=[f"{VNAME[veh]}: coupled pressure / velocity / vorticity / modeled viscous dissipation at "
                    f"{short(c)} (nearest U*), synchronized with the pressure-difference and neck-velocity "
                    f"waveforms and the impedance phasor" + (" (still at theta nearest 90 deg)" if p.suffix == ".png"
                                                             else f" ({len(frames)} frames, {FPS} fps)")],
            notes=[COUPLED_CAPTION, COUPLED_LIMIT, SCOPE, VERDICT_NOTE[veh],
                   "Z = R + iX shown is the block estimate of the authoritative case record (analysis cycles 14-25 "
                   "or the follow-up continuation record), not computed from the visualization window."])
    return dict(file=str(Path(out).relative_to(OUT)).replace("\\", "/"), frames=len(frames), fps=FPS,
                limits={k: list(lim[k]) for k in keys}, Z=Zrec, arg_Z_deg=phi_deg, u_fundamental=A)


FLUID = "#cfe6fb"          # fluid (blue)
SOLID = "0.62"             # solids (grey, hatched)
NECK = "tab:orange"        # neck boundary
LINER = "tab:purple"       # lined inner wall
ORIENT_TEXT = ("The global FEM models acoustic pressure throughout the payload fairing. A lined wall is represented by "
               "an impedance boundary assembled from many local resonator cells. CFD resolves oscillatory flow through "
               "one representative neck. The pressure–flow relationship extracted from that local model supplies "
               "the nonlinear impedance R+iX used to correct the global FEM.")
SIDES_NOTE = ("The CFD boundary definitions do not assign a fairing-interior or backing-cavity side: the upper reservoir "
              "carries the imposed oscillatory velocity (inlet patch), the lower one is an open pressure reference "
              "(outlet patch, p = 0), and neither contains the backing cavity, whose stiffness enters through the "
              "resonator model and the FEM. The CFD domain is drawn with the same neck axis as the liner cell for "
              "comparison only.")


def fairing_geom(veh):
    import yaml
    y = yaml.safe_load((ROOT / "configs" / f"{VEH[veh]}.yaml").read_text(encoding="utf-8"))
    flat = {}

    def walk(d):
        for k, v in d.items():
            if isinstance(v, dict):
                if "value" in v and not isinstance(v["value"], dict):
                    flat.setdefault(k, v["value"])
                walk(v)
    walk(y)
    out = dict(R=float(flat["R_fair_m"]), H=float(flat["H_total_m"]), Hb=float(flat["H_barrel_m"]))
    try:
        pm = y["payload_model"]
        case = pm["cases"][pm["primary_case"]["value"]]
        out.update(z0=float(pm["z_base_m"]["value"]), Lz=float(pm["axial_extent_m"]["value"]),
                   rp=float(case["r_out_m"]["value"]))
    except (KeyError, TypeError):
        out.update(z0=None)
    return out


def liner_cell(veh, c):
    fin = json.loads((VDIR / VEH[veh] / "finalist.json").read_text(encoding="utf-8"))
    gr = fin["design"]["groups"][int(bank_of(c)[1:])]
    Dc = fin["design"]["D"]
    V = RM.cavity_volume_for_target_f0(gr["f0"], RM.neck_area(gr["r_neck"]),
                                       RM.effective_neck_length(gr["L_neck"], gr["r_neck"]))
    return dict(r=gr["r_neck"] * 1e3, L=gr["L_neck"] * 1e3, D=Dc * 1e3, w=math.sqrt(V / Dc) * 1e3, V_cm3=V * 1e6,
                f0=gr["f0"], coverage=fin.get("coverage"))


def draw_fairing(ax, fg, patch_frac=0.45, small=False):
    from matplotlib.patches import Polygon
    R, H, Hb = fg["R"], fg["H"], fg["Hb"]
    ax.add_patch(Polygon([(-R, 0), (-R, Hb), (0, H), (R, Hb), (R, 0)], closed=True, fc=FLUID, ec="none"))
    if fg.get("z0") is not None:
        ax.add_patch(Rectangle((-fg["rp"], fg["z0"]), 2 * fg["rp"], fg["Lz"], fc=SOLID, ec="k", lw=0.6,
                               hatch="////"))
    ax.plot([-R, -R, 0, R, R], [0, Hb, H, Hb, 0], color=LINER, lw=2.0 if small else 3.2, solid_capstyle="butt")
    ax.plot([-R, -R, 0, R, R], [0, Hb, H, Hb, 0], color="k", lw=0.6)
    ax.plot([0, 0], [0, H], color="0.3", lw=0.7, ls=(0, (6, 3, 1, 3)))
    ph = 0.07 * H
    zp = patch_frac * Hb
    ax.add_patch(Rectangle((R - 0.09 * R, zp - ph / 2), 0.18 * R, ph, fc="none", ec="k", lw=1.0, zorder=6))
    ax.set_xlim(-R * 1.15, R * 1.15)
    ax.set_ylim(-0.04 * H, H * 1.03)
    ax.set_aspect("equal")
    ax.axis("off")
    return (R - 0.09 * R, zp - ph / 2, 0.18 * R, ph)


def draw_locator(ax, veh, c, meta):
    """Small fairing-to-liner-to-CFD locator for the coupled animation."""
    fg = fairing_geom(veh)
    rect = draw_fairing(ax, fg, small=True)
    ax.text(0, fg["H"] * 0.04, "payload", ha="center", fontsize=5.2) if fg.get("z0") is None else \
        ax.text(0, fg["z0"] + fg["Lz"] / 2, "payload", ha="center", va="center", fontsize=5.2)
    ax.annotate("representative\nliner patch\n→ one resonator\nneck = this\nlocal CFD model\n(not to scale)",
                (rect[0] + rect[2], rect[1] + rect[3] / 2), (fg["R"] * 1.25, fg["H"] * 0.62), fontsize=5.4,
                va="center", arrowprops=dict(arrowstyle="->", lw=0.6), annotation_clip=False)
    ax.text(0, fg["H"] * 1.05, f"{VNAME[veh]} fairing (lined wall: purple)", ha="center", fontsize=5.6)
    ax.set_xlim(-fg["R"] * 1.15, fg["R"] * 3.2)


def orientation_figure(veh, c, out):
    """Multiscale orientation: fairing -> liner patch -> one neck = the local CFD domain -> impedance coupling."""
    from matplotlib.patches import ConnectionPatch, FancyBboxPatch
    D, meta = load(c)
    g = {k: v * 1e3 for k, v in meta["geometry"].items() if k != "wedge_deg"}     # mm
    lc = liner_cell(veh, c)
    fg = fairing_geom(veh)
    a, Ln = g["a"], g["z2"] - g["z1"]
    assert abs(a - lc["r"]) < 0.02 and abs(Ln - lc["L"]) < 0.02, "CFD neck differs from the liner-cell neck"
    tw = 6.0                                             # schematic structural-wall thickness (mm), not modelled
    top_int = 45.0                                       # fairing interior shown above the facesheet (mm)
    # panel 2 data box (mm): x along the wall, y wall-normal (interior up)
    y2 = (-tw - 6, lc["D"] + Ln + top_int)
    x2 = (-0.85 * lc["w"], 0.85 * lc["w"])
    # panel 3 data box (mm): CFD section, neck centred at the liner neck height for comparison
    zc = (g["z1"] + g["z2"]) / 2
    off = (lc["D"] + Ln / 2) - zc                        # shift CFD z so the neck mid-planes line up
    y3 = (min(g["z0"] + off, y2[0]) - 6, max(g["z3"] + off, y2[1]) + 6)
    y2 = y3
    x3 = (-g["R"] - 22, g["R"] + 22)
    W2, W3 = x2[1] - x2[0], x3[1] - x3[0]
    Hd = y3[1] - y3[0]
    s = 5.4 / Hd                                         # inches per mm, shared by panels 2 and 3
    w1 = 3.4
    figw = w1 + (W2 + W3) * s + 1.6
    fig = plt.figure(figsize=(figw, 5.6 + 3.7))
    gs = fig.add_gridspec(2, 3, width_ratios=[w1, W2 * s, W3 * s], height_ratios=[5.6, 3.7], wspace=0.08,
                          hspace=0.12, left=0.02, right=0.98, top=0.97, bottom=0.03)
    # ---- (1) fairing
    ax1 = fig.add_subplot(gs[0, 0])
    rect = draw_fairing(ax1, fg)
    R, H, Hb = fg["R"], fg["H"], fg["Hb"]
    if fg.get("z0") is not None:
        ax1.text(0, fg["z0"] + fg["Lz"] / 2, "payload\n(solid)", ha="center", va="center", fontsize=6.5,
                 bbox=dict(fc="w", ec="none", alpha=0.8, pad=1))
    ax1.text(0, Hb * 1.12, "payload-fairing\nacoustic cavity\n(fluid; global FEM)", ha="center", fontsize=6.3,
             bbox=dict(fc=FLUID, ec="none", pad=0.5))
    ax1.annotate("lined inner wall", (-R, Hb * 0.35), (-R * 1.12, -0.03 * H), fontsize=6.5, color=LINER,
                 arrowprops=dict(arrowstyle="->", color=LINER, lw=0.8))
    ax1.annotate("representative\nliner patch", (rect[0] + rect[2], rect[1] + rect[3]),
                 (R * 1.08, rect[1] + 0.12 * H), fontsize=6.3, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax1.set_xlim(-R * 1.2, R * 2.0)
    ax1.text(0.07 * R, H * 0.75, "axis of\nsymmetry", fontsize=5.8, color="0.3")
    ax1.text(0, -0.035 * H, f"{VNAME[veh]}{'' if 'representative' in VNAME[veh] else ' representative'} fairing, R = {R:.2f} m, H = {H:.1f} m", ha="center",
             va="top", fontsize=6.2)
    # ---- (2) liner patch
    ax2 = fig.add_subplot(gs[0, 1])
    w, Dm, r = lc["w"], lc["D"], lc["r"]
    ax2.add_patch(Rectangle((x2[0], -tw), W2, tw, fc=SOLID, ec="k", hatch="////", lw=0.8))
    ax2.add_patch(Rectangle((x2[0], 0), W2, Dm, fc=FLUID, ec="none"))
    ax2.add_patch(Rectangle((x2[0], Dm + Ln), W2, y2[1] - Dm - Ln, fc=FLUID, ec="none"))
    for xc in (-w, 0.0, w):
        for x_lo, x_hi in ((xc - w / 2, xc - r), (xc + r, xc + w / 2)):
            lo, hi = max(x_lo, x2[0]), min(x_hi, x2[1])
            if hi > lo:
                ax2.add_patch(Rectangle((lo, Dm), hi - lo, Ln, fc=SOLID, ec="k", hatch="////", lw=0.6))
        if x2[0] < xc - r and xc + r < x2[1]:
            ax2.add_patch(Rectangle((xc - r, Dm), 2 * r, Ln, fc=FLUID, ec="none"))
            ax2.plot([xc - r, xc - r], [Dm, Dm + Ln], color=NECK, lw=2.0)
            ax2.plot([xc + r, xc + r], [Dm, Dm + Ln], color=NECK, lw=2.0)
    for xp in (-w / 2, w / 2):
        ax2.add_patch(Rectangle((xp - 0.8, 0), 1.6, Dm, fc=SOLID, ec="k", lw=0.5))
    ax2.plot([0, 0], [y2[0], y2[1]], color="0.3", lw=0.7, ls=(0, (6, 3, 1, 3)))
    box = dict(fc="w", ec="none", alpha=0.8, pad=1)
    ax2.text(x2[0] + 3, y2[1] - 4, "fairing-interior side (fluid)", fontsize=6.6, va="top", bbox=box)
    ax2.text(-w / 2 + 3, Dm * 0.5, "backing cavity\n(backing-cavity side, fluid)", fontsize=6.4, va="center",
             bbox=box)
    ax2.text(x2[0] + 3, -tw / 2, "structural wall (schematic thickness)", fontsize=6.2, va="center")
    ax2.annotate("solid facesheet", (w * 0.32, Dm + Ln / 2), (w * 0.28, Dm + Ln + 14), fontsize=6.4,
                 arrowprops=dict(arrowstyle="->", lw=0.6))
    ax2.annotate("resonator neck", (r * 0.9, Dm + Ln * 0.8), (-w * 0.47, Dm + Ln + 26), fontsize=6.4, color=NECK,
                 arrowprops=dict(arrowstyle="->", lw=0.7, color=NECK))
    ax2.annotate("", (r + 9, Dm + Ln + 13), (r + 9, Dm - 13),
                 arrowprops=dict(arrowstyle="<|-|>", lw=1.0, mutation_scale=8))
    ax2.text(r + 11, Dm - 10, "local wall-normal\nflow direction", fontsize=6.0)
    ax2.text(1.5, y2[0] + 2, "axis of symmetry\n(of one neck)", fontsize=5.8, color="0.3")
    ax2.text(0.5, 0.0, f"representative liner patch, bank {bank_of(c)}: neck r = {r:.2f} mm, L = {Ln:.2f} mm, "
             f"cavity depth D = {Dm:.0f} mm (cells sized for f₀ = {lc['f0']:.1f} Hz)", transform=ax2.transAxes,
             ha="center", va="top", fontsize=6.2)
    ax2.set_xlim(*x2)
    ax2.set_ylim(*y2)
    ax2.set_aspect("equal")
    ax2.axis("off")
    # ---- (3) CFD domain (same mm scale, same neck)
    ax3 = fig.add_subplot(gs[0, 2])
    Rt, z0, z1, z2, z3 = g["R"], g["z0"] + off, g["z1"] + off, g["z2"] + off, g["z3"] + off
    ax3.add_patch(Rectangle((-Rt, z0), 2 * Rt, z3 - z0, fc=FLUID, ec="none"))
    for xs in (a, -Rt):
        ax3.add_patch(Rectangle((xs, z1), Rt - a, z2 - z1, fc=SOLID, ec="k", hatch="////", lw=0.6))
    ax3.plot([a, a], [z1, z2], color=NECK, lw=2.0)
    ax3.plot([-a, -a], [z1, z2], color=NECK, lw=2.0)
    ax3.plot([Rt, Rt], [z0, z3], color="k", lw=1.6)
    ax3.plot([-Rt, -Rt], [z0, z3], color="k", lw=1.6)
    ax3.plot([0, 0], [z0, z3], color="0.3", lw=0.7, ls=(0, (6, 3, 1, 3)))
    ax3.plot([-Rt, Rt], [z3, z3], color="tab:red", lw=2.2)
    ax3.plot([-Rt, Rt], [z0, z0], color="tab:green", lw=2.2)
    ax3.text(0, z3 + 1.5, "inlet patch: imposed\noscillatory velocity", ha="center", va="bottom", fontsize=6.0,
             color="tab:red")
    ax3.text(0, z0 - 1.5, "outlet patch: p = 0\n(open pressure reference)", ha="center", va="top", fontsize=6.0,
             color="tab:green")
    ax3.text(0, (z2 + z3) / 2 + 6, "fluid reservoir", ha="center", fontsize=6.2, bbox=box)
    ax3.text(0, (z0 + z1) / 2 - 6, "fluid reservoir", ha="center", fontsize=6.2, bbox=box)
    ax3.text(Rt * 0.55 + a * 0.45, (z1 + z2) / 2, "solid\nfacesheet", ha="center", va="center", fontsize=5.6,
             bbox=box)
    ax3.annotate("resonator neck", (a * 0.6, z2 - Ln * 0.1), (-Rt - 18, z2 + 18), fontsize=6.2, color=NECK,
                 arrowprops=dict(arrowstyle="->", lw=0.7, color=NECK), annotation_clip=False)
    ax3.text(1.0, z0 + 3, "axis of\nsymmetry", fontsize=5.6, color="0.3")
    ax3.annotate("", (Rt + 6, (z1 + z2) / 2 + 13), (Rt + 6, (z1 + z2) / 2 - 13),
                 arrowprops=dict(arrowstyle="<|-|>", lw=1.0, mutation_scale=8), annotation_clip=False)
    ax3.text(Rt + 8, (z1 + z2) / 2, "local\nwall-normal\nflow", fontsize=5.6, va="center")
    ax3.plot([x3[1] - 22 - 10, x3[1] - 22], [y3[0] + 3, y3[0] + 3], color="k", lw=1.8, solid_capstyle="butt")
    ax3.text(x3[1] - 27, y3[0] + 4, "10 mm", ha="center", fontsize=5.8)
    ax3.text(0.5, 0.0, f"local CFD domain, not to fairing scale: 2D axisymmetric {meta['geometry']['wedge_deg']:.0f}° "
             "wedge, incompressible, laminar", transform=ax3.transAxes, ha="center", va="top", fontsize=6.2)
    ax3.set_xlim(*x3)
    ax3.set_ylim(*y3)
    ax3.set_aspect("equal")
    ax3.axis("off")
    # zoom connectors
    for (xa, ya) in ((rect[0], rect[1] + rect[3]), (rect[0], rect[1])):
        fig.add_artist(ConnectionPatch((xa, ya), (x2[0], y2[1] if ya > rect[1] else y2[0]), ax1.transData,
                                       ax2.transData, color="0.4", lw=0.7, ls="--"))
    ax2.add_patch(Rectangle((-w / 2 + 2, Dm - 14), w - 4, Ln + 28, fc="none", ec="0.3", lw=0.8, ls="--"))
    for yy in (Dm - 14, Dm + Ln + 14):
        fig.add_artist(ConnectionPatch((w / 2 - 2, yy), (-Rt, yy), ax2.transData, ax3.transData, color="0.4",
                                       lw=0.7, ls="--"))
    # ---- (4) coupling diagram
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 40)
    ax4.axis("off")
    f0 = c["f0_Hz"]
    c0, c1 = meta["cycles"]
    m = (D["tp"] >= c0 / f0 - 1e-9) & (D["tp"] < (c0 + 1) / f0)
    mq = (D["tq"] >= c0 / f0 - 1e-9) & (D["tq"] < (c0 + 1) / f0)
    Z = Z_record(c)
    boxes = [(1, "Δp(t) = ρ (p_up − p_down)\nacross the neck (CFD)"),
             (26, "q(t): volume flow through the neck\n(CFD; ū = q / πa²)"),
             (51, f"Z = R + iX = Δp / ū\n(fundamental, block-averaged)\nR = {Z['R']:.2f}, X = {Z['X']:.2f} Rayl"),
             (76, "global FEM impedance boundary\n(lined wall of the fairing;\ncorrected FEM)")]
    for k, (x, txt) in enumerate(boxes):
        ax4.add_patch(FancyBboxPatch((x, 17), 22, 21, boxstyle="round,pad=0.4", fc="#f5f5f5", ec="0.4", lw=0.8))
        ax4.text(x + 11, 36.5, txt, ha="center", va="top", fontsize=6.6)
        if k < 3:
            ax4.annotate("", (x + 25, 27), (x + 22.6, 27), arrowprops=dict(arrowstyle="-|>", lw=1.2,
                                                                         mutation_scale=10))
    sub = ax4.inset_axes([1.8, 18.5, 20.4, 9.5], transform=ax4.transData)
    sub.plot(360 * (f0 * D["tp"][m] - c0), D["dp_pa"][m], color="tab:red", lw=1)
    sub2 = ax4.inset_axes([26.8, 18.5, 20.4, 9.5], transform=ax4.transData)
    qf = D["u_down"][mq] * math.pi * (a / 1e3) ** 2 * 1e6
    sub2.plot(360 * (f0 * D["tq"][mq] - c0), qf, color="tab:blue", lw=1)
    for s_, yl in ((sub, "Pa"), (sub2, "cm³/s")):
        s_.set_xticks([0, 180, 360])
        s_.tick_params(labelsize=5)
        s_.set_xlabel("phase (deg)", fontsize=5, labelpad=1)
        s_.set_ylabel(yl, fontsize=5, labelpad=1)
        s_.grid(alpha=0.3)
    sub3 = ax4.inset_axes([57.5, 18.2, 9, 8.4], transform=ax4.transData)
    sub3.annotate("", (Z["R"], Z["X"]), (0, 0), arrowprops=dict(arrowstyle="->", lw=1.2))
    sub3.plot([0, Z["R"]], [0, 0], color="tab:green", lw=1.2)
    sub3.plot([Z["R"], Z["R"]], [0, Z["X"]], color="tab:purple", lw=1.2)
    sc = max(Z["R"], Z["X"]) * 1.15
    sub3.set_xlim(-0.1 * sc, sc)
    sub3.set_ylim(-0.1 * sc, sc)
    sub3.set_aspect("equal")
    sub3.set_xticks([])
    sub3.set_yticks([])
    sub4 = ax4.inset_axes([80, 17.6, 14, 10.5], transform=ax4.transData)
    draw_fairing(sub4, fg, small=True)
    ax4.text(50, 13.5, ORIENT_TEXT, ha="center", va="top", fontsize=7.2, wrap=True,
             bbox=dict(fc="w", ec="none"))
    ax4.text(50, 4.8, "The local CFD domain does not represent the fairing, and the CFD does not calculate whole-fairing "
             "attenuation. " + SIDES_NOTE, ha="center", va="top", fontsize=6.0, color="0.25", wrap=True)
    for t in ax4.texts[-2:]:
        t._get_wrap_line_width = lambda: fig.get_figwidth() * fig.dpi * 0.86
    # numbering
    for axx, n in ((ax1, "1"), (ax2, "2"), (ax3, "3"), (ax4, "4")):
        axx.text(0.0, 1.0, n, transform=axx.transAxes, fontsize=10, fontweight="bold", va="top")
    fig.canvas.draw()
    px2 = (ax2.transData.transform((r, 0)) - ax2.transData.transform((-r, 0)))[0]
    px3 = (ax3.transData.transform((a, 0)) - ax3.transData.transform((-a, 0)))[0]
    save(fig, out, [f"{VNAME[veh]}: multiscale orientation (1) payload fairing with lined inner wall, (2) one "
                    "representative liner patch, (3) one resonator neck mapped onto the local axisymmetric CFD domain "
                    "(same mm scale and identical neck geometry as panel 2), (4) coupling Δp(t) → q(t) "
                    "→ Z = R + iX → global FEM impedance boundary"],
         [ORIENT_TEXT, SIDES_NOTE, SCOPE, VERDICT_NOTE[veh],
          "Panel 1 uses the representative axisymmetric fairing of the vehicle configuration (cylinder + conical "
          "nose; not the real interior shape). The structural-wall thickness in panel 2 is schematic (not modelled). "
          f"Z shown is the block estimate of the authoritative record of {c['case']}."], dpi=170, bbox_inches="tight")
    return dict(file=str(Path(out).relative_to(OUT)).replace("\\", "/"), neck_mm=dict(liner=[r, lc["L"]],
                cfd=[a, Ln]), neck_px_liner=float(px2), neck_px_cfd=float(px3))


def Z_record(c):
    """Block estimate of the authoritative case (follow-up member record when continued)."""
    st = c["final_status"]
    if st["member"]:
        fol = VDIR / VEH[c["vehicle"]] / "statistical_followup"
        rep = next(r for r in json.loads((fol / "_followup_result.json").read_text(encoding="utf-8"))["member_reports"]
                   if r["member"] == st["member"])
        e = rep["estimate"]
        src = f"statistical follow-up member {st['member']}"
    else:
        from ..vehicles import cfd_supplemental as S
        e = S.block_estimate(str(ROOT / c["source_dir"]))
        src = c["source_dir"]
    return dict(R=float(e["R"]), X=float(e["X"]), h_R=float(e["h_R"]), h_X=float(e["h_X"]), source=src)


def geometry_figure(veh, c, out):
    D, meta = load(c)
    g = meta["geometry"]
    fin = json.loads((VDIR / VEH[veh] / "finalist.json").read_text(encoding="utf-8"))
    bank = int(bank_of(c)[1:])
    gr = fin["design"]["groups"][bank]
    Dcav = fin["design"]["D"]
    S = RM.neck_area(gr["r_neck"])
    Le = RM.effective_neck_length(gr["L_neck"], gr["r_neck"])
    V = RM.cavity_volume_for_target_f0(gr["f0"], S, Le)
    mm = 1000.0
    a, R, z0, z1, z2, z3 = (g[k] * mm for k in ("a", "R", "z0", "z1", "z2", "z3"))
    fig = plt.figure(figsize=(13.5, 6.4))
    gsp = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.25)
    # (a) computational domain, meridional half-plane to scale
    ax = fig.add_subplot(gsp[0, 0])
    ax.add_patch(Rectangle((0, z0), R, z3 - z0, fc="#e3f2fd", ec="none"))
    ax.add_patch(Rectangle((a, z1), R - a, z2 - z1, fc="0.55", ec="k", lw=1.0, hatch="///"))
    ax.plot([0, 0], [z0, z3], color="0.3", lw=1.0, ls=(0, (6, 3, 1, 3)))
    ax.plot([R, R], [z0, z3], color="k", lw=2.2)
    ax.plot([0, R], [z3, z3], color="tab:red", lw=2.5)
    ax.plot([0, R], [z0, z0], color="tab:green", lw=2.5)
    for xx in np.linspace(0.15 * R, 0.85 * R, 4):
        ax.add_patch(FancyArrowPatch((xx, z3 + 9), (xx, z3 + 1), arrowstyle="<->", mutation_scale=8, color="tab:red"))
    ax.text(R / 2, z3 + 11, f"inlet: imposed oscillatory velocity u_z = \u2212U_in sin(2\u03c0 f\u2080 t)\n"
            f"(U_in = U_neck (a/R)\u00b2)", ha="center", fontsize=7.2, color="tab:red")
    ax.text(R / 2, z0 - 3, "outlet: pressure p = 0 (reference)", ha="center", va="top", fontsize=7.2,
            color="tab:green")
    ax.text(R + 1.2, (z2 + z3) / 2, "solid wall (no slip)", rotation=90, va="center", fontsize=7.2)
    ax.text(-1.2, z2 + 0.6 * (z3 - z2), "axis of symmetry", rotation=90, va="center", ha="right", fontsize=7.2,
            color="0.3")
    ax.text((a + R) / 2, (z1 + z2) / 2, "facesheet\n(solid)", ha="center", va="center", fontsize=7,
            bbox=dict(fc="w", ec="none", alpha=0.8, pad=1))
    ax.text(a / 2, (z1 + z2) / 2, "neck", ha="center", va="center", fontsize=7.5, fontweight="bold")
    ax.annotate("", (a, z2 + 3), (0, z2 + 3), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(a / 2, z2 + 4, f"a = {a:.2f} mm", ha="center", fontsize=6.8)
    ax.annotate("", (R + 2.5, z2), (R + 2.5, z1), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(R + 3.5, (z1 + z2) / 2, f"L = {z2 - z1:.2f} mm", va="center", fontsize=6.8)
    ax.annotate("", (R, z0 + 6), (0, z0 + 6), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(R / 2, z0 + 7, f"R_tube = {R:.2f} mm", ha="center", fontsize=6.8)
    ax.annotate("", (R * 0.7, z1), (R * 0.7, z0), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(R * 0.7 - 0.8, z0 + 0.35 * (z1 - z0), f"{z1 - z0:.0f} mm", ha="right", fontsize=6.8)
    ax.annotate("", (R * 0.7, z3), (R * 0.7, z2), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(R * 0.7 - 0.8, z2 + 0.7 * (z3 - z2), f"{z3 - z2:.0f} mm", ha="right", fontsize=6.8)
    for zz, lab in ((z2 + 15, "p_up plane"), (z1 - 15, "p_down plane"), ((z1 + z2) / 2, None)):
        if lab:
            ax.plot([0, R], [zz, zz], color="tab:orange", lw=0.8, ls="--")
            ax.text(R * 0.98, zz + 0.6, lab, ha="right", fontsize=6.3, color="tab:orange")
    ax.annotate("", (-6, z0 + 12), (-6, z0 + 2), arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("", (-6 + 8, z0 + 2), (-6, z0 + 2), arrowprops=dict(arrowstyle="->", lw=1))
    ax.text(-6, z0 + 13, "z", ha="center", fontsize=8)
    ax.text(-6 + 9, z0 + 2, "r", va="center", fontsize=8)
    ax.text(0.5, -0.07, f"Bank b{bank}, f\u2080 = {c['f0_Hz']:.3f} Hz; wedge {g['wedge_deg']:.1f}\u00b0, "
            f"{meta['n_cells']} cells (nominal mesh)", transform=ax.transAxes, ha="center", fontsize=7.5)
    ax.set_xlim(-12, R + 22)
    ax.set_ylim(z0 - 12, z3 + 18)
    ax.set_aspect("equal")
    ax.axis("off")
    # (b) visual revolution of the axisymmetric domain
    ax3 = fig.add_subplot(gsp[0, 1], projection="3d")
    th = np.linspace(0, 1.5 * np.pi, 60)
    for (rr, za, zb, col, al) in ((R, z0, z1, "#90caf9", 0.25), (R, z2, z3, "#90caf9", 0.25),
                                  (a, z1, z2, "#1e88e5", 0.45)):
        T, Z = np.meshgrid(th, [za, zb])
        ax3.plot_surface(rr * np.cos(T), rr * np.sin(T), Z, color=col, alpha=al, linewidth=0)
    rr = np.linspace(a, R, 8)
    for zz in (z1, z2):
        T, RR = np.meshgrid(th, rr)
        ax3.plot_surface(RR * np.cos(T), RR * np.sin(T), np.full_like(T, zz), color="0.5", alpha=0.6, linewidth=0)
    w = math.radians(g["wedge_deg"]) * 4
    for ang in (-w / 2, w / 2):
        ax3.plot([0, R * math.cos(ang), R * math.cos(ang), 0, 0], [0, R * math.sin(ang), R * math.sin(ang), 0, 0],
                 [z0, z0, z3, z3, z0], color="tab:red", lw=1.0)
    ax3.plot([0, 0], [0, 0], [z0, z3], color="k", lw=0.8, ls="--")
    ax3.set_box_aspect((1, 1, (z3 - z0) / (2 * R)))
    ax3.set_axis_off()
    ax3.view_init(elev=18, azim=-60)
    ax3.text2D(0.02, -0.02, "Visual revolution of the axisymmetric computational\ndomain (not a 3D CFD calculation).\n"
               "Red outline: the computational wedge\n(angle exaggerated 4\u00d7 for visibility).",
               transform=ax3.transAxes, fontsize=6.6, va="top")
    # (c) the full resonator cell for context
    axc = fig.add_subplot(gsp[0, 2])
    Dm = Dcav * mm
    wcell = math.sqrt(V / Dcav) * mm
    r_n, L_n = gr["r_neck"] * mm, gr["L_neck"] * mm
    cx = wcell / 2
    axc.add_patch(Rectangle((0, 0), wcell, Dm, fc="#fff3e0", ec="k", lw=1.2))
    axc.add_patch(Rectangle((0, Dm), cx - r_n, L_n, fc="0.55", ec="k", hatch="///"))
    axc.add_patch(Rectangle((cx + r_n, Dm), wcell - cx - r_n, L_n, fc="0.55", ec="k", hatch="///"))
    axc.annotate("backing cavity (depth D = %.0f mm, V = %.1f cm³)\nNOT in the CFD domain: its stiffness enters\n"
                 "through the impedance model and the FEM" % (Dm, V * 1e6), (wcell * 0.12, Dm * 0.15),
                 (wcell * 0.5, -14), ha="center", va="top", fontsize=6.6, arrowprops=dict(arrowstyle="->", lw=0.7))
    axc.annotate("facesheet with neck\nr = %.2f mm, L = %.2f mm" % (r_n, L_n), (wcell * 0.92, Dm + L_n / 2),
                 (wcell + 8, Dm + L_n + 22), fontsize=6.6, arrowprops=dict(arrowstyle="->", lw=0.7))
    axc.text(cx, Dm + L_n + (z3 - z2) + 4, "fairing interior (sound field)", ha="center", fontsize=6.8,
             color="0.3")
    axc.set_xlim(-8, wcell + 75)
    axc.set_ylim(-40, Dm + L_n + (z3 - z2) + 14)
    axc.set_aspect("equal")
    axc.axis("off")
    axc.text(0.5, -0.13, f"Context: one resonator cell of bank b{bank} (frozen finalist; square cell of depth D sized\n"
             "for the bank's tuning frequency). Local resonator-cell model, not the complete fairing.",
             transform=axc.transAxes, ha="center", va="top", fontsize=6.8)
    save(fig, out, [f"{VNAME[veh]}: geometry of the local CFD model (a) meridional section of the computational "
                    "domain to scale, (b) visual revolution, (c) resonator-cell context"],
         [SCOPE, "The CFD domain is a 2D axisymmetric wedge of the neck in the facesheet between an upstream tube "
          "(imposed oscillatory velocity) and a downstream tube (pressure reference). The backing cavity is not "
          "part of the CFD domain, and the CFD boundary definitions do not assign either tube to the fairing-interior or the "
          "backing-cavity side.",
          "Incompressible, laminar pimpleFoam; pressure planes 15 mm above and below the facesheet define "
          "\u0394p = \u03c1 (p_up \u2212 p_down) for the impedance.", VERDICT_NOTE[veh]], bbox_inches="tight")
    return dict(bank=bank, f0=c["f0_Hz"], a_mm=a, L_mm=z2 - z1, R_tube_mm=R, tubes_mm=[z1 - z0, z3 - z2],
                cavity_D_mm=Dm, cavity_V_cm3=V * 1e6)


def render(only_veh=None):
    pl = plan()
    results = dict(stills=[], montages=[], animations=[], coupled=[], geometry=[], orientation=[], limits={})
    for veh in VEH:
        if only_veh and veh != only_veh:
            continue
        cs = [c for c in pl["cases"] if c["vehicle"] == veh]
        fr = frame(cs)
        own = {c["dir"]: case_limits(load(c)[0]) for c in cs}
        shared = union_limits(list(own.values()))
        lims = {"shared": shared, **own}
        results["limits"][veh] = dict(shared=shared, individual=own, frame=fr)
        vd = OUT / veh
        # stills
        for c in cs:
            D, meta = load(c)
            P = Panel(c, D, meta)
            tagc = c["dir"].replace("/", "__")
            for key in FIELDS:
                for ph in PHASES:
                    i = phase_index(D, c["f0_Hz"], ph, 0)
                    for sh in (True, False):
                        lim = shared if sh else own[c["dir"]]
                        p = vd / "stills" / f"{tagc}__{key}__ph{ph:03d}__{'shared' if sh else 'individual'}.png"
                        pts = still(c, P, D, key, i, lim[key], fr, sh, p, lim["umag"][1])
                        results["stills"].append(dict(file=str(p.relative_to(OUT)).replace("\\", "/"), case=c["dir"],
                                                      field=key, phase_target=ph,
                                                      theta=float(theta_of(D["t"][i], c["f0_Hz"])),
                                                      t=float(D["t"][i]), limits=list(lim[key]),
                                                      arrow_points=None if pts is None else pts.tolist()))
            print("stills", c["dir"], flush=True)
        # montages
        by = {c["slot"]: c for c in cs if c["group"] == "A"}
        speed = [by["low"], by["ustar"], by["high"]]
        numer = [by["ustar"]] + [c for c in cs if c["group"] == "B"]
        numer.sort(key=lambda c: ["nominal", "sens_refine", "sens_halfdt"].index(c["case"].split("/")[0]))
        for key in FIELDS:
            for sh in (True, False):
                p = vd / f"{veh}_speed_{key}_{'shared' if sh else 'individual'}.png"
                info, arr = montage(veh, speed, key, lims, sh, fr, p, "speed")
                results["montages"].append(dict(file=str(p.relative_to(OUT)).replace("\\", "/"), kind="speed",
                                                field=key, shared=sh, panels=info,
                                                limits=list(shared[key]) if sh else
                                                {c["dir"]: list(own[c["dir"]][key]) for c in speed},
                                                arrow_points={d: a.tolist() for d, a in arr}))
            p = vd / f"{veh}_numerics_{key}_shared.png"
            info, arr = montage(veh, numer, key, lims, True, fr, p, "numerics")
            results["montages"].append(dict(file=str(p.relative_to(OUT)).replace("\\", "/"), kind="numerics",
                                            field=key, shared=True, panels=info, limits=list(shared[key]),
                                            arrow_points={d: a.tolist() for d, a in arr}))
            print("montages", veh, key, flush=True)
        # per-case animations (individual scale, fixed over all frames)
        for c in cs:
            for key in ("umag", "logw", "phi"):
                p = vd / "animations" / f"{c['dir'].replace('/', '__')}__{key}.gif"
                a = animation(c, key, own[c["dir"]], fr, p)
                a.update(case=c["dir"], field=key)
                results["animations"].append(a)
            print("animations", c["dir"], flush=True)
        # coupled animation at the nearest-U* nominal case
        cz = by["ustar"]
        Z = Z_record(cz)
        a = coupled(veh, cz, own[cz["dir"]], fr, vd / f"{veh}_coupled_mechanism.gif", Z)
        a.update(case=cz["dir"])
        results["coupled"].append(a)
        o = orientation_figure(veh, cz, vd / f"{veh}_multiscale_orientation.png")
        o.update(vehicle=veh)
        results["orientation"].append(o)
        g = geometry_figure(veh, cz, vd / f"{veh}_geometry.png")
        g.update(vehicle=veh, file=f"{veh}/{veh}_geometry.png")
        results["geometry"].append(g)
        print("coupled + geometry", veh, flush=True)
    prev = OUT / "_viz_results.json"
    old = json.loads(prev.read_text()) if (only_veh and prev.exists()) else None
    if old:
        for k in ("stills", "montages", "animations", "coupled", "geometry", "orientation"):
            old[k] = [x for x in old[k] if not str(x.get("file", "")).startswith(only_veh + "/")] + results[k]
        old["limits"].update(results["limits"])
        results = old
    prev.write_text(json.dumps(results, indent=0, default=float))
    nb = OUT / "_notes.json"
    allnotes = json.loads(nb.read_text(encoding="utf-8")) if nb.exists() else {}
    allnotes.update(NOTES)
    nb.write_text(json.dumps(allnotes, indent=1), encoding="utf-8")


if __name__ == "__main__":
    from .. import cli
    if cli.show_help("figures", "cfd-fields"):
        raise SystemExit(0)
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "extract":
        extract(sys.argv[2:] or None)
    elif cmd == "render":
        render(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        sys.exit(__doc__)
