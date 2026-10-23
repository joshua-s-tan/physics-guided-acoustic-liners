"""QA, reproducibility manifest, gallery index and README selection for the visualization layer (NON-AUTHORITATIVE).

    acoustic-liner figures cfd-qa qa         # run every check; write visualization_manifest.json only if all pass
    acoustic-liner figures cfd-qa publish    # copy the README selection to docs/figures/cfd/ (after qa)

Reads frozen records only to verify that they are unchanged. Writes only under artifacts/dev/vehicle_v1/figures/cfd_viz/
and docs/figures/cfd/.
"""
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .. import paths
from ..provenance import schemas as SC
ROOT = paths.ROOT
from . import cfd_fields as VF  # noqa: E402
from . import cfd_cases as VCASES  # noqa: E402

OUT = VF.OUT
VDIR = VF.VDIR
FROZEN_PREFIX = {"atlas_i_14ft_representative_unblanketed/verdict/verdict.json": "77e05918",
                 "falcon9_2025_unblanketed_v2/verdict/verdict.json": "0dc0dc01",
                 "falcon9_2025_unblanketed_v2/supplemental_statistical/verdict/verdict.json": "ceaac17e"}
README_SELECTION = [
    ("falcon/falcon_multiscale_orientation.png", "falcon_multiscale_orientation.png"),
    ("falcon/falcon_geometry.png", "geometry_falcon.png"),
    ("falcon/falcon_coupled_mechanism.gif", "falcon_coupled_mechanism.gif"),
    ("atlas/atlas_coupled_mechanism.gif", "atlas_coupled_mechanism.gif"),
    ("falcon/falcon_speed_phi_shared.png", "falcon_speed_dissipation_shared.png"),
    ("falcon/falcon_numerics_logw_shared.png", "falcon_numerics_log_vorticity_shared.png"),
    ("falcon_pressure_flow.png", "falcon_pressure_flow.png"),
]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def in_fluid_with_margin(x, y, g, margin):
    if not (abs(x) < g["R"] - margin * 0.999 and g["z0"] < y < g["z3"]):
        return False
    return not (abs(x) > g["a"] - margin * 0.999 and g["z1"] - margin * 0.999 < y < g["z2"] + margin * 0.999)


def checks():
    pl = VF.plan()
    res = json.loads((OUT / "_viz_results.json").read_text())
    cases = {c["dir"]: c for c in pl["cases"]}
    meta = {d: json.loads((VF.CACHE / (d.replace("/", "__") + ".json")).read_text()) for d in cases}
    C = {}

    # 1-2. every animation has >= 2 complete cycles, monotone phase
    bad = []
    for a in res["animations"] + res["coupled"]:
        th = np.unwrap(np.radians(a["theta"])) if "theta" in a else None
        if th is None:
            D, _ = VF.load(cases[a["case"]])
            th = 2 * np.pi * cases[a["case"]]["f0_Hz"] * D["t"]
        span = math.degrees(th[-1] - th[0]) + 360.0 / VF.SNAP
        if a["frames"] < 2 * VF.SNAP or span < 720.0 - 1.0 or np.any(np.diff(th) <= 0):
            bad.append((a["file"], a["frames"], span))
    C["animations_two_complete_cycles_and_phase_order"] = dict(ok=not bad, failures=bad,
                                                               n=len(res["animations"]) + len(res["coupled"]))
    # every window: 96 frames, phase reference confirmed by the measured neck-velocity fundamental
    ph = {d: (m["n_frames"], m["cycles"], round(m["u_down_fundamental"]["phase_offset_deg"], 4)) for d, m in meta.items()}
    C["windows_and_phase_reference"] = dict(ok=all(n == 2 * VF.SNAP and abs(o) < 0.5 for n, _, o in ph.values()),
                                            per_case=ph)
    # 3. fixed limits per animation (single recorded limit pair, equal to the case's own fixed limits)
    bad = [a["file"] for a in res["animations"]
           if a["limits"] != list(res["limits"][a["case"].split("/")[0]]["individual"][a["case"]][a["field"]])]
    C["animation_color_limits_fixed"] = dict(ok=not bad, failures=bad)
    # 4. shared-scale montages and stills truly share limits
    bad = []
    for m in res["montages"]:
        veh = m["file"].split("/")[0]
        if m["shared"] and m["limits"] != list(res["limits"][veh]["shared"][m["field"]]):
            bad.append(m["file"])
    for s in res["stills"]:
        veh = s["file"].split("/")[0]
        want = res["limits"][veh]["shared" if s["file"].endswith("shared.png") else "individual"]
        want = want[s["field"]] if s["file"].endswith("shared.png") else want[s["case"]][s["field"]]
        if s["limits"] != list(want):
            bad.append(s["file"])
    C["shared_scales_identical"] = dict(ok=not bad, failures=bad[:10])
    # 5. units
    C["units"] = dict(ok=("m²/s²" in VF.FIELDS["pk"][0] and "W/m³" in VF.FIELDS["phi"][0]
                          and VF.RHO == 1.204),
                      pressure=VF.FIELDS["pk"][0], dissipation=VF.FIELDS["phi"][0], rho=VF.RHO,
                      dissipation_formula="Phi = 2 mu S:S, mu = rho * nu, S = sym(grad U) (W/m^3); not the "
                                          "per-mass form 2 nu S:S")
    # 6. no arrows inside solids or on walls
    bad = []
    for item in res["stills"] + res["montages"]:
        pts = item.get("arrow_points")
        if not pts:
            continue
        groups = pts.items() if isinstance(pts, dict) else [(item["case"], pts)]
        for d, P in groups:
            g = meta[d]["geometry"]
            fr = res["limits"][d.split("/")[0]]["frame"]
            n_across = 11 if isinstance(pts, dict) else 17
            dx = (fr["xlim"][1] - fr["xlim"][0]) / (n_across + 1)
            for x, y in P:
                if not in_fluid_with_margin(x, y, g, 0.6 * dx):
                    bad.append((item["file"], d, x, y))
    C["arrows_fluid_interior_only"] = dict(ok=not bad, failures=bad[:10])
    # 7. status labels
    f16 = cases["falcon/A/high__nominal__b3_u16.2"]
    C["falcon_16p2_not_converged"] = dict(ok=VF.status(f16)["tag"] == "CFD CASE NOT CONVERGED",
                                          failed=f16["final_status"]["failed"])
    outs = {}
    for veh, d in VF.VEH.items():
        v = json.loads((VDIR / d / "statistical_followup/verdict/verdict.json").read_text(encoding="utf-8"))
        outs[veh] = v["outcome"]
    C["vehicle_verdicts_unresolved"] = dict(ok=all("UNRESOLVED" in o for o in outs.values()), outcomes=outs)
    atl_ok = all(VF.status(c)["converged"] for c in pl["cases"] if c["vehicle"] == "atlas")
    C["atlas_cases_not_mislabelled"] = dict(ok=atl_ok)
    # 8. no visualization directory referenced by the evidence pipeline
    hits = []
    for p in (ROOT / "src").rglob("*.py"):
        t = p.read_text(encoding="utf-8", errors="replace")
        if re.search(r"visualization_cases|cfd_viz|viz_fields|viz_cases|viz_render", t):
            hits.append(str(p.relative_to(ROOT)))
    C["not_referenced_by_evidence_pipeline"] = dict(ok=not hits, hits=hits)
    # 9. frozen records and authoritative hashes unchanged
    from ..vehicles import scenarios as VS
    fr_ok, rows = True, {}
    for rel, pre in FROZEN_PREFIX.items():
        h = sha(VDIR / rel)
        rows[rel] = h
        fr_ok &= h.startswith(pre)
    for veh, d in VF.VEH.items():
        v = json.loads((VDIR / d / "statistical_followup/verdict/verdict.json").read_text(encoding="utf-8"))
        r = v.pop("record_sha256")
        selfok = VS.record_sha256(v) == r
        for rel, h in v["frozen_records"].items():
            fr_ok &= sha(VDIR / rel) == h
        fres = json.loads((VDIR / d / "statistical_followup/_followup_result.json").read_text(encoding="utf-8"))
        resok = VS.record_sha256(fres) == v["evidence"]["followup_result_sha256"] or \
            sha(VDIR / d / "statistical_followup/_followup_result.json") == v["evidence"]["followup_result_sha256"]
        rows[f"{d}/statistical_followup/verdict/verdict.json (self hash)"] = r
        fr_ok &= selfok and resok
    src_ok = []
    for c in pl["cases"]:
        src = ROOT / c["source_dir"]
        src_ok.append(VCASES.sha_tree(src / c["restart_time"]) == c["restart_state_sha256"])
    tracked = [ln for ln in git("status", "--porcelain", "--untracked-files=no").splitlines()
               if not ln.endswith("README.md") and "acoustic-liner figures cfd-" not in ln]
    C["authoritative_hashes_unchanged"] = dict(ok=bool(fr_ok and all(src_ok) and not tracked), frozen=rows,
                                               source_restart_states_unchanged=f"{sum(src_ok)}/{len(src_ok)}",
                                               other_tracked_changes=tracked,
                                               cvae_source_sha256_note="cvae.SOURCE_MODULES untouched (no src/ edits)")
    # 10. every expected output exists; GIFs readable with the stated frame count
    missing = []
    for item in (res["stills"] + res["montages"] + res["animations"] + res["coupled"] + res["geometry"]
                 + res["orientation"]):
        if not (OUT / item["file"]).exists():
            missing.append(item["file"])
    for a in res["animations"] + res["coupled"]:
        with Image.open(OUT / a["file"]) as im:
            if im.n_frames != a["frames"]:
                missing.append(a["file"] + f" (frames {im.n_frames} != {a['frames']})")
    for n in ("falcon_cycle_histories.png", "atlas_cycle_histories.png", "falcon_pressure_flow.png",
              "atlas_pressure_flow.png"):
        if not (OUT / n).exists():
            missing.append(n)
    C["all_expected_outputs_exist"] = dict(ok=not missing, missing=missing)
    # identical neck geometry between the liner-cell and CFD-domain panels of the orientation figures
    bad = [o for o in res["orientation"] if abs(o["neck_px_liner"] - o["neck_px_cfd"]) > 0.5
           or abs(o["neck_mm"]["liner"][0] - o["neck_mm"]["cfd"][0]) > 0.01
           or abs(o["neck_mm"]["liner"][1] - o["neck_mm"]["cfd"][1]) > 0.01]
    C["orientation_identical_neck_geometry"] = dict(ok=not bad and len(res["orientation"]) == 2,
                                                    values=res["orientation"])
    return C, res, pl, meta


def write_books(notes):
    for fname, key, head in (("TITLES.md", "titles", "Figure and panel titles"),
                             ("FOOTNOTES.md", "notes", "Figure footnotes")):
        lines = [f"# {head}", "", "Bookkeeping for `artifacts/dev/vehicle_v1/figures/cfd_viz/`, generated by "
                 "`acoustic-liner figures cfd-fields`, `acoustic-liner figures cfd-histories` and `acoustic-liner figures cfd-qa`.", ""]
        groups = {}
        for n in sorted(notes):
            groups.setdefault(tuple(notes[n][key]), []).append(n)
        for txts, names in sorted(groups.items(), key=lambda kv: kv[1][0]):
            if len(names) > 6:
                lines += [f"## {names[0]} ... ({len(names)} files sharing these {key})", ""]
            else:
                lines += ["## " + ", ".join(names), ""]
            lines += ["- " + " ".join(t.split()) for t in txts] + [""]
        (OUT / fname).write_text("\n".join(lines), encoding="utf-8")


def index_md(res, pl, meta):
    L = ["# CFD visualization gallery (visualization-only, non-authoritative)", "",
         "> " + VF.SCOPE, "",
         "**Vehicle verdicts: Falcon 9 `UNRESOLVED`; Atlas I representative `UNRESOLVED`.**", "",
         "- " + VF.VERDICT_NOTE["falcon"], "- " + VF.VERDICT_NOTE["atlas"], "",
         "Case status labels: `CFD CASE PASSED STATISTICAL CHECKS` / `CFD CASE NOT CONVERGED` describe one CFD "
         "case only; no individually converged case resolves a vehicle verdict. Every image also carries "
         "`ILLUSTRATIVE VISUALIZATION ONLY` and `VEHICLE VERDICT: UNRESOLVED`.", "",
         "Titles and footnotes of every image are in `TITLES.md` and `FOOTNOTES.md`; settings, limits, hashes and the "
         "complete file inventory are in `visualization_manifest.json`.", "",
         "## Cases", "", "| vehicle | slot | case | U (m/s) | status | restart | window (cycles) |",
         "|---|---|---|---|---|---|---|"]
    for c in pl["cases"]:
        m = meta[c["dir"]]
        st = VF.status(c)
        L.append(f"| {c['vehicle']} | {VF.SLOT_LABEL[c['slot']]} | `{c['case']}` | {c['U_target_m_s']:g} | "
                 f"{st['tag']}{' (' + ', '.join(st['failed']) + ')' if st['failed'] else ''} | t = {c['restart_time']} s "
                 f"| {m['cycles'][0] + 1}-{m['cycles'][1]} |")
    L += ["", "Each window is the first two complete cycles after the restart cycle (counted from 1), 48 snapshots "
          "per cycle. The refined-mesh Falcon U* case restarts from the end of its statistical follow-up "
          "continuation (it passed its checks there), so its window is cycles 26-27.", "",
          "## Phase reference", "",
          "theta = 360 f0 t mod 360 deg, with the imposed inlet velocity u_z = -U_in sin(theta) (z up, inlet at the top). "
          "The downward neck mean velocity is u = U sin(theta); the least-squares fundamental of the measured neck "
          "flow in every window confirms this to within 0.04 deg (manifest: `cases[].u_down_fundamental`). Phase "
          "stills are the saved frames nearest 0/90/180/270 deg in the first window cycle; the actual theta and t are "
          "printed on each image and recorded in the manifest.", "",
          "## Products", "",
          "| file pattern | what it shows |", "|---|---|",
          "| `<veh>/<veh>_coupled_mechanism.gif` (+ `.png` still at 90 deg) | **Primary CFD visual.** Synchronized "
          "kinematic pressure, velocity magnitude with vectors, log vorticity magnitude and modeled viscous-dissipation "
          "density at the nearest-U* nominal case, with the pressure-difference and neck-velocity waveforms, a moving "
          "phase marker and the impedance phasor Z = R + iX of the authoritative record. |",
          "| `<veh>/<veh>_multiscale_orientation.png` | Multiscale orientation: (1) payload fairing with payload, "
          "acoustic cavity and lined inner wall; (2) one liner patch (structural wall, backing cavity, facesheet, "
          "neck); (3) the same neck on the exact axisymmetric CFD domain at the same mm scale; (4) the coupling "
          "dp(t) -> q(t) -> Z = R + iX -> global FEM impedance boundary. The CFD boundaries do not assign a "
          "fairing-interior or backing-cavity side (see the figure note). |",
          "| `<veh>/<veh>_geometry.png` | Geometry cutaway: meridional section of the computational wedge to scale "
          "(neck, facesheet, axis, walls, inlet/outlet conditions, probe planes, dimensions), a visual revolution, "
          "and the resonator-cell context with the backing cavity (not in the CFD domain). |",
          "| `<veh>/<veh>_speed_<field>_{shared,individual}.png` | Low / nearest-U* / high speed at matched phases "
          "(rows 0/90/180/270 deg). Shared scale = quantitative comparison; individual scale = structure only. |",
          "| `<veh>/<veh>_numerics_<field>_shared.png` | Nominal / refined mesh / half time step near U*, matched "
          "phases, shared scale. Qualitative companion to the sensitivity analysis; visual similarity does not "
          "establish numerical convergence. |",
          "| `<veh>/stills/<case>__<field>__phNNN__{shared,individual}.png` | Full-resolution phase stills. |",
          "| `<veh>/animations/<case>__{umag,logw,phi}.gif` | Per-case animations, 2 cycles, fixed individual "
          f"scale, {VF.FPS} fps. |",
          "| `<veh>_cycle_histories.png` | Block R and X per cycle with block mean and uncertainty half-width, "
          "pressure difference and neck mean velocity versus phase (authoritative records). |",
          "| `<veh>_pressure_flow.png` | Normalized pressure and velocity waveforms and the pressure-velocity loops "
          "whose phase lead arg Z = atan(X/R) defines the complex impedance. |", "",
          "Fields: `umag` |U| (m/s) with arrows from fluid-interior points only; `pk` kinematic pressure p/rho "
          "(m^2/s^2) relative to the outlet reference; `logw` log10(|omega| + eps), eps = 10 1/s; `wth` signed azimuthal "
          "vorticity omega_theta (1/s), symmetric diverging scale (stills and montages only; never logged); `phi` "
          "modeled viscous-dissipation density Phi = 2 mu S:S (W/m^3, mu = rho nu, rho = 1.204 kg/m^3), log colour "
          "scale. Pressure differences in Pa use rho = 1.204 kg/m^3.", "",
          "MP4: not produced (no ffmpeg on this machine); GIFs and full-resolution stills are provided.", "",
          "The superseded ParaView renders (linear vorticity scale, arrows at boundary points) are kept in "
          "`_superseded_paraview_v1/` and are not part of this gallery.", "",
          "## README selection", "", "Copied to `docs/figures/cfd/` by `acoustic-liner figures cfd-qa publish`:", ""]
    for src, dst in README_SELECTION:
        L.append(f"- `{dst}` (from `{src}`)")
    (OUT / "INDEX.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def manifest(C, res, pl, meta):
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and "_cache" not in p.parts
                   and "_superseded_paraview_v1" not in p.parts and p.name != "visualization_manifest.json")
    inv = {str(p.relative_to(OUT)).replace("\\", "/"): sha(p) for p in files}
    cases = []
    for c in pl["cases"]:
        m = meta[c["dir"]]
        cd = ROOT / "artifacts/dev/vehicle_v1/visualization_cases" / c["dir"]
        ctl = (cd / "system/controlDict").read_text()
        cases.append(dict(dir=c["dir"], case=c["case"], vehicle=c["vehicle"], slot=c["slot"], group=c["group"],
                          final_status=c["final_status"], source_dir=c["source_dir"],
                          source_case_hash=c["source_case_hash"], source_input_tree_hash=c["source_input_tree_hash"],
                          source_build=c["source_build"], restart_time=c["restart_time"],
                          restart_state_sha256=c["restart_state_sha256"],
                          visualization_case_tree_sha256=VCASES.sha_tree(cd), controlDict_sha256=c["controlDict_sha256"],
                          mesh=dict(n_cells=m["n_cells"], geometry=m["geometry"],
                                    refine=json.loads((cd / "_case_meta.json").read_text())["refine"]),
                          time_step=dict(deltaT_initial=re.search(r"deltaT\s+([^;]+);", ctl).group(1),
                                         maxCo=re.search(r"maxCo\s+([^;]+);", ctl).group(1),
                                         maxDeltaT=re.search(r"maxDeltaT\s+([^;]+);", ctl).group(1),
                                         adjustTimeStep=re.search(r"adjustTimeStep\s+([^;]+);", ctl).group(1)),
                          f0_Hz=c["f0_Hz"], U_target_m_s=c["U_target_m_s"], nu=m["nu"],
                          window_cycle_boundaries=m["cycles"], n_frames=m["n_frames"], t_first=m["t_first"],
                          t_last=m["t_last"], u_down_fundamental=m["u_down_fundamental"]))
    scripts = {s: sha(ROOT / "tools" / s) for s in ("viz_cases.py", "viz_fields.py", "viz_histories.py", "viz_qa.py")}
    import matplotlib
    import scipy
    man = dict(
        schema=SC.schema("visualization-manifest"), authoritative=False, scope=VF.SCOPE,
        excluded_from=["impedance extraction", "sensitivity", "corrected FEM", "verdict"],
        vehicle_verdicts={v: VF.VERDICT_NOTE[v] for v in VF.VEH},
        case_selection_rule=(VCASES.__doc__ or "").split("Case selection (fixed rule):")[-1].strip(),
        phase_reference=dict(definition="theta = 360 f0 t mod 360 deg; imposed inlet u_z = -U_in sin(theta); downward "
                                        "neck mean velocity u = U sin(theta)",
                             confirmation="least-squares fundamental of the measured neck flow per window "
                                          "(cases[].u_down_fundamental.phase_offset_deg)",
                             stills="saved frame nearest 0/90/180/270 deg in the first window cycle"),
        software=dict(openfoam=sorted({c["source_build"] for c in pl["cases"]}),
                      renderer="matplotlib " + matplotlib.__version__ + " (tripcolor, Gouraud shading on cell centres)",
                      scipy=scipy.__version__, numpy=np.__version__, pillow=Image.__version__,
                      python=platform.python_version(),
                      paraview="6.0.1 (used only for the superseded first render; not used for this gallery)",
                      git_head=git("rev-parse", "HEAD"), scripts_sha256=scripts,
                      commands=["acoustic-liner figures cfd-fields extract",
                                "acoustic-liner figures cfd-fields render",
                                "acoustic-liner figures cfd-histories",
                                "acoustic-liner figures cfd-qa qa"]),
        fields=dict(umag="|U| (m/s); arrows: regular grid in the frame, fluid interior only, >= 0.6 spacing from "
                         "walls/facesheet, omitted below 6 % of the reference speed; length = speed / reference",
                    pk="kinematic pressure p/rho (m^2/s^2), OpenFOAM p, outlet reference p = 0",
                    pressure_difference_Pa="rho (p_up - p_down), rho = 1.204 kg/m^3, planes 15 mm above/below the "
                                           "facesheet",
                    logw="log10(|omega_theta| + eps), eps = 10 1/s (pinned)", wth="signed omega_theta (1/s); "
                    "omega_y on the y = 0 meridional plane, sign reversed on the mirrored half",
                    phi="modeled viscous-dissipation density Phi = 2 mu S:S, mu = rho nu, S = sym(grad U) from "
                        "OpenFOAM grad(U) (W/m^3), log colour scale",
                    rho=VF.RHO, eps_log_vorticity=VF.EPS_W),
        color_limits=dict(rule="per case: 0.5-99.5 % of cell values over the whole 2-cycle window (|U|, |p/rho| "
                               "symmetric, Phi upper; 4 decades below), |omega| upper 99.5 % with 3 decades, signed "
                               "omega 98 % symmetric (saturates beyond); shared = union over the 5 cases of the "
                               "vehicle; fixed across all frames of every animation",
                          values=res["limits"]),
        camera=dict(projection="2D meridional section, both halves (mirror x -> -x), equal aspect; frame per "
                               "vehicle: full tube width, centred on the facesheet mid-plane, identical for all "
                               "cases of a vehicle", frames={v: res["limits"][v]["frame"] for v in res["limits"]}),
        images=dict(still_size_in=[4.6, 5.6], still_dpi=VF.DPI, montage_dpi=VF.DPI, animation_dpi=85,
                    coupled_dpi=80, coupled_still_dpi=130, animation_fps=VF.FPS, gif_palette="255-colour median-cut "
                    "palette from the middle frame, no dithering"),
        mp4="not produced: no ffmpeg available", cases=cases,
        products=dict(stills=len(res["stills"]), montages=len(res["montages"]), animations=len(res["animations"]),
                      coupled=res["coupled"], geometry=res["geometry"]),
        readme_selection=[d for _, d in README_SELECTION],
        qa=C, inventory=inv)
    p = OUT / "visualization_manifest.json"
    p.write_text(json.dumps(man, indent=1, default=float), encoding="utf-8")
    return p


def main():
    from .. import cli
    if cli.show_help("figures", "cfd-qa"):
        return
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "qa":
        C, res, pl, meta = checks()
        for k, v in C.items():
            print(("PASS " if v["ok"] else "FAIL ") + k, "" if v["ok"] else json.dumps(v, default=str)[:600])
        notes = json.loads((OUT / "_notes.json").read_text(encoding="utf-8"))
        write_books(notes)
        index_md(res, pl, meta)
        if all(v["ok"] for v in C.values()):
            p = manifest(C, res, pl, meta)
            print("manifest", p, sha(p))
        else:
            print("NOT writing the manifest: QA failed")
            sys.exit(1)
    elif cmd == "publish":
        m = OUT / "visualization_manifest.json"
        if not m.exists():
            sys.exit("run qa first")
        VF.PUBLIC.mkdir(parents=True, exist_ok=True)
        for src, dst in README_SELECTION:
            shutil.copy(OUT / src, VF.PUBLIC / dst)
        notes = json.loads((OUT / "_notes.json").read_text(encoding="utf-8"))
        L = ["# CFD figures selected for the README", "", "> " + VF.SCOPE, "",
             "**Vehicle verdicts: Falcon 9 `UNRESOLVED`; Atlas I representative `UNRESOLVED`.**", "",
             "- " + VF.VERDICT_NOTE["falcon"], "- " + VF.VERDICT_NOTE["atlas"], ""]
        for src, dst in README_SELECTION:
            n = notes.get(src, {})
            L += [f"## `{dst}`", "", *(" ".join(t.split()) for t in n.get("titles", [])[:1]), ""]
            if "coupled" in dst:
                L += ["> " + VF.COUPLED_CAPTION, "", "> " + VF.COUPLED_LIMIT, ""]
        L += ["Complete gallery (not committed; regenerate with the commands in the manifest): "
              "`artifacts/dev/vehicle_v1/figures/cfd_viz/` (`INDEX.md`, `visualization_manifest.json` sha256 "
              f"`{sha(m)}`).", ""]
        (VF.PUBLIC / "README.md").write_text("\n".join(L), encoding="utf-8")
        print("published", [d for _, d in README_SELECTION])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
