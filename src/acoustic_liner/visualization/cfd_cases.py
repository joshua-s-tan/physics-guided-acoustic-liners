"""Visualization-only CFD continuations (NON-AUTHORITATIVE).

Builds copies under artifacts/dev/vehicle_v1/visualization_cases/<vehicle>/<group>/<slot>__<case>/ from the final
saved state of each selected case, plus run_viz.sh (WSL). Nothing here is ever read by impedance extraction,
sensitivity, corrected FEM or verdict code; source case directories are only read, never written.

    acoustic-liner figures cfd-cases prepare
    bash artifacts/dev/vehicle_v1/visualization_cases/run_viz.sh A|B [jobs]     (in WSL)

Case selection (fixed rule): the controlling bank of the frozen five-case selection (the only bank with mesh and
time-step variants); low = its low-speed pair source, ustar = its U*-slot pair source, high = its high-slot pair
source (all from statistical_followup/_pairs.json). Group A = the three nominal sources per vehicle; group B = the
two variants of the U*-slot pair.
"""
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path

from .. import paths
from ..provenance import schemas as SC
ROOT = paths.ROOT
VDIR = ROOT / "artifacts/dev/vehicle_v1"
OUT = VDIR / "visualization_cases"
VEH = {"falcon": "falcon9_2025_unblanketed_v2", "atlas": "atlas_i_14ft_representative_unblanketed"}
SNAPS_PER_CYCLE = 48
FULL_CYCLES = 2                    # complete cycles after the restart cycle is finished
TIME_RE = re.compile(r"^\d+(\.\d+)?(e-?\d+)?$")
SCOPE = ("These visualization-only continuations illustrate local, phase-resolved flow in a representative resonator "
         "neck. They show the modeled mechanisms associated with nonlinear impedance - neck acceleration, separation, "
         "vortex formation and viscous dissipation. They are not full-fairing acoustic CFD, do not simulate propagating "
         "sound through the fairing and are not used as numerical evidence in the vehicle verdict. Global attenuation "
         "is evaluated separately using the payload-resolving FEM with CFD-derived impedance corrections.")


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_tree(d):
    h = hashlib.sha256()
    for p in sorted(x for x in Path(d).rglob("*") if x.is_file()):
        h.update(p.relative_to(d).as_posix().encode() + b"\0" + sha_file(p).encode() + b"\n")
    return h.hexdigest()


def build(case):
    m = re.search(r"Build\s*:\s*(\S+)", (Path(case) / "log.pimpleFoam").read_text(errors="replace"))
    return m.group(1) if m else None


def latest_time(case):
    ts = [d for d in Path(case).iterdir() if d.is_dir() and TIME_RE.match(d.name) and d.name != "0"]
    return max(ts, key=lambda d: float(d.name))


def selection(vehicle):
    pairs = json.loads((VDIR / VEH[vehicle] / "statistical_followup/_pairs.json").read_text())["pairs"]
    by_slot = {}
    for p in pairs:
        by_slot.setdefault(p["slot"], []).append(p)
    out = []
    for slot in ("low", "ustar", "high"):
        p = by_slot[slot][0]
        out.append(dict(group="A", slot=slot, case=f"nominal/{p['name']}", src=p["source_dir"], pair=None,
                        gate_failed=p["source"]["failed"], continue_pair=p["continue_pair"]))
    for p in by_slot["ustar"]:
        out.append(dict(group="B", slot="ustar", case=p["case"], src=p["variant_dir"], pair=p["case"],
                        gate_failed=p["variant"]["failed"], continue_pair=p["continue_pair"]))
    return out


def prepare():
    if OUT.exists() and any(OUT.iterdir()):
        sys.exit(f"{OUT} already holds visualization cases; remove it first to rebuild")
    plan = dict(schema=SC.schema("visualization-cases"), authoritative=False, scope=SCOPE,
                excluded_from=["impedance extraction", "sensitivity", "corrected FEM", "verdict"],
                snapshots_per_cycle=SNAPS_PER_CYCLE, full_cycles_after_restart=FULL_CYCLES,
                phase_reference="imposed inlet velocity u_z = -U_in*sin(2*pi*f0*t) (incompressible: neck volume "
                                "flow is in phase); phase theta = 360*f0*t mod 360 deg",
                cases=[])
    for veh in VEH:
        for s in selection(veh):
            plan["cases"].append(prep_case(veh, s))
    (OUT / "_viz_plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    write_runner(plan)
    print(f"prepared {len(plan['cases'])} visualization cases under {OUT}")


def prep_case(veh, s):
    src = ROOT / s["src"].replace("\\", "/")
    meta = json.loads((src / "_case_meta.json").read_text())
    f0 = float(meta["f0_Hz"])
    tr = latest_time(src)
    t_r = float(tr.name)
    end_cycle = math.ceil(t_r * f0 - 1e-9) + FULL_CYCLES
    dst = OUT / veh / s["group"] / f"{s['slot']}__{s['case'].replace('/', '__')}"
    dst.mkdir(parents=True)
    for sub in ("constant", "system"):
        shutil.copytree(src / sub, dst / sub)
    shutil.copytree(tr, dst / tr.name)
    shutil.copy(src / "_case_meta.json", dst / "_case_meta.json")
    cd = dst / "system/controlDict"
    txt = cd.read_text()
    new = re.sub(r"^startFrom\s+[^;]+;", "startFrom       latestTime;", txt, flags=re.M)
    new = re.sub(r"^endTime\s+[^;]+;", f"endTime         {end_cycle / f0!r};", new, flags=re.M)
    new = re.sub(r"writeInterval\s+0\.001;", f"writeInterval {1.0 / (SNAPS_PER_CYCLE * f0)!r};", new, count=1)
    new = re.sub(r"purgeWrite\s+\d+;", "purgeWrite      0;", new, count=1)
    new = new.replace("writeCompression off;", "writeCompression off; timePrecision 10;", 1)
    for k in ("startFrom       latestTime;", "purgeWrite      0;", "timePrecision 10;",
              f"writeInterval {1.0 / (SNAPS_PER_CYCLE * f0)!r};"):
        if k not in new:
            raise SystemExit(f"{cd}: controlDict edit failed ({k})")
    cd.write_text(new, newline="\n")
    return dict(
        vehicle=veh, group=s["group"], slot=s["slot"], case=s["case"], pair=s["pair"],
        dir=dst.relative_to(OUT).as_posix(), source_dir=s["src"].replace("\\", "/"),
        source_case_hash=meta["case_hash"], source_input_tree_hash=meta.get("input_tree_hash"),
        source_build=build(src), restart_time=tr.name, restart_state_sha256=sha_tree(tr),
        f0_Hz=f0, U_target_m_s=meta["uneck_m_s"], end_cycle=end_cycle, end_time=end_cycle / f0,
        analysis_cycles=[end_cycle - FULL_CYCLES + 1, end_cycle],
        write_interval_s=1.0 / (SNAPS_PER_CYCLE * f0), controlDict_sha256=sha_file(cd),
        stability_gates_failed_on_existing_history=s["gate_failed"],
        pair_needed_continuation=s["continue_pair"])


def reprepare_continued(vehicle, case_dir):
    """Rebuild one group-B case from the final state of its statistical follow-up continuation (used only after the
    follow-up verdict exists and that continued member passed its checks). The old copy is kept as <dir>.precontinuation."""
    plan = json.loads((OUT / "_viz_plan.json").read_text(encoding="utf-8"))
    i = next(k for k, c in enumerate(plan["cases"]) if c["dir"] == case_dir)
    old = plan["cases"][i]
    fol = VDIR / VEH[vehicle] / "statistical_followup"
    res = json.loads((fol / "_followup_result.json").read_text(encoding="utf-8"))
    member = old["pair"].replace("/", "__") + "/variant"
    rep = next(r for r in res["member_reports"] if r["member"] == member)
    if not (rep["ok"] and (fol / "verdict/verdict.json").exists()):
        raise SystemExit(f"{member}: no final record or checks not passed; not re-preparing")
    d = OUT / case_dir
    d.rename(d.with_name(d.name + ".precontinuation"))
    src = (fol / "continuation" / member).relative_to(ROOT).as_posix()
    new = prep_case(vehicle, dict(group=old["group"], slot=old["slot"], case=old["case"], src=src, pair=old["pair"],
                                  gate_failed=rep["failed"], continue_pair=old["pair_needed_continuation"]))
    new.update(restart_from="statistical follow-up continuation final state (member passed its checks)",
               continuation_member=member, continuation_stitched_sha256=rep["stitched_sha256"],
               superseded_precontinuation=old)
    new.pop("stability_gates_failed_on_existing_history")
    new["stability_gates_failed_after_continuation"] = rep["failed"]
    plan["cases"][i] = new
    (OUT / "_viz_plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    print("re-prepared", case_dir, "from", src, "end_cycle", new["end_cycle"])


def final_status(c):
    """Final recorded stability status of the authoritative case a visualization copy came from: the statistical
    follow-up member report when that case was continued, else the gates on its existing history (_pairs.json)."""
    fol = VDIR / VEH[c["vehicle"]] / "statistical_followup"
    pairs = json.loads((fol / "_pairs.json").read_text(encoding="utf-8"))["pairs"]
    res = fol / "_followup_result.json"
    reports = {r["member"]: r for r in json.loads(res.read_text(encoding="utf-8"))["member_reports"]}         if res.exists() else {}
    role = "source" if c["group"] == "A" else "variant"
    mine = [q for q in pairs if (q["name"] == c["case"].split("/")[-1] if role == "source" else q["case"] == c["pair"])]
    for q in mine:
        r = reports.get(q["case"].replace("/", "__") + "/" + role)
        if r is not None:
            return dict(converged=bool(r["ok"]), failed=r["failed"], member=r["member"],
                        basis="statistical follow-up continuation (final record)")
    failed = sorted({g for q in mine for g in q[role]["failed"]})
    return dict(converged=not failed, failed=failed, member=None, basis="existing history (not continued)")


def record_status():
    plan = json.loads((OUT / "_viz_plan.json").read_text(encoding="utf-8"))
    for c in plan["cases"]:
        c["final_status"] = final_status(c)
        print(c["dir"], c["final_status"])
    (OUT / "_viz_plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")


def write_runner(plan):
    a = " ".join(c["dir"] for c in plan["cases"] if c["group"] == "A")
    b = " ".join(c["dir"] for c in plan["cases"] if c["group"] == "B")
    fol = VEH["falcon"]
    script = f"""#!/usr/bin/env bash
# Visualization-only continuations (NON-AUTHORITATIVE). Usage: [VIZ_ONLY="dirs"] bash run_viz.sh A|B [parallel jobs, default 6]
# Copies the prepared cases to ~/visualization_cases and runs there (serial solver per case, no decomposition).
source /usr/lib/openfoam/openfoam2606/etc/bashrc || true     # before set -u: the OpenFOAM bashrc reads unset vars
set -u
GROUP="${{1:?usage: run_viz.sh A|B [jobs]}}"; N="${{2:-6}}"
HERE="$(cd "$(dirname "$0")" && pwd)"; WORK="$HOME/visualization_cases"
# never compete with an unfinished authoritative solver
for pid in $(pgrep -x pimpleFoam); do
  cwd="$(readlink -f /proc/$pid/cwd)"
  case "$cwd" in "$WORK"/*) ;; *) echo "authoritative solver still running in $cwd; not starting"; exit 1;; esac
done
if [ "$GROUP" = A ]; then LIST="{a}"
elif [ "$GROUP" = B ]; then
  LIST="{b}"
  # the Falcon U*-slot pair was continued by the statistical follow-up: wait for its final recorded status
  if [ ! -f "$HERE/../{fol}/statistical_followup/verdict/verdict.json" ]; then
    echo "Falcon statistical follow-up has no final record yet; running Atlas group B only"
    LIST="$(printf '%s\\n' $LIST | grep '^atlas/')"
  fi
else echo "group must be A or B"; exit 1; fi
# optional VIZ_ONLY="<dir> <dir>": restrict to these entries of the group list (the guards above still apply)
if [ -n "${{VIZ_ONLY:-}}" ]; then
  LIST="$(for c in $LIST; do case " $VIZ_ONLY " in *" $c "*) echo "$c";; esac; done)"
fi
[ -z "$LIST" ] && {{ echo "nothing to run"; exit 0; }}
mkdir -p "$WORK"
run() {{
  c="$1"; w="$WORK/$c"
  [ -f "$w/_viz_status.json" ] && grep -q done "$w/_viz_status.json" && {{ echo "skip $c"; exit 0; }}
  mkdir -p "$(dirname "$w")"; rm -rf "$w"; cp -a "$HERE/$c" "$w"
  ( cd "$w" && pimpleFoam > log.pimpleFoam 2>&1 && postProcess -func 'grad(U)' > log.post 2>&1 \\
      && postProcess -func vorticity >> log.post 2>&1 && postProcess -func Q >> log.post 2>&1 ); rc=$?
  s=failed; [ $rc -eq 0 ] && s=done
  printf '{{"state": "%s", "rc": %d, "host": "%s", "finished": "%s"}}\\n' "$s" $rc "$(hostname)" "$(date -Is)" > "$w/_viz_status.json"
  echo "$s $c"
}}
export -f run; export WORK HERE
printf '%s\\n' $LIST | xargs -P "$N" -I{{}} bash -c 'run "$@"' _ {{}}
echo "visualization group $GROUP finished"
"""
    (OUT / "run_viz.sh").write_text(script, newline="\n")


if __name__ == "__main__":
    from .. import cli
    if cli.show_help("figures", "cfd-cases"):
        raise SystemExit(0)
    if sys.argv[1:] == ["prepare"]:
        prepare()
    elif sys.argv[1:] == ["runner"]:          # rewrite run_viz.sh from the existing plan
        write_runner(json.loads((OUT / "_viz_plan.json").read_text(encoding="utf-8")))
    elif sys.argv[1:] == ["status"]:          # record each case's final stability status in the plan
        record_status()
    elif sys.argv[1:2] == ["reprepare"]:     # reprepare <vehicle> <case dir>
        reprepare_continued(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
