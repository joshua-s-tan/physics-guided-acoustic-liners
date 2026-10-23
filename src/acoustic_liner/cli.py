"""The one command of the package: `acoustic-liner <group> <tool> [arguments]`.

Each tool is a module of the package with its own arguments, unchanged; this file only names them and hands over.
`acoustic-liner` lists the groups, `acoustic-liner <group>` lists its tools, and
`acoustic-liner <group> <tool> --help` shows that tool's own help. A tool with an argument parser prints it
itself. For a tool that reads its own arguments the usage is in `TOOL_USAGE`, and `-h` or `--help` prints it and
returns before the tool is loaded: no input is read, no file or folder is written and nothing is computed.
"""
from __future__ import annotations

import runpy
import sys

# group -> tool -> (module under acoustic_liner, one-line description)
COMMANDS = {
    "design": {
        "schema": ("design.schema", "self-test of the design description"),
    },
    "evaluation": {
        "study": ("evaluation.study", "the LinerGen study: train, pilot, development, seal, verify"),
        "shards": ("evaluation.shards", "the sealed final evaluation in shards: manifest, run, merge"),
    },
    "fem": {
        "scenarios": ("fem.scenarios", "print the fairing scenarios"),
        "transfer-audit": ("fem.transfer_audit", "oracle-to-FEM transfer audit: run, summarize"),
    },
    "vehicles": {
        "search": ("vehicles.search", "vehicle search, finalist freeze, initial FEM, audit seal"),
        "cfd-bundle": ("vehicles.cfd_bundle", "CFD bundle: bundle, status, returned, aggregate, corrected-fem"),
        "cfd-rerun": ("vehicles.cfd_rerun", "operating-point reruns: select, merge"),
        "cfd-targeted": ("vehicles.cfd_targeted", "targeted sensitivity: select, amend, aggregate"),
        "cfd-supplemental": ("vehicles.cfd_supplemental", "block-estimator follow-up: estimate, select, aggregate, decide"),
        "statistical-followup": ("vehicles.statistical_followup", "statistical follow-up: table, pairs, continue, finish"),
        "verdict": ("vehicles.verdict", "final interval and verdict of one vehicle"),
    },
    "figures": {
        "physics": ("visualization.physics_figure", "the physics-model figures"),
        "study": ("visualization.study_figures", "the LinerGen study figures from the frozen records"),
        "vehicles": ("visualization.vehicle_figures", "the vehicle figures from the frozen records"),
        "fem": ("visualization.fem_animation", "global FEM acoustic-field still and animation"),
        "cfd-cases": ("visualization.cfd_cases", "visualization-only CFD continuations: prepare, runner, status"),
        "cfd-fields": ("visualization.cfd_fields", "CFD field figures: extract, render"),
        "cfd-histories": ("visualization.cfd_histories", "CFD history figures"),
        "cfd-qa": ("visualization.cfd_qa", "CFD figure checks and manifest: qa, publish"),
    },
    "provenance": {
        "scenario-table": ("provenance.scenario_table", "write the vehicle scenario provenance table"),
        "continuity-fixture": ("provenance.continuity_fixture", "freeze the v1 continuity fixture (maintainer)"),
        "checkpoint": ("provenance.checkpoint_manifest", "verify a downloaded LinerGen checkpoint"),
    },
    "verify": {
        "public": ("provenance.public_verifier", "verify the public evidence of a fresh clone (standard library and NumPy)"),
        "release": ("provenance.release_verifier", "verify release artifacts"),
    },
}


def _tool_usage(group, tool, arguments, text):
    head = f"usage: acoustic-liner {group} {tool} [-h]" + (f" {arguments}" if arguments else "")
    return f"{head}\n\n{text}\n  -h, --help        show this help and exit"


# (group, tool) -> usage, for every tool that reads its own arguments (no argument parser).
TOOL_USAGE = {(g, t): _tool_usage(g, t, a, x) for g, t, a, x in [
    ("design", "schema", "",
     "Self-test of the design description: print its dimensions and a round-trip error.\n\n"
     "  (no arguments)"),
    ("fem", "scenarios", "[scenario.yaml]",
     "Print a fairing scenario and run its unit self-test.\n\n"
     "  scenario.yaml     the scenario file (default: configs/falcon9_2025_blanketed.yaml)"),
    ("figures", "physics", "[output folder]",
     "Write the physics-model figures (one combined image and six single panels) and their manifest.\n\n"
     "  output folder     where the images are written (default: artifacts/dev/vehicle_v1/figures/fig1)"),
    ("figures", "study", "[output folder]",
     "Write the LinerGen study figures and their manifest from the frozen study records.\n\n"
     "  output folder     where the images are written (default: artifacts/dev/physgen_v1/figures)"),
    ("figures", "vehicles", "[output folder]",
     "Write the vehicle figures and their manifest from the frozen vehicle records.\n\n"
     "  output folder     where the images are written (default: artifacts/dev/vehicle_v1/figures/frozen)"),
    ("figures", "fem", "[falcon | atlas | publish]",
     "Write the global FEM acoustic-field still and animation of one vehicle.\n\n"
     "  falcon | atlas    the vehicle (default: falcon)\n"
     "  publish           copy the selected figures to docs/figures/fem/"),
    ("figures", "cfd-cases", "prepare | runner | status | reprepare <vehicle> <case folder>",
     "Visualization-only CFD continuations (never read by the impedance, FEM or verdict code).\n\n"
     "  prepare           build the visualization copies and run_viz.sh\n"
     "  runner            rewrite run_viz.sh from the existing plan\n"
     "  status            record each case's final stability status in the plan\n"
     "  reprepare         rebuild one case from the final state of its follow-up continuation"),
    ("figures", "cfd-fields", "extract [case folder ...] | render [falcon | atlas]",
     "Figures of the visualization-only CFD fields.\n\n"
     "  extract           read the cell data of the two-cycle window into the cache (all cases, or those named)\n"
     "  render            draw stills, montages, animations, geometry and histories (both vehicles, or one)"),
    ("figures", "cfd-histories", "",
     "Write the cycle-resolved impedance histories and pressure-flow figures of the visualization cases.\n\n"
     "  (no arguments)"),
    ("figures", "cfd-qa", "qa | publish",
     "Checks and manifest of the CFD figures.\n\n"
     "  qa                run every check; write visualization_manifest.json only if all pass\n"
     "  publish           copy the README selection to docs/figures/cfd/ (after qa)"),
    ("provenance", "scenario-table", "",
     "Write docs/VEHICLE-SCENARIO-PROVENANCE.md from the vehicle scenario files.\n\n"
     "  (no arguments)"),
    ("provenance", "continuity-fixture", "",
     "Maintainer tool: freeze the v1 continuity fixture (tests/fixtures/v1_continuity_reference.json) from the\n"
     "historical v1 solver. The environment variable ACOUSTIC_LINER_V1_REFERENCE_TREE must name the historical v1\n"
     "tree, which is not distributed. The committed fixture needs no regeneration to run the tests.\n\n"
     "  (no arguments)"),
    ("verify", "release", "artifact.json [artifact.json ...]",
     "Verify release artifacts. Fails closed: the provenance of every artifact is checked against the live\n"
     "repository, its scientific invariants are recomputed and the mandatory tests are run. A complete release\n"
     "names both a scenario-FEM and a generic-benchmark artifact.\n\n"
     "  artifact.json     one or more artifact files to verify"),
]}


def show_help(group, tool, argv=None):
    """Print the usage of a tool that reads its own arguments if `-h` or `--help` is among them; True if printed.
    The command calls this before it loads the tool, and the tool calls it first when it is run directly."""
    argv = sys.argv[1:] if argv is None else argv
    if not any(a in ("-h", "--help") for a in argv):
        return False
    print(TOOL_USAGE[(group, tool)])
    return True


def usage(group=None):
    if group is None:
        lines = ["usage: acoustic-liner <group> <tool> [arguments]", "", "groups:"]
        lines += [f"  {g:<12}{', '.join(COMMANDS[g])}" for g in COMMANDS]
    else:
        lines = [f"usage: acoustic-liner {group} <tool> [arguments]", "", "tools:"]
        lines += [f"  {t:<22}{d}" for t, (_, d) in COMMANDS[group].items()]
    return "\n".join(lines)


def program(group, tool):
    """The program name a tool shows in its own usage and records as its command."""
    return f"acoustic-liner {group} {tool}"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0 if argv else 2
    group = argv[0]
    if group not in COMMANDS:
        print(f"acoustic-liner: unknown group {group!r}\n\n{usage()}", file=sys.stderr)
        return 2
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(usage(group))
        return 0 if len(argv) > 1 else 2
    tool = argv[1]
    if tool not in COMMANDS[group]:
        print(f"acoustic-liner: unknown tool {tool!r} in group {group!r}\n\n{usage(group)}", file=sys.stderr)
        return 2
    if (group, tool) in TOOL_USAGE and show_help(group, tool, argv[2:]):
        return 0
    saved = sys.argv
    sys.argv = [program(group, tool)] + argv[2:]
    try:
        runpy.run_module(f"acoustic_liner.{COMMANDS[group][tool][0]}", run_name="__main__", alter_sys=True)
    finally:
        sys.argv = saved
    return 0
