"""The one `acoustic-liner` command: every listed tool is a package module with its own entry block, the command
hands over the arguments unchanged, and a tool shows and records the public program name."""
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner import cli  # noqa: E402

PKG = ROOT / "src" / "acoustic_liner"
TOOLS = [(g, t, m) for g, tools in cli.COMMANDS.items() for t, (m, _) in tools.items()]


def run(*args):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    return subprocess.run([sys.executable, "-m", "acoustic_liner", *args], capture_output=True, text=True, env=env,
                          cwd=ROOT)


def test_groups_follow_the_package():
    assert list(cli.COMMANDS) == ["design", "evaluation", "fem", "vehicles", "figures", "provenance", "verify"]
    assert len(TOOLS) == 25 and len({m for _, _, m in TOOLS}) == 25


@pytest.mark.parametrize("group,tool,module", TOOLS)
def test_every_tool_is_a_module_with_an_entry_block(group, tool, module):
    f = PKG / (module.replace(".", "/") + ".py")
    tree = ast.parse(f.read_text(encoding="utf-8"))
    assert any(isinstance(n, ast.If) and "__name__" in ast.unparse(n.test) and "__main__" in ast.unparse(n.test)
               for n in tree.body), f"{module} has no `if __name__ == \"__main__\":` block"


def test_usage_and_errors():
    r = run()
    assert r.returncode == 2 and "usage: acoustic-liner <group> <tool> [arguments]" in r.stdout
    r = run("--help")
    assert r.returncode == 0 and "vehicles" in r.stdout
    r = run("vehicles")
    assert r.returncode == 2 and "cfd-bundle" in r.stdout
    r = run("nonsense")
    assert r.returncode == 2 and "unknown group 'nonsense'" in r.stderr
    r = run("vehicles", "nonsense")
    assert r.returncode == 2 and "unknown tool 'nonsense'" in r.stderr


@pytest.mark.parametrize("group,tool", [("vehicles", "verdict"), ("vehicles", "cfd-bundle"), ("vehicles", "search"),
                                        ("evaluation", "study"), ("evaluation", "shards"), ("fem", "transfer-audit")])
def test_a_tool_shows_the_public_program_name(group, tool):
    r = run(group, tool, "--help")
    assert r.returncode == 0, r.stderr[-400:]
    assert f"usage: acoustic-liner {group} {tool}" in r.stdout and "casef" not in r.stdout


# The tools that read their own arguments (no argument parser): (group, tool) -> the arguments their usage shows.
SELF_PARSED = {
    ("design", "schema"): "",
    ("fem", "scenarios"): " [scenario.yaml]",
    ("figures", "physics"): " [output folder]",
    ("figures", "study"): " [output folder]",
    ("figures", "vehicles"): " [output folder]",
    ("figures", "fem"): " [falcon | atlas | publish]",
    ("figures", "cfd-cases"): " prepare | runner | status | reprepare <vehicle> <case folder>",
    ("figures", "cfd-fields"): " extract [case folder ...] | render [falcon | atlas]",
    ("figures", "cfd-histories"): "",
    ("figures", "cfd-qa"): " qa | publish",
    ("provenance", "scenario-table"): "",
    ("provenance", "continuity-fixture"): "",
    ("verify", "release"): " artifact.json [artifact.json ...]",
}
HELP_CASES = [(g, t, f) for (g, t) in SELF_PARSED for f in ("-h", "--help")]
# Files a tool could write when it runs, watched by the help tests.
WATCHED = [ROOT / "docs" / "VEHICLE-SCENARIO-PROVENANCE.md",
           ROOT / "tests" / "fixtures" / "v1_continuity_reference.json",
           ROOT / "docs" / "figures" / "cfd" / "README.md", ROOT / "docs" / "figures" / "fem" / "README.md"]


def state():
    """Names at the top of the repository, and size, time and content of the watched files."""
    top = sorted(p.name for p in ROOT.iterdir() if p.name not in (".pytest_cache", "__pycache__"))
    return top, [(str(p), p.stat().st_size, p.stat().st_mtime_ns, p.read_bytes()) if p.exists() else (str(p), None)
                 for p in WATCHED]


def test_every_tool_has_an_argument_parser_or_a_registered_usage():
    """The 13 tools without an argument parser are exactly the registered ones, and each calls the shared rule."""
    without = set()
    for group, tool, module in TOOLS:
        src = (PKG / (module.replace(".", "/") + ".py")).read_text(encoding="utf-8")
        if "argparse" not in src:
            without.add((group, tool))
            assert f'cli.show_help("{group}", "{tool}"' in src, module
    assert without == set(SELF_PARSED) == set(cli.TOOL_USAGE) and len(without) == 13


@pytest.mark.parametrize("group,tool,flag", HELP_CASES)
def test_help_prints_the_usage_and_changes_nothing(tmp_path, group, tool, flag):
    """The public command, run in an empty folder: the usage is printed, the status is 0, standard error is empty,
    and no file or folder is written: none in the working folder (a folder named after the flag was once created
    there), none at the top of the repository, and no watched file is rewritten."""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    env.pop("ACOUSTIC_LINER_V1_REFERENCE_TREE", None)
    before = state()
    r = subprocess.run([sys.executable, "-m", "acoustic_liner", group, tool, flag], capture_output=True, text=True,
                       env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stderr[-400:]
    assert r.stdout == cli.TOOL_USAGE[(group, tool)] + "\n" and r.stderr == ""
    assert r.stdout.splitlines()[0] == f"usage: acoustic-liner {group} {tool} [-h]{SELF_PARSED[(group, tool)]}"
    assert "-h, --help        show this help and exit" in r.stdout
    assert list(tmp_path.iterdir()) == [] and state() == before


@pytest.mark.parametrize("group,tool,flag", HELP_CASES)
def test_the_command_shows_help_before_it_loads_the_tool(monkeypatch, capsys, group, tool, flag):
    def refuse(*a, **k):
        raise AssertionError("help must not load or run the tool")
    monkeypatch.setattr(cli.runpy, "run_module", refuse)
    assert cli.main([group, tool, flag]) == 0 and cli.main([group, tool, "some-argument", flag]) == 0
    assert capsys.readouterr().out == (cli.TOOL_USAGE[(group, tool)] + "\n") * 2


def refuse_writes(monkeypatch):
    """Any attempt to create a folder or write a file through the usual calls fails the test."""
    import pathlib

    def refuse(*a, **k):
        raise AssertionError(f"help must not write: {a[:2]}")
    for name in ("mkdir", "write_text", "write_bytes", "touch", "unlink", "rename"):
        monkeypatch.setattr(pathlib.Path, name, refuse)
    for name in ("makedirs", "mkdir", "remove", "replace"):
        monkeypatch.setattr(os, name, refuse)


@pytest.mark.parametrize("group,tool,flag", HELP_CASES)
def test_a_tool_run_directly_shows_help_first(tmp_path, monkeypatch, capsys, group, tool, flag):
    """The tool's own entry code, run as the program without the command in front: the first thing it does is the
    shared help rule, so it returns before it reads an argument, an input or the environment."""
    import runpy
    import warnings
    module = f"acoustic_liner.{cli.COMMANDS[group][tool][0]}"
    __import__(module)                                  # imports happen here; the entry code is what is tested
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [cli.program(group, tool), flag])
    refuse_writes(monkeypatch)
    before = state()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)             # the module is already imported
        try:
            runpy.run_module(module, run_name="__main__")
        except SystemExit as e:
            assert e.code in (0, None)
    monkeypatch.undo()
    assert capsys.readouterr().out == cli.TOOL_USAGE[(group, tool)] + "\n"
    assert list(tmp_path.iterdir()) == [] and state() == before


# ---- the four tools that used to act on `--help` ---------------------------------------------------------------------
def test_the_vehicle_figure_tool_no_longer_creates_a_folder_named_after_the_flag(tmp_path, monkeypatch, capsys):
    """Before M15 the flag was read as the output folder: the folder was created and the figures were drawn."""
    from acoustic_liner.visualization import vehicle_figures as F
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["tool", "--help"])
    for name in ("fig_impedance", "pair_estimates", "fig_fem", "fig_ledger", "manifest"):
        monkeypatch.setattr(F, name, lambda *a, **k: pytest.fail("help must not draw"))
    assert F.main() is None
    assert not (tmp_path / "--help").exists() and list(tmp_path.iterdir()) == []
    assert capsys.readouterr().out.startswith("usage: acoustic-liner figures vehicles")


def test_the_history_figure_tool_no_longer_runs_on_help(monkeypatch, capsys):
    """Before M15 the tool ignored its arguments: it read the plan, drew every figure and rewrote the notes."""
    from acoustic_liner.visualization import cfd_histories as H
    monkeypatch.setattr(sys, "argv", ["tool", "-h"])
    monkeypatch.setattr(H.VF, "plan", lambda *a, **k: pytest.fail("help must not read the plan"))
    for name in ("cycle_histories", "pressure_flow"):
        monkeypatch.setattr(H, name, lambda *a, **k: pytest.fail("help must not draw"))
    assert H.main() is None
    assert capsys.readouterr().out.startswith("usage: acoustic-liner figures cfd-histories")


def test_the_scenario_table_tool_no_longer_rewrites_its_document_on_help(tmp_path):
    """Before M15 the tool ignored its arguments and rewrote docs/VEHICLE-SCENARIO-PROVENANCE.md (now generated on
    request only, and not distributed)."""
    doc = ROOT / "docs" / "VEHICLE-SCENARIO-PROVENANCE.md"
    look = lambda: (doc.stat().st_mtime_ns, doc.read_bytes()) if doc.exists() else None
    before = look()
    r = run("provenance", "scenario-table", "--help")
    assert r.returncode == 0 and "wrote docs/VEHICLE-SCENARIO-PROVENANCE.md" not in r.stdout
    assert look() == before


def test_the_fixture_tool_no_longer_runs_on_help(tmp_path):
    """Before M15 the tool ignored its arguments: it stopped for the missing variable, or with the variable set it
    loaded the historical solver and rewrote the fixture. With help it does neither, whatever the variable says."""
    fixture = ROOT / "tests" / "fixtures" / "v1_continuity_reference.json"
    before = (fixture.stat().st_mtime_ns, fixture.read_bytes())
    for value in (None, str(tmp_path), str(ROOT)):
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        env.pop("ACOUSTIC_LINER_V1_REFERENCE_TREE", None)
        if value:
            env["ACOUSTIC_LINER_V1_REFERENCE_TREE"] = value
        r = subprocess.run([sys.executable, "-m", "acoustic_liner", "provenance", "continuity-fixture", "--help"],
                           capture_output=True, text=True, env=env, cwd=tmp_path)
        assert r.returncode == 0 and r.stderr == "" and "ACOUSTIC_LINER_V1_REFERENCE_TREE" in r.stdout
        assert "unset/invalid" not in r.stdout + r.stderr and "wrote " not in r.stdout
    assert (fixture.stat().st_mtime_ns, fixture.read_bytes()) == before and list(tmp_path.iterdir()) == []


def test_the_fixture_tool_reads_only_the_new_variable(tmp_path, monkeypatch):
    """The variable was renamed in M15 and the old name is not an alias: with only the old name set, the tool
    stops as if no tree were named."""
    from acoustic_liner.provenance import continuity_fixture as CF
    src = (PKG / "provenance" / "continuity_fixture.py").read_text(encoding="utf-8")
    assert "CASEF" not in src and src.count('os.environ.get("ACOUSTIC_LINER_V1_REFERENCE_TREE", "")') == 1
    assert src.count("os.environ") == 1
    monkeypatch.delenv("ACOUSTIC_LINER_V1_REFERENCE_TREE", raising=False)
    monkeypatch.setenv("CASEF_V1_REFERENCE_TREE", str(tmp_path))
    with pytest.raises(SystemExit, match="ACOUSTIC_LINER_V1_REFERENCE_TREE unset/invalid"):
        CF.build_reference()


def test_the_release_verifier_shows_help_without_running_its_checks(monkeypatch, capsys):
    from acoustic_liner.provenance import release_verifier as V
    monkeypatch.setattr(V, "run_mandatory_tests", lambda *a, **k: pytest.fail("help must not run the tests"))
    assert V.main(["--help"]) == 0 and V.main(["artifact.json", "-h"]) == 0
    assert capsys.readouterr().out == (cli.TOOL_USAGE[("verify", "release")] + "\n") * 2


# ---- an argument that is not help is not mistaken for help ----------------------------------------------------------
@pytest.mark.parametrize("group,tool,arguments,status,text", [
    ("figures", "cfd-cases", ["nonsense"], 1, "acoustic-liner figures cfd-cases prepare"),
    ("figures", "cfd-cases", ["--helpful"], 1, "acoustic-liner figures cfd-cases prepare"),
    ("figures", "cfd-cases", [], 1, "acoustic-liner figures cfd-cases prepare"),
    ("figures", "cfd-fields", ["help"], 1, "acoustic-liner figures cfd-fields extract"),
    ("figures", "cfd-fields", ["-help"], 1, "acoustic-liner figures cfd-fields extract"),
    ("figures", "cfd-qa", ["--h"], 1, "acoustic-liner figures cfd-qa qa"),
    ("verify", "release", [], 1, "RELEASE VERIFICATION FAILED: no artifacts named"),
    ("fem", "scenarios", ["no-such-scenario.yaml"], 1, "no-such-scenario.yaml"),
])
def test_an_invalid_argument_still_fails(tmp_path, group, tool, arguments, status, text):
    """These calls failed before M15 and fail the same way now: the tool's own message on standard error, a
    non-zero status, and no usage of the help rule on standard output."""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([sys.executable, "-m", "acoustic_liner", group, tool, *arguments], capture_output=True,
                       text=True, env=env, cwd=tmp_path)
    assert r.returncode == status and text in r.stderr
    assert not r.stdout.startswith("usage: acoustic-liner") and list(tmp_path.iterdir()) == []


def test_a_tool_with_an_argument_parser_still_refuses_an_unknown_argument():
    r = run("vehicles", "cfd-bundle", "returned", "--bundle", "B", "--nonsense")
    assert r.returncode == 2 and "unrecognized arguments: --nonsense" in r.stderr


def test_arguments_are_handed_over_unchanged(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.runpy, "run_module", lambda name, **k: seen.update(name=name, argv=list(sys.argv), **k))
    before = list(sys.argv)
    assert cli.main(["vehicles", "cfd-rerun", "merge", "--bundle", "B", "--rerun", "R"]) == 0
    assert seen == dict(name="acoustic_liner.vehicles.cfd_rerun", run_name="__main__", alter_sys=True,
                        argv=["acoustic-liner vehicles cfd-rerun", "merge", "--bundle", "B", "--rerun", "R"])
    assert sys.argv == before


def test_a_small_tool_runs_end_to_end():
    r = run("fem", "scenarios")
    assert r.returncode == 0 and "scenario: falcon9_2025_blanketed" in r.stdout, r.stderr[-400:]


def test_the_console_script_is_declared():
    assert 'acoustic-liner = "acoustic_liner.cli:main"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
