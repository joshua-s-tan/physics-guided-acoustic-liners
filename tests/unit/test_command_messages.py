"""Messages that tell the user which command to run next name a command of `acoustic-liner`, with the arguments
that command takes. Two messages named a historical script until M15; the complete messages are tested here."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from acoustic_liner import cli
from acoustic_liner.vehicles import cfd_bundle, cfd_rerun, cfd_targeted

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "acoustic_liner"
RETURNED = "acoustic-liner vehicles cfd-bundle returned --bundle <bundle> --groups nominal"
AGGREGATE = "acoustic-liner vehicles cfd-bundle aggregate --bundle <bundle>"


def test_a_missing_returned_manifest_names_the_public_command(tmp_path):
    with pytest.raises(cfd_rerun.RerunError) as e:
        cfd_rerun._verify_returned(tmp_path, {})
    assert str(e.value) == f"{tmp_path / '_returned_manifest.json'} missing: run `{RETURNED}` first"


def test_a_missing_nominal_result_names_the_public_command(tmp_path, monkeypatch):
    monkeypatch.setattr(cfd_targeted.VC, "load_bundle", lambda bdir: (None, None, None, None))
    with pytest.raises(cfd_targeted.TargetedError) as e:
        cfd_targeted._ctx(tmp_path)
    assert str(e.value) == f"run `{AGGREGATE}` on the nominal results first"
    with pytest.raises(cfd_targeted.TargetedError) as e:
        cfd_targeted._ctx(tmp_path, tmp_path / "other_result.json")
    assert str(e.value) == f"run `{AGGREGATE}` on the nominal results first"


@pytest.mark.parametrize("command", [RETURNED, AGGREGATE])
def test_the_named_command_exists_and_takes_the_arguments_shown(command, monkeypatch):
    """The command line of the message, with a bundle folder in place of `<bundle>`, is accepted by the tool's own
    argument parser and reaches the named action."""
    program, group, tool, action, *arguments = command.split()
    assert program == "acoustic-liner" and cli.COMMANDS[group][tool][0] == "vehicles.cfd_bundle"
    seen = {}
    monkeypatch.setattr(cfd_bundle, "returned_manifest", lambda bundle, groups: seen.update(
        action="returned", bundle=bundle, groups=groups))
    monkeypatch.setattr(cfd_bundle, "aggregate", lambda bundle: seen.update(action="aggregate", bundle=bundle))
    cfd_bundle.main([action] + [a.replace("<bundle>", "B") for a in arguments])
    assert seen["action"] == action and seen["bundle"] == "B"
    if action == "returned":
        assert seen["groups"] == ["nominal"]


def test_the_command_help_lists_the_arguments_of_the_message():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([sys.executable, "-m", "acoustic_liner", "vehicles", "cfd-bundle", "returned", "--help"],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0 and "--bundle" in r.stdout and "--groups" in r.stdout


def test_no_message_of_the_package_names_a_historical_script():
    """No code string tells the user to run a script of the historical layout."""
    import ast
    import re
    old = re.compile(r"`(?:python\s+)?(?:tools/|vehicle_[a-z_]+\s|study_[a-z0-9_]+\s|physgen_[a-z_]+\s)")
    found = []
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and old.search(node.value):
                found.append((path.relative_to(SRC).as_posix(), node.lineno, node.value[:80]))
    assert found == []
