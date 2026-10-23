"""Public documentation: nothing in the repository depends on a document that is neither distributed nor archived.

* every repository file the package names in a code string exists, or is one of the archived documents;
* comments and docstrings of the package and the dependency lock name no document path that is neither present nor
  archived;
* the maintained documents name no archived document, every relative link of theirs resolves, and every command
  they show is a command of the CLI;
* the README states the pre-registered verdict rule and the scope of the refactoring check.

The archived documents (the pre-registrations, protocols, seal, checkpoint lock and the long-form documentation)
are kept in the private archive and are not distributed. No command reads them. A few code strings still carry the
name of a protocol, because frozen records identify the protocol by that name.
"""
import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

from acoustic_liner import cli

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "acoustic_liner"
MODULES = sorted(SRC.rglob("*.py"))
MAINTAINED = ["README.md", "environment-openfoam.md"]
# kept in the private archive, not distributed
ARCHIVED = ["docs/PHYS-GEN-V1-PREREGISTRATION.md", "docs/VEHICLE-VERDICT-RULE-PREREG.md",
            "docs/VEHICLE-CFD-TARGETED-SENSITIVITY-PREREG.md", "docs/VEHICLE-STATISTICAL-FOLLOWUP-PROTOCOL.md",
            "docs/FALCON-SUPPLEMENTAL-STATISTICAL-IMPEDANCE-PROTOCOL.md", "docs/CFD-RESOLUTION-FOLLOWUP-PROTOCOL.md",
            "docs/FALCON-HIGH-SPEED-PAIR-PROTOCOL.md", "docs/FALCON-BOUNDED-SENSITIVITY-PROTOCOL.md",
            "docs/PHYS-GEN-V1-SEAL.json", "docs/PHYS-GEN-V1-CHECKPOINT-LOCK.json", "docs/POST-HOC-FOLLOWUPS-REPORT.md",
            "docs/VEHICLE-SCENARIO-PROVENANCE.md", "docs/LIMITATIONS.md", "docs/PROVENANCE.md", "docs/EQUIVALENCE.md",
            "docs/CODE-MAP.md"]
# a path under one of the repository's document or record folders
REPO_PATH = re.compile(r"(?<![\w/.\-])((?:docs|release|evidence|configs|finalists)/[\w.\-/]*\.[A-Za-z]{2,5})(?![\w/])")
DOC_PATH = re.compile(r"(?<![\w/.\-])(docs/[\w.\-/]+\.md)(?![\w/])")


def docstring_nodes(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


def code_strings(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    doc = docstring_nodes(tree)
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc]


def prose(path):
    """Comments and docstrings of one module, as text."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    out = [n.value for n in ast.walk(tree)
           if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) in docstring_nodes(tree)]
    out += [t.string for t in tokenize.generate_tokens(io.StringIO(src).readline) if t.type == tokenize.COMMENT]
    return "\n".join(out)


def known(path):
    """A path that is distributed, or one of the archived documents."""
    return (ROOT / path).exists() or path in ARCHIVED


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.relative_to(SRC).as_posix())
def test_every_repository_file_a_module_names_in_code_exists(path):
    named = {m for s in code_strings(path) for m in REPO_PATH.findall(s)}
    missing = sorted(m for m in named if not known(m))
    assert not missing, missing


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.relative_to(SRC).as_posix())
def test_comments_and_docstrings_name_no_missing_document(path):
    missing = sorted({m for m in DOC_PATH.findall(prose(path)) if not known(m)})
    assert not missing, missing


def test_the_dependency_files_name_no_missing_document():
    for name in ("requirements.lock", "requirements.txt", "requirements-dev.txt", "pyproject.toml"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert not [m for m in DOC_PATH.findall(text) if not known(m)], name
    # the lock's comment names an archived document; the lock is not edited, its SHA-256 is in the run records
    assert "docs/PROVENANCE.md" in (ROOT / "requirements.lock").read_text(encoding="utf-8")


@pytest.mark.parametrize("doc", MAINTAINED)
def test_every_relative_link_of_a_maintained_document_resolves(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    links = [m for m in re.findall(r"\]\(([^)\s]+)\)", text) if not re.match(r"[a-z]+://|#|mailto:", m)]
    missing = sorted({m for m in links if not ((ROOT / doc).parent / m.split("#")[0]).exists()})
    assert not missing, missing


@pytest.mark.parametrize("doc", MAINTAINED)
def test_a_maintained_document_names_no_missing_document_and_no_historical_command(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    assert not [m for m in DOC_PATH.findall(text) if not (ROOT / m).exists()]
    assert not [m for m in ARCHIVED if m in text or m.split("/")[-1] in text]        # no archived document named
    assert not re.search(r"python\s+tools/|python\s+-m\s+casef|src/casef", text)


@pytest.mark.parametrize("doc", MAINTAINED)
def test_every_command_a_document_shows_is_a_command_of_the_cli(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    shown = set(re.findall(r"acoustic-liner ([a-z][a-z\-]*) ([a-z][a-z\-]*)", text))
    assert all(g in cli.COMMANDS and t in cli.COMMANDS[g] for g, t in shown), sorted(shown)
    if doc == "README.md":
        assert {("verify", "public"), ("provenance", "checkpoint"), ("evaluation", "study")} <= shown


def test_the_documents_are_the_distributed_ones():
    for doc in MAINTAINED:
        assert (ROOT / doc).is_file(), doc
    assert not (ROOT / "docs" / "FULL-TRAINING-MANIFEST.md").exists()
    assert sorted(p.name for p in (ROOT / "docs").iterdir() if p.is_dir()) == ["figures"]


def test_the_readme_names_the_package_and_its_layout():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "src/acoustic_liner/" in readme and "casef" not in readme and "tools/" not in readme
    for folder in ("physics", "design", "models", "optimization", "evaluation", "fem", "cfd", "vehicles",
                   "visualization", "provenance"):
        assert (SRC / folder).is_dir() and f"  {folder}/" in readme, folder


# archived documents a code string may still name, and why: none of them is opened by any command
NAMED_BY_CODE = {
    "docs/FALCON-SUPPLEMENTAL-STATISTICAL-IMPEDANCE-PROTOCOL.md": "protocol name carried by frozen records",
    "docs/VEHICLE-STATISTICAL-FOLLOWUP-PROTOCOL.md": "protocol name carried by frozen records; a figure caption",
    "docs/VEHICLE-CFD-TARGETED-SENSITIVITY-PREREG.md": "a label of the CFD rerun record",
    "docs/VEHICLE-SCENARIO-PROVENANCE.md": "the file `provenance scenario-table` writes",
}


def test_code_names_no_seal_lock_or_pre_registration_and_only_the_listed_protocols():
    named = {m for path in MODULES for text in code_strings(path) for m in REPO_PATH.findall(text)}
    archived = named & set(ARCHIVED)
    assert archived <= set(NAMED_BY_CODE), sorted(archived - set(NAMED_BY_CODE))
    for path in MODULES:
        src = path.read_text(encoding="utf-8")
        for doc in ("PHYS-GEN-V1-SEAL", "PHYS-GEN-V1-CHECKPOINT-LOCK", "PHYS-GEN-V1-PREREGISTRATION",
                    "VEHICLE-VERDICT-RULE-PREREG"):
            assert doc not in src, (path.name, doc)
