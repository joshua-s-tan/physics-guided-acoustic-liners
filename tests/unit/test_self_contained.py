"""The package is self-contained: every module imports only the standard library, the declared dependencies and the
package itself; no module changes `sys.path`; no module reads the variable that points at the historical v1 tree.

One maintainer tool is the exception to the last two rules, by design: `provenance/continuity_fixture.py` rebuilds
the frozen v1 continuity fixture from the historical v1 tree, which it is given through that variable. Nothing
imports it, and no test or workflow needs the tree.
"""
import ast
import importlib
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "acoustic_liner"
MODULES = sorted(p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py"))

# import names of the dependencies declared in pyproject.toml
DEPENDENCIES = {"numpy", "scipy", "torch", "matplotlib", "skfem", "gmsh", "meshio", "cma", "sklearn", "yaml", "PIL"}
ALLOWED_ABS = set(sys.stdlib_module_names) | DEPENDENCIES | {"acoustic_liner"}
REFERENCE_TREE_TOOL = "provenance/continuity_fixture.py"
REFERENCE_TREE_MODULES = {"fem_interior_nr"}            # the v1 solver, imported by the maintainer tool only


def _uses_sys_path(tree):
    """True iff EXECUTABLE code accesses `sys.path` (an Attribute `.path` on the name `sys`). Ignores docstrings
    /comments that merely mention the string 'sys.path'."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "path" \
                and isinstance(node.value, ast.Name) and node.value.id == "sys":
            return True
    return False


def _abs_import_top_levels(src):
    """Top-level names of every ABSOLUTE import (Import, or ImportFrom with level==0). Relative imports
    (from . import x) have level>0 and are intra-package -> always allowed, so excluded here."""
    tree = ast.parse(src)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue                                   # relative intra-package import
            if node.module:
                names.append(node.module.split(".")[0])
    return names


def test_the_dependency_names_are_the_declared_ones():
    text = (SRC.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    for dist in ("numpy", "scipy", "torch", "matplotlib", "scikit-fem", "gmsh", "meshio", "cma", "scikit-learn",
                 "pyyaml", "pillow"):
        assert f'"{dist}"' in text, dist


@pytest.mark.parametrize("mod", MODULES)
def test_module_has_no_reference_tree_import(mod):
    src = (SRC / mod).read_text(encoding="utf-8")
    tree = ast.parse(src)
    if mod != REFERENCE_TREE_TOOL:
        # no sys.path manipulation (AST, not substring: docstrings may mention it), no reference-tree variable
        assert not _uses_sys_path(tree), f"{mod} accesses sys.path in executable code"
        if mod == "cli.py":
            # the usage text of the fixture tool names the variable; the command itself never reads the environment
            assert "os.environ" not in src and "getenv" not in src and "import os" not in src, \
                f"{mod} reads the environment"
        else:
            assert "V1_REFERENCE_TREE" not in src, f"{mod} reads the reference-tree env var"
    # every absolute import must be the standard library, a declared dependency or the package itself
    allowed = ALLOWED_ABS | (REFERENCE_TREE_MODULES if mod == REFERENCE_TREE_TOOL else set())
    for top in _abs_import_top_levels(src):
        assert top in allowed, f"{mod} has a non-self-contained absolute import {top!r}"


def test_only_figure_modules_import_pillow():
    users = sorted(m for m in MODULES if "PIL" in _abs_import_top_levels((SRC / m).read_text(encoding="utf-8")))
    assert users == ["visualization/cfd_fields.py", "visualization/cfd_qa.py", "visualization/fem_animation.py"]


def test_the_reference_tree_tool_is_imported_by_nothing():
    name = REFERENCE_TREE_TOOL[:-3].split("/")[-1]
    for mod in MODULES:
        if mod != REFERENCE_TREE_TOOL:
            src = (SRC / mod).read_text(encoding="utf-8")
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert name not in [a.name for a in node.names] and (node.module or "").split(".")[-1] != name, mod
                elif isinstance(node, ast.Import):
                    assert all(a.name.split(".")[-1] != name for a in node.names), mod


def test_modules_import_cleanly():
    """Each module imports as part of the package (no run-time dependency on a tree outside it)."""
    for mod in MODULES:
        if mod.endswith("__main__.py"):
            continue
        dotted = "acoustic_liner." + mod[:-3].replace("/", ".")
        importlib.import_module(dotted[:-len(".__init__")] if dotted.endswith(".__init__") else dotted)
