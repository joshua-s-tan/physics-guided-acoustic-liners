"""The layout of the acoustic_liner package, and the check that it stands alone: no released file imports, loads
or names the historical package.

During the migration a migrated module could import a module that was not migrated yet, through a marked bridge.
The last bridge was removed in M11 and the bridge checks were removed in M15; this test stays.
"""
import importlib
import re
from pathlib import Path

import acoustic_liner

PKG = Path(acoustic_liner.__file__).resolve().parent
SUBPACKAGES = ("physics", "design", "models", "optimization", "evaluation", "vehicles", "fem", "cfd",
               "visualization", "provenance")


def test_subpackages_import():
    for name in SUBPACKAGES:
        importlib.import_module(f"acoustic_liner.{name}")
    importlib.import_module("acoustic_liner.paths")


def test_repo_root_found():
    from acoustic_liner import paths
    assert (paths.ROOT / "pyproject.toml").exists()


def bridges():
    out = []
    for f in sorted(PKG.rglob("*.py")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"^\s*(from\s+casef_v2\b|import\s+casef_v2\b)", line):
                out.append((f.relative_to(PKG).as_posix(), i, line.strip()))
    return out


# Historical identifiers a released file may not contain in code: the historical package name, every historical
# schema identifier (`casef_<name>/<n>`, and the pre-release checkpoint-manifest identifier) and the historical
# generator class name. They live only in the reader mapping of provenance/schemas.py (the earlier identifiers a
# reader accepts) and in archived evidence.
HISTORICAL_IDENTIFIERS = ("casef_v2", "casef_", "linergen-checkpoint-manifest/", "ConditionalGenerator")
READER_MAPPING = "provenance/schemas.py"


def casef_v2_references():
    """Every place a released acoustic_liner file imports or dynamically loads casef_v2: import statements,
    importlib / __import__ / runpy / pkgutil calls, and any code string naming casef_v2 or another historical
    identifier (docstrings and comments are text, not loads, and are not counted)."""
    import ast
    out = []
    for f in sorted(PKG.rglob("*.py")):
        if f.relative_to(PKG).as_posix() == READER_MAPPING:
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        docs = {id(n.body[0].value) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0].value, "value", None), str)}
        for n in ast.walk(tree):
            hit = None
            if isinstance(n, ast.Import) and any(a.name.split(".")[0] == "casef_v2" for a in n.names):
                hit = "import"
            elif isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "casef_v2":
                hit = "from-import"
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs \
                    and any(k in n.value for k in HISTORICAL_IDENTIFIERS):
                hit = "string"
            if hit:
                out.append((f.relative_to(PKG).as_posix(), n.lineno, hit))
    return out


def test_no_released_file_imports_or_loads_casef_v2():
    """Passes normally, with no expected failure and no exception list: the package has no import, no dynamic
    load and no code string that names the historical package or another historical identifier."""
    assert casef_v2_references() == []
    assert bridges() == []


def test_historical_identifiers_appear_in_the_reader_mapping_only_as_accepted_earlier_names():
    from acoustic_liner.provenance import schemas as SC
    assert not any("casef" in v for v in SC.CURRENT.values())
    old = [v for vs in SC.EARLIER.values() for v in vs if "casef" in v]
    assert old and all(v.startswith("casef_") and "/" in v for v in old)
