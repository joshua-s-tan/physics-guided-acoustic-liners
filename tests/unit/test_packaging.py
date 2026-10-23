"""Packaging: every third-party module the package imports belongs to a distribution that `pyproject.toml` declares
as a direct dependency, and `requirements.lock` pins each declared distribution to one exact version.

Pillow is the case this guards. Three visualization modules import `PIL` directly, so Pillow is declared on its own;
the package does not rely on Matplotlib to install it.
"""
import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "acoustic_liner"
MODULES = sorted(p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py"))
VISUALIZATION = [m for m in MODULES if m.startswith("visualization/")]

# import name -> distribution name, for every third-party module the package may import
DISTRIBUTION = {"numpy": "numpy", "scipy": "scipy", "torch": "torch", "matplotlib": "matplotlib", "skfem": "scikit-fem",
                "gmsh": "gmsh", "meshio": "meshio", "cma": "cma", "sklearn": "scikit-learn", "yaml": "pyyaml",
                "PIL": "pillow"}
# the maintainer tool that rebuilds the frozen continuity fixture imports the historical v1 solver from a tree it is
# given; that module is not a distribution (tests/unit/test_self_contained.py)
NOT_A_DISTRIBUTION = {"provenance/continuity_fixture.py": {"fem_interior_nr"}}
PILLOW_TESTED_VERSION = "11.3.0"


def name(requirement):
    return re.split(r"[\s<>=!~;\[]", requirement.strip(), maxsplit=1)[0].lower().replace("_", "-")


def declared():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    return [name(r) for r in project["dependencies"]]


def third_party_imports(mod):
    """Top-level names of the absolute imports of one module that are neither the standard library nor the package."""
    out = set()
    for node in ast.walk(ast.parse((SRC / mod).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            out.add(node.module.split(".")[0])
    return out - set(sys.stdlib_module_names) - {"acoustic_liner"} - NOT_A_DISTRIBUTION.get(mod, set())


def lock_pins():
    """{distribution: [exact versions]} from the `name==version` lines of requirements.lock."""
    pins = {}
    for line in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([^\s;]+)", line.strip())
        if m:
            pins.setdefault(name(m.group(1)), []).append(m.group(2))
    return pins


def test_the_declared_dependencies_are_the_mapped_distributions():
    assert sorted(declared()) == sorted(DISTRIBUTION.values())
    assert len(set(declared())) == len(declared())


def test_pillow_is_a_direct_dependency():
    assert "pillow" in declared()


@pytest.mark.parametrize("mod", MODULES)
def test_every_third_party_import_is_a_declared_distribution(mod):
    for top in sorted(third_party_imports(mod)):
        assert top in DISTRIBUTION, f"{mod} imports {top!r}, which maps to no declared distribution"
        assert DISTRIBUTION[top] in declared(), f"{mod} imports {top!r}; {DISTRIBUTION[top]} is not declared"


def test_the_direct_imports_of_the_visualization_modules_are_declared():
    """The visualization modules import Pillow, Matplotlib and NumPy directly; each is declared on its own."""
    used = set().union(*(third_party_imports(m) for m in VISUALIZATION))
    assert "PIL" in used and "matplotlib" in used
    assert {DISTRIBUTION[u] for u in used} <= set(declared())
    pil = sorted(m for m in VISUALIZATION if "PIL" in third_party_imports(m))
    assert pil == ["visualization/cfd_fields.py", "visualization/cfd_qa.py", "visualization/fem_animation.py"]
    assert pil == sorted(m for m in MODULES if "PIL" in third_party_imports(m))


def test_the_lock_pins_every_declared_dependency_to_one_exact_version():
    pins = lock_pins()
    for dist in declared():
        assert len(pins.get(dist, [])) == 1, f"{dist}: {pins.get(dist)}"
        assert re.fullmatch(r"\d+(\.\d+)+([.+\-]?[A-Za-z0-9.+\-]+)?", pins[dist][0]), (dist, pins[dist])


def test_the_lock_pins_the_tested_pillow_version():
    assert lock_pins()["pillow"] == [PILLOW_TESTED_VERSION]


def test_the_direct_requirements_are_the_declared_dependencies():
    """`requirements.txt`, the input of the lock, lists exactly the distributions `pyproject.toml` declares."""
    lines = [x.strip() for x in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    direct = [name(x) for x in lines if x and not x.startswith("#")]
    assert sorted(direct) == sorted(declared()) and "pillow" in direct


def test_the_lock_lists_pillow_as_a_direct_requirement():
    text = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    block = text[text.index("\npillow=="):].split("\n", 2)[2]
    via = []
    for line in block.splitlines():
        if not line.startswith("    #"):
            break
        via.append(line.strip("# ").strip())
    assert "-r requirements.txt" in via and "matplotlib" in via
