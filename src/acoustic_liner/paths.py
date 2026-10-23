"""Repository-relative path resolution ONLY. No absolute developer paths,
no dependence on the current working directory. The repo root is found by walking up from this file to
the directory that contains the release markers (pyproject.toml + src).
"""
from __future__ import annotations

from pathlib import Path


_MARKERS = ("pyproject.toml", "src")


def repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if all((parent / m).exists() for m in _MARKERS):
            return parent
    raise RuntimeError("could not locate repo root (missing pyproject.toml + src/)")
ROOT = repo_root()
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS_RELEASE = ARTIFACTS / "release"
ARTIFACTS_HISTORICAL = ARTIFACTS / "historical"
DOCS = ROOT / "docs"
OPENFOAM = ROOT / "openfoam"


def under(*parts: str) -> Path:
    """Resolve a path relative to the repo root (creates nothing)."""
    return ROOT.joinpath(*parts)
