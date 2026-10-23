"""SHA-256 of files: the content hash every provenance record and manifest is built from.
"""
from __future__ import annotations

import hashlib, platform, subprocess, sys, time, json, os
from pathlib import Path
from typing import Iterable, Optional


def sha256_file(path: str | Path) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_files(paths: Iterable[str | Path]) -> dict:
    return {str(p): sha256_file(p) for p in paths}


PACKAGE = Path(__file__).resolve().parents[1]
GENERATED = ("__pycache__",)
GENERATED_SUFFIXES = (".pyc", ".pyo")


def package_source_files() -> list:
    """Every file of the installed package as it is on disk (committed or not), relative to the package, without
    generated files (`__pycache__`, compiled bytecode)."""
    return sorted(p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob("*")
                  if p.is_file() and not set(p.relative_to(PACKAGE).parts) & set(GENERATED)
                  and p.suffix not in GENERATED_SUFFIXES)


def package_file_sha256(rel: str) -> str:
    """SHA-256 of one package file with LF line endings (a checkout that converts line endings hashes the same)."""
    return hashlib.sha256((PACKAGE / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def package_source_sha256() -> str:
    """Fingerprint of the package source: SHA-256 of canonical JSON {relative path: SHA-256 of the file with LF line
    endings}. Any edit, added or removed file changes it. It identifies cached results and shard inputs; it is not
    a release identity and nothing refuses to run because of it."""
    files = {f: package_file_sha256(f) for f in package_source_files()}
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
