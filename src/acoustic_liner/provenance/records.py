r"""Provenance capture for every processed artifact (remediation §2 rule 7, §3.2; review rounds 2-3).

Each artifact stores: schema/project version, git commit + dirty flag, command line, UTC timestamp,
random seeds, Python version, dependency-lock SHA-256, configuration SHA-256, input-artifact SHA-256
values, source-file SHA-256 values, and a status in {complete, incomplete, invalid}.

`status='complete'` is COMPUTED, never trusted from the caller, and it is computed against an ARTIFACT
POLICY that declares, per artifact kind, whether a configuration, source files, upstream inputs, and random
seeds are required. A real dependency LOCK (requirements.lock) is required -- an unpinned requirements.txt
does NOT satisfy it.
"""
from __future__ import annotations

import hashlib, platform, subprocess, sys, time, json, os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional
from .hashing import sha256_file


SCHEMA_VERSION = "provenance-1"


class ProvenanceError(RuntimeError):
    """Raised in strict mode when 'complete' is requested but required provenance is missing."""


@dataclass(frozen=True)
class ArtifactPolicy:
    """Per-kind completeness requirements (review round 3 finding 2; round 4 named roles).

    `required_source_roles` / `required_input_roles` name the roles a release artifact of this kind must
    provide (e.g. finalists need [evaluation, checkpoint, request_manifest, physical_scenario]). Role-level
    enforcement is wired as stages 3-4 land; the role lists are declared here now so the requirement is
    explicit and testable."""
    kind: str
    config_required: bool = False
    min_source_files: int = 0
    min_input_artifacts: int = 0
    seeds_required: bool = False
    required_source_roles: tuple = ()
    required_input_roles: tuple = ()
    release_allowed: bool = True          # `generic` is NOT allowed for artifacts/release/


# Registry of artifact kinds. `generic` requires nothing beyond the base (commit/clean/lock) and is NOT
# allowed for release artifacts. Named source/input roles are declared per kind (enforced as stages land).
POLICIES = {
    "generic": ArtifactPolicy("generic", release_allowed=False),
    "scenario": ArtifactPolicy("scenario", config_required=True, min_source_files=1,
                               required_source_roles=("scenarios.py",)),
    "dataset": ArtifactPolicy("dataset", config_required=True, min_source_files=1, seeds_required=True,
                              required_source_roles=("dataset.py", "oracle_v2.py", "mass_model.py", "spectra.py")),
    "evaluation": ArtifactPolicy("evaluation", config_required=True, min_source_files=1,
                                 min_input_artifacts=1, seeds_required=True,
                                 required_input_roles=("dataset", "checkpoint", "request_manifest")),
    "finalists": ArtifactPolicy("finalists", config_required=True, min_source_files=1, min_input_artifacts=1,
                                required_input_roles=("evaluation", "checkpoint", "request_manifest", "physical_scenario")),
    "fem": ArtifactPolicy("fem", config_required=True, min_source_files=1, min_input_artifacts=1,
                          required_input_roles=("physical_scenario", "finalists")),
    "cfd": ArtifactPolicy("cfd", config_required=True, min_source_files=1, min_input_artifacts=1,
                          required_input_roles=("physical_scenario", "finalists")),
    "sensitivity": ArtifactPolicy("sensitivity", config_required=True, min_source_files=1, min_input_artifacts=1,
                                  required_input_roles=("cfd",)),
    "relabel": ArtifactPolicy("relabel", config_required=True, min_source_files=1, min_input_artifacts=1,
                              required_input_roles=("physical_scenario", "finalists", "sensitivity")),
    "generic_benchmark": ArtifactPolicy("generic_benchmark", config_required=True, min_source_files=1,
                                        required_source_roles=("reference_cavity.py", "acoustics.py", "fem_solver.py")),
}


class ReleasePolicyError(RuntimeError):
    """Raised when a non-releasable artifact policy (e.g. 'generic') is used for a release artifact."""


def assert_release_policy(policy) -> None:
    """Reject a policy not allowed for anything placed under artifacts/release/ (review round 4)."""
    pol = _resolve_policy(policy)
    if not pol.release_allowed:
        raise ReleasePolicyError(f"artifact policy {pol.kind!r} is not permitted for release artifacts")


def _normalize_artifacts(items):
    """Accept EITHER a sequence of paths (no named roles) OR a {role: path} mapping (review round 4.1,
    finding 1). Returns (roles:set, {key -> sha256|None}, count). A policy that declares required roles is
    satisfied ONLY by the role-mapping form that provides those roles; a bare list provides NO roles and
    therefore fails such a policy (fail-closed -- arbitrary files can no longer satisfy a named-role policy).
    In role form the hash map is keyed BY ROLE so a missing/unresolved role hash is visible and blocking."""
    if items is None:
        return set(), {}, 0
    if isinstance(items, dict):
        roles = {str(r) for r in items.keys()}
        hashes = {str(r): sha256_file(p) for r, p in items.items()}
        return roles, hashes, len(items)
    lst = [str(p) for p in items]
    return set(), {p: sha256_file(p) for p in lst}, len(lst)


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parent,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def git_commit() -> Optional[str]:
    return _git("rev-parse", "HEAD")


def git_dirty() -> Optional[bool]:
    status = _git("status", "--porcelain")
    if status is None:
        return None
    return bool(status.strip())


def _lock_path() -> Path:
    from .. import paths
    return paths.ROOT / "requirements.lock"


def dependency_lock_sha256() -> Optional[str]:
    """Hash of the ACTUAL generated lock file ONLY. An unpinned requirements.txt / pyproject.toml is NOT a
    lock and must not satisfy the dependency-lock requirement (review round 3, finding 1)."""
    lock = _lock_path()
    return sha256_file(lock) if lock.is_file() else None


def dependency_lock_status() -> Optional[str]:
    """Read the machine-readable `# lock-status:` marker from requirements.lock: 'final' | 'provisional'.
    A provisional lock (direct deps only, no full transitive closure) does NOT satisfy release completeness
    (review round 4, finding 3). Returns None if no lock file."""
    lock = _lock_path()
    if not lock.is_file():
        return None
    for line in lock.read_text().splitlines():
        s = line.strip().lower()
        if s.startswith("# lock-status:"):
            return s.split(":", 1)[1].strip()
    return "unmarked"          # a lock with no status marker is treated as not-final


def requirements_txt_sha256() -> Optional[str]:
    """Convenience hash of the (unpinned) requirements.txt -- informational only, NOT a lock."""
    from .. import paths
    root = paths.ROOT
    return sha256_file(root / "requirements.txt")


def _resolve_policy(policy) -> ArtifactPolicy:
    if isinstance(policy, ArtifactPolicy):
        return policy
    if policy in POLICIES:
        return POLICIES[policy]
    raise ValueError(f"unknown artifact policy {policy!r}; known: {sorted(POLICIES)}")


def _completeness_issues(rec: dict, pol: ArtifactPolicy, n_source: int, n_input: int, n_seeds: int) -> list:
    """Reasons a record may NOT be 'complete'. Computed from base provenance + the artifact policy."""
    issues = []
    if not rec.get("git_commit"):
        issues.append("no_git_commit")
    if rec.get("dirty_worktree") is None:
        issues.append("unknown_worktree_state")
    elif rec.get("dirty_worktree"):
        issues.append("dirty_worktree")
    if not rec.get("dependency_lock_sha256"):
        issues.append("no_dependency_lock")
    elif rec.get("dependency_lock_status") != "final":
        issues.append(f"provisional_dependency_lock({rec.get('dependency_lock_status')})")
    if pol.config_required and not rec.get("configuration_sha256"):
        issues.append("no_configuration_hash")
    if any(v is None for v in rec.get("input_artifact_sha256", {}).values()):
        issues.append("missing_input_artifact_hash")
    if any(v is None for v in rec.get("source_file_sha256", {}).values()):
        issues.append("missing_source_file_hash")
    if n_source < pol.min_source_files:
        issues.append(f"too_few_source_files(<{pol.min_source_files})")
    if n_input < pol.min_input_artifacts:
        issues.append(f"too_few_input_artifacts(<{pol.min_input_artifacts})")
    if pol.seeds_required and n_seeds == 0:
        issues.append("no_random_seeds")
    # Named-role enforcement (review round 4.1, finding 1): counts alone do NOT satisfy a policy that
    # declares required roles -- each required role must have been PROVIDED (via the {role: path} form).
    provided_source_roles = set(rec.get("source_roles", []))
    provided_input_roles = set(rec.get("input_roles", []))
    for role in pol.required_source_roles:
        if role not in provided_source_roles:
            issues.append(f"missing_source_role({role})")
    for role in pol.required_input_roles:
        if role not in provided_input_roles:
            issues.append(f"missing_input_role({role})")
    return issues


def capture(*, project_version: str, policy="generic", config_path: Optional[str] = None,
            input_artifacts: Iterable[str | Path] = (), source_files: Iterable[str | Path] = (),
            seeds: Optional[dict] = None, status: str = "incomplete", strict: bool = False,
            extra: Optional[dict] = None) -> dict:
    """Build a provenance record. `status='complete'` is COMPUTED against the artifact `policy`: it holds
    only when git commit + clean worktree + real dependency lock are present AND the policy's config /
    source-file / input-artifact / seed requirements are met AND every provided hash resolved. Otherwise
    the record is demoted to 'incomplete' with `incomplete_reasons` (or, with `strict=True`, raises
    ProvenanceError). 'invalid' is always honored."""
    if status not in ("complete", "incomplete", "invalid"):
        raise ValueError(f"status must be complete|incomplete|invalid; got {status!r}")
    pol = _resolve_policy(policy)
    src_roles, src_hashes, n_source = _normalize_artifacts(source_files)
    in_roles, in_hashes, n_input = _normalize_artifacts(input_artifacts)
    # role -> repo-relative path manifest, so every recorded hash is later re-verifiable against a real file
    from .. import paths
    root = paths.ROOT

    def _rel(p):
        p = Path(p)
        try:
            return (p if not p.is_absolute() else p).resolve().relative_to(root).as_posix()
        except ValueError:
            return str(p)                      # left as-is; verification will reject a non-relative/escaping path

    role_paths = {}
    if isinstance(source_files, dict):
        role_paths.update({str(r): _rel(p) for r, p in source_files.items()})
    if isinstance(input_artifacts, dict):
        role_paths.update({str(r): _rel(p) for r, p in input_artifacts.items()})
    configuration_path = _rel(config_path) if config_path else None
    seeds = seeds or {}
    rec = dict(
        schema_version=SCHEMA_VERSION,
        project_version=project_version,
        artifact_kind=pol.kind,
        git_commit=git_commit(),
        dirty_worktree=git_dirty(),
        command_line=" ".join(sys.argv),
        utc_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        random_seeds=seeds,
        python_version=platform.python_version(),
        platform=platform.platform(),
        dependency_lock_sha256=dependency_lock_sha256(),
        dependency_lock_status=dependency_lock_status(),
        requirements_txt_sha256=requirements_txt_sha256(),
        configuration_sha256=(sha256_file(config_path) if config_path else None),
        configuration_path=configuration_path,
        input_artifact_sha256=in_hashes,
        source_file_sha256=src_hashes,
        input_roles=sorted(in_roles),
        source_roles=sorted(src_roles),
        role_paths=role_paths,
    )
    issues = _completeness_issues(rec, pol, n_source, n_input, len(seeds))
    if status == "complete" and issues:
        if strict:
            raise ProvenanceError(f"cannot mark '{pol.kind}' complete; missing provenance: {issues}")
        status = "incomplete"
        rec["incomplete_reasons"] = issues
    rec["status"] = status
    if extra:
        rec["extra"] = extra
    return rec


def _is_release_artifact_path(path: str | Path) -> bool:
    """True if `path` lands under an `artifacts/release/` directory (case-insensitive). The destination is
    CANONICALIZED first (`resolve(strict=False)`) so a lexical bypass like `artifacts/dev/../release/x.json`
    -- which resolves INSIDE the release directory -- is still caught (review round 4.1-b, finding 1)."""
    parts = [x.lower() for x in Path(path).resolve(strict=False).parts]
    return any(parts[i] == "artifacts" and parts[i + 1] == "release" for i in range(len(parts) - 1))


def write_artifact(path: str | Path, payload: dict, provenance: dict) -> None:
    """Persist an artifact atomically. Any write under `artifacts/release/` is GATED (review round 4.1):
    the artifact's policy kind must be release-allowed (never `generic`) AND its provenance must ACTUALLY be
    complete. The stored `status` string is NOT trusted -- completeness is RECOMPUTED from the provenance
    record via `_completeness_issues` (review round 4.1-b, finding 2), so a fabricated
    `{"artifact_kind": "fem", "status": "complete"}` with no commit/lock/hashes/roles is rejected.
    Development / incomplete outputs must be written elsewhere."""
    p = Path(path)
    if _is_release_artifact_path(p):
        kind = provenance.get("artifact_kind", "generic")
        pol = _resolve_policy(kind)
        assert_release_policy(pol)           # raises ReleasePolicyError for 'generic'/non-release kinds
        # Do NOT trust provenance["status"]. Recompute from the record, deriving counts from its recorded
        # mappings, so a hand-forged 'complete' cannot bypass commit/lock/hash/role/cleanliness checks.
        n_source = len(provenance.get("source_file_sha256", {}) or {})
        n_input = len(provenance.get("input_artifact_sha256", {}) or {})
        n_seeds = len(provenance.get("random_seeds", {}) or {})
        issues = _completeness_issues(provenance, pol, n_source, n_input, n_seeds)
        if issues or provenance.get("status") != "complete":
            raise ReleasePolicyError(
                f"release artifact {p} requires genuinely-complete provenance; recomputed issues={issues}, "
                f"stored status={provenance.get('status')!r}")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump({"provenance": provenance, "data": payload}, f, indent=1, default=float, allow_nan=False)
    os.replace(tmp, p)
