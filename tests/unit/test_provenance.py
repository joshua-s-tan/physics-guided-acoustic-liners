# -*- coding: utf-8 -*-
"""Provenance must be FAIL-CLOSED (remediation §6): 'complete' is computed, never trusted."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner.provenance import hashing as HASH  # noqa: E402
from acoustic_liner.provenance import records as PROV


def test_complete_demoted_when_provenance_missing(tmp_path):
    # capture() COMPUTES status, never trusts the claimed 'complete': a policy whose REQUIRED provenance is
    # absent (fem requires config + source + inputs) is demoted to incomplete -- robust to the repo now being a
    # clean, committed checkout with a final lock (where a generic capture would legitimately be complete).
    rec = PROV.capture(project_version="test", policy="fem", status="complete")
    assert rec["status"] == "incomplete"
    assert rec.get("incomplete_reasons")


def test_completeness_issues_new_api():
    # a fully-populated rec with a FINAL lock and clean git -> no issues under generic policy
    good = {"git_commit": "abc", "dirty_worktree": False, "dependency_lock_sha256": "l",
            "dependency_lock_status": "final", "configuration_sha256": "c",
            "input_artifact_sha256": {}, "source_file_sha256": {}}
    assert PROV._completeness_issues(good, PROV.POLICIES["generic"], n_source=1, n_input=1, n_seeds=1) == []


def test_missing_input_hash_blocks_complete(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    rec = PROV.capture(project_version="test", input_artifacts=[missing], status="complete")
    assert rec["status"] == "incomplete"
    assert "missing_input_artifact_hash" in rec["incomplete_reasons"]


def test_strict_mode_raises(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(PROV.ProvenanceError):
        PROV.capture(project_version="test", input_artifacts=[missing], status="complete", strict=True)


def test_policy_requires_config_source_inputs():
    rec = PROV.capture(project_version="test", policy="fem", status="complete")
    assert rec["status"] == "incomplete"
    r = rec["incomplete_reasons"]
    assert "no_configuration_hash" in r
    assert any(x.startswith("too_few_source_files") for x in r)
    assert any(x.startswith("too_few_input_artifacts") for x in r)


def test_dataset_policy_requires_seeds():
    rec = PROV.capture(project_version="test", policy="dataset", status="complete")
    assert "no_random_seeds" in rec["incomplete_reasons"]


def test_unpinned_requirements_is_not_a_lock():
    # a record with no dependency lock hash must flag it, regardless of requirements.txt existing
    fake = {"git_commit": "abc", "dirty_worktree": False, "dependency_lock_sha256": None,
            "configuration_sha256": "c", "input_artifact_sha256": {}, "source_file_sha256": {}}
    issues = PROV._completeness_issues(fake, PROV.POLICIES["generic"], n_source=1, n_input=1, n_seeds=1)
    assert "no_dependency_lock" in issues


def test_generated_lock_is_recognized():
    # requirements.lock was generated in this repo -> dependency_lock_sha256 resolves to it (not None)
    assert PROV.dependency_lock_sha256() is not None
    # and it is NOT the same file as requirements.txt
    assert PROV.dependency_lock_sha256() != PROV.requirements_txt_sha256()


def test_provisional_lock_blocks_complete():
    # BEHAVIOR (independent of the repo's own lock): a provisional-marked record must flag completeness ...
    rec = {"git_commit": "abc", "dirty_worktree": False, "dependency_lock_sha256": "l",
           "dependency_lock_status": "provisional", "configuration_sha256": "c",
           "input_artifact_sha256": {}, "source_file_sha256": {}}
    issues = PROV._completeness_issues(rec, PROV.POLICIES["generic"], n_source=1, n_input=1, n_seeds=1)
    assert any(x.startswith("provisional_dependency_lock") for x in issues)
    # ... and a FINAL-marked record must NOT be flagged for lock status.
    issues_final = PROV._completeness_issues(dict(rec, dependency_lock_status="final"),
                                          PROV.POLICIES["generic"], n_source=1, n_input=1, n_seeds=1)
    assert not any(x.startswith("provisional_dependency_lock") for x in issues_final)


def test_repo_lock_is_final():
    """The release lock has been finalized (genuine transitive lock via pip-compile)."""
    assert PROV.dependency_lock_status() == "final"


def test_generic_policy_forbidden_for_release():
    with pytest.raises(PROV.ReleasePolicyError):
        PROV.assert_release_policy("generic")
    PROV.assert_release_policy("fem")           # a real kind is allowed


def test_required_roles_not_satisfied_by_count(tmp_path):
    """A named-role policy is NOT satisfied by count alone: wrong/absent roles must flag (round 4.1)."""
    pol = PROV.POLICIES["fem"]      # requires input roles: physical_scenario, finalists
    rec = {"git_commit": "a", "dirty_worktree": False, "dependency_lock_sha256": "l",
           "dependency_lock_status": "final", "configuration_sha256": "c",
           "input_artifact_sha256": {"x": "h"}, "source_file_sha256": {"y": "h"},
           "input_roles": ["unrelated"], "source_roles": ["scenarios.py"]}
    issues = PROV._completeness_issues(rec, pol, n_source=1, n_input=1, n_seeds=1)
    assert "missing_input_role(physical_scenario)" in issues
    assert "missing_input_role(finalists)" in issues


def test_capture_accepts_role_mapping_and_records_roles(tmp_path):
    f = tmp_path / "a.json"; f.write_text("{}")
    rec = PROV.capture(project_version="t", policy="fem",
                    input_artifacts={"physical_scenario": f, "finalists": f}, config_path=str(f),
                    source_files={"imap": f})
    assert set(rec["input_roles"]) == {"physical_scenario", "finalists"}
    # the fem-required input roles are satisfied -> no missing_input_role reasons (other gaps may remain)
    assert not any(r.startswith("missing_input_role") for r in rec.get("incomplete_reasons", []))


def test_release_write_rejects_generic_policy(tmp_path):
    rel = tmp_path / "artifacts" / "release" / "x.json"
    prov = PROV.capture(project_version="t", policy="generic")
    with pytest.raises(PROV.ReleasePolicyError):
        PROV.write_artifact(rel, {"k": 1}, prov)
    assert not rel.exists()


def test_release_write_rejects_incomplete_provenance(tmp_path):
    rel = tmp_path / "artifacts" / "release" / "y.json"
    prov = PROV.capture(project_version="t", policy="fem", status="incomplete")   # release-allowed kind, not complete
    with pytest.raises(PROV.ReleasePolicyError):
        PROV.write_artifact(rel, {"k": 1}, prov)
    assert not rel.exists()


def test_release_gate_not_lexically_bypassable(tmp_path):
    """`artifacts/dev/../release/x.json` resolves INSIDE artifacts/release/ and must be gated (round 4.1-b)."""
    sneaky = tmp_path / "artifacts" / "dev" / ".." / "release" / "x.json"
    prov = PROV.capture(project_version="t", policy="generic")
    with pytest.raises(PROV.ReleasePolicyError):
        PROV.write_artifact(sneaky, {"k": 1}, prov)
    assert not (tmp_path / "artifacts" / "release" / "x.json").exists()


def test_release_write_rejects_forged_complete_status(tmp_path):
    """A hand-forged provenance dict must not pass: completeness is RECOMPUTED, not trusted (round 4.1-b)."""
    rel = tmp_path / "artifacts" / "release" / "forged.json"
    forged = {"artifact_kind": "fem", "status": "complete"}   # no commit/lock/hashes/roles
    with pytest.raises(PROV.ReleasePolicyError):
        PROV.write_artifact(rel, {"k": 1}, forged)
    assert not rel.exists()


def test_nonrelease_write_is_allowed(tmp_path):
    out = tmp_path / "artifacts" / "dev" / "z.json"
    prov = PROV.capture(project_version="t", policy="generic")
    PROV.write_artifact(out, {"k": 1}, prov)      # not under artifacts/release/ -> no release gate
    assert out.is_file()


def test_sha256_roundtrip(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert HASH.sha256_file(f) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert HASH.sha256_file(tmp_path / "missing") is None
