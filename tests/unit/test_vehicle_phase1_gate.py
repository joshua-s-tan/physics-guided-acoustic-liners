# -*- coding: utf-8 -*-
"""Phase 1 gate: vehicle search unlocks only on a complete, valid, unedited, sealed FEM-transfer audit of the
released checkpoint; and the audit's search cache is reused only under an equal key."""
import copy
import hashlib
import json

import pytest

from acoustic_liner.fem import transfer_audit as FTA
from acoustic_liner.evaluation import splits as PCN
from acoustic_liner.evaluation import conditions as COND
from acoustic_liner.models import checkpoint as CKPT
from acoustic_liner.provenance import hashing as H
from acoustic_liner.provenance import schemas as SC
from acoustic_liner.vehicles import search as VP


@pytest.fixture(scope="module")
def good():
    conds = COND.synthetic_conditions(FTA.N_CONDITIONS, PCN.named_seeds()["fem_audit"], FTA.SPLIT_NAME)
    rows = []
    for i, c in enumerate(conds):
        o, f = 0.5 + 0.01 * i, 0.4 + 0.012 * i
        rows.append(dict(cond_index=i, condition_hash=PCN.condition_hash(c), feasible=True, oracle=o, fem=f, diff=o - f,
                         corners=[dict(corner_id=[k, a], value=f) for k in ("kappa_lo", "kappa_hi") for a in ("low", "high")]))
    return dict(study="physgen_v1_fem_transfer_audit", split=FTA.SPLIT_NAME, split_hash=PCN.split_hash(conds),
                n_conditions=FTA.N_CONDITIONS, budget=FTA.BUDGET, seed=FTA.SEED, arm="D",
                base_seed=PCN.named_seeds()["evaluation"], checkpoint_model_sha256=CKPT.released_weights_sha256(),
                fem=dict(mesh_size=FTA.MESH_SIZE), rows=rows, summary=json.loads(json.dumps(FTA.summarize(rows))))


SEALS = {      # the three seal forms a reader accepts: written by the historical implementation, by the release
    # that recorded source scopes, and by this package
    "historical": lambda h: dict(record_sha256=h, source_sha256="a" * 64),
    "version 2 (source scopes)": lambda h: dict(record_sha256=h, provenance_schema_version=2,
                                                public_audit_scorer_source_sha256="b" * 64),
    "current": lambda h: dict(schema=SC.schema("fem-audit-seal"), record_sha256=h, package_source_sha256="c" * 64),
}


def _install(monkeypatch, tmp_path, rec, seal="current"):
    p = tmp_path / "fem_transfer_audit.json"
    p.write_text(json.dumps(rec))
    monkeypatch.setattr(FTA, "RECORD", p)
    s = tmp_path / "fem_transfer_audit.seal.json"
    if seal:
        h = hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        s.write_text(json.dumps(SEALS[seal](h)))
    elif s.exists():
        s.unlink()
    return p


@pytest.mark.parametrize("seal", list(SEALS))
def test_valid_sealed_audit_unlocks(monkeypatch, tmp_path, good, seal):
    _install(monkeypatch, tmp_path, good, seal)
    assert VP.require_phase1()["summary"]["n_evaluated"] == 20


def test_a_seal_of_an_unsupported_schema_is_refused(monkeypatch, tmp_path, good):
    p = _install(monkeypatch, tmp_path, good)
    s = p.with_name("fem_transfer_audit.seal.json")
    s.write_text(json.dumps(dict(json.loads(s.read_text()), schema="acoustic-liner/fem-audit-seal/9")))
    with pytest.raises(VP.VehiclePipelineError, match="not a supported version"):
        VP.require_phase1()


def _drop(r):
    r["rows"].pop()


def _infeasible(r):
    r["rows"][3].update(feasible=False)


def _nan(r):
    r["rows"][4]["fem"] = float("nan")


def _failure(r):
    r["summary"]["transfer_failure"] = True


def _edited(r):
    r["rows"][5]["oracle"] += 0.5
    r["rows"][5]["diff"] += 0.5


@pytest.mark.parametrize("mutate", [
    _drop, _infeasible, _nan, _failure, _edited,
    lambda r: r.update(split_hash="0" * 64),
    lambda r: r.update(checkpoint_model_sha256="0" * 64),
    lambda r: r["fem"].update(mesh_size=0.35),
    lambda r: r.update(budget=512),
    lambda r: r["rows"][0].update(condition_hash="0" * 64),
])
def test_failed_partial_stale_or_edited_audit_rejected(monkeypatch, tmp_path, good, mutate):
    rec = copy.deepcopy(good)
    mutate(rec)
    _install(monkeypatch, tmp_path, rec)
    with pytest.raises(VP.VehiclePipelineError):
        VP.require_phase1()


def test_unsealed_or_edited_after_seal_rejected_and_a_source_edit_does_not_matter(monkeypatch, tmp_path, good):
    p = _install(monkeypatch, tmp_path, good, seal=None)
    with pytest.raises(VP.VehiclePipelineError, match="not sealed"):
        VP.require_phase1()
    _install(monkeypatch, tmp_path, good)
    p.write_text(json.dumps(dict(good, n_conditions=20.0)))       # same values, different bytes
    with pytest.raises(VP.VehiclePipelineError, match="changed after sealing"):
        VP.require_phase1()
    _install(monkeypatch, tmp_path, good)
    monkeypatch.setattr(H, "package_source_sha256", lambda: "f" * 64)      # the code was edited since sealing
    VP.require_phase1()


def test_search_refuses_without_valid_audit(monkeypatch, tmp_path):
    monkeypatch.setattr(FTA, "RECORD", tmp_path / "missing.json")
    monkeypatch.setattr(VP, "RUN_DIR", tmp_path)
    with pytest.raises(VP.VehiclePipelineError, match="record missing"):
        VP.search("falcon", 1)


def test_the_search_cache_key_changes_with_every_input_it_depends_on(monkeypatch):
    """Cached search winners are reused only under an equal key: any change of the conditions, the checkpoint
    weights or configuration, the seeds or budget, the package source or the numerical environment gives a new key."""
    conds = FTA.audit_conditions()
    ck = dict(model_sha256="m" * 64, binding=dict(config_sha256="c" * 64))
    base = PCN.named_seeds()["evaluation"]
    key = FTA.cache_key(conds, ck, base)
    assert set(key) == {"split_hash", "model_sha256", "config_sha256", "base_seed", "budget", "seed", "arm",
                        "package_source_sha256", "environment"}
    assert key == FTA.cache_key(conds, ck, base)                                         # deterministic
    assert FTA.cache_key(conds[:-1], ck, base) != key
    assert FTA.cache_key(conds, dict(ck, model_sha256="n" * 64), base) != key
    assert FTA.cache_key(conds, dict(ck, binding=dict(config_sha256="d" * 64)), base) != key
    assert FTA.cache_key(conds, ck, base + 1) != key
    monkeypatch.setattr(FTA, "BUDGET", FTA.BUDGET * 2)
    assert FTA.cache_key(conds, ck, base) != key
    monkeypatch.undo()
    monkeypatch.setattr(H, "package_source_sha256", lambda: "f" * 64)
    assert FTA.cache_key(conds, ck, base) != key
    monkeypatch.undo()
    from acoustic_liner.evaluation import study as ST
    monkeypatch.setattr(ST, "_environment", lambda: dict(python="0", torch="0", numpy="0", torch_threads=1))
    assert FTA.cache_key(conds, ck, base) != key


def test_the_package_fingerprint_sees_edits_additions_and_removals_but_not_bytecode(tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "a.py").write_text("x = 1\n")
    (pkg / "sub" / "b.py").write_text("y = 2\n")
    monkeypatch.setattr(H, "PACKAGE", pkg)
    f0 = H.package_source_sha256()
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "a.cpython-311.pyc").write_bytes(b"\0")
    (pkg / "sub" / "b.pyc").write_bytes(b"\0")
    assert H.package_source_sha256() == f0                                               # generated files ignored
    (pkg / "a.py").write_bytes(b"x = 1\r\n")
    assert H.package_source_sha256() == f0                                               # line endings ignored
    (pkg / "a.py").write_text("x = 2\n")
    f1 = H.package_source_sha256()
    assert f1 != f0                                                                      # an edit
    (pkg / "c.py").write_text("")
    f2 = H.package_source_sha256()
    assert f2 != f1                                                                      # an added file
    (pkg / "c.py").unlink()
    assert H.package_source_sha256() == f1                                               # removed again
    (pkg / "sub" / "b.py").rename(pkg / "b.py")
    assert H.package_source_sha256() != f1                                               # a moved file
