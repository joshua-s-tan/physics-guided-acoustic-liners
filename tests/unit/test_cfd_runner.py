# -*- coding: utf-8 -*-
"""Unit tests for the batch CFD runner (acoustic_liner.cfd.runner): it is pure orchestration over the already-tested
stages, so these exercise the state machine -- prepare->manifest, status classification (valid/needs_extend/
failed/missing), the WSL resume manifest, and fail-closed four-bank aggregation -- with the heavy physics
(post_case, build_per_bank_rx) monkeypatched. OpenFOAM is never invoked."""
import json
import os

import pytest

from acoustic_liner.cfd import runner as RUN
from acoustic_liner.cfd import postprocess as POST
from acoustic_liner.cfd import impedance_correction as RL

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(RUN.__file__)))))
_FINALIST = os.path.join(_ROOT, "finalists", "falcon_v2.1_finalist_geometry.json")


def _blanketed():
    from acoustic_liner.fem import scenarios as S
    return list(S.falcon9_2025_blanketed().spl_dB)


def _row(name="L2_b0", case_hash="ch", geom="g", **over):
    r = dict(bank_id=name, case_hash=case_hash, geometry_hash=geom, u_target=6.0, u_amp=6.2, mesh_ok=True,
             R_phasor=2.0, X_phasor=21.0, R_eff=2.05, X_accel_xcheck=20.7, R_drift=0.01, X_drift=0.02,
             thd_p=0.05, thd_u=0.04, probe_clean=True)
    r.update(over)
    return r


# ---- prepare delegates to gen_cases and writes the authoritative plan ------------------------------------


def test_load_manifest_missing_raises(tmp_path):
    with pytest.raises(RUN.RunnerError):
        RUN.load_manifest(str(tmp_path))


# ---- classification --------------------------------------------------------------------------------------
def _validate(row, entry, geom="g"):
    from acoustic_liner.cfd import validation as VAL
    return VAL.validate_cfd_row(row, expected_bank_id=entry["bank_id"], expected_case_hash=entry["case_hash"],
                                expected_geometry_hash=geom)


def test_classify_valid():
    e = dict(bank_id="L2_b0", case_hash="ch")
    row = _row()
    ok, rep = _validate(row, e)
    assert RUN._classify(row, ok, rep) == RUN.VALID


def test_classify_needs_extend_on_convergence_only():
    e = dict(bank_id="L2_b0", case_hash="ch")
    row = _row(R_drift=0.2)                       # only R convergence fails; probe + mesh clean
    ok, rep = _validate(row, e)
    assert not ok and RUN._classify(row, ok, rep) == RUN.NEEDS_EXTEND


def test_classify_failed_when_mesh_bad():
    e = dict(bank_id="L2_b0", case_hash="ch")
    row = _row(R_drift=0.2, mesh_ok=False)        # mesh failure is not fixable by more cycles
    ok, rep = _validate(row, e)
    assert RUN._classify(row, ok, rep) == RUN.FAILED


def test_classify_failed_when_probe_dirty():
    e = dict(bank_id="L2_b0", case_hash="ch")
    row = _row(R_drift=0.2, probe_clean=False)
    ok, rep = _validate(row, e)
    assert RUN._classify(row, ok, rep) == RUN.FAILED


# ---- scan_status + resume manifest -----------------------------------------------------------------------
def test_scan_status_and_resume(tmp_path, monkeypatch):
    names = ["v_case", "ext_case", "fail_case", "miss_case"]
    for n in names:
        os.makedirs(tmp_path / n, exist_ok=True)
    manifest = dict(geometry_hash="g", cases=[
        dict(name="v_case", bank_id="L2_b0", case_hash="h0", u_target=6.0),
        dict(name="ext_case", bank_id="L2_b1", case_hash="h1", u_target=6.0),
        dict(name="fail_case", bank_id="L2_b2", case_hash="h2", u_target=6.0),
        dict(name="miss_case", bank_id="L2_b3", case_hash="h3", u_target=0.5),
    ])
    rows = {
        "v_case": _row(name="L2_b0", case_hash="h0"),
        "ext_case": _row(name="L2_b1", case_hash="h1", R_drift=0.3),
        "fail_case": _row(name="L2_b2", case_hash="h2", mesh_ok=False),
    }
    monkeypatch.setattr(RUN, "_case_ran", lambda cd: os.path.basename(cd) != "miss_case")
    monkeypatch.setattr(POST, "post_case", lambda cd, n_periods=3: rows[os.path.basename(cd)])

    st = RUN.scan_status(manifest, str(tmp_path))
    by = {c["name"]: c["state"] for c in st["cases"]}
    assert by == {"v_case": RUN.VALID, "ext_case": RUN.NEEDS_EXTEND,
                  "fail_case": RUN.FAILED, "miss_case": RUN.MISSING}
    assert st["n_banks"] == 4 and st["all_valid"] is False

    rm = RUN.resume_manifest(st, str(tmp_path), extend_cycles=40)
    acts = {a["name"]: a["action"] for a in rm["actions"]}
    assert acts == {"ext_case": "extend", "miss_case": "run"}   # valid + hard-failed are not re-queued
    assert rm["done"] is False                                   # done tracks all_valid, not "no actions"
    assert rm["blocked"] == ["fail_case"]                        # hard-failed surfaced as blocked evidence
    assert any("pimpleFoam" in s for s in rm["shell"])           # only executable run commands in shell
    assert rm["manual"] and all("pimpleFoam" not in m for m in rm["manual"])  # extends are non-executable notes


def test_resume_only_failed_is_not_done(tmp_path):
    st = dict(cases=[dict(name="x", bank_id="L2_b0", state=RUN.FAILED)], summary={}, n_banks=1, all_valid=False)
    rm = RUN.resume_manifest(st, str(tmp_path))
    assert rm["done"] is False and rm["shell"] == [] and rm["blocked"] == ["x"]


def test_resume_done_when_all_valid(tmp_path):
    st = dict(cases=[dict(name="a", bank_id="b0", state=RUN.VALID)], summary={}, n_banks=1, all_valid=True)
    rm = RUN.resume_manifest(st, str(tmp_path))
    assert rm["done"] is True and rm["actions"] == []


# ---- aggregate derives bank set from the manifest and is fail-closed --------------------------------------
def _manifest_4():
    return dict(geometry_hash="g",
                cases=[dict(name=f"c{j}", bank_id=f"L2_b{j}", case_hash="h", u_target=0.5) for j in range(4)])


def test_aggregate_complete(monkeypatch):
    monkeypatch.setattr(RL, "build_per_bank_rx",
                        lambda m, d, n, n_periods=3: ([[(0.5, 1.0, 20.0)]] * 4, [(0.5, 1.0, 20.0)] * 4, []))
    out = RUN.aggregate(_manifest_4(), "base")
    assert out["n_banks"] == 4 and len(out["per_bank_RX"]) == 4 and all(out["per_bank_anchor"])


def test_aggregate_raises_on_empty_bank(monkeypatch):
    monkeypatch.setattr(RL, "build_per_bank_rx",
                        lambda m, d, n, n_periods=3: ([[(0.5, 1.0, 20.0)], None, [(0.5, 1.0, 20.0)], [(0.5, 1.0, 20.0)]],
                                                      [(0.5, 1.0, 20.0), None, (0.5, 1.0, 20.0), (0.5, 1.0, 20.0)], []))
    with pytest.raises(RUN.RunnerError):
        RUN.aggregate(_manifest_4(), "base")


def test_aggregate_rejects_wrong_bank_count():
    # a 3-bank manifest must be rejected -- the count is derived, not trusted from a caller.
    m3 = dict(geometry_hash="g",
              cases=[dict(name=f"c{j}", bank_id=f"L2_b{j}", case_hash="h", u_target=0.5) for j in range(3)])
    with pytest.raises(RUN.RunnerError):
        RUN.aggregate(m3, "base")


# ---- resumable state round-trip --------------------------------------------------------------------------
