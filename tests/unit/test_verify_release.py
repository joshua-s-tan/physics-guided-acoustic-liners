# -*- coding: utf-8 -*-
"""FAIL-CLOSED release verifier (§8 gate H; #5). Uses REAL runner records (not `"x"` stubs), then mutates
each field/invariant to prove rejection: NaN/Inf, fake hashes, bare records, incomplete/forged provenance,
missing mandatory tests, omitted artifacts, inconsistent areas, mismatched frequency keys, altered dB."""
import copy
import json
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("gmsh")
pytest.importorskip("skfem")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner.provenance import release_verifier as V  # noqa: E402
from acoustic_liner.fem import artifacts as FART  # noqa: E402
from acoustic_liner.fem import solver as FS
from acoustic_liner.fem.verification import reference_cavity as RC
from acoustic_liner.fem import scenarios as S  # noqa: E402
from acoustic_liner.provenance import records as PROV  # noqa: E402
from acoustic_liner.provenance import records as P
from acoustic_liner.physics import absorption as AC  # noqa: E402
from acoustic_liner.physics import constants as C  # noqa: E402


# ---- real records (built once) ----
SCEN = S.falcon9_2025_blanketed()
FEM = FS.run_scenario_fem(SCEN, SCEN.band_hz, wall_beta=1.0 / 3.0 + 0j, coverage=0.6, bare_beta=0.01 + 0j)
CAV = RC.ReferenceCavity(name="generic_boundary_benchmark_v1", R_outer=2.6, H=10.0,
                         R_payload=1.5, H_payload=4.0, max_A_other=43.6)
BENCH = RC.run_reference_cavity_benchmark(CAV, SCEN.band_hz, wall_beta=1.0 / 3.0 + 0j, A_other=40.0)


# ============================ scientific-data recomputation (no provenance) ============================
def test_real_fem_data_passes_recomputation():
    V.verify_fem_data(copy.deepcopy(FEM))


def test_fem_scenario_binding_enforced():
    bad = copy.deepcopy(FEM); bad["scenario_sha256"] = "0" * 64          # valid hex, wrong value
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_fem_data(bad)
    bad = copy.deepcopy(FEM); bad["scenario_sha256"] = "z" * 64          # not hex
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_fem_data(bad)
    bad = copy.deepcopy(FEM); bad["scenario_name"] = "falcon9_other"
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_fem_data(bad)


def test_fem_rejects_nan_and_inconsistent_coverage():
    bad = copy.deepcopy(FEM); bad["measured_wall_area"] = float("nan")
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_fem_data(bad)
    bad = copy.deepcopy(FEM); bad["coverage_consistency"]["mass_treated_area"] *= 1.5   # != requested*A_wall
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_fem_data(bad)
    bad = copy.deepcopy(FEM); bad["coverage_consistency"]["achieved_mesh_coverage"] += 0.2  # != treated/mesh
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_fem_data(bad)


def test_fem_rejects_freqkey_and_ms_tampering():
    bad = copy.deepcopy(FEM); bad["ms_by_freq"].pop(next(iter(bad["ms_by_freq"])))
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_fem_data(bad)
    bad = copy.deepcopy(FEM)
    k = next(iter(bad["ms_by_freq"])); bad["ms_by_freq"][k] = -1.0
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_fem_data(bad)


def test_fem_rejects_nonpositive_frequency():
    bad = copy.deepcopy(FEM); bad["freqs"][0] = -1.0
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_fem_data(bad)


def test_fem_requires_bare_beta_for_partial_coverage():
    # FEM was run at coverage=0.6, so bare_beta is LOAD-BEARING and cannot be dropped or nulled.
    for mut in (lambda p: p["boundary_coeffs"].__setitem__("bare_beta", None),
                lambda p: p["boundary_coeffs"].pop("bare_beta"),
                lambda p: p["boundary_coeffs"]["bare_beta"].__setitem__("real", -1.0)):  # active
        bad = copy.deepcopy(FEM); mut(bad)
        with pytest.raises(V.ReleaseVerificationError):
            V.verify_fem_data(bad)


def test_fem_rejects_source_and_damping_tampering():
    for mut in (lambda p: p["source_model"].pop("p0"),                      # missing incident amplitude
                lambda p: p["boundary_coeffs"].__setitem__("wall_beta_is_callable", 1),  # not a bool
                lambda p: p["damping_model"].__setitem__("kappa", p["damping_model"]["kappa"] * 1.5),
                lambda p: p["damping_model"].__setitem__("sabine_const", 0.2)):
        bad = copy.deepcopy(FEM); mut(bad)
        with pytest.raises(V.ReleaseVerificationError):
            V.verify_fem_data(bad)


def test_fem_rejects_negative_partition_error():
    bad = copy.deepcopy(FEM); bad["boundary_partition"]["wall"] = -1e-4
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_fem_data(bad)


def test_real_benchmark_data_passes():
    V.verify_benchmark_data(copy.deepcopy(BENCH))


def test_benchmark_rejects_dB_and_unit_tampering():
    bad = copy.deepcopy(BENCH); k = next(iter(bad["delta_dB"])); bad["delta_dB"][k] += 0.5   # != 10log10(e/v)
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_benchmark_data(bad)
    bad = copy.deepcopy(BENCH); bad["payload_beta"]["real"] *= 2.0                    # break beta=Z_air*Y
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_benchmark_data(bad)
    bad = copy.deepcopy(BENCH); bad["note"] = "great Falcon result"                   # mislabel
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_benchmark_data(bad)
    bad = copy.deepcopy(BENCH); bad["max_abs_delta_dB"] = 999.0                        # != recomputed
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_benchmark_data(bad)


def test_benchmark_requires_model_records():
    for key in ("source_model", "explicit_a_other_model", "volumetric_loss_model"):
        bad = copy.deepcopy(BENCH); bad.pop(key)                     # a deleted model record must be rejected
        with pytest.raises(V.ReleaseVerificationError):
            V.verify_benchmark_data(bad)
    # tamper the explicit model's geometry hash away from the cavity hash
    bad = copy.deepcopy(BENCH); bad["explicit_a_other_model"]["geometry_hash"] = "0" * 64
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_benchmark_data(bad)
    # tamper a per-frequency volumetric loss factor
    bad = copy.deepcopy(BENCH)
    k = next(iter(bad["volumetric_loss_model"]["loss_factor_by_freq"]))
    bad["volumetric_loss_model"]["loss_factor_by_freq"][k] *= 2.0
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_benchmark_data(bad)


def test_benchmark_rejects_rho_and_partition_tampering():
    bad = copy.deepcopy(BENCH); bad["air_properties"]["rho_air"] = 999.0
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_benchmark_data(bad)
    bad = copy.deepcopy(BENCH); bad["boundary_partition"]["wall"] = -1e-4     # errors are nonnegative
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_benchmark_data(bad)


def test_benchmark_rejects_open_roundtrip_with_faked_error():
    """A DIFFERENT passive admittance that reproduces a self-consistent roundtrip_alpha but a DIFFERENT
    absorption than the requested alpha_other -- while claiming roundtrip_error == 0 -- must be rejected."""
    bad = copy.deepcopy(BENCH)
    alpha0 = bad["alpha_other"]
    # find a passive zeta whose diffuse alpha differs from the requested one, then its admittance
    zeta2 = AC.zeta_from_diffuse_alpha(alpha0) + 0.5           # shift on the rigid branch -> different alpha
    Y2 = 1.0 / (zeta2 * C.Z_AIR)
    a2 = AC.diffuse_alpha_from_admittance(complex(Y2, 0.0), C.Z_AIR)
    assert abs(a2 - alpha0) > 1e-3                             # genuinely different absorption
    bad["payload_admittance"] = dict(real=Y2, imag=0.0)
    bad["payload_beta"] = dict(real=C.Z_AIR * Y2, imag=0.0)   # keep beta = Z_air*Y self-consistent
    bad["roundtrip_alpha"] = a2                               # self-consistent with the NEW admittance
    bad["roundtrip_error"] = 0.0                              # but falsely claims a closed round-trip
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_benchmark_data(bad)


# ============================ provenance verification (simulate a complete repo) ============================
def _captured_prov(monkeypatch):
    """Provenance produced through the PRODUCTION API `capture()` (not hand-built), with the live git/lock
    probes monkeypatched to simulate a clean release checkout with a FINAL lock."""
    monkeypatch.setattr(P, "git_commit", lambda: "c0ffeecommit")
    monkeypatch.setattr(P, "git_dirty", lambda: False)
    monkeypatch.setattr(P, "dependency_lock_status", lambda: "final")
    monkeypatch.setattr(P, "dependency_lock_sha256", lambda: "l" * 64)
    src = ROOT / "src" / "acoustic_liner"
    cfg = ROOT / "configs" / "falcon9_2025_blanketed.yaml"       # a real repo-relative config
    return PROV.capture(project_version="t", policy="generic_benchmark", config_path=str(cfg),
                     source_files={"reference_cavity.py": src / "fem" / "verification" / "reference_cavity.py",
                                   "acoustics.py": src / "physics" / "absorption.py",
                                   "fem_solver.py": src / "fem" / "solver.py"},
                     status="complete")


def test_captured_provenance_passes_production_path(monkeypatch):
    prov = _captured_prov(monkeypatch)
    assert prov["status"] == "complete"                          # capture() itself computed complete
    assert set(prov["role_paths"]) == {"reference_cavity.py", "acoustics.py", "fem_solver.py"}
    assert prov["configuration_path"] == "configs/falcon9_2025_blanketed.yaml"
    V.verify_provenance(prov, "generic_benchmark", ROOT)         # production artifact passes verification


def test_provenance_mutations_fail(monkeypatch):
    prov = _captured_prov(monkeypatch)
    for mut in (
        lambda p: p.update(status="incomplete"),
        lambda p: p.update(git_commit="deadbeef"),                          # != HEAD
        lambda p: p["source_file_sha256"].__setitem__("acoustics.py", "0" * 64),  # forged hash
        lambda p: p.pop("role_paths"),                                      # missing manifest
        lambda p: p["role_paths"].pop("acoustics.py"),                      # missing one role path
        lambda p: p["role_paths"].__setitem__("extra.py", "src/acoustic_liner/physics/constants.py"),  # extra role
        lambda p: p["role_paths"].__setitem__("acoustics.py", "/etc/passwd"),           # absolute
        lambda p: p["role_paths"].__setitem__("acoustics.py", "../../../etc/passwd"),    # traversal escape
        lambda p: p.update(configuration_sha256="0" * 64),                  # config hash mismatch
        lambda p: p.update(configuration_path="../outside.yaml"),           # config path escape
    ):
        bad = copy.deepcopy(prov); mut(bad)
        with pytest.raises(V.ReleaseVerificationError):
            V.verify_provenance(bad, "generic_benchmark", ROOT)


def _sim_clean_final(monkeypatch):
    monkeypatch.setattr(P, "git_commit", lambda: "c0ffeecommit")
    monkeypatch.setattr(P, "git_dirty", lambda: False)
    monkeypatch.setattr(P, "dependency_lock_status", lambda: "final")
    monkeypatch.setattr(P, "dependency_lock_sha256", lambda: "l" * 64)


def test_fem_writer_produces_complete_verifiable_artifact(tmp_path, monkeypatch):
    """The PRODUCTION FEM writer path: write_fem_artifact -> capture -> {provenance,data} wrapper ->
    verify_artifact_file passes under simulated clean/final conditions, and BOTH required input roles
    (physical_scenario AND finalists) are present with live-verifiable hashes. Guards the regression where
    the writer omitted `finalists`, so the `fem` policy could never be marked complete."""
    _sim_clean_final(monkeypatch)
    cfg = ROOT / "configs" / "falcon9_2025_blanketed.yaml"
    finalists = ROOT / "README.md"                         # any real repo file: exercises the finalists role
    art = tmp_path / "fem_release.json"
    prov = FART.write_fem_artifact(art, copy.deepcopy(FEM), cfg, finalists, status="complete")
    assert prov["status"] == "complete"                    # the fem policy is satisfiable via the writer
    assert set(prov["input_artifact_sha256"]) == {"physical_scenario", "finalists"}
    assert set(prov["role_paths"]) >= {"physical_scenario", "finalists"}
    for role in ("physical_scenario", "finalists"):
        assert V._is_hex64(prov["input_artifact_sha256"][role])
    assert V.verify_artifact_file(art, ROOT) == "scenario-FEM"


def test_fem_writer_missing_finalists_is_a_type_error(tmp_path, monkeypatch):
    _sim_clean_final(monkeypatch)
    cfg = ROOT / "configs" / "falcon9_2025_blanketed.yaml"
    with pytest.raises(TypeError):                         # finalists_path is now mandatory
        FART.write_fem_artifact(tmp_path / "x.json", copy.deepcopy(FEM), cfg)


def test_source_input_role_collision_rejected(monkeypatch):
    prov = _captured_prov(monkeypatch)
    prov["input_artifact_sha256"] = {"acoustics.py": "a" * 64}   # same name as a source role
    prov["input_roles"] = ["acoustics.py"]
    prov["role_paths"]["acoustics.py"] = "src/acoustic_liner/physics/absorption.py"
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_provenance(prov, "generic_benchmark", ROOT)


def test_provisional_lock_or_no_git_fails(monkeypatch):
    prov = _captured_prov(monkeypatch)
    monkeypatch.setattr(P, "dependency_lock_status", lambda: "provisional")   # the REAL current state
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_provenance(prov, "generic_benchmark", ROOT)
    monkeypatch.setattr(P, "dependency_lock_status", lambda: "final")
    monkeypatch.setattr(P, "git_commit", lambda: None)                        # not a worktree
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_provenance(prov, "generic_benchmark", ROOT)


# ============================ wrapper / CLI gates ============================
def test_bare_record_and_wrapper(tmp_path, monkeypatch):
    bare = tmp_path / "bare.json"; bare.write_text(json.dumps(FEM))
    with pytest.raises(V.ReleaseVerificationError):
        V.verify_artifact_file(bare, ROOT)                     # bare (no provenance/data) rejected


def test_zero_args_fails():
    assert V.main([]) == 1                                     # a release must name its artifact set


def test_mandatory_gate_fails_without_fixture(tmp_path, monkeypatch):
    """G1b: the mandatory continuity gate is SELF-CONTAINED (no ACOUSTIC_LINER_V1_REFERENCE_TREE requirement) but stays
    FAIL-CLOSED on the committed fixture: a MISSING fixture makes the gate fail before any test runs."""
    monkeypatch.delenv("ACOUSTIC_LINER_V1_REFERENCE_TREE", raising=False)
    with pytest.raises(V.ReleaseVerificationError, match="fixture missing"):
        V.run_mandatory_tests(tmp_path)                 # empty repo root -> no fixture -> fail-closed


def test_mandatory_gate_fails_on_tampered_fixture(tmp_path, monkeypatch):
    """A fixture whose self SHA-256 does not verify (tampered/corrupt) must fail the gate before running tests."""
    import json
    monkeypatch.delenv("ACOUSTIC_LINER_V1_REFERENCE_TREE", raising=False)
    fx = tmp_path / "tests" / "fixtures" / "v1_continuity_reference.json"
    fx.parent.mkdir(parents=True)
    fx.write_text(json.dumps({"mesh_fingerprint": "0" * 64, "self_sha256": "deadbeef"}))
    with pytest.raises(V.ReleaseVerificationError, match="self SHA-256"):
        V.run_mandatory_tests(tmp_path)
