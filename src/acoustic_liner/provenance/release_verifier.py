# -*- coding: utf-8 -*-
"""FAIL-CLOSED release verifier (remediation §8 gate H; step 6 #5).

This is NOT a presence checker. It (a) requires a `{provenance, data}` wrapper and verifies the provenance is
GENUINELY complete against the LIVE repo (git commit + clean worktree, FINAL dependency lock + hash, config
hash, required roles, and re-hashed referenced files via a role->path manifest); (b) binds a Falcon FEM
artifact to the AUTHORITATIVE scenario by exact name + canonical SHA-256; (c) RECOMPUTES every scientific
cross-field invariant (coverage/area/kappa/A_other, frequency-key equality, finite-positive results, model
IDs, alpha=A/S, beta=Z_air*Y, delta_dB=10log10(explicit/volumetric)) and rejects NaN/Inf and booleans-as-
numbers; (d) executes an explicit list of MANDATORY pytest node IDs and rejects any missing/skipped/xfailed/
deselected node; (e) fails with zero artifacts unless a release manifest is given.

Because a real release requires a git worktree with a clean commit and a FINAL lock, this verifier correctly
FAILS in a bare checkout with a provisional lock -- that is the point.

CLI: `acoustic-liner verify release <artifact.json ...>`  (must name >=1 artifact) -> exit 0 iff all pass.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

MANDATORY_NODES = (
    "tests/integration/test_v1_v2_continuity.py::test_real_mesh_matches_authoritative_scenario",
    "tests/integration/test_v1_v2_continuity.py::test_migrated_solver_matches_v1_full_coverage",
    "tests/integration/test_v1_v2_continuity.py::test_defective_solver_differs_from_v1",
    "tests/integration/test_v1_v2_continuity.py::test_coverage_invariant_over_full_range",
    "tests/integration/test_v1_v2_continuity.py::test_generic_boundary_vs_volume_sensitivity",
)

_HEX = set("0123456789abcdef")


class ReleaseVerificationError(RuntimeError):
    """A release artifact or evidence gate failed verification."""


# --------------------------- numeric helpers (reject NaN/Inf and bool-as-number) --------------------------- #
def _num(rec: dict, key: str, kind: str) -> float:
    if key not in rec:
        raise ReleaseVerificationError(f"{kind}: missing numeric field {key!r}")
    v = rec[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ReleaseVerificationError(f"{kind}: {key} must be a real number, not {type(v).__name__}")
    f = float(v)
    if not math.isfinite(f):
        raise ReleaseVerificationError(f"{kind}: {key} must be finite; got {v}")
    return f


def _approx(a: float, b: float, rel: float = 1e-6, abs_: float = 1e-9) -> bool:
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), abs_)


def _is_hex64(s) -> bool:
    return isinstance(s, str) and len(s) == 64 and set(s) <= _HEX


# ------------------------------------------ provenance verification ---------------------------------------- #
def verify_provenance(prov: dict, expected_kind: str, repo_root: Path) -> None:
    """Recompute completeness against the LIVE repo and the artifact policy. Raises on any failure."""
    from . import hashing as HASH
    from . import records as PROV
    if not isinstance(prov, dict):
        raise ReleaseVerificationError("artifact has no provenance block")
    if prov.get("schema_version") != PROV.SCHEMA_VERSION:
        raise ReleaseVerificationError(f"provenance schema_version != {PROV.SCHEMA_VERSION}")
    if prov.get("artifact_kind") != expected_kind:
        raise ReleaseVerificationError(f"artifact_kind {prov.get('artifact_kind')!r} != {expected_kind!r}")
    if prov.get("status") != "complete":
        raise ReleaseVerificationError(f"provenance status is {prov.get('status')!r}, not 'complete'")
    # LIVE git + lock + config
    cur = PROV.git_commit()
    if not cur:
        raise ReleaseVerificationError("no git commit (not a worktree) -> provenance cannot be complete")
    if prov.get("git_commit") != cur:
        raise ReleaseVerificationError("provenance git_commit != current HEAD")
    if PROV.git_dirty() is not False:
        raise ReleaseVerificationError("git worktree is dirty or unknown")
    if PROV.dependency_lock_status() != "final":
        raise ReleaseVerificationError(f"dependency lock is not final ({PROV.dependency_lock_status()})")
    if prov.get("dependency_lock_sha256") != PROV.dependency_lock_sha256():
        raise ReleaseVerificationError("dependency lock hash mismatch")
    # recompute completeness against the policy (roles included)
    pol = PROV._resolve_policy(expected_kind)
    n_src = len(prov.get("source_file_sha256", {}) or {})
    n_in = len(prov.get("input_artifact_sha256", {}) or {})
    n_seed = len(prov.get("random_seeds", {}) or {})
    issues = PROV._completeness_issues(prov, pol, n_src, n_in, n_seed)
    if issues:
        raise ReleaseVerificationError(f"recomputed provenance is incomplete: {issues}")

    repo_root = repo_root.resolve()

    def _resolve_under_root(rel, what):
        p = Path(rel)
        if p.is_absolute():
            raise ReleaseVerificationError(f"{what} path must be relative: {rel}")
        full = (repo_root / p).resolve()
        if repo_root not in full.parents and full != repo_root:
            raise ReleaseVerificationError(f"{what} path escapes repo_root: {rel}")
        return full

    # configuration: recorded repo-relative path, contained, live-rehashed to equal configuration_sha256
    cfg_hash = prov.get("configuration_sha256")
    cfg_path = prov.get("configuration_path")
    if not cfg_hash or not cfg_path:
        raise ReleaseVerificationError("provenance lacks configuration_sha256/configuration_path")
    if not _is_hex64(cfg_hash):
        raise ReleaseVerificationError("configuration_sha256 is not 64-char lowercase hex")
    if HASH.sha256_file(_resolve_under_root(cfg_path, "configuration")) != cfg_hash:
        raise ReleaseVerificationError("configuration file live hash != configuration_sha256")

    # role_paths must cover EXACTLY the recorded source ∪ input roles (no missing, no extra); source and
    # input role names must not collide; every path relative + under repo_root + hash lowercase-hex-64 + live.
    src_hashes = prov.get("source_file_sha256") or {}
    in_hashes = prov.get("input_artifact_sha256") or {}
    if set(src_hashes) & set(in_hashes):
        raise ReleaseVerificationError(f"source/input role name collision: {set(src_hashes) & set(in_hashes)}")
    role_paths = prov.get("role_paths")
    if not isinstance(role_paths, dict):
        raise ReleaseVerificationError("provenance lacks a verifiable role_paths manifest")
    all_roles = set(src_hashes) | set(in_hashes)
    if set(role_paths) != all_roles:
        raise ReleaseVerificationError(
            f"role_paths must cover exactly the recorded roles; missing={all_roles - set(role_paths)}, "
            f"extra={set(role_paths) - all_roles}")
    hashes = {**src_hashes, **in_hashes}
    for role, rel in role_paths.items():
        recorded = hashes.get(role)
        if not _is_hex64(recorded):
            raise ReleaseVerificationError(f"role {role!r} hash is not 64-char lowercase hex")
        actual = HASH.sha256_file(_resolve_under_root(rel, f"role {role!r}"))
        if actual is None:
            raise ReleaseVerificationError(f"referenced file for role {role!r} not found: {rel}")
        if actual != recorded:
            raise ReleaseVerificationError(f"file hash mismatch for role {role!r} ({rel})")


# ------------------------------------------- scientific validators ----------------------------------------- #
def verify_fem_data(data: dict) -> None:
    """Recompute the scenario-FEM scientific invariants against the AUTHORITATIVE scenario."""
    from ..fem import scenarios as S
    from ..evaluation import requests as R
    from ..fem import solver as FS
    from ..fem.verification import reference_cavity as RC
    from ..fem import damping
    if not isinstance(data, dict):
        raise ReleaseVerificationError("FEM data must be a dict")
    scen = S.falcon9_2025_blanketed()
    if data.get("scenario_name") != scen.name:
        raise ReleaseVerificationError("scenario_name does not match the authoritative scenario")
    sha = data.get("scenario_sha256")
    if not _is_hex64(sha):
        raise ReleaseVerificationError("scenario_sha256 is not a 64-char lowercase hex hash")
    if sha != R.canonical_scenario_sha256(scen):
        raise ReleaseVerificationError("scenario_sha256 != canonical hash (artifact not bound to current scenario)")
    wall = _num(data, "measured_wall_area", "FEM")
    vol = _num(data, "measured_volume", "FEM")
    kappa = _num(data, "kappa", "FEM")
    A_other = _num(data, "A_other", "FEM")
    coverage = _num(data, "coverage", "FEM")
    # reproduce the scenario consistency gate on the recorded measured area/volume/kappa
    S.assert_fem_consistency(scen, wall, vol, A_other, expected_kappa=kappa)
    if not _approx(A_other, kappa * scen.A_wall, rel=1e-6):
        raise ReleaseVerificationError("A_other != kappa * A_wall")
    # coverage invariant recompute
    cc = data.get("coverage_consistency")
    if not isinstance(cc, dict):
        raise ReleaseVerificationError("missing coverage_consistency")
    for k in ("requested_coverage", "achieved_mesh_coverage", "treated_mesh_area", "mass_treated_area",
              "mesh_wall_area", "abs_coverage_error", "tol"):
        _num(cc, k, "coverage_consistency")
    if cc.get("passed") is not True:
        raise ReleaseVerificationError("coverage_consistency.passed is not True")
    if not _approx(cc["requested_coverage"], coverage):
        raise ReleaseVerificationError("coverage_consistency.requested != top-level coverage")
    if not _approx(cc["mass_treated_area"], coverage * scen.A_wall, rel=1e-6):
        raise ReleaseVerificationError("mass_treated_area != requested coverage * authoritative A_wall")
    if not _approx(cc["achieved_mesh_coverage"], cc["treated_mesh_area"] / cc["mesh_wall_area"], rel=1e-6):
        raise ReleaseVerificationError("achieved_mesh_coverage != treated/mesh area")
    if not _approx(cc["abs_coverage_error"], abs(cc["achieved_mesh_coverage"] - cc["requested_coverage"])):
        raise ReleaseVerificationError("recorded abs_coverage_error != recomputed")
    if cc["abs_coverage_error"] > cc["tol"] + 1e-12:
        raise ReleaseVerificationError("coverage abs error exceeds tol")
    if not _approx(cc["tol"], FS.COVERAGE_CONSISTENCY_TOL):
        raise ReleaseVerificationError("coverage tol != fixed production threshold")
    # frequency-key equality + finite-positive mean-squares
    freqs = data.get("freqs")
    if not isinstance(freqs, list) or not freqs:
        raise ReleaseVerificationError("freqs must be a non-empty list")
    fset = set()
    for f in freqs:
        if isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(float(f)) or float(f) <= 0:
            raise ReleaseVerificationError("FEM freq not finite-positive")
        fset.add(float(f))
    ms = data.get("ms_by_freq") or {}
    if {float(k) for k in ms} != fset:
        raise ReleaseVerificationError("ms_by_freq keys != freqs")
    for k, v in ms.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or float(v) <= 0:
            raise ReleaseVerificationError(f"ms_by_freq[{k}] must be finite and > 0")
    # model IDs / constants
    if (data.get("source_model") or {}).get("source_model") != "v1_base_radiation_plus_incident":
        raise ReleaseVerificationError("source_model id is not the restored v1 model")
    dm = data.get("damping_model") or {}
    if dm.get("a_other_model") != "equivalent_distributed_volumetric_damping":
        raise ReleaseVerificationError("damping_model id is not the volumetric model")
    if not _approx(_num(dm, "coef", "damping_model"), damping.VOLUMETRIC_COEF, rel=1e-9):
        raise ReleaseVerificationError("damping coef != production constant")
    # physical ranges
    if not (0.0 <= coverage <= 1.0):
        raise ReleaseVerificationError("coverage out of [0,1]")
    if not (0.0 <= kappa <= 1.0) or A_other < 0 or wall <= 0 or vol <= 0:
        raise ReleaseVerificationError("kappa/A_other/wall/volume out of physical range")
    if _num(cc, "treated_mesh_area", "cc") < 0 or _num(cc, "mesh_wall_area", "cc") <= 0:
        raise ReleaseVerificationError("treated/mesh area out of range")
    # unique frequencies; mesh-wall equality; anchor
    if len(fset) != len(freqs):
        raise ReleaseVerificationError("freqs contains duplicates")
    if not _approx(cc["mesh_wall_area"], wall, rel=1e-6):
        raise ReleaseVerificationError("coverage_consistency.mesh_wall_area != measured_wall_area")
    if data.get("anchor") not in ("low", "high", "mid"):
        raise ReleaseVerificationError("anchor must be low|high|mid")
    # consistency record equality
    con = data.get("consistency")
    if not isinstance(con, dict):
        raise ReleaseVerificationError("missing consistency record")
    if not (_approx(_num(con, "mesh_wall_area", "consistency"), wall) and
            _approx(_num(con, "mesh_volume", "consistency"), vol) and
            _approx(_num(con, "expected_kappa", "consistency"), kappa)):
        raise ReleaseVerificationError("consistency record disagrees with measured area/volume/kappa")
    # boundary partition errors present + small
    bp = data.get("boundary_partition")
    if not isinstance(bp, dict) or not {"source", "wall", "axis"} <= set(bp):
        raise ReleaseVerificationError("boundary_partition missing source/wall/axis")
    for tag in ("source", "wall", "axis"):
        e = _num(bp, tag, "boundary_partition")
        if not (0.0 <= e <= FS.FEM_PARTITION_TOL):   # relative area errors are nonnegative by construction
            raise ReleaseVerificationError(f"boundary_partition {tag} error out of [0, {FS.FEM_PARTITION_TOL}]")
    # source-model constants beyond the ID: base admittance == 1.0 and a finite p0
    sm = data.get("source_model") or {}
    if not _approx(_num(sm, "base_admittance_normalized", "source_model"), 1.0):
        raise ReleaseVerificationError("source base admittance != 1.0")
    _num(sm, "p0", "source_model")                       # incident-source amplitude must be a finite number
    # boundary coefficients: per-freq wall beta present for every freq, passive (Re>=0)
    bcoeffs = data.get("boundary_coeffs") or {}
    bc = bcoeffs.get("wall_beta_by_freq") or {}
    if {float(k) for k in bc} != fset:
        raise ReleaseVerificationError("wall_beta_by_freq keys != freqs")
    for k, b in bc.items():
        if _num(b, "real", "wall_beta") < -1e-9:
            raise ReleaseVerificationError(f"wall beta at {k} is active (Re<0)")
        _num(b, "imag", "wall_beta")
    # wall_beta_is_callable must be a genuine bool (not a truthy stand-in)
    if not isinstance(bcoeffs.get("wall_beta_is_callable"), bool):
        raise ReleaseVerificationError("boundary_coeffs.wall_beta_is_callable must be a bool")
    # bare_beta is LOAD-BEARING for any coverage < 1 (untreated wall facets use it): require it present, finite
    # and passive there; require it absent (None) only where consistent with full coverage.
    bare = bcoeffs.get("bare_beta")
    if coverage < 1.0 - 1e-12:
        if not isinstance(bare, dict):
            raise ReleaseVerificationError("coverage < 1 requires a bare_beta coefficient, but none was recorded")
        if _num(bare, "real", "bare_beta") < -1e-9:
            raise ReleaseVerificationError("bare_beta is active (Re<0)")
        _num(bare, "imag", "bare_beta")
    elif bare is not None:
        raise ReleaseVerificationError("full coverage must not record a bare_beta (whole wall is treated)")
    # full damping record: A_other/kappa/volume + Sabine constant + per-frequency eta reproduced
    if not (_approx(_num(dm, "A_other", "damping"), A_other) and _approx(_num(dm, "cavity_volume", "damping"), vol)):
        raise ReleaseVerificationError("damping record A_other/volume disagree")
    if not _approx(_num(dm, "kappa", "damping"), kappa, rel=1e-9):
        raise ReleaseVerificationError("damping record kappa != top-level kappa")
    if not _approx(_num(dm, "sabine_const", "damping"), damping.SABINE_CONST, rel=1e-9):
        raise ReleaseVerificationError("damping record sabine_const != production Sabine constant")
    lf = dm.get("loss_factor_by_freq") or {}
    if {float(k) for k in lf} != fset:
        raise ReleaseVerificationError("damping loss_factor_by_freq keys != freqs")
    for k, eta in lf.items():
        exp = damping.VOLUMETRIC_COEF * A_other / (vol * float(k))
        if not _approx(float(eta), exp, rel=1e-6):
            raise ReleaseVerificationError(f"loss factor at {k} != COEF*A_other/(V*f)")


def verify_benchmark_data(data: dict) -> None:
    """Recompute the generic-benchmark scientific invariants."""
    from ..fem import solver as FS
    from ..fem.verification import reference_cavity as RC
    from ..physics import absorption as acoustics
    from ..physics import constants
    if not isinstance(data, dict):
        raise ReleaseVerificationError("benchmark data must be a dict")
    if data.get("interpretation") != "generic_benchmark_model_form_sensitivity":
        raise ReleaseVerificationError("benchmark interpretation must be the generic model-form sensitivity")
    if "NOT part of the Falcon robustness envelope" not in str(data.get("note", "")):
        raise ReleaseVerificationError("benchmark must be labeled outside the Falcon robustness envelope")
    if not _is_hex64(data.get("geometry_hash")):
        raise ReleaseVerificationError("geometry_hash is not 64-char lowercase hex")
    if data.get("diffuse_mapping_id") != acoustics.DIFFUSE_MAPPING_ID:
        raise ReleaseVerificationError("diffuse_mapping_id mismatch")
    if int(_num(data, "diffuse_quadrature_n", "benchmark")) != acoustics.DIFFUSE_QUADRATURE_N:
        raise ReleaseVerificationError("diffuse_quadrature_n mismatch")
    ap = data.get("air_properties") or {}
    if not (_approx(_num(ap, "z_air", "air"), constants.Z_AIR) and _approx(_num(ap, "c_air", "air"), constants.C_AIR)):
        raise ReleaseVerificationError("air_properties mismatch")
    A_other = _num(data, "A_other", "benchmark")
    alpha = _num(data, "alpha_other", "benchmark")
    a_max = _num(data, "alpha_other_max", "benchmark")
    S_other = _num(data, "measured_S_other", "benchmark")
    if alpha > a_max + 1e-9:
        raise ReleaseVerificationError("alpha_other exceeds the achievable diffuse maximum")
    if not _approx(alpha, A_other / S_other, rel=2e-2):     # alpha = A_other / S_other (mesh S_other)
        raise ReleaseVerificationError("alpha_other != A_other / measured_S_other")
    rt_tol = _num(data, "roundtrip_tol", "benchmark")
    if not _approx(rt_tol, RC.BENCHMARK_ROUNDTRIP_TOL, rel=1e-9) or rt_tol <= 0:
        raise ReleaseVerificationError("roundtrip_tol != fixed positive production tolerance")
    if _num(data, "roundtrip_error", "benchmark") > rt_tol:
        raise ReleaseVerificationError("diffuse round-trip error exceeds tol")
    Y = data.get("payload_admittance") or {}; beta = data.get("payload_beta") or {}
    if not _approx(_num(beta, "real", "payload_beta"), constants.Z_AIR * _num(Y, "real", "payload_admittance"), rel=1e-9):
        raise ReleaseVerificationError("payload_beta != Z_air * payload_admittance (units)")
    if data.get("delta_convention") != "10*log10(explicit_ms/volumetric_ms)":
        raise ReleaseVerificationError("delta convention mismatch")
    ex = data.get("explicit_ms") or {}; vo = data.get("volumetric_ms") or {}; dd = data.get("delta_dB") or {}
    if not (set(ex) == set(vo) == set(dd)) or not ex:
        raise ReleaseVerificationError("explicit/volumetric/delta frequency keys disagree")
    recomputed_max = 0.0
    for k in ex:
        e = ex[k]; v = vo[k]
        for name, val in (("explicit", e), ("volumetric", v)):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)) or float(val) <= 0:
                raise ReleaseVerificationError(f"{name}_ms[{k}] must be finite and > 0")
        d = 10.0 * math.log10(float(e) / float(v))
        if not _approx(d, float(dd[k]), rel=1e-6, abs_=1e-9):
            raise ReleaseVerificationError(f"delta_dB[{k}] != 10log10(explicit/volumetric)")
        recomputed_max = max(recomputed_max, abs(d))
    if not _approx(_num(data, "max_abs_delta_dB", "benchmark"), recomputed_max, rel=1e-6):
        raise ReleaseVerificationError("max_abs_delta_dB != recomputed maximum")
    # canonical cavity binding: reconstruct the ReferenceCavity from recorded dims and recompute hash + areas
    dims = data.get("cavity_dims")
    if not isinstance(dims, dict):
        raise ReleaseVerificationError("benchmark missing cavity_dims for canonical binding")
    try:
        cav = RC.ReferenceCavity(name=data.get("cavity_name", "generic_boundary_benchmark_v1"),
                                 R_outer=dims["R_outer"], H=dims["H"], R_payload=dims["R_payload"],
                                 H_payload=dims["H_payload"], z_payload=dims.get("z_payload"),
                                 max_A_other=dims["max_A_other"])
    except Exception as e:
        raise ReleaseVerificationError(f"cavity_dims do not reconstruct a valid ReferenceCavity: {e}")
    if cav.geometry_hash() != data["geometry_hash"]:
        raise ReleaseVerificationError("geometry_hash != recomputed canonical cavity hash")
    if not (_approx(_num(data, "measured_source_area", "bench"), cav.source_area, rel=3e-2) and
            _approx(_num(data, "measured_wall_area", "bench"), cav.lined_wall_area, rel=3e-2) and
            _approx(S_other, cav.S_other, rel=3e-2) and
            _approx(_num(data, "measured_volume", "bench"), cav.fluid_volume, rel=3e-2)):
        raise ReleaseVerificationError("measured areas/volume disagree with the canonical cavity")
    if not _approx(a_max, acoustics.diffuse_alpha_max(), rel=1e-9):
        raise ReleaseVerificationError("alpha_other_max != exact production diffuse maximum")
    # positive physical ranges
    if not (A_other > 0 and S_other > 0 and 0.0 < alpha <= 1.0):
        raise ReleaseVerificationError("A_other/S_other/alpha_other out of positive physical range")
    # area_tol / volume_rel_error present and consistent
    at = _num(data, "area_tol", "benchmark"); vre = _num(data, "volume_rel_error", "benchmark")
    if vre > at:
        raise ReleaseVerificationError("volume_rel_error exceeds area_tol")
    # boundary partition present + small (source/wall/payload/axis)
    bp = data.get("boundary_partition")
    if not isinstance(bp, dict) or not {"source", "wall", "payload", "axis"} <= set(bp):
        raise ReleaseVerificationError("benchmark boundary_partition missing tags")
    for tag in ("source", "wall", "payload", "axis"):
        e = _num(bp, tag, "boundary_partition")
        if not (0.0 <= e <= at):                     # relative area errors are nonnegative and within area_tol
            raise ReleaseVerificationError(f"benchmark partition {tag} error out of [0, area_tol={at}]")
    # freqs unique + finite-positive + matching every result map
    freqs = data.get("freqs")
    if not isinstance(freqs, list) or not freqs:
        raise ReleaseVerificationError("benchmark freqs must be a non-empty list")
    fset = set()
    for f in freqs:
        if isinstance(f, bool) or not isinstance(f, (int, float)) or not math.isfinite(float(f)) or float(f) <= 0:
            raise ReleaseVerificationError("benchmark freq not finite-positive")
        fset.add(float(f))
    if len(fset) != len(freqs):
        raise ReleaseVerificationError("benchmark freqs contain duplicates")
    if {float(k) for k in ex} != fset:
        raise ReleaseVerificationError("explicit/volumetric/delta keys != freqs")
    # Diffuse round-trip must CLOSE against the ORIGINAL alpha_other, not merely against the recorded
    # roundtrip_alpha: recompute alpha from the payload admittance and require it to match (a) the requested
    # alpha_other, (b) the recorded roundtrip_alpha, and (c) that roundtrip_error == abs(alpha_from_Y - alpha).
    # This blocks a different passive admittance that reproduces the right beta but a different absorption while
    # claiming a zero round-trip error.
    rt_a = acoustics.diffuse_alpha_from_admittance(complex(_num(Y, "real", "Y"), _num(Y, "imag", "Y")), constants.Z_AIR)
    if not _approx(rt_a, alpha, abs_=rt_tol):
        raise ReleaseVerificationError("diffuse alpha from admittance != requested alpha_other (round-trip open)")
    if not _approx(rt_a, _num(data, "roundtrip_alpha", "benchmark"), abs_=1e-9):
        raise ReleaseVerificationError("roundtrip_alpha != recomputed diffuse alpha from admittance")
    if not _approx(_num(data, "roundtrip_error", "benchmark"), abs(rt_a - alpha), abs_=1e-12):
        raise ReleaseVerificationError("roundtrip_error != abs(alpha_from_Y - alpha_other)")
    # full complex beta = Z_air * Y (imag too), wall beta passive, and rho_air == the production constant
    if not _approx(_num(beta, "imag", "payload_beta"), constants.Z_AIR * _num(Y, "imag", "payload_admittance"), abs_=1e-12):
        raise ReleaseVerificationError("payload_beta imag != Z_air * Y imag")
    wb = data.get("wall_beta") or {}
    if _num(wb, "real", "wall_beta") < -1e-9:
        raise ReleaseVerificationError("benchmark wall_beta is active (Re<0)")
    _num(wb, "imag", "wall_beta")
    if not _approx(_num(ap, "rho_air", "air"), constants.RHO_AIR, rel=1e-9):
        raise ReleaseVerificationError("air_properties.rho_air != production RHO_AIR")
    # ------- model records must be present and internally consistent (they can otherwise be deleted) -------
    sm = data.get("source_model") or {}
    if sm.get("source_model") != "v1_base_radiation_plus_incident":
        raise ReleaseVerificationError("benchmark source_model id is not the restored v1 model")
    if not _approx(_num(sm, "base_admittance_normalized", "source_model"), 1.0):
        raise ReleaseVerificationError("benchmark source base admittance != 1.0")
    _num(sm, "p0", "source_model")
    ea = data.get("explicit_a_other_model") or {}
    if not _is_hex64(ea.get("geometry_hash")) or ea["geometry_hash"] != data["geometry_hash"]:
        raise ReleaseVerificationError("explicit_a_other_model geometry_hash missing or != cavity geometry_hash")
    if not (_approx(_num(ea, "A_other", "explicit_model"), A_other) and
            _approx(_num(ea, "S_other", "explicit_model"), S_other)):
        raise ReleaseVerificationError("explicit_a_other_model A_other/S_other disagree with the benchmark")
    if not _approx(_num(ea, "alpha_other", "explicit_model"), A_other / S_other, rel=2e-2):
        raise ReleaseVerificationError("explicit_a_other_model alpha_other != A_other/S_other")
    from ..fem import damping as _damping
    vl = data.get("volumetric_loss_model") or {}
    if vl.get("a_other_model") != "equivalent_distributed_volumetric_damping":
        raise ReleaseVerificationError("volumetric_loss_model id is not the volumetric model")
    kappa_bench = A_other / cav.lined_wall_area          # benchmark-analogue kappa recomputed from the cavity
    if not (_approx(_num(vl, "A_other", "vol_model"), A_other) and
            _approx(_num(vl, "cavity_volume", "vol_model"), _num(data, "measured_volume", "bench")) and
            _approx(_num(vl, "kappa", "vol_model"), kappa_bench, rel=2e-2) and
            _approx(_num(vl, "coef", "vol_model"), _damping.VOLUMETRIC_COEF, rel=1e-9) and
            _approx(_num(vl, "sabine_const", "vol_model"), _damping.SABINE_CONST, rel=1e-9)):
        raise ReleaseVerificationError("volumetric_loss_model constants disagree with production")
    vlf = vl.get("loss_factor_by_freq") or {}
    if {float(k) for k in vlf} != fset:
        raise ReleaseVerificationError("volumetric_loss_model loss_factor_by_freq keys != freqs")
    vvol = _num(data, "measured_volume", "bench")
    for k, eta in vlf.items():
        exp = _damping.VOLUMETRIC_COEF * A_other / (vvol * float(k))
        if not _approx(float(eta), exp, rel=1e-6):
            raise ReleaseVerificationError(f"volumetric_loss_model eta at {k} != COEF*A_other/(V*f)")


# ------------------------------------------- artifact + CLI ------------------------------------------------ #
_KIND_BY_SHAPE = {"generic_benchmark_model_form_sensitivity": ("benchmark", "generic_benchmark")}


def verify_artifact_file(path: str | Path, repo_root: Path) -> str:
    """Require the {provenance,data} wrapper, verify provenance completeness, then the scientific invariants."""
    obj = json.loads(Path(path).read_text())      # NOTE: writers must use allow_nan=False
    if not (isinstance(obj, dict) and "provenance" in obj and "data" in obj):
        raise ReleaseVerificationError(f"{path}: release artifact must be a {{provenance, data}} wrapper")
    data = obj["data"]
    if data.get("interpretation") == "generic_benchmark_model_form_sensitivity":
        verify_provenance(obj["provenance"], "generic_benchmark", repo_root)
        verify_benchmark_data(data)
        return "generic-benchmark"
    verify_provenance(obj["provenance"], "fem", repo_root)
    verify_fem_data(data)
    return "scenario-FEM"


def run_mandatory_tests(repo_root: Path, nodes=MANDATORY_NODES) -> None:
    """Execute the EXPLICIT mandatory node IDs. Reject a non-zero run, or any missing/skipped/xfailed/
    deselected node. Substring 'skipped' detection alone is insufficient -- we assert each node PASSED."""
    # G1b (§14 step 9): the mandatory v1 continuity gate is SELF-CONTAINED -- it runs against a COMMITTED, HASHED
    # fixture of frozen v1 outputs (no live reference tree). Require the fixture present + self-hash-valid, else
    # the continuity nodes would SKIP -> FAIL (fail-closed, same strength as before, no reference-tree dependency).
    _fx = repo_root / "tests" / "fixtures" / "v1_continuity_reference.json"
    if not _fx.is_file():
        raise ReleaseVerificationError(
            "v1 continuity fixture missing (tests/fixtures/v1_continuity_reference.json): mandatory continuity "
            "would SKIP -> FAIL. Regenerate via acoustic-liner provenance continuity-fixture.")
    try:
        _doc = json.loads(_fx.read_text(encoding="utf-8"))
        _d = {k: v for k, v in _doc.items() if k != "self_sha256"}
        _calc = hashlib.sha256(json.dumps(_d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    except Exception as e:
        raise ReleaseVerificationError(f"v1 continuity fixture unreadable ({e}): mandatory continuity -> FAIL")
    if _doc.get("self_sha256") != _calc:
        raise ReleaseVerificationError("v1 continuity fixture failed its self SHA-256 check (tampered/corrupt)")
    env = dict(os.environ, NO_COLOR="1", PY_COLORS="0")   # belt: disable color via env too
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *nodes, "-p", "no:cacheprovider", "--no-header", "-rA", "-q",
         "--color=no"],
        cwd=str(repo_root), capture_output=True, text=True, env=env)
    # pytest can still emit ANSI color to a pipe in some environments; strip it so the PASSED-node check is robust.
    out = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise ReleaseVerificationError(f"mandatory tests did not all pass:\n{out[-2000:]}")
    for bad in ("skipped", "xfailed", "deselected", "error"):
        if bad in out.lower():
            raise ReleaseVerificationError(f"mandatory tests reported {bad!r}:\n{out[-1500:]}")
    for node in nodes:
        if f"PASSED {node}" not in out:
            raise ReleaseVerificationError(f"mandatory node did not PASS (missing/renamed?): {node}")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    from .. import cli
    if cli.show_help("verify", "release", argv):
        return 0
    from .. import paths
    repo_root = paths.ROOT
    if not argv:
        print("RELEASE VERIFICATION FAILED: no artifacts named (a release must validate its artifact set)",
              file=sys.stderr)
        return 1
    problems = []
    try:
        run_mandatory_tests(repo_root)
    except ReleaseVerificationError as e:
        problems.append(str(e))
    seen = set()
    for p in argv:
        try:
            kind = verify_artifact_file(p, repo_root)
            seen.add(kind)
            print(f"OK   {p}  ({kind})")
        except (ReleaseVerificationError, json.JSONDecodeError, OSError) as e:
            problems.append(f"{p}: {e}")
    # the current step-6 stage requires BOTH artifact kinds -- one artifact is not a complete release set
    required_kinds = {"scenario-FEM", "generic-benchmark"}
    if not required_kinds <= seen:
        problems.append(f"incomplete artifact set: missing {sorted(required_kinds - seen)} "
                        f"(a complete release must include both a scenario-FEM and a generic-benchmark artifact)")
    if problems:
        print("RELEASE VERIFICATION FAILED:", file=sys.stderr)
        for pr in problems:
            print(f"  - {pr}", file=sys.stderr)
        return 1
    print("RELEASE VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
