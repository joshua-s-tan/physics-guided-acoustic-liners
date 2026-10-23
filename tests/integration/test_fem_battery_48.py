# -*- coding: utf-8 -*-
"""§4.8 FEM battery (step 6 #6) -- routed through the MIGRATED production solver `acoustic_liner.fem.solver` and the
authoritative Falcon scenario. This battery gathers the load-bearing FEM invariants that are settled physics on
the existing production API:

  #1  f_cov = 0 reproduces the UNTREATED baseline (whole wall bare at the bare impedance);
  #3  equal patch & bare impedance -> zero NR at EVERY placement (anchor/coverage cannot manufacture NR);
  #5  patch admittance == 1/z_s over the frequency grid (locally-reacting normalized admittance mapping);
  #6  bare-wall admittance reproduces the requested diffuse absorption (Paris round-trip on the design band);
  #7  A_other reproduces the requested Sabine decay (modal T60 == Sabine T60 at the Falcon uncertainty corners);
  #9  mesh / coverage discretization convergence (finer mesh reduces both the solution change and the achieved-
      coverage error -- i.e. refine the mesh, never loosen the tolerance).

Two §4.8 items are intentionally NOT in this file because they require machinery that is not settled here:
  #8  source-boundary duct reflection MAGNITUDE and PHASE needs a plane-wave duct reflection extractor (or a v1
      `fem_full_duct_radiation` cross-check); and
  #10 the uncertainty-corner ROBUST MINIMUM ("robust value == minimum stored") presupposes the 4-corner robust-
      placement evaluation that is step 8.
Both are tracked separately (in the historical remediation plan, which is not distributed) so this battery makes no fabricated claim about them.

Items #2 (f_cov=1 matches the v1 reference) and #4 (coverage area matches requested) are already MANDATORY
gates in tests/integration/test_v1_v2_continuity.py and are not duplicated here.

Cross-checks against the v1 REFERENCE tree; SKIPS if it or the FEM toolchain is unavailable. Point
ACOUSTIC_LINER_V1_REFERENCE_TREE at the resonator checkout.
"""
import math
import os
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("skfem")
pytest.importorskip("gmsh")
pytest.importorskip("scipy")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from acoustic_liner.fem import mesh as MESH  # noqa: E402  (migrated production solver)
from acoustic_liner.fem import solver as FS
from acoustic_liner.fem.verification import reference_cavity as RC
from acoustic_liner.fem import scenarios as S  # noqa: E402  (authoritative scenario)
from acoustic_liner.physics import absorption as AC  # noqa: E402
from acoustic_liner.fem import damping as DMP  # noqa: E402
from acoustic_liner.physics import constants as C  # noqa: E402

SCEN = S.falcon9_2025_blanketed()
BANDS = SCEN.band_hz
ZETA_WALL = 3.0 + 0.0j
BETA_WALL = 1.0 / ZETA_WALL
BETA_BARE = 0.01 + 0.0j


def _shared_mesh(mesh_size=0.35):
    g = SCEN.geometry
    return MESH.build_fairing_interior_mesh(g.R_fair, g.H_barrel, g.H_total, mesh_size=mesh_size)


# --------------------------------------------------------------------------- #
#  #1  f_cov = 0 reproduces the untreated baseline
# --------------------------------------------------------------------------- #
def test_zero_coverage_is_untreated_baseline():
    """At coverage 0 the WHOLE wall is bare at `bare_beta`; the field must be identical (to numerical precision)
    to a wall entirely at that same impedance -- i.e. the untreated baseline. No treated patch exists, so the
    result must not differ from the untreated reference at any frequency."""
    mesh = _shared_mesh()
    tagged = MESH.tag_boundary_facets(mesh)
    for f in BANDS:
        # untreated reference: whole wall at BETA_BARE via the kernel (no treated facets)
        ref = FS.solve_interior(mesh, f, tagged, wall_beta=BETA_BARE, A_other=0.0)
        # production runner at coverage 0 (all wall bare at BETA_BARE), same volumetric A_other=0 path
        ev = FS.run_scenario_fem(SCEN, [f], wall_beta=BETA_WALL, coverage=0.0, bare_beta=BETA_BARE)
        got = ev["ms_by_freq"][float(f)]
        # both apply the same volumetric A_other (nominal kappa) and the same all-bare wall -> identical field.
        # Compare the runner (with A_other) against the runner's own untreated reference to isolate coverage=0:
        ref_runner = FS.run_scenario_fem(SCEN, [f], wall_beta=BETA_BARE, coverage=1.0)["ms_by_freq"][float(f)]
        assert got == pytest.approx(ref_runner, rel=1e-9), f"coverage=0 != full-bare wall at {f} Hz"
        assert np.isfinite(ref) and ref > 0


# --------------------------------------------------------------------------- #
#  #3  equal patch & bare impedance -> zero NR at every placement
# --------------------------------------------------------------------------- #
def test_equal_patch_and_bare_impedance_gives_zero_nr_everywhere():
    """If the patch and the bare wall share the SAME impedance, the wall is uniform and no placement (coverage
    or anchor) can manufacture noise reduction. NR = 10 log10(ref/test) must be ~0 for every coverage/anchor."""
    beta = BETA_WALL
    for cov in (0.3, 0.5, 0.7):
        for anchor in ("low", "high", "mid"):
            ref = FS.run_scenario_fem(SCEN, BANDS, wall_beta=beta, coverage=1.0)
            test = FS.run_scenario_fem(SCEN, BANDS, wall_beta=beta, coverage=cov, bare_beta=beta, anchor=anchor)
            for f in BANDS:
                nr = 10.0 * math.log10(ref["ms_by_freq"][float(f)] / test["ms_by_freq"][float(f)])
                assert abs(nr) < 1e-6, f"equal-impedance NR {nr} at cov={cov} anchor={anchor} f={f}"


# --------------------------------------------------------------------------- #
#  #5  patch admittance == 1 / z_s over the frequency grid
# --------------------------------------------------------------------------- #
def test_patch_admittance_equals_inverse_surface_impedance():
    """The locally-reacting patch admittance recovered from a requested diffuse absorption must equal 1/z_s,
    where z_s = zeta * Z_air is the (dimensional) surface impedance on the rigid branch (zeta >= 1). The
    normalized boundary coefficient the solver consumes is beta = Z_air * Y = 1/zeta."""
    for alpha in (0.05, 0.2, 0.5, AC.diffuse_alpha_max() * 0.99):
        zeta = AC.zeta_from_diffuse_alpha(alpha)             # rigid branch, zeta >= 1
        assert zeta >= 1.0 - 1e-12
        Y = AC.admittance_from_diffuse_alpha(alpha, C.Z_AIR)
        z_s = zeta * C.Z_AIR                                 # dimensional surface impedance
        assert Y == pytest.approx(1.0 / z_s, rel=1e-9)       # Y == 1/z_s
        # the normalized coefficient the FEM consumes is dimensionless beta = Z_air*Y = 1/zeta
        assert C.Z_AIR * Y == pytest.approx(1.0 / zeta, rel=1e-9)


# --------------------------------------------------------------------------- #
#  #6  bare-wall admittance reproduces the requested diffuse absorption
# --------------------------------------------------------------------------- #
def test_bare_wall_admittance_reproduces_requested_diffuse_absorption():
    """Round-trip on the design band: a requested diffuse absorption -> admittance -> diffuse absorption must
    return the request to the production round-trip tolerance (the same Paris integral the benchmark uses)."""
    for alpha in (0.03, 0.1, 0.3, 0.6):
        if alpha > AC.diffuse_alpha_max():
            continue
        Y = AC.admittance_from_diffuse_alpha(alpha, C.Z_AIR)
        back = AC.diffuse_alpha_from_admittance(Y, C.Z_AIR)
        assert back == pytest.approx(alpha, abs=RC.BENCHMARK_ROUNDTRIP_TOL)


# --------------------------------------------------------------------------- #
#  #7  A_other reproduces the requested Sabine decay at the uncertainty corners
# --------------------------------------------------------------------------- #
def test_A_other_reproduces_sabine_decay_at_corners():
    """The volumetric loss factor is DERIVED so the small-loss modal T60 equals the Sabine T60. Verify the
    identity at the Falcon uncertainty corners (kappa_eval_range endpoints) across the design band, and record
    that the small-loss approximation error stays bounded (~0.10% at the worst Falcon eta)."""
    lo, hi = SCEN.kappa_eval_range
    worst_rel = 0.0
    for kappa in (lo, SCEN.kappa_nominal, hi):
        A_other = SCEN.A_other(kappa)
        for f in BANDS:
            # modal T60 from the derived eta must equal the Sabine T60 (raises if not, within 1e-6)
            DMP.assert_sabine_consistency(A_other, SCEN.volume, float(f))
            eta = DMP.volumetric_loss_factor(A_other, SCEN.volume, float(f))
            worst_rel = max(worst_rel, DMP.small_loss_relative_error(eta))
    assert worst_rel < 2e-3, f"small-loss approximation error {worst_rel} exceeds the recorded ~0.10% bound"


# --------------------------------------------------------------------------- #
#  #9  mesh / coverage discretization convergence
# --------------------------------------------------------------------------- #
def test_mesh_and_coverage_discretization_converge():
    """Refining the mesh must (a) leave the full-coverage solution stable (coarse vs fine agree within a stated
    band) and (b) REDUCE the achieved-coverage error -- the preregistered response to discretization error is a
    finer mesh, never a looser tolerance."""
    coarse = FS.run_scenario_fem(SCEN, BANDS, wall_beta=BETA_WALL, coverage=1.0, mesh_size=0.35)
    fine = FS.run_scenario_fem(SCEN, BANDS, wall_beta=BETA_WALL, coverage=1.0, mesh_size=0.25)
    # (a) solution convergence: Falcon-weighted |dB| between coarse and fine is small
    e = np.array([10.0 ** (L / 10.0) for L in SCEN.spl_dB]); w = e / e.sum()
    d = np.array([10.0 * math.log10(coarse["ms_by_freq"][float(f)] / fine["ms_by_freq"][float(f)]) for f in BANDS])
    weighted = float(np.sum(w * np.abs(d)))
    assert weighted <= 0.5, f"full-coverage solution not mesh-converged: weighted |dB| {weighted}"
    # (b) coverage discretization: a finer mesh reduces the worst achieved-coverage error across placements
    def worst_cov_err(mesh_size):
        worst = 0.0
        for cov in (0.3, 0.5, 0.7):
            for a in ("low", "high", "mid"):
                cc = FS.run_scenario_fem(SCEN, [BANDS[0]], wall_beta=BETA_WALL, coverage=cov,
                                         bare_beta=BETA_BARE, anchor=a, mesh_size=mesh_size)["coverage_consistency"]
                worst = max(worst, cc["abs_coverage_error"])
        return worst
    worst_coarse = worst_cov_err(0.35)
    worst_fine = worst_cov_err(0.25)
    assert worst_fine <= worst_coarse + 1e-9, (
        f"finer mesh did not reduce coverage error: coarse {worst_coarse:.4f} -> fine {worst_fine:.4f}")
    assert worst_fine <= FS.COVERAGE_CONSISTENCY_TOL
