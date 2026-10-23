# -*- coding: utf-8 -*-
"""Mass model: the SHARED-WALL partition term (§4.6). Proves (a) a finite N x N grid converges to the analytic
shared-wall areal formula, (b) shared-wall is exactly HALF the historical individual-cell term, (c) f_cov=0 ->
0 kg, (d) named BOM terms sum to the legacy lump, (e) area-fraction and input validation."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner.physics import mass as M  # noqa: E402


def test_finite_grid_converges_to_analytic_shared_wall():
    side = 0.05                                   # 50 mm cells
    analytic = 2.0 / side                         # shared-wall wall length per unit area
    prev_err = 1e9
    for N in (2, 5, 10, 50, 200):
        for boundary in (True, False):
            approx = M.finite_grid_wall_length_per_area(N, side, include_outer_boundary=boundary)
            err = abs(approx - analytic) / analytic
            if N >= 50:
                assert err < 0.03                 # within 3% by N=50
        # error must shrink as the grid refines (monotone convergence, boundary-included branch)
        err_incl = abs(M.finite_grid_wall_length_per_area(N, side, True) - analytic) / analytic
        assert err_incl <= prev_err + 1e-12
        prev_err = err_incl
    # in the limit both boundary conventions -> exactly 2/side
    assert M.finite_grid_wall_length_per_area(5000, side, True) == pytest.approx(analytic, rel=2e-3)
    assert M.finite_grid_wall_length_per_area(5000, side, False) == pytest.approx(analytic, rel=2e-3)


def test_shared_wall_is_half_of_individual_cell():
    n = 400.0
    shared = M.partition_wall_length_per_area(n, "shared_wall")
    individual = M.partition_wall_length_per_area(n, "individual_cell")
    assert shared == pytest.approx(0.5 * individual)
    assert shared == pytest.approx(2.0 * (n ** 0.5))


def test_partition_areal_mass_uses_shared_coefficient():
    # rho*wall_t*D*2*sqrt(n)
    m = M.partition_areal_mass(n_cells_per_m2=400.0, wall_t=0.0006, D=0.05, rho_wall=1550.0)
    assert m == pytest.approx(1550.0 * 0.0006 * 0.05 * 2.0 * 20.0)


def test_zero_coverage_zero_mass():
    bom = M.InstallBOM()
    out = M.total_installed_mass(liner_areal=2.59, f_cov=0.0, A_wall=165.878, bom=bom, m_fixed=0.0)
    assert out["m_total"] == 0.0 and out["A_treated"] == 0.0


def test_named_bom_sums_to_legacy_lump():
    assert M.InstallBOM().install_areal_mass() == pytest.approx(M.MU_A_LUMPED_LEGACY)


def test_install_breakdown_is_named_and_sums_to_aggregate():
    out = M.total_installed_mass(liner_areal=2.59, f_cov=1.0, A_wall=100.0, bom=M.InstallBOM())
    bd = out["install_breakdown"]
    assert set(bd) == {"adhesive", "fastener", "frame_edge", "manufacturing_margin"}
    assert sum(bd.values()) == pytest.approx(out["m_install"])
    assert bd["adhesive"] == pytest.approx(0.10 * 100.0)      # named term is auditable, not recombined


def _grp(area_frac, s=0.03, D=0.05):
    return dict(area_frac=area_frac, cavity_volume=s * s * D)      # footprint a=s^2 -> V_cav=s^2*D


def test_density_derived_from_cavity_geometry():
    # local density is D/V_cav = 1/footprint -- the SAME geometry feasibility uses (no injected density)
    D, s = 0.05, 0.03
    V = s * s * D
    assert M.local_cell_density(V, D) == pytest.approx(D / V)
    assert M.local_cell_density(V, D) == pytest.approx(1.0 / (s * s))


def test_total_mass_breakdown_and_full_coverage():
    bom = M.InstallBOM()
    groups = [_grp(0.5), _grp(0.5)]
    la = M.liner_areal_mass(groups, D=0.05, face_t=0.0008, wall_t=0.0006)
    out = M.total_installed_mass(la, f_cov=1.0, A_wall=165.878, bom=bom)
    assert out["A_treated"] == pytest.approx(165.878)
    assert out["m_total"] == pytest.approx(out["m_liner"] + out["m_install"] + out["m_fixed"])
    assert out["m_install"] == pytest.approx(M.MU_A_LUMPED_LEGACY * 165.878)


def test_area_fraction_sum_must_be_one():
    with pytest.raises(M.MassModelError):                          # sum 1.4
        M.liner_areal_mass([_grp(0.7), _grp(0.7)], D=0.05, face_t=0.0008, wall_t=0.0006)
    with pytest.raises(M.MassModelError):                          # sum 0.5 undercounts partitions
        M.liner_areal_mass([_grp(0.5)], D=0.05, face_t=0.0008, wall_t=0.0006)


def test_negative_area_fraction_rejected():
    with pytest.raises(M.MassModelError):
        M.liner_areal_mass([_grp(1.0), _grp(-0.2)], D=0.05, face_t=0.0008, wall_t=0.0006)


def test_negative_backing_thickness_rejected_at_construction():
    with pytest.raises(M.MassModelError):
        M.InstallBOM(backing_t=-0.001)
    with pytest.raises(M.MassModelError):
        M.InstallBOM(adhesive_areal=-0.1)


def test_bad_inputs_rejected():
    with pytest.raises(M.MassModelError):
        M.partition_wall_length_per_area(400.0, topology="honeycomb_unknown")
    with pytest.raises(M.MassModelError):
        M.total_installed_mass(liner_areal=2.59, f_cov=1.2, A_wall=165.878)
    with pytest.raises(M.MassModelError):
        M.finite_grid_wall_length_per_area(0, 0.05)
    with pytest.raises(M.MassModelError):
        M.local_cell_density(0.0, 0.05)                            # V_cav must be > 0
