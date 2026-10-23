# -*- coding: utf-8 -*-
"""FEM boundary models (§4.2/§4.3) — analytic unit tests (no mesh): restored source/radiation coefficients,
explicit A_other boundary with the <=0.95 guard, and the coverage<->area invariant."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner.fem import boundary as B  # noqa: E402
from acoustic_liner.fem.verification import reference_cavity as RC  # noqa: E402


def _bench():
    return RC.ReferenceCavity(name="generic_boundary_benchmark_v1", R_outer=2.6, H=10.0,
                              R_payload=1.5, H_payload=4.0, max_A_other=43.6)


class _FakeVehicle:
    scenario_kind = "vehicle"
    is_vehicle = True
    S_other = 50.0
    def alpha_other(self, a):
        return a / self.S_other


class _ForgedBenchmark:
    """A duck-typed look-alike with the RIGHT strings but arbitrary S_other and no real geometry hash."""
    scenario_kind = "generic_boundary_benchmark"
    is_vehicle = False
    S_other = 999.0
    def alpha_other(self, a):
        return a / self.S_other


def test_restored_source_and_radiation_coefficients():
    k = 2.5
    assert B.base_radiation_robin_coeff(k) == pytest.approx(1j * k)      # v1 base radiation (was dropped)
    assert B.incident_source_coeff(k, p0=1.0) == pytest.approx(2j * k)   # v1 incident source (unchanged)
    assert B.incident_source_coeff(k, p0=2.0) == pytest.approx(4j * k)
    assert B.base_admittance_normalized() == 1.0
    assert B.SOURCE_MODEL == "v1_base_radiation_plus_incident"


def test_alpha_other_and_ceiling():
    assert B.alpha_other(10.0, 100.0) == pytest.approx(0.10)
    with pytest.raises(B.FEMBoundaryError):
        B.alpha_other(96.0, 100.0)          # 0.96 > 0.95 ceiling -> enlarge S_other
    with pytest.raises(B.FEMBoundaryError):
        B.alpha_other(10.0, 0.0)            # S_other must be > 0
    with pytest.raises(B.FEMBoundaryError):
        B.alpha_other(-1.0, 100.0)          # A_other >= 0


def test_other_boundary_admittance_validates_and_roundtrips():
    Z_air = 415.0
    freqs = [50.0, 100.0, 200.0]
    bench = _bench()
    def inv(a):                                  # alpha -> physical Y (rigid-branch resistive)
        zeta = (1 + (1 - a) ** 0.5) / (1 - (1 - a) ** 0.5) if 0 < a < 1 else 1.0
        return 1.0 / (zeta * Z_air)
    def inv_back(Y):                             # Y -> alpha (normal-incidence round-trip)
        zeta = 1.0 / (complex(Y) * Z_air)
        return B.normal_incidence_alpha(zeta)
    y = B.other_boundary_admittance(freqs, A_other=20.0, cavity=bench, admittance_from_alpha=inv,
                                    Z_air=Z_air, alpha_from_admittance=inv_back)
    assert len(y) == 3
    # an ACTIVE admittance is rejected INSIDE the function
    with pytest.raises(B.FEMBoundaryError):
        B.other_boundary_admittance(freqs, A_other=20.0, cavity=bench,
                                    admittance_from_alpha=lambda a: -0.001 + 0j, Z_air=Z_air)
    # bad Z_air / tol rejected
    with pytest.raises(B.FEMBoundaryError):
        B.other_boundary_admittance(freqs, A_other=20.0, cavity=bench, admittance_from_alpha=inv, Z_air=0.0)


def test_explicit_boundary_is_machine_enforced_generic_only():
    # a VEHICLE scenario must be rejected from explicit-boundary mode (fail closed) -> forces volumetric
    with pytest.raises(B.FEMBoundaryError):
        B.other_boundary_admittance([50.0], A_other=20.0, cavity=_FakeVehicle(),
                                    admittance_from_alpha=lambda a: 0.001 + 0j)
    with pytest.raises(B.FEMBoundaryError):
        B.explicit_a_other_record(20.0, _FakeVehicle())
    # a FORGED duck-typed look-alike (right strings, arbitrary S_other, no real hash) must ALSO fail
    with pytest.raises(B.FEMBoundaryError):
        B.explicit_a_other_record(20.0, _ForgedBenchmark())
    with pytest.raises(B.FEMBoundaryError):
        B.other_boundary_admittance([50.0], A_other=20.0, cavity=_ForgedBenchmark(),
                                    admittance_from_alpha=lambda a: 0.001 + 0j)
    # only an actual ReferenceCavity is accepted; its record carries a real 64-char geometry hash
    rec = B.explicit_a_other_record(20.0, _bench())
    assert rec["scenario_kind"] == "generic_boundary_benchmark" and len(rec["geometry_hash"]) == 64


def test_require_roundtrip_cannot_be_omitted():
    bench = _bench()
    with pytest.raises(B.FEMBoundaryError):     # production evidence path: inverse must be supplied
        B.other_boundary_admittance([50.0], A_other=20.0, cavity=bench,
                                    admittance_from_alpha=lambda a: 0.001 + 0j, require_roundtrip=True)


def test_frequencies_validated():
    bench = _bench()
    inv = lambda a: 0.001 + 0j
    with pytest.raises(B.FEMBoundaryError):     # empty
        B.other_boundary_admittance([], A_other=20.0, cavity=bench, admittance_from_alpha=inv)
    with pytest.raises(B.FEMBoundaryError):     # non-positive
        B.other_boundary_admittance([50.0, -1.0], A_other=20.0, cavity=bench, admittance_from_alpha=inv)
    with pytest.raises(B.FEMBoundaryError):     # non-finite
        B.other_boundary_admittance([float("nan")], A_other=20.0, cavity=bench, admittance_from_alpha=inv)


def test_admittance_units_beta_is_Zair_times_Y():
    # freeze units: physical Y -> dimensionless beta = Z_air * Y (guards a ~415x scaling error)
    Y = 0.0024 + 0.0003j
    assert B.to_normalized_beta(Y, Z_air=415.0) == pytest.approx(415.0 * Y)
    with pytest.raises(B.FEMBoundaryError):
        B.to_normalized_beta(Y, Z_air=-1.0)


def test_coverage_area_invariant_passes_when_consistent():
    r = B.assert_coverage_area_consistency(
        requested_coverage=0.60, achieved_mesh_coverage=0.601,
        treated_mesh_area=99.5, mesh_wall_area=165.6,       # ratio ~0.6009
        mass_treated_area=99.53, scenario_wall_area=165.878)  # ratio ~0.6
    assert r["ratio_mesh_coverage"] == pytest.approx(0.601, abs=2e-3)
    assert r["ratio_mass_coverage"] == pytest.approx(0.60, abs=2e-3)


def test_coverage_area_invariant_rejects_mismatch():
    with pytest.raises(B.FEMBoundaryError):
        # mass model charged 60% but the FEM only treated 40% of the wall -> invalid run
        B.assert_coverage_area_consistency(
            requested_coverage=0.60, achieved_mesh_coverage=0.60,
            treated_mesh_area=66.0, mesh_wall_area=165.6,      # ratio ~0.40
            mass_treated_area=99.5, scenario_wall_area=165.878)  # ratio ~0.60
    with pytest.raises(B.FEMBoundaryError):
        B.assert_coverage_area_consistency(1.2, 0.6, 99.0, 165.6, 99.0, 165.878)  # coverage>1


def test_admittance_passivity_and_absorption_roundtrip():
    Z_air = 415.0
    # a passive resistive termination zeta=3 -> Y = 1/(3 Z_air), Re(beta)=1/3 >= 0
    Y_passive = 1.0 / (3.0 * Z_air)
    B.assert_passive_admittance(Y_passive, Z_air)
    B.assert_passive_admittance([Y_passive, 0.0 + 0j], Z_air)
    # an ACTIVE admittance (negative real normalized) must be rejected
    with pytest.raises(B.FEMBoundaryError):
        B.assert_passive_admittance(-1.0 / (3.0 * Z_air), Z_air)
    with pytest.raises(B.FEMBoundaryError):
        B.assert_passive_admittance(float("inf") + 0j, Z_air)
    # absorption round-trip: alpha(zeta=1) = 1 (matched), alpha(rigid) -> 0
    assert B.normal_incidence_alpha(1.0) == pytest.approx(1.0)
    assert B.normal_incidence_alpha(1e9) == pytest.approx(0.0, abs=1e-6)


def test_source_record_has_no_a_other_model():
    rec = B.source_model_record(p0=1.0)
    assert rec["source_model"] == "v1_base_radiation_plus_incident"
    assert rec["base_admittance_normalized"] == 1.0
    assert "a_other_model" not in rec           # source metadata must NOT mislabel A_other (Falcon=volumetric)


def test_explicit_a_other_record_from_benchmark():
    bench = _bench()
    rec = B.explicit_a_other_record(A_other=20.0, cavity=bench)
    assert rec["a_other_model"] == "explicit_other_boundary_admittance"
    assert rec["S_other"] == pytest.approx(bench.S_other)
    assert rec["alpha_other"] == pytest.approx(20.0 / bench.S_other) and rec["max_alpha_other"] == 0.95
