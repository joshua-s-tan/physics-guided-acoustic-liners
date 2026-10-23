# -*- coding: utf-8 -*-
"""Scenario invariants (remediation §4.1 + review round 2). Pins geometry-derived area, the fail-closed
consistency gate (must survive `python -O`), per-corner kappa validation, record validation, and the
in-band energy audit that catches a wrong/mislabelled spectrum."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner.fem import scenarios as S  # noqa: E402

CFG = ROOT / "configs" / "falcon9_2025_blanketed.yaml"


def falcon():
    return S.load_scenario(CFG)


def _sem():
    return S.ClaimSemantics(id="x", metric="relative_dOASPL_only",
                            absolute_post_treatment_spl_allowed=False, validated_retrofit_claim_allowed=False,
                            blankets_modelled=False, blanket_replacement_claim_allowed=False,
                            alpha_bare_source="modelling_assumption", kappa_source="modelling_assumption")


def test_yaml_is_source_of_truth_and_loads():
    s = falcon()
    assert s.name == "falcon9_2025_blanketed"
    # correct Falcon spectrum (NOT Ariane 6's [131,132,...])
    assert s.spl_dB == (120.0, 120.0, 119.8, 120.5, 121.5, 122.0, 121.5, 120.5)
    assert s.oaspl_dB == pytest.approx(131.4)


def test_derived_area_and_volume():
    s = falcon()
    assert s.A_wall == pytest.approx(165.878, abs=1e-3)
    assert s.volume == pytest.approx(187.595, abs=1e-3)


def test_inband_energy_consistent_with_listed_oaspl():
    s = falcon()
    assert s.inband_oaspl <= s.oaspl_dB          # 129.83 <= 131.4
    assert s.inband_oaspl == pytest.approx(129.83, abs=0.05)


def test_historical_kappa_range_preserved():
    s = falcon()
    assert s.kappa_nominal == pytest.approx(0.23270570707355803, rel=1e-12)
    assert s.kappa_eval_range[0] == pytest.approx(0.20270570707355803, rel=1e-12)
    assert s.kappa_eval_range[1] == pytest.approx(0.262705707073558, rel=1e-12)


def test_consistency_gate_raises_not_asserts():
    s = falcon()
    # matching mesh passes at nominal kappa
    S.assert_fem_consistency(s, 165.878, s.volume, s.kappa_nominal * 165.878, expected_kappa=s.kappa_nominal)
    # historical 83.78 mismatch must RAISE (an explicit exception, not an assert)
    with pytest.raises(S.ScenarioConsistencyError):
        S.assert_fem_consistency(s, 83.781, s.volume, 19.4963, expected_kappa=s.kappa_nominal)


def test_consistency_gate_validates_uncertainty_corner():
    s = falcon()
    klo = s.kappa_eval_range[0]
    # applying the low-kappa corner passes when expected_kappa is that corner...
    S.assert_fem_consistency(s, 165.878, s.volume, klo * 165.878, expected_kappa=klo)
    # ...but is (correctly) rejected if checked against nominal
    with pytest.raises(S.ScenarioConsistencyError):
        S.assert_fem_consistency(s, 165.878, s.volume, klo * 165.878, expected_kappa=s.kappa_nominal)


def test_bad_geometry_rejected():
    with pytest.raises(S.ScenarioValidationError):
        S.FairingGeometry(R_fair=2.6, H_barrel=6.7, H_total=6.7)   # H_total !> H_barrel


def test_wrong_spectrum_energy_audit_would_reject():
    """The mislabelled Ariane spectrum ([131..128]) has in-band ~141 dB > Falcon total 131.4 -> rejected."""
    with pytest.raises(S.ScenarioValidationError):
        S.PhysicalScenario(
            name="bad", geometry=S.FairingGeometry(2.6, 6.7, 13.1),
            source=S.SourceConfig("x", "2025", "http://x", "present", 0.6), semantics=_sem(),
            spl_dB=(131.0, 132.0, 133.0, 133.0, 132.0, 131.0, 129.5, 128.0),
            oaspl_dB=131.4, alpha_bare_nominal=0.05, alpha_bare_range=(0.02, 0.10),
            kappa_nominal=0.2327, kappa_eval_range=(0.2027, 0.2627))


def test_semantics_loaded_and_enforced():
    s = falcon()
    assert s.semantics.id == "falcon_with_blankets_mpe_weighted_incremental_treatment"
    assert s.semantics.metric == "relative_dOASPL_only"
    s.semantics.assert_claim_allowed("relative_dOASPL")            # allowed
    for forbidden in ("absolute_post_treatment_spl", "validated_retrofit", "blanket_replacement"):
        with pytest.raises(S.ClaimNotAllowedError):
            s.semantics.assert_claim_allowed(forbidden)


def test_missing_semantics_rejected():
    with pytest.raises(S.ScenarioValidationError):
        S.PhysicalScenario(
            name="s", geometry=S.FairingGeometry(2.6, 6.7, 13.1),
            source=S.SourceConfig("x", "2025", "http://x", "present", 0.6), semantics=None,
            spl_dB=(120.0, 120.0, 119.8, 120.5, 121.5, 122.0, 121.5, 120.5),
            oaspl_dB=131.4, alpha_bare_nominal=0.05, alpha_bare_range=(0.02, 0.10),
            kappa_nominal=0.2327, kappa_eval_range=(0.2027, 0.2627))


def test_geometry_kind_and_todict():
    s = falcon()
    assert s.geometry.kind == "representative_axisymmetric_cone_approximation"
    d = s.to_dict()
    assert d["semantics"]["id"] == "falcon_with_blankets_mpe_weighted_incremental_treatment"
    assert d["geometry"]["kind"] == "representative_axisymmetric_cone_approximation"


def test_nonfinite_and_negative_fields_rejected():
    g = S.FairingGeometry(2.6, 6.7, 13.1)
    src = S.SourceConfig("n", "2025", "http://x", "present", 0.6)
    base = dict(name="s", geometry=g, source=src, semantics=_sem(),
                spl_dB=(120.0, 120.0, 119.8, 120.5, 121.5, 122.0, 121.5, 120.5), oaspl_dB=131.4,
                alpha_bare_nominal=0.05, alpha_bare_range=(0.02, 0.10),
                kappa_nominal=0.2327, kappa_eval_range=(0.2027, 0.2627))
    with pytest.raises(S.ScenarioValidationError):        # NaN Q
        S.PhysicalScenario(**{**base, "Q_fixed": float("nan")})
    with pytest.raises(S.ScenarioValidationError):        # NaN pressure ref
        S.PhysicalScenario(**{**base, "pressure_ref_uPa": float("nan")})
    with pytest.raises(S.ScenarioValidationError):        # empty source url
        S.SourceConfig("n", "2025", "", "present", 0.6)
    with pytest.raises(S.ScenarioValidationError):        # negative mass target
        S.SyntheticBenchmarkRequest("r", (120.0,)*8, 83.78, 0.2327, 0.95, -5.0)
    with pytest.raises(S.ScenarioValidationError):        # non-positive attenuation target
        S.SyntheticBenchmarkRequest("r", (120.0,)*8, 83.78, 0.2327, -0.1, 100.0)


def test_expected_kappa_must_be_in_declared_interval():
    s = falcon()
    with pytest.raises(S.ScenarioConsistencyError):       # 0.30 is outside [0.2027, 0.2627]
        S.assert_fem_consistency(s, 165.878, s.volume, 0.30 * 165.878, expected_kappa=0.30)


def test_consistency_gate_rejects_nonfinite_inputs():
    s = falcon()
    with pytest.raises(S.ScenarioConsistencyError):
        S.assert_fem_consistency(s, float("nan"), s.volume, s.A_other_nominal, expected_kappa=s.kappa_nominal)


def test_quoted_bool_semantics_flag_rejected(tmp_path):
    """A quoted YAML string 'false' must NOT be coerced to a bool (bool('false') is True). round 4.1."""
    import yaml
    d = yaml.safe_load(CFG.read_text())
    d["semantics"]["absolute_post_treatment_spl_allowed"] = "false"   # quoted string, not a YAML bool
    p = tmp_path / "bad_flag.yaml"; p.write_text(yaml.safe_dump(d))
    with pytest.raises(S.ScenarioValidationError):
        S.load_scenario(p)


def test_quoted_is_vehicle_rejected(tmp_path):
    import yaml
    d = yaml.safe_load(CFG.read_text())
    d["is_vehicle"] = "true"                                          # quoted string, not a YAML bool
    p = tmp_path / "bad_veh.yaml"; p.write_text(yaml.safe_dump(d))
    with pytest.raises(S.ScenarioValidationError):
        S.load_scenario(p)


def test_geometry_kind_and_provenance_required():
    with pytest.raises(S.ScenarioValidationError):
        S.FairingGeometry(2.6, 6.7, 13.1, kind="exact_falcon_cad")    # not an allowed kind
    with pytest.raises(S.ScenarioValidationError):
        S.FairingGeometry(2.6, 6.7, 13.1, provenance="   ")           # empty provenance
    # Falcon is pinned to the representative kind
    assert falcon().geometry.kind == "representative_axisymmetric_cone_approximation"


def test_malformed_synthetic_request_rejected():
    # frequencies used as SPL levels (the earlier test's mistake) -> rejected by plausible-level check
    with pytest.raises(S.ScenarioValidationError):
        S.SyntheticBenchmarkRequest(request_id="bad", spl_dB=S.DESIGN_BAND_HZ, A_wall=83.78, kappa=0.2327,
                                    target_dOASPL_dB=0.95, mass_target_kg=100.0)
    # a well-formed one is accepted
    ok = S.SyntheticBenchmarkRequest(request_id="bench-0",
                                     spl_dB=(120.0, 120.0, 119.8, 120.5, 121.5, 122.0, 121.5, 120.5),
                                     A_wall=83.78, kappa=0.2327, target_dOASPL_dB=0.95, mass_target_kg=100.0,
                                     spectrum_shape_ref="falcon9-shape")
    assert ok.is_vehicle is False and falcon().is_vehicle is True
