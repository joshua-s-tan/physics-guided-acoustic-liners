# -*- coding: utf-8 -*-
"""Request construction: physical vehicles use the scenario-DERIVED A_wall (never randomized), the scenario
is reconstructed+revalidated (a minimal/tampered record cannot pass), requests are immutable, and the
finalist gate rejects a randomized-area vehicle finalist (§4.1, stages 3-4; round 4.1-c hardening)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner.fem import scenarios as S  # noqa: E402
from acoustic_liner.evaluation import requests as R  # noqa: E402

CFG = ROOT / "configs" / "falcon9_2025_blanketed.yaml"


def falcon():
    return S.load_scenario(CFG)


def _amax(spl_dB, kappa, A_wall):
    # deterministic stub physics ceiling (monotone in A_wall) -- stands in for baselines._achievable_max
    return 0.01 * A_wall + 2.0


def test_physical_request_uses_scenario_derived_area():
    s = falcon()
    req = R.physical_request_from_scenario(s, _amax)
    assert req.is_vehicle is True
    assert req.A_wall == pytest.approx(s.A_wall)          # 165.878, NOT a random draw
    assert req.A_wall == pytest.approx(165.878, abs=1e-3)
    assert req.kappa == pytest.approx(s.kappa_nominal)
    assert req.spl_dB == s.spl_dB
    # carries the full scenario record + a matching canonical hash (incl. semantics)
    assert req.scenario_record()["semantics"]["id"] == s.semantics.id
    assert req.scenario_sha256 == R.canonical_scenario_sha256(s)


def test_minimal_fabricated_record_rejected():
    """The exact bypass: a minimal self-hashed record with A_wall=83.78, kappa_nominal, spl_dB. It lacks a
    complete geometry/source/semantics block, so reconstruction+validation must reject it."""
    minimal = {"name": "falcon9_2025_blanketed", "A_wall": 83.78, "kappa_nominal": 0.2327,
               "spl_dB": [120.0, 120.0, 119.8, 120.5, 121.5, 122.0, 121.5, 120.5]}
    blob = json.dumps(minimal, sort_keys=True, separators=(",", ":"))
    sha = hashlib.sha256(blob.encode()).hexdigest()       # a *matching* hash cannot save an incomplete record
    with pytest.raises((R.RequestSeparationError, S.ScenarioValidationError)):
        R.PhysicalScenarioRequest(request_id="ref:falcon9", scenario_name="falcon9_2025_blanketed",
                                  achievable_max_dB=3.0, dOASPL_target_dB=1.5,
                                  scenario_json=blob, scenario_sha256=sha)


def test_injected_derived_area_is_normalized_away():
    """A COMPLETE record with injected A_wall=83.78 must NOT leave two conflicting areas in the artifact:
    the export and the stored hash are NORMALIZED to the authoritative recomputed record (round 4.1-d)."""
    s = falcon()
    rec = s.to_dict()
    rec["A_wall"] = 83.78                                  # forged derived value
    blob = R._canonical_json(rec)
    sha = hashlib.sha256(blob.encode()).hexdigest()        # self-consistent supplied (json, sha)
    req = R.PhysicalScenarioRequest(request_id="ref:falcon9", scenario_name=s.name,
                                    achievable_max_dB=3.0, dOASPL_target_dB=1.5,
                                    scenario_json=blob, scenario_sha256=sha)
    assert req.A_wall == pytest.approx(165.878, abs=1e-3)          # property recomputed
    assert req.scenario_record()["A_wall"] == pytest.approx(165.878, abs=1e-3)   # export no longer 83.78
    assert req.scenario_sha256 == R.canonical_scenario_sha256(s)  # authoritative canonical hash


def test_tampered_scenario_hash_rejected():
    s = falcon()
    blob = R._canonical_json(s.to_dict())
    with pytest.raises(R.RequestSeparationError):
        R.PhysicalScenarioRequest(request_id="ref:falcon9", scenario_name=s.name,
                                  achievable_max_dB=3.0, dOASPL_target_dB=1.5,
                                  scenario_json=blob, scenario_sha256="deadbeef")


def test_scenario_record_export_is_a_fresh_copy():
    """No caller-owned mutable state: mutating one exported record must not affect the request or another export."""
    s = falcon()
    req = R.physical_request_from_scenario(s, _amax)
    r1 = req.scenario_record(); r1["A_wall"] = -999; r1["semantics"]["id"] = "hacked"
    r2 = req.scenario_record()
    assert r2["A_wall"] != -999 and r2["semantics"]["id"] == s.semantics.id
    assert req.A_wall == pytest.approx(165.878, abs=1e-3)


def test_synthetic_request_is_randomized_and_not_a_vehicle():
    rng = np.random.default_rng(0)
    spl = (120.0, 120.0, 119.8, 120.5, 121.5, 122.0, 121.5, 120.5)
    req = R.synthetic_request(rng, spl, A_wall_range=(60.0, 110.0), kappa_range=(0.2, 0.26),
                              achievable_max_fn=_amax, request_id="interp:0", spectrum_shape_ref="falcon-shape")
    assert isinstance(req, S.SyntheticBenchmarkRequest)
    assert req.is_vehicle is False
    assert 60.0 <= req.A_wall <= 110.0


def test_request_set_is_frozen_and_uncontaminable():
    s = falcon()
    rng = np.random.default_rng(1)
    phys = [R.physical_request_from_scenario(s, _amax)]
    synth = [R.synthetic_request(rng, s.spl_dB, (60.0, 110.0), (0.2, 0.26), _amax, f"interp:{i}")
             for i in range(3)]
    rs = R.RequestSet(physical=phys, synthetic=synth)
    m = rs.to_manifest()
    assert m["n_physical"] == 1 and m["n_synthetic"] == 3 and len(m["request_set_sha256"]) == 64
    # collections are tuples -> cannot append after construction, cannot rebind (frozen dataclass)
    assert isinstance(rs.physical, tuple) and isinstance(rs.synthetic, tuple)
    with pytest.raises(AttributeError):
        rs.physical.append(synth[0])                       # tuple has no append
    with pytest.raises(Exception):
        rs.physical = (synth[0],)                          # frozen -> FrozenInstanceError
    # a synthetic cannot be smuggled into the physical list at construction either
    with pytest.raises(R.RequestSeparationError):
        R.RequestSet(physical=synth, synthetic=())


def test_finalist_gate_accepts_scenario_area_and_rejects_random_area():
    s = falcon()
    R.assert_finalist_uses_scenario_area(s.A_wall, s)                 # exact derived area -> OK
    R.assert_finalist_uses_scenario_area(s.A_wall * 1.001, s)         # within 0.5% -> OK
    with pytest.raises(R.FinalistConsistencyError):
        R.assert_finalist_uses_scenario_area(83.78, s)               # the historical randomized area -> void
