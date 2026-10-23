# -*- coding: utf-8 -*-
"""Vehicle pipeline wiring: approved-hash pin, mass rule, condition mapping, selection rule, near-threshold mixture,
finalist refusal of placeholders / tampering, and the payload-resolving four-corner FEM descending from a finalist."""
import json

import numpy as np
import pytest

from acoustic_liner.design import schema as DS
from acoustic_liner.fem import robust as DRUN
from acoustic_liner.physics import scorer as PS
from acoustic_liner.vehicles import search as VP
from acoustic_liner.vehicles import scenarios as V


@pytest.fixture(scope="module")
def atlas():
    return VP.scenario("atlas")


@pytest.fixture(scope="module")
def falcon():
    return VP.scenario("falcon")


def test_approved_hash_pinned(monkeypatch):
    monkeypatch.setitem(VP.APPROVED_SCENARIO_HASHES, "atlas", "0" * 64)
    with pytest.raises(VP.VehiclePipelineError):
        VP.scenario("atlas")


def test_atlas_mass_limit_scales_with_wall_area(atlas, falcon):
    m = VP.mass_limit("atlas", atlas)
    assert m["m_total_kg"] == pytest.approx(400.0 * atlas.geometry.lined_wall_area / falcon.geometry.lined_wall_area)
    assert m["m_total_kg"] == pytest.approx(285.03, abs=0.01)
    assert VP.mass_limit("falcon", falcon)["m_total_kg"] == 400.0


def test_condition_uses_scenario_levels(falcon):
    c = VP.condition(falcon, "east_range", 400.0)
    assert c["spl"] == falcon.model_levels("east_range")
    assert c["A_wall"] == pytest.approx(falcon.geometry.lined_wall_area)
    assert c["kappa"] == falcon.acoustics["kappa_nominal"] and c["dOASPL_target"] == VP.TARGET_DB


def test_selection_rule_and_tie_breaks():
    runs = [dict(seed=0, best_val=1.0, mass=300, coverage=0.5, geometry_hash="b"),
            dict(seed=1, best_val=1.0, mass=300, coverage=0.5, geometry_hash="a"),
            dict(seed=2, best_val=1.0, mass=250, coverage=0.9, geometry_hash="z"),
            dict(seed=3, best_val=None)]
    assert VP.select(runs)["seed"] == 2
    runs[2]["mass"] = 300
    runs[2]["coverage"] = 0.6
    assert VP.select(runs)["seed"] == 1
    with pytest.raises(VP.VehiclePipelineError):
        VP.select([dict(seed=0, best_val=None)])


def test_near_threshold_mixture_is_half_generator():
    calls = []
    gen = lambda rng, n: calls.append(n) or [("g", i) for i in range(n)]
    out = VP.mixture_fn(gen)(np.random.default_rng(0), 32)
    assert calls == [16] and len(out) == 32 and out[:16] == [("g", i) for i in range(16)]


@pytest.fixture(scope="module")
def finalist(atlas):
    cond = VP.condition(atlas, "qualification", VP.mass_limit("atlas", atlas)["m_total_kg"])
    rng = np.random.default_rng(5)
    while True:
        d = DS.vector_to_design(DS.sample_random(rng))
        q = PS.quality_eval_exact(d, cond)
        if q["feasible_budget"] and 0.2 < q["best_cov"] < 0.9:
            break
    rec = dict(study="vehicle_physgen_v1_search", design=d, design_sha256=VP._sha(d), geometry_hash="g" * 64,
               coverage=q["best_cov"], mass_kg=q["mass"], status="test", **V.finalist_coverage(atlas))
    return V.stamp(rec, atlas.identity("qualification"), "finalist")


def test_load_finalist_refuses_placeholder_and_tampering(atlas, finalist, tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps(finalist, default=float))
    assert VP.load_finalist(atlas, p)["design_sha256"] == finalist["design_sha256"]
    for bad in (dict(finalist, study="placeholder"), dict(finalist, design_sha256="0" * 64),
                {k: v for k, v in finalist.items() if k != "scenario_hash"}):
        p.write_text(json.dumps(bad, default=float))
        with pytest.raises((VP.VehiclePipelineError, V.ScenarioMixError)):
            VP.load_finalist(atlas, p)


def test_four_corner_fem_descends_from_finalist(atlas, falcon, finalist):
    recs = [VP.four_corner_fem(atlas, finalist, c, case, 0.35)
            for c, case in (("qualification", "fill_055"), ("acceptance", "fill_060"))]
    for r in recs:
        assert {tuple(c["corner_id"]) for c in r["corners"]} == set(DRUN.PREREGISTERED_CORNERS)
        assert r["robust"]["robust_value"] == pytest.approx(min(c["value"] for c in r["corners"]), abs=1e-12)
        assert r["payload"]["geometry_sha256"] == atlas.payload_cases[r["payload_case"]]["geometry_sha256"]
        assert V.verify_lineage(atlas, [finalist, r])
        assert "may include payload-associated losses" in r["a_other_note"]
    with pytest.raises(V.ScenarioMixError):                 # Atlas finalist cannot parent a Falcon FEM
        VP.four_corner_fem(falcon, finalist, "west_range", "equal_blocked_area_solid", 0.35)
