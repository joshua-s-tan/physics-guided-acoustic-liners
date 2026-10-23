# -*- coding: utf-8 -*-
"""Payload-resolving vehicle FEM: payload cases, mesh domain / connectivity, identity propagation through
finalist -> FEM -> CFD case -> CFD result -> corrected FEM, stale / cross-vehicle rejection, `python -O`."""
import copy
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from acoustic_liner.vehicles import fem as VF
from acoustic_liner.vehicles import scenarios as V

ROOT = Path(__file__).resolve().parents[2]
FREQS = [63.0, 125.0]
BETA = 0.05 + 0.0j


@pytest.fixture(scope="module")
def falcon():
    return V.falcon9_unblanketed()


@pytest.fixture(scope="module")
def atlas():
    return V.atlas_i_14ft()


def _mut(vs, fn):
    raw = copy.deepcopy(vs.raw)
    fn(raw)
    return V.VehicleScenario(raw)


# ---- payload cases ------------------------------------------------------------------------------------------------
def test_falcon_primary_solid_and_annulus_sensitivity(falcon):
    c = falcon.payload_cases
    assert falcon.payload_primary == "equal_blocked_area_solid"
    assert c["equal_blocked_area_solid"]["role"] == "primary" and c["keep_in_annulus"]["role"] == "sensitivity"
    s, a = c["equal_blocked_area_solid"], c["keep_in_annulus"]
    assert s["r_in_m"] == 0.0 and a["r_in_m"] > 0.9
    assert s["blocked_area_m2"] == pytest.approx(falcon.payload_equivalent["blocked_area_m2"], rel=1e-12)
    assert a["blocked_area_m2"] == pytest.approx(s["blocked_area_m2"], rel=1e-12)
    assert a["displaced_volume_m3"] == pytest.approx(s["displaced_volume_m3"], rel=1e-12)


def test_falcon_five_tier_placement_fixed_and_reproducible(falcon):
    for c in falcon.payload_cases.values():
        assert c["axial_extent_m"] == pytest.approx(5 * 46.0 * 0.0254, abs=1e-12)
        assert c["z_base_m"] == pytest.approx(0.429, abs=1e-12)
        assert c["z_top_m"] == pytest.approx(6.271, abs=1e-12)
        assert c["z_top_m"] <= falcon.geometry.H_barrel
    again = V.falcon9_unblanketed()
    assert {k: c["geometry_sha256"] for k, c in again.payload_cases.items()} == \
           {k: c["geometry_sha256"] for k, c in falcon.payload_cases.items()}


def test_atlas_fill_cases_geometrically_consistent(atlas):
    c = atlas.payload_cases
    assert atlas.payload_primary == "fill_055"
    R = atlas.geometry.R_fair
    for cid, r_exp in (("fill_050", 1.482), ("fill_055", 1.554), ("fill_060", 1.623)):
        x = c[cid]
        assert x["r_out_m"] == pytest.approx(r_exp, abs=5e-4)
        assert x["blocked_area_m2"] == pytest.approx(x["fill"] * math.pi * R ** 2, rel=1e-12)
        assert x["displaced_volume_m3"] == pytest.approx(x["blocked_area_m2"] * x["axial_extent_m"], rel=1e-12)
        assert x["fluid_volume_m3"] == pytest.approx(atlas.geometry.cavity_volume - x["displaced_volume_m3"], rel=1e-12)
        assert x["role"] == ("primary" if cid == "fill_055" else "sensitivity")
    assert c["fill_050"]["r_out_m"] < c["fill_055"]["r_out_m"] < c["fill_060"]["r_out_m"]
    assert c["fill_055"]["axial_extent_m"] == pytest.approx(0.8 * atlas.geometry.H_barrel, rel=1e-12)


@pytest.mark.parametrize("mutate", [
    lambda d: d["payload_model"]["cases"]["equal_blocked_area_solid"]["r_out_m"].update(value=2.1),
    lambda d: d["payload_model"]["z_base_m"].update(value=0.5),
    lambda d: d["payload_model"]["surface_condition"].update(value="absorbing"),
    lambda d: d["payload_model"]["cases"]["keep_in_annulus"]["role"].update(value="primary"),
    lambda d: d["payload_model"]["primary_case"].update(value="keep_in_annulus"),
    lambda d: d.pop("payload_model"),
])
def test_falcon_payload_model_fails_closed(falcon, mutate):
    with pytest.raises(V.ScenarioValidationError):
        _mut(falcon, mutate)


def test_atlas_fill_outside_published_range_rejected(atlas):
    def m(d):
        c = d["payload_model"]["cases"]["fill_060"]
        c["fill"]["value"] = 0.65
        c["r_out_m"]["value"] = 2.0955 * math.sqrt(0.65)
    with pytest.raises(V.ScenarioValidationError):
        _mut(atlas, m)


# ---- mesh domain ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("which,case", [("falcon", "equal_blocked_area_solid"), ("falcon", "keep_in_annulus"),
                                        ("atlas", "fill_050"), ("atlas", "fill_055"), ("atlas", "fill_060")])
def test_mesh_excludes_payload_and_stays_connected(which, case, request):
    vs = request.getfixturevalue(which)
    c = vs.payload_cases[case]
    mesh = VF.build_vehicle_mesh(vs.geometry, c, 0.35)
    assert VF.fluid_components(mesh) == 1
    cen = mesh.p[:, mesh.t].mean(axis=1)
    inside = ((cen[0] > c["r_in_m"] + 1e-9) & (cen[0] < c["r_out_m"] - 1e-9)
              & (cen[1] > c["z_base_m"] + 1e-9) & (cen[1] < c["z_top_m"] - 1e-9))
    assert not inside.any()
    from acoustic_liner.fem import mesh as MESH
    assert MESH.mesh_physical_volume(mesh) == pytest.approx(c["fluid_volume_m3"], rel=1e-9)
    tagged = VF.tag_vehicle_facets(mesh, vs.geometry)
    pay = sum(tagged["area_by_id"][f] for f in tagged["sets"]["payload"])
    assert pay == pytest.approx(c["surface_area_m2"], rel=1e-9)


def test_payload_changes_the_field(falcon, atlas):
    s = VF.run_vehicle_fem(falcon, FREQS, BETA)
    a = VF.run_vehicle_fem(falcon, FREQS, BETA, payload_case="keep_in_annulus")
    lo = VF.run_vehicle_fem(atlas, FREQS, BETA, payload_case="fill_050")
    hi = VF.run_vehicle_fem(atlas, FREQS, BETA, payload_case="fill_060")
    for f in FREQS:
        assert abs(math.log10(s["ms_by_freq"][f] / a["ms_by_freq"][f])) > 1e-3
        assert abs(math.log10(lo["ms_by_freq"][f] / hi["ms_by_freq"][f])) > 1e-3


def test_fem_deterministic(atlas):
    r1, r2 = VF.run_vehicle_fem(atlas, FREQS, BETA), VF.run_vehicle_fem(atlas, FREQS, BETA)
    assert r1["mesh_sha256"] == r2["mesh_sha256"] and r1["ms_by_freq"] == r2["ms_by_freq"]
    assert V.record_sha256(r1) == V.record_sha256(r2)


def test_legacy_scenario_rejected(falcon):
    with pytest.raises(VF.VehicleFEMError):
        VF.run_vehicle_fem(falcon.to_physical_scenario(), FREQS, BETA)


# ---- identity propagation ------------------------------------------------------------------------------------------
def _chain(vs, case=None, corner=None):
    ident = vs.identity(corner, case)
    fin = V.stamp({"geometry_hash": "g" * 64, "coverage": 0.5}, ident, "finalist")
    fem = VF.run_vehicle_fem(vs, FREQS, BETA, corner_id=corner, payload_case=case, parent=fin)
    cc = V.stamp({"case_hash": "c" * 64}, ident, "cfd_case", fem)
    cr = V.stamp({"impedance": [1.0, 2.0]}, ident, "cfd_result", cc)
    cf = VF.run_vehicle_fem(vs, FREQS, BETA, corner_id=corner, payload_case=case, stage="corrected_fem", parent=cr)
    return [fin, fem, cc, cr, cf]


@pytest.mark.parametrize("which", ["falcon", "atlas"])
def test_identity_propagates_through_every_stage(which, request):
    vs = request.getfixturevalue(which)
    recs = _chain(vs)
    assert V.verify_lineage(vs, recs)
    for r in recs:
        assert r["scenario_hash"] == vs.scenario_hash
        assert r["payload_geometry_sha256"] == vs.payload_cases[vs.payload_primary]["geometry_sha256"]
        assert r["payload_bc_sha256"] == vs.payload_bc["sha256"]


def test_cross_vehicle_and_stale_rejected(falcon, atlas):
    recs = _chain(falcon)
    with pytest.raises(V.ScenarioMixError):
        V.verify_lineage(atlas, recs)
    stale = _mut(falcon, lambda d: d["acoustics"]["Q_fixed"].update(value=14.0))
    with pytest.raises(V.ScenarioMixError):
        V.verify_lineage(stale, recs)
    with pytest.raises(V.ScenarioMixError):                   # FEM refuses an Atlas parent for a Falcon run
        VF.run_vehicle_fem(falcon, FREQS, BETA, parent=_chain(atlas)[0])


def test_wrong_case_tampered_parent_missing_identity_rejected(falcon):
    recs = _chain(falcon)
    with pytest.raises(V.ScenarioMixError):
        falcon.require_identity(recs[1], payload_case="keep_in_annulus")
    bad = [dict(r) for r in recs]
    bad[2]["case_hash"] = "x" * 64                             # edit a record after its child was stamped
    with pytest.raises(V.ScenarioMixError):
        V.verify_lineage(falcon, bad)
    with pytest.raises(V.ScenarioMixError):
        falcon.require_identity({"scenario_name": falcon.name, "scenario_hash": falcon.scenario_hash})
    with pytest.raises(V.ScenarioMixError):
        V.verify_lineage(falcon, [recs[1], recs[0]])


def test_gates_survive_python_O():
    code = (
        "import copy, sys\n"
        "from acoustic_liner.vehicles import scenarios as V\n"
        "f = V.falcon9_unblanketed(); a = V.atlas_i_14ft()\n"
        "rec = V.stamp({}, f.identity(), 'finalist')\n"
        "try:\n    a.require_identity(rec); sys.exit('cross-vehicle accepted')\nexcept V.ScenarioMixError: pass\n"
        "raw = copy.deepcopy(f.raw); raw['payload_model']['z_base_m']['value'] = 0.5\n"
        "try:\n    V.VehicleScenario(raw); sys.exit('bad placement accepted')\nexcept V.ScenarioValidationError: pass\n"
        "assert False, 'asserts must be stripped'\n")
    env = dict(__import__("os").environ, PYTHONPATH=str(ROOT / "src"))
    p = subprocess.run([sys.executable, "-O", "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr + p.stdout
