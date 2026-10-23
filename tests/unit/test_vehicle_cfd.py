# -*- coding: utf-8 -*-
"""Vehicle CFD bundles: real-parent requirement, identity lineage, demanding-corner velocity source, immutable
plans, resume status, returned-results tamper detection and cross-vehicle refusal."""
import json
import shutil

import numpy as np
import pytest

from acoustic_liner.design import schema as DS
from acoustic_liner.physics import scorer as PS
from acoustic_liner.vehicles import cfd_bundle as VC
from acoustic_liner.vehicles import search as VP
from acoustic_liner.vehicles import scenarios as V


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    base = tmp_path_factory.mktemp("veh")
    mp.setattr(VP, "RUN_DIR", base)
    mp.setattr(VP, "MESH_SIZES", (0.35,))
    vs = VP.scenario("falcon")
    cond = VP.condition(vs, "west_range", 400.0)
    rng = np.random.default_rng(7)
    while True:
        d = DS.vector_to_design(DS.sample_random(rng))
        q = PS.quality_eval_exact(d, cond)
        if q["feasible_budget"] and 0.2 < q["best_cov"] < 0.9:
            break
    fin = V.stamp(dict(study="vehicle_physgen_v1_search", design=d, design_sha256=VP._sha(d), geometry_hash="g" * 64,
                       coverage=q["best_cov"], mass_kg=q["mass"], status="finalist", **V.finalist_coverage(vs)),
                  vs.identity("west_range"), "finalist")
    (base / vs.name).mkdir()
    (base / vs.name / "finalist.json").write_text(json.dumps(fin, default=float))
    fin = VP.load_finalist(vs)
    yield mp, base, vs, fin
    mp.undo()


def _fem(env):
    mp, base, vs, fin = env
    r = VP.four_corner_fem(vs, fin, "west_range", vs.payload_primary, 0.35)
    (base / vs.name / "initial_fem.json").write_text(json.dumps(
        dict(finalist_sha256=V.record_sha256(fin), records=[r]), default=float))
    return r


def test_bundle_refuses_without_initial_fem(env, tmp_path):
    with pytest.raises(VC.BundleError):
        VC.bundle("falcon", allow_dirty=True, out=tmp_path / "b")


@pytest.fixture(scope="module")
def bundle(env, tmp_path_factory):
    _fem(env)
    out = tmp_path_factory.mktemp("bundle") / "b"
    return out, VC.bundle("falcon", allow_dirty=True, out=out, cycles=2)


def test_bundle_lineage_and_identity(env, bundle):
    _, _, vs, fin = env
    out, rec = bundle
    vs2, fin2, fem, rec2 = VC.load_bundle(out)
    assert rec2 == rec and V.verify_lineage(vs, [fin, fem, rec])
    assert rec["velocity_source_corner"] == "west_range"                  # higher 50-250 Hz OASPL than East
    assert set(rec["both_source_corners"]) == {"west_range", "east_range"}
    assert rec["payload_geometry_sha256"] == vs.payload_cases[vs.payload_primary]["geometry_sha256"]
    assert rec["n_expected"] == sum(len(v) for v in rec["expected_cases"].values()) > 0
    for f in ("run_cases.sh", "run_one.sh", "_bundle.json", "_parent_fem.json", "cfd_finalist.json"):
        assert (out / f).is_file()
    assert b"\r\n" not in (out / "run_one.sh").read_bytes()


def test_bundle_immutable_and_plan_tamper_detected(env, bundle, tmp_path):
    out, _ = bundle
    with pytest.raises(VC.BundleError):
        VC.bundle("falcon", allow_dirty=True, out=out)
    cp = tmp_path / "copy"
    shutil.copytree(out, cp)
    plan = cp / "nominal" / "_cfd_case_plan.json"
    plan.write_text(plan.read_text().replace("west_range", "east_range"))
    with pytest.raises(VC.BundleError):
        VC.load_bundle(cp)


def test_cross_vehicle_bundle_refused(bundle, tmp_path):
    out, rec = bundle
    cp = tmp_path / "x"
    shutil.copytree(out, cp)
    (cp / "_bundle.json").write_text(json.dumps(dict(rec, vehicle="atlas"), default=float))
    with pytest.raises((V.ScenarioMixError, VP.VehiclePipelineError, FileNotFoundError)):
        VC.load_bundle(cp)


def test_status_lists_every_case_as_runnable(bundle):
    out, rec = bundle
    st = VC.status(out)["nominal"]
    assert st["summary"] == {"missing": rec["groups"]["nominal"]["n_cases"]}
    assert len(st["resume"]["shell"]) == rec["groups"]["nominal"]["n_cases"] and not st["resume"]["done"]


def test_returned_groups_nominal_only(bundle, tmp_path):
    out, rec = bundle
    cp = tmp_path / "n"
    shutil.copytree(out, cp)
    assert VC.returned_manifest(cp, ["nominal"])["groups"] == ["nominal"]
    for bad in (["sens_refine"], ["nominal", "sens_unknown"], []):
        with pytest.raises(VC.BundleError):
            VC.returned_manifest(cp, bad)


def test_returned_results_tamper_detected(bundle, tmp_path):
    out, rec = bundle
    cp = tmp_path / "r"
    shutil.copytree(out, cp)
    VC.returned_manifest(cp)
    case = rec["expected_cases"]["nominal"][0]["name"]
    (cp / "nominal" / case / "system" / "controlDict").write_text("edited after return")
    with pytest.raises(VC.BundleError):
        VC.aggregate(cp)
