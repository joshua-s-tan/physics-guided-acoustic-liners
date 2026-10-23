"""The public robust FEM workflow, `acoustic_liner.fem.robust.evaluate_design_robust_fem`: inputs reach the FEM
solver, the four pre-registered corners run, the robust value and its controlling corner are returned, each corner's
artifact and provenance are written, and malformed or incomplete input fails closed.

Everything is synthetic and public: the design, spectrum, wall area and kappa are the first row of the test-only
fixture `tests/fixtures/torch_oracle_rows.json` (made from a fixed seed by `tests/fixtures/make_torch_oracle_rows.py`),
and the fairing is the FEM-audit geometry rule applied to that wall area. One real FEM evaluation runs, at partial
coverage so that the placement anchors matter; the failure cases replay its solver outputs.
"""
import hashlib
import json
import math
from pathlib import Path

import pytest

from acoustic_liner.design import schema as DS
from acoustic_liner.evaluation import requests as R
from acoustic_liner.fem import placement as RP, robust as ROB, solver as FS, transfer_audit as FTA
from acoustic_liner.fem.errors import FEMSolverError
from acoustic_liner.physics import oracle as O2

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "torch_oracle_rows.json"
COVERAGE = 0.6
MESH_SIZE = 0.25


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    out = tmp_path_factory.mktemp("robust")
    row = json.loads(FIXTURE.read_text(encoding="utf-8"))["rows"][0]
    cond = dict(spl=row["spectrum"]["spl_dB"], A_wall=row["A_wall"], kappa=row["kappa_nominal"])
    scen = FTA.synthetic_scenario(cond, 0)
    design, _ = DS.design_from_decoder_vector(row["decoder_vector"])
    design["groups"] = [dict(g, cavity_volume=O2._group_cavity_volume(g)) for g in design["groups"]]
    cfg, fin = out / "scenario.json", out / "design.json"
    cfg.write_text(json.dumps(scen.to_dict(), sort_keys=True, default=str, indent=1))
    fin.write_text(json.dumps(dict(design=design), sort_keys=True, indent=1))
    calls, real = [], FS.run_scenario_fem

    def spy(scenario, freqs, **kw):
        r = real(scenario, freqs, **kw)
        calls.append(dict(kw, freqs=list(freqs), scenario=scenario, result=r))
        return r
    FS.run_scenario_fem = spy
    try:
        kw = dict(coverage=COVERAGE, alpha_bare=ROB.ALPHA_BARE_NOMINAL, out_dir=out, config_path=cfg,
                  finalists_path=fin, scenario_sha256=R.canonical_scenario_sha256(scen), mesh_size=MESH_SIZE)
        corners, robust = ROB.evaluate_design_robust_fem(scen, O2, design, **kw)
    finally:
        FS.run_scenario_fem = real
    return dict(scen=scen, design=design, kw=kw, corners=corners, robust=robust, calls=calls, cfg=cfg, fin=fin)


def test_inputs_reach_the_fem_solver(case):
    s, kw = case["scen"], case["kw"]
    treated = [c for c in case["calls"] if "anchor" in c]
    bare = [c for c in case["calls"] if "anchor" not in c]
    assert len(treated) == 4 and len(bare) == 2                    # four corners, one bare run per kappa level
    assert {(c["kappa"], c["anchor"]) for c in treated} == \
        {(k, a) for k in (min(s.kappa_eval_range), max(s.kappa_eval_range)) for a in ("low", "high")}
    assert all(c["coverage"] == kw["coverage"] and c["mesh_size"] == MESH_SIZE and c["scenario"] is s
               and c["freqs"] == list(s.band_hz) and callable(c["wall_beta"]) for c in treated)
    assert all(c["coverage"] == 1.0 for c in bare)


def test_required_corners_execute(case):
    ids = [tuple(c["corner_id"]) for c in case["corners"]]
    assert sorted(ids) == sorted(ROB.PREREGISTERED_CORNERS) and len(ids) == 4
    assert all(c["ok"] is True and math.isfinite(c["value"]) for c in case["corners"])
    by = {tuple(c["corner_id"]): c["value"] for c in case["corners"]}
    assert by[("kappa_lo", "low")] != by[("kappa_lo", "high")]       # partial coverage: the anchor matters


def test_robust_value_and_controlling_corner(case):
    """Worst case = minimum dOASPL; on an exact tie the gate picks the first tied corner in canonical order."""
    low = min(c["value"] for c in case["corners"])
    tied = sorted(tuple(c["corner_id"]) for c in case["corners"] if c["value"] == low)
    assert case["robust"]["robust_value"] == low
    assert tuple(case["robust"]["robust_corner_id"]) == tied[0]
    assert case["robust"]["tie_broken"] is (len(tied) > 1) and case["robust"]["n_corners"] == 4


def test_provenance_is_written_correctly(case):
    for c in case["corners"]:
        p = Path(c["artifact_path"])
        assert p.name == f"corner_{c['corner_id'][0]}_{c['corner_id'][1]}.json" and sha(p) == c["artifact_sha256"]
        doc = json.loads(p.read_text(encoding="utf-8"))
        assert set(doc) == {"provenance", "data"}
        prov = doc["provenance"]
        assert prov["artifact_kind"] == "fem" and prov["input_artifact_sha256"] == \
            dict(physical_scenario=sha(case["cfg"]), finalists=sha(case["fin"]))
        assert sorted(prov["source_file_sha256"]) == ["fem_boundary.py", "fem_solver.py", "scenarios.py"]
        assert doc["data"]["scenario_sha256"] == case["kw"]["scenario_sha256"] == c["fem_scenario_sha256"]
        assert doc["data"]["coverage"] == c["coverage"] and doc["data"]["kappa"] == c["kappa"]


@pytest.fixture
def replay(case, monkeypatch, tmp_path):
    """The recorded solver outputs, replayed by their arguments (no FEM), so the failure cases run in milliseconds."""
    by_key = {(c.get("kappa"), c.get("anchor"), c["coverage"]): c["result"] for c in case["calls"]}
    monkeypatch.setattr(FS, "run_scenario_fem",
                        lambda scen, freqs, **kw: by_key[(kw.get("kappa"), kw.get("anchor"), kw["coverage"])])
    return dict(case["kw"], out_dir=tmp_path)


def test_replay_reproduces_the_real_result(case, replay):
    corners, robust = ROB.evaluate_design_robust_fem(case["scen"], O2, case["design"], **replay)
    assert robust == case["robust"] and [c["value"] for c in corners] == [c["value"] for c in case["corners"]]


def test_wrong_scenario_binding_fails_closed(case, replay):
    with pytest.raises(ROB.DryRunError, match="scenario hash"):
        ROB.evaluate_design_robust_fem(case["scen"], O2, case["design"], **dict(replay, scenario_sha256="0" * 64))


def test_non_finite_corner_fails_closed(case, replay, monkeypatch):
    good = FS.run_scenario_fem

    def nan_treated(scen, freqs, **k):
        r = good(scen, freqs, **k)
        if k.get("anchor") == "high":
            r = dict(r, ms_by_freq={f: float("nan") for f in r["ms_by_freq"]})
        return r
    monkeypatch.setattr(FS, "run_scenario_fem", nan_treated)
    with pytest.raises((ROB.DryRunError, ValueError)):
        ROB.evaluate_design_robust_fem(case["scen"], O2, case["design"], **replay)


def test_ill_formed_corner_registration_fails_closed(case, replay):
    dup = (ROB.PREREGISTERED_CORNERS[0],) * 2
    with pytest.raises(RP.RobustSelectionError):
        ROB.evaluate_design_robust_fem(case["scen"], O2, case["design"], **dict(replay, corners=dup))
    with pytest.raises(RP.RobustSelectionError):
        ROB.evaluate_design_robust_fem(case["scen"], O2, case["design"], **dict(replay, corners=()))


def test_incomplete_design_fails_closed(case, replay):
    with pytest.raises(KeyError):
        ROB.evaluate_design_robust_fem(case["scen"], O2, {k: v for k, v in case["design"].items() if k != "groups"},
                                       **replay)


def test_invalid_coverage_fails_closed(case):
    with pytest.raises(FEMSolverError):
        FS.run_scenario_fem(case["scen"], list(case["scen"].band_hz), wall_beta=1.0 + 0j, coverage=1.5,
                            mesh_size=MESH_SIZE)
