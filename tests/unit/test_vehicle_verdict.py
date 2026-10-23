# -*- coding: utf-8 -*-
"""Vehicle verdict: preregistered conservative interval, PASS/FAIL/UNRESOLVED, fail-closed evidence, immutable
publication rendered from the JSON."""
import json
import os
import shutil
import stat

import pytest

from acoustic_liner.vehicles import search as VP
from acoustic_liner.vehicles import verdict as VV

T = VP.TARGET_DB


def fem(base, d02=0.02, d015=0.01, spread=0.1):
    return {f: {0.2: b + d02, 0.15: b + d015, 0.1: b} for f, b in (("broadband", base + spread), ("resonant_tone", base))}


def test_interval_formula():
    iv = VV.interval(fem(3.0), 0.2, 0.1)
    assert iv["F01"] == 3.0 and iv["F01_hi"] == pytest.approx(3.1) and iv["M"] == pytest.approx(0.02)
    assert iv["L"] == pytest.approx(3.0 - 0.02 - 0.2) and iv["U"] == pytest.approx(3.1 + 0.02 + 0.1)


def test_mesh_allowance_uses_largest_difference_either_sign():
    f = fem(3.0)
    f["broadband"][0.15] = 3.1 - 0.05
    assert VV.interval(f, 0, 0)["M"] == pytest.approx(0.05)


def test_missing_mesh_value_gives_no_interval():
    f = fem(3.0)
    del f["broadband"][0.15]
    assert VV.interval(f, 0, 0) is None and VV.decide(None, []) == VV.UNRESOLVED


def test_complete_synthetic_pass_and_fail():
    assert VV.decide(VV.interval(fem(3.0), 0.2, 0.1), []) == VV.PASS
    assert VV.decide(VV.interval(fem(0.5), 0.1, 0.1), []) == VV.FAIL


def test_interval_crossing_target_is_unresolved():
    iv = VV.interval(fem(T + 0.05), 0.2, 0.1)
    assert iv["L"] < T < iv["U"] and VV.decide(iv, []) == VV.UNRESOLVED


@pytest.mark.parametrize("reason", ["unresolved banks: ['b2']", "targeted sensitivity escalation triggered: [...]",
                                    "corrected FEM missing", "returned file changed after its manifest"])
def test_any_reason_blocks_a_pass(reason):
    assert VV.decide(VV.interval(fem(5.0), 0.0, 0.0), [reason]) == VV.UNRESOLVED


def test_sensitivity_allowance():
    tr = dict(fem_nominal_0p1=3.0, fem_envelope_combos_0p1=[dict(fem_robust_db=2.9), dict(fem_robust_db=3.05),
                                                          dict(fem_robust_db=None), dict(fem_robust_db=2.95)])
    assert VV.sensitivity_allowance(tr) == pytest.approx((0.1, 0.05))


def test_collect_missing_or_wrong_vehicle_evidence_is_unresolved(tmp_path):
    ev, reasons = VV.collect("falcon", tmp_path / "nope", tmp_path / "nope2")
    assert reasons and "failed verification" in reasons[0]


def _verdict(outcome="PASS"):
    return dict(schema=VV.SCHEMA, vehicle="falcon", scenario_name="s", outcome=outcome, target_db=T,
                primary=dict(source_corner="west_range", payload_case="solid", interval=VV.interval(fem(3.0), 0.2, 0.1)),
                reported_not_verdict={"east_range|annulus": dict(VV.interval(fem(0.8), 0.2, 0.1), below_target=True)},
                reasons=[], evidence=dict(targeted=dict(bank=3)), wording=VV.WORDING, record_sha256="r")


def test_render_from_json_claim_discipline():
    md = VV.render(_verdict())
    assert "**PASS**" in md and "Simulation-based" in md and "not formal grid convergence" in md
    assert "targeted operating-region mesh/time-step sensitivity" in md and "not full-range CFD convergence" in md
    assert "High-speed mesh sensitivity is not quantified" in md and "mesh-time-step interaction is not quantified" in md
    assert "not a proven global bound over the entire velocity range" in md
    assert "may include payload-associated losses" in md and "partial-coverage" in md
    low = md.lower().replace("not experimentally validated", "").replace("not flight-qualified", "")
    assert "experimentally validated" not in low and "flight-qualified" not in low
    assert "| `east_range|annulus` |" in md and "**yes**" in md


def test_publish_immutable_atomic_readonly(tmp_path):
    out = tmp_path / "verdict"
    VV.publish(_verdict(), out)
    js = json.loads((out / "verdict.json").read_text())
    assert (out / "VERDICT.md").read_text() == VV.render(js)                       # rendered only from the JSON
    assert not os.stat(out / "verdict.json").st_mode & stat.S_IWRITE
    VV.publish(_verdict(), out)                                                     # identical: no-op
    with pytest.raises(VV.VerdictError):
        VV.publish(_verdict("FAIL"), out)
    assert not list(out.glob("*.tmp"))
    for p in out.iterdir():
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
    shutil.rmtree(out)


# ---- fail-closed provenance on a real (synthetic-design) bundle ----------------------------------------------
from test_vehicle_cfd import bundle, env  # noqa: E402,F401  (module fixtures: finalist + initial FEM + CFD bundle)


def test_collect_on_bundle_without_results_lists_missing_evidence(env, bundle, tmp_path):
    out, _ = bundle
    _, reasons = VV.collect("falcon", out, tmp_path / "no_targeted")
    assert any("returned CFD results" in r for r in reasons) and any("CFD result missing" in r for r in reasons)


def test_collect_wrong_vehicle_and_edited_parent(env, bundle, tmp_path):
    out, _ = bundle
    _, reasons = VV.collect("atlas", out, tmp_path / "t")
    assert reasons
    cp = tmp_path / "edited"
    shutil.copytree(out, cp)
    p = cp / "_parent_fem.json"
    d = json.loads(p.read_text()); d["mesh_size"] = 0.5
    p.write_text(json.dumps(d))
    _, reasons = VV.collect("falcon", cp, tmp_path / "t")
    assert reasons and VV.decide(VV.interval(fem(5.0), 0, 0), reasons) == VV.UNRESOLVED


def test_stopped_rerun_merge_is_named_as_the_reason(env, bundle, tmp_path):
    out, _ = bundle
    b = tmp_path / "veh" / "bundle"
    shutil.copytree(out, b)
    (tmp_path / "veh" / "cfd_rerun_24c").mkdir()
    (tmp_path / "veh" / "cfd_rerun_24c" / "_merge_report.json").write_text(json.dumps(dict(
        rerun_states={"b1_u12": "needs_extend"}, banks=dict(b0=dict(status="adequate"), b1=dict(status="needs_rerun")))))
    ev, reasons = VV.collect("falcon", b, tmp_path / "t")
    assert any("unresolved banks ['b1']" in r for r in reasons)
    assert ev["rerun_merge_reports"]["cfd_rerun_24c"]["unresolved_banks"] == ["b1"]
