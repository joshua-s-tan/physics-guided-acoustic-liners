# -*- coding: utf-8 -*-
"""Targeted CFD sensitivity: the preregistered selection rules, envelope perturbation and escalation triggers."""
import pytest

from acoustic_liner.provenance import schemas as SC

from acoustic_liner.vehicles import cfd_targeted as T

ROWS = [("b1_u2", 2.0, 1.9), ("b1_u6", 6.0, 5.8), ("b1_u12", 12.0, 11.5), ("b1_u20", 20.0, 19.0), ("b1_u33", 33.0, 31.0)]


def test_velocities_low_ustar_high():
    assert T.pick_velocities(ROWS, 11.0) == [("low", "b1_u2"), ("ustar", "b1_u12"), ("high", "b1_u33")]


@pytest.mark.parametrize("u_star,expect", [
    (1.0, [("low", "b1_u6"), ("ustar", "b1_u2"), ("high", "b1_u33")]),       # U* case = lowest -> low moves inward
    (40.0, [("low", "b1_u2"), ("ustar", "b1_u33"), ("high", "b1_u20")]),     # U* case = highest -> high moves inward
])
def test_collision_moves_slot_inward(u_star, expect):
    assert T.pick_velocities(ROWS, u_star) == expect


def test_few_velocities_uses_all_with_one_ustar():
    p = T.pick_velocities(ROWS[:2], 6.0)
    assert sorted(n for _, n in p) == ["b1_u2", "b1_u6"] and dict(p)["ustar"] == "b1_u6"


def test_bank_rule_largest_with_low_index_tie():
    assert T.pick_bank([0.1, 0.3, 0.2, 0.05]) == 1
    assert T.pick_bank([0.1, 0.3, 0.3005, 0.05]) == 1


def test_controlling_forcing_lower_fem_ties_to_tone():
    assert T.controlling_forcing(dict(broadband=3.0, resonant_tone=3.2)) == "broadband"
    assert T.controlling_forcing(dict(broadband=3.0, resonant_tone=3.0005)) == "resonant_tone"
    assert T.controlling_forcing(dict(broadband=3.4, resonant_tone=3.2)) == "resonant_tone"


def test_perturbation_keeps_anchor_and_shifts_others():
    rx = [[(0.4, 10.0, 2.0), (6.0, 12.0, 3.0)]]
    out = T.perturb_tables(rx, [(0.4, 10.0, 2.0)], lambda j, u, r, x: 100.0, 0.02, 0.01, 1, -1)
    assert out[0][0] == (0.4, 10.0, 2.0) and out[0][1] == pytest.approx((6.0, 14.0, 2.0))


def test_sources_substitute_operating_point_reruns(tmp_path):
    import hashlib
    import json
    b, r = tmp_path / "b", tmp_path / "rerun"
    (b / "nominal").mkdir(parents=True)
    (r / "nominal_24c").mkdir(parents=True)
    mk = lambda n, h: dict(name=n, bank_id="b1", u_target=6.0, case_hash="c", input_tree_hash=h)
    (b / "nominal" / "_cfd_case_plan.json").write_text(json.dumps(dict(geometry_hash="g", cases=[mk("b1_u6", "old"), mk("b1_u2", "keep")])))
    (r / "nominal_24c" / "_cfd_case_plan.json").write_text(json.dumps(dict(geometry_hash="g", cases=[mk("b1_u6", "new")])))
    (r / "_rerun.json").write_text("{}")
    sha = hashlib.sha256(b"{}").hexdigest()
    src = T._sources(dict(bdir=b, cr=dict(operating_point_rerun=dict(rerun_dir="../rerun", rerun_sha256=sha))))
    assert src["b1_u6"][1]["input_tree_hash"] == "new" and src["b1_u6"][0] == (r / "nominal_24c" / "b1_u6").resolve()
    assert src["b1_u2"][1]["input_tree_hash"] == "keep"
    with pytest.raises(T.TargetedError):
        T._sources(dict(bdir=b, cr=dict(operating_point_rerun=dict(rerun_dir="../rerun", rerun_sha256="0" * 64))))


def test_triggers():
    ok = dict(fem_range_db=0.05, max_norm_change=0.05, ustar_shift=0.05, verdicts=[True, True])
    assert T.triggers(**ok) == []
    for k, v in (("fem_range_db", 0.11), ("max_norm_change", 0.11), ("ustar_shift", 0.11), ("verdicts", [True, False])):
        assert len(T.triggers(**dict(ok, **{k: v}))) == 1


# ---- prereg amendment 1: five-case operating-region protocol ------------------------------------------------
PICKS = [("low", "b0_u2"), ("ustar", "b0_u6"), ("high", "b0_u25.8")]
LEGACY = [dict(group=g, name=n, slot=s) for s, n in PICKS for g in ("sens_refine", "sens_halfdt")] + \
         [dict(group="sens_refine_halfdt", name="b0_u6", slot="ustar")]


def test_five_case_protocol_retains_operating_region_cases():
    got = [(c["group"], c["slot"]) for c in T.protocol_cases(PICKS)]
    assert sorted(got) == sorted(T.FIVE_CASE) and len(got) == 5
    keep, drop = T.split_legacy(LEGACY)
    assert keep == T.protocol_cases(PICKS)
    assert [(d["group"], d["name"]) for d in drop] == [("sens_refine", "b0_u25.8"), ("sens_refine_halfdt", "b0_u6")]


def _legacy_tdir(tmp_path):
    import json
    t = tmp_path / "cfd_targeted"
    for c in LEGACY:
        (t / c["group"] / c["name"]).mkdir(parents=True)
    (t / "sens_halfdt" / "b0_u2" / "_status.json").write_text('{"state": "done", "rc": 0}')
    (t / "sens_refine" / "b0_u25.8" / "_status.json").write_text('{"state": "running"}')
    (t / "_targeted_selection.json").write_text(json.dumps(dict(cases=LEGACY, picks=PICKS)))
    return t


def test_seven_case_selection_needs_bound_amendment(tmp_path):
    t = _legacy_tdir(tmp_path)
    with pytest.raises(T.TargetedError):
        T.retained_cases(t)
    am = T.amend(t)
    assert am["completed_before_amendment"] == ["sens_halfdt/b0_u2"] and am["numerical_outputs_inspected"] is False
    assert am["schema"] == SC.schema("cfd-targeted-amendment") and "prereg_sha256" not in am
    assert am["state_at_amendment"]["sens_refine_halfdt/b0_u6"] == "not started" and len(am["dropped"]) == 2
    cases, sha = T.retained_cases(t)
    assert cases == T.split_legacy(LEGACY)[0] and sha
    script = (t / "run_targeted.sh").read_text()
    assert "b0_u25.8" in script and "sens_refine/b0_u25.8" not in script and "sens_refine_halfdt" not in script
    with pytest.raises(T.TargetedError):
        T.amend(t)                                                                  # immutable
    (t / "_targeted_selection.json").write_text((t / "_targeted_selection.json").read_text() + " ")
    with pytest.raises(T.TargetedError):
        T.retained_cases(t)                                                         # selection edited after binding


def test_five_case_selection_needs_no_amendment(tmp_path):
    import json
    t = tmp_path / "t"
    t.mkdir()
    (t / "_targeted_selection.json").write_text(json.dumps(dict(protocol=T.PROTOCOL, cases=T.protocol_cases(PICKS))))
    assert T.retained_cases(t) == (T.protocol_cases(PICKS), None)
    with pytest.raises(T.TargetedError):
        T.amend(t)


@pytest.mark.parametrize("name,legacy", [("cfd-targeted-selection", "prereg_sha256"),
                                         ("cfd-targeted-amendment", "prereg_sha256"),
                                         ("cfd-targeted-result", "prereg_sha256"),
                                         ("statistical-impedance-estimate", "protocol_sha256"),
                                         ("fem-audit-seal", "record_sha256"),
                                         ("fem-field-manifest", "historical_source_sha256")])
def test_both_record_formats_are_read_and_nothing_else(name, legacy):
    """Version 1 (written during the study: no schema field, the fields of that version) and version 2 (schema field)
    are read; a record with neither, or another schema, is refused."""
    assert SC.record_version({legacy: "0" * 64}, name) == 1
    assert SC.record_version(dict(schema=SC.schema(name)), name) == 2
    with pytest.raises(SC.SchemaError):
        SC.record_version({}, name)
    with pytest.raises(SC.SchemaError):
        SC.record_version(dict(schema=SC.schema(name)[:-1] + "9"), name)


@pytest.mark.parametrize("name", ["vehicle-verdict", "falcon-supplemental-verdict", "vehicle-statistical-followup"])
def test_verdict_records_version_2_and_their_earlier_identifiers(name):
    assert SC.schema(name).endswith("/2")
    assert SC.record_version(dict(schema=SC.schema(name)), name) == 2
    for old in SC.EARLIER[name]:
        assert SC.record_version(dict(schema=old), name) == 1
    with pytest.raises(SC.SchemaError):
        SC.record_version(dict(schema="acoustic-liner/vehicle-verdict/3"), name)
