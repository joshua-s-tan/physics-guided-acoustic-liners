"""The active CFD workflow, in the form the vehicle pipeline calls it: generic bank ids, the default plan file and a
scenario-bound finalist record.

These tests replace historical tests of code that is not part of this package (an older wrapper around the case
generator, a runner state file, a propagation step and a self-test). They test the current behaviour; they do not
reproduce the omitted code."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
FINALIST = ROOT / "finalists" / "falcon_v2.1_finalist_geometry.json"


@pytest.fixture(autouse=True)
def _restore_case_globals():
    """generate() assigns the neck geometry on cfd.case by design; restore it so later tests see the module as
    imported."""
    from acoustic_liner.cfd import case
    saved = dict(vars(case))
    yield
    for k in [k for k in vars(case) if k not in saved]:
        delattr(case, k)
    for k, v in saved.items():
        setattr(case, k, v)


def _vehicle_plan(tmp_path, **kw):
    """gen_cases in the vehicle pipeline's call form (vehicle_cfd.bundle): a scenario-bound finalist record,
    generic bank ids, the default plan file."""
    from acoustic_liner.cfd import case_generation as CG
    from acoustic_liner.fem import scenarios as SC
    fin = json.loads(FINALIST.read_text(encoding="utf-8"))
    rec = tmp_path / "cfd_finalist.json"
    rec.write_text(json.dumps(dict(geometry=fin["geometry"], coverage=fin["coverage"], scenario="vehicle_test"),
                              sort_keys=True, indent=1, default=float))
    out = tmp_path / "nominal"
    plan = CG.gen_cases(str(out), scenario="vehicle_test", source_corner="kappa_hi_low", finalist_path=str(rec),
                        spectrum=list(SC.falcon9_2025_blanketed().spl_dB), cycles=4, **kw)
    return plan, out


def test_gen_cases_writes_the_manifest_the_runner_loads(tmp_path):
    """The plan written by gen_cases is the manifest the runner loads; with nothing run, every case is missing."""
    from acoustic_liner.cfd import runner as RUN
    plan, out = _vehicle_plan(tmp_path)
    assert plan["n_cases"] == 28 and (out / "_cfd_case_plan.json").exists()
    man = RUN.load_manifest(str(out))
    assert man["scenario"] == "vehicle_test" and man["cases"] == json.loads(json.dumps(plan["cases"]))
    assert {c["bank_id"] for c in man["cases"]} == {"b0", "b1", "b2", "b3"}
    st = RUN.scan_status(man, str(out))                    # nothing ran: every case MISSING, fail closed
    assert st["summary"] == {RUN.MISSING: 28} and not st["all_valid"]
    assert len(RUN.resume_manifest(st, str(out))["shell"]) == 28


def test_manifest_binding_fails_on_meta_tamper(tmp_path):
    """The per-bank impedance table binds to the manifest: an edited case record is refused."""
    from acoustic_liner.cfd import impedance_correction as RL
    plan, out = _vehicle_plan(tmp_path)
    man = dict(geometry_hash=plan["geometry_hash"], cases=plan["cases"])
    per_bank, anchors, reports = RL.build_per_bank_rx(man, str(out), 4)
    assert len(per_bank) == 4 and len(reports) == len(plan["cases"])
    mp = out / plan["cases"][0]["name"] / "_case_meta.json"
    meta = json.loads(mp.read_text()); meta["f0_Hz"] += 5.0
    mp.write_text(json.dumps(meta))
    with pytest.raises(RL.ProvenanceError):
        RL.build_per_bank_rx(man, str(out), 4)


def test_manifest_binding_fails_on_input_tree_edit(tmp_path):
    """An edited input file of a case is refused."""
    from acoustic_liner.cfd import impedance_correction as RL
    plan, out = _vehicle_plan(tmp_path)
    man = dict(geometry_hash=plan["geometry_hash"], cases=plan["cases"])
    with open(out / plan["cases"][0]["name"] / "system" / "controlDict", "a") as f:
        f.write("\n// tampered\n")
    with pytest.raises(RL.ProvenanceError):
        RL.build_per_bank_rx(man, str(out), 4)


def test_input_tree_hash_ignores_blockmesh_output(tmp_path):
    """Mesh output does not change the input-tree hash; an edited input file still does."""
    from acoustic_liner.cfd import impedance_correction as RL
    plan, out = _vehicle_plan(tmp_path)
    man = dict(geometry_hash=plan["geometry_hash"], cases=plan["cases"])
    case = out / plan["cases"][0]["name"]
    (case / "constant" / "polyMesh").mkdir()
    (case / "constant" / "polyMesh" / "points").write_text("(0 0 0)\n")
    RL.build_per_bank_rx(man, str(out), 4)
    with open(case / "constant" / "transportProperties", "a") as f:
        f.write("\n// tampered\n")
    with pytest.raises(RL.ProvenanceError):
        RL.build_per_bank_rx(man, str(out), 4)


def _synthetic_case(tmp_path, R_true, X_true, split=False):
    """A solved case's files with KNOWN neck impedance: _case_meta.json, a checkMesh log and the three probe planes
    (pUp, pDown, fluxMid) as OpenFOAM surfaceFieldValue.dat histories."""
    from acoustic_liner.cfd import postprocess as POST
    case = tmp_path / "case"
    case.mkdir()
    d_mm, f0, U = 4.0, 126.51, 0.5
    (case / "_case_meta.json").write_text(json.dumps(dict(bank_id="b0", d_mm=d_mm, f0_Hz=f0, uneck_m_s=U,
                                                          geometry_hash="g", case_hash="c")))
    (case / "log.checkMesh").write_text("Mesh OK.\n")
    w, T = 2 * math.pi * f0, 1 / f0
    t = np.arange(0, 20 * T, T / 400.0)
    u = U * np.cos(w * t)
    dp = R_true * u + X_true * (-U * np.sin(w * t))
    a_neck = math.pi * (d_mm / 2000.0) ** 2
    q = u * a_neck / (POST.FLOW_SIGN * POST.WEDGE_FACTOR)
    for name, v in (("pUp", dp / POST.RHO), ("pDown", 0 * dp), ("fluxMid", q)):
        for td in (["0", "0.05"] if split and name == "pUp" else ["0"]):
            d = case / "postProcessing" / name / td
            d.mkdir(parents=True)
            (d / "surfaceFieldValue.dat").write_text(
                "# Time value\n" + "".join(f"{a:.12e}\t{b:.12e}\n" for a, b in zip(t, v)))
    return case


def test_post_case_recovers_known_R_X_from_probe_files(tmp_path):
    """Known neck resistance and reactance are recovered from probe files (probe files -> post_case ->
    validate_cfd_row)."""
    from acoustic_liner.cfd import postprocess as POST
    from acoustic_liner.cfd import validation as VAL
    row = POST.post_case(str(_synthetic_case(tmp_path, 0.687, 25.09)))
    assert row["probe_clean"] and row["mesh_ok"]
    assert abs(row["R_eff"] - 0.687) < 1e-2 and abs(row["R_phasor"] - 0.687) < 1e-2
    assert abs(row["X_phasor"] - 25.09) < 5e-2 and abs(row["u_amp"] - 0.5) < 1e-3
    assert row["R_drift"] < 1e-3 and row["X_drift"] < 1e-3 and row["thd_p"] < 1e-3
    ok, rep = VAL.validate_cfd_row(row, expected_bank_id="b0", expected_case_hash="c", expected_geometry_hash="g")
    assert ok, rep


def test_post_case_fails_closed_on_a_restart_split(tmp_path):
    from acoustic_liner.cfd import postprocess as POST
    from acoustic_liner.cfd import validation as VAL
    row = POST.post_case(str(_synthetic_case(tmp_path, 0.687, 25.09, split=True)))
    assert row["probe_clean"] is False and "restart split" in row["probe_error"] and math.isnan(row["R_phasor"])
    ok, rep = VAL.validate_cfd_row(row, expected_bank_id="b0", expected_case_hash="c", expected_geometry_hash="g")
    assert not ok and rep["failed"]


# Frozen anchor: the hashes of the first case generated for the Falcon 9 blanketed scenario with the `L2_b` bank
# prefix (captured from the generator before it was generalized). They must never drift.
_ANCHOR_NAME = "L2_b0_u0.5"
_ANCHOR_CASE_HASH = "24cc2580ca9306abee2015ae0615495f13ae28f28bb43cc68b68358d41bfbf4b"
_ANCHOR_INPUT_TREE_HASH = "d638effe4387c6ba5dbd8e21f8ba3777f3d47787e873d4c11c57335fdbb93f2b"


def test_explicit_inputs_reproduce_the_frozen_case_hashes(tmp_path):
    """The frozen per-case hashes, reached through gen_cases with explicit inputs (the historical test reached
    them through a wrapper that supplied the same inputs)."""
    from acoustic_liner.cfd import case_generation as CG
    from acoustic_liner.fem import scenarios as SC
    scen = SC.falcon9_2025_blanketed()
    plan = CG.gen_cases(str(tmp_path), scenario=scen.name, source_corner="kappa_hi_low",
                        finalist_path=os.fspath(FINALIST), spectrum=list(scen.spl_dB),
                        bank_prefix="L2_b", plan_filename="_l2_case_plan.json", cycles=12)
    assert plan["n_cases"] == 28
    assert plan["scenario"] == "falcon9_2025_blanketed"
    assert plan["source_corner"] == "kappa_hi_low"
    anchor = next(c for c in plan["cases"] if c["name"] == _ANCHOR_NAME)
    assert anchor["case_hash"] == _ANCHOR_CASE_HASH
    assert anchor["input_tree_hash"] == _ANCHOR_INPUT_TREE_HASH
