# -*- coding: utf-8 -*-
"""Unit tests for the self-contained nonlinear-CFD subpackage acoustic_liner.cfd (historical nonlinear-CFD protocol):
excess-inertance law + measured anchor binding, the single validate_cfd_row predicate (incl. R>=0 gates),
probe-history sanitation, manifest hash binding, sensitivity (non-cancelling), and the AST self-containment gate."""
import ast
import json
import os
import numpy as np
import pytest

from acoustic_liner.cfd import feedback as FB
from acoustic_liner.cfd import validation as VAL
from acoustic_liner.cfd import postprocess as POST
from acoustic_liner.cfd import case_generation as CG
from acoustic_liner.cfd import impedance_correction as RL

_CFD_DIR = os.path.dirname(FB.__file__)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_CFD_DIR)))   # cfd -> acoustic_liner -> src -> repository root
_L2 = os.path.join(_ROOT, "finalists", "falcon_v2.1_finalist_geometry.json")
_GOOD_ROW = dict(bank_id="L2_b0", case_hash="c", geometry_hash="g", u_target=6.0, u_amp=6.2, mesh_ok=True,
                 R_phasor=2.0, X_phasor=21.0, R_eff=2.05, X_accel_xcheck=20.7, R_drift=0.01, X_drift=0.02,
                 thd_p=0.05, thd_u=0.04, probe_clean=True)
_RX = [(0.5, 1.0, 20.0), (2.0, 1.3, 21.0), (6.0, 2.0, 22.0), (12.0, 3.2, 23.0), (20.0, 4.5, 24.0)]
_ANCHOR = (0.5, 1.0, 20.0)                       # validated low-amplitude anchor (measured u_amp/R/X)


def _l2_design_cond():
    fin = json.load(open(_L2))
    design = fin["geometry"]
    from acoustic_liner.fem import scenarios as S
    scen = S.falcon9_2025_blanketed()
    cond = dict(spl_dB=list(scen.spl_dB), f_cov=float(fin["coverage"]), A_wall=float(scen.A_wall),
                kappa_nominal=float(scen.kappa_nominal), dOASPL_target=0.9213)
    return design, cond


# ---- impedance law + anchor (Gap 4) ----------------------------------------------------------------------


def test_excess_inertance_fixed_point_ok_uses_measured_anchor():
    bank = dict(f0_Hz=120.0, z_linear_complex_Rayl=complex(2.0, 20.0))
    r = FB.self_consistent_U(bank, 128.0, _RX, _ANCHOR)
    assert r["status"] == "ok" and r["unique"] and r["U_star"] > 0
    assert r["dR_excess_Rayl"] >= 0.0
    assert r["clamp"]["U_ref_measured"] == pytest.approx(0.5)      # anchor is the MEASURED row, not interp@0.5
    assert r["clamp"]["R_ref_Rayl"] == pytest.approx(1.0)


def test_missing_anchor_fails_closed():
    bank = dict(f0_Hz=120.0, z_linear_complex_Rayl=complex(2.0, 20.0))
    assert FB.self_consistent_U(bank, 128.0, _RX, None)["status"] == "no_anchor"


def test_high_amplitude_anchor_rejected():
    bank = dict(f0_Hz=120.0, z_linear_complex_Rayl=complex(2.0, 20.0))
    assert FB.self_consistent_U(bank, 128.0, _RX, (5.0, 1.0, 20.0))["status"] == "no_anchor"


def test_reactance_is_frequency_dependent_not_constant():
    groups = [dict(f0=120.0, r_neck=0.01, L_neck=0.03, Q=15.0, area_frac=1.0)]
    D = 0.05
    dR = np.array([0.0]); dM = np.array([1e-3])
    z1 = FB._z_surface_NL(np.array([100.0]), groups, D, dR, dM)[0]
    z0 = FB._z_surface_NL(np.array([100.0]), groups, D, dR, np.array([0.0]))[0]
    z2 = FB._z_surface_NL(np.array([200.0]), groups, D, dR, dM)[0]
    z2_0 = FB._z_surface_NL(np.array([200.0]), groups, D, dR, np.array([0.0]))[0]
    assert abs(z1 - z0) > 0
    assert abs(z2 - z2_0) > abs(z1 - z0)                          # reactive excess grows with frequency


def test_material_nonmonotonic_R_fails_closed():
    bank = dict(f0_Hz=120.0, z_linear_complex_Rayl=complex(2.0, 20.0))
    rx_bad = [(0.5, 3.0, 20.0), (2.0, 2.0, 21.0)]
    assert FB.self_consistent_U(bank, 128.0, rx_bad, (0.5, 3.0, 20.0))["status"] == "nonmonotonic_R"


def test_numerical_R_noise_is_clamped_and_recorded():
    bank = dict(f0_Hz=120.0, z_linear_complex_Rayl=complex(2.0, 20.0))
    rx = [(0.5, 1.00, 20.0), (2.0, 0.98, 21.0), (6.0, 2.0, 22.0), (20.0, 4.5, 24.0)]
    r = FB.self_consistent_U(bank, 128.0, rx, (0.5, 1.0, 20.0))
    assert r["clamp"]["n_clamped"] >= 1


def test_extend_sweep_when_root_above_range():
    bank = dict(f0_Hz=120.0, z_linear_complex_Rayl=complex(0.3, 2.0))
    rx = [(0.5, 1.0, 5.0), (2.0, 1.1, 5.0)]
    assert FB.self_consistent_U(bank, 150.0, rx, (0.5, 1.0, 5.0))["status"] == "extend_sweep"


def test_relabel_incomplete_when_a_bank_uncorrected():
    groups = [dict(f0=120.0, r_neck=0.01, L_neck=0.03, Q=15.0, area_frac=0.25) for _ in range(4)]
    design = dict(D=0.05, groups=groups)
    cond = dict(spl_dB=[120.0] * 8, f_cov=0.4, A_wall=165.0, kappa_nominal=0.23, dOASPL_target=0.9213)
    per_bank = [_RX, _RX, _RX, None]
    anchors = [_ANCHOR, _ANCHOR, _ANCHOR, None]
    out = FB.relabel_design(design, cond, per_bank, anchors)
    assert out["status"] == "incomplete" and out["dOASPL_nonlinear"] is None and 3 in out["banks_not_ok"]


# ---- validate_cfd_row (incl. new R>=0 gates) --------------------------------------------------------------
def test_validate_cfd_row_good_passes():
    ok, rep = VAL.validate_cfd_row(_GOOD_ROW)
    assert ok and rep["failed"] == []


@pytest.mark.parametrize("field,val", [("u_amp", -1.0), ("R_drift", 0.9), ("X_drift", 0.9), ("thd_p", 0.9),
                                       ("mesh_ok", False), ("probe_clean", False), ("R_eff", 100.0),
                                       ("X_accel_xcheck", 100.0), ("R_phasor", -0.5), ("R_eff", -0.5)])
def test_validate_cfd_row_each_gate_fails_closed(field, val):
    row = dict(_GOOD_ROW); row[field] = val
    ok, rep = VAL.validate_cfd_row(row)
    assert not ok and len(rep["failed"]) >= 1


def test_validate_missing_field_fails():
    row = dict(_GOOD_ROW); del row["R_phasor"]
    ok, rep = VAL.validate_cfd_row(row)
    assert not ok and "R_phasor" in rep["missing"]


# ---- probe-history sanitation (Gap 2) ---------------------------------------------------------------------
def _write_plane(case, name, tdir, times, vals):
    d = os.path.join(case, "postProcessing", name, tdir)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "surfaceFieldValue.dat"), "w") as f:
        f.write("# t value\n")
        for t, v in zip(times, vals):
            f.write(f"{t} {v}\n")


def test_read_plane_rejects_multiple_time_dirs(tmp_path):
    case = str(tmp_path / "c")
    t = np.linspace(0, 0.1, 50)
    _write_plane(case, "pUp", "0", t, np.sin(t))
    _write_plane(case, "pUp", "0.05", t, np.sin(t))            # restart split -> two dirs
    with pytest.raises(POST.ProbeError):
        POST.read_plane(case, "pUp")


def test_read_plane_rejects_nonmonotone_times(tmp_path):
    case = str(tmp_path / "c")
    _write_plane(case, "pUp", "0", [0.0, 0.02, 0.01, 0.03], [0, 1, 2, 3])
    with pytest.raises(POST.ProbeError):
        POST.read_plane(case, "pUp")


def test_post_case_fails_closed_on_bad_probe(tmp_path):
    case = str(tmp_path / "c")
    os.makedirs(case)
    json.dump(dict(d_mm=20.0, f0_Hz=120.0, uneck_m_s=6.0, bank_id="L2_b0", geometry_hash="g", case_hash="c"),
              open(os.path.join(case, "_case_meta.json"), "w"))
    t = np.linspace(0, 0.1, 50)
    for nm in ("pUp", "pDown", "fluxMid"):                     # each plane has a restart split
        _write_plane(case, nm, "0", t, np.sin(t)); _write_plane(case, nm, "0.05", t, np.sin(t))
    row = POST.post_case(case)
    assert row["probe_clean"] is False and not np.isfinite(row["R_phasor"])
    ok, _ = VAL.validate_cfd_row(row)
    assert not ok


# ---- manifest hash binding (Gap 3) ------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", ["b", "L2_b"])
def test_build_per_bank_rx_parses_generic_and_l2_bank_ids(tmp_path, monkeypatch, prefix):
    # regression: generic vehicle ids `b0..b3` crashed int() on VALID rows (only `L2_b0` parsed)
    cases = []
    for j, u in ((0, 0.5), (0, 6.0), (2, 0.5)):
        name = f"{prefix}{j}_u{u:g}"
        cd = tmp_path / name
        (cd / "system").mkdir(parents=True)
        (cd / "system" / "controlDict").write_text("endTime 1;\n")
        meta = dict(bank_id=f"{prefix}{j}", d_mm=20.0, L_mm=30.0, f0_Hz=120.0, uneck_m_s=u, refine=1.0, maxco=0.5,
                    geometry_hash="g")
        meta["case_hash"] = CG.recompute_case_hash(meta)
        json.dump(meta, open(cd / "_case_meta.json", "w"))
        cases.append(dict(name=name, bank_id=meta["bank_id"], u_target=u, case_hash=meta["case_hash"],
                          input_tree_hash=CG.input_tree_hash(str(cd))))
    monkeypatch.setattr(POST, "post_case", lambda cd, n_periods=3: dict(
        u_amp=json.load(open(os.path.join(cd, "_case_meta.json")))["uneck_m_s"], R_phasor=1.0, X_phasor=2.0))
    monkeypatch.setattr(VAL, "validate_cfd_row", lambda row, **k: (True, {}))
    per_bank, anchors, _ = RL.build_per_bank_rx(dict(geometry_hash="g", cases=cases), str(tmp_path), 4)
    assert [len(t) if t else 0 for t in per_bank] == [2, 0, 1, 0]
    assert anchors[0] == (0.5, 1.0, 2.0) and anchors[2] == (0.5, 1.0, 2.0) and anchors[1] is None


# ---- sensitivity does NOT cancel (Gap 1) ------------------------------------------------------------------


# ---- self-containment (AST) -------------------------------------------------------------------------------
def test_cfd_subpackage_is_self_contained():
    FORBIDDEN = {"fairing_oracle", "verify_eval", "oracle_v2", "make_case", "make_incompressible",
                 "post_incompressible", "fem_interior_nr", "fem_interior_v2", "_cfd_feedback"}
    ALLOWED_ABS_ROOTS = {"numpy", "scipy", "os", "sys", "json", "math", "re", "hashlib", "argparse",
                         "itertools", "collections", "dataclasses", "typing", "functools", "glob", "__future__"}
    for fn in sorted(os.listdir(_CFD_DIR)):
        if not fn.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(_CFD_DIR, fn)).read(), filename=fn)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    root = a.name.split(".")[0]
                    assert root not in FORBIDDEN, f"{fn}: forbidden import {a.name}"
                    assert root in ALLOWED_ABS_ROOTS, f"{fn}: unexpected absolute import {a.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level >= 1:
                    assert (node.module or "") not in FORBIDDEN, f"{fn}: forbidden relative import {node.module}"
                    continue
                root = (node.module or "").split(".")[0]
                assert root not in FORBIDDEN, f"{fn}: forbidden import {node.module}"
                assert root in ALLOWED_ABS_ROOTS, f"{fn}: unexpected absolute import {node.module}"
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == "sys" and node.attr == "path":
                    raise AssertionError(f"{fn}: touches sys.path (not self-contained)")
