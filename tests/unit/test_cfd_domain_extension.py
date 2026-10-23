# -*- coding: utf-8 -*-
"""Unit tests for the optional reservoir extension of the CFD neck case (cfd.case.build_neck_mesh H_ext,
cfd.case_generation.generate H_ext_mm). The default must reproduce the recorded Falcon b3 16.2 m/s case exactly;
an extended case may differ only in the appended mesh blocks, the moved inlet/outlet faces and its identity."""
import json
import os

from acoustic_liner.cfd import case as MI
from acoustic_liner.cfd import case_generation as CG

# Recorded Falcon bank b3, 16.2 m/s source case (run `sens_halfdt__b3_u16.2/source`, 12 cycles).
FALCON_B3 = dict(bank_id="b3", d_mm=19.9046, L_mm=5.12627841739253, f0=251.546, uneck=16.2,
                 geometry_hash="ca871cc15528866560f35e70d2d7be1be5c279f2fc304e5d2c6595764fccad6a")
RECORDED_CASE_HASH = "5b8f8607ef90ce491929aeecda0b4062411738f510b0731d449dd5d68e44b5f1"
RECORDED_INPUT_TREE_HASH = "f797788cb2bc19f5727c06a635fbc3bade0d8d6a5f4cbf1b12289ce0225fbdd4"


def _tree(d):
    out = {}
    for dp, _, fns in os.walk(d):
        for fn in fns:
            p = os.path.join(dp, fn)
            out[os.path.relpath(p, d).replace(os.sep, "/")] = open(p, "rb").read()
    return out


def _section(text, name):
    text = text.replace("\r\n", "\n")
    return text.split(f"\n{name}\n(\n", 1)[1].split("\n);", 1)[0].splitlines()


def test_default_reproduces_recorded_case(tmp_path):
    _, meta = CG.generate(str(tmp_path / "a"), cycles=12, **FALCON_B3)
    assert meta["case_hash"] == RECORDED_CASE_HASH
    assert meta["input_tree_hash"] == RECORDED_INPUT_TREE_HASH
    assert "H_ext_mm" not in meta
    _, meta0 = CG.generate(str(tmp_path / "b"), cycles=12, H_ext_mm=0.0, **FALCON_B3)
    assert _tree(tmp_path / "a") == _tree(tmp_path / "b")


def test_extension_cell_size_matches_reservoir_end_cell():
    end = MI.far_cell(MI.H_RES, MI.NZ_RES, MI.G_ORIF_LO)
    assert MI.n_ext(0.040) == 35
    assert abs(0.040 / MI.n_ext(0.040) - end) / end < 0.01


def test_extended_case_changes_only_the_declared_extension(tmp_path):
    CG.generate(str(tmp_path / "c"), cycles=25, **FALCON_B3)
    _, meta = CG.generate(str(tmp_path / "e"), cycles=25, H_ext_mm=40.0, **FALCON_B3)
    c, e = _tree(tmp_path / "c"), _tree(tmp_path / "e")
    assert set(c) == set(e)
    differ = {k for k in c if c[k] != e[k]}
    assert differ == {"system/blockMeshDict", "_case_meta.json"}       # forcing, probes, fields, settings equal
    bc, be = c["system/blockMeshDict"].decode(), e["system/blockMeshDict"].decode()
    vc, ve = _section(bc, "vertices"), _section(be, "vertices")
    hc, he = _section(bc, "blocks"), _section(be, "blocks")
    assert ve[:len(vc)] == vc and len(ve) == len(vc) + 10                # original vertices first, unchanged
    assert he[:len(hc)] == hc and len(he) == len(hc) + 4                 # original blocks first, unchanged
    assert all(" 35 1) simpleGrading (" in h and h.endswith(" 1.0 1)") for h in he[len(hc):])   # uniform axially
    assert meta["H_ext_mm"] == 40.0 and meta["H_res_mm"] == 40.0 and meta["n_ext"] == 35
    assert meta["case_hash"] != RECORDED_CASE_HASH
    assert CG.recompute_case_hash(meta) == meta["case_hash"]
    assert CG.recompute_case_hash(json.loads(c["_case_meta.json"])) == RECORDED_CASE_HASH
