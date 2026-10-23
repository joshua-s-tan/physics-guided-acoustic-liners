# -*- coding: utf-8 -*-
"""Generic reference-cavity benchmark (§4.2): geometry DERIVED from dimensions with a unique payload axial
position + positive clearance, DISJOINT tagged boundary areas (source base != lined wall), S_other large
enough for every kappa corner, and a geometry hash. Not a vehicle."""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner.fem.verification import reference_cavity as RC  # noqa: E402


def _cav(**kw):
    d = dict(name="generic_boundary_benchmark_v1", R_outer=2.6, H=10.0, R_payload=1.5, H_payload=4.0,
             max_A_other=43.6)
    d.update(kw)
    return RC.ReferenceCavity(**d)


def test_required_S_other_for_falcon_high_corner():
    A_hi = 0.262705707073558 * 165.878
    assert RC.required_S_other(A_hi) == pytest.approx(45.87, abs=0.02)


def test_disjoint_named_areas_and_defaults():
    c = _cav()
    assert c.z_payload == pytest.approx((10.0 - 4.0) / 2.0)         # centered by default
    assert c.source_area == pytest.approx(math.pi * 2.6**2)        # base cap only
    assert c.lined_wall_area == pytest.approx(2*math.pi*2.6*10.0 + math.pi*2.6**2)  # lateral + TOP cap
    assert c.S_other == pytest.approx(2*math.pi*1.5*4.0 + 2*math.pi*1.5**2)
    assert c.rigid_area == 0.0
    # source cap is NOT part of the lined wall (disjoint): total = sum of tags
    assert c.total_boundary_area == pytest.approx(c.source_area + c.lined_wall_area + c.S_other)
    assert c.S_other >= RC.required_S_other(43.6)
    assert c.scenario_kind == "generic_boundary_benchmark" and c.is_vehicle is False


def test_payload_clearance_required():
    with pytest.raises(RC.ReferenceCavityError):
        _cav(H_payload=10.0)                       # touches base+apex, no clearance
    with pytest.raises(RC.ReferenceCavityError):
        _cav(z_payload=0.0)                        # sits on the base (source boundary)
    with pytest.raises(RC.ReferenceCavityError):
        _cav(R_payload=2.7)                        # wider than outer
    with pytest.raises(RC.ReferenceCavityError):
        _cav(z_payload=6.5)                        # top (6.5+4=10.5) beyond apex


def test_S_other_too_small_rejected():
    with pytest.raises(RC.ReferenceCavityError):
        _cav(R_payload=0.4, H_payload=0.5)


def test_vehicle_name_and_kind_guards():
    for bad in ("falcon9_ref", "NASA_cavity", "electron_bench"):
        with pytest.raises(RC.ReferenceCavityError):
            _cav(name=bad)
    with pytest.raises(RC.ReferenceCavityError):
        _cav(is_vehicle=True)
    with pytest.raises(RC.ReferenceCavityError):
        _cav(scenario_kind="vehicle")


def test_alpha_other_rejects_bad_A_other():
    c = _cav()
    with pytest.raises(RC.ReferenceCavityError):
        c.alpha_other(-1.0)
    with pytest.raises(RC.ReferenceCavityError):
        c.alpha_other(float("nan"))
    with pytest.raises(RC.ReferenceCavityError):
        c.alpha_other(c.S_other)                   # alpha_other = 1.0 > 0.95


def test_geometry_hash_stable_and_sensitive():
    a = _cav().geometry_hash()
    assert a == _cav().geometry_hash()             # deterministic
    assert a != _cav(R_payload=1.6).geometry_hash()  # sensitive to dimensions


def test_area_closure_is_necessary_not_sufficient():
    c = _cav()
    tot = c.total_boundary_area
    c.assert_area_closure(c.source_area*1.001, c.lined_wall_area*0.999, c.S_other*1.002, tot*1.0)
    with pytest.raises(RC.ReferenceCavityError):    # wall off by 10%
        c.assert_area_closure(c.source_area, c.lined_wall_area*1.10, c.S_other, tot)
    with pytest.raises(RC.ReferenceCavityError):    # sum != total boundary
        c.assert_area_closure(c.source_area, c.lined_wall_area, c.S_other, tot*0.5)


def _facet_setup(c):
    """Build a per-facet area map whose tag sums equal the analytic areas (source/wall/payload)."""
    allf = list(range(30))
    sets = {"source": set(range(0, 5)), "lined_wall": set(range(5, 22)), "S_other": set(range(22, 30))}
    area_by_id = {}
    for tag, ids in sets.items():
        analytic = {"source": c.source_area, "lined_wall": c.lined_wall_area, "S_other": c.S_other}[tag]
        per = analytic / len(ids)
        for fid in ids:
            area_by_id[fid] = per
    return set(allf), sets, area_by_id


def test_facet_partition_gate_valid():
    c = _cav()
    allf, sets, area_by_id = _facet_setup(c)
    res = c.assert_facet_partition(sets, allf, area_by_id)
    assert res["n_facets"] == 30


def test_facet_partition_computes_areas_not_trusts_aggregate():
    """The exact reproduced loophole: move a source facet into a nonempty 'rigid' tag but keep the claimed
    source area. Gate must reject (rigid must be empty when rigid_area==0; areas computed from facets)."""
    c = _cav()
    allf, sets, area_by_id = _facet_setup(c)
    moved = dict(sets)
    moved["source"] = set(range(0, 4))          # drop facet 4 from source
    moved["rigid"] = {4}                         # park it in rigid, though rigid_area == 0
    with pytest.raises(RC.ReferenceCavityError):
        c.assert_facet_partition(moved, allf, area_by_id)


def test_facet_partition_rejects_bad_partitions():
    c = _cav()
    allf, sets, area_by_id = _facet_setup(c)
    # overlapping tags
    ov = {**sets, "source": set(range(0, 6))}
    with pytest.raises(RC.ReferenceCavityError):
        c.assert_facet_partition(ov, allf, area_by_id)
    # untagged gap (facet 29 left out) -> union != all
    gap = {**sets, "S_other": set(range(22, 29))}
    with pytest.raises(RC.ReferenceCavityError):
        c.assert_facet_partition(gap, allf, area_by_id)
    # unknown tag
    with pytest.raises(RC.ReferenceCavityError):
        c.assert_facet_partition({**sets, "mystery": set()}, allf, area_by_id)
    # missing a facet's area in the map
    bad_map = dict(area_by_id); del bad_map[10]
    with pytest.raises(RC.ReferenceCavityError):
        c.assert_facet_partition(sets, allf, bad_map)
    # a facet with non-finite/zero area
    inf_map = {**area_by_id, 10: float("inf")}
    with pytest.raises(RC.ReferenceCavityError):
        c.assert_facet_partition(sets, allf, inf_map)
    # inconsistent facet-area map: computed S_other area 20% high
    hi_map = dict(area_by_id)
    for fid in sets["S_other"]:
        hi_map[fid] *= 1.2
    with pytest.raises(RC.ReferenceCavityError):
        c.assert_facet_partition(sets, allf, hi_map)
    # infinite tolerance rejected
    with pytest.raises(RC.ReferenceCavityError):
        c.assert_facet_partition(sets, allf, area_by_id, tol=float("inf"))
