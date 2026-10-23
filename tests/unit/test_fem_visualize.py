# -*- coding: utf-8 -*-
"""Visualization-only FEM field export: conformance with the production solver, frozen-record reproduction,
phase reconstruction, complex differencing, boundary tags, deterministic band selection, manifest binding, and
no modification of hash-bound solver files or frozen records."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from acoustic_liner.fem import mesh as MESH
from acoustic_liner.fem import solver as FS
from acoustic_liner.visualization import fem_fields as FV
from acoustic_liner.vehicles import fem as VF
from acoustic_liner.vehicles import scenarios as V

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/dev/vehicle_v1"
_skip_no_records = pytest.mark.skipif(
    not (RUN / "falcon9_2025_unblanketed_v2/initial_fem.json").exists(),
    reason="optional full-evidence test: needs the private full-evidence-v1 archive extracted to artifacts/dev "
           "(not distributed; see evidence/manifest.json)")


def needs_records(test):
    return pytest.mark.full_evidence(_skip_no_records(test))


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---- conformance with the production solver on a small mesh (fast, no records needed) -----------------------------
@pytest.fixture(scope="module")
def small():
    vs = V.falcon9_unblanketed()
    case = vs.payload_cases[vs.payload_primary]
    mesh = VF.build_vehicle_mesh(vs.geometry, case, 0.6)
    tagged = VF.tag_vehicle_facets(mesh, vs.geometry)
    return mesh, tagged, MESH.mesh_physical_volume(mesh)


@pytest.mark.parametrize("cov", [1.0, 0.5])
@pytest.mark.parametrize("payload_beta", [0.0, 0.08 - 0.02j])
def test_solve_field_matches_production(small, cov, payload_beta):
    mesh, tagged, vol = small
    treated = None if cov >= 1 else MESH.select_coverage_facets(mesh, tagged, cov, anchor="low")
    extra = [] if payload_beta == 0 else [(tagged["sets"]["payload"], payload_beta)]
    kw = dict(A_other=3.0, volume=vol, p0=1.0, treated_facets=treated, bare_beta=0.006 + 0j, extra_robin=extra)
    p, ms = FV.solve_field(mesh, 125.0, tagged, 0.4 - 0.1j, **kw)
    ref = FS.solve_interior(mesh, 125.0, tagged, wall_beta=0.4 - 0.1j, **kw)
    assert p.shape == (mesh.p.shape[1],) and np.iscomplexobj(p)
    assert abs(ms - ref) / ref < 1e-12


def test_phase_reconstruction_matches_phasor():
    rng = np.random.default_rng(0)
    p = rng.normal(size=50) + 1j * rng.normal(size=50)
    frames = [p.real * math.cos(2 * math.pi * k / 36) - p.imag * math.sin(2 * math.pi * k / 36) for k in range(36)]
    assert np.allclose(frames[0], p.real) and np.allclose(frames[9], -p.imag)
    assert np.allclose(frames[5], np.real(p * np.exp(1j * 2 * math.pi * 5 / 36)))      # e^{+i omega t}
    assert np.allclose(np.mean(np.square(frames), axis=0), np.abs(p) ** 2 / 2)          # cycle mean = |p|^2/2


def test_difference_is_complex_not_scalar():
    pb = np.array([1.0 + 0j, 1.0 + 0j])
    pt = np.array([1.0 + 0j, -1.0 + 0j])                      # same magnitude, opposite phase at node 1
    D = FV.derived(dict(p_bare=pb, p_treated=pt))
    assert np.allclose(D["abs_diff"], [0.0, 2.0])             # |p_t - p_b|, not ||p_t| - |p_b|| (= 0)
    assert np.allclose(D["rel_dB"], [0.0, 0.0])
    assert D["floor"] == pytest.approx(FV.REL_FLOOR)


def test_band_selection_rule_and_tie_break():
    f = [100.0, 200.0, 400.0]
    i, g = FV.pick_band(f, [120.0, 120.0, 110.0], [3.0, 3.0, 20.0])
    assert i == 0 and g[0] == g[1]                            # tie -> lower frequency
    i, _ = FV.pick_band(f, [120.0, 120.0, 110.0], [3.0, 6.0, 1.0])
    assert i == 1


# ---- frozen-record reproduction, tags, manifest (needs the local frozen records) ----------------------------------
@pytest.fixture(scope="module")
def falcon_export(tmp_path_factory):
    frozen = [RUN / "falcon9_2025_unblanketed_v2" / n for n in ("initial_fem.json", "finalist.json",
                                                              "verdict/verdict.json")]
    before = {p: _sha(p) for p in frozen}
    R = FV.compute("falcon")
    out = tmp_path_factory.mktemp("fem_viz")
    man = FV.export(R, out)
    return R, man, out, before


@needs_records
def test_selected_frequency_follows_rule(falcon_export):
    R, man, _, _ = falcon_export
    sel = man["selection"]
    i, g = FV.pick_band(sel["bands_hz"], sel["spl_dB"], sel["nr_band"])
    assert man["frequency_hz"] == sel["bands_hz"][i] == 200.0
    assert sel["mesh_size"] == 0.1 and sel["source_corner"] == "west_range"


@needs_records
def test_field_reproduces_frozen_record(falcon_export):
    R, man, _, _ = falcon_export
    v = man["verification"]
    assert v["ms_bare_rel_err"] < 1e-10 and v["ms_treated_rel_err"] < 1e-10 and v["nr_abs_err_dB"] < 1e-8
    assert man["mesh"]["mesh_sha256"] == man["selection"]["record_mesh_sha256"]


@needs_records
def test_same_mesh_and_source_for_both_states(falcon_export):
    R, _, out, _ = falcon_export
    import meshio
    a, b = (meshio.read(out / f"falcon_fem_field_{s}.vtu") for s in ("bare", "treated"))
    assert np.array_equal(a.points, b.points)
    assert np.array_equal(a.cells_dict["triangle"], b.cells_dict["triangle"])
    assert R["p_bare"].shape == R["p_treated"].shape


@needs_records
def test_boundary_tags_and_treated_mask(falcon_export):
    R, _, out, _ = falcon_export
    import meshio
    m = meshio.read(out / "falcon_fem_boundaries.vtu")
    tag = m.cell_data["boundary_tag"][0]
    assert set(tag) == set(FV.BOUNDARY_TAGS.values())
    treated, bare = m.cell_data["treated_wall"][0], m.cell_data["bare_wall"][0]
    wall = tag == FV.BOUNDARY_TAGS["wall"]
    assert treated[wall].all() and not bare.any()             # full coverage: every wall facet treated
    assert not treated[~wall].any()


@needs_records
def test_manifest_binds_inputs_and_outputs(falcon_export):
    R, man, out, _ = falcon_export
    assert man["visualization_only"] is True
    for k in ("scenario_hash", "finalist_sha256", "design_sha256", "geometry_hash", "initial_fem_sha256"):
        assert isinstance(man[k], str) and len(man[k]) == 64
    assert man["identity"]["payload_geometry_sha256"] and man["identity"]["payload_bc_sha256"]
    for name, sha in man["outputs"].items():
        assert _sha(out / name) == sha
    assert man["schema"] == "acoustic-liner/fem-field-manifest/2" and "historical_source_sha256" not in man
    assert set(man["public_source_files"]) == set(FV.PUBLIC_SOURCE_FILES)
    assert json.loads((out / "falcon_fem_field_manifest.json").read_text())["outputs"] == man["outputs"]


@needs_records
def test_frozen_records_unmodified(falcon_export):
    _, _, _, before = falcon_export
    for p, sha in before.items():
        assert _sha(p) == sha, p
