# -*- coding: utf-8 -*-
r"""§14 Step 9 / G1b: FREEZE the v1 FEM continuity reference into a committed, hashed fixture so the mandatory
continuity gate runs SELF-CONTAINED (no live reference tree at test/verify_release time).

Run ONCE with the v1 reference tree available (it imports the v1 solver `fem_interior_nr`). It builds the shared
fairing-interior mesh EXACTLY as the continuity test does, records a mesh fingerprint, evaluates the v1 solver at
the design bands and a fine grid (both lined and rigid), and writes tests/fixtures/v1_continuity_reference.json
with an embedded self SHA-256. The continuity test then compares the MIGRATED solver against this frozen v1
reference and FAILS CLOSED if the mesh drifts (fingerprint mismatch) or the fixture is tampered (hash mismatch).

    ACOUSTIC_LINER_V1_REFERENCE_TREE=/path/to/resonator_updated_6 \
        acoustic-liner provenance continuity-fixture
"""
from __future__ import annotations
import hashlib
import json
import os
import sys

import numpy as np

from .. import paths
ROOT = str(paths.ROOT)
from ..fem import mesh as MESH  # noqa: E402
from ..fem import scenarios as S  # noqa: E402

FIXTURE = os.path.join(ROOT, "tests", "fixtures", "v1_continuity_reference.json")
MESH_SIZE = 0.35
ZETA_WALL = 3.0 + 0.0j
GRID = np.linspace(45.0, 260.0, 60)


def mesh_fingerprint(mesh):
    """Stable SHA-256 over the mesh vertices + connectivity (rounded) — ties the frozen v1 values to the exact
    mesh they were computed on, so mesh/toolchain drift is caught rather than silently compared to a stale ref."""
    p = np.round(np.asarray(mesh.p, float), 9)
    t = np.asarray(mesh.t, np.int64)
    h = hashlib.sha256()
    h.update(b"p"); h.update(np.ascontiguousarray(p).tobytes())
    h.update(b"t"); h.update(np.ascontiguousarray(t).tobytes())
    h.update(b"shape"); h.update(str((p.shape, t.shape)).encode())
    return h.hexdigest()


def _self_sha256(doc: dict) -> str:
    d = {k: v for k, v in doc.items() if k != "self_sha256"}
    return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_reference():
    v1_root = os.environ.get("ACOUSTIC_LINER_V1_REFERENCE_TREE", "")
    v1_fc = os.path.join(v1_root, "fairing_case") if v1_root else ""
    if not (v1_root and os.path.isdir(v1_fc)
            and os.path.isfile(os.path.join(v1_root, "fem_full_duct_radiation.py"))):
        raise SystemExit("ACOUSTIC_LINER_V1_REFERENCE_TREE unset/invalid: point it at the resonator checkout to FREEZE.")
    for p in (v1_root, v1_fc):
        if p not in sys.path:
            sys.path.insert(0, p)
    import fem_interior_nr as FIN               # v1 reference solver

    scen = S.falcon9_2025_blanketed()
    g = scen.geometry
    # mesh determinism guard: build twice, require identical fingerprint (else the fixture cannot be trusted)
    m1 = MESH.build_fairing_interior_mesh(g.R_fair, g.H_barrel, g.H_total, mesh_size=MESH_SIZE)
    m2 = MESH.build_fairing_interior_mesh(g.R_fair, g.H_barrel, g.H_total, mesh_size=MESH_SIZE)
    fp1, fp2 = mesh_fingerprint(m1), mesh_fingerprint(m2)
    if fp1 != fp2:
        raise SystemExit(f"mesh is NON-DETERMINISTIC across builds ({fp1[:12]} != {fp2[:12]}); "
                         "cannot freeze a mesh-keyed v1 reference on this toolchain.")
    mesh = m1
    base_v1, wall_v1 = FIN._facet_groups(mesh)
    bands = [float(f) for f in scen.band_hz]
    v1_band = [float(FIN.solve_interior(mesh, base_v1, wall_v1, f, zsn=(lambda ff: ZETA_WALL))[0]) for f in bands]
    v1_grid = [float(FIN.solve_interior(mesh, base_v1, wall_v1, f, zsn=(lambda ff: ZETA_WALL))[0]) for f in GRID]
    v1_rigid_grid = [float(FIN.solve_interior(mesh, base_v1, wall_v1, f, zsn=None)[0]) for f in GRID]

    doc = dict(
        fixture="v1_fem_continuity_reference",
        purpose="frozen v1 fem_interior_nr.solve_interior outputs for the SELF-CONTAINED continuity gate (G1b)",
        v1_solver="fem_interior_nr.solve_interior",
        scenario="falcon9_2025_blanketed",
        scenario_sha256=None,
        mesh_size=MESH_SIZE, mesh_fingerprint=fp1,
        mesh_R_fair=float(g.R_fair), mesh_H_barrel=float(g.H_barrel), mesh_H_total=float(g.H_total),
        zeta_wall=[ZETA_WALL.real, ZETA_WALL.imag],
        design_bands_hz=bands, grid_hz=[float(f) for f in GRID],
        v1_band_ms=v1_band, v1_grid_ms=v1_grid, v1_rigid_grid_ms=v1_rigid_grid,
        toolchain=dict(note="skfem/gmsh pinned via requirements.lock; regenerate only after a reviewed toolchain change"),
    )
    try:
        from ..evaluation import requests as R
        doc["scenario_sha256"] = R.canonical_scenario_sha256(scen)
    except Exception:
        pass
    doc["self_sha256"] = _self_sha256(doc)
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    with open(FIXTURE, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {FIXTURE}\n  mesh_fingerprint {fp1}\n  self_sha256 {doc['self_sha256']}\n  "
          f"{len(bands)} design bands, {len(GRID)} grid freqs")


if __name__ == "__main__":
    from .. import cli
    if cli.show_help("provenance", "continuity-fixture"):
        raise SystemExit(0)
    build_reference()
