# -*- coding: utf-8 -*-
"""OPTIONAL integration test (SK-1, owner decision 2026-10-09): re-verify the committed v1 continuity fixture
against the LIVE historical v1 solver. It is not collected by the normal suite (`pytest`, `pytest tests`, or
`pytest tests/optional` without --v1-reference); run it explicitly:

    ACOUSTIC_LINER_V1_REFERENCE_TREE=<historical v1 tree> pytest --v1-reference tests/optional
    ACOUSTIC_LINER_V1_REFERENCE_TREE=<historical v1 tree> pytest tests/optional/test_v1_live_reference.py

It needs the historical v1 tree outside this repository. Invoked without it, it fails (it does not skip). The
mandatory continuity gate (tests/integration/test_v1_v2_continuity.py) runs self-contained on the committed fixture.
"""
import importlib.util
import os
import sys

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "_v1_v2_continuity", os.path.join(os.path.dirname(__file__), "..", "integration", "test_v1_v2_continuity.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)


def test_frozen_fixture_still_matches_live_v1():
    """Maintenance regression: re-run the LIVE v1 solver on the shared mesh and confirm the COMMITTED fixture still
    reproduces it exactly (guards fixture staleness)."""
    v1_root = os.environ.get("ACOUSTIC_LINER_V1_REFERENCE_TREE")
    assert v1_root and os.path.isdir(v1_root), \
        "this optional test needs the historical v1 tree: set ACOUSTIC_LINER_V1_REFERENCE_TREE"
    for p in (v1_root, os.path.join(v1_root, "fairing_case")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import fem_interior_nr as FIN
    mesh = C._build_shared_mesh()
    base_v1, wall_v1 = FIN._facet_groups(mesh)
    live_band = np.array([FIN.solve_interior(mesh, base_v1, wall_v1, f, zsn=(lambda ff: C.ZETA_WALL))[0]
                          for f in C.DESIGN_BANDS])
    assert np.allclose(live_band, np.asarray(C.V1REF["v1_band_ms"], float), rtol=1e-9, atol=0.0), \
        "committed v1 continuity fixture no longer matches the live v1 solver -- re-freeze after review"
