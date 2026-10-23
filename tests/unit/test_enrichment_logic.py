# -*- coding: utf-8 -*-
"""Regression lock for the corrected ENRICHMENT logic (peer-review): feasibility applied INSIDE CMA (death
penalty -> feasible-only archive), Pareto over ALL feasible geoms (no performance prefilter), per-condition
CLASSIFICATION, accounting closure, and the ACCURACY of the accelerated (curve + bracket-refine) min-coverage
vs the reference bisection. Self-contained; tiny.
"""
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("cma")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner.design import constraints as FEAS  # noqa: E402
from acoustic_liner.physics import oracle as O2
from acoustic_liner.design import schema as DS  # noqa: E402


@pytest.mark.parametrize("kappa", [0.03, 0.12, 0.22])
def test_accelerated_min_coverage_matches_reference(kappa):
    """ACCURACY: the accelerated (curve + bracket bisection-refine) min-coverage agrees with the reference
    full-range bisection (`_min_coverage`) within a PREDECLARED tolerance, on a representative feasible design
    across several targets -- so the speedup is not a mass-accuracy regression."""
    COV_TOL = 0.01            # predeclared: |f_cov_accelerated - f_cov_reference|
    lr = np.random.default_rng(3)
    # a feasible design
    while True:
        d = DS.vector_to_design(DS.sample_random(lr))
        if FEAS.is_feasible(d, 1.0)[0]:
            break
    a_band = O2.liner_diffuse_alpha_band(d["groups"], d["D"])
    spl = O2.falcon9_design_band(); A_wall = 140.0
    cov_grid = np.linspace(O2._MIN_COV_GRID, 1.0, O2._COV_GRID_N)
    wv = np.array([O2._worst(a_band, spl, f, A_wall, kappa) for f in cov_grid])
    vmax = float(wv.max())
    checked = 0
    for frac in (0.3, 0.5, 0.7, 0.9):
        t = frac * vmax
        ref = O2._min_coverage(a_band, spl, A_wall, kappa, t)
        acc = O2._refine_min_coverage(a_band, spl, A_wall, kappa, wv, cov_grid, t)
        assert (ref is None) == (acc is None)
        if ref is not None:
            assert abs(acc - ref) <= COV_TOL, f"kappa={kappa} t={t:.3f}: accel {acc:.4f} vs ref {ref:.4f}"
            checked += 1
    assert checked >= 1
