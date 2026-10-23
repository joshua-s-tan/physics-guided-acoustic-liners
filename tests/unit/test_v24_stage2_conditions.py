# -*- coding: utf-8 -*-
"""`acoustic_liner.evaluation.historical_splits.v24_conditions.falcon_conditions_both_scenarios`: the Falcon conditions the PHYS-GEN leakage
barrier hashes. (phys-gen: the v2.4 stage-2 expert-data gate tests were removed with that code.)
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner.evaluation.historical_splits import v24_conditions as V24C  # noqa: E402


@pytest.fixture(scope="module")
def falcon_conds():
    return V24C.falcon_conditions_both_scenarios()


# --------------------------------------------------------------------------- #
#  Falcon (both scenarios)
# --------------------------------------------------------------------------- #
def test_falcon_both_scenarios_shape(falcon_conds):
    assert len(falcon_conds) == 3          # blanketed, unblanketed west, unblanketed east
    ids = {c["id"] for c in falcon_conds}
    assert "falcon9_2025_blanketed" in ids
    assert "falcon9_2025_unblanketed_west" in ids
    assert "falcon9_2025_unblanketed_east" in ids
    for c in falcon_conds:
        assert len(c["weights"]) == 8
        assert abs(sum(c["weights"]) - 1.0) < 1e-6
        assert c["A_wall"] > 0.0
