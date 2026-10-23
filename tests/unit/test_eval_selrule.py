# -*- coding: utf-8 -*-
"""The post-training eval harness (acoustic_liner.evaluation.historical_splits.selection_rule) must be SELF-CONTAINED (no legacy/reference-tree eval
import; G1b) and produce well-formed generate-verify-rerank metrics against the migrated oracle + feasibility."""
import ast
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("cma")
pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner.evaluation.historical_splits import selection_rule as SEL  # noqa: E402

NET = dict(latent=6, hidden=32)


def test_eval_harness_is_self_contained():
    """It may import only from within acoustic_liner (relative) or the scientific stack; NEVER a legacy `evaluate`
    module or anything under a reference/fairing tree (preserves G1b self-containment)."""
    src = (ROOT / "src" / "acoustic_liner" / "evaluation" / "historical_splits" / "selection_rule.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    abs_imports = set()
    rel_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            abs_imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:                 # relative import within the package
                if node.module:
                    rel_modules.add(node.module.split(".")[0])
            elif node.module:
                abs_imports.add(node.module.split(".")[0])
    allowed_abs = {"numpy", "json", "os", "sys", "time", "math", "__future__"}
    assert abs_imports <= allowed_abs, f"unexpected absolute imports: {abs_imports - allowed_abs}"
    assert "evaluate" not in rel_modules and "evaluate" not in abs_imports
    assert not any("fairing" in m or "reference" in m for m in abs_imports | rel_modules)


def test_fresh_requests_use_independent_rng_streams():
    """Spectrum and environment (A_wall/kappa) must come from SEPARATELY spawned RNG streams, and a different
    seed/group yields an independent (confirmatory) set - different in BOTH spectra and A_wall."""
    dev = SEL.build_fresh_test_requests(n=6, seed=31415)
    conf = SEL.build_fresh_test_requests(n=6, seed=27182, group="confirmatory")
    assert [r["A_wall"] for r in dev] != [r["A_wall"] for r in conf]
    assert dev[0]["spl_dB"] != conf[0]["spl_dB"]
    assert all(r["group"] == "confirmatory" for r in conf)
    # environment draws are not a deterministic echo of the spectrum stream: A_wall varies across requests
    assert len({round(r["A_wall"], 3) for r in dev}) > 1
