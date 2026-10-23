# -*- coding: utf-8 -*-
"""cVAE v2.7 differentiable-oracle unit tests: forward parity with oracle.label_v2 and usable autograd."""
import io
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from acoustic_liner.physics import differentiable as VT

_EXPERT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures", "torch_oracle_rows.json")


def _rows(n=12):
    d = json.load(io.open(_EXPERT, encoding="utf-8"))
    rows = d["rows"] if isinstance(d, dict) and "rows" in d else d
    return rows[:n]


def test_forward_matches_label_v2():
    """The torch oracle reproduces label_v2's dOASPL_worst and total mass to ~machine precision."""
    orc = VT.TorchOracle()
    for r in _rows():
        dvec = torch.tensor(r["decoder_vector"], dtype=torch.float64)
        dO, mT = orc.evaluate(dvec, r["spectrum"]["spl_dB"], r["A_wall"], r["kappa_nominal"])
        assert abs(float(dO) - r["dOASPL_worst_dB"]) <= 1e-6 * max(1.0, abs(r["dOASPL_worst_dB"]))
        assert abs(float(mT) - r["total_installed_mass_kg"]) <= 1e-6 * max(1.0, r["total_installed_mass_kg"])


def test_autograd_finite_and_nonzero():
    """Backprop through the frozen path yields finite gradients w.r.t. the decoder vector."""
    orc = VT.TorchOracle()
    r = _rows(1)[0]
    dvec = torch.tensor(r["decoder_vector"], dtype=torch.float64, requires_grad=True)
    dO, mT = orc.evaluate(dvec, r["spectrum"]["spl_dB"], r["A_wall"], r["kappa_nominal"])
    g = torch.autograd.grad(dO, dvec, retain_graph=True)[0]
    assert torch.all(torch.isfinite(g))
    assert float(torch.linalg.norm(g)) > 0.0
    gm = torch.autograd.grad(mT, dvec)[0]
    assert torch.all(torch.isfinite(gm))
