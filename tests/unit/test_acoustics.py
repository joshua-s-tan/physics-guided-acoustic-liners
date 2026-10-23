# -*- coding: utf-8 -*-
"""Production diffuse-absorption mapping (§4.2 #3): monotone, invertible, round-trip exact."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acoustic_liner.physics import absorption as A  # noqa: E402


def test_diffuse_alpha_monotone_and_bounded():
    a1 = A.diffuse_alpha_from_zeta(1.0)          # matched-ish -> highest
    a3 = A.diffuse_alpha_from_zeta(3.0)
    a50 = A.diffuse_alpha_from_zeta(50.0)         # near-rigid -> low
    assert 0.0 < a50 < a3 < a1 <= 1.0


def test_roundtrip_alpha_zeta_alpha():
    for target in (0.1, 0.3, 0.5, 0.7):
        zeta = A.zeta_from_diffuse_alpha(target)
        assert zeta >= 1.0
        assert A.diffuse_alpha_from_zeta(zeta) == pytest.approx(target, abs=1e-6)


def test_admittance_roundtrip_uses_diffuse_mapping():
    Z = A.Z_AIR
    for target in (0.15, 0.4, 0.6):
        Y = A.admittance_from_diffuse_alpha(target, Z)
        assert Y.real > 0
        assert A.diffuse_alpha_from_admittance(Y, Z) == pytest.approx(target, abs=1e-6)


def test_bad_inputs_rejected():
    with pytest.raises(A.AcousticsError):
        A.diffuse_alpha_from_zeta(0.5)            # zeta < 1
    with pytest.raises(A.AcousticsError):
        A.zeta_from_diffuse_alpha(1.5)            # alpha not in (0,1)
    with pytest.raises(A.AcousticsError):
        A.admittance_from_diffuse_alpha(0.3, Z_air=0.0)
    with pytest.raises(A.AcousticsError):
        A.diffuse_alpha_from_admittance(-0.01 + 0j)   # non-passive
