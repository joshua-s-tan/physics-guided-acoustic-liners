"""Operating conditions of the evaluation: the synthetic condition family, the Falcon condition, and the
condition and split identities (SHA-256 of canonical JSON).
"""
from __future__ import annotations

import hashlib
import numpy as np
from .historical_splits import selection_rule as SEL
from ..fem import scenarios as S
import os, sys, json, csv, argparse, hashlib, platform


FALCON_TARGET_DB = 0.9213                 # the owner's unchanged Falcon bar
FALCON_MASS_TARGET_KG = 400.0             # the documented Falcon request mass budget
DEV_TARGET_FRAC = 0.5                     # approved default 7 (= v2.1 confirmatory rule)


# --------------------------------------------------------------------------- #
#  Conditions
# --------------------------------------------------------------------------- #
def _from_request(r, group):
    return dict(id=r["id"], group=group, weights=[float(x) for x in r["weights"]],
                spl=[float(x) for x in r["spl_dB"]], dOASPL_target=float(r["dOASPL_target"]),
                m_total_target=float(r["m_total_target"]), A_wall=float(r["A_wall"]), kappa=float(r["kappa"]))


def synthetic_conditions(n, seed, group):
    """Deterministic synthetic conditions (identical generator to the v2.1 confirmatory set: independent spectrum
    and environment streams; target = DEV_TARGET_FRAC x representative achievable max, floor 0.25 dB)."""
    return [_from_request(r, group) for r in SEL.build_fresh_test_requests(n=n, seed=seed, target_frac=DEV_TARGET_FRAC,
                                                                          group=group)]


def falcon_condition():
    scen = S.falcon9_2025_blanketed()
    spl = np.asarray(scen.spl_dB, float)
    return dict(id="falcon9_2025_blanketed", group="falcon_ood_dev", weights=list(map(float, SEL._weights_from_spl(spl))),
                spl=spl.tolist(), dOASPL_target=FALCON_TARGET_DB, m_total_target=FALCON_MASS_TARGET_KG,
                A_wall=float(scen.A_wall), kappa=float(scen.kappa_nominal))


def condition_hash(c) -> str:
    view = {k: c[k] for k in ("weights", "spl", "dOASPL_target", "m_total_target", "A_wall", "kappa")}
    return hashlib.sha256(_canonical_json(view).encode()).hexdigest()


def split_hash(conds) -> str:
    return hashlib.sha256(_canonical_json([condition_hash(c) for c in conds]).encode()).hexdigest()


def _round(x):
    """Round a float to 12 significant digits so the canonical fingerprint ignores repr noise but not real
    numeric change. Non-floats pass through."""
    if isinstance(x, float):
        if not np.isfinite(x):
            return str(x)                          # 'nan'/'inf' are stringified (and rejected upstream anyway)
        return float(f"{x:.12g}")
    return x


def _canonicalize(obj):
    if isinstance(obj, dict):
        return {k: _canonicalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    if isinstance(obj, float):
        return _round(obj)
    if isinstance(obj, (np.floating,)):
        return _round(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _canonical_json(obj) -> str:
    return json.dumps(_canonicalize(obj), sort_keys=True, separators=(",", ":"))
