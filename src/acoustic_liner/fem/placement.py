# -*- coding: utf-8 -*-
"""Robust-placement aggregation over pre-registered uncertainty/placement corners (§4.4; §4.8 #10).

The robustness metric per corner is a spectrum-weighted interior OASPL reduction dOASPL [dB]; a design's ROBUST
value is the WORST case = the MINIMUM dOASPL over the corner set (higher dOASPL is better, so the worst corner
is the smallest). This module owns the FAIL-CLOSED selection that Step 8 inherits:

  * the corner set must EXACTLY equal the pre-registered set -- missing, duplicated, or unexpected corners fail;
  * every corner must be OK and carry a finite value -- a failed or non-finite corner fails closed;
  * the minimum AND its corner identity are recomputed INDEPENDENTLY here (never trusted from an upstream field);
  * selection is ORDER-INVARIANT (records are keyed and canonically sorted, so shuffling cannot change it);
  * ties are broken DETERMINISTICALLY by the canonical corner-id ordering.

It computes nothing scientific beyond the min; the per-corner values are produced upstream (e.g. from the FEM).
"""
from __future__ import annotations

import math
from typing import Iterable


class RobustSelectionError(RuntimeError):
    """A robust-placement corner set or value failed the fail-closed selection gate."""


def _canonical_id(cid) -> tuple:
    """Normalize a corner id to a tuple of strings (so ids are hashable, comparable, and order-invariant)."""
    if isinstance(cid, (list, tuple)):
        return tuple(str(x) for x in cid)
    return (str(cid),)


def _canonical_key(cid: tuple) -> str:
    return "|".join(cid)


def select_robust_minimum(corner_records: Iterable[dict], expected_corner_ids) -> dict:
    """Select the robust (worst-case = minimum) dOASPL over the pre-registered corners, fail-closed.

    `corner_records`: an iterable of dicts, each with
        - "corner_id": hashable id (a tuple/list of labels, e.g. ("kappa_hi", "high"), or a string);
        - "value":     the corner's dOASPL [dB] (the metric MINIMIZED across corners);
        - "ok":        bool, True iff the corner completed successfully.
    `expected_corner_ids`: the EXACT pre-registered corner-id set that must all be present (no more, no less).

    Returns dict(robust_value, robust_corner_id, per_corner, n_corners, tie_broken, expected_corner_ids).
    Raises RobustSelectionError on a missing / duplicated / unexpected / non-finite / failed corner.
    """
    expected = [_canonical_id(c) for c in expected_corner_ids]
    expected_set = set(expected)
    if len(expected_set) != len(expected):
        raise RobustSelectionError("expected_corner_ids contains duplicates -- the pre-registration is ill-formed")
    if not expected_set:
        raise RobustSelectionError("expected_corner_ids is empty -- no corners were pre-registered")

    values: dict[tuple, float] = {}
    for rec in corner_records:
        if not isinstance(rec, dict):
            raise RobustSelectionError(f"corner record must be a dict; got {type(rec).__name__}")
        if "corner_id" not in rec or "value" not in rec or "ok" not in rec:
            raise RobustSelectionError(f"corner record missing required keys (corner_id/value/ok): {rec!r}")
        cid = _canonical_id(rec["corner_id"])
        if cid in values:
            raise RobustSelectionError(f"duplicate corner {cid} -- each corner may appear exactly once")
        if rec["ok"] is not True:
            raise RobustSelectionError(f"corner {cid} is not ok (failed corner) -- fail closed")
        v = rec["value"]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise RobustSelectionError(f"corner {cid} value must be a finite real number; got {v!r}")
        values[cid] = float(v)

    seen = set(values)
    if seen != expected_set:
        missing = sorted(_canonical_key(c) for c in (expected_set - seen))
        unexpected = sorted(_canonical_key(c) for c in (seen - expected_set))
        raise RobustSelectionError(
            f"corner set does not match the pre-registered set exactly; missing={missing}, unexpected={unexpected}")

    # Independent recomputation of the minimum AND its corner identity (never trusted from upstream).
    min_val = min(values.values())
    tied = sorted((cid for cid, v in values.items() if v == min_val), key=_canonical_key)  # deterministic tie-break
    robust_cid = tied[0]

    return dict(
        robust_value=min_val,
        robust_corner_id=robust_cid,
        per_corner={_canonical_key(cid): v for cid, v in values.items()},
        n_corners=len(values),
        tie_broken=len(tied) > 1,
        expected_corner_ids=[_canonical_key(c) for c in expected],
    )
