# -*- coding: utf-8 -*-
"""Complete geometric feasibility checks (remediation §4.7).

Replaces the historical partial check (depth / neck-radius / areal-mass / area-sum) with the full constraint
set, and gives inactive banks an EXPLICIT convention so a dormant, physically-impossible value (the Electron
CMA bank had `L_neck > D`) is masked/canonicalized rather than silently retained.

Physics-light and injectable, like the rest of the release package: each group supplies its already-derived
`cavity_volume` [m^3] and neck geometry; this module checks manufacturability and tiling, it does not re-run
the resonator model. `check_feasibility` returns a structured result (never a bare bool) with per-constraint
reasons, a canonicalized design, and the explicit truncated-cell rule in force.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional
from ..physics.oracle import _enrich_groups

# Only rules that are ACTUALLY IMPLEMENTED are exposed (review round 4.1-e finding 7). The continuous
# analytic areal mass formula (physics.mass, shared-wall 2*sqrt(n)) IMPLEMENTS `count_fractional_area`:
# boundary cells contribute their fractional footprint. Finite-panel rules (exclude/round-down) would need
# explicit finite-panel geometry and are intentionally NOT offered until that geometry exists.
_TRUNCATED_CELL_RULES = ("count_fractional_area",)


class FeasibilityInputError(ValueError):
    """The design record is malformed (missing fields, non-finite values) -- distinct from an INFEASIBLE
    but well-formed design, which is reported in the result, not raised."""


@dataclass(frozen=True)
class FeasibilityLimits:
    """Documented manufacturing limits / conventions. Placeholders to be replaced by a real process spec;
    values are explicit so the feasibility verdict is reproducible and auditable."""
    min_neck_r_m: float = 0.0005          # >= 0.5 mm drillable (v1)
    max_neck_r_m: float = 0.050           # <= 50 mm neck radius (manufacturing upper bound, placeholder)
    max_depth_m: float = 0.050            # <= 50 mm radial stand-off (backing depth D)
    allow_folded_neck: bool = False       # if False, require 0 < L_neck <= D
    min_wall_t_m: float = 0.0003          # 0.3 mm min partition wall
    max_wall_t_m: float = 0.005           # 5 mm max
    min_face_t_m: float = 0.0003          # 0.3 mm min facesheet
    max_face_t_m: float = 0.005           # 5 mm max
    ligament_min_m: float = 0.0010        # min material between a neck opening edge and the cell wall
    sigma_min: float = 0.01               # min open-area (porosity) ratio
    sigma_max: float = 0.25               # max open-area ratio
    area_frac_sum_tol: float = 0.02       # |sum(area_frac) - 1| tolerance
    inactive_area_frac_eps: float = 1e-6  # area_frac <= eps -> bank is INACTIVE
    truncated_cell_rule: str = "count_fractional_area"

    def __post_init__(self):
        if self.truncated_cell_rule not in _TRUNCATED_CELL_RULES:
            raise FeasibilityInputError(
                f"truncated_cell_rule {self.truncated_cell_rule!r} not in {_TRUNCATED_CELL_RULES}")
        # Validate the limits THEMSELVES (review round 4.1-f): finite, non-negative, and internally
        # consistent -- contradictory limits would silently pass or fail every design.
        for nm in ("min_neck_r_m", "max_neck_r_m", "max_depth_m", "min_wall_t_m", "max_wall_t_m",
                   "min_face_t_m", "max_face_t_m", "ligament_min_m", "sigma_min", "sigma_max",
                   "area_frac_sum_tol", "inactive_area_frac_eps"):
            v = getattr(self, nm)
            if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) and float(v) >= 0):
                raise FeasibilityInputError(f"FeasibilityLimits.{nm} must be a finite number >= 0; got {v!r}")
        for lo, hi, label in ((self.min_neck_r_m, self.max_neck_r_m, "neck_r"),
                              (self.min_wall_t_m, self.max_wall_t_m, "wall_t"),
                              (self.min_face_t_m, self.max_face_t_m, "face_t"),
                              (self.sigma_min, self.sigma_max, "sigma")):
            if not (lo < hi):
                raise FeasibilityInputError(f"{label} range invalid: min {lo} must be < max {hi}")
        if not (0.0 <= self.area_frac_sum_tol < 1.0):
            raise FeasibilityInputError(f"area_frac_sum_tol must be in [0,1); got {self.area_frac_sum_tol}")
        if not (self.max_depth_m > 0):
            raise FeasibilityInputError(f"max_depth_m must be > 0; got {self.max_depth_m}")
        if not (0.0 < self.inactive_area_frac_eps < 1.0):
            raise FeasibilityInputError(f"inactive_area_frac_eps must be in (0,1); got {self.inactive_area_frac_eps}")


def _finite(name: str, v) -> float:
    if v is None:
        raise FeasibilityInputError(f"{name} is required but missing (None)")
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise FeasibilityInputError(f"{name} must be a real number; got {v!r}")
    if not math.isfinite(x):
        raise FeasibilityInputError(f"{name} must be finite; got {v}")
    return x


def _neck_area(r_neck: float) -> float:
    return math.pi * r_neck * r_neck


def check_feasibility(design: dict, limits: Optional[FeasibilityLimits] = None,
                      coverage: Optional[float] = None, require_coverage: bool = False) -> dict:
    """Check one design. Returns a dict:
        {feasible: bool, failures: [str], per_group: [{index, active, failures}], area_frac_sum: float,
         coverage: float|None, canonicalized: <design copy with inactive banks masked>,
         truncated_cell_rule: str, inactive_banks: [int]}
    A malformed record raises FeasibilityInputError; an INFEASIBLE well-formed record is reported, not raised.

    `require_coverage=True` (the release/FEM installed-design path) makes a missing `coverage` an INFEASIBILITY,
    so local-cell feasibility can never be mistaken for complete installed-design feasibility. The coverage
    <-> mesh-area <-> mass-area invariant itself is enforced in the FEM path (§4.2/step 6), which passes the
    achieved value here."""
    lim = limits or FeasibilityLimits()
    if not isinstance(design, dict) or "groups" not in design:
        raise FeasibilityInputError("design must be a dict with a 'groups' list")
    groups = design["groups"]
    if not groups:
        raise FeasibilityInputError("design has no groups")
    D = _finite("D", design.get("D"))
    face_t = _finite("face_t", design.get("face_t", 0.0008))
    wall_t = _finite("wall_t", design.get("wall_t", 0.0006))

    failures: list[str] = []

    # --- design-level constraints ---
    if not (0.0 < D <= lim.max_depth_m + 1e-12):
        failures.append(f"backing depth D={D} not in (0, {lim.max_depth_m}]")
    if not (lim.min_face_t_m <= face_t <= lim.max_face_t_m):
        failures.append(f"facesheet thickness {face_t} outside [{lim.min_face_t_m}, {lim.max_face_t_m}]")
    if not (lim.min_wall_t_m <= wall_t <= lim.max_wall_t_m):
        failures.append(f"wall thickness {wall_t} outside [{lim.min_wall_t_m}, {lim.max_wall_t_m}]")

    if coverage is not None:
        coverage = _finite("coverage", coverage)
        if not (0.0 <= coverage <= 1.0):
            failures.append(f"coverage {coverage} not in [0,1]")
    elif require_coverage:
        failures.append("coverage is required for installed-design feasibility but was not supplied")

    # --- area-fraction sum over ACTIVE banks must be ~1 ---
    area_frac_sum = 0.0
    per_group = []
    inactive_banks = []
    canon_groups = []

    for i, g in enumerate(groups):
        af = _finite(f"group[{i}].area_frac", g.get("area_frac"))
        gf: list[str] = []

        # (1) area fraction MUST be in [0,1] BEFORE any inactive classification -- a negative fraction is
        # infeasible, never "inactive" (review round 4.1-e finding 1).
        if not (0.0 <= af <= 1.0 + 1e-9):
            gf.append(f"area_frac {af} not in [0,1]")
            failures.extend(f"group[{i}]: {m}" for m in gf)
            per_group.append(dict(index=i, active=None, failures=gf, masked=False))
            canon_groups.append(dict(g))               # left untouched; the design is already infeasible
            continue

        active = af > lim.inactive_area_frac_eps
        if not active:
            inactive_banks.append(i)
            # (6) FULLY canonicalize an inactive bank: discard the (possibly impossible) dormant geometry and
            # write ONE deterministic valid placeholder cell, so nothing physically impossible is retained
            # (the Electron CMA L_neck>D / cavity_volume=-1 case). Only standard fields + _inactive remain.
            s0 = 2.0 * (lim.min_neck_r_m + lim.ligament_min_m)      # smallest cell that fits the min neck
            D_ref = D if (math.isfinite(D) and D > 0) else lim.max_depth_m
            # FULLY deterministic placeholder: no field (f0 included) is taken from the dormant input.
            cg = dict(f0=0.0, area_frac=0.0, r_neck=lim.min_neck_r_m,
                      L_neck=min(0.5 * D_ref, D_ref), cavity_volume=s0 * s0 * D_ref, _inactive=True)
            per_group.append(dict(index=i, active=False, failures=[], masked=True))
            canon_groups.append(cg)
            continue

        area_frac_sum += af
        r_neck = _finite(f"group[{i}].r_neck", g.get("r_neck"))
        L_neck = _finite(f"group[{i}].L_neck", g.get("L_neck"))
        V_cav = _finite(f"group[{i}].cavity_volume", g.get("cavity_volume"))

        # neck radius within manufacturing range
        if not (lim.min_neck_r_m <= r_neck <= lim.max_neck_r_m):
            gf.append(f"r_neck {r_neck} outside [{lim.min_neck_r_m}, {lim.max_neck_r_m}]")
        # (8) neck length: require 0 < L_neck <= D. A folded/protruding neck is NOT feasible yet: even with
        # allow_folded_neck=True we flag it as unsupported (its mass/volume effects are not modeled), so it
        # can never yield a feasible verdict (review round 4.1-e finding 8).
        if L_neck <= 0:
            gf.append(f"L_neck {L_neck} must be > 0")
        elif L_neck > D:
            if lim.allow_folded_neck:
                gf.append(f"L_neck {L_neck} > D {D}: folded/protruding-neck geometry and its mass/volume "
                          f"effects are NOT modeled yet (unsupported)")
            else:
                gf.append(f"L_neck {L_neck} > D {D} (folded/protruding neck not enabled)")
        # cavity volume positive
        if V_cav <= 0:
            gf.append(f"cavity_volume {V_cav} must be > 0")
        # (5) only compute footprint when BOTH D and V_cav are positive -- never divide by a nonpositive D
        # (a nonpositive D is already recorded as a design-level failure; here we just avoid the crash).
        if D > 0 and V_cav > 0:
            a_foot = V_cav / D
            s = math.sqrt(a_foot)
            # neck opening must fit inside its cell with the required ligament on each side:
            # 2*r_neck + 2*ligament <= s   (=> no overlapping geometry within the cell)
            if 2.0 * r_neck + 2.0 * lim.ligament_min_m > s + 1e-12:
                gf.append(f"neck opening (2r={2*r_neck:.4g}) + ligament exceeds cell side s={s:.4g}")
            # porosity / open-area ratio = neck open area / cell footprint, within manufacturing range
            sigma = _neck_area(r_neck) / a_foot
            if not (lim.sigma_min <= sigma <= lim.sigma_max):
                gf.append(f"open-area ratio sigma={sigma:.4g} outside [{lim.sigma_min}, {lim.sigma_max}]")

        per_group.append(dict(index=i, active=True, failures=gf, masked=False))
        failures.extend(f"group[{i}]: {m}" for m in gf)
        canon_groups.append(dict(g))

    if abs(area_frac_sum - 1.0) > lim.area_frac_sum_tol:
        failures.append(f"active area fractions sum to {area_frac_sum:.4f} (must be 1 +/- {lim.area_frac_sum_tol})")

    canonicalized = dict(design)
    canonicalized["groups"] = canon_groups

    return dict(feasible=(len(failures) == 0), failures=failures, per_group=per_group,
                area_frac_sum=area_frac_sum, coverage=coverage, canonicalized=canonicalized,
                truncated_cell_rule=lim.truncated_cell_rule, inactive_banks=inactive_banks)


def is_feasible(design, f_cov):
    """Full geometric feasibility (require_coverage=True). Returns (bool, result). Malformed/non-finite inputs
    make it False (fail-closed) rather than raising, so the builder can reject-and-continue."""
    try:
        res = check_feasibility(dict(design, groups=_enrich_groups(design)),
                                     coverage=float(f_cov), require_coverage=True)
        return bool(res["feasible"]), res
    except FeasibilityInputError:
        return False, None
