# -*- coding: utf-8 -*-
"""Authoritative scenario objects (remediation §3 rule 3, §4.1; review round 2).

ONE serialized record drives every downstream layer. The YAML config in `configs/` is the SINGLE source
of truth; this module LOADS and VALIDATES it and never duplicates its values. Geometry-derived A_wall /
volume close the oracle->FEM scenario mismatch (historical F1 used a synthetic A_wall=83.78 m^2 vs the FEM
geometry 165.878 m^2).

Two request types are strictly separate:
  * PhysicalScenario           -- a named vehicle / qualification envelope; geometry-DERIVED A_wall+volume.
  * SyntheticBenchmarkRequest  -- randomized A_wall/kappa/target for the ML benchmark ONLY.

Hardening (review round 2):
  * fail-closed gate raises `ScenarioConsistencyError` (NOT `assert`, which vanishes under `python -O`);
  * `assert_fem_consistency` takes an explicit `expected_kappa` so any uncertainty corner can be validated;
  * `__post_init__` validates geometry, band/level counts, monotone band centers, ranges-contain-nominal,
    Q, fill factor, finiteness, and plausible SPL magnitudes;
  * scenarios carry `oaspl_dB` + `pressure_ref_uPa`; an in-band energy audit guards against a wrong
    spectrum (in-band 50-250 Hz energy must not exceed the listed all-frequency OASPL).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence
from pathlib import Path
import math

DESIGN_BAND_HZ = (50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0)
DEFAULT_FEM_TOL = 0.005          # 0.5% area/volume/kappa consistency tolerance (remediation §4.1)
_SPL_MIN_DB, _SPL_MAX_DB = 80.0, 180.0   # plausible fairing-interior 1/3-octave band level bounds


class ScenarioValidationError(ValueError):
    """A scenario/request record is internally invalid (bad geometry, counts, ranges, levels, ...)."""


class ScenarioConsistencyError(RuntimeError):
    """The FEM mesh does not integrate to the scenario's derived area/volume/kappa within tolerance.
    A RuntimeError subclass (NOT assert) so the gate survives `python -O`."""


class ClaimNotAllowedError(RuntimeError):
    """A result/claim was requested that the scenario's frozen semantics forbid (e.g. an absolute
    post-treatment SPL for an incremental-treatment scenario)."""


_ALLOWED_METRICS = ("relative_dOASPL_only",)
_ALLOWED_SOURCES = ("modelling_assumption", "measured", "published")
_CLAIM_TYPES = ("relative_dOASPL", "absolute_post_treatment_spl", "validated_retrofit", "blanket_replacement")
# Geometry kinds we allow. A named vehicle scenario must NOT be labelled exact CAD unless it is one
# (review round 4/4.1); today only the representative axisymmetric cone approximation exists.
_ALLOWED_GEOMETRY_KINDS = ("representative_axisymmetric_cone_approximation",)


def _require_bool(value, label: str) -> bool:
    """Fail-closed boolean: reject `bool("false") -> True` style coercion (review round 4.1, finding 3).
    Only a genuine YAML/Python bool is accepted; a quoted string, 0/1, or None raises."""
    if not isinstance(value, bool):
        raise ScenarioValidationError(
            f"{label} must be a real boolean (YAML true/false), not {type(value).__name__} {value!r}")
    return value


@dataclass(frozen=True)
class ClaimSemantics:
    """Machine-readable, ENFORCEABLE claim policy for a scenario (review round 4, finding 2). Loaded from
    the YAML `semantics:` block, carried on the scenario, propagated into every artifact, and used by the
    claim ledger / provenance.release_verifier to reject forbidden claims. YAML comments are NOT enforceable; this is."""
    id: str
    metric: str
    absolute_post_treatment_spl_allowed: bool
    validated_retrofit_claim_allowed: bool
    blankets_modelled: bool
    blanket_replacement_claim_allowed: bool
    alpha_bare_source: str
    kappa_source: str

    def __post_init__(self):
        if not str(self.id).strip():
            raise ScenarioValidationError("semantics.id must be non-empty")
        if self.metric not in _ALLOWED_METRICS:
            raise ScenarioValidationError(f"semantics.metric must be one of {_ALLOWED_METRICS}; got {self.metric}")
        for fld in ("absolute_post_treatment_spl_allowed", "validated_retrofit_claim_allowed",
                    "blankets_modelled", "blanket_replacement_claim_allowed"):
            if not isinstance(getattr(self, fld), bool):
                raise ScenarioValidationError(f"semantics.{fld} must be bool")
        for fld in ("alpha_bare_source", "kappa_source"):
            if getattr(self, fld) not in _ALLOWED_SOURCES:
                raise ScenarioValidationError(f"semantics.{fld} must be one of {_ALLOWED_SOURCES}")

    def claim_allowed(self, claim: str) -> bool:
        if claim not in _CLAIM_TYPES:
            raise ValueError(f"unknown claim type {claim!r}; known: {_CLAIM_TYPES}")
        return {
            "relative_dOASPL": self.metric == "relative_dOASPL_only",
            "absolute_post_treatment_spl": self.absolute_post_treatment_spl_allowed,
            "validated_retrofit": self.validated_retrofit_claim_allowed,
            "blanket_replacement": self.blanket_replacement_claim_allowed,
        }[claim]

    def assert_claim_allowed(self, claim: str) -> None:
        if not self.claim_allowed(claim):
            raise ClaimNotAllowedError(
                f"claim {claim!r} is forbidden by scenario semantics {self.id!r} "
                f"(metric={self.metric})")


def _all_finite(xs: Sequence[float]) -> bool:
    return all(math.isfinite(float(x)) for x in xs)


@dataclass(frozen=True)
class FairingGeometry:
    """Axisymmetric fairing interior: cylindrical barrel + conical nose. Lengths in metres.

    NOTE (review round 4): this is a REPRESENTATIVE axisymmetric cone approximation, NOT verified exact
    Falcon CAD. SpaceX publishes the 5.2 m outer diameter and 13.1 m fairing height; the R_fair=2.6 m
    interior radius, the 6.7 m barrel/nose split, and the conical nose are MODELLING ASSUMPTIONS. The
    derived `lined_wall_area` (165.878 m^2) is therefore the modeled representative interior wall area, not
    the exact real Falcon interior surface. `kind`/`provenance` record this."""
    R_fair: float
    H_barrel: float
    H_total: float
    kind: str = "representative_axisymmetric_cone_approximation"
    provenance: str = ("SpaceX 5.2 m OD (-> R_fair 2.6 m interior) and 13.1 m height are published; "
                       "6.7 m barrel/nose split and conical nose are modelling assumptions (no verified CAD).")

    def __post_init__(self):
        if not _all_finite((self.R_fair, self.H_barrel, self.H_total)):
            raise ScenarioValidationError("geometry has non-finite values")
        if self.R_fair <= 0 or self.H_barrel <= 0:
            raise ScenarioValidationError("R_fair and H_barrel must be > 0")
        if self.H_total <= self.H_barrel:
            raise ScenarioValidationError(f"H_total ({self.H_total}) must exceed H_barrel ({self.H_barrel})")
        # kind/provenance must be present and honest (review round 4.1, finding 4): a scenario cannot
        # silently ship as exact CAD, and an empty provenance string is not acceptable.
        if self.kind not in _ALLOWED_GEOMETRY_KINDS:
            raise ScenarioValidationError(
                f"geometry.kind {self.kind!r} not in allowed kinds {_ALLOWED_GEOMETRY_KINDS}")
        if not str(self.provenance).strip():
            raise ScenarioValidationError("geometry.provenance must be a non-empty description of what is published vs assumed")

    @property
    def H_nose(self) -> float:
        return self.H_total - self.H_barrel

    @property
    def lined_wall_area(self) -> float:
        barrel = 2.0 * math.pi * self.R_fair * self.H_barrel
        cone = math.pi * self.R_fair * math.hypot(self.R_fair, self.H_nose)
        return barrel + cone

    @property
    def cavity_volume(self) -> float:
        return math.pi * self.R_fair ** 2 * self.H_barrel + (1.0 / 3.0) * math.pi * self.R_fair ** 2 * self.H_nose


@dataclass(frozen=True)
class SourceConfig:
    name: str
    revision: str
    url: str
    blankets: str
    fill_factor: Optional[float]
    table_page: str = ""
    range_site: str = ""
    notes: str = ""

    def __post_init__(self):
        if self.fill_factor is not None and not (0.0 < float(self.fill_factor) <= 1.0):
            raise ScenarioValidationError(f"fill_factor must be in (0,1] or null; got {self.fill_factor}")
        for fld in ("name", "revision", "url"):
            if not str(getattr(self, fld)).strip():
                raise ScenarioValidationError(f"source.{fld} must be non-empty")


@dataclass(frozen=True)
class PhysicalScenario:
    name: str
    geometry: FairingGeometry
    source: SourceConfig
    semantics: ClaimSemantics
    spl_dB: tuple
    oaspl_dB: float
    alpha_bare_nominal: float
    alpha_bare_range: tuple
    kappa_nominal: float
    kappa_eval_range: tuple
    pressure_ref_uPa: float = 20.0
    Q_fixed: float = 15.0
    band_hz: tuple = DESIGN_BAND_HZ
    is_vehicle: bool = True

    def __post_init__(self):
        if not isinstance(self.semantics, ClaimSemantics):
            raise ScenarioValidationError("semantics must be a ClaimSemantics (loaded from the YAML 'semantics:' block)")
        _validate_bands_and_levels(self.band_hz, self.spl_dB)
        _validate_range_contains(self.alpha_bare_range, self.alpha_bare_nominal, "alpha_bare", lo_min=0.0, hi_max=1.0)
        _validate_range_contains(self.kappa_eval_range, self.kappa_nominal, "kappa", lo_min=0.0, hi_max=1.0)
        if not math.isfinite(self.oaspl_dB):
            raise ScenarioValidationError("oaspl_dB must be finite")
        if not (math.isfinite(self.Q_fixed) and self.Q_fixed > 0):
            raise ScenarioValidationError(f"Q_fixed must be finite and > 0; got {self.Q_fixed}")
        if not (math.isfinite(self.pressure_ref_uPa) and self.pressure_ref_uPa > 0):
            raise ScenarioValidationError(f"pressure_ref_uPa must be finite and > 0; got {self.pressure_ref_uPa}")
        if not isinstance(self.is_vehicle, bool):
            raise ScenarioValidationError("is_vehicle must be bool")
        if not str(self.name).strip():
            raise ScenarioValidationError("scenario name must be non-empty")
        # energy audit: in-band (design bands) energy must not exceed the listed all-frequency OASPL
        if self.inband_oaspl > self.oaspl_dB + 1e-6:
            raise ScenarioValidationError(
                f"in-band OASPL {self.inband_oaspl:.2f} dB EXCEEDS listed total {self.oaspl_dB:.2f} dB "
                f"-- impossible for one spectrum (likely a wrong/mislabelled spectrum)")

    @property
    def A_wall(self) -> float:
        return self.geometry.lined_wall_area

    @property
    def volume(self) -> float:
        return self.geometry.cavity_volume

    @property
    def A_other_nominal(self) -> float:
        return self.kappa_nominal * self.A_wall

    def A_other(self, kappa: Optional[float] = None) -> float:
        return (self.kappa_nominal if kappa is None else float(kappa)) * self.A_wall

    @property
    def inband_oaspl(self) -> float:
        """Overall SPL of just the design band (energy sum), for the in-band consistency audit."""
        return 10.0 * math.log10(sum(10.0 ** (float(L) / 10.0) for L in self.spl_dB))

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(A_wall=self.A_wall, volume=self.volume, A_other_nominal=self.A_other_nominal,
                 inband_oaspl=self.inband_oaspl)
        return d


@dataclass(frozen=True)
class SyntheticBenchmarkRequest:
    """ML-benchmark request ONLY: randomized A_wall/kappa/target. Never a flight-vehicle design."""
    request_id: str
    spl_dB: tuple
    A_wall: float
    kappa: float
    target_dOASPL_dB: float
    mass_target_kg: float
    spectrum_shape_ref: str = ""
    band_hz: tuple = DESIGN_BAND_HZ
    is_vehicle: bool = False

    def __post_init__(self):
        _validate_bands_and_levels(self.band_hz, self.spl_dB)
        if not (self.A_wall > 0 and math.isfinite(self.A_wall)):
            raise ScenarioValidationError("A_wall must be finite and > 0")
        if not (0.0 <= self.kappa <= 1.0):
            raise ScenarioValidationError(f"kappa must be in [0,1]; got {self.kappa}")
        if not (math.isfinite(self.target_dOASPL_dB) and self.target_dOASPL_dB > 0):
            raise ScenarioValidationError(f"target_dOASPL_dB must be finite and > 0; got {self.target_dOASPL_dB}")
        if not (math.isfinite(self.mass_target_kg) and self.mass_target_kg > 0):
            raise ScenarioValidationError(f"mass_target_kg must be finite and > 0; got {self.mass_target_kg}")
        if not isinstance(self.is_vehicle, bool):
            raise ScenarioValidationError("is_vehicle must be bool")


def _validate_bands_and_levels(band_hz, spl_dB) -> None:
    if len(band_hz) != 8 or len(spl_dB) != 8:
        raise ScenarioValidationError(f"expected exactly 8 bands and 8 SPL values; got {len(band_hz)}/{len(spl_dB)}")
    if not _all_finite(band_hz) or not _all_finite(spl_dB):
        raise ScenarioValidationError("band/SPL values must be finite")
    if any(b2 <= b1 for b1, b2 in zip(band_hz, band_hz[1:])):
        raise ScenarioValidationError(f"band centers must be strictly increasing; got {band_hz}")
    bad = [L for L in spl_dB if not (_SPL_MIN_DB <= float(L) <= _SPL_MAX_DB)]
    if bad:
        raise ScenarioValidationError(
            f"SPL values {bad} dB outside plausible [{_SPL_MIN_DB},{_SPL_MAX_DB}] dB "
            f"-- likely a malformed record (e.g. frequencies used as levels)")


def _validate_range_contains(rng, nominal, label, lo_min, hi_max) -> None:
    if len(rng) != 2:
        raise ScenarioValidationError(f"{label}_range must be (lo, hi)")
    lo, hi = float(rng[0]), float(rng[1])
    if not (lo_min <= lo < hi <= hi_max):
        raise ScenarioValidationError(f"{label}_range {rng} must satisfy {lo_min} <= lo < hi <= {hi_max}")
    if not (lo <= float(nominal) <= hi):
        raise ScenarioValidationError(f"{label}_nominal {nominal} not inside {label}_range {rng}")


def assert_fem_consistency(scenario: PhysicalScenario, mesh_wall_area: float, mesh_volume: float,
                           A_other_applied: float, expected_kappa: float,
                           tol: float = DEFAULT_FEM_TOL) -> dict:
    """Fail-closed invariant (§4.1): the FEM mesh must integrate to the scenario's derived area/volume and
    the APPLIED kappa must match `expected_kappa` (nominal OR a specific uncertainty corner) within `tol`.
    Raises ScenarioConsistencyError on any breach (survives `python -O`). Returns achieved errors, which
    the caller MUST record in the FEM artifact."""
    for nm, val in (("mesh_wall_area", mesh_wall_area), ("mesh_volume", mesh_volume),
                    ("A_other_applied", A_other_applied), ("expected_kappa", expected_kappa), ("tol", tol)):
        if not math.isfinite(float(val)):
            raise ScenarioConsistencyError(f"{nm} must be finite; got {val}")
    if not (mesh_wall_area > 0 and mesh_volume > 0):
        raise ScenarioConsistencyError("mesh wall area and volume must be positive")
    if A_other_applied < 0:
        raise ScenarioConsistencyError(f"A_other_applied must be >= 0; got {A_other_applied}")
    if not (0.0 < tol < 1.0):
        raise ScenarioConsistencyError(f"tol must be in (0,1); got {tol}")
    if not (0.0 <= float(expected_kappa) <= 1.0):
        raise ScenarioConsistencyError(f"expected_kappa must be in [0,1]; got {expected_kappa}")
    klo, khi = scenario.kappa_eval_range
    if not (klo - 1e-9 <= float(expected_kappa) <= khi + 1e-9):
        raise ScenarioConsistencyError(
            f"expected_kappa {expected_kappa} outside the scenario's declared uncertainty interval "
            f"[{klo}, {khi}] -- an undeclared corner cannot be validated")
    e_area = abs(mesh_wall_area - scenario.A_wall) / scenario.A_wall
    e_vol = abs(mesh_volume - scenario.volume) / scenario.volume
    kappa_applied = A_other_applied / mesh_wall_area
    e_kappa = abs(kappa_applied - float(expected_kappa))
    achieved = dict(rel_area_err=e_area, rel_volume_err=e_vol, kappa_applied=kappa_applied,
                    expected_kappa=float(expected_kappa), abs_kappa_err=e_kappa, tol=tol,
                    scenario_A_wall=scenario.A_wall, mesh_wall_area=mesh_wall_area,
                    scenario_volume=scenario.volume, mesh_volume=mesh_volume)
    if e_area > tol:
        raise ScenarioConsistencyError(
            f"FEM wall area {mesh_wall_area:.4f} vs scenario {scenario.A_wall:.4f} (rel {e_area:.4f} > {tol})")
    if e_vol > tol:
        raise ScenarioConsistencyError(
            f"FEM volume {mesh_volume:.4f} vs scenario {scenario.volume:.4f} (rel {e_vol:.4f} > {tol})")
    if e_kappa > tol:
        raise ScenarioConsistencyError(
            f"applied kappa {kappa_applied:.4f} vs expected {float(expected_kappa):.4f} (abs {e_kappa:.4f} > {tol})")
    return achieved


def scenario_from_dict(d: dict) -> PhysicalScenario:
    """Reconstruct and FULLY VALIDATE a PhysicalScenario from a primitive-field mapping (YAML, or a
    `to_dict()` record round-tripped through JSON). Derived keys (A_wall/volume/A_other_nominal/inband_oaspl)
    present in a `to_dict()` record are IGNORED and RECOMPUTED from geometry/spectrum -- so a record with a
    forged/stale derived value (e.g. an injected A_wall=83.78) cannot survive reconstruction. Requires the
    complete structured geometry/source/semantics blocks; missing fields raise KeyError/ScenarioValidationError."""
    if not isinstance(d, dict):
        raise ScenarioValidationError("scenario record must be a mapping")
    for key in ("name", "geometry", "source", "semantics", "spl_dB", "oaspl_dB",
                "alpha_bare_nominal", "alpha_bare_range", "kappa_nominal", "kappa_eval_range"):
        if key not in d:
            raise ScenarioValidationError(f"scenario record missing required field {key!r}")
    g = d["geometry"]
    if not isinstance(g, dict):
        raise ScenarioValidationError("geometry must be a structured mapping (R_fair/H_barrel/H_total/kind/provenance)")
    geom = FairingGeometry(R_fair=float(g["R_fair"]), H_barrel=float(g["H_barrel"]), H_total=float(g["H_total"]),
                           kind=g.get("kind", "representative_axisymmetric_cone_approximation"),
                           provenance=g.get("provenance", ""))
    s = d["source"]
    if not isinstance(s, dict):
        raise ScenarioValidationError("source must be a structured mapping")
    src = SourceConfig(name=s["name"], revision=str(s["revision"]), url=s["url"], blankets=s["blankets"],
                       fill_factor=(None if s.get("fill_factor") is None else float(s["fill_factor"])),
                       table_page=s.get("table_page", ""), range_site=s.get("range_site", ""),
                       notes=s.get("notes", ""))
    sm = d["semantics"]
    if not isinstance(sm, dict):
        raise ScenarioValidationError("semantics must be a structured mapping (the enforceable claim block)")
    # Semantics flags and is_vehicle MUST be real booleans -- no bool("false")->True coercion (round 4.1).
    sem = ClaimSemantics(
        id=sm["id"], metric=sm["metric"],
        absolute_post_treatment_spl_allowed=_require_bool(sm["absolute_post_treatment_spl_allowed"],
                                                          "semantics.absolute_post_treatment_spl_allowed"),
        validated_retrofit_claim_allowed=_require_bool(sm["validated_retrofit_claim_allowed"],
                                                       "semantics.validated_retrofit_claim_allowed"),
        blankets_modelled=_require_bool(sm["blankets_modelled"], "semantics.blankets_modelled"),
        blanket_replacement_claim_allowed=_require_bool(sm["blanket_replacement_claim_allowed"],
                                                        "semantics.blanket_replacement_claim_allowed"),
        alpha_bare_source=sm["alpha_bare_source"], kappa_source=sm["kappa_source"])
    is_vehicle = True if "is_vehicle" not in d else _require_bool(d["is_vehicle"], "is_vehicle")
    return PhysicalScenario(
        name=d["name"], geometry=geom, source=src, semantics=sem,
        spl_dB=tuple(float(x) for x in d["spl_dB"]), oaspl_dB=float(d["oaspl_dB"]),
        alpha_bare_nominal=float(d["alpha_bare_nominal"]), alpha_bare_range=tuple(float(x) for x in d["alpha_bare_range"]),
        kappa_nominal=float(d["kappa_nominal"]), kappa_eval_range=tuple(float(x) for x in d["kappa_eval_range"]),
        pressure_ref_uPa=float(d.get("pressure_ref_uPa", 20.0)), Q_fixed=float(d.get("Q_fixed", 15.0)),
        band_hz=tuple(float(x) for x in d.get("band_hz", DESIGN_BAND_HZ)), is_vehicle=is_vehicle)


def load_scenario(path: str | Path) -> PhysicalScenario:
    """Load and VALIDATE a PhysicalScenario from its authoritative YAML (the single source of truth)."""
    import yaml
    with open(path) as f:
        d = yaml.safe_load(f)
    if "semantics" not in d:
        raise ScenarioValidationError("scenario YAML must contain an enforceable 'semantics:' block")
    return scenario_from_dict(d)


def falcon9_2025_blanketed() -> PhysicalScenario:
    """Convenience loader for the Falcon 9 scenario from its authoritative YAML (no duplicated values)."""
    from .. import paths
    return load_scenario(paths.CONFIGS / "falcon9_2025_blanketed.yaml")


if __name__ == "__main__":
    from .. import cli
    if cli.show_help("fem", "scenarios"):
        raise SystemExit(0)
    import sys
    from .. import paths
    cfg = sys.argv[1] if len(sys.argv) > 1 else str(paths.CONFIGS / "falcon9_2025_blanketed.yaml")
    s = load_scenario(cfg)
    print(f"scenario: {s.name}  (loaded from {cfg})")
    print(f"  A_wall (derived) = {s.A_wall:.4f} m^2 ; volume = {s.volume:.4f} m^3")
    print(f"  in-band OASPL = {s.inband_oaspl:.2f} dB (listed total {s.oaspl_dB:.1f} dB) -> "
          f"{'energy-consistent' if s.inband_oaspl <= s.oaspl_dB else 'INCONSISTENT'}")
    print(f"  A_other(nominal kappa {s.kappa_nominal}) = {s.A_other_nominal:.4f} sabins")
    ach = assert_fem_consistency(s, 165.878, s.volume, s.kappa_nominal * 165.878, expected_kappa=s.kappa_nominal)
    print(f"  UNIT self-test (formula check, NOT a real mesh integration) PASS: "
          f"rel_area_err={ach['rel_area_err']:.2e} abs_kappa_err={ach['abs_kappa_err']:.2e}")
