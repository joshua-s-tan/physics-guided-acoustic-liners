# -*- coding: utf-8 -*-
"""Evaluation request construction with a HARD physical/synthetic separation (remediation §4.1, §3 rule 3;
stages 3-4).

The historical `evaluate.make_requests` drew a RANDOM `A_wall = uniform(*A_WALL_RANGE)` for EVERY request,
including named physical vehicles (`ref:falcon9`), so a real vehicle was evaluated at a synthetic wall area
instead of its geometry-derived one (165.878 m^2 for Falcon). This module makes that impossible:

  * `PhysicalScenarioRequest` -- built ONLY from a COMPLETE, RE-VALIDATED `PhysicalScenario`. It stores the
    scenario as IMMUTABLE canonical JSON + its SHA-256 (never a caller-owned mutable dict) and reconstructs
    the scenario on every access, RECOMPUTING all derived quantities (A_wall/volume/A_other/in-band OASPL)
    from geometry/spectrum -- so a minimal or tampered record (e.g. an injected `A_wall=83.78`) cannot pass.
    (Review round 4.1 finding 5 + round 4.1-c: every downstream physical artifact carries the full validated
    scenario or its hash, incl. claim semantics, for `provenance.release_verifier`.)
  * `SyntheticBenchmarkRequest` (in `scenarios.py`) -- randomized `A_wall`/`kappa`/`target`; ML benchmark ONLY.

`RequestSet` is FROZEN with tuple collections and revalidates on export, so neither list can be contaminated
after construction. `assert_finalist_uses_scenario_area` is the finalist-selection gate; the migrated
evaluation/finalist writer must CALL it (scheduled for §14 step 8) -- until then step 4 stays partial.
"""
from __future__ import annotations
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ..fem import scenarios as S
import numpy as np

DEFAULT_AREA_TOL = S.DEFAULT_FEM_TOL          # reuse the 0.5% consistency tolerance for the finalist gate


class RequestSeparationError(RuntimeError):
    """A request crossed the physical/synthetic boundary (e.g. a vehicle request whose record is incomplete
    or does not reconstruct to a valid scenario, or a synthetic request labelled as a vehicle)."""


class FinalistConsistencyError(RuntimeError):
    """A named-vehicle finalist was evaluated at a wall area that does not match the scenario's derived
    A_wall -- the §4.1 mismatch the release exists to prevent. RuntimeError (not assert) so it survives -O."""


def _canonical_json(record: dict) -> str:
    """Deterministic, NaN-free canonical JSON for hashing/storage. `allow_nan=False` rejects Inf/NaN."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False, default=float)


def canonical_scenario_sha256(scenario: S.PhysicalScenario) -> str:
    """Stable SHA-256 of the scenario's serialized record. The SAME function is used when building a physical
    request and when `provenance.release_verifier` re-checks it, so a drifted scenario is caught."""
    return hashlib.sha256(_canonical_json(scenario.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PhysicalScenarioRequest:
    """An evaluation request for a NAMED physical vehicle. The scenario is held as IMMUTABLE canonical JSON;
    all physical quantities are DERIVED by reconstructing+revalidating that scenario, never stored/trusted.

    Construct via `physical_request_from_scenario`. Direct construction still works but is held to the same
    bar: the record must reconstruct to a fully valid `PhysicalScenario` and its hash must match."""
    request_id: str
    scenario_name: str
    achievable_max_dB: float
    dOASPL_target_dB: float
    scenario_json: str
    scenario_sha256: str
    group: str = "reference"
    is_vehicle: bool = True

    def __post_init__(self):
        if self.is_vehicle is not True:
            raise RequestSeparationError("PhysicalScenarioRequest.is_vehicle must be True")
        if not str(self.request_id).strip():
            raise S.ScenarioValidationError("request_id must be non-empty")
        if not isinstance(self.scenario_json, str) or not self.scenario_json.strip():
            raise RequestSeparationError("scenario_json must be a non-empty canonical JSON string")
        # the SUPPLIED hash must match the SUPPLIED json (reject a tampered/mismatched input hash)
        if hashlib.sha256(self.scenario_json.encode("utf-8")).hexdigest() != self.scenario_sha256:
            raise RequestSeparationError("scenario_sha256 does not match scenario_json")
        # RECONSTRUCT + FULLY VALIDATE the scenario from primitives; recomputes A_wall/volume/A_other/OASPL.
        try:
            rec = json.loads(self.scenario_json)
        except json.JSONDecodeError as e:
            raise RequestSeparationError(f"scenario_json is not valid JSON: {e}")
        scen = S.scenario_from_dict(rec)          # raises ScenarioValidationError on any incomplete/bad field
        if not scen.is_vehicle:
            raise RequestSeparationError("a PhysicalScenarioRequest scenario must have is_vehicle=True")
        if self.scenario_name != scen.name:
            raise RequestSeparationError(
                f"scenario_name {self.scenario_name!r} != reconstructed scenario name {scen.name!r}")
        # NORMALIZE storage to the authoritative canonical record (round 4.1-d): any ignored/injected derived
        # value in the supplied json (e.g. A_wall=83.78) is replaced by the recomputed authoritative record,
        # so the exported scenario_record() and hash can NEVER disagree with the A_wall property. The stored
        # hash becomes the authoritative scenario hash (== canonical_scenario_sha256(scen)).
        normalized_json = _canonical_json(scen.to_dict())
        object.__setattr__(self, "scenario_json", normalized_json)
        object.__setattr__(self, "scenario_sha256",
                           hashlib.sha256(normalized_json.encode("utf-8")).hexdigest())
        if not (math.isfinite(self.achievable_max_dB) and self.achievable_max_dB >= 0):
            raise S.ScenarioValidationError("achievable_max_dB must be finite and >= 0")
        if not (math.isfinite(self.dOASPL_target_dB) and self.dOASPL_target_dB > 0):
            raise S.ScenarioValidationError("dOASPL_target_dB must be finite and > 0")

    # --- derived, recomputed from the immutable scenario on demand (never stored mutable state) ---
    @property
    def scenario(self) -> S.PhysicalScenario:
        return S.scenario_from_dict(json.loads(self.scenario_json))

    @property
    def A_wall(self) -> float:
        return self.scenario.A_wall

    @property
    def kappa(self) -> float:
        return self.scenario.kappa_nominal

    @property
    def spl_dB(self) -> tuple:
        return self.scenario.spl_dB

    def scenario_record(self) -> dict:
        """A FRESH decoded copy of the scenario record (callers cannot mutate our stored state)."""
        return json.loads(self.scenario_json)

    def to_dict(self) -> dict:
        return dict(request_id=self.request_id, scenario_name=self.scenario_name, group=self.group,
                    is_vehicle=self.is_vehicle, A_wall=self.A_wall, kappa=self.kappa,
                    spl_dB=list(self.spl_dB), achievable_max_dB=self.achievable_max_dB,
                    dOASPL_target_dB=self.dOASPL_target_dB, scenario_sha256=self.scenario_sha256,
                    scenario_record=self.scenario_record())


def _validate_factory_args(target_frac: float, min_target_dB: float, amax: float) -> None:
    if not (0.0 < float(target_frac) <= 1.0):
        raise S.ScenarioValidationError(f"target_frac must be in (0,1]; got {target_frac}")
    if not (math.isfinite(min_target_dB) and float(min_target_dB) >= 0):
        raise S.ScenarioValidationError(f"min_target_dB must be finite and >= 0; got {min_target_dB}")
    if not (math.isfinite(amax) and amax >= 0):
        raise S.ScenarioValidationError(f"achievable_max must be finite and >= 0; got {amax}")


def physical_request_from_scenario(scenario: S.PhysicalScenario, achievable_max_fn: Callable[..., float],
                                   target_frac: float = 0.6, min_target_dB: float = 0.5,
                                   request_id: Optional[str] = None, group: str = "reference") -> PhysicalScenarioRequest:
    """Build a `PhysicalScenarioRequest` from an authoritative scenario. `A_wall`/`kappa`/`spl` come from the
    SCENARIO (derived); `achievable_max_fn(spl_dB, kappa, A_wall)` supplies the physics-ceiling (injected so
    this module needs no ML/oracle imports). `dOASPL_target = max(min_target_dB, target_frac * achievable)`."""
    if not isinstance(scenario, S.PhysicalScenario):
        raise RequestSeparationError("physical_request_from_scenario requires a PhysicalScenario")
    if not scenario.is_vehicle:
        raise RequestSeparationError("physical_request_from_scenario requires a vehicle scenario")
    A_wall = float(scenario.A_wall)
    kappa = float(scenario.kappa_nominal)
    spl = tuple(float(x) for x in scenario.spl_dB)
    amax = float(achievable_max_fn(spl_dB=spl, kappa=kappa, A_wall=A_wall))
    _validate_factory_args(target_frac, min_target_dB, amax)
    target = max(float(min_target_dB), float(target_frac) * amax)
    blob = _canonical_json(scenario.to_dict())
    return PhysicalScenarioRequest(
        request_id=request_id or f"ref:{scenario.name}", scenario_name=scenario.name,
        achievable_max_dB=amax, dOASPL_target_dB=target, scenario_json=blob,
        scenario_sha256=hashlib.sha256(blob.encode("utf-8")).hexdigest(), group=group)


def synthetic_request(rng, spl_dB: Sequence[float], A_wall_range: tuple, kappa_range: tuple,
                      achievable_max_fn: Callable[..., float], request_id: str, target_frac: float = 0.6,
                      min_target_dB: float = 0.5, spectrum_shape_ref: str = "",
                      mass_target_kg: float = 100.0) -> S.SyntheticBenchmarkRequest:
    """Build a randomized ML-benchmark request. `A_wall`/`kappa` ARE drawn from the given ranges -- the ONLY
    legitimate place to randomize wall area -- and it produces a SyntheticBenchmarkRequest (is_vehicle=False),
    never a vehicle."""
    if not str(request_id).strip():
        raise S.ScenarioValidationError("request_id must be non-empty")
    S._validate_range_contains(A_wall_range, A_wall_range[0], "A_wall", lo_min=0.0, hi_max=math.inf)
    S._validate_range_contains(kappa_range, kappa_range[0], "kappa", lo_min=0.0, hi_max=1.0)
    A_wall = float(rng.uniform(*A_wall_range))
    kappa = float(rng.uniform(*kappa_range))
    amax = float(achievable_max_fn(spl_dB=tuple(float(x) for x in spl_dB), kappa=kappa, A_wall=A_wall))
    _validate_factory_args(target_frac, min_target_dB, amax)
    target = max(float(min_target_dB), float(target_frac) * amax)
    return S.SyntheticBenchmarkRequest(
        request_id=request_id, spl_dB=tuple(float(x) for x in spl_dB), A_wall=A_wall, kappa=kappa,
        target_dOASPL_dB=target, mass_target_kg=mass_target_kg, spectrum_shape_ref=spectrum_shape_ref)


@dataclass(frozen=True)
class RequestSet:
    """A frozen evaluation request set: named physical vehicles + randomized synthetic benchmarks, kept
    strictly separate. Collections are normalized to tuples and revalidated on export, so neither list can be
    contaminated after construction."""
    physical: tuple = ()
    synthetic: tuple = ()

    def __post_init__(self):
        # coerce to immutable tuples (a caller-passed list can no longer be mutated through us)
        object.__setattr__(self, "physical", tuple(self.physical))
        object.__setattr__(self, "synthetic", tuple(self.synthetic))
        self._validate()

    def _validate(self) -> None:
        for r in self.physical:
            if not isinstance(r, PhysicalScenarioRequest):
                raise RequestSeparationError("physical[] must contain only PhysicalScenarioRequest")
        for r in self.synthetic:
            if not isinstance(r, S.SyntheticBenchmarkRequest):
                raise RequestSeparationError("synthetic[] must contain only SyntheticBenchmarkRequest")
            if r.is_vehicle:
                raise RequestSeparationError("a synthetic request must not be flagged is_vehicle")

    def to_manifest(self, tag: str = "final") -> dict:
        from dataclasses import asdict
        self._validate()                                    # revalidate on every export
        phys = [r.to_dict() for r in self.physical]
        synth = [asdict(r) for r in self.synthetic]
        body = dict(tag=tag, physical=phys, synthetic=synth, n_physical=len(phys), n_synthetic=len(synth))
        body["request_set_sha256"] = hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()
        return body


def assert_finalist_uses_scenario_area(finalist_A_wall: float, scenario: S.PhysicalScenario,
                                       tol: float = DEFAULT_AREA_TOL) -> float:
    """Finalist-selection gate (§4.1): a named-vehicle finalist must have been evaluated at the scenario's
    DERIVED A_wall (within `tol`). Raises FinalistConsistencyError otherwise. Returns the relative error.
    This is the concrete guard that the historical randomized-A_wall finalist can never be shipped again.
    NOTE: not yet invoked by a production writer (the evaluation/finalist migration is §14 step 8); step 4
    stays PARTIAL until it is."""
    if not (math.isfinite(finalist_A_wall) and finalist_A_wall > 0):
        raise FinalistConsistencyError(f"finalist A_wall must be finite and > 0; got {finalist_A_wall}")
    if not (0.0 < tol < 1.0):
        raise FinalistConsistencyError(f"tol must be in (0,1); got {tol}")
    derived = scenario.A_wall
    rel = abs(finalist_A_wall - derived) / derived
    if rel > tol:
        raise FinalistConsistencyError(
            f"finalist evaluated at A_wall {finalist_A_wall:.4f} m^2 but scenario {scenario.name!r} derives "
            f"{derived:.4f} m^2 (rel {rel:.4f} > {tol}); a named vehicle must use its geometry-derived area")
    return rel


def oracle_achievable_max_fn(oracle):
    """An `achievable_max_fn(spl_dB, kappa, A_wall)` for `requests.physical_request_from_scenario`: the max robust
    worst-case dOASPL a representative tuned liner reaches (a physics-ceiling ESTIMATE, oracle-injected)."""
    def fn(*, spl_dB, kappa, A_wall):
        design = dict(D=0.05, face_t=0.0008, wall_t=0.0006,
                      groups=[dict(f0=f0, r_neck=r, L_neck=0.03, area_frac=0.25, Q=15.0)
                              for f0, r in ((126.0, 0.012), (160.0, 0.010), (196.0, 0.008), (251.0, 0.006))])
        base = dict(design, A_wall=float(A_wall), kappa_nominal=float(kappa),
                    spectrum=np.asarray(spl_dB, dtype=float))
        return float(oracle.min_coverage_for_target(base, dOASPL_target=1e9).get("max_dOASPL_worst_dB", 0.0))
    return fn
