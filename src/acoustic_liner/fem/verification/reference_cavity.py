r"""Generic reference-cavity benchmark for the boundary-vs-volume model-form study (remediation §4.2; owner
decision 2026-09-26; round 4.1-h disjoint-boundary + enforcement corrections).

METHODOLOGICAL boundary-model benchmark, NOT a vehicle (`scenario_kind="generic_boundary_benchmark"`,
`is_vehicle=False`). It compares an EXPLICIT non-wall payload boundary against Falcon's equivalent distributed
(volumetric) damping, contributing a MODEL-FORM SENSITIVITY (interpretation frozen in REMEDIATION-PLAN: this
is NOT part of the formal Falcon robustness envelope).

Geometry is uniquely defined from dimensions [m]: an axisymmetric OUTER cylinder `(R_outer, H)` with a coaxial
INTERNAL cylindrical payload `(R_payload, H_payload)` at axial position `z_payload` (bottom), defaulting to
centered. Positive top/bottom clearance is required so the payload touches neither the base nor the apex. The
boundary is partitioned into DISJOINT tagged areas -- a facet is never both source and lined wall:

  * source/radiation boundary : base end cap at z=0            (area = pi R_outer^2)
  * lineable wall             : outer lateral + top end cap    (2 pi R_outer H + pi R_outer^2)
  * payload S_other           : payload lateral + 2 end caps   (2 pi R_payload H_payload + 2 pi R_payload^2)
  * remaining rigid boundary  : none in this benchmark         (0)

`assert_matches_mesh` validates EACH tagged area against the mesh and proves the tag sets are disjoint (they
sum to the total boundary area).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Optional
import os
import tempfile
from ..errors import FEMSolverError
from ..mesh import _facet_axisym_area
from ..solver import _require_passive_beta
from ..mesh import assert_boundary_partition
from ..mesh import mesh_physical_volume
from ..solver import solve_interior


MAX_ALPHA_OTHER = 0.95
SCENARIO_KIND = "generic_boundary_benchmark"
_MIN_CLEARANCE = 1e-3          # m, minimum axial gap between payload ends and base/apex


class ReferenceCavityError(ValueError):
    """The generic reference cavity is under-specified or unrealizable (payload doesn't fit / no clearance,
    S_other too small for a kappa corner, bad dimensions, or a forbidden/vehicle label)."""


def required_S_other(max_A_other: float, max_alpha: float = MAX_ALPHA_OTHER) -> float:
    """Minimum explicit-boundary area to carry `max_A_other` sabins at `alpha_other <= max_alpha`."""
    A = float(max_A_other)
    if not (math.isfinite(A) and A > 0):
        raise ReferenceCavityError(f"max_A_other must be finite and > 0; got {max_A_other}")
    if not (0.0 < max_alpha <= 1.0):
        raise ReferenceCavityError(f"max_alpha must be in (0,1]; got {max_alpha}")
    return A / max_alpha
_FORBIDDEN_NAME_TOKENS = ("falcon", "nasa", "electron", "ariane", "vulcan", "atlas", "delta", "starship")


@dataclass(frozen=True)
class ReferenceCavity:
    """A fully-specified generic cavity DERIVED from dimensions. `scenario_kind` is fixed and `is_vehicle`
    is always False. `z_payload` is the payload bottom; None -> centered."""
    name: str
    R_outer: float
    H: float
    R_payload: float
    H_payload: float
    max_A_other: float
    z_payload: Optional[float] = None
    note: str = "methodological boundary-model benchmark; NOT Falcon/NASA/Electron"
    scenario_kind: str = SCENARIO_KIND
    is_vehicle: bool = False

    def __post_init__(self):
        if self.is_vehicle:
            raise ReferenceCavityError("reference cavity must have is_vehicle=False (it is not a vehicle)")
        if self.scenario_kind != SCENARIO_KIND:
            raise ReferenceCavityError(f"scenario_kind must be {SCENARIO_KIND!r}; got {self.scenario_kind!r}")
        low = str(self.name).lower()
        if not low.strip():
            raise ReferenceCavityError("name must be non-empty")
        if any(tok in low for tok in _FORBIDDEN_NAME_TOKENS):
            raise ReferenceCavityError(f"reference cavity name {self.name!r} must not name a real vehicle")
        for nm in ("R_outer", "H", "R_payload", "H_payload", "max_A_other"):
            v = getattr(self, nm)
            if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) and float(v) > 0):
                raise ReferenceCavityError(f"{nm} must be a finite number > 0; got {v!r}")
        if self.R_payload >= self.R_outer:
            raise ReferenceCavityError(f"R_payload {self.R_payload} must be < R_outer {self.R_outer}")
        # axial placement: default centered; require positive clearance to base AND apex (unique, disjoint)
        z0 = (self.H - self.H_payload) / 2.0 if self.z_payload is None else float(self.z_payload)
        if not math.isfinite(z0):
            raise ReferenceCavityError("z_payload must be finite")
        object.__setattr__(self, "z_payload", z0)
        if z0 < _MIN_CLEARANCE:
            raise ReferenceCavityError(f"payload bottom z={z0} needs >= {_MIN_CLEARANCE} m clearance above base")
        if z0 + self.H_payload > self.H - _MIN_CLEARANCE:
            raise ReferenceCavityError(
                f"payload top z={z0 + self.H_payload} needs >= {_MIN_CLEARANCE} m clearance below apex (H={self.H})")
        need = required_S_other(self.max_A_other)
        if self.S_other < need - 1e-9:
            raise ReferenceCavityError(
                f"S_other {self.S_other:.4f} m^2 (from geometry) too small: need >= {need:.4f} m^2 to carry "
                f"max_A_other {self.max_A_other:.4f} at alpha_other <= {MAX_ALPHA_OTHER}; enlarge the payload")

    # --- DERIVED, DISJOINT tagged boundary areas (a facet is never in two sets) ---
    @property
    def source_area(self) -> float:
        """Radiation/source boundary: the base end cap at z=0 (payload has clearance, so the base is fully source)."""
        return math.pi * self.R_outer ** 2

    @property
    def lined_wall_area(self) -> float:
        """Lineable wall: outer cylinder lateral + TOP end cap (the base cap is the source, NOT lined wall)."""
        return 2.0 * math.pi * self.R_outer * self.H + math.pi * self.R_outer ** 2

    @property
    def S_other(self) -> float:
        """Explicit internal payload boundary: payload cylinder lateral + two end caps."""
        return 2.0 * math.pi * self.R_payload * self.H_payload + 2.0 * math.pi * self.R_payload ** 2

    @property
    def rigid_area(self) -> float:
        """Remaining rigid boundary (none in this benchmark)."""
        return 0.0

    @property
    def fluid_volume(self) -> float:
        return math.pi * self.R_outer ** 2 * self.H - math.pi * self.R_payload ** 2 * self.H_payload

    @property
    def total_boundary_area(self) -> float:
        return self.source_area + self.lined_wall_area + self.S_other + self.rigid_area

    def alpha_other(self, A_other: float) -> float:
        A = float(A_other)
        if not (math.isfinite(A) and A >= 0):
            raise ReferenceCavityError(f"A_other must be finite and >= 0; got {A_other}")
        a = A / self.S_other
        if a > MAX_ALPHA_OTHER + 1e-12:
            raise ReferenceCavityError(f"alpha_other {a:.3f} > {MAX_ALPHA_OTHER} for A_other={A_other}")
        return a

    def geometry_hash(self) -> str:
        """Canonical SHA-256 of the defining dimensions (for provenance in explicit-boundary artifacts)."""
        rec = dict(name=self.name, R_outer=self.R_outer, H=self.H, R_payload=self.R_payload,
                   H_payload=self.H_payload, z_payload=self.z_payload, max_A_other=self.max_A_other,
                   scenario_kind=self.scenario_kind)
        blob = json.dumps(rec, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def assert_area_closure(self, mesh_source_area: float, mesh_lined_wall_area: float, mesh_S_other: float,
                            mesh_total_boundary_area: float, tol: float = 0.02) -> dict:
        """AREA-CLOSURE check only: each analytic tagged area matches the mesh's integrated area for that tag,
        AND the tagged areas sum to the total boundary area. NOTE: area closure is NECESSARY but NOT SUFFICIENT
        for facet DISJOINTNESS -- overlapping tags plus an equal untagged gap can also close. True disjointness
        (pairwise-empty facet-ID intersections, union == all boundary facets) requires the actual mesh facet
        sets and is a PRODUCTION-MIGRATION gate (`assert_facet_partition`), NOT proven here. Raises on mismatch."""
        if not (math.isfinite(tol) and 0.0 < tol < 1.0):
            raise ReferenceCavityError(f"tol must be finite in (0,1); got {tol}")
        pairs = {"source": (mesh_source_area, self.source_area),
                 "lined_wall": (mesh_lined_wall_area, self.lined_wall_area),
                 "S_other": (mesh_S_other, self.S_other)}
        checks = {}
        for k, (m, a) in pairs.items():
            if not (math.isfinite(float(m)) and a > 0):
                raise ReferenceCavityError(f"{k} area invalid (mesh={m}, analytic={a})")
            checks[k] = abs(float(m) - a) / a
        # AREA CLOSURE (necessary, NOT sufficient for disjointness): the tagged areas must sum to the total
        # boundary area. True facet disjointness is proven only by `assert_facet_partition` on real facet sets.
        tag_sum = float(mesh_source_area) + float(mesh_lined_wall_area) + float(mesh_S_other)
        checks["area_closure_sum"] = abs(tag_sum - float(mesh_total_boundary_area)) / float(mesh_total_boundary_area)
        bad = {k: v for k, v in checks.items() if not (math.isfinite(v) and v <= tol)}
        if bad:
            raise ReferenceCavityError(f"analytic-vs-mesh area-closure mismatch beyond tol={tol}: {bad}")
        return checks

    def assert_facet_partition(self, facet_id_sets: dict, all_boundary_facets, facet_area_by_id: dict,
                               tol: float = 0.02) -> dict:
        """TRUE facet-set disjointness gate for production migration. `facet_id_sets` maps each tag
        ('source','lined_wall','S_other'[,'rigid']) to the set of mesh facet IDs carrying it;
        `facet_area_by_id` maps EVERY boundary facet ID to its integrated area. Each tag's area is COMPUTED
        HERE from its facet set + that per-facet map -- never trusted from a separate aggregate (which let a
        moved facet keep its old area). Proves: (a) exactly the declared tags are present (rigid empty/absent
        when analytic rigid area is 0); (b) tags PAIRWISE DISJOINT; (c) their UNION equals every boundary
        facet; (d) every boundary facet has one finite positive area; (e) each tag's COMPUTED area matches the
        analytic area within `tol`."""
        if not (isinstance(tol, (int, float)) and math.isfinite(tol) and 0.0 < tol < 0.5):
            raise ReferenceCavityError(f"tol must be finite in (0, 0.5); got {tol}")
        analytic = {"source": self.source_area, "lined_wall": self.lined_wall_area, "S_other": self.S_other}
        # rigid: allowed only when this benchmark actually has rigid area (> 0). It is 0 here, so a nonempty
        # or present-but-declared 'rigid' tag is rejected.
        sets = {k: set(v) for k, v in facet_id_sets.items()}
        allowed_tags = set(analytic) | ({"rigid"} if self.rigid_area > 0 else set())
        unknown = set(sets) - allowed_tags
        if unknown:
            raise ReferenceCavityError(f"unknown facet tag(s) {sorted(unknown)}; allowed {sorted(allowed_tags)}")
        if self.rigid_area == 0 and sets.get("rigid"):
            raise ReferenceCavityError("rigid tag must be empty/absent when analytic rigid_area == 0")
        missing = set(analytic) - set(sets)
        if missing:
            raise ReferenceCavityError(f"facet_id_sets missing required tags: {sorted(missing)}")
        # per-facet area map: every boundary facet present with a finite positive area
        allb = set(all_boundary_facets)
        area_map = {int(k): float(v) for k, v in facet_area_by_id.items()}
        for fid in allb:
            a = area_map.get(fid)
            if a is None or not (math.isfinite(a) and a > 0):
                raise ReferenceCavityError(f"boundary facet {fid} lacks a finite positive area (got {a})")
        # pairwise disjoint
        keys = list(sets)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                inter = sets[keys[i]] & sets[keys[j]]
                if inter:
                    raise ReferenceCavityError(f"tags {keys[i]!r} and {keys[j]!r} share facets {sorted(inter)[:5]}")
        # union == all boundary facets
        union = set().union(*sets.values()) if sets else set()
        if union != allb:
            raise ReferenceCavityError(
                f"tag union != all boundary facets: untagged={sorted(allb - union)[:5]}, "
                f"foreign={sorted(union - allb)[:5]}")
        # COMPUTE each tag area from its facets (not trusted) and compare to analytic
        errs = {}
        for k, a in analytic.items():
            computed = sum(area_map[fid] for fid in sets[k])
            e = abs(computed - a) / a if a > 0 else float("inf")
            if not (math.isfinite(e) and e <= tol):
                raise ReferenceCavityError(f"tag {k!r} computed area {computed} vs analytic {a} (rel {e} > {tol})")
            errs[k] = e
        return dict(area_errors=errs, n_facets=len(allb), tags={k: len(v) for k, v in sets.items()})


BENCHMARK_ROUNDTRIP_TOL = 1e-6


def build_reference_cavity_mesh(cavity, mesh_size: float = 0.25):
    """Axisymmetric cross-section of a generic reference cavity: outer rectangle (R_outer x H) with the
    coaxial payload rectangle (R_payload x H_payload at z_payload) CUT OUT. The payload surface becomes an
    internal boundary (the explicit non-wall S_other)."""
    import gmsh
    import meshio
    import skfem
    Ro, H, Rp, Hp, zp = cavity.R_outer, cavity.H, cavity.R_payload, cavity.H_payload, cavity.z_payload
    fd, fn = tempfile.mkstemp(prefix="acoustic_liner_refcav_", suffix=".msh")
    os.close(fd)
    try:
        gmsh.initialize()
        try:
            gmsh.model.add("reference_cavity")
            occ = gmsh.model.occ
            outer = occ.addRectangle(0.0, 0.0, 0.0, Ro, H)
            payload = occ.addRectangle(0.0, zp, 0.0, Rp, Hp)
            occ.cut([(2, outer)], [(2, payload)])
            occ.synchronize()
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
            gmsh.model.mesh.generate(2)
            gmsh.write(fn)
        finally:
            gmsh.finalize()
        md = meshio.read(fn)
        return skfem.MeshTri(md.points[:, :2].T, md.cells_dict["triangle"].T)
    finally:
        try:
            os.remove(fn)
        except OSError:
            pass


def tag_cavity_facets(mesh, cavity, tol: float = 1e-6) -> dict:
    """Partition reference-cavity boundary facets into DISJOINT tags: 'axis' (r=0), 'source' (base z=0),
    'wall' (outer lateral r=R_outer OR top cap z=H), 'payload' (everything else -> the cut payload surface)."""
    bnd = list(mesh.boundary_facets())
    pts = mesh.p
    Ro, H = cavity.R_outer, cavity.H
    sets = {"source": set(), "axis": set(), "wall": set(), "payload": set()}
    area_by_id = {}
    for fid in bnd:
        a, b = pts[:, mesh.facets[0, fid]], pts[:, mesh.facets[1, fid]]
        mid_r = 0.5 * (a[0] + b[0]); mid_z = 0.5 * (a[1] + b[1])
        area_by_id[fid] = _facet_axisym_area(mesh, fid)
        if mid_r < tol:
            sets["axis"].add(fid)
        elif mid_z < tol:
            sets["source"].add(fid)
        elif mid_r > Ro - tol or mid_z > H - tol:
            sets["wall"].add(fid)
        else:
            sets["payload"].add(fid)
    return dict(sets=sets, area_by_id=area_by_id, all=set(bnd))


def run_reference_cavity_benchmark(cavity, freqs, wall_beta, A_other, *, p0: float = 1.0,
                                   mesh_size: float = 0.25, area_tol: float = 0.03) -> dict:
    """GENERIC-BENCHMARK boundary-vs-volume model-form study (#3). On ONE identical mesh/source/wall config
    it runs TWO solves for the same non-wall absorption A_other:
      * EXPLICIT: A_other as a Robin boundary on the payload surface (admittance from the PRODUCTION diffuse
        map, MANDATORY round-trip; beta = Z_air*Y);
      * VOLUMETRIC: A_other as equivalent distributed damping k^2->k^2(1-i*eta) (Falcon's model).
    Reports the per-frequency dB difference as a GENERIC MODEL-FORM SENSITIVITY -- explicitly NOT part of the
    Falcon robustness envelope. Machine-enforced generic-only (rejects vehicles)."""
    from .. import boundary as fem_boundary
    from ...physics import absorption as acoustics
    from .. import damping
    from ...physics import constants
    fem_boundary._require_generic_benchmark(cavity)      # ReferenceCavity, generic, is_vehicle False
    wall_beta = _require_passive_beta("wall_beta", wall_beta)
    fs = [float(f) for f in freqs]
    if not fs:
        raise FEMSolverError("freqs must be non-empty")
    for f in fs:
        if not (math.isfinite(f) and f > 0):
            raise FEMSolverError(f"every frequency must be finite and > 0; got {f}")

    mesh = build_reference_cavity_mesh(cavity, mesh_size=mesh_size)
    tagged = tag_cavity_facets(mesh, cavity)
    # geometry-derived, disjoint source/wall/payload/axis area checks against the analytic cavity
    partition = assert_boundary_partition(
        tagged, {"source": cavity.source_area, "wall": cavity.lined_wall_area,
                 "payload": cavity.S_other, "axis": 0.0}, tol=area_tol)
    mesh_volume = mesh_physical_volume(mesh)
    vol_err = abs(mesh_volume - cavity.fluid_volume) / cavity.fluid_volume
    if vol_err > area_tol:
        raise FEMSolverError(f"mesh volume {mesh_volume:.4f} vs analytic {cavity.fluid_volume:.4f} > {area_tol}")

    # align the achievable-absorption range with the selected real rigid-branch diffuse inversion
    alpha_o = cavity.alpha_other(A_other)
    a_max = acoustics.diffuse_alpha_max()
    if alpha_o > a_max + 1e-9:
        raise FEMSolverError(
            f"alpha_other {alpha_o:.4f} exceeds the rigid-branch diffuse max {a_max:.4f}; enlarge S_other "
            f"(payload) or adopt another validated impedance branch")

    # EXPLICIT payload admittance via the AUTHORITATIVE policy path (generic-only guard + PRODUCTION diffuse
    # map + MANDATORY round-trip), so production cannot drift from the validated boundary policy.
    roundtrip_tol = BENCHMARK_ROUNDTRIP_TOL
    Ys = fem_boundary.other_boundary_admittance(
        fs, A_other, cavity,
        admittance_from_alpha=lambda a: acoustics.admittance_from_diffuse_alpha(a, constants.Z_AIR),
        Z_air=constants.Z_AIR,
        alpha_from_admittance=lambda Y: acoustics.diffuse_alpha_from_admittance(Y, constants.Z_AIR),
        roundtrip_tol=roundtrip_tol, require_roundtrip=True)
    Y = complex(Ys[0])                                   # alpha_other is frequency-independent here
    a_rt = acoustics.diffuse_alpha_from_admittance(Y, constants.Z_AIR)
    beta_payload = fem_boundary.to_normalized_beta(Y, constants.Z_AIR)
    payload_facets = tagged["sets"]["payload"]

    explicit, volumetric, delta_dB = {}, {}, {}
    for f in fs:
        ex = solve_interior(mesh, f, tagged, wall_beta=wall_beta, A_other=0.0, p0=p0,
                            extra_robin=[(payload_facets, beta_payload)])
        vo = solve_interior(mesh, f, tagged, wall_beta=wall_beta, A_other=A_other, volume=mesh_volume, p0=p0)
        explicit[f] = ex; volumetric[f] = vo
        delta_dB[f] = 10.0 * math.log10(ex / vo)
    kappa_bench = A_other / cavity.lined_wall_area       # benchmark-analogue kappa for the loss-factor record
    return dict(
        interpretation="generic_benchmark_model_form_sensitivity",
        note="NOT part of the Falcon robustness envelope; not vehicle validation",
        cavity_name=cavity.name, geometry_hash=cavity.geometry_hash(),
        cavity_dims=dict(R_outer=cavity.R_outer, H=cavity.H, R_payload=cavity.R_payload,
                         H_payload=cavity.H_payload, z_payload=cavity.z_payload, max_A_other=cavity.max_A_other),
        freqs=fs, mesh_size=mesh_size, area_tol=area_tol, volume_rel_error=vol_err,
        air_properties=constants.AIR_PROPERTIES,
        measured_source_area=sum(tagged["area_by_id"][f] for f in tagged["sets"]["source"]),
        measured_wall_area=sum(tagged["area_by_id"][f] for f in tagged["sets"]["wall"]),
        measured_S_other=sum(tagged["area_by_id"][f] for f in tagged["sets"]["payload"]),
        measured_volume=mesh_volume, boundary_partition=partition,
        A_other=A_other, alpha_other=alpha_o, alpha_other_max=a_max,
        diffuse_mapping_id=acoustics.DIFFUSE_MAPPING_ID, diffuse_quadrature_n=acoustics.DIFFUSE_QUADRATURE_N,
        roundtrip_tol=roundtrip_tol, roundtrip_alpha=a_rt, roundtrip_error=abs(a_rt - alpha_o),
        source_model=fem_boundary.source_model_record(p0=p0),
        explicit_a_other_model=fem_boundary.explicit_a_other_record(A_other, cavity),
        volumetric_loss_model=damping.loss_factor_record(A_other, mesh_volume, kappa_bench, fs),
        payload_admittance=dict(real=Y.real, imag=Y.imag),
        payload_beta=dict(real=beta_payload.real, imag=beta_payload.imag),
        wall_beta=dict(real=wall_beta.real, imag=wall_beta.imag),
        delta_convention="10*log10(explicit_ms/volumetric_ms)",
        explicit_ms=explicit, volumetric_ms=volumetric, delta_dB=delta_dB,
        max_abs_delta_dB=max(abs(v) for v in delta_dB.values()))
