# -*- coding: utf-8 -*-
"""Payload-resolving axisymmetric FEM for the vehicle scenarios (`vehicles.scenarios.VehicleScenario` only).

The acoustic domain is the representative fairing interior (cylinder + cone) with the scenario's payload case CUT OUT
(a solid coaxial cylinder or a coaxial annulus, fixed axial placement). The payload surface carries the scenario's
documented boundary condition (rigid, beta = 0: the natural Neumann condition); payload and other absorption stay
lumped in A_other as volumetric damping, evaluated with the MEASURED FLUID volume. Every gate fails closed with
`VehicleFEMError` (never `assert`), and every result is stamped with the vehicle identity (scenario hash, source corner,
payload geometry hash, payload BC hash) plus a mesh hash.

The production solver primitives are reused unchanged from `fem.solver` (whose bytes are pinned by release provenance).
"""
from __future__ import annotations

import hashlib
import math
import os
import tempfile

import numpy as np

from ..fem import errors as FERR
from ..fem import mesh as MESH
from ..fem import solver as FS
from . import scenarios as V


class VehicleFEMError(RuntimeError):
    pass


def build_vehicle_mesh(geometry, case, mesh_size=0.35):
    """Cross-section (r, z) of the cylinder + cone interior minus the payload rectangle [r_in, r_out] x [z_base, z_top]."""
    import gmsh
    import meshio
    import skfem
    R, Hb, H = geometry.R_fair, geometry.H_barrel, geometry.H_total
    fd, fn = tempfile.mkstemp(prefix="acoustic_liner_vehicle_", suffix=".msh")
    os.close(fd)
    try:
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("vehicle_interior")
            occ = gmsh.model.occ
            pts = [occ.addPoint(0.0, 0.0, 0.0), occ.addPoint(R, 0.0, 0.0), occ.addPoint(R, Hb, 0.0),
                   occ.addPoint(0.0, H, 0.0)]
            ls = [occ.addLine(pts[i], pts[(i + 1) % 4]) for i in range(4)]
            outer = occ.addPlaneSurface([occ.addCurveLoop(ls)])
            pay = occ.addRectangle(case["r_in_m"], case["z_base_m"], 0.0, case["r_out_m"] - case["r_in_m"],
                                   case["z_top_m"] - case["z_base_m"])
            occ.cut([(2, outer)], [(2, pay)])
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


def tag_vehicle_facets(mesh, geometry, tol=1e-6):
    """DISJOINT tags: 'axis' (r=0), 'source' (z=0), 'wall' (barrel r=R or cone line), 'payload' (all others)."""
    R, Hb, H = geometry.R_fair, geometry.H_barrel, geometry.H_total
    sets = {"source": set(), "axis": set(), "wall": set(), "payload": set()}
    area_by_id = {}
    bnd = list(mesh.boundary_facets())
    for fid in bnd:
        a, b = mesh.p[:, mesh.facets[0, fid]], mesh.p[:, mesh.facets[1, fid]]
        r, z = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
        area_by_id[fid] = MESH._facet_axisym_area(mesh, fid)
        on_barrel = z <= Hb + tol and abs(r - R) < tol
        on_cone = z >= Hb - tol and abs(r - R * (H - z) / (H - Hb)) < tol
        if r < tol:
            sets["axis"].add(fid)
        elif z < tol:
            sets["source"].add(fid)
        elif on_barrel or on_cone:
            sets["wall"].add(fid)
        else:
            sets["payload"].add(fid)
    return dict(sets=sets, area_by_id=area_by_id, all=set(bnd))


def fluid_components(mesh):
    """Number of edge-connected components of the fluid triangulation (must be 1)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    f2t = mesh.f2t
    inner = f2t[1] >= 0
    a, b = f2t[0, inner], f2t[1, inner]
    n = mesh.t.shape[1]
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return int(connected_components(g, directed=False)[0])


def mesh_sha256(mesh):
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.round(mesh.p, 12)).tobytes())
    h.update(np.ascontiguousarray(mesh.t.astype(np.int64)).tobytes())
    return h.hexdigest()


def _rel(got, exp):
    return abs(got - exp) / abs(exp) if exp else abs(got)


def run_vehicle_fem(vs, freqs, wall_beta, *, corner_id=None, payload_case=None, kappa=None, coverage=1.0,
                    bare_beta=None, anchor="low", p0=1.0, mesh_size=0.2, tol=FS.FEM_CONSISTENCY_TOL_DEFAULT,
                    stage="fem", parent=None):
    """Payload-resolving vehicle FEM. Returns a record stamped with the vehicle identity (and `parent` if given)."""
    if not isinstance(vs, V.VehicleScenario):
        raise VehicleFEMError(f"run_vehicle_fem requires a vehicle_scenarios.VehicleScenario; got {type(vs).__name__}")
    if stage not in ("fem", "corrected_fem"):
        raise VehicleFEMError(f"stage must be 'fem' or 'corrected_fem'; got {stage!r}")
    ident = vs.identity(corner_id, payload_case)
    if parent is not None:
        vs.require_identity(parent, ident["source_corner"], ident["payload_case"])
    case = vs.payload_cases[ident["payload_case"]]
    if not (isinstance(coverage, (int, float)) and not isinstance(coverage, bool) and 0.0 <= coverage <= 1.0):
        raise VehicleFEMError(f"coverage must be a number in [0, 1]; got {coverage!r}")
    fs = [float(f) for f in freqs]
    if not fs or any(not (math.isfinite(f) and f > 0) for f in fs):
        raise VehicleFEMError(f"freqs must be a non-empty list of finite positive values; got {fs}")
    g = vs.geometry
    ps = vs.to_physical_scenario(ident["source_corner"])

    mesh = build_vehicle_mesh(g, case, mesh_size)
    ncomp = fluid_components(mesh)
    if ncomp != 1:
        raise VehicleFEMError(f"fluid domain has {ncomp} disconnected components (payload seals a region)")
    tagged = tag_vehicle_facets(mesh, g)
    if not tagged["sets"]["payload"]:
        raise VehicleFEMError("no payload facets: the payload was not cut out of the mesh")
    A_wall = g.lined_wall_area
    try:
        partition = MESH.assert_boundary_partition(
            tagged, {"source": math.pi * g.R_fair ** 2, "wall": A_wall, "payload": case["surface_area_m2"],
                     "axis": 0.0}, tol=4 * tol)
    except FERR.FEMSolverError as e:
        raise VehicleFEMError(f"boundary partition gate failed: {e}") from e
    area = {k: sum(tagged["area_by_id"][f] for f in s) for k, s in tagged["sets"].items()}
    vol = MESH.mesh_physical_volume(mesh)
    checks = dict(wall_area=_rel(area["wall"], A_wall), payload_area=_rel(area["payload"], case["surface_area_m2"]),
                  fluid_volume=_rel(vol, case["fluid_volume_m3"]))
    bad = {k: v for k, v in checks.items() if not (math.isfinite(v) and v <= tol)}
    if bad:
        raise VehicleFEMError(f"mesh consistency gate failed (tol {tol}): {bad}; refine the mesh")

    k = float(ps.kappa_nominal if kappa is None else kappa)
    A_other = ps.A_other(k)
    if abs(A_other / A_wall - k) > 1e-12:
        raise VehicleFEMError("A_other is not kappa * lined wall area")
    if coverage >= 1.0:
        treated, treated_area = None, area["wall"]
    else:
        if bare_beta is None:
            raise VehicleFEMError("coverage < 1 requires bare_beta (silently rigid wall is not allowed)")
        treated = MESH.select_coverage_facets(mesh, tagged, coverage, anchor=anchor) if coverage > 0 else set()
        treated_area = sum(tagged["area_by_id"][f] for f in treated)
    cov_err = abs(treated_area / area["wall"] - coverage)
    if cov_err > FS.COVERAGE_CONSISTENCY_TOL:
        raise VehicleFEMError(f"achieved coverage differs from requested by {cov_err:.4f}; refine the mesh")

    beta_p = complex(vs.payload_bc["beta_real"], vs.payload_bc["beta_imag"])
    extra = [] if beta_p == 0 else [(tagged["sets"]["payload"], beta_p)]
    ms, wb = {}, {}
    for f in fs:
        beta = complex(wall_beta(f) if callable(wall_beta) else wall_beta)
        wb[f] = dict(real=beta.real, imag=beta.imag)
        ms[f] = FS.solve_interior(mesh, f, tagged, wall_beta=beta, A_other=A_other, volume=vol, p0=p0,
                                 treated_facets=treated, bare_beta=bare_beta, extra_robin=extra)
    rec = dict(
        solver="vehicle_fem/payload_resolving_axisym_v1", mesh_size=mesh_size, mesh_sha256=mesh_sha256(mesh),
        n_triangles=int(mesh.t.shape[1]), fluid_components=ncomp,
        payload=dict(case, surface_condition=vs.payload_bc["surface_condition"]),
        measured=dict(area, fluid_volume=vol), consistency=checks, consistency_tol=tol,
        boundary_partition=partition, kappa=k, A_other=A_other, A_other_damping_volume="measured_fluid_volume",
        coverage=coverage, anchor=anchor, achieved_coverage=treated_area / area["wall"],
        bare_beta=None if bare_beta is None else dict(real=complex(bare_beta).real, imag=complex(bare_beta).imag),
        freqs=fs, wall_beta_by_freq=wb, ms_by_freq=ms,
        limitations=["representative axisymmetric interior, not vehicle CAD",
                     "payload is a representative rigid obstruction, not a spacecraft model; its absorption is in A_other",
                     "relative dOASPL only"])
    return V.stamp(rec, ident, stage, parent)
