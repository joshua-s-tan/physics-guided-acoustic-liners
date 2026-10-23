"""Meshes of the fairing interior (r, z half-plane), their measured volume, the disjoint boundary-facet
tags (source, wall, axis) with axisymmetric facet areas, the facet-partition gate, and the selection of
the lined wall facets for a coverage fraction and placement anchor.
"""
from __future__ import annotations

import math
import os
import tempfile
import numpy as np
from .errors import FEMSolverError
from .forms import _forms


def build_fairing_interior_mesh(R_fair: float, H_barrel: float, H_total: float, mesh_size: float = 0.35):
    """Axisymmetric fairing interior cross-section: base (z=0), barrel wall, conical nose wall, axis (r=0)."""
    import gmsh
    import meshio
    import skfem
    if not all(math.isfinite(x) and x > 0 for x in (R_fair, H_barrel, H_total, mesh_size)):
        raise FEMSolverError("mesh dimensions must be finite and > 0")
    if H_total <= H_barrel:
        raise FEMSolverError("H_total must exceed H_barrel")
    fd, fn = tempfile.mkstemp(prefix="acoustic_liner_fairing_", suffix=".msh")   # collision-safe unique path
    os.close(fd)
    try:                                           # OUTER finally removes the temp file even if gmsh fails
        gmsh.initialize()
        try:
            gmsh.model.add("fairing_interior")
            occ = gmsh.model.occ
            p1 = occ.addPoint(0.0, 0.0, 0.0)
            p2 = occ.addPoint(R_fair, 0.0, 0.0)
            p3 = occ.addPoint(R_fair, H_barrel, 0.0)
            p4 = occ.addPoint(0.0, H_total, 0.0)
            ls = [occ.addLine(p1, p2), occ.addLine(p2, p3), occ.addLine(p3, p4), occ.addLine(p4, p1)]
            occ.addPlaneSurface([occ.addCurveLoop(ls)])
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
            os.remove(fn)                          # clean up the temporary mesh file (any failure path)
        except OSError:
            pass


def mesh_physical_volume(mesh) -> float:
    """Physical volume of the axisymmetric body of revolution: V = 2*pi * integral(r dA) over the cross-
    section. Computed from the assembled r-weighted mass matrix (ones^T M ones = integral(r dA))."""
    skfem, BilinearForm, LinearForm, FacetBasis, spsolve, axisym_stiffness, axisym_mass = _forms()
    basis = skfem.Basis(mesh, skfem.ElementTriP1())
    M = axisym_mass.assemble(basis)
    ones = np.ones(basis.N)
    r_integral = float(np.real(ones @ (M @ ones)))
    return 2.0 * math.pi * r_integral


def _facet_axisym_area(mesh, fid: int) -> float:
    """Axisymmetric area of a boundary facet: 2*pi*r_mid*length (0 on the r=0 axis)."""
    a, b = mesh.p[:, mesh.facets[0, fid]], mesh.p[:, mesh.facets[1, fid]]
    r_mid = 0.5 * (a[0] + b[0])
    length = float(np.hypot(a[0] - b[0], a[1] - b[1]))
    return 2.0 * math.pi * r_mid * length


def tag_boundary_facets(mesh, tol: float = 1e-6) -> dict:
    """Partition boundary facets into DISJOINT tags: 'source' (base z=0), 'axis' (r=0 symmetry), 'wall'
    (r>0, z>0). Returns {sets: {tag: set(ids)}, area_by_id: {id: area}, all: set(ids)}."""
    bnd = list(mesh.boundary_facets())
    pts = mesh.p
    sets = {"source": set(), "axis": set(), "wall": set()}
    area_by_id = {}
    for fid in bnd:
        a, b = pts[:, mesh.facets[0, fid]], pts[:, mesh.facets[1, fid]]
        mid_r = 0.5 * (a[0] + b[0]); mid_z = 0.5 * (a[1] + b[1])
        area_by_id[fid] = _facet_axisym_area(mesh, fid)
        if mid_r < tol:
            sets["axis"].add(fid)
        elif mid_z < tol:
            sets["source"].add(fid)
        else:
            sets["wall"].add(fid)
    return dict(sets=sets, area_by_id=area_by_id, all=set(bnd))


def assert_boundary_partition(tagged: dict, expected_areas: dict, tol: float = 0.02) -> dict:
    """Real-mesh facet-partition gate for the fairing interior: tags DISJOINT, union == all boundary facets,
    every facet has a finite (>=0; axis facets may be ~0) area, and each tag's COMPUTED area (summed from its
    facets) matches its expected analytic area within `tol`. Areas are computed from facets, never trusted."""
    if not (math.isfinite(tol) and 0.0 < tol < 0.5):
        raise FEMSolverError(f"tol must be finite in (0,0.5); got {tol}")
    sets = tagged["sets"]; area_by_id = tagged["area_by_id"]; allb = tagged["all"]
    keys = list(sets)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if sets[keys[i]] & sets[keys[j]]:
                raise FEMSolverError(f"tags {keys[i]!r}/{keys[j]!r} overlap")
    union = set().union(*sets.values()) if sets else set()
    if union != set(allb):
        raise FEMSolverError(f"tag union != boundary facets (untagged {sorted(set(allb)-union)[:5]})")
    for fid in allb:
        area = area_by_id.get(fid)
        if area is None or not (math.isfinite(area) and area >= 0):
            raise FEMSolverError(f"facet {fid} has invalid area {area}")
    errs = {}
    for tag, exp in expected_areas.items():
        got = sum(area_by_id[f] for f in sets[tag])
        e = abs(got - exp) / exp if exp > 0 else abs(got)
        if not (math.isfinite(e) and e <= tol):
            raise FEMSolverError(f"tag {tag!r} computed area {got:.4f} vs expected {exp:.4f} (rel {e:.4f} > {tol})")
        errs[tag] = e
    return errs


def select_coverage_facets(mesh, tagged: dict, coverage: float, anchor: str = "low") -> set:
    """Select a CONTIGUOUS axial band of WALL facets covering `coverage` of total wall AREA (axisym area
    weighted). anchor 'low' grows from the base, 'high' from the apex, 'mid' grows CENTER-OUT from the wall's
    mid-height. coverage>=1 -> all wall facets, coverage<=0 -> none."""
    if anchor not in ("low", "high", "mid"):
        raise FEMSolverError(f"anchor must be one of low|high|mid; got {anchor!r}")
    if not (math.isfinite(coverage) and 0.0 <= coverage <= 1.0):
        raise FEMSolverError(f"coverage must be finite in [0,1]; got {coverage}")
    wall = sorted(tagged["sets"]["wall"])
    ab = tagged["area_by_id"]
    if coverage >= 1.0:
        return set(wall)
    if coverage <= 0.0:
        return set()
    pts = mesh.p
    zc = {f: 0.5 * (pts[1, mesh.facets[0, f]] + pts[1, mesh.facets[1, f]]) for f in wall}
    if anchor == "mid":
        z_center = 0.5 * (min(zc.values()) + max(zc.values()))      # center-out from wall mid-height
        order = sorted(wall, key=lambda f: abs(zc[f] - z_center))
    else:
        order = sorted(wall, key=lambda f: zc[f], reverse=(anchor == "high"))
    total = sum(ab[f] for f in wall)
    target = coverage * total
    # choose the contiguous prefix whose cumulative area is CLOSEST to the target (minimizes overshoot);
    # always keep at least one facet. Requested coverage stays authoritative -- this only reduces the
    # unavoidable discretization gap between requested and achievable facet-area coverage.
    cum, best, best_err, pref = 0.0, None, float("inf"), []
    for f in order:
        cum += ab[f]; pref.append(f)
        err = abs(cum - target)
        if err < best_err:
            best_err, best = err, list(pref)
        elif cum >= target:
            break                       # past target and error growing -> closest prefix already found
    return set(best if best else order[:1])
