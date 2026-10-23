# -*- coding: utf-8 -*-
"""VISUALIZATION-ONLY complex pressure-field export of the payload-resolving vehicle FEM.

Opt-in and self-contained: `fem.solver` and `vehicles.fem` are NOT modified (their bytes are pinned by release
provenance). `solve_field` repeats the assembly of `fem.solver.solve_interior` term by term and additionally returns
the nodal complex pressure; `verify_against_production` proves on every export that the field reproduces the
production solver's mean-square pressure and the frozen record's band noise reduction. Nothing here writes to, or is
read by, any evidence record. Both vehicle verdicts remain UNRESOLVED.

Phasor convention: e^{+i omega t} (the solver's volumetric loss k^2 (1 - i eta) decays only under it), so
p(x, t) = Re p_hat(x) cos(omega t) - Im p_hat(x) sin(omega t). Pressures are normalized by the incident source
amplitude p0 (= 1): "normalized complex acoustic pressure p / p_inc", not calibrated SPL.

Frequency selection (deterministic, written before rendering, from recorded values only): in the authoritative
initial-FEM record (the vehicle's `authoritative` source corner | payload case at that mesh size) take the robust
corner; band i's share of the broadband reduction is g_i = w_i (1 - 10^(-NR_i/10)) with w_i the energy weights of
`oracle.delta_oaspl`; select argmax g_i, ties to the lower frequency. Field appearance plays no part.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ..physics import absorption as acoustics
from ..physics import constants
from ..fem import robust as DRUN
from ..fem import forms as FORMS
from ..fem import mesh as MESH
from ..fem import solver as FS
from ..physics import oracle as O2
from ..design import canonical as CAN
from ..vehicles import fem as VF
from ..vehicles import search as VP
from ..provenance import hashing as H
from ..provenance import schemas as SC

PUBLIC_SOURCE_FILES = ("visualization/fem_fields.py", "fem/solver.py", "vehicles/fem.py", "vehicles/search.py")

C_AIR = FS.C_AIR
VISUALIZATION_ONLY = True
PHASOR = "e^{+i omega t}; p(x,t) = Re(p_hat) cos(omega t) - Im(p_hat) sin(omega t)"
NORMALIZATION = "p / p_inc, p_inc = incident source amplitude p0 = 1 (dimensionless); not calibrated SPL"
BOUNDARY_TAGS = {"source": 1, "wall": 2, "payload": 3, "axis": 4}
REL_FLOOR = 1e-6          # relative-level floor: |p| clipped below REL_FLOOR * max|p_bare| before 20 log10


CFD_CORRECTED_OMITTED = {
    "falcon": "omitted: no completed central CFD-corrected FEM exists (the 16.2 m/s impedance pair did not satisfy the "
              "statistical-stability checks)",
    "atlas": "omitted by project decision: only the bare and frozen linear-impedance states are shown; the vehicle "
             "verdict remains UNRESOLVED"}


class FEMVisualizeError(RuntimeError):
    pass


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


def solve_field(mesh, freq, tagged, wall_beta, A_other=0.0, volume=None, p0=1.0, treated_facets=None,
                bare_beta=None, extra_robin=None):
    """Same system as `fem.solver.solve_interior` (production base radiation), returning (p_hat, ms / vol)."""
    skfem, BilinearForm, LinearForm, FacetBasis, spsolve, axisym_stiffness, axisym_mass = FORMS._forms()
    wall_beta = FS._require_passive_beta("wall_beta", wall_beta)
    k = 2.0 * math.pi * freq / C_AIR
    eta = 0.0
    if A_other and A_other > 0:
        from ..fem import damping
        eta = damping.volumetric_loss_factor(float(A_other), float(volume), freq)
    kc2 = (k ** 2) * (1.0 - 1j * eta)
    el = skfem.ElementTriP1()
    basis = skfem.Basis(mesh, el)
    A = axisym_stiffness.assemble(basis).astype(complex) - kc2 * axisym_mass.assemble(basis).astype(complex)
    sets = tagged["sets"]
    base = np.array(sorted(sets["source"]), dtype=int)

    def robin_matrix(facets, beta):
        fb = FacetBasis(mesh, el, facets=np.array(sorted(facets), dtype=int))

        @BilinearForm(dtype=complex)
        def robin(u, v, w):
            return 1j * k * beta * w.x[0] * u * v
        return robin.assemble(fb)

    fb_base = FacetBasis(mesh, el, facets=base)

    @BilinearForm(dtype=complex)
    def rad(u, v, w):
        return 1j * k * w.x[0] * u * v
    A = A + rad.assemble(fb_base)
    wall_all = sets["wall"]
    if treated_facets is None:
        if wall_beta != 0 and wall_all:
            A = A + robin_matrix(wall_all, wall_beta)
    else:
        treated = set(treated_facets)
        untreated = wall_all - treated
        if wall_beta != 0 and treated:
            A = A + robin_matrix(treated, wall_beta)
        if untreated:
            bb = FS._require_passive_beta("bare_beta", bare_beta)
            if bb != 0:
                A = A + robin_matrix(untreated, bb)
    for facets, beta in (extra_robin or []):
        b_ = FS._require_passive_beta("extra_robin_beta", beta)
        if b_ != 0 and len(facets):
            A = A + robin_matrix(set(facets), b_)

    @LinearForm(dtype=complex)
    def src(v, w):
        return 2j * k * p0 * w.x[0] * v
    b = src.assemble(fb_base)
    M = axisym_mass.assemble(basis)
    p = spsolve(A.tocsc(), b)
    if not np.all(np.isfinite(p)):
        raise FEMVisualizeError(f"non-finite field at {freq} Hz")
    ms = float(np.real(np.vdot(p, M @ p)))
    vol = float(np.real(np.ones(len(p)) @ (M @ np.ones(len(p)))))
    return p, ms / vol


def authoritative_case(vehicle):
    """(source corner, payload case, mesh size) of the vehicle's frozen initial-FEM authoritative entry."""
    d = json.loads((VP.RUN_DIR / VP.scenario(vehicle).name / "initial_fem.json").read_text())
    corner_case, mesh = d["authoritative"].split(" at mesh ")
    corner, case = corner_case.split("|")
    return corner, case, float(mesh.rstrip(" m")), d


def pick_band(f, L, nr):
    """(index, shares g_i) of the band with the largest share of the broadband reduction; ties -> lower frequency."""
    f, L, nr = (np.asarray(x, float) for x in (f, L, nr))
    e = 10.0 ** ((L - L.max()) / 10.0)
    w = e / e.sum()
    g = w * (1.0 - 10.0 ** (-nr / 10.0))
    return max(range(len(f)), key=lambda i: (g[i], -f[i])), g


def select_frequency(vehicle):
    vs = VP.scenario(vehicle)
    corner, case, h, d = authoritative_case(vehicle)
    rec = next(r for r in d["records"] if r["source_corner"] == corner and r["payload_case"] == case
               and r["mesh_size"] == h and r["stage"] == "fem")
    rid = rec["robust"]["robust_corner_id"]
    c = next(c for c in rec["corners"] if list(c["corner_id"]) == list(rid))
    ps = vs.to_physical_scenario(corner)
    f = np.asarray(ps.band_hz, float)
    L = np.asarray(ps.spl_dB, float)
    nr = np.asarray(c["nr_band"], float)
    best, g = pick_band(f, L, nr)
    return dict(rule="argmax_i w_i (1 - 10^(-NR_i/10)), w_i = spectrum energy weights of oracle.delta_oaspl; "
                     "ties -> lower frequency; robust corner of the authoritative initial-FEM record",
                freq=float(f[best]), index=int(best), bands_hz=f.tolist(), share=(g / g.sum()).tolist(),
                nr_band=nr.tolist(), spl_dB=L.tolist(), source_corner=corner, payload_case=case, mesh_size=h,
                robust_corner=list(rid), kappa=float(c["kappa"]), anchor=c["anchor"],
                record_mesh_sha256=c["mesh_sha256"], record_coverage=rec["coverage"])


def compute(vehicle):
    """Bare and lined (frozen linear resonator-model impedance) complex fields at the selected band."""
    vs = VP.scenario(vehicle)
    fin = VP.load_finalist(vs)
    sel = select_frequency(vehicle)
    corner, case, h = sel["source_corner"], sel["payload_case"], sel["mesh_size"]
    g = vs.geometry
    pc = vs.payload_cases[case]
    mesh = VF.build_vehicle_mesh(g, pc, h)
    msha = VF.mesh_sha256(mesh)
    if msha != sel["record_mesh_sha256"]:
        raise FEMVisualizeError(f"mesh hash {msha[:12]} differs from the frozen record {sel['record_mesh_sha256'][:12]}")
    tagged = VF.tag_vehicle_facets(mesh, g)
    vol = MESH.mesh_physical_volume(mesh)
    ps = vs.to_physical_scenario(corner)
    A_other = ps.A_other(sel["kappa"])
    design = dict(CAN.canonical_design(fin["design"]))
    design["groups"] = [dict(gr, cavity_volume=O2._group_cavity_volume(gr)) for gr in design["groups"]]
    beta_fn = DRUN._wall_beta_fn(O2, design["groups"], design["D"])
    bare = constants.Z_AIR * acoustics.admittance_from_diffuse_alpha(DRUN.ALPHA_BARE_NOMINAL, constants.Z_AIR)
    cov = float(fin["coverage"])
    treated = None if cov >= 1.0 else MESH.select_coverage_facets(mesh, tagged, cov, anchor=sel["anchor"])
    beta_p = complex(vs.payload_bc["beta_real"], vs.payload_bc["beta_imag"])
    extra = [] if beta_p == 0 else [(tagged["sets"]["payload"], beta_p)]
    f = sel["freq"]
    common = dict(A_other=A_other, volume=vol, p0=1.0, extra_robin=extra)
    beta_t = complex(beta_fn(f))
    p_bare, ms_bare = solve_field(mesh, f, tagged, bare, **common)
    p_trt, ms_trt = solve_field(mesh, f, tagged, beta_t, treated_facets=treated,
                                bare_beta=None if treated is None else bare, **common)
    ver = verify_against_production(mesh, f, tagged, bare, beta_t, treated, common, ms_bare, ms_trt, sel)
    treated_set = tagged["sets"]["wall"] if treated is None else set(treated)
    ident = vs.identity(corner, case)
    return dict(vehicle=vehicle, vs=vs, fin=fin, sel=sel, mesh=mesh, mesh_sha256=msha, tagged=tagged,
                treated_facets=treated_set, p_bare=p_bare, p_treated=p_trt, ms_bare=ms_bare, ms_treated=ms_trt,
                beta_bare=bare, beta_treated=beta_t, A_other=A_other, volume=vol, identity=ident,
                verification=ver, payload=pc, coverage=cov)


def verify_against_production(mesh, f, tagged, bare, beta_t, treated, common, ms_bare, ms_trt, sel):
    kw = dict(A_other=common["A_other"], volume=common["volume"], p0=1.0, extra_robin=common["extra_robin"])
    prod_bare = FS.solve_interior(mesh, f, tagged, wall_beta=bare, **kw)
    prod_trt = FS.solve_interior(mesh, f, tagged, wall_beta=beta_t, treated_facets=treated,
                                bare_beta=None if treated is None else bare, **kw)
    nr = 10.0 * math.log10(ms_bare / ms_trt)
    nr_rec = sel["nr_band"][sel["index"]]
    out = dict(ms_bare_rel_err=abs(ms_bare - prod_bare) / prod_bare, ms_treated_rel_err=abs(ms_trt - prod_trt) / prod_trt,
               nr_field_dB=nr, nr_record_dB=nr_rec, nr_abs_err_dB=abs(nr - nr_rec))
    if not (out["ms_bare_rel_err"] < 1e-10 and out["ms_treated_rel_err"] < 1e-10 and out["nr_abs_err_dB"] < 1e-8):
        raise FEMVisualizeError(f"field does not reproduce the production solver / frozen record: {out}")
    return out


def derived(R):
    pb, pt = R["p_bare"], R["p_treated"]
    floor = REL_FLOOR * float(np.max(np.abs(pb)))
    return dict(diff=pt - pb, abs_diff=np.abs(pt - pb),
                rel_dB=20.0 * np.log10(np.maximum(np.abs(pt), floor) / np.maximum(np.abs(pb), floor)),
                floor=floor)


def boundary_lines(R):
    m, t = R["mesh"], R["tagged"]["sets"]
    rows = []
    for name, tag in BOUNDARY_TAGS.items():
        for fid in sorted(t[name]):
            rows.append((m.facets[0, fid], m.facets[1, fid], tag, int(name == "wall" and fid in R["treated_facets"]),
                         int(name == "wall" and fid not in R["treated_facets"])))
    return np.array(rows, dtype=np.int64)


def export(R, out_dir):
    """ParaView VTU files (bare, treated, difference, boundary) plus a manifest binding every input and output."""
    import meshio
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = R["mesh"]
    pts = np.column_stack([m.p[0], m.p[1], np.zeros(m.p.shape[1])])
    cells = [("triangle", m.t.T.astype(np.int64))]
    D = derived(R)
    files = {}

    def write(name, pdata):
        fn = out / f"{R['vehicle']}_{name}.vtu"
        meshio.write(fn, meshio.Mesh(pts, cells, point_data=pdata), binary=False)
        files[fn.name] = sha_file(fn)

    for state, p in (("bare", R["p_bare"]), ("treated", R["p_treated"])):
        write(f"fem_field_{state}", dict(p_real=p.real, p_imag=p.imag, p_abs=np.abs(p),
                                         p_phase_deg=np.degrees(np.angle(p))))
    write("fem_field_treated_minus_bare", dict(dp_real=D["diff"].real, dp_imag=D["diff"].imag, dp_abs=D["abs_diff"],
                                               rel_level_dB=D["rel_dB"]))
    bl = boundary_lines(R)
    fn = out / f"{R['vehicle']}_fem_boundaries.vtu"
    meshio.write(fn, meshio.Mesh(pts, [("line", bl[:, :2])],
                                 cell_data=dict(boundary_tag=[bl[:, 2]], treated_wall=[bl[:, 3]],
                                                bare_wall=[bl[:, 4]])), binary=False)
    files[fn.name] = sha_file(fn)
    vs, fin, sel = R["vs"], R["fin"], R["sel"]
    fin_path = VP.RUN_DIR / vs.name / "finalist.json"
    ini_path = VP.RUN_DIR / vs.name / "initial_fem.json"
    man = dict(
        schema=SC.schema("fem-field-manifest"), visualization_only=True, not_evidence="never read by any verdict, sensitivity or evidence record",
        vehicle=R["vehicle"], scenario_name=vs.name, scenario_hash=vs.scenario_hash,
        identity=R["identity"], finalist_sha256=sha_file(fin_path), design_sha256=fin["design_sha256"],
        geometry_hash=fin["geometry_hash"], initial_fem_sha256=sha_file(ini_path),
        mesh=dict(mesh_size_m=sel["mesh_size"], mesh_sha256=R["mesh_sha256"], n_nodes=int(m.p.shape[1]),
                  n_triangles=int(m.t.shape[1]), element="P1 triangle, axisymmetric (r, z)"),
        frequency_hz=sel["freq"], selection=sel, source_corner=sel["source_corner"],
        payload_case=sel["payload_case"], payload=R["payload"],
        kappa=sel["kappa"], A_other_m2=R["A_other"], alpha_bare=DRUN.ALPHA_BARE_NOMINAL,
        coverage=R["coverage"], placement_anchor=sel["anchor"],
        states=dict(bare=dict(wall_beta=[R["beta_bare"].real, R["beta_bare"].imag],
                              impedance="bare wall, diffuse alpha = 0.05 everywhere on the lined-wall boundary"),
                    treated=dict(wall_beta=[R["beta_treated"].real, R["beta_treated"].imag],
                                 impedance="frozen finalist linear resonator-model surface impedance "
                                           "(initial FEM, wall_impedance = linear_oracle_z_surface); not CFD-corrected")),
        cfd_corrected_state=CFD_CORRECTED_OMITTED[R["vehicle"]],
        solver="direct sparse LU (scipy spsolve), no iterative tolerance; assembly identical to "
               "fem_solver.solve_interior (production base radiation)",
        source_model="base plane z = 0: radiation Robin + incident wave of amplitude p0 = 1",
        normalization=NORMALIZATION, units="dimensionless (p / p_inc)", phasor=PHASOR,
        relative_level=dict(formula="20 log10(max(|p_t|, floor) / max(|p_b|, floor))", floor=D["floor"],
                            floor_rule=f"{REL_FLOOR} * max|p_bare|"),
        difference="|p_hat_treated - p_hat_bare| (complex-field difference)",
        boundary_tags=BOUNDARY_TAGS, verification=R["verification"],
        # the files of this package that compute a field now (SHA-256 with LF line endings)
        public_source_files={n: H.package_file_sha256(n) for n in PUBLIC_SOURCE_FILES},
        outputs=files)
    (out / f"{R['vehicle']}_fem_field_manifest.json").write_text(json.dumps(man, indent=1, default=float))
    return man
