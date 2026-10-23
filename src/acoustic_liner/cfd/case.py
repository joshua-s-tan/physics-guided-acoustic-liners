"""OpenFOAM dictionaries of the incompressible imposed-oscillating-flow neck case (pimpleFoam, laminar):
the graded neck-in-reservoir block mesh, controlDict with its sampling function objects, the initial
and boundary fields, transport, turbulence, schemes and solution settings.
"""
from __future__ import annotations

import os, argparse, math
from .mesh import Verts
from .mesh import Block
from .mesh import _FOAM_HEADER
from .mesh import q
from .mesh import R_NECK
from .mesh import L_NECK
from .mesh import R_TUBE
from .mesh import WEDGE_DEG


HERE = os.path.dirname(os.path.abspath(__file__))
NU = 1.82e-5 / 1.2                 # kinematic viscosity [m^2/s]
F0 = 126.51
A_NECK = math.pi * R_NECK**2
A_TUBE = math.pi * R_TUBE**2
AREA_RATIO = A_NECK / A_TUBE       # inlet(tube) velocity = u_neck * AREA_RATIO
H_RES = 40.0e-3                    # reservoir length each side of the neck
NR_ANN = 24                       # radial cells r_neck -> R_tube (reservoir annulus)
NZ_RES = 70                       # axial cells per reservoir
G_FROM = 6.0                      # annulus radial grading toward the neck edge
G_ORIF_HI = 0.2                   # axial clustering toward a high-z orifice mouth
G_ORIF_LO = 5.0                   # axial clustering toward a low-z orifice mouth


# bore mesh: uniform tube radius r_neck, length 3*L (lead-in / measure / lead-out)
NR = 40           # radial cells 0->r_neck (graded to wall for the Stokes layer)
NZ_SEG = 60       # axial cells per L-segment
G_WALL = 0.15     # cluster radial cells at the wall


def far_cell(H_res, nz, g):
    """Axial size of the cell farthest from the neck in a reservoir block of length H_res, nz cells, end ratio g."""
    g = max(g, 1.0 / g)
    r = g ** (1.0 / (nz - 1))
    return g * H_res * (r - 1.0) / (r ** nz - 1.0)


def n_ext(H_ext, H_res=H_RES, ref=1.0):
    """Axial cells of a uniform extension block whose cell size matches the reservoir's far-end cell."""
    return max(1, round(H_ext / far_cell(H_res, int(NZ_RES * ref), G_ORIF_LO)))


def build_neck_mesh(V, H_res=H_RES, ref=1.0, H_ext=0.0):
    """H_ext > 0 adds a uniform block of length H_ext beyond each reservoir end (z < 0 and z > Z3), after the
    original blocks, so the original vertices, blocks, cells and sampling planes are unchanged."""
    nr = int(NR * ref); nra = int(NR_ANN * ref)
    nzr = int(NZ_RES * ref); nzs = int(NZ_SEG * ref)     # mesh-refinement factor
    Z0, Z1, Z2, Z3 = 0.0, H_res, H_res + L_NECK, H_res + L_NECK + H_res
    recv_c = Block(V, 0.0,    R_NECK, Z0, Z1, nr, nzr, G_WALL, G_ORIF_HI)
    recv_a = Block(V, R_NECK, R_TUBE, Z0, Z1, nra, nzr, G_FROM, G_ORIF_HI)
    neck   = Block(V, 0.0,    R_NECK, Z1, Z2, nr, nzs, G_WALL, 1.0)
    driv_c = Block(V, 0.0,    R_NECK, Z2, Z3, nr, nzr, G_WALL, G_ORIF_LO)
    driv_a = Block(V, R_NECK, R_TUBE, Z2, Z3, nra, nzr, G_FROM, G_ORIF_LO)
    blocks = [recv_c, recv_a, neck, driv_c, driv_a]
    inlet = [driv_c.face_zhi(), driv_a.face_zhi()]
    outlet = [recv_c.face_zlo(), recv_a.face_zlo()]
    walls = [recv_a.face_rhi(), driv_a.face_rhi(),
             recv_a.face_zhi(), driv_a.face_zlo(), neck.face_rhi()]
    if H_ext > 0:
        ne = n_ext(H_ext, H_res, ref)
        lo_c = Block(V, 0.0,    R_NECK, Z0 - H_ext, Z0, nr, ne, G_WALL, 1.0)
        lo_a = Block(V, R_NECK, R_TUBE, Z0 - H_ext, Z0, nra, ne, G_FROM, 1.0)
        hi_c = Block(V, 0.0,    R_NECK, Z3, Z3 + H_ext, nr, ne, G_WALL, 1.0)
        hi_a = Block(V, R_NECK, R_TUBE, Z3, Z3 + H_ext, nra, ne, G_FROM, 1.0)
        blocks += [lo_c, lo_a, hi_c, hi_a]
        inlet = [hi_c.face_zhi(), hi_a.face_zhi()]
        outlet = [lo_c.face_zlo(), lo_a.face_zlo()]
        walls += [lo_a.face_rhi(), hi_a.face_rhi()]
        axis_ext = [lo_c.face_rlo(), hi_c.face_rlo()]
    axis = [recv_c.face_rlo(), neck.face_rlo(), driv_c.face_rlo()] + (axis_ext if H_ext > 0 else [])
    wm = [b.face_wedge_m() for b in blocks]; wp = [b.face_wedge_p() for b in blocks]

    def patch(name, ptype, faces):
        body = "\n".join(q(f) for f in faces)
        return (f"    {name}\n    {{\n        type {ptype};\n"
                f"        faces\n        (\n{body}\n        );\n    }}\n")
    boundary = (patch("inlet", "patch", inlet) + patch("outlet", "patch", outlet) +
                patch("walls", "wall", walls) + patch("axis", "empty", axis) +
                patch("front", "wedge", wm) + patch("back", "wedge", wp))
    return _FOAM_HEADER("dictionary", "blockMeshDict") + f"""
scale 1;
vertices
(
{V.dump()}
);
blocks
(
{chr(10).join(b.hexs() for b in blocks)}
);
edges ( );
boundary
(
{boundary});
mergePatchPairs ( );
"""


def build_controldict_neck(endTime, H_res=H_RES, deltaT=2e-6, maxCo=0.5,
                           maxDeltaT=2e-5, writeInterval=1e-3):
    Z1, Z2 = H_res, H_res + L_NECK
    zmid = Z1 + 0.5 * L_NECK
    # planes bracket the neck (out of the on-axis jet not needed for area-averaged p)
    fos = (_sfv("pUp",   Z2 + 0.015, "areaAverage", "p") +
           _sfv("pDown", Z1 - 0.015, "areaAverage", "p") +
           _sfv("pUp2",  Z2 + 0.025, "areaAverage", "p") +
           _sfv("pDown2",Z1 - 0.025, "areaAverage", "p") +
           _sfv("fluxMid", zmid, "areaNormalIntegrate", "U") +
           _sfv("fluxUp",  Z2 + 0.015, "areaNormalIntegrate", "U") +
           _sfv("fluxDown",Z1 - 0.015, "areaNormalIntegrate", "U"))
    return _FOAM_HEADER("dictionary", "controlDict") + f"""
application     pimpleFoam;
startFrom       startTime; startTime 0; stopAt endTime;
endTime         {endTime};
deltaT          {deltaT};
adjustTimeStep  yes; maxCo {maxCo}; maxDeltaT {maxDeltaT};
writeControl    adjustableRunTime; writeInterval {writeInterval};
purgeWrite      4; writeFormat ascii; writePrecision 8; writeCompression off;
timeFormat      general; timePrecision 10; runTimeModifiable yes;
functions
{{
{fos}}}
"""


def build_fields(u0):
    """Incompressible fields. Imposed inlet velocity (0 0 -u0)*sin; kinematic p."""
    U = _FOAM_HEADER("volVectorField", "U", "0") + f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{{
    inlet
    {{
        type            uniformFixedValue;
        uniformValue
        {{
            type   sine; frequency {F0}; amplitude 1;
            scale  (0 0 {-u0:.6g}); level (0 0 0); t0 0;
        }}
    }}
    outlet          {{ type pressureInletOutletVelocity; value uniform (0 0 0); }}
    walls           {{ type noSlip; }}
    axis            {{ type empty; }}
    front           {{ type wedge; }}
    back            {{ type wedge; }}
}}
"""
    # kinematic pressure [m^2/s^2]; reference 0 at outlet
    p = _FOAM_HEADER("volScalarField", "p", "0") + """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet           { type zeroGradient; }
    outlet          { type fixedValue; value uniform 0; }
    walls           { type zeroGradient; }
    axis            { type empty; }
    front           { type wedge; }
    back            { type wedge; }
}
"""
    return {"U": U, "p": p}


def build_transport():
    return _FOAM_HEADER("dictionary", "transportProperties") + f"""
transportModel  Newtonian;
nu              {NU:.6g};
"""


def build_turbulence():
    return _FOAM_HEADER("dictionary", "turbulenceProperties") + """
simulationType  laminar;
"""


def _sfv(name, zpos, op, field):
    return f"""    {name}
    {{
        type surfaceFieldValue; libs (fieldFunctionObjects);
        writeControl timeStep; writeInterval 1; writeFields false;
        surfaceFormat none; regionType sampledSurface; name {name}Surf;
        operation {op}; fields ({field});
        sampledSurfaceDict
        {{ type plane; planeType pointAndNormal;
          pointAndNormalDict {{ point (0 0 {zpos:.6f}); normal (0 0 1); }}
          interpolate true; }}
    }}
"""


def build_fvschemes():
    return _FOAM_HEADER("dictionary", "fvSchemes") + """
ddtSchemes      { default backward; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      Gauss linearUpwindV grad(U);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
"""


def build_fvsolution():
    return _FOAM_HEADER("dictionary", "fvSolution") + """
solvers
{
    p
    {
        solver GAMG; smoother GaussSeidel;
        tolerance 1e-8; relTol 0.01;
    }
    pFinal { $p; relTol 0; }
    "(U|k|epsilon)" { solver PBiCGStab; preconditioner DILU; tolerance 1e-8; relTol 0.01; }
    "(U|k|epsilon)Final" { $U; relTol 0; }
}
PIMPLE { nOuterCorrectors 2; nCorrectors 2; nNonOrthogonalCorrectors 1; }
relaxationFactors { equations { ".*" 1; } }
"""
