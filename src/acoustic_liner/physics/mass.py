# -*- coding: utf-8 -*-
"""Total-installed-mass model with the CORRECTED shared-wall partition term (remediation §4.6; frozen owner
decision: construction topology = SHARED-WALL square-cell grid).

The historical v1 term (`fairing_oracle.areal_mass`) used `4 * wall_t * D * sqrt(n)` per unit group area,
i.e. it charged every cell for all 4 of its walls. In a continuous square-cell grid the internal walls are
SHARED between adjacent cells, so the asymptotic wall length per unit footprint area is `2/side = 2*sqrt(n)`,
not `4/side`. The old expression therefore roughly DOUBLED partition mass. This module uses the shared-wall
coefficient (2.0) and proves convergence of a finite N x N grid to that analytic areal formula in the tests.

Mass is built from NAMED terms (facesheet, partition walls, backing, edge/frame, adhesive, fasteners) rather
than one unexplained lump; every allowance is a documented placeholder to be replaced by a real panel BOM.
Per the spec, do NOT describe an output as "hardware-feasible" on mass alone -- geometry/BOM feasibility is in
`feasibility.py`.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

# --- documented material constant (carried from v1 fairing_oracle for continuity) ------------------- #
RHO_WALL = 1550.0          # kg/m^3, light CFRP/panel material (areal-mass model)

# Wall-length-per-unit-area coefficient by construction topology. For a square grid of pitch `s` the wall
# length per unit footprint area is coeff/s. SHARED walls -> 2/s; each cell owning all 4 walls -> 4/s.
_TOPOLOGY_COEFF = {"shared_wall": 2.0, "individual_cell": 4.0}
DEFAULT_TOPOLOGY = "shared_wall"


class MassModelError(ValueError):
    """A mass-model input is invalid (negative density/thickness, unknown topology, bad coverage, ...)."""


def _require_positive(name: str, value: float, allow_zero: bool = False) -> float:
    v = float(value)
    if not math.isfinite(v) or (v < 0) or (v == 0 and not allow_zero):
        raise MassModelError(f"{name} must be finite and {'>= 0' if allow_zero else '> 0'}; got {value}")
    return v


def partition_wall_length_per_area(n_cells_per_m2: float, topology: str = DEFAULT_TOPOLOGY) -> float:
    """Analytic (asymptotic) partition-wall LENGTH per unit footprint area [m / m^2] for square cells of
    density `n_cells_per_m2` (so pitch s = 1/sqrt(n)). Shared-wall grid -> 2*sqrt(n)."""
    if topology not in _TOPOLOGY_COEFF:
        raise MassModelError(f"unknown topology {topology!r}; known: {sorted(_TOPOLOGY_COEFF)}")
    n = _require_positive("n_cells_per_m2", n_cells_per_m2, allow_zero=True)
    return _TOPOLOGY_COEFF[topology] * math.sqrt(n)


def finite_grid_wall_length_per_area(n_side: int, side: float, include_outer_boundary: bool = True) -> float:
    """Wall length per unit footprint area for a FINITE `n_side x n_side` grid of square cells of pitch
    `side`. Used by the convergence test: as `n_side -> inf` this tends to the analytic shared-wall value
    `2/side`. With outer boundary walls included there are (n_side+1) grid lines per direction; excluded
    (walls truncated at the patch edge / shared with a neighbouring patch) there are (n_side-1)."""
    if int(n_side) < 1:
        raise MassModelError(f"n_side must be >= 1; got {n_side}")
    s = _require_positive("side", side)
    lines_per_dir = (n_side + 1) if include_outer_boundary else (n_side - 1)
    L = n_side * s
    total_wall_length = 2.0 * lines_per_dir * L            # two orthogonal directions
    return total_wall_length / (L * L)                     # = 2*(n_side±1)/(n_side*s) -> 2/s


def partition_areal_mass(n_cells_per_m2: float, wall_t: float, D: float, rho_wall: float = RHO_WALL,
                         topology: str = DEFAULT_TOPOLOGY) -> float:
    """Partition-wall mass per unit footprint area [kg/m^2] = rho * wall_t * D * (wall length per area)."""
    wall_t = _require_positive("wall_t", wall_t, allow_zero=True)
    D = _require_positive("D", D, allow_zero=True)
    rho_wall = _require_positive("rho_wall", rho_wall)
    return rho_wall * wall_t * D * partition_wall_length_per_area(n_cells_per_m2, topology)


@dataclass(frozen=True)
class InstallBOM:
    """NAMED installation allowances [kg/m^2 of treated area] (remediation §4.6: named terms, not a lump).
    Defaults are DOCUMENTED PLACEHOLDERS whose sum (0.30) reproduces the historical lumped `mu_A=0.30`; each
    is to be replaced by a real panel BOM. `backing_t` adds a full-area backing sheet (0 = none). ALL fields
    are validated at construction (fail-closed): a negative term can never reach the mass calc."""
    adhesive_areal: float = 0.10          # bonding adhesive per treated area
    fastener_areal: float = 0.12          # mechanical fasteners / rivets
    frame_edge_areal: float = 0.08        # edge framing / close-outs, amortized over treated area
    manufacturing_margin: float = 0.0     # explicit contingency (default none)
    backing_t: float = 0.0                # m; full-area backing sheet thickness (0 = none)

    def __post_init__(self):
        for nm in ("adhesive_areal", "fastener_areal", "frame_edge_areal", "manufacturing_margin", "backing_t"):
            _require_positive(nm, getattr(self, nm), allow_zero=True)

    def install_areal_mass(self) -> float:
        return self.adhesive_areal + self.fastener_areal + self.frame_edge_areal + self.manufacturing_margin

    def backing_areal_mass(self, rho_wall: float = RHO_WALL) -> float:
        return _require_positive("rho_wall", rho_wall) * self.backing_t


# The legacy lumped allowance, kept only to document continuity with v1 (sum of the named terms above).
MU_A_LUMPED_LEGACY = 0.30


AREA_FRAC_SUM_TOL = 0.02       # must MATCH feasibility.FeasibilityLimits.area_frac_sum_tol


def local_cell_density(cavity_volume: float, D: float) -> float:
    """Cells per m^2 of a group's PATCH, derived from the SAME geometry the feasibility layer uses:
    footprint a = cavity_volume / D, so n_local = 1/a = D / cavity_volume. This guarantees mass and
    feasibility describe the SAME cell footprint (review round 4.1-e finding 3) -- no independently injected
    density that could disagree."""
    V = _require_positive("cavity_volume", cavity_volume)
    D = _require_positive("D", D)
    return D / V


def liner_areal_mass(groups, D: float, face_t: float, wall_t: float, rho_wall: float = RHO_WALL,
                     topology: str = DEFAULT_TOPOLOGY, bom: InstallBOM | None = None,
                     area_frac_sum_tol: float = AREA_FRAC_SUM_TOL) -> float:
    """Liner mass per unit WALL area [kg/m^2]: facesheet (full area) + shared-wall partitions (area-fraction
    weighted over groups) + optional backing sheet. Each group provides `area_frac` and `cavity_volume`;
    local cell density is DERIVED (`local_cell_density`) from that same cavity geometry, not injected. Active
    group fractions MUST sum to one within `area_frac_sum_tol` (fail-closed: a sum of 0.5 undercounts
    partitions and is rejected). Install allowances are added at TREATED-area level by `total_installed_mass`."""
    D = _require_positive("D", D)
    face_t = _require_positive("face_t", face_t, allow_zero=True)
    wall_t = _require_positive("wall_t", wall_t, allow_zero=True)
    if not (math.isfinite(area_frac_sum_tol) and 0.0 <= area_frac_sum_tol < 1.0):
        raise MassModelError(f"area_frac_sum_tol must be finite in [0,1); got {area_frac_sum_tol}")
    bom = bom or InstallBOM()
    m = rho_wall * face_t + bom.backing_areal_mass(rho_wall)          # facesheet (+ backing) over full wall
    total_af = 0.0
    for g in groups:
        af = float(g["area_frac"])
        if not (0.0 <= af <= 1.0 + 1e-9):
            raise MassModelError(f"area_frac must be in [0,1]; got {af}")
        if af <= 1e-6:
            continue                                                 # inactive bank contributes no partitions
        total_af += af
        n_local = local_cell_density(g["cavity_volume"], D)          # cells per m^2 WITHIN this group's patch
        m += af * partition_areal_mass(n_local, wall_t, D, rho_wall, topology)
    if abs(total_af - 1.0) > area_frac_sum_tol:
        raise MassModelError(
            f"active group area fractions sum to {total_af:.4f} (must be 1 +/- {area_frac_sum_tol})")
    return m


def treated_area(f_cov: float, A_wall: float) -> float:
    if not (-1e-9 <= float(f_cov) <= 1.0 + 1e-9):
        raise MassModelError(f"f_cov must be in [0,1]; got {f_cov}")
    A_wall = _require_positive("A_wall", A_wall)
    return min(1.0, max(0.0, float(f_cov))) * A_wall


def total_installed_mass(liner_areal: float, f_cov: float, A_wall: float, bom: InstallBOM | None = None,
                         m_fixed: float = 0.0) -> dict:
    """Total installed liner mass [kg] with a NAMED per-term breakdown (review round 4.1-f). `f_cov=0` and
    `m_fixed=0` -> 0 kg. Returns the aggregate `m_install` AND `install_breakdown` (each named allowance in kg)
    so the BOM stays auditable and later mass-sensitivity analysis can vary individual terms."""
    bom = bom or InstallBOM()
    m_fixed = _require_positive("m_fixed", m_fixed, allow_zero=True)
    liner_areal = _require_positive("liner_areal", liner_areal, allow_zero=True)
    A_t = treated_area(f_cov, A_wall)
    m_liner = liner_areal * A_t
    install_breakdown = dict(adhesive=bom.adhesive_areal * A_t, fastener=bom.fastener_areal * A_t,
                             frame_edge=bom.frame_edge_areal * A_t, manufacturing_margin=bom.manufacturing_margin * A_t)
    m_install = sum(install_breakdown.values())
    return dict(m_total=m_liner + m_install + m_fixed, m_liner=m_liner, m_install=m_install,
                install_breakdown=install_breakdown, m_fixed=m_fixed, A_treated=A_t,
                install_areal=bom.install_areal_mass(), topology=DEFAULT_TOPOLOGY)
