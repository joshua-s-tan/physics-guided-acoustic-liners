"""Robust FEM evaluation of one design: the partial-coverage FEM at the four pre-registered corners
(kappa low/high x placement anchor low/high), each written as a provenance-bound artifact, and the
robust (worst-case) value with its controlling corner through the fail-closed selection gate.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
from ..physics import absorption as acoustics
from ..physics import constants
from ..provenance import hashing as HASH
from . import artifacts as FART
from . import solver as FS
from . import placement as RP


ALPHA_BARE_NOMINAL = 0.05                         # frozen bare-wall diffuse absorption assumption (§ blanket semantics)


# The FOUR pre-registered uncertainty x placement corners: uncertainty axis = kappa at the scenario's
# kappa_eval_range endpoints {lo, hi}; placement axis = coverage anchor {low, high}. Immutable and named.
KAPPA_LEVELS = ("kappa_lo", "kappa_hi")
ANCHORS = ("low", "high")
PREREGISTERED_CORNERS = tuple((kl, an) for kl in KAPPA_LEVELS for an in ANCHORS)   # 4 corners


class DryRunError(RuntimeError):
    """The thin skeleton hit an invalid wiring state (a real component rejected an input)."""


def _group_cavity_volume(oracle, g: dict) -> float:
    """Cavity volume of one resonator group, computed EXACTLY as the oracle does (reuse its frozen RM)."""
    S = oracle.RM.neck_area(g["r_neck"])
    L_eff = oracle.RM.effective_neck_length(g["L_neck"], g["r_neck"])
    return float(oracle.RM.cavity_volume_for_target_f0(g["f0"], S, L_eff))


def _wall_beta_fn(oracle, groups, D: float):
    """Candidate -> per-frequency NORMALIZED wall admittance beta(f) = Z_air / z_s(f), from the oracle's
    complex specific surface impedance z_s. Complex and passive for a passive liner (Re z_s >= 0)."""
    def beta(f: float) -> complex:
        z_s = complex(oracle.z_surface(np.array([float(f)], dtype=float), groups, D)[0])
        return constants.Z_AIR / z_s
    return beta


def evaluate_design_robust_fem(scenario, oracle, design, *, coverage, alpha_bare, out_dir, config_path,
                               finalists_path, scenario_sha256, corners=PREREGISTERED_CORNERS,
                               wall_beta_fn=None, mesh_size=None):
    """FEM robust (worst-case) evaluation of an ALREADY-feasible, cavity-enriched `design` over the pre-registered
    uncertainty x placement corners, with the fail-closed #10 gate, an independent cross-check, and scenario-hash
    threading. Returns (corner_records, robust). SHARED by the dry-run skeleton and the physical-finalist path so
    both use identical, audited FEM/robust mechanics.

    `wall_beta_fn` (optional): a callable f->normalized beta = Z_air/z_s(f) to OVERRIDE the linear surface
    impedance -- used by the nonlinear-CFD corrected-FEM rerun (`cfd.impedance_correction`) to feed the excess-inertance
    surface impedance into the SAME audited FEM/robust mechanics. Default None uses the linear z_surface.
    mesh_size (optional): FEM mesh size; None keeps the run_scenario_fem default."""
    band = list(scenario.band_hz)
    spl = list(scenario.spl_dB)
    beta_fn = wall_beta_fn if wall_beta_fn is not None else _wall_beta_fn(oracle, design["groups"], design["D"])
    bare_beta = constants.Z_AIR * acoustics.admittance_from_diffuse_alpha(alpha_bare, constants.Z_AIR)
    kappa_lo, kappa_hi = min(scenario.kappa_eval_range), max(scenario.kappa_eval_range)
    kappa_by_level = {"kappa_lo": kappa_lo, "kappa_hi": kappa_hi}

    mkw = {} if mesh_size is None else dict(mesh_size=float(mesh_size))
    bare_ms_cache: dict[str, dict] = {}

    def _bare_ms(klevel: str) -> dict:
        if klevel not in bare_ms_cache:
            ev = FS.run_scenario_fem(scenario, band, wall_beta=bare_beta, kappa=kappa_by_level[klevel], coverage=1.0, **mkw)
            bare_ms_cache[klevel] = ev["ms_by_freq"]
        return bare_ms_cache[klevel]

    corner_records = []
    for (klevel, anchor) in corners:
        kappa = kappa_by_level[klevel]
        treated = FS.run_scenario_fem(scenario, band, wall_beta=beta_fn, kappa=kappa, coverage=coverage,
                                      bare_beta=bare_beta, anchor=anchor, **mkw)
        bare_ms = _bare_ms(klevel)
        nr_band = [10.0 * math.log10(bare_ms[float(f)] / treated["ms_by_freq"][float(f)]) for f in band]
        d_oaspl = float(oracle.delta_oaspl(np.asarray(nr_band), np.asarray(spl, dtype=float)))
        if not math.isfinite(d_oaspl):
            raise DryRunError(f"corner ({klevel},{anchor}) produced a non-finite dOASPL")
        art_path = Path(out_dir) / f"corner_{klevel}_{anchor}.json"
        prov = FART.write_fem_artifact(art_path, treated, config_path=config_path,
                                     finalists_path=finalists_path, status="complete")
        corner_records.append(dict(
            corner_id=(klevel, anchor), value=d_oaspl, ok=True,
            kappa=kappa, anchor=anchor, coverage=coverage, nr_band=nr_band,
            fem_scenario_sha256=treated["scenario_sha256"],
            artifact_path=str(art_path), artifact_sha256=HASH.sha256_file(art_path),
            provenance_status=prov.get("status"),
        ))

    robust = RP.select_robust_minimum(corner_records, corners)
    by_id = {tuple(str(x) for x in c["corner_id"]): c for c in corner_records}
    chosen = by_id[tuple(robust["robust_corner_id"])]
    direct = float(oracle.delta_oaspl(np.asarray(chosen["nr_band"]), np.asarray(spl, dtype=float)))
    if abs(direct - robust["robust_value"]) > 1e-9:
        raise DryRunError(f"robust value {robust['robust_value']} != direct eval of its corner {direct}")
    for c in corner_records:
        if c["fem_scenario_sha256"] != scenario_sha256:
            raise DryRunError("a corner FEM artifact is not bound to the authoritative scenario hash")
    return corner_records, robust
