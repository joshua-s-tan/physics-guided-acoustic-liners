"""Inputs of the CFD-corrected FEM: the manifest-bound per-bank (u_amp, R, X) tables and u_target = 0.5
anchors from validated rows only, and the excess-inertance wall admittance for the FEM boundary.
"""
from __future__ import annotations

import json, os
import numpy as np
from ..physics import constants
from . import feedback as FB
from . import postprocess as POST
from . import validation as VAL
from . import case_generation as CG


ANCHOR_U_TARGET = 0.5           # the low-amplitude anchor case (historical nonlinear-CFD protocol sec.1)


class ProvenanceError(RuntimeError):
    """A CFD case failed to bind to the committed manifest (hash mismatch / missing case). Fail closed."""


def build_per_bank_rx(manifest, cases_base_dir, n_banks, *, n_periods=3):
    """Bind every case to the COMMITTED manifest before use (fail closed): recompute case_hash from the case's
    own _case_meta.json AND recompute the input-tree hash over the OpenFOAM dictionaries, and require BOTH to
    equal the manifest's pinned values; require bank_id + geometry_hash to match. Only manifest-bound, VALID rows
    enter the per-bank (u_amp, R_phasor, X_phasor) tables. The per-bank ANCHOR is the manifest's u_target==0.5
    case's measured (u_amp, R, X) (never an interp at a fictitious 0.5). Returns (per_bank_RX, per_bank_anchor,
    reports)."""
    geometry_hash = manifest["geometry_hash"]
    per_bank = [[] for _ in range(n_banks)]
    anchors = [None] * n_banks
    reports = []
    for entry in manifest["cases"]:
        cd = os.path.join(cases_base_dir, entry["name"])
        meta_path = os.path.join(cd, "_case_meta.json")
        if not os.path.isfile(meta_path):
            raise ProvenanceError(f"{entry['name']}: _case_meta.json missing under {cases_base_dir}")
        meta = json.load(open(meta_path))
        # 1) recompute case_hash from metadata; 2) recompute input-tree hash over the dictionaries; pin BOTH
        if CG.recompute_case_hash(meta) != entry["case_hash"]:
            raise ProvenanceError(f"{entry['name']}: case_hash mismatch vs manifest (metadata edited)")
        if CG.input_tree_hash(cd) != entry["input_tree_hash"]:
            raise ProvenanceError(f"{entry['name']}: input_tree_hash mismatch vs manifest (case tree edited)")
        if meta.get("bank_id") != entry["bank_id"] or meta.get("geometry_hash") != geometry_hash:
            raise ProvenanceError(f"{entry['name']}: bank_id/geometry_hash mismatch vs manifest")
        row = POST.post_case(cd, n_periods=n_periods)
        ok, rep = VAL.validate_cfd_row(row, expected_bank_id=entry["bank_id"], expected_case_hash=entry["case_hash"],
                                       expected_geometry_hash=geometry_hash)
        rep["case"] = entry["name"]; reports.append(rep)
        if not ok:
            continue
        bid = str(entry["bank_id"])
        j = int(bid.split("_b")[-1] if "_b" in bid else bid.lstrip("b"))     # L2 `L2_b0` and generic `b0` ids
        rx = (float(row["u_amp"]), float(row["R_phasor"]), float(row["X_phasor"]))
        per_bank[j].append(rx)
        if abs(float(entry["u_target"]) - ANCHOR_U_TARGET) < 1e-9:      # the validated low-amplitude anchor
            anchors[j] = rx
    return [(t if t else None) for t in per_bank], anchors, reports


def _nl_beta_fn(groups, D, dR, dM):
    """Callable f -> normalized beta = Z_air / z_surface_NL(f) with the per-bank excess-inertance correction."""
    def beta(f):
        z_s = complex(FB._z_surface_NL(np.array([float(f)], float), groups, D, dR, dM)[0])
        return constants.Z_AIR / z_s
    return beta
