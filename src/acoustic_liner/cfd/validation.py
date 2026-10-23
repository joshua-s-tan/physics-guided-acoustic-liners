"""The CFD-row admissibility predicate: a row enters the per-bank impedance tables only if every gate
passes; each gate records its measured value and the failed gates (fail-closed).
"""
from __future__ import annotations

import numpy as np


# Pre-registered tolerances (fixed before consuming any CFD output).
DRIFT_MAX = 0.05            # R and X inter-window convergence drift (relative)
VEL_TOL = 0.5              # |u_amp - u_target| / u_target
R_XCHECK_TOL = 0.15        # relative |R_eff - R_phasor| / |R_phasor|  (energy vs fundamental-phasor resistance)
X_XCHECK_TOL = 0.15        # relative |X_phasor - X_accel_xcheck| / |X_phasor|
THD_MAX = 0.15             # pressure/velocity total harmonic distortion (flagged as limitation above this)
_REQUIRED = ("bank_id", "case_hash", "geometry_hash", "u_target", "u_amp", "mesh_ok",
             "R_phasor", "X_phasor", "R_eff", "X_accel_xcheck",
             "R_drift", "X_drift", "thd_p", "thd_u", "probe_clean")


def _rel(a, b):
    d = abs(float(b))
    return abs(float(a) - float(b)) / d if d > 0 else (0.0 if float(a) == 0.0 else np.inf)


def validate_cfd_row(row, *, expected_bank_id=None, expected_case_hash=None, expected_geometry_hash=None):
    """Return (ok: bool, report: dict). `row` must carry every _REQUIRED field. Optional expected_* pin the
    row to a specific requested bank/case/geometry (exact identity)."""
    checks = {}
    missing = [k for k in _REQUIRED if k not in row or row[k] is None]
    if missing:
        return False, dict(ok=False, missing=missing)

    # identity
    checks["bank_identity"] = (expected_bank_id is None or row["bank_id"] == expected_bank_id)
    checks["case_hash"] = (expected_case_hash is None or row["case_hash"] == expected_case_hash)
    checks["geometry_hash"] = (expected_geometry_hash is None or row["geometry_hash"] == expected_geometry_hash)

    # measured-velocity agreement + positivity
    u_t, u_a = float(row["u_target"]), float(row["u_amp"])
    checks["u_amp_positive"] = u_a > 0.0
    checks["u_amp_agreement"] = (u_t > 0.0 and _rel(u_a, u_t) < VEL_TOL)

    # mesh + probe hygiene + finiteness
    checks["mesh_ok"] = bool(row["mesh_ok"])
    checks["probe_clean"] = bool(row["probe_clean"])
    checks["finite"] = all(np.isfinite(float(row[k])) for k in ("R_phasor", "X_phasor", "R_eff", "X_accel_xcheck"))
    # passivity: a physical neck resistance is non-negative (both the fundamental-phasor and energy measures)
    checks["R_phasor_nonneg"] = float(row["R_phasor"]) >= 0.0
    checks["R_eff_nonneg"] = float(row["R_eff"]) >= 0.0

    # R and X convergence between independent cycle windows
    checks["R_converged"] = float(row["R_drift"]) < DRIFT_MAX
    checks["X_converged"] = float(row["X_drift"]) < DRIFT_MAX

    # cross-checks: R_eff (energy) vs R_phasor ; X_phasor vs X_accel cross-check
    checks["R_eff_vs_phasor"] = _rel(row["R_eff"], row["R_phasor"]) < R_XCHECK_TOL
    checks["X_phasor_vs_xcheck"] = _rel(row["X_phasor"], row["X_accel_xcheck"]) < X_XCHECK_TOL

    # THD within acceptance (recorded either way; above threshold is a hard fail, not a silent accept)
    checks["thd_p"] = float(row["thd_p"]) <= THD_MAX
    checks["thd_u"] = float(row["thd_u"]) <= THD_MAX

    ok = all(checks.values())
    report = dict(ok=bool(ok), bank_id=row["bank_id"], u_target=u_t, u_amp=u_a,
                  R_phasor=float(row["R_phasor"]), X_phasor=float(row["X_phasor"]),
                  R_eff=float(row["R_eff"]), R_drift=float(row["R_drift"]), X_drift=float(row["X_drift"]),
                  thd_p=float(row["thd_p"]), thd_u=float(row["thd_u"]),
                  checks=checks, failed=[k for k, v in checks.items() if not v])
    return ok, report
