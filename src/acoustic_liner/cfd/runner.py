"""Status and resume handling over a case plan, without running the solver: per-case classification
(VALID, NEEDS_EXTEND, FAILED, MISSING), the resume action manifest, and the aggregation of
manifest-bound valid rows into per-bank impedance tables.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from . import postprocess as POST
from . import validation as VAL
from . import impedance_correction as RL


VALID, NEEDS_EXTEND, FAILED, MISSING = "valid", "needs_extend", "failed", "missing"


# a case not-yet-run is marked by the absence of the solver's function-object output tree.
_RUN_MARKER = "postProcessing"
REQUIRE_N_BANKS = 4            # the current release requires complete four-bank CFD evidence (prereg / item 6)


class RunnerError(RuntimeError):
    """A batch-runner stage could not proceed (e.g. not all banks have a valid converged R/X table). Fails
    closed -- the runner never fabricates or holds a missing/invalid case."""


def load_manifest(outbase, plan_filename="_cfd_case_plan.json"):
    path = os.path.join(outbase, plan_filename)
    if not os.path.isfile(path):
        raise RunnerError(f"manifest {path!r} not found -- run prepare() first")
    return json.load(open(path))


# ---- stage 2: status (convergence + extend detection) -----------------------------------------------------
def _case_ran(case_dir):
    """A case has solver output iff its OpenFOAM function-object tree exists (fail-closed: absent => not run)."""
    return os.path.isdir(os.path.join(case_dir, _RUN_MARKER))


def _classify(row, ok, report):
    """Map a post_case row + validate report to VALID / NEEDS_EXTEND / FAILED. NEEDS_EXTEND is reserved for a
    case whose ONLY failing gates are the inter-window convergence drifts (R/X) while probe + mesh are clean --
    i.e. physically sound but under-run, fixable by more cycles. Any other failure is a hard FAILED."""
    if ok:
        return VALID
    failed = set(report.get("failed", []))
    convergence_only = bool(failed) and failed <= {"R_converged", "X_converged"}
    if convergence_only and bool(row.get("probe_clean")) and bool(row.get("mesh_ok")):
        return NEEDS_EXTEND
    return FAILED


def scan_status(manifest, cases_base_dir, *, n_periods=3):
    """Classify every manifest case against what the human has produced on disk. Does NOT mutate any case.
    Returns dict(cases=[{name,bank_id,state,report?}], summary={state:count}, n_banks, all_valid)."""
    geometry_hash = manifest["geometry_hash"]
    per_case = []
    for entry in manifest["cases"]:
        cd = os.path.join(cases_base_dir, entry["name"])
        rec = dict(name=entry["name"], bank_id=entry["bank_id"], u_target=entry["u_target"])
        if not os.path.isdir(cd) or not _case_ran(cd):
            rec["state"] = MISSING
            per_case.append(rec)
            continue
        row = POST.post_case(cd, n_periods=n_periods)
        ok, rep = VAL.validate_cfd_row(row, expected_bank_id=entry["bank_id"],
                                       expected_case_hash=entry["case_hash"],
                                       expected_geometry_hash=geometry_hash)
        rec["state"] = _classify(row, ok, rep)
        rec["failed"] = rep.get("failed", [])
        per_case.append(rec)
    summary = dict(Counter(c["state"] for c in per_case))
    n_banks = len({c["bank_id"] for c in manifest["cases"]})
    return dict(cases=per_case, summary=summary, n_banks=n_banks,
                all_valid=all(c["state"] == VALID for c in per_case))


def resume_manifest(status, cases_base_dir, *, extend_cycles=None):
    """From a scan_status result, build an ACTION MANIFEST of what remains. Fail-closed semantics:
      * `done` is TRUE iff EVERY case is VALID (status['all_valid']) -- NOT merely that nothing is runnable; a
        batch of only hard-failed cases is done=False with those cases surfaced under `blocked`.
      * MISSING cases get an EXECUTABLE WSL command (fresh blockMesh/checkMesh/solver) in `shell`.
      * NEEDS_EXTEND cases get a NON-executable directive in `manual` (extension semantics are case-specific and
        human-run; the runner never silently rehashes a committed case tree) -- it is flagged executable=False
        so no caller mistakes a note for a command.
      * FAILED cases are blocked evidence, not work: listed in `blocked`, never queued.
    Returns dict(actions=[{name,action,dir,executable,cmd|note}], shell=[str], manual=[str], blocked=[str], done)."""
    actions, blocked = [], []
    for c in status["cases"]:
        cd = os.path.join(cases_base_dir, c["name"])
        if c["state"] == MISSING:
            cmd = f"cd {cd} && blockMesh > log.blockMesh && checkMesh > log.checkMesh && pimpleFoam > log.pimpleFoam 2>&1"
            actions.append(dict(name=c["name"], action="run", dir=cd, executable=True, cmd=cmd))
        elif c["state"] == NEEDS_EXTEND:
            note = f"extend endTime (more cycles) and restart {cd}" + (f"; suggest cycles>={extend_cycles}" if extend_cycles else "")
            actions.append(dict(name=c["name"], action="extend", dir=cd, executable=False, note=note))
        elif c["state"] == FAILED:
            blocked.append(c["name"])
    shell = [a["cmd"] for a in actions if a.get("executable")]
    manual = [a["note"] for a in actions if not a.get("executable")]
    return dict(actions=actions, shell=shell, manual=manual, blocked=blocked, done=bool(status.get("all_valid")))


# ---- stage 3: aggregate (provenance-bound per-bank R/X) ----------------------------------------------------
def _manifest_bank_indices(manifest):
    """Derive the DISTINCT bank indices declared by the manifest (fail-closed on an unparseable id). Accepts
    both generic (`b0`) and L2 (`L2_b0`) bank ids, matching cfd.impedance_correction.build_per_bank_rx's own parsing."""
    idx = set()
    for e in manifest["cases"]:
        bid = str(e["bank_id"])
        tail = bid.split("_b")[-1] if "_b" in bid else bid.lstrip("b")
        try:
            idx.add(int(tail))
        except ValueError:
            raise RunnerError(f"unparseable bank_id {bid!r} in manifest")
    return idx


def aggregate(manifest, cases_base_dir, *, n_periods=3, require_banks=REQUIRE_N_BANKS):
    """Bind every case to the committed manifest (RL.build_per_bank_rx fails closed on any hash mismatch) and
    build the per-bank (u_amp,R,X) tables + per-bank u_target=0.5 anchors from ONLY valid rows. The bank count
    and identities are DERIVED FROM THE MANIFEST, never trusted from a caller: with require_banks set (default
    4), the manifest must declare EXACTLY that many distinct banks, identified exactly {0..require_banks-1}.
    Raises RunnerError on the wrong bank set, or if any bank has no valid row or no u_target=0.5 anchor --
    complete four-bank evidence is mandatory before relabel."""
    banks = _manifest_bank_indices(manifest)
    if require_banks is not None and banks != set(range(require_banks)):
        raise RunnerError(f"manifest declares banks {sorted(banks)}; the current release requires exactly "
                          f"{require_banks} distinct banks identified {{0..{require_banks - 1}}}")
    n_banks = len(banks)
    per_bank_RX, anchors, reports = RL.build_per_bank_rx(manifest, cases_base_dir, n_banks, n_periods=n_periods)
    empty = [j for j, t in enumerate(per_bank_RX) if not t]
    no_anchor = [j for j, a in enumerate(anchors) if a is None]
    if empty or no_anchor:
        raise RunnerError(f"incomplete CFD evidence: banks with no valid row={empty}, "
                          f"banks with no u_target=0.5 anchor={no_anchor}")
    return dict(per_bank_RX=per_bank_RX, per_bank_anchor=anchors, reports=reports, n_banks=n_banks)
