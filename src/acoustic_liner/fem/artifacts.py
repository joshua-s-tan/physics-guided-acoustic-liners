"""The scenario-FEM release artifact: the evidence wrapped with its provenance (the only production path).
"""
from __future__ import annotations



def write_fem_artifact(out_path, evidence: dict, config_path, finalists_path, status: str = "complete") -> dict:
    """Write a scenario-FEM release artifact as a {provenance, data} wrapper. Provenance is produced through
    `provenance.capture` (role->path source files + config), so the PRODUCTION path yields exactly what the
    verifier requires (in a bare/provisional checkout capture demotes status to incomplete -- by design).

    The `fem` policy requires BOTH input roles (`physical_scenario` and `finalists`); a FEM artifact is only
    complete when it is bound to the finalist set it evaluated, so `finalists_path` is mandatory here."""
    from ..provenance import records as PROV
    from .. import paths
    src = paths.ROOT / "src" / "acoustic_liner" / "fem"         # the public files that fill the three source roles
    source_roles = {"fem_solver.py": src / "solver.py", "scenarios.py": src / "scenarios.py",
                    "fem_boundary.py": src / "boundary.py"}
    prov = PROV.capture(project_version=evidence.get("scenario_sha256", "")[:12] or "fem", policy="fem",
                     config_path=str(config_path), source_files=source_roles,
                     input_artifacts={"physical_scenario": config_path, "finalists": finalists_path},
                     status=status)
    PROV.write_artifact(out_path, evidence, prov)
    return prov
