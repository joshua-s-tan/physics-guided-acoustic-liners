# -*- coding: utf-8 -*-
"""OPTIONAL fixture-regeneration AUDIT (NOT part of the normal test suite).

This is the ONLY place the historical reference tree is imported. It regenerates the committed frozen fixture
`migration_fixtures.json` that the normal, self-contained equivalence tests read. Run manually:

    ACOUSTIC_LINER_V1_REFERENCE_TREE=<historical v1 tree> \
        python tests/fixtures/regenerate_fixtures.py

The committed fixture carries, for reproducibility: reference-source SHA-256 hashes, the exact inputs/seeds/units,
the expected outputs computed FROM THE REFERENCE implementation, the generation command, dependency versions, and
the tolerance policy the tests must apply. GitHub CI never runs this; it reads the committed JSON.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "migration_fixtures.json"

# Tolerance policy (applied by the normal tests). Acoustics were shown byte-identical, so acoustic tolerances are
# tight; masses are deterministic closed-form, so exact.
TOLERANCE = dict(
    acoustics_abs=1e-9, acoustics_rel=1e-9,
    spectra_abs=1e-12,
    legacy_mass_rel=1e-9,
    note="acoustic outputs are expected byte-identical (migration preserves the math); tolerances guard only "
         "float formatting/interp. Masses are closed-form deterministic.")

# Canonical fixture designs (release-feasible band; deterministic). NOT scientific finalists.
DESIGNS = {
    "four_group_feasible": dict(
        D=0.05, face_t=0.0008, wall_t=0.0006, groups=[
            dict(f0=126.0, r_neck=0.012, L_neck=0.03, area_frac=0.25, Q=15.0),
            dict(f0=160.0, r_neck=0.010, L_neck=0.03, area_frac=0.25, Q=15.0),
            dict(f0=196.0, r_neck=0.008, L_neck=0.03, area_frac=0.25, Q=15.0),
            dict(f0=251.0, r_neck=0.006, L_neck=0.03, area_frac=0.25, Q=15.0)]),
    "two_group_feasible": dict(
        D=0.05, face_t=0.0008, wall_t=0.0006, groups=[
            dict(f0=196.0, r_neck=0.008, L_neck=0.03, area_frac=0.5, Q=15.0),
            dict(f0=251.0, r_neck=0.006, L_neck=0.03, area_frac=0.5, Q=15.0)]),
}
COVERAGES = [0.0, 0.6, 1.0]
A_WALL = 165.878                         # authoritative Falcon scenario A_wall (fixture input; not sourced here)
KAPPA_NOMINAL = 0.23270570707355803
Z_FREQS = [63.0, 125.0, 200.0, 250.0]    # z_surface probe frequencies
SPECTRA_SEEDS = [0, 1]
SPECTRA_ROWS = 6                         # first-K sample_family rows to freeze


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ref = os.environ.get("ACOUSTIC_LINER_V1_REFERENCE_TREE", "")
    if not ref:
        print("ACOUSTIC_LINER_V1_REFERENCE_TREE unset -- this AUDIT needs the reference tree.", file=sys.stderr)
        return 2
    ref = Path(ref)
    sys.path.insert(0, str(ref / "fairing_case" / "v2"))
    import oracle_v2 as O2R           # reference oracle (+ RM/OR/MM via its own bootstrap)
    import spectra as SPR
    import mass_model as MMR          # v1 (legacy) total-mass model

    src_hashes = {
        "resonator_model.py": _sha256(ref / "resonator_model.py"),
        "fairing_oracle.py": _sha256(ref / "fairing_case" / "fairing_oracle.py"),
        "oracle_v2.py": _sha256(ref / "fairing_case" / "v2" / "oracle_v2.py"),
        "spectra.py": _sha256(ref / "fairing_case" / "v2" / "spectra.py"),
        "launch_environments.py": _sha256(ref / "fairing_case" / "launch_environments.py"),
        "mass_model_v1.py": _sha256(ref / "fairing_case" / "v2" / "mass_model.py"),
    }
    spec = O2R.falcon9_design_band().tolist()

    cases = []
    for dname, design in DESIGNS.items():
        a_band = O2R.liner_diffuse_alpha_band(design["groups"], design["D"]).tolist()
        zf = np.array(Z_FREQS, dtype=float)
        zs = O2R.z_surface(zf, design["groups"], design["D"])
        for cov in COVERAGES:
            d = dict(design, f_cov=cov, A_wall=A_WALL, spectrum=np.array(spec),
                     kappa_nominal=KAPPA_NOMINAL)
            lab = O2R.label_v2(dict(d))
            cases.append(dict(
                design_name=dname, coverage=cov,
                expected_diffuse_alpha_band=a_band,
                expected_z_surface=[dict(f=float(f), real=float(z.real), imag=float(z.imag))
                                    for f, z in zip(Z_FREQS, zs)],
                expected_dOASPL_worst_dB=float(lab["dOASPL_worst_dB"]),
                expected_dOASPL_nominal_dB=float(lab["dOASPL_nominal_dB"]),
                expected_robust_ngrid=int(lab["robust_ngrid"]),
                reference_areal_mass_kg_m2=float(lab["areal_mass_kg_m2"]),   # v1 4*sqrt(n) areal
                reference_total_mass_kg=float(lab["total_installed_mass_kg"]),  # v1 total (mu_A install)
            ))

    spectra_fix = {}
    for seed in SPECTRA_SEEDS:
        rows = SPR.sample_family(SPECTRA_ROWS, seed=seed)
        spectra_fix[str(seed)] = [dict(spl_dB=r["spl_dB"], weights=r["weights"]) for r in rows]

    fixture = dict(
        _schema="casef-migration-fixtures/1",      # the frozen schema identifier of the committed fixture
        provenance=dict(
            reference_source_sha256=src_hashes,
            generated_by="tests/fixtures/regenerate_fixtures.py",
            command="ACOUSTIC_LINER_V1_REFERENCE_TREE=<ref> python tests/fixtures/regenerate_fixtures.py",
            dependency_versions=dict(python=platform.python_version(),
                                     numpy=np.__version__, scipy=__import__("scipy").__version__),
            units=dict(f0="Hz", r_neck="m", L_neck="m", D="m", face_t="m", wall_t="m",
                       A_wall="m^2", z_surface="Pa*s/m", mass="kg", areal_mass="kg/m^2",
                       dOASPL="dB", alpha="dimensionless"),
            reference_oracle_version=getattr(O2R, "ORACLE_V2_VERSION", "unknown"),
            legacy_mass_model="v1 fairing_oracle.areal_mass (4*wall_t*D*sqrt(n_local)) + mass_model.total_mass "
                              "(m_areal*A_treated + m_fixed + mu_A*A_treated, mu_A=%.3f)" % MMR.MU_A_DEFAULT,
            legacy_mu_A=float(MMR.MU_A_DEFAULT),
            # the note stored in the committed fixture, kept word for word (it names the historical package)
            note="expected_* are computed FROM THE REFERENCE implementation; the normal suite compares the "
                 "migrated casef_v2 modules against these frozen values WITHOUT importing the reference tree.",
        ),
        tolerance=TOLERANCE,
        inputs=dict(designs=DESIGNS, coverages=COVERAGES, A_wall=A_WALL, kappa_nominal=KAPPA_NOMINAL,
                    z_freqs=Z_FREQS, spectrum_falcon9_design_band=spec,
                    spectra_seeds=SPECTRA_SEEDS, spectra_rows=SPECTRA_ROWS),
        oracle_cases=cases,
        spectra_family=spectra_fix,
    )
    OUT.write_text(json.dumps(fixture, indent=2, sort_keys=True))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); reference oracle {fixture['provenance']['reference_oracle_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
