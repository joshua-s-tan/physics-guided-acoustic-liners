"""Generator of `torch_oracle_rows.json`, the fixture of the differentiable-oracle tests (not a test itself).

    python tests/fixtures/make_torch_oracle_rows.py            compare the committed fixture with its regeneration
    python tests/fixtures/make_torch_oracle_rows.py --write    (maintainer) write the fixture

The fixture is test-only. It is synthetic, it is made by public code alone, and it is not part of the scientific
evaluation evidence: no result in the README or under `evidence/` is computed from it.

Inputs (twelve rows, from the fixed seed SEED):
  * four synthetic conditions (spectrum, wall area, kappa): `evaluation.conditions.synthetic_conditions(4, SEED,
    GROUP)`;
  * three designs per condition, drawn in order from `numpy.random.default_rng(SEED)`: a uniform sample of the design
    box (`design.schema.sample_random`) and one more uniform number for the coverage (COVERAGE_RANGE), kept when
    `design.constraints.is_feasible` accepts the design at that coverage. The third design of every condition is
    placed at full coverage (1.0), the upper edge of the coverage range;
  * a row stores the decoder vector of the design (`design.schema.decoder_vector`), the spectrum, the wall area and
    kappa.

Expected values: the two labels of every row are computed by the NumPy oracle, `physics.oracle.label_v2`, from the
row's own stored inputs. The Torch implementation under test (`physics.differentiable.TorchOracle`) is not used and
Torch is not imported here, so the expected values are independent of the code the tests check.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "torch_oracle_rows.json"
SCHEMA = "acoustic-liner/test-fixture-torch-oracle-rows/2"
SEED = 20261012
GROUP = "torch-oracle-fixture"
N_CONDITIONS = 4
DESIGNS_PER_CONDITION = 3
COVERAGE_RANGE = (0.3, 1.0)
MAX_DRAWS = 100000
FIELDS = ["decoder_vector", "spectrum.spl_dB", "A_wall", "kappa_nominal", "dOASPL_worst_dB", "total_installed_mass_kg"]


def _package():
    try:
        import acoustic_liner  # noqa: F401
    except ImportError:                                   # run from a checkout that is not installed
        sys.path.insert(0, str(HERE.parents[1] / "src"))
    from acoustic_liner.design import constraints, schema
    from acoustic_liner.evaluation import conditions
    from acoustic_liner.physics import oracle
    return constraints, schema, conditions, oracle


def inputs():
    """The twelve rows without their labels, and the number of designs drawn to find them."""
    FEAS, DS, COND, _ = _package()
    rng = np.random.default_rng(SEED)
    rows, draws = [], 0
    for c in COND.synthetic_conditions(N_CONDITIONS, SEED, GROUP):
        kept = 0
        while kept < DESIGNS_PER_CONDITION:
            draws += 1
            if draws > MAX_DRAWS:
                raise RuntimeError("no feasible design found")
            design = DS.vector_to_design(DS.sample_random(rng))
            f_cov = COVERAGE_RANGE[0] + (COVERAGE_RANGE[1] - COVERAGE_RANGE[0]) * float(rng.random())
            if kept == DESIGNS_PER_CONDITION - 1:
                f_cov = 1.0
            if not FEAS.is_feasible(design, f_cov)[0]:
                continue
            kept += 1
            rows.append(dict(decoder_vector=DS.decoder_vector(design, f_cov),
                             spectrum=dict(spl_dB=[float(x) for x in c["spl"]]),
                             A_wall=float(c["A_wall"]), kappa_nominal=float(c["kappa"])))
    return rows, draws


def labels(row):
    """(dOASPL_worst_dB, total_installed_mass_kg) of one row, by the NumPy oracle from the row's stored inputs."""
    _, DS, _, O2 = _package()
    design, f_cov = DS.design_from_decoder_vector(row["decoder_vector"])
    lab = O2.label_v2(dict(design, f_cov=f_cov, A_wall=row["A_wall"], spectrum=row["spectrum"]["spl_dB"],
                           kappa_nominal=row["kappa_nominal"]))
    return float(lab["dOASPL_worst_dB"]), float(lab["total_installed_mass_kg"])


def build():
    rows, draws = inputs()
    for r in rows:
        r["dOASPL_worst_dB"], r["total_installed_mass_kg"] = labels(r)
    return dict(
        _schema=SCHEMA,
        provenance=dict(
            what="Twelve labelled rows for the differentiable-oracle tests (tests/unit/test_v27_oracle_torch.py).",
            use="test-only; synthetic; not part of the scientific evaluation evidence",
            generator="tests/fixtures/make_torch_oracle_rows.py", seed=SEED, condition_group=GROUP,
            conditions=N_CONDITIONS, designs_per_condition=DESIGNS_PER_CONDITION, coverage_range=list(COVERAGE_RANGE),
            designs_drawn=draws, rows=len(rows), fields=FIELDS,
            inputs="synthetic conditions and uniform samples of the design box, from the seed, by public code "
                   "(the module docstring of the generator gives the exact method)",
            labels="dOASPL_worst_dB and total_installed_mass_kg are computed by the NumPy oracle "
                   "(acoustic_liner.physics.oracle.label_v2) from the other four fields of the same row; the Torch "
                   "oracle is not used"),
        rows=rows)


def text(doc):
    return json.dumps(doc, indent=1) + "\n"


if __name__ == "__main__":
    new = text(build())
    if "--write" in sys.argv[1:]:
        FIXTURE.write_text(new, encoding="utf-8", newline="\n")
    same = FIXTURE.read_bytes() == new.encode("utf-8")
    print(json.dumps(dict(fixture=FIXTURE.name, sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                          equals_regeneration=same)))
    sys.exit(0 if same else 1)
