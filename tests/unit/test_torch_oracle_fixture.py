"""The fixture of the differentiable-oracle tests (`tests/fixtures/torch_oracle_rows.json`) is what its generator
says: twelve synthetic rows made from a fixed seed by public code, labelled by the NumPy oracle. It is test-only and
is not part of the scientific evaluation evidence.
"""
import ast
import importlib.util
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE = FIXTURES / "torch_oracle_rows.json"
GENERATOR = FIXTURES / "make_torch_oracle_rows.py"
DOC = json.loads(FIXTURE.read_text(encoding="utf-8"))
INPUT_TOLERANCE = 1e-12          # regenerated inputs against the stored ones, relative
LABEL_TOLERANCE = 1e-9           # NumPy oracle against the stored labels, relative (the Torch tests allow 1e-6)


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("make_torch_oracle_rows", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def numbers_only(x):
    if isinstance(x, dict):
        return all(numbers_only(v) for v in x.values())
    if isinstance(x, list):
        return all(numbers_only(v) for v in x)
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def close(a, b, tol):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_the_fixture_declares_what_it_is(gen):
    p = DOC["provenance"]
    assert DOC["_schema"] == gen.SCHEMA == "acoustic-liner/test-fixture-torch-oracle-rows/2"
    assert p["use"] == "test-only; synthetic; not part of the scientific evaluation evidence"
    assert p["generator"] == "tests/fixtures/make_torch_oracle_rows.py" and p["seed"] == gen.SEED == 20261012
    assert (p["conditions"], p["designs_per_condition"], p["rows"]) == (4, 3, 12) and p["fields"] == gen.FIELDS
    assert "source" not in p and "source_sha256" not in p          # no row comes from a stored dataset


def test_the_rows_are_twelve_rows_of_numbers_with_the_six_fields():
    rows = DOC["rows"]
    assert len(rows) == 12 and numbers_only(rows)
    for r in rows:
        assert sorted(r) == ["A_wall", "dOASPL_worst_dB", "decoder_vector", "kappa_nominal", "spectrum",
                             "total_installed_mass_kg"]
        assert list(r["spectrum"]) == ["spl_dB"] and len(r["spectrum"]["spl_dB"]) == 8
        assert len(r["decoder_vector"]) == 19
    assert len({tuple(r["decoder_vector"]) for r in rows}) == 12
    assert len({r["A_wall"] for r in rows}) == 4 and [r["decoder_vector"][-1] for r in rows[2::3]] == [1.0] * 4


def test_the_inputs_are_the_ones_the_seed_gives(gen):
    rows, draws = gen.inputs()
    assert draws == DOC["provenance"]["designs_drawn"] and len(rows) == 12
    for new, old in zip(rows, DOC["rows"]):
        for a, b in zip(new["decoder_vector"] + new["spectrum"]["spl_dB"] + [new["A_wall"], new["kappa_nominal"]],
                        old["decoder_vector"] + old["spectrum"]["spl_dB"] + [old["A_wall"], old["kappa_nominal"]]):
            assert close(a, b, INPUT_TOLERANCE)


def test_every_design_is_feasible_at_its_coverage():
    from acoustic_liner.design import constraints as FEAS, schema as DS
    for r in DOC["rows"]:
        design, f_cov = DS.design_from_decoder_vector(r["decoder_vector"])
        assert FEAS.is_feasible(design, f_cov)[0] and 0.3 <= f_cov <= 1.0


def test_the_labels_are_the_numpy_oracle_on_the_stored_inputs(gen):
    for r in DOC["rows"]:
        d_oaspl, mass = gen.labels(r)
        assert close(d_oaspl, r["dOASPL_worst_dB"], LABEL_TOLERANCE)
        assert close(mass, r["total_installed_mass_kg"], LABEL_TOLERANCE)


def test_the_generator_does_not_use_the_implementation_under_test():
    """The expected values are independent of the Torch oracle: the generator imports neither Torch nor the
    differentiable module."""
    names = set()
    for n in ast.walk(ast.parse(GENERATOR.read_text(encoding="utf-8"))):
        if isinstance(n, ast.Import):
            names |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            names |= {n.module or ""} | {f"{n.module}.{a.name}" for a in n.names}
    assert not [x for x in names if x.split(".")[0] == "torch" or "differentiable" in x]
    assert "acoustic_liner.physics.oracle" in names
