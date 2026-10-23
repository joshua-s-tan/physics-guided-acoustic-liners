"""tests/optional holds explicitly invoked integration tests that need resources outside this repository (SK-1:
the historical v1 tree). Directory collection skips them unless --v1-reference is given; naming a file runs it."""
from pathlib import Path

OPTIONAL = Path(__file__).resolve().parent / "optional"


def pytest_addoption(parser):
    parser.addoption("--v1-reference", action="store_true", default=False,
                     help="collect tests/optional (needs the historical v1 tree in ACOUSTIC_LINER_V1_REFERENCE_TREE)")


def pytest_ignore_collect(collection_path, config):
    p = Path(collection_path).resolve()
    if (p == OPTIONAL or OPTIONAL in p.parents) and not config.getoption("--v1-reference"):
        return True
    return None

