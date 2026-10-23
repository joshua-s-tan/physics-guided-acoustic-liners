"""The public verifier (`acoustic-liner verify public`) on the compact public evidence: it is independent of the code
that produced the results (standard library and NumPy), it recomputes every statistic from the per-cell and per-pair
values, it passes on the distributed files, and it fails on every edited input: a changed value, a stale manifest
hash, a superseded schema identifier, a retired manifest layout.

Only files distributed in this repository are read. The tamper tests run on a copy of them. No check needs an
undistributed file. All results are simulation-based.
"""
import ast
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from acoustic_liner import paths
from acoustic_liner.provenance import public_verifier as PV, schemas as SC

ROOT = paths.ROOT
POSTHOC_NAME = "posthoc-followups.json"


def dump(d):
    return json.dumps(d, indent=1) + "\n"


FORBIDDEN = ("torch", "scipy", "cma", "skfem", "matplotlib", "acoustic_liner.evaluation", "acoustic_liner.fem",
             "acoustic_liner.models", "acoustic_liner.physics", "acoustic_liner.optimization", "acoustic_liner.design",
             "acoustic_liner.vehicles", "acoustic_liner.cfd", "acoustic_liner.visualization")
# the headline lines of the released result, as the verifier recomputes them from the 1,500 cells and the 20 pairs
RESULT_LINES = [
    "PASS  1,500 cells = 100 conditions x 5 seeds x 3 budgets",
    "PASS  RECOMPUTED  D_minus_A at 128: mean +1.050 dB [0.937, 1.165] recomputed",
    "PASS  RECOMPUTED  D_minus_B at 128: mean +0.904 dB [0.811, 0.999] recomputed",
    "PASS  RECOMPUTED  C32_minus_B32 at 128: mean +1.424 dB [1.263, 1.586] recomputed",
    "PASS  RECOMPUTED  D_minus_A at 512: mean +0.743 dB [0.669, 0.819] recomputed",
    "PASS  RECOMPUTED  D_minus_B at 512: mean +0.458 dB [0.412, 0.505] recomputed",
    "PASS  RECOMPUTED  D_minus_A at 2048: mean +0.354 dB [0.313, 0.396] recomputed",
    "PASS  RECOMPUTED  D_minus_B at 2048: mean +0.208 dB [0.174, 0.245] recomputed",
    "PASS  RECOMPUTED  parameter count 1,141,010 from the architecture (summary and checkpoint manifest agree)",
    "PASS  RECOMPUTED  FEM audit: r 0.957, median |err| 0.374, bias +0.073, max 1.180 dB recomputed from 20 pairs",
    "PASS  CONSISTENT  every listed figure matches its SHA-256",
]
N_PUBLIC_CHECKS = 53                      # PASS lines of the public mode, without the optional checkpoint file


# ---- independence
def test_the_verifier_runs_in_a_fresh_interpreter_without_any_forbidden_module():
    """A fresh interpreter runs the complete public mode through the command; Torch, SciPy and the evaluation, FEM,
    model, physics, optimization, design, vehicle, CFD and visualization code of the package are never imported."""
    code = ("import sys\n"
            "from acoustic_liner import cli\n"
            "try:\n"
            "    code = cli.main(['verify', 'public'])\n"
            "except SystemExit as e:\n"
            "    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)\n"
            f"forbidden = {FORBIDDEN!r}\n"
            "bad = sorted(m for m in sys.modules if any(m == f or m.startswith(f + '.') for f in forbidden))\n"
            "print('LOADED ' + ','.join(bad))\n"
            "sys.exit(code)\n")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
    lines = r.stdout.splitlines()
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    assert lines[-1] == "LOADED " and lines[-2] == "public verification passed"
    assert not [x for x in lines if x.startswith("FAIL")] and set(RESULT_LINES) <= set(lines)


def test_the_verifier_source_imports_only_the_standard_library_numpy_and_the_schema_names():
    allowed = {"__future__", "argparse", "collections", "hashlib", "json", "pickle", "re", "sys", "zipfile", "numpy",
               "tempfile", "pathlib"}
    assert allowed - {"numpy"} <= set(sys.stdlib_module_names) | {"__future__"}
    for mod in (PV,):
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                assert {a.name.split(".")[0] for a in n.names} <= allowed, (mod.__name__, ast.dump(n))
            elif isinstance(n, ast.ImportFrom):
                if n.level == 0:
                    assert n.module.split(".")[0] in allowed, (mod.__name__, n.module)
                else:                                   # relative: only paths and the schema names
                    assert (n.level, n.module, sorted(a.name for a in n.names)) in (
                        (2, None, ["paths"]), (1, None, ["schemas"])), ast.dump(n)


def test_the_restated_seed_label_reproduces_the_registered_bootstrap_stream():
    """The verifier restates the frozen label; it must be the label the study's stream derives from."""
    from acoustic_liner.evaluation import splits as PCN
    assert PV.SEED_DOMAIN_LABEL == PCN.ROOT_LABEL
    assert PCN.STREAM_NAMES[PV.BOOTSTRAP_STREAM] == "bootstrap"
    a = PV.bootstrap_rng().integers(0, 100, size=1000)
    b = np.random.default_rng(PCN.streams()["bootstrap"]).integers(0, 100, size=1000)
    assert (a == b).all()


def cells_of(fe):
    cells = []
    for r in fe["cells"]["rows"]:
        c = {k: {} for k in "ABCD"} | {"proposals": {}}
        for k, v in zip(fe["cells"]["columns"], r):
            if "." in k:
                g, f = k.split(".")
                c[g][f] = v
            else:
                c[k] = v
        cells.append(c)
    return cells


def test_the_independent_statistics_equal_the_package_functions_on_all_cells():
    """Two implementations, one result: the verifier's NumPy code and the study's own functions agree exactly on the
    1,500 cells and the 20 audit pairs. (The verifier never calls the study's functions.)"""
    from acoustic_liner.evaluation import statistics as STAT
    from acoustic_liner.fem import transfer_audit as FA
    fe = json.loads((ROOT / "evidence" / "linergen-final-evaluation.json").read_text(encoding="utf-8"))
    cells = cells_of(fe)
    assert len(cells) == 1500
    for b in fe["budgets"]:
        for x, y, kx in (("D", "A", "best_val"), ("D", "B", "best_val"), ("C", "B", "initial32_best_val"),
                         ("D", "C", "best_val")):
            mine, theirs = PV.contrast(cells, b, x, y, kx=kx, ky=kx), STAT.contrast(cells, b, x, y, kx=kx, ky=kx)
            theirs["seed_means"] = {str(k): v for k, v in theirs["seed_means"].items()}
            assert mine == theirs
    fa = json.loads((ROOT / "evidence" / "fem-transfer-audit.json").read_text(encoding="utf-8"))
    assert PV.audit_summary(fa["rows"]) == FA.summarize(fa["rows"]) == fa["summary"]
    assert PV.TRANSFER_FAILURE_MEDIAN_DB == FA.TRANSFER_FAILURE_MEDIAN_DB


def test_a_contrast_is_the_paired_mean_with_the_condition_as_the_unit():
    """Hand-checkable case: two conditions, two seeds; the mean is the mean of the per-condition means."""
    cells = [dict(budget=8, cond_index=c, seed=s, A=dict(best_val=a), D=dict(best_val=d))
             for c, s, a, d in ((0, 0, 1.0, 2.0), (0, 1, 1.0, 4.0), (1, 0, 3.0, 3.5), (1, 1, 5.0, None))]
    r = PV.contrast(cells, 8, "D", "A")
    assert r["n_conditions"] == 2 and r["mean"] == pytest.approx((2.0 + 0.5) / 2)
    assert r["seed_means"] == {"0": pytest.approx(0.75), "1": pytest.approx(3.0)}        # a missing value is left out
    assert r["ci95"][0] <= r["mean"] <= r["ci95"][1] and r["min_seed_mean"] == pytest.approx(0.75)
    assert PV.contrast(cells, 8, "D", "A") == r                                          # the stream is re-derived


def test_the_parameter_count_formula_equals_the_model():
    from acoustic_liner.models import linergen as LG
    n = int(sum(p.numel() for p in LG.LinerGenerator().parameters()))
    assert PV.parameter_count(LG.CONFIG) == n == 1141010


# ---- the copy of the public files
@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A copy of the public files the verifier reads; the verifier is pointed at it."""
    for d in ("evidence", "release", "docs/figures"):          # only what the verifier reads; nothing else in docs/
        shutil.copytree(ROOT / d, tmp_path / d)
    shutil.copyfile(ROOT / "README.md", tmp_path / "README.md")
    monkeypatch.setattr(PV, "ROOT", tmp_path)
    monkeypatch.setattr(PV, "EV", tmp_path / "evidence")
    monkeypatch.setattr(PV, "CHECKPOINT_MANIFEST", tmp_path / "release" / "checkpoint-manifest.json")
    return tmp_path


def run(capsys, argv=()):
    capsys.readouterr()
    code = PV.main(list(argv))
    return code, capsys.readouterr().out.splitlines()


def edit(path, old, new, count=1):
    text = path.read_bytes().decode("utf-8")
    assert text.count(old) >= count, old
    path.write_bytes(text.replace(old, new, count).encode("utf-8"))


def failed(lines):
    return [x for x in lines if x.startswith("FAIL")]


CM_LINE = ("FAIL  CONSISTENT  release/checkpoint-manifest.json matches the SHA-256 and size in "
           "evidence/manifest.json")
SUMMARY_LINES = [f"FAIL  CONSISTENT  evidence/{n} matches the SHA-256 and size in evidence/manifest.json" for n in (
    "linergen-final-evaluation.json", "fem-transfer-audit.json", "falcon9-vehicle.json", "atlas-i-vehicle.json",
    "figures-manifest.json")]


def test_the_untampered_copy_passes_every_public_check(tree, capsys):
    code, lines = run(capsys)
    assert code == 0 and failed(lines) == [] and lines[-1] == "public verification passed"
    assert len([x for x in lines if x.startswith("PASS")]) == N_PUBLIC_CHECKS and set(RESULT_LINES) <= set(lines)
    assert [x for x in lines if x.startswith("SKIP")] == [
        "SKIP  checkpoint file not downloaded (artifacts/dev/physgen_v1/train/selected.pt); see README"]
    assert not [x for x in lines if "NOT CHECKED" in x or "lock" in x.lower() or "seal" in x.lower()]
    assert not (tree / "docs" / "PHYS-GEN-V1-CHECKPOINT-LOCK.json").exists()       # nothing archived is read


def test_every_check_line_says_whether_it_recomputes_or_compares(tree, capsys):
    _, lines = run(capsys)
    checks = [x for x in lines if x.startswith(("PASS", "FAIL"))]
    labelled = [x for x in checks if x[6:].startswith(("RECOMPUTED  ", "CONSISTENT  "))]
    unlabelled = sorted(x for x in checks if x not in labelled)
    assert len(labelled) == N_PUBLIC_CHECKS - len(unlabelled)
    assert all("contains no local paths" in x or "1,500 cells" in x for x in unlabelled), unlabelled


def test_the_manifest_lists_every_summary_with_its_hash():
    man = json.loads((ROOT / "evidence" / "manifest.json").read_text(encoding="utf-8"))
    hs = man["historical_summaries"]
    assert man["schema"] == SC.schema("evidence-manifest") and hs["format"] == SC.ARCHIVED_SUMMARY_FORMAT
    assert len(hs["files"]) == 5 and sorted(man["current_summaries"]) == [POSTHOC_NAME, "readme-figures.json"]
    for name, e in list(hs["files"].items()) + list(man["current_summaries"].items()):
        raw = (ROOT / "evidence" / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == e["sha256"] and len(raw) == e["bytes"]


# ---- tampering: every edited input fails
def test_an_edited_cell_value_fails_the_hash_and_the_recomputed_statistic(tree, capsys):
    p = tree / "evidence" / "linergen-final-evaluation.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    i = d["cells"]["columns"].index("D.best_val")
    row = next(r for r in d["cells"]["rows"] if r[2] == 128)
    row[i] += 0.5
    p.write_text(dump(d), encoding="utf-8", newline="\n")
    code, lines = run(capsys)
    f = failed(lines)
    assert code == 1 and SUMMARY_LINES[0] in f and lines[-1] == "public verification FAILED"
    assert any(x.startswith("FAIL  RECOMPUTED  D_minus_A at 128") for x in f)
    assert any(x.startswith("FAIL  RECOMPUTED  D_minus_B at 128") for x in f)
    assert not any("at 2048" in x for x in f)                       # only the edited budget's contrasts move


def test_an_edited_stored_aggregate_fails_because_it_is_recomputed_not_trusted(tree, capsys):
    p = tree / "evidence" / "linergen-final-evaluation.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["aggregate"]["512"]["D_minus_A"]["mean"] += 0.25
    d["aggregate"]["512"]["D_minus_A"]["ci95"][0] -= 0.01
    p.write_text(dump(d), encoding="utf-8", newline="\n")
    code, lines = run(capsys)
    assert code == 1 and any(x.startswith("FAIL  RECOMPUTED  D_minus_A at 512: mean +0.743") for x in failed(lines))


def test_an_edited_audit_pair_fails(tree, capsys):
    p = tree / "evidence" / "fem-transfer-audit.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["rows"][3]["fem"] += 0.3
    p.write_text(dump(d), encoding="utf-8", newline="\n")
    code, lines = run(capsys)
    f = failed(lines)
    assert code == 1 and any(x.startswith("FAIL  RECOMPUTED  FEM audit: r ") for x in f)
    assert any("every diff = oracle - FEM" in x for x in f)


def test_an_edited_verdict_interval_or_outcome_fails(tree, capsys):
    p = tree / "evidence" / "falcon9-vehicle.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    v = d["verdicts"][0]
    v["interval"] = [v["target_db"] + 0.1, v["target_db"] + 0.2]         # would be PASS; the outcome says UNRESOLVED
    p.write_text(dump(d), encoding="utf-8", newline="\n")
    code, lines = run(capsys)
    assert code == 1 and any("follows the interval rule" in x for x in failed(lines))


# ---- the interval rule at its boundary (PASS needs L > T; FAIL needs U < T)
def stored_verdicts():
    return [(f, vd["stage"], vd["outcome"], vd["interval"], vd["target_db"])
            for f in ("falcon9-vehicle.json", "atlas-i-vehicle.json")
            for vd in json.loads((ROOT / "evidence" / f).read_text(encoding="utf-8"))["verdicts"]]


def test_every_stored_verdict_is_unresolved_and_has_no_interval():
    """The released verdicts: five, all unresolved, none with a complete interval. The boundary rule below decides
    none of them."""
    stored = stored_verdicts()
    assert len(stored) == 5 and all(iv is None and outcome.endswith("UNRESOLVED") for _, _, outcome, iv, _ in stored)
    assert {t for *_, t in stored} == {0.9213}


def verdict_line(tree, capsys, lower, upper, outcome):
    """The verifier's line for the first Falcon verdict after its interval and outcome are set (PASS or FAIL)."""
    p = tree / "evidence" / "falcon9-vehicle.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["verdicts"][0].update(interval=[lower, upper], outcome=outcome)
    p.write_text(dump(d), encoding="utf-8", newline="\n")
    assert json.loads(p.read_text(encoding="utf-8"))["verdicts"][0]["interval"] == [lower, upper]    # exact floats
    _, lines = run(capsys)
    line = [x for x in lines
            if f"  {d['scenario_name']} {d['verdicts'][0]['stage']}: {outcome} follows the interval rule" in x]
    assert len(line) == 1
    return line[0][:4]


T = 0.9213
JUST_BELOW, JUST_ABOVE = np.nextafter(T, -np.inf).item(), np.nextafter(T, np.inf).item()


@pytest.mark.parametrize("lower,name,rule", [
    (JUST_BELOW, "below the target", "UNRESOLVED"),
    (T, "exactly equal to the target", "UNRESOLVED"),
    (JUST_ABOVE, "the smallest representable value above the target", "PASS"),
])
def test_a_pass_needs_the_lower_bound_strictly_above_the_target(tree, capsys, lower, name, rule):
    """A recorded PASS is accepted only when L > T; at L = T and below, only UNRESOLVED is accepted. The verifier
    and the formal verdict code (`vehicles.verdict.decide`) give the same outcome at each point."""
    from acoustic_liner.vehicles import verdict as VV
    assert JUST_BELOW < T < JUST_ABOVE and not (JUST_BELOW < (JUST_BELOW + T) / 2 < T)      # adjacent floats
    assert VV.decide(dict(L=lower, U=lower + 1.0), [], target=T) == rule, name
    assert verdict_line(tree, capsys, lower, lower + 1.0, "PASS") == ("PASS" if rule == "PASS" else "FAIL")
    assert verdict_line(tree, capsys, lower, lower + 1.0, "UNRESOLVED") == ("PASS" if rule == "UNRESOLVED" else "FAIL")


@pytest.mark.parametrize("upper,rule", [(JUST_BELOW, "FAIL"), (T, "UNRESOLVED"), (JUST_ABOVE, "UNRESOLVED")])
def test_a_fail_needs_the_upper_bound_strictly_below_the_target(tree, capsys, upper, rule):
    from acoustic_liner.vehicles import verdict as VV
    assert VV.decide(dict(L=upper - 1.0, U=upper), [], target=T) == rule
    assert verdict_line(tree, capsys, upper - 1.0, upper, "FAIL") == ("PASS" if rule == "FAIL" else "FAIL")
    assert verdict_line(tree, capsys, upper - 1.0, upper, "UNRESOLVED") == ("PASS" if rule == "UNRESOLVED" else "FAIL")


def test_an_edited_figure_fails(tree, capsys):
    fig = tree / "docs" / "figures" / "readme" / "fig4_paired_improvement.png"
    fig.write_bytes(fig.read_bytes() + b"\0")
    code, lines = run(capsys)
    assert code == 1 and failed(lines) == ["FAIL  CONSISTENT  every listed figure matches its SHA-256"]


def test_a_private_path_in_a_summary_fails(tree, capsys):
    edit(tree / "evidence" / "fem-transfer-audit.json", '"purpose": "', '"purpose": "C:\\\\Users\\\\someone ', 1)
    code, lines = run(capsys)
    assert code == 1 and any(x.startswith("FAIL  fem-transfer-audit.json contains no local paths") for x in failed(lines))


# ---- stale manifest hashes
def test_a_stale_summary_hash_fails_for_that_summary_only(tree, capsys):
    """A summary changes by one byte of white space; the manifest still names the old hash."""
    edit(tree / "evidence" / "atlas-i-vehicle.json", '"target_dB"', '"target_dB" ', 1)
    code, lines = run(capsys)
    assert code == 1 and failed(lines) == [SUMMARY_LINES[3]]


def test_a_stale_checkpoint_manifest_hash_fails(tree, capsys):
    cm = tree / "release" / "checkpoint-manifest.json"
    cm.write_bytes(cm.read_bytes() + b"\n")
    code, lines = run(capsys)
    assert code == 1 and CM_LINE in failed(lines)


def test_a_stale_size_fails_although_the_hash_is_right(tree, capsys):
    m = tree / "evidence" / "manifest.json"
    d = json.loads(m.read_text(encoding="utf-8"))
    d["historical_summaries"]["files"]["falcon9-vehicle.json"]["bytes"] += 1
    m.write_text(dump(d), encoding="utf-8", newline="\n")
    code, lines = run(capsys)
    assert code == 1 and failed(lines) == [SUMMARY_LINES[2]]


def rehash_in_manifest(tree, section, name):
    """Edit the manifest entry together with the file (what one copy of the hashes cannot catch)."""
    p, m = tree / "evidence" / name, tree / "evidence" / "manifest.json"
    d = json.loads(m.read_text(encoding="utf-8"))
    (d[section]["files"] if section == "historical_summaries" else d[section])[name].update(
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)
    m.write_text(dump(d), encoding="utf-8", newline="\n")


def test_a_file_edited_together_with_its_manifest_entry_is_caught_only_by_a_recomputation(tree, capsys):
    """The manifest is the one list of hashes: an edit made to a summary and to its entry together passes the
    CONSISTENT check. A changed value is still caught where the verifier recomputes it; a change it does not
    recompute (here white space) is not caught."""
    p = tree / "evidence" / "atlas-i-vehicle.json"
    edit(p, '"target_dB"', '"target_dB" ', 1)
    rehash_in_manifest(tree, "historical_summaries", "atlas-i-vehicle.json")
    code, lines = run(capsys)
    assert code == 0 and failed(lines) == []
    q = tree / "evidence" / "fem-transfer-audit.json"
    d = json.loads(q.read_text(encoding="utf-8"))
    d["rows"][3]["fem"] += 0.3
    q.write_text(dump(d), encoding="utf-8", newline="\n")
    rehash_in_manifest(tree, "historical_summaries", "fem-transfer-audit.json")
    code, lines = run(capsys)
    assert code == 1 and all(x.startswith("FAIL  RECOMPUTED  FEM audit") for x in failed(lines)) and failed(lines)


def test_an_edited_historical_hash_in_the_checkpoint_manifest_fails(tree, capsys):
    cm = tree / "release" / "checkpoint-manifest.json"
    d = json.loads(cm.read_text(encoding="utf-8"))
    d["historical_binding"]["architecture_sha256"] = "f" * 64
    d["historical_binding"]["source_sha256"] = "e" * 64
    cm.write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8", newline="\n")
    code, lines = run(capsys)
    assert code == 1 and failed(lines) == [
        CM_LINE, "FAIL  CONSISTENT  architecture and its recorded hash: summary = checkpoint manifest",
        "FAIL  CONSISTENT  recorded LinerGen source hash: final-test record = checkpoint manifest"]


# ---- unsupported historical schemas
def test_the_superseded_evidence_manifest_identifier_fails_every_summary(tree, capsys):
    """The identifier of the retired manifest format is not accepted, although every hash in the file is right."""
    edit(tree / "evidence" / "manifest.json", "acoustic-liner/evidence-manifest/1", "linergen-evidence-manifest/1")
    code, lines = run(capsys)
    assert code == 1 and [x for x in failed(lines) if "size in evidence/manifest.json" in x][:5] == SUMMARY_LINES


def test_the_retired_evidence_manifest_layout_is_refused(tree):
    """A manifest in the retired layout (one flat `files` table, a schema field per file) is not a manifest this
    verifier reads: it stops, it does not verify."""
    m = tree / "evidence" / "manifest.json"
    d = json.loads(m.read_text(encoding="utf-8"))
    hs = d.pop("historical_summaries")
    old = dict(d, schema="linergen-evidence-manifest/1",
               files={n: dict(e, schema=hs["format"]) for n, e in hs["files"].items()})
    m.write_text(dump(old), encoding="utf-8", newline="\n")
    with pytest.raises(KeyError):
        PV.main([])


@pytest.mark.parametrize("where", ["evidence manifest", "checkpoint manifest", "both"])
def test_the_superseded_checkpoint_manifest_identifier_fails(tree, capsys, where):
    new, old = "acoustic-liner/checkpoint-manifest/1", "linergen-checkpoint-manifest/1"
    if where in ("evidence manifest", "both"):
        edit(tree / "evidence" / "manifest.json", new, old)
    if where in ("checkpoint manifest", "both"):
        cm = tree / "release" / "checkpoint-manifest.json"
        edit(cm, new, old)
        if where == "both":                         # the evidence manifest also names the edited file's hash and size
            m = tree / "evidence" / "manifest.json"
            d = json.loads(m.read_text(encoding="utf-8"))
            d["checkpoint_manifest"].update(sha256=hashlib.sha256(cm.read_bytes()).hexdigest(), bytes=cm.stat().st_size)
            m.write_text(dump(d), encoding="utf-8", newline="\n")
    code, lines = run(capsys)
    assert code == 1 and CM_LINE in failed(lines)


def test_an_unregistered_summary_format_fails_every_summary(tree, capsys):
    edit(tree / "evidence" / "manifest.json", '"format": "linergen-public-evidence/1"',
         '"format": "linergen-public-evidence/2"')
    code, lines = run(capsys)
    assert code == 1 and [x for x in failed(lines) if "size in evidence/manifest.json" in x][:5] == SUMMARY_LINES


# ---- the checkpoint reader
def test_the_checkpoint_reader_refuses_any_other_pickled_name(tmp_path):
    p = tmp_path / "x.pt"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("x/data.pkl", pickle.dumps(dict(model=Path("."))))
    with pytest.raises(PV.CheckpointFormatError):
        PV.read_checkpoint(p)


# ---- the post-hoc follow-ups (evidence/posthoc-followups.json)
POSTHOC = "evidence/posthoc-followups.json"


def test_the_posthoc_summary_is_listed_and_recomputed(tree, capsys):
    raw = (ROOT / POSTHOC).read_bytes()
    man = json.loads((ROOT / "evidence" / "manifest.json").read_text(encoding="utf-8"))
    assert man["current_summaries"][POSTHOC_NAME]["sha256"] == hashlib.sha256(raw).hexdigest()
    code, lines = run(capsys)
    assert ("PASS  RECOMPUTED  post-hoc resolution_followup (atlas): PASS follows the interval rule (T = 0.9213 dB)"
            in lines)


def test_an_edited_posthoc_summary_fails_its_listed_hash(tree, capsys):
    edit(tree / POSTHOC, '"outcome": "PASS"', '"outcome": "UNRESOLVED"')
    code, lines = run(capsys)
    assert code == 1 and ("FAIL  CONSISTENT  evidence/posthoc-followups.json matches the SHA-256 and size in "
                          "evidence/manifest.json") in failed(lines)


def _rehash(tree, monkeypatch):
    """List the edited summary's new hash, so that only the recomputation can catch the edit."""
    rehash_in_manifest(tree, "current_summaries", POSTHOC_NAME)


def test_an_edited_atlas_fem_value_fails_the_recomputed_interval(tree, capsys, monkeypatch):
    s = json.loads((tree / POSTHOC).read_text(encoding="utf-8"))
    st = s["stages"][0]
    st["fem_primary"]["R-X-"]["broadband"]["0.1"] -= 0.01
    (tree / POSTHOC).write_text(json.dumps(s), encoding="utf-8")
    _rehash(tree, monkeypatch)
    code, lines = run(capsys)
    assert code == 1 and any("post-hoc resolution_followup (atlas): L" in x for x in failed(lines))


@pytest.mark.parametrize("lower", [0.9213, 0.9])
def test_the_atlas_pass_needs_the_lower_bound_strictly_above_the_target(tree, capsys, monkeypatch, lower):
    s = json.loads((tree / POSTHOC).read_text(encoding="utf-8"))
    st = s["stages"][0]
    shift = st["interval"]["L"] - lower
    for t in st["fem_primary"].values():
        for f in t.values():
            for h in f:
                f[h] -= shift
    st["sensitivity"]["fem_nominal_0p1"] -= shift
    for c in st["sensitivity"]["fem_envelope_combos_0p1"]:
        c["fem_robust_db"] -= shift
    st["interval"] = {k: (v - shift if k in ("F01", "F01_hi", "L", "U") else v) for k, v in st["interval"].items()}
    (tree / POSTHOC).write_text(json.dumps(s), encoding="utf-8")
    _rehash(tree, monkeypatch)
    code, lines = run(capsys)
    assert code == 1 and ("FAIL  RECOMPUTED  post-hoc resolution_followup (atlas): PASS follows the interval rule "
                          "(T = 0.9213 dB)") in lines


def test_a_bounded_label_without_a_complete_chain_fails(tree, capsys, monkeypatch):
    edit(tree / POSTHOC, '"label": "NOT BOUNDED"', '"label": "BOUNDED ABOVE TARGET"')
    _rehash(tree, monkeypatch)
    code, lines = run(capsys)
    assert code == 1 and any("post-hoc bounded_sensitivity" in x for x in failed(lines))


def test_an_accepted_highspeed_pair_with_a_failed_gate_fails(tree, capsys, monkeypatch):
    s = json.loads((tree / POSTHOC).read_text(encoding="utf-8"))
    st = next(x for x in s["stages"] if x["stage"] == "highspeed_pair")
    st["runs"]["V"]["accepted"], st["pair_accepted"] = True, True
    (tree / POSTHOC).write_text(json.dumps(s), encoding="utf-8")
    _rehash(tree, monkeypatch)
    code, lines = run(capsys)
    assert code == 1 and any("post-hoc highspeed_pair" in x for x in failed(lines))


# ---- the README figures (evidence/readme-figures.json)
def test_every_readme_figure_is_registered_with_its_current_hash():
    rf = json.loads((ROOT / "evidence" / "readme-figures.json").read_text(encoding="utf-8"))
    assert rf["schema"] == SC.schema("readme-figures") and len(rf["figures"]) == 12
    for f in rf["figures"]:
        assert hashlib.sha256((ROOT / f["path"]).read_bytes()).hexdigest() == f["sha256"], f["path"]
        if f["script"]:
            assert hashlib.sha256((ROOT / f["script"]["path"]).read_bytes()).hexdigest() == f["script"]["sha256"]


def test_an_edited_readme_figure_fails(tree, capsys):
    p = tree / "docs" / "figures" / "readme" / "fig6_designs_mass.png"
    p.write_bytes(p.read_bytes() + b"\0")
    code, lines = run(capsys)
    assert code == 1 and "FAIL  CONSISTENT  every listed figure matches its SHA-256" in lines
