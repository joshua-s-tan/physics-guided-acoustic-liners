"""The checkpoint manifest (`release/checkpoint-manifest.json`) and the checkpoint verification of
`acoustic-liner provenance checkpoint`.

The released checkpoint is a download and is not part of the repository, so no test here reads it. The committed
manifest is checked against this code (architecture, coordinate transform) and the evidence manifest. The
verification itself is exercised on a small synthetic checkpoint of the same file form, written by the test, with a
manifest describing it in a temporary directory: the complete file passes, and every edit of the file or of the
manifest fails. The synthetic checkpoint is test-only; it is not a model and no result is computed from it.
"""
import collections
import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from acoustic_liner import paths
from acoustic_liner.models import checkpoint as CKPT, linergen as LG
from acoustic_liner.provenance import checkpoint_manifest as CM, public_verifier as PV, schemas as SC

ROOT = paths.ROOT
MANIFEST = json.loads((ROOT / "release" / "checkpoint-manifest.json").read_text(encoding="utf-8"))
SPEC = ("public_architecture_sha256", "public_coordinate_transform_sha256")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- the committed manifest
def test_the_manifest_has_the_public_schema_and_names_the_released_asset():
    assert MANIFEST["schema"] == SC.schema("checkpoint-manifest") == "acoustic-liner/checkpoint-manifest/1"
    asset = MANIFEST["asset"]
    assert asset["save_as"] == "artifacts/dev/physgen_v1/train/selected.pt" == CM.DEFAULT.relative_to(ROOT).as_posix()
    assert len(asset["sha256"]) == 64 and asset["size_bytes"] > 0
    assert MANIFEST["model"]["python_class"] == f"{LG.LinerGenerator.__module__}.{LG.LinerGenerator.__name__}"


def test_this_code_reproduces_the_manifest_architecture_and_coordinate_transform():
    now = CM.spec_identities()
    assert tuple(now) == SPEC
    for name in SPEC:
        assert MANIFEST[name] == now[name]
    assert CKPT.released_weights_sha256() == MANIFEST["model"]["weights_sha256"]


def test_the_historical_binding_has_the_three_recorded_hashes():
    hb = MANIFEST["historical_binding"]
    assert sorted(hb) == sorted(CM.HISTORICAL_BINDING) and all(len(v) == 64 for v in hb.values())
    assert not set(hb.values()) & {MANIFEST[name] for name in SPEC}     # historical values are not this code's


def test_the_manifest_agrees_with_the_architecture():
    model = MANIFEST["model"]
    assert model["architecture"] == LG.CONFIG and model["dtypes"] == ["float64"]
    assert model["n_parameters"] == PV.parameter_count(LG.CONFIG) == int(
        sum(p.numel() for p in LG.LinerGenerator().parameters()))
    assert model["n_tensors"] == len(LG.LinerGenerator().state_dict())


def test_the_evidence_manifest_names_this_checkpoint_manifest():
    e = json.loads((ROOT / "evidence" / "manifest.json").read_text(encoding="utf-8"))["checkpoint_manifest"]
    p = ROOT / "release" / "checkpoint-manifest.json"
    assert e["path"] == "release/checkpoint-manifest.json" and e["schema"] == MANIFEST["schema"]
    assert e["sha256"] == sha(p) and e["bytes"] == p.stat().st_size


# ---- verification, on a synthetic checkpoint
def write_checkpoint(path, scale=1.0, **binding):
    """A small file of the released checkpoint's form: float64 tensors, the binding record, the epoch."""
    g = torch.Generator().manual_seed(7)
    model = collections.OrderedDict(
        (name, scale * torch.randn(shape, generator=g, dtype=torch.float64))
        for name, shape in (("inp.weight", (4, 3)), ("inp.bias", (4,)), ("head.weight", (2, 4)), ("head.bias", (2,))))
    config = dict(epochs=3, lr=0.001)
    hb = MANIFEST["historical_binding"]
    b = dict(arch_sha256=hb["architecture_sha256"], source_sha256=hb["source_sha256"],
             scorer_sha256=hb["scorer_sha256"], config=config, config_sha256="c" * 64, training_hash="a" * 64,
             validation_hash="b" * 64, torch=str(torch.__version__), numpy="2")
    b.update(binding)
    torch.save(dict(model=model, binding=b, epoch=2), path)
    return path


def manifest_for(ckpt):
    """A manifest of the committed form describing `ckpt`."""
    d = CM.describe(ckpt)
    m = json.loads(json.dumps(MANIFEST))
    m["asset"].update(sha256=sha(ckpt), size_bytes=ckpt.stat().st_size)
    m["model"].update({k: d[k] for k in ("weights_sha256", "n_tensors", "n_parameters", "dtypes", "selected_epoch")})
    m["binding"]["configuration_sha256"] = d["configuration_sha256"]
    m["historical_binding"] = {k: d[k] for k in CM.HISTORICAL_BINDING}
    return m


@pytest.fixture
def box(tmp_path, monkeypatch):
    ckpt = write_checkpoint(tmp_path / "selected.pt")
    path = tmp_path / "release" / "checkpoint-manifest.json"
    path.parent.mkdir()
    path.write_text(json.dumps(manifest_for(ckpt), indent=1) + "\n", encoding="utf-8", newline="\n")
    monkeypatch.setattr(CM, "ROOT", tmp_path)
    monkeypatch.setattr(CM, "MANIFEST", path)
    return dict(ckpt=ckpt, manifest=path, dir=tmp_path)


def verify(capsys, path, **kw):
    capsys.readouterr()
    ok = CM.verify(path, **kw)
    lines = capsys.readouterr().out.splitlines()
    return ok, [x for x in lines if x.startswith("FAIL")], lines


def rewrite(path, change):
    d = json.loads(path.read_text(encoding="utf-8"))
    change(d)
    path.write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8", newline="\n")


def test_the_complete_file_verifies(box, capsys):
    ok, fails, lines = verify(capsys, box["ckpt"])
    assert ok is True and fails == [] and lines[-1] == "checkpoint verified"
    assert len([x for x in lines if x.startswith("PASS")]) == 14
    assert not any("NOT CHECKED" in x or "lock" in x for x in lines)
    assert CM.main([str(box["ckpt"])]) == 0


def test_the_reader_without_torch_gives_the_same_description(box):
    mine, theirs = PV.describe_checkpoint(box["ckpt"]), CM.describe(box["ckpt"])
    assert mine == {k: theirs[k] for k in mine} and len(mine) == 9
    cm = json.loads(box["manifest"].read_text(encoding="utf-8"))
    lines = []
    assert PV.verify_checkpoint(box["ckpt"], cm, out=lines.append) is True
    assert lines[-1] == "checkpoint verified" and not [x for x in lines if x.startswith("FAIL")]


def test_a_missing_file_stops(box):
    with pytest.raises(SystemExit, match="no checkpoint at"):
        CM.verify(box["dir"] / "absent.pt")


@pytest.mark.parametrize("change", ["one byte", "appended byte", "other weights"])
def test_a_file_that_is_not_the_published_bytes_stops_before_it_is_loaded(box, change, monkeypatch):
    raw = bytearray(box["ckpt"].read_bytes())
    if change == "one byte":
        raw[len(raw) // 2] ^= 1
        box["ckpt"].write_bytes(bytes(raw))
    elif change == "appended byte":
        box["ckpt"].write_bytes(bytes(raw) + b"\0")
    else:
        write_checkpoint(box["ckpt"], scale=1.0000001)
    monkeypatch.setattr(CM, "describe", lambda p: pytest.fail("the file was loaded"))
    with pytest.raises(SystemExit, match="not the published asset"):
        CM.verify(box["ckpt"])
    cm = json.loads(box["manifest"].read_text(encoding="utf-8"))
    with pytest.raises(SystemExit, match="not the published asset"):
        PV.verify_checkpoint(box["ckpt"], cm, out=lambda s: None)


@pytest.mark.parametrize("change,line", [
    (lambda m: m["model"].update(weights_sha256="0" * 64), "FAIL  weight hash matches the manifest"),
    (lambda m: m["model"].update(n_parameters=27), "FAIL  n_parameters matches the manifest"),
    (lambda m: m["model"].update(selected_epoch=3), "FAIL  selected_epoch matches the manifest"),
    (lambda m: m.update(schema="linergen-checkpoint-manifest/1"),
     "FAIL  manifest schema is the public checkpoint-manifest schema"),
    (lambda m: m["binding"].update(configuration_sha256="0" * 64), "FAIL  configuration_sha256 matches the manifest"),
    (lambda m: m["historical_binding"].update(source_sha256="0" * 64), "FAIL  historical source_sha256 matches the manifest"),
    (lambda m: m.update(public_architecture_sha256="0" * 64),
     "FAIL  public_architecture_sha256 recomputed from this code matches the manifest"),
    (lambda m: m.pop("public_coordinate_transform_sha256"),
     "FAIL  public_coordinate_transform_sha256 recomputed from this code matches the manifest"),
])
def test_an_edited_manifest_fails_the_check_of_the_edited_value_only(box, capsys, change, line):
    rewrite(box["manifest"], change)
    ok, fails, lines = verify(capsys, box["ckpt"])
    assert ok is False and fails == [line] and lines[-1] == "checkpoint NOT verified"
    assert CM.main([str(box["ckpt"])]) == 1


def test_source_scope_fields_of_the_manifest_are_not_compared_with_this_checkout(box, capsys):
    """The manifest's source-scope hashes describe the release it was written with; editing the code (or the field)
    does not fail the check."""
    rewrite(box["manifest"], lambda m: m.update(public_model_source_sha256="0" * 64))
    ok, fails, _ = verify(capsys, box["ckpt"])
    assert ok is True and fails == []


def test_the_weight_hash_depends_on_names_and_values():
    a = collections.OrderedDict(w=torch.zeros(2, dtype=torch.float64), b=torch.ones(1, dtype=torch.float64))
    h = CM.weights_sha256(a)
    assert h == CM.weights_sha256(collections.OrderedDict(reversed(list(a.items()))))      # order does not matter
    assert h != CM.weights_sha256(dict(a, b=torch.ones(1, dtype=torch.float64) * 2))
    assert h != CM.weights_sha256(dict(w=a["w"], c=a["b"]))
