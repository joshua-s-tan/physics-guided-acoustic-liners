# -*- coding: utf-8 -*-
"""Verify a downloaded LinerGen checkpoint against `release/checkpoint-manifest.json`.

    acoustic-liner provenance checkpoint [path]          default path: artifacts/dev/physgen_v1/train/selected.pt
    acoustic-liner provenance checkpoint [path] --full   also run the study's own binding check (models.checkpoint.load_checkpoint)

Checks, in order:
  1. file SHA-256 and size equal the manifest (the bytes are the published asset);
  2. the file loads with `torch.load(weights_only=True)` (no project code is unpickled), and the weight hash recomputed
     from the tensors equals the manifest;
  3. tensor count, parameter count and dtype equal the manifest; the configuration hash equals the manifest; the
     three source hashes recorded in the file by the historical implementation equal the manifest's
     `historical_binding`; the architecture and coordinate-transform identities recomputed from this code equal the
     manifest (this code builds the same network and decodes designs the same way).
`--full` additionally re-checks the training binding and schedule with the study code itself; it unpickles the full
file, so it runs only after checks 1-3 have passed.

The manifest was written once, from the trusted training output; this tool does not rewrite it. Its
`public_model_source_sha256` and `public_scorer_source_sha256` fields record the source of the release it was
written with; they are not compared with this checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .. import paths
from . import schemas as SC
ROOT = paths.ROOT
MANIFEST = ROOT / "release" / "checkpoint-manifest.json"
DEFAULT = ROOT / "artifacts" / "dev" / "physgen_v1" / "train" / "selected.pt"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_weights_only(path):
    import torch
    from torch.torch_version import TorchVersion
    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(path, weights_only=True)


def weights_sha256(state_dict):
    """Same content hash as `models.checkpoint.model_sha256`: sorted tensor names plus raw tensor bytes."""
    h = hashlib.sha256()
    for k, v in sorted(state_dict.items()):
        h.update(k.encode())
        h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def describe(path):
    ck = load_weights_only(path)
    sd, b = ck["model"], ck["binding"]
    return dict(
        weights_sha256=weights_sha256(sd),
        n_tensors=len(sd),
        n_parameters=int(sum(v.numel() for v in sd.values())),
        dtypes=sorted({str(v.dtype).replace("torch.", "") for v in sd.values()}),
        selected_epoch=int(ck["epoch"]),
        architecture_sha256=b["arch_sha256"],
        configuration_sha256=b["config_sha256"],
        configuration=b["config"],
        source_sha256=b["source_sha256"],
        scorer_sha256=b["scorer_sha256"],
        training_split_hash=b["training_hash"],
        validation_split_hash=b["validation_hash"],
        recorded_with=dict(torch=b["torch"], numpy=b["numpy"]),
    )


HISTORICAL_BINDING = ("architecture_sha256", "source_sha256", "scorer_sha256")


def spec_identities():
    """The architecture and coordinate-transform identities, recomputed from this code."""
    from ..models import checkpoint as CKPT
    return CKPT.spec_identities()


def verify(path, full=False):
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + name)
        ok &= bool(cond)

    p = Path(path)
    if not p.is_file():
        sys.exit(f"no checkpoint at {p}; download {m['asset']['filename']} and save it as {m['asset']['save_as']}")
    check("file SHA-256 matches the manifest", sha256_file(p) == m["asset"]["sha256"])
    check("file size matches the manifest", p.stat().st_size == m["asset"]["size_bytes"])
    if not ok:
        sys.exit("stopping: the file is not the published asset")
    d = describe(p)
    check("weight hash matches the manifest", d["weights_sha256"] == m["model"]["weights_sha256"])
    for k in ("n_tensors", "n_parameters", "dtypes", "selected_epoch"):
        check(f"{k} matches the manifest", d[k] == m["model"][k])
    check("manifest schema is the public checkpoint-manifest schema", m.get("schema") == SC.schema("checkpoint-manifest"))
    check("configuration_sha256 matches the manifest", d["configuration_sha256"] == m["binding"]["configuration_sha256"])
    for k in HISTORICAL_BINDING:
        check(f"historical {k} matches the manifest", d[k] == m["historical_binding"][k])
    for k, v in spec_identities().items():
        check(f"{k} recomputed from this code matches the manifest", v == m.get(k))
    if full and ok:
        from ..models import checkpoint as CKPT
        try:
            CKPT.load_checkpoint(str(p))
            check("study binding and training schedule (models.checkpoint.load_checkpoint)", True)
        except Exception as e:                       # report the study's own refusal reason
            print(f"      {type(e).__name__}: {e}")
            check("study binding and training schedule (models.checkpoint.load_checkpoint)", False)
    print("checkpoint verified" if ok else "checkpoint NOT verified")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", nargs="?", default=str(DEFAULT))
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args(argv)
    return 0 if verify(a.path, a.full) else 1


if __name__ == "__main__":
    sys.exit(main())
