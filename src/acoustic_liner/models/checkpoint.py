"""LinerGen checkpoint identity: the binding recorded at training time (architecture, coordinate transform,
configuration, scorer, schedule, seeds and split hashes), its compatibility checks, and checkpoint loading.

The loader refuses a checkpoint this code cannot reproduce: a different architecture specification or coordinate
transform (both recomputed from this code), a configuration that does not match its hash, another scorer, node
gradient, schedule or seeds, or weights that do not match their recorded content hash. It does not compare source
files: editing the code does not lock the checkpoint out.

The released checkpoint was trained by the historical implementation. Its binding keeps the three source hashes of
that implementation under their original keys; they must equal the `historical_binding` of
`release/checkpoint-manifest.json`, whose architecture and coordinate-transform identities this code must reproduce.
A checkpoint trained by this package (binding version 3) records the two specification identities itself, and the
package source fingerprint for information only. Version-2 bindings, which also recorded source-scope hashes, are
read with the same checks.
"""
from __future__ import annotations

import copy
import hashlib
import json
import numpy as np
import torch
from ..design import coordinates as PC
from ..evaluation import splits as PCN
from . import linergen as PM
from . import objective as PO
from .. import paths
from ..provenance import hashing as H
from .objective import curriculum


SCORER = "quality_eval_exact"
NODE_GRAD = "cap_only"
TAU_SPEC = dict(kind="geometric", start=PO.TAU_COV_START, end=PO.TAU_COV_END, formula="start*(end/start)**q",
                q="(p - ramp_end) / (sharpen_end - ramp_end) clipped to [0, 1]")


class CheckpointError(RuntimeError):
    """A checkpoint is unbound, mis-bound or inconsistent with the registered training policy."""


# =================================================================================================================
#  hashes / binding
# =================================================================================================================
def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=list).encode()).hexdigest()


def model_sha256(model):
    """Content hash of the parameters (file bytes depend on pickle sharing, not on content)."""
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


COORDINATE_PROBES = 16                    # fixed probe set of the coordinate-transform identity
HISTORICAL_BINDING_KEYS = ("arch_sha256", "source_sha256", "scorer_sha256")
SPEC_BINDING_KEYS = ("public_architecture_sha256", "public_coordinate_transform_sha256")
BINDING_VERSION = 3                         # 2: also recorded the model and scorer source-scope hashes
SUPPORTED_BINDING_VERSIONS = (2, 3)
RELEASE_MANIFEST = paths.ROOT / "release" / "checkpoint-manifest.json"


def coordinate_probe_inputs():
    """The registered probe set: COORDINATE_PROBES points of the unconstrained 18-dimensional design space, built
    with exact arithmetic only (no random numbers, no transcendental function), float64, C order."""
    k = np.arange(COORDINATE_PROBES * PC.DIM, dtype=np.float64).reshape(COORDINATE_PROBES, PC.DIM)
    return ((k * 37.0) % 61.0) / 61.0 * 6.0 - 3.0


def _array_identity(a):
    a = np.ascontiguousarray(a)
    return dict(shape=list(a.shape), dtype=str(a.dtype), order="C", sha256=hashlib.sha256(a.tobytes()).hexdigest())


def coordinate_transform_specification():
    """Probe inputs (order, shape, dtype, bytes) together with what `decode_np` returns for each: the 19-entry
    optimizer vector and the decoded design as [D, face_t, wall_t, then f0, r_neck, L_neck, area_frac, Q per bank]."""
    u = coordinate_probe_inputs()
    vecs, designs = [], []
    for row in u:
        d, v = PC.decode_np(row)
        vecs.append(np.asarray(v, dtype=np.float64))
        designs.append([d["D"], d["face_t"], d["wall_t"]] + [g[k] for g in d["groups"]
                                                             for k in ("f0", "r_neck", "L_neck", "area_frac", "Q")])
    return dict(function="design.coordinates.decode_np", inputs=_array_identity(u),
                outputs=dict(optimizer_vector=_array_identity(np.array(vecs)),
                             design=_array_identity(np.array(designs, dtype=np.float64))))


def coordinate_transform_sha256():
    spec = json.dumps(coordinate_transform_specification(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(spec.encode("ascii")).hexdigest()


def spec_identities():
    """The two specification identities, recomputed from this code."""
    return dict(public_architecture_sha256=PM.public_architecture_sha256(),
                public_coordinate_transform_sha256=coordinate_transform_sha256())


def released_manifest():
    return json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))


def released_weights_sha256():
    """The weight hash of the released selected checkpoint (`release/checkpoint-manifest.json`)."""
    return released_manifest()["model"]["weights_sha256"]


def require_released(ck):
    """The vehicle and FEM-audit stages use the selected checkpoint of the study, nothing else."""
    if ck.get("model_sha256") != released_weights_sha256():
        raise CheckpointError("checkpoint is not the released selected checkpoint (release/checkpoint-manifest.json)")


def binding(cfg, training, validation):
    return dict(provenance_schema_version=BINDING_VERSION, config=copy.deepcopy(cfg),
                config_sha256=_sha(cfg), **spec_identities(), package_source_sha256=H.package_source_sha256(),
                training_hash=PCN.split_hash(training),
                validation_hash=PCN.split_hash(validation), scorer=SCORER,
                node_grad=NODE_GRAD, tau_schedule=dict(TAU_SPEC), seeds=PCN.named_seeds(),
                torch=torch.__version__, numpy=np.__version__)


def checkpoint_identity(ck):
    """The identity block a record keeps about the checkpoint it used. The weights and configuration hashes always;
    for a checkpoint of the historical implementation its two recorded source hashes, labelled historical; for a
    checkpoint trained by this package its architecture identity and the source fingerprint it recorded."""
    b = ck["binding"]
    out = dict(model_sha256=ck.get("model_sha256"), config_sha256=b["config_sha256"])
    if is_historical_binding(b):
        out.update(historical_arch_sha256=b["arch_sha256"], historical_source_sha256=b["source_sha256"])
    else:
        out.update(public_architecture_sha256=b["public_architecture_sha256"])
        if "package_source_sha256" in b:                                    # version 3
            out["checkpoint_package_source_sha256"] = b["package_source_sha256"]
        if "public_model_source_sha256" in b:                               # version 2
            out["public_model_source_sha256"] = b["public_model_source_sha256"]
    return out


CHECKPOINT_IDENTITY_KEYS = ("model_sha256", "config_sha256", "historical_arch_sha256", "historical_source_sha256",
                            "public_architecture_sha256", "checkpoint_package_source_sha256",
                            "public_model_source_sha256")


def is_historical_binding(b):
    """True for a binding written by the historical implementation (it has no provenance schema version)."""
    return "provenance_schema_version" not in b


def _identity_checks(b):
    """Compatibility of a binding with this code. A historical binding: its three recorded hashes must equal the
    released manifest's `historical_binding`, and this code must reproduce the manifest's architecture and
    coordinate-transform identities. A binding of this package: its two specification identities must equal this
    code's. A binding may not mix the two kinds."""
    hist, spec = [k for k in HISTORICAL_BINDING_KEYS if k in b], [k for k in SPEC_BINDING_KEYS if k in b]
    now = spec_identities()
    if is_historical_binding(b):
        if spec or len(hist) != len(HISTORICAL_BINDING_KEYS):
            raise CheckpointError(f"historical checkpoint binding has the wrong identity fields: {sorted(hist + spec)}")
        m = released_manifest()
        hb = m["historical_binding"]
        return dict(arch=b["arch_sha256"] == hb["architecture_sha256"]
                    and now["public_architecture_sha256"] == m["public_architecture_sha256"],
                    source=b["source_sha256"] == hb["source_sha256"] and b["scorer_sha256"] == hb["scorer_sha256"],
                    coordinate_transform=now["public_coordinate_transform_sha256"]
                    == m["public_coordinate_transform_sha256"])
    if b["provenance_schema_version"] not in SUPPORTED_BINDING_VERSIONS or hist or len(spec) != len(SPEC_BINDING_KEYS):
        raise CheckpointError(f"checkpoint binding has the wrong identity fields: {sorted(hist + spec)}")
    return dict(arch=b["public_architecture_sha256"] == now["public_architecture_sha256"], source=True,
                coordinate_transform=b["public_coordinate_transform_sha256"]
                == now["public_coordinate_transform_sha256"])


# =================================================================================================================
#  checkpoint loader (the ONLY way a trained generator reaches the study driver)
# =================================================================================================================
def _check_binding(b, training=None, validation=None):
    need = ("config", "config_sha256", "training_hash", "validation_hash", "scorer", "node_grad", "tau_schedule",
            "seeds")
    miss = [k for k in need if k not in b]
    if miss:
        raise CheckpointError(f"checkpoint binding missing {miss}")
    cfg = b["config"]
    ident = _identity_checks(b)
    checks = dict(arch=ident["arch"], config=b["config_sha256"] == _sha(cfg),
                  historical_binding=ident["source"], scorer=b["scorer"] == SCORER == cfg.get("scorer"),
                  coordinate_transform=ident["coordinate_transform"],
                  node_grad=b["node_grad"] == NODE_GRAD == cfg.get("node_grad"),
                  tau_schedule=b["tau_schedule"] == TAU_SPEC and cfg.get("tau_corner") == 0.0,
                  seeds=b["seeds"] == PCN.named_seeds())
    if training is not None:
        checks["training_split"] = b["training_hash"] == PCN.split_hash(training)
    if validation is not None:
        checks["validation_split"] = b["validation_hash"] == PCN.split_hash(validation)
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise CheckpointError(f"checkpoint binding rejected: {bad}")


def load_checkpoint(path, training=None, validation=None):
    """Load a selected (or state) checkpoint; rejects any binding / policy / schedule mismatch."""
    ck = torch.load(path, weights_only=False)
    if "binding" not in ck or "model" not in ck or "history" not in ck:
        raise CheckpointError("not a PHYS-GEN checkpoint")
    _check_binding(ck["binding"], training, validation)
    cfg = ck["binding"]["config"]
    for h in ck["history"]:
        ph, w, tau = curriculum(h["p"], cfg)
        if h["tau_cov"] != tau or h["w_perf"] != w or h["phase"] != ph:
            raise CheckpointError(f"history step {h['step']} departs from the registered tau / weight schedule")
    model = PM.build(0)
    model.load_state_dict(ck["model"])
    model.eval()
    if "model_sha256" in ck and ck["model_sha256"] != model_sha256(model):
        raise CheckpointError("model weights do not match the recorded content hash")
    return model, ck
