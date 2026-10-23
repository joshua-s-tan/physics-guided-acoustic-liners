# -*- coding: utf-8 -*-
r"""PHYS-GEN-V1 conditions: named seeds, the stratified training distribution, the frozen training / training-validation
split, the leakage barrier and the OOD scorer (pre-registration sec. 6, 11, 13).

SEEDS. One root label -> `SeedSequence.spawn` named streams, collision-checked against CONSUMED_SEEDS and every
v2.4-v2.8 derived seed (a collision is a hard failure, not a redraw).

TRAINING DISTRIBUTION (sec. 6). Latin hypercube over nine marginals, so every marginal stratum receives exactly equal
weight: the four spectrum-shape parameters of the frozen synthetic family (`spectra.PARAM_BOX`), spectral level, wall
area, kappa, attenuation-target fraction and mass target. The attenuation target is built exactly as the evaluation
family builds it (`max(0.25, frac x oracle achievable max)`); the evaluation family's fixed values (frac 0.5,
m_total 200 kg) lie inside the training ranges. No historical design or label is used.

EVALUATION SPLITS (sec. 13-14). `pilot_physgen_v1`, `development_physgen_v1`, `final_physgen_v1` use the established
evaluation generator `evaluation.conditions.synthetic_conditions` with new named seeds. Before the seal they are BUILT ONLY TO
HASH them for the leakage barrier; nothing here scores, trains on or selects with them.

LEAKAGE (sec. 6). Training and training-validation condition hashes must not intersect each other or any forbidden
split (every v2.4-v2.8 pilot/development/final split, the three PHYS-GEN evaluation splits, Falcon); a standardized
nearest-neighbour near-duplicate check is also applied. Any failure raises (fail closed).
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import stat
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..physics import spectra as SP
from . import conditions as COND
from . import leakage as LEAK

ROOT_LABEL = b"casef-physgen-v1-root"
STREAM_NAMES = ("train_conditions", "validation_conditions", "model_init", "training_noise", "validation_noise",
                "pilot_split", "development_split", "final_split", "evaluation", "bootstrap", "fem_audit")

# ---- r1 pins (Phase I; never chosen on any held-out outcome) ---------------------------------------------------------
N_TRAIN = 2048
N_VALIDATION = 128
PILOT_N, DEVELOPMENT_N, FINAL_N = 10, 50, 100
LEVEL_DB_RANGE = SP.LEVEL_DB_RANGE              # (120, 145) dB; weights are level-invariant, stratified anyway
A_WALL_RANGE = (50.0, 200.0)                    # = evaluation family
KAPPA_RANGE = (0.0, 0.30)                       # = evaluation family
TARGET_FRAC_RANGE = (0.30, 0.70)                # evaluation family uses 0.5
M_TOTAL_RANGE = (100.0, 300.0)                  # kg; evaluation family uses 200
TARGET_FLOOR_DB = 0.25                          # = evaluation family
LHS_NAMES = SP.PARAM_NAMES + ("level_db", "A_wall", "kappa", "target_frac", "m_total_target")
LHS_BOX = SP.PARAM_BOX + (LEVEL_DB_RANGE, A_WALL_RANGE, KAPPA_RANGE, TARGET_FRAC_RANGE, M_TOTAL_RANGE)
OOD_QUANTILE = 0.99                             # deployment threshold: training-validation quantile of the distance
NEAR_QUANTILE = 0.95                            # near-threshold band [q0.95, q0.99)
NEAR_MIX_FRACTION = 0.5                         # post-lock deployment: generator share of the 32 near-threshold proposals
NEAR_DUP_FRAC = LEAK.NEAR_DUP_FRAC               # 1% of the training median nearest-neighbour distance
VEHICLE_WEIGHT_TOL = 1e-6                       # max |shape-weight difference| at or below which a spectrum IS a vehicle's


class PhysgenConditionError(RuntimeError):
    """Seed collision, malformed condition, or a leakage-barrier failure (fail closed)."""


# =================================================================================================================
#  seeds
# =================================================================================================================
def root_seed() -> int:
    return int.from_bytes(hashlib.sha256(ROOT_LABEL).digest()[:8], "big")


def streams():
    return dict(zip(STREAM_NAMES, np.random.SeedSequence(root_seed()).spawn(len(STREAM_NAMES))))


@functools.lru_cache(maxsize=1)
def named_seeds():
    from .historical_splits import v24 as SV
    seeds = {n: SV.derived_seed(ss) for n, ss in streams().items()}
    check_seed_collisions(seeds)
    return dict(seeds)


def check_seed_collisions(seeds):
    from .historical_splits import v24 as SV
    from .historical_splits import v28 as SV28
    try:
        SV28.check_seed_collisions(seeds)                    # CONSUMED + v2.4-v2.7, and pairwise distinct
    except Exception as e:
        raise PhysgenConditionError(f"PHYS-GEN seed collision: {e}") from e
    prior = set(SV28.v28_named_seeds().values()) | {SV28.pilot_split_seed()}
    bad = {n: s for n, s in seeds.items() if s in prior}
    if bad:
        raise PhysgenConditionError(f"PHYS-GEN seed(s) collide with v2.8 seeds: {bad}")
    if len(set(seeds.values())) != len(seeds):
        raise PhysgenConditionError("PHYS-GEN named seeds are not pairwise distinct")
    return seeds


# =================================================================================================================
#  training distribution
# =================================================================================================================
@functools.lru_cache(maxsize=1)
def _amax_fn():
    from ..physics import oracle as O2
    from .requests import oracle_achievable_max_fn
    return oracle_achievable_max_fn(O2)


def _build_one(args):
    i, row, group = args
    p = dict(zip(LHS_NAMES, map(float, row)))
    spec = SP.spectrum_from_params({k: p[k] for k in SP.PARAM_NAMES}, p["level_db"])
    spl = np.asarray(spec["spl_dB"], float)
    if SP.band_span_dB(spl) > SP.MAX_BAND_SPAN_DB or float(np.max(np.abs(np.diff(spl)))) > SP.MAX_ADJ_JUMP_DB:
        raise PhysgenConditionError(f"{group}_{i}: spectrum outside the validated family bounds")
    amax = float(_amax_fn()(spl_dB=spl, kappa=p["kappa"], A_wall=p["A_wall"]))
    return dict(id=f"{group}_{i}", group=group, weights=[float(w) for w in spec["weights"]], spl=spl.tolist(),
                dOASPL_target=float(max(TARGET_FLOOR_DB, p["target_frac"] * amax)),
                m_total_target=p["m_total_target"], A_wall=p["A_wall"], kappa=p["kappa"],
                strata={k: p[k] for k in LHS_NAMES})


def sample_training_conditions(n, seed, group, workers=0):
    """n stratified (LHS, every marginal stratum equal weight) training-family conditions, deterministic in `seed`."""
    X = SP._lhs(int(n), LHS_BOX, np.random.default_rng(np.random.SeedSequence(int(seed))))
    jobs = [(i, X[i], group) for i in range(int(n))]
    if workers and n > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            return list(ex.map(_build_one, jobs, chunksize=8))
    return [_build_one(j) for j in jobs]


def condition_hash(c) -> str:
    return COND.condition_hash(c)


def split_hash(conds) -> str:
    return COND.split_hash(conds)


def training_splits(workers=0, cache_dir=None, n_train=N_TRAIN, n_val=N_VALIDATION):
    """The (training, training_validation) splits from their named streams. With `cache_dir` they are FROZEN: written
    once, atomically (temp file + os.replace), then made read-only; every later call loads that artifact and verifies
    its content hashes; it is never regenerated or overwritten."""
    path = None if cache_dir is None else frozen_split_path(cache_dir, n_train, n_val)
    if path and os.path.exists(path):
        return load_frozen_splits(path)
    s = named_seeds()
    tr = sample_training_conditions(n_train, s["train_conditions"], "training", workers)
    va = sample_training_conditions(n_val, s["validation_conditions"], "training_validation", workers)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(dict(training=tr, training_validation=va, training_hash=split_hash(tr),
                           validation_hash=split_hash(va), seeds=s), f)
        os.replace(tmp, path)
        os.chmod(path, stat.S_IREAD)
        return load_frozen_splits(path)
    return tr, va


def frozen_split_path(cache_dir, n_train=N_TRAIN, n_val=N_VALIDATION):
    return os.path.join(cache_dir, f"physgen_v1_splits_{n_train}_{n_val}.json")


def load_frozen_splits(path, expected=None):
    """Load the frozen split artifact; verify its recorded hashes (and `expected` {training_hash, validation_hash,
    file_sha256}, e.g. those of the released study). Raises on any mismatch."""
    raw = open(path, "rb").read()
    doc = json.loads(raw)
    tr, va = doc["training"], doc["training_validation"]
    got = dict(training_hash=split_hash(tr), validation_hash=split_hash(va), file_sha256=hashlib.sha256(raw).hexdigest())
    if got["training_hash"] != doc["training_hash"] or got["validation_hash"] != doc["validation_hash"]:
        raise PhysgenConditionError("frozen split artifact: content does not match its recorded hashes")
    if expected is not None:
        bad = [k for k, v in expected.items() if got.get(k) != v]
        if bad:
            raise PhysgenConditionError(f"frozen split artifact does not match the expected hashes: {bad}")
    return tr, va


def split_file_sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def full_barrier():
    """Everything the leakage barrier checks for the definitive run."""
    return dict(forbidden_hashes=forbidden_condition_hashes(), vehicle_spectra=protected_vehicle_spectra(),
                forbidden_features=feature_matrix(falcon_conditions()))


# =================================================================================================================
#  evaluation splits (built ONLY to hash before the seal) + forbidden set
# =================================================================================================================
def evaluation_split(name):
    s = named_seeds()
    n, key = {"pilot_physgen_v1": (PILOT_N, "pilot_split"), "development_physgen_v1": (DEVELOPMENT_N, "development_split"),
              "final_physgen_v1": (FINAL_N, "final_split")}[name]
    return COND.synthetic_conditions(n, s[key], name)


@functools.lru_cache(maxsize=1)
def forbidden_condition_hashes():
    """Hashes of every split training must avoid: v2.4-v2.8 evaluation splits (via evaluation.historical_splits.v28's registry, which
    includes the v2.8 final_test hashes WITHOUT scoring it), the three PHYS-GEN evaluation splits, and Falcon."""
    from .historical_splits import v24 as SV
    from .historical_splits import v25 as SV25
    from .historical_splits import v26 as SV26
    from .historical_splits import v28 as SV28
    h = set(SV28.eval_split_condition_hashes())          # v2.7/v2.8 pilot+development, v2.3 final_test, Falcon
    for conds in (SV.build_development_v24(SV.v24_named_seeds()["split_construction"]),
                  SV25.build_development_v25(SV25.v25_named_seeds()["split_construction"]),
                  SV26.build_development_v26(SV26.v26_named_seeds()["split_construction"])):
        h |= {condition_hash(c) for c in conds}           # no try/except: a failing builder fails closed
    for name in ("pilot_physgen_v1", "development_physgen_v1", "final_physgen_v1"):
        h |= {condition_hash(c) for c in evaluation_split(name)}
    h |= {condition_hash(c) for c in falcon_conditions()}
    return frozenset(h)


def falcon_conditions():
    """Falcon blanketed + unblanketed West and East (hashed / compared only; never scored here)."""
    from .historical_splits import v24_conditions as V24C
    return V24C.falcon_conditions_both_scenarios()


# Atlas I/II/IIA: the sealed external table is NOT in this repository and is never opened, loaded or transformed
# (`evaluation.leakage` sealed guard). It cannot be hashed without opening it; it is protected by construction instead:
# training conditions come only from the synthetic family generator, which has no runtime data path to any vehicle.
ATLAS_I_II_POLICY = "not present in repository; never opened; protected by construction (no runtime path)"


def protected_vehicle_spectra():
    """Shape weights of every protected vehicle spectrum: Falcon blanketed / unblanketed West / East, and every
    reference environment in `physics.environments` (Atlas V reconstruction, Ariane 6, NASA GEVS, Falcon 9 table).
    Read from the constant table directly (outside the sealed guard); weights only, never scored."""
    from ..physics import environments as LE
    out = {c["id"]: np.asarray(c["weights"], float) for c in falcon_conditions()}
    for name, env in LE.ENVIRONMENTS.items():
        out[f"reference:{name}"] = np.asarray(SP.repr_from_spl(env["spl"])["weights"], float)
    return out


def feature_matrix(conds):
    from ..models import linergen as PM
    return PM.condition_features(conds).numpy()


def leakage_report(training, validation, forbidden_hashes, forbidden_features=None, vehicle_spectra=None):
    """Fail-closed leakage barrier. Raises PhysgenConditionError on any failure; returns the report.
    `vehicle_spectra` {name: 8 shape weights}: no training / validation spectrum may equal a protected vehicle
    spectrum (max |dw| <= VEHICLE_WEIGHT_TOL); the minimum distances are reported."""
    ht = [condition_hash(c) for c in training]
    hv = [condition_hash(c) for c in validation]
    checks = dict(unique_training=len(set(ht)) == len(ht), unique_validation=len(set(hv)) == len(hv),
                  training_vs_validation=not (set(ht) & set(hv)),
                  training_vs_forbidden=not (set(ht) & set(forbidden_hashes)),
                  validation_vs_forbidden=not (set(hv) & set(forbidden_hashes)))
    for c in training + validation:
        LEAK.assert_no_sealed({k: v for k, v in c.items() if k != "strata"}, f"condition {c.get('id')}")
        if c.get("group") not in ("training", "training_validation"):
            raise PhysgenConditionError(f"condition {c.get('id')}: group {c.get('group')!r} is not a training group")
    Ft = feature_matrix(training)
    mu, sd = Ft.mean(0), Ft.std(0) + 1e-12
    Ts = (Ft - mu) / sd
    tol = float(NEAR_DUP_FRAC * np.median(LEAK.nn_dist(Ts, Ts, exclude_self=True)))
    dmin = dict(validation_vs_training=float(LEAK.nn_dist((feature_matrix(validation) - mu) / sd, Ts).min()))
    if forbidden_features is not None and len(forbidden_features):
        Fs = (np.asarray(forbidden_features, float) - mu) / sd
        dmin["forbidden_vs_training"] = float(LEAK.nn_dist(Fs, Ts).min())
        dmin["forbidden_vs_validation"] = float(LEAK.nn_dist(Fs, (feature_matrix(validation) - mu) / sd).min())
    for k, v in dmin.items():
        checks[f"no_near_dup_{k}"] = v >= tol
    vmin = {}
    if vehicle_spectra:
        W = np.array([c["weights"] for c in training + validation], float)
        for name, w in vehicle_spectra.items():
            vmin[name] = float(np.abs(W - np.asarray(w, float)).max(1).min())
            checks[f"not_vehicle_spectrum:{name}"] = vmin[name] > VEHICLE_WEIGHT_TOL
    rep = dict(ok=all(checks.values()), checks=checks, near_dup_tol=tol, min_distances=dmin,
               vehicle_min_weight_distance=vmin, atlas_i_ii=ATLAS_I_II_POLICY,
               training_hash=split_hash(training), validation_hash=split_hash(validation),
               n_forbidden=len(forbidden_hashes))
    if not rep["ok"]:
        raise PhysgenConditionError(f"leakage barrier failed: {[k for k, v in checks.items() if not v]}")
    return rep


# =================================================================================================================
#  OOD (sec. 11): Mahalanobis distance in the generator's fixed condition-feature space, fit on TRAINING only
# =================================================================================================================
class OODScorer:
    def __init__(self, training, validation):
        F = feature_matrix(training)
        self.mu = F.mean(0)
        self.prec = np.linalg.pinv(np.cov(F, rowvar=False), rcond=1e-10, hermitian=True)   # weights sum to 1: singular
        dv = self.score(validation)
        self.threshold = float(np.quantile(dv, OOD_QUANTILE))
        self.near = float(np.quantile(dv, NEAR_QUANTILE))
        self.training_hash, self.validation_hash = split_hash(training), split_hash(validation)

    def score(self, conds):
        X = feature_matrix(conds) - self.mu
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", X, self.prec, X), 0.0))

    def classify(self, conds):
        return ["beyond" if d >= self.threshold else "near" if d >= self.near else "in" for d in self.score(conds)]

    def record(self):
        return dict(method="mahalanobis_pinv_condition_features", quantile=OOD_QUANTILE, near_quantile=NEAR_QUANTILE,
                    threshold=self.threshold, near=self.near, near_mix_fraction=NEAR_MIX_FRACTION,
                    training_hash=self.training_hash, validation_hash=self.validation_hash)
