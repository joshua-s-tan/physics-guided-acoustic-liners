r"""PHYS-GEN-V1 trainer (pre-registration sec. 6-9, 16), Phase I.

LOSS (sec. 7: exactly four categories), per step on `conds_per_step` training conditions x `pop_per_cond` noise draws:
  1. robust physics performance   -w_perf(p) * mean J_train          (soft-coverage objective, node_grad="cap_only",
                                                                       hard worst corner, tau_cov(p) geometric)
  2. residual feasibility / mass  none: every coordinate and every coverage node is feasible / in budget BY
                                   CONSTRUCTION (sec. 4; coefficient 0, nothing left to penalize)
  3. quality-conditioned diversity -w_div * mean_c logdet(K_c + eps I) / |S_c|   over S_c = proposals of condition c
                                   whose detached node-max value is within `div_window_db` of the condition's batch best;
                                   K = RBF kernel on normalized 19-D CMA coordinates (DPP-style repulsion, eigenvalue
                                   floor eps); repels only near-duplicates (bandwidth << typical distance)
  4. numerical regularization     w_sat * mean relu(|u| - sat_u)^2   (keeps u off the sigmoid saturation plateau)
CURRICULUM (sec. 9) over progress p = step / total_steps: (1) p < stabilize: w_perf = w_perf_start (feasibility /
diversity stabilization); (2) p < ramp_end: w_perf ramps linearly to 1; (3) p < sharpen_end: tau_cov sharpens along the
registered geometric schedule `models.objective.tau_cov_schedule`; (4) fixed final weights (w_perf = 1, tau_cov_end).
The worst corner stays HARD (tau_corner = 0) throughout: it is what the end-to-end gradient validation covered.

VALIDATION AND CHECKPOINT SELECTION (sec. 9): only on the training-validation split, only with the authoritative exact
scorer `physics.scorer.quality_eval_exact`, fixed validation noise, at epoch 0 (= Arm B, the untrained generator) and
every `val_every` epochs. Registered lexicographic rule: (1) feasibility >= 0.98 and mass compliance == 1.00;
(2) unique fraction >= 0.95 and the diversity-rank gate; (3) highest mean best-of-population exact value; (4) earliest
epoch on an exact tie.

DETERMINISM / RESUME: float64 CPU, fixed torch thread count, deterministic algorithms; the full state (model, optimizer,
scheduler, numpy RNG states, epoch permutation and position, selection state, complete history) is saved, and resuming
reproduces an uninterrupted run bit for bit. Checkpoints carry a binding (architecture / config / source / split /
scorer hashes, node_grad = cap_only, tau schedule); `load_checkpoint` rejects anything else.

ACCESS: the trainer sees only the training and training-validation splits (leakage barrier enforced at construction)
and runs under `evaluation.leakage.sealed_guard`. It never loads a pilot, development, final, Falcon or Atlas condition.
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from ..evaluation import splits as PCN
from ..design import coordinates as PC
from . import linergen as PM
from . import objective as PO
from ..physics import scorer as PS
from ..optimization import seeded_cma as HO
from ..evaluation import leakage as LEAK
from ..design import canonical as CAN
from .checkpoint import NODE_GRAD
from .checkpoint import SCORER
from .checkpoint import _check_binding
from .checkpoint import binding
from .objective import curriculum
from .checkpoint import model_sha256


NORM_DIM = 19


# ---- r1 pins (Phase I; fixed after conformance smoke only, never chosen on any held-out outcome) ---------------------
CONFIG = dict(
    n_train=PCN.N_TRAIN, n_val=PCN.N_VALIDATION, conds_per_step=16, pop_per_cond=16, epochs=200,
    optimizer="adam", lr=3e-4, betas=(0.9, 0.999), weight_decay=0.0, grad_clip=1.0, warmup_frac=0.02, final_lr_frac=0.1,
    stabilize=0.05, ramp_end=0.30, sharpen_end=0.80, w_perf_start=0.1,
    w_div=0.02, div_window_db=0.5, div_bandwidth=0.05, div_eps=1e-3,
    w_sat=1e-3, sat_u=12.0, tau_corner=0.0, node_grad=NODE_GRAD, scorer=SCORER,
    val_every=5, val_pop=HO.N_PROPOSALS, unique_tol=1e-3, rank_min_median=3.0, feas_min=0.98, mass_min=1.0,
    torch_threads=8)
SMOKE_CONFIG = dict(CONFIG, n_train=32, n_val=8, conds_per_step=8, pop_per_cond=8, epochs=8, val_every=4,
                    val_pop=16)


class TrainingError(RuntimeError):
    """Non-finite loss, broken invariant, or a training gate failure (fail closed)."""


def lr_factor(step, total, cfg):
    w = max(1, int(round(cfg["warmup_frac"] * total)))
    if step < w:
        return (step + 1) / w
    s = int(cfg["sharpen_end"] * total)
    if step < s:
        return 1.0
    q = min(1.0, (step - s) / max(1, total - s))
    return cfg["final_lr_frac"] + (1.0 - cfg["final_lr_frac"]) * 0.5 * (1.0 + math.cos(math.pi * q))


def normalized_design(geom):
    return (geom["cma_vec"] - torch.as_tensor(HO._LO, dtype=PM.DTYPE)) / torch.as_tensor(HO._SPAN, dtype=PM.DTYPE)


def diversity_term(x, q, cfg):
    """x [C, P, 19] normalized designs, q [C, P] detached quality -> mean over conditions of logdet(K_S+eps I)/|S|."""
    vals = []
    for c in range(x.shape[0]):
        keep = q[c] >= q[c].max() - cfg["div_window_db"]
        xs = x[c][keep]
        n = xs.shape[0]
        if n < 2:
            vals.append(torch.zeros((), dtype=PM.DTYPE))
            continue
        d2 = ((xs.unsqueeze(0) - xs.unsqueeze(1)) ** 2).sum(-1) / NORM_DIM
        K = torch.exp(-d2 / (2.0 * cfg["div_bandwidth"] ** 2)) + cfg["div_eps"] * torch.eye(n, dtype=PM.DTYPE)
        vals.append(torch.linalg.slogdet(K)[1] / n)
    return torch.stack(vals).mean()


def loss_terms(model, z, feats, ct, n_cond, pop, p, cfg):
    phase, w_perf, tau = curriculum(p, cfg)
    u = model(z, feats)
    g = PC.decode_t(u)
    out = PO.train_objective(g, ct, tau, tau_corner=float(cfg["tau_corner"]), node_grad=NODE_GRAD)
    perf = out["J"].mean()
    q = out["W_nodes"].detach().max(1).values.view(n_cond, pop)
    div = diversity_term(normalized_design(g).view(n_cond, pop, NORM_DIM), q, cfg)
    sat = torch.relu(u.abs() - cfg["sat_u"]).pow(2).mean()
    loss = -w_perf * perf - cfg["w_div"] * div + cfg["w_sat"] * sat
    return loss, dict(phase=phase, w_perf=w_perf, tau_cov=tau, perf=float(perf.detach()), div=float(div.detach()),
                      sat=float(sat.detach()))


# =================================================================================================================
#  validation (exact scorer only)
# =================================================================================================================
def _score(args):
    design, cond = args
    e = PS.quality_eval_exact(design, cond)
    return dict(pf=bool(e["physically_feasible"]), fb=bool(e["feasible_budget"]), v=e["best_val"], cov=e["best_cov"],
                mass=e["mass"])


def effective_rank(X):
    lam = np.clip(np.linalg.eigvalsh(np.cov(np.asarray(X, float), rowvar=False)), 0.0, None)
    if lam.sum() <= 0:
        return 0.0
    pr = lam / lam.sum(); pr = pr[pr > 0]
    return float(np.exp(-(pr * np.log(pr)).sum()))


def _qd_cells(recs, cond):
    best = max([r["v"] for r in recs if r["fb"]] + [0.0])
    cells = set()
    for r in recs:
        if not r["fb"]:
            continue
        mb = int(np.digitize(r["mass"] / cond["m_total_target"], [0.5, 0.9]))
        cb = min(3, int(r["cov"] * 4))
        pb = 0 if r["v"] <= 0.0 else (1 if r["v"] < 0.5 * best else 2)
        cells.add((mb, cb, pb))
    return len(cells) / 36.0


def validate(model, validation, cfg, pool=None):
    """Generator-only proposals (validation noise stream, identical every call) scored by the exact scorer."""
    rng = np.random.default_rng(PCN.streams()["validation_noise"])
    P = int(cfg["val_pop"])
    with torch.no_grad():
        z = torch.as_tensor(PM.sample_noise(rng, len(validation) * P), dtype=PM.DTYPE)
        u = model(z, PM.condition_features([c for c in validation for _ in range(P)]))
        X = normalized_design(PC.decode_t(u)).numpy()
    designs = HO.designs_from_u(u.numpy())
    jobs = [(d, validation[i // P]) for i, d in enumerate(designs)]
    recs = list(pool.map(_score, jobs, chunksize=16)) if pool is not None else [_score(j) for j in jobs]
    per = []
    for ci, cond in enumerate(validation):
        R = recs[ci * P:(ci + 1) * P]; Xc = X[ci * P:(ci + 1) * P]
        keys = [CAN.geometry_key(d) for d in designs[ci * P:(ci + 1) * P]]
        D2 = np.sqrt(((Xc[:, None] - Xc[None]) ** 2).sum(-1) / NORM_DIM) + np.eye(P) * 1e9
        uniq = np.mean([(D2[i].min() > cfg["unique_tol"]) and keys.count(keys[i]) == 1 for i in range(P)])
        vals = [r["v"] for r in R if r["fb"]]
        winner = max((r for r in R if r["fb"]), key=lambda r: r["v"], default=None)
        per.append(dict(feas=np.mean([r["pf"] for r in R]),
                        mass=np.mean([r["fb"] and r["mass"] <= cond["m_total_target"] for r in R]),
                        winner_ok=winner is not None and winner["mass"] <= cond["m_total_target"],
                        best=max(vals) if vals else float("-inf"), mean=float(np.mean(vals)) if vals else float("-inf"),
                        dead=np.mean([(not r["fb"]) or r["v"] <= 0.0 for r in R]), unique=float(uniq),
                        erank=effective_rank(Xc), mpd=float(np.mean(D2[np.triu_indices(P, 1)])),
                        qd=_qd_cells(R, cond)))
    m = dict(feasibility=float(np.mean([r["feas"] for r in per])), mass_compliance=float(np.mean([r["mass"] for r in per])),
             winner_compliance=float(np.mean([r["winner_ok"] for r in per])),
             perf=float(np.mean([r["best"] for r in per])), mean_value=float(np.mean([r["mean"] for r in per])),
             dead_fraction=float(np.mean([r["dead"] for r in per])), unique_fraction=float(np.mean([r["unique"] for r in per])),
             erank_median=float(np.median([r["erank"] for r in per])), erank_min=float(min(r["erank"] for r in per)),
             mean_pairwise=float(np.mean([r["mpd"] for r in per])), qd_coverage=float(np.mean([r["qd"] for r in per])),
             n_exact_evals=len(jobs))
    m["rank_gate"] = bool(m["erank_median"] >= cfg["rank_min_median"])
    m["gate1"] = bool(m["feasibility"] >= cfg["feas_min"] and m["mass_compliance"] >= cfg["mass_min"])
    m["gate2"] = bool(m["unique_fraction"] >= 0.95 and m["rank_gate"])
    # noise sensitivity (sec. 8): mean |du/dz_j| on the first conditions
    zz = z[: min(len(z), 4 * P)].clone().requires_grad_(True)
    uu = model(zz, PM.condition_features([c for c in validation[:4] for _ in range(P)])[: zz.shape[0]])
    sens = torch.stack([torch.autograd.grad(uu[:, k].sum(), zz, retain_graph=True)[0].abs().mean(0)
                        for k in range(PC.DIM)]).mean(0)
    m["noise_sensitivity_min"] = float(sens.min()); m["noise_sensitivity"] = sens.tolist()
    return m


def selection_key(m, epoch):
    """Registered lexicographic rule (sec. 9); larger is better."""
    return (bool(m["gate1"] and m["gate2"]), float(m["perf"]), -int(epoch))


# =================================================================================================================
#  trainer
# =================================================================================================================
class Trainer:
    def __init__(self, cfg, training, validation, barrier, out_dir, workers=0, _state=None, progress=None):
        """barrier: `evaluation.splits.full_barrier()` dict (definitive run) or a frozenset of forbidden hashes.
        progress: optional callable(dict) called once per epoch and once per validation (wall-clock timing only;
        never stored in the deterministic state)."""
        if cfg["node_grad"] != NODE_GRAD or cfg["scorer"] != SCORER or cfg["tau_corner"] != 0.0:
            raise TrainingError("registered policy violated (node_grad / scorer / tau_corner)")
        if len(training) != cfg["n_train"] or len(validation) != cfg["n_val"]:
            raise TrainingError("split sizes differ from the config")
        if cfg["n_train"] % cfg["conds_per_step"]:
            raise TrainingError("n_train must be a multiple of conds_per_step")
        if not isinstance(barrier, dict):
            barrier = dict(forbidden_hashes=frozenset(barrier))
        self.leakage = PCN.leakage_report(training, validation, **barrier)
        self.cfg, self.training, self.validation, self.out_dir, self.workers = cfg, training, validation, out_dir, workers
        self.progress, self._last_val_s = progress, 0.0
        self.bind = binding(cfg, training, validation)
        torch.set_num_threads(int(cfg["torch_threads"]))
        torch.use_deterministic_algorithms(True)
        self.steps_per_epoch = cfg["n_train"] // cfg["conds_per_step"]
        self.total_steps = self.steps_per_epoch * cfg["epochs"]
        s = PCN.named_seeds()
        self.model = PM.build(s["model_init"])
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]),
                                    weight_decay=cfg["weight_decay"])
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda k: lr_factor(k, self.total_steps, cfg))
        self.feats = PM.condition_features(training)
        if _state is None:
            ss = PCN.streams()["training_noise"].spawn(2)
            self.noise_rng, self.order_rng = np.random.default_rng(ss[0]), np.random.default_rng(ss[1])
            self.step, self.epoch, self.pos, self.perm = 0, 0, 0, None
            self.history, self.validations, self.selected = [], [], None
        else:
            self._restore(_state)
        os.makedirs(out_dir, exist_ok=True)

    # ---- state -------------------------------------------------------------------------------------------------
    def state(self):
        return dict(binding=self.bind, model=self.model.state_dict(), optimizer=self.opt.state_dict(),
                    scheduler=self.sched.state_dict(), noise_rng=self.noise_rng.bit_generator.state,
                    order_rng=self.order_rng.bit_generator.state, step=self.step, epoch=self.epoch, pos=self.pos,
                    perm=None if self.perm is None else self.perm.tolist(), history=self.history,
                    validations=self.validations, selected=self.selected, torch_rng=torch.get_rng_state())

    def _restore(self, st):
        self.model.load_state_dict(st["model"]); self.opt.load_state_dict(st["optimizer"])
        self.sched.load_state_dict(st["scheduler"])
        self.noise_rng, self.order_rng = np.random.default_rng(), np.random.default_rng()
        self.noise_rng.bit_generator.state = st["noise_rng"]; self.order_rng.bit_generator.state = st["order_rng"]
        self.step, self.epoch, self.pos = st["step"], st["epoch"], st["pos"]
        self.perm = None if st["perm"] is None else np.asarray(st["perm"])
        self.history, self.validations, self.selected = st["history"], st["validations"], st["selected"]
        torch.set_rng_state(st["torch_rng"])

    def save(self, name="state.pt"):
        torch.save(self.state(), os.path.join(self.out_dir, name))

    @classmethod
    def resume(cls, out_dir, training, validation, barrier, workers=0, progress=None):
        st = torch.load(os.path.join(out_dir, "state.pt"), weights_only=False)
        _check_binding(st["binding"], training, validation)
        return cls(st["binding"]["config"], training, validation, barrier, out_dir, workers, _state=st,
                   progress=progress)

    # ---- loop --------------------------------------------------------------------------------------------------
    def _emit(self, rec):
        if self.progress is not None:
            self.progress(rec)

    def _validate(self, pool):
        t0 = time.perf_counter()
        m = validate(self.model, self.validation, self.cfg, pool)
        m["epoch"] = self.epoch
        self.validations.append(m)
        self._last_val_s = time.perf_counter() - t0
        self._emit(dict(kind="validation", epoch=self.epoch, seconds=self._last_val_s,
                        **{k: m[k] for k in ("perf", "mean_value", "dead_fraction", "feasibility", "mass_compliance",
                                             "unique_fraction", "erank_median", "gate1", "gate2")}))
        key = selection_key(m, self.epoch)
        if self.selected is None or key > tuple(self.selected["key"]):
            path = os.path.join(self.out_dir, "selected.pt")
            torch.save(dict(binding=self.bind, model=self.model.state_dict(), model_sha256=model_sha256(self.model),
                            epoch=self.epoch, metrics=m, history=self.history), path)
            self.selected = dict(epoch=self.epoch, key=list(key), metrics=m, model_sha256=model_sha256(self.model))

    def run(self, stop_at_step=None):
        cfg, C, P = self.cfg, self.cfg["conds_per_step"], self.cfg["pop_per_cond"]
        pool = ProcessPoolExecutor(max_workers=self.workers) if self.workers else None
        t_run, t_epoch, steps_run, epochs_run, t_train = time.perf_counter(), time.perf_counter(), 0, 0, 0.0
        try:
            with LEAK.sealed_guard():
                if self.step == 0 and not self.validations:
                    self._validate(pool)                                     # epoch 0 = untrained = Arm B
                while self.step < self.total_steps:
                    if stop_at_step is not None and self.step >= stop_at_step:
                        self.save(); return self
                    if self.perm is None:
                        self.perm, self.pos = self.order_rng.permutation(cfg["n_train"]), 0
                    idx = self.perm[self.pos:self.pos + C]
                    conds = [self.training[i] for i in idx for _ in range(P)]
                    z = torch.as_tensor(PM.sample_noise(self.noise_rng, C * P), dtype=PM.DTYPE)
                    feats = self.feats[torch.as_tensor(np.repeat(idx, P))]
                    p = self.step / self.total_steps
                    loss, info = loss_terms(self.model, z, feats, PO.condition_tensors(conds), C, P, p, cfg)
                    if not torch.isfinite(loss):
                        raise TrainingError(f"non-finite loss at step {self.step}")
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    gn = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg["grad_clip"]))
                    if not math.isfinite(gn):
                        raise TrainingError(f"non-finite gradient at step {self.step}")
                    self.opt.step(); self.sched.step()
                    self.history.append(dict(step=self.step, epoch=self.epoch, p=p, loss=float(loss.detach()), grad_norm=gn,
                                             lr=self.opt.param_groups[0]["lr"], **info))
                    self.step += 1; self.pos += C; steps_run += 1
                    if self.pos >= cfg["n_train"]:
                        self.epoch += 1; self.perm = None
                        now = time.perf_counter(); epochs_run += 1; t_train += now - t_epoch
                        ep = self.history[-self.steps_per_epoch:]
                        vals_left = sum(1 for e in range(self.epoch, cfg["epochs"] + 1)
                                        if e % cfg["val_every"] == 0 or e == cfg["epochs"])
                        eta = t_train / steps_run * (self.total_steps - self.step) + vals_left * self._last_val_s
                        self._emit(dict(kind="epoch", epoch=self.epoch, epochs=cfg["epochs"], seconds=now - t_epoch,
                                        elapsed=now - t_run, eta=eta,
                                        phase=info["phase"], tau_cov=info["tau_cov"], w_perf=info["w_perf"],
                                        lr=self.opt.param_groups[0]["lr"],
                                        loss=float(np.mean([h["loss"] for h in ep])),
                                        perf=float(np.mean([h["perf"] for h in ep]))))
                        if self.epoch % cfg["val_every"] == 0 or self.epoch == cfg["epochs"]:
                            self._validate(pool)
                        self.save()
                        t_epoch = time.perf_counter()
                self.save()
                return self
        finally:
            if pool is not None:
                pool.shutdown()
