"""Statistics of the evaluation: paired per-condition differences and the clustered bootstrap (the
condition is the resampling unit; all seeds move together), aggregated per budget.
"""
from __future__ import annotations

import numpy as np
from . import splits as PCN


N_BOOT = 10000


# =================================================================================================================
#  aggregate (clustered bootstrap; condition is the resampling unit, all seeds move together)
# =================================================================================================================
def _diff_table(cells, budget, x, y, kx="best_val", ky="best_val"):
    by = {}
    for c in cells:
        if c["budget"] == budget and c[x][kx] is not None and c[y][ky] is not None:
            by.setdefault(c["cond_index"], {})[c["seed"]] = c[x][kx] - c[y][ky]
    return by


def contrast(cells, budget, x, y, n_boot=N_BOOT, kx="best_val", ky="best_val"):
    by = _diff_table(cells, budget, x, y, kx, ky)
    if not by:
        return None
    ids = sorted(by)
    cm = np.array([np.mean(list(by[i].values())) for i in ids])
    seeds = sorted({s for i in ids for s in by[i]})
    seed_means = {s: float(np.mean([by[i][s] for i in ids if s in by[i]])) for s in seeds}
    rng = np.random.default_rng(PCN.streams()["bootstrap"])
    boot = cm[rng.integers(0, len(cm), size=(n_boot, len(cm)))].mean(1)
    return dict(n_conditions=len(ids), mean=float(cm.mean()), ci95=[float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975))], seed_means=seed_means, min_seed_mean=float(min(seed_means.values())))


def aggregate(cells, budgets):
    out = {}
    for b in budgets:
        cb = [c for c in cells if c["budget"] == b]
        out[str(b)] = dict(
            D_minus_A=contrast(cells, b, "D", "A"), D_minus_B=contrast(cells, b, "D", "B"),
            # generator-only claim, matched at 32 charged calls: best of C's 32 vs best of B's first 32 proposals
            C32_minus_B32=contrast(cells, b, "C", "B", kx="initial32_best_val", ky="initial32_best_val"),
            # labelled diagnostic only (budget-mismatched: 32 calls vs B's full-budget winner)
            diagnostic_C32_minus_Bfinal=contrast(cells, b, "C", "B", kx="initial32_best_val"),
            raw_proposal_feasibility=float(np.mean([c["proposals"]["raw_feasibility"] for c in cb])),
            unique_fraction=float(np.mean([c["proposals"]["unique_fraction"] for c in cb])),
            winner_feasibility={a: float(np.mean([c[a]["feasible"] for c in cb])) for a in "ABCD"},
            winner_mass_compliance={a: float(np.mean([c[a]["mass_ok"] for c in cb])) for a in "ABCD"})
    return out
