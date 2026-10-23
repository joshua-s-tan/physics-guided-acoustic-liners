"""How an operating condition is encoded as the model's conditioning vector.
"""
from __future__ import annotations



def condition_vector(cond):
    return list(map(float, cond["weights"])) + [float(cond["dOASPL_target"]), float(cond["m_total_target"]),
                                                float(cond["A_wall"]), float(cond["kappa"])]
