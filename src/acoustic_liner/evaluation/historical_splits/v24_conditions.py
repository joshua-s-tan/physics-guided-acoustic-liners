# -*- coding: utf-8 -*-
r"""Historical cVAE v2.4 stage 2: the Falcon conditions of both scenarios
(`falcon_conditions_both_scenarios`), which the leakage barrier (`evaluation.splits`) hashes into the
forbidden condition set. The rest of the historical module (the stage-2 training-condition leakage gates)
is not distributed.
"""

from __future__ import annotations

import numpy as np

from . import selection_rule as SEL
from ... import paths
from ...fem import scenarios as S
from .. import conditions as COND


# --------------------------------------------------------------------------- #
#  Falcon (both scenarios): blanketed (existing) + unblanketed (Table 5-6, West + East)
# --------------------------------------------------------------------------- #
def falcon_conditions_both_scenarios() -> list:
    """Blanketed (registered v2.3 condition) + unblanketed West/East (configs/falcon9_2025_unblanketed_
    replacement.yaml, transcribed-only Table 5-6). Used ONLY for the leakage gate here -- scoring the
    unblanketed scenario is deferred to after the method-lock (owner decision 2026-09-29)."""
    import yaml

    blanketed = COND.falcon_condition()
    yaml_path = paths.CONFIGS / "falcon9_2025_unblanketed_replacement.yaml"
    unblank = S.load_scenario(yaml_path)
    west_spl = np.asarray(unblank.spl_dB, float)
    west = dict(id="falcon9_2025_unblanketed_west", group="falcon_ood_dev",
               weights=list(map(float, SEL._weights_from_spl(west_spl))), spl=west_spl.tolist(),
               dOASPL_target=COND.FALCON_TARGET_DB, m_total_target=COND.FALCON_MASS_TARGET_KG,
               A_wall=float(unblank.A_wall), kappa=float(unblank.kappa_nominal))
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    east_spl = np.asarray(raw["range_spectra"]["east_band_dB"], float)
    east = dict(id="falcon9_2025_unblanketed_east", group="falcon_ood_dev",
               weights=list(map(float, SEL._weights_from_spl(east_spl))), spl=east_spl.tolist(),
               dOASPL_target=COND.FALCON_TARGET_DB, m_total_target=COND.FALCON_MASS_TARGET_KG,
               A_wall=float(unblank.A_wall), kappa=float(unblank.kappa_nominal))
    return [blanketed, west, east]
