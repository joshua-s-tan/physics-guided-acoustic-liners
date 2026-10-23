# -*- coding: utf-8 -*-
"""Launch acoustic environments (design-band 1/3-octave SPL) for SYNTHETIC benchmarks, held-out reference
evaluation, and frozen comparisons.

MIGRATED from `fairing_case/launch_environments.py` (frozen SHA-256
028d037381e8e64053f35b0d3ee7f1d38a1637d4b90e3bc0b8d526c269df856b) for the §14 step-8 self-containment migration
(the historical migration record, which is not distributed).

GUARDRAILS (owner):
  * These tables are NOT a source of truth for the Falcon PHYSICAL scenario. The authoritative Falcon spectrum,
    A_wall, volume and OASPL come from the YAML scenario loader (`fem.scenarios` / `configs/*.yaml`); this
    module may support synthetic benchmarks and frozen comparisons but must NEVER override the scenario. The
    finalist pipeline sources the physical request from the scenario, not from here.
  * BLANKET SEMANTICS: the Falcon spectrum is a fixed, published interior-MPE weighting/forcing input -- NOT a
    bare-fairing spectrum and NOT an absolute post-treatment prediction.
  * NASA GEVS is a QUALIFICATION-SPECTRUM envelope with NO implied NASA geometry.
  * ELECTRON is DORMANT: the historical anchored reconstruction is retained for provenance only and is NOT
    served for use pending an official-source rebuild. Requesting it raises.

Absolute-level caveat (v2 round-9 audit; blocks v2.1, NOT v2.0): the design-band values are 1/3-octave envelope
/ reconstruction levels, NOT an energy-consistent simultaneous spectrum normalized to the listed OASPL. For the
anchored reconstructions the in-band (50-250 Hz) energy sum EXCEEDS the listed total OASPL. v2.0 uses only the
NORMALIZED SHAPE (level-invariant), so this does not affect it; it must be reconciled before any absolute-level
(v2.1) claim.
"""
from __future__ import annotations

import numpy as np

CENTERS_HZ = [50, 63, 80, 100, 125, 160, 200, 250]


class EnvironmentUnavailableError(KeyError):
    """A requested environment is dormant/not served (e.g. Electron pending official-source rebuild)."""


# Active environments (design-band SPL, dB ref 20 uPa). Falcon 9 + NASA GEVS are VERBATIM published tables;
# Atlas V + Ariane 6 are ANCHORED RECONSTRUCTIONS (shape reconstructed to published OASPL/endpoints).
ENVIRONMENTS = {
    "Falcon 9": dict(
        spl=[120.0, 120.0, 119.8, 120.5, 121.5, 122.0, 121.5, 120.5], oaspl=131.4,
        fairing="5.2 m composite", provenance="VERBATIM published 1/3-octave table",
        source="SpaceX Falcon User's Guide (Falcon 9 payload acoustic env., OASPL 131.4 dB; peak ~122 dB @160 Hz).",
        note="mid-band peak (~160 Hz). NOTE: the authoritative Falcon scenario is the YAML loader, not this table."),
    "Ariane 6": dict(
        spl=[131.0, 132.0, 133.0, 133.0, 132.0, 131.0, 129.5, 128.0], oaspl=139.5,
        fairing="5.4 m (long/short)",
        provenance="ANCHORED RECONSTRUCTION (design-band shape reconstructed to published OASPL + endpoints)",
        source="Arianespace Ariane 6 User's Manual (OASPL 139.5 dB; diffuse space-averaged field).",
        note="high level, low-mid peak (~80-100 Hz)."),
    "Atlas V": dict(
        spl=[126.0, 127.5, 129.0, 130.0, 130.0, 129.0, 127.5, 126.0], oaspl=135.0,
        fairing="5 m short PLF",
        provenance="ANCHORED RECONSTRUCTION (design-band shape reconstructed to published MPE characteristics)",
        source="ULA Atlas V Launch Services User's Guide, payload acoustic MPE (1/3-octave).",
        note="low-mid peak (~100-125 Hz), broad."),
    "NASA GEVS envelope": dict(
        spl=[126.0, 127.5, 129.5, 130.5, 132.0, 133.0, 133.5, 134.0], oaspl=144.9,
        fairing="qualification envelope (multi-vehicle)",
        provenance="VERBATIM published 1/3-octave table",
        source="NASA/JPL PD-ED-1259, 'Acoustic Noise Requirement', Table 1 (multi-vehicle envelope, OASPL 144.9 dB).",
        note="QUALIFICATION spectrum (no implied NASA geometry); high-energy broadband, rising across the band."),
}

# DORMANT: retained for provenance only; NOT served for use (owner: do not revive pending official-source rebuild).
DORMANT_ENVIRONMENTS = {
    "Electron": dict(
        spl=[116.0, 118.0, 120.0, 122.0, 124.0, 126.0, 127.0, 127.5], oaspl=129.0, fairing="1.2 m",
        provenance="ANCHORED RECONSTRUCTION (dormant; pending official-source rebuild)",
        source="Rocket Lab Electron Payload User's Guide 7.0 (historical reconstruction).",
        note="DORMANT -- not served; retained for provenance/frozen-comparison only."),
}

# Active evaluation order (Electron excluded -- dormant).
ORDER = ["Falcon 9", "Atlas V", "Ariane 6", "NASA GEVS envelope"]

# In-band energy sum exceeds listed total OASPL for these anchored reconstructions (level caveat above).
LEVEL_INCONSISTENT = ("Atlas V", "Ariane 6")


def spl_array(name: str) -> np.ndarray:
    if name in DORMANT_ENVIRONMENTS and name not in ENVIRONMENTS:
        raise EnvironmentUnavailableError(
            f"environment {name!r} is DORMANT (not served pending official-source rebuild)")
    return np.array(ENVIRONMENTS[name]["spl"], dtype=float)


def inband_oaspl(name: str) -> float:
    """Overall SPL of just the design band (50-250 Hz) for this environment."""
    p2 = 10.0 ** (spl_array(name) / 10.0)
    return float(10.0 * np.log10(p2.sum()))


def level_consistency_audit() -> dict:
    """Per-environment {total_oaspl, inband_oaspl, gap_dB, consistent}. `consistent` means the in-band energy
    sum does NOT exceed the listed all-frequency OASPL (the physical requirement)."""
    out = {}
    for name in ORDER:
        tot = float(ENVIRONMENTS[name]["oaspl"])
        ib = inband_oaspl(name)
        out[name] = dict(total_oaspl=tot, inband_oaspl=ib, gap_dB=ib - tot, consistent=bool(ib <= tot))
    return out
