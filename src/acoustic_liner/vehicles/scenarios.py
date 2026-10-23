# -*- coding: utf-8 -*-
"""Provenance-classified physical vehicle scenarios (schema `acoustic-liner/vehicle-scenario/1`; the scenario files
of the archived evidence carry the historical identifier, which the reader accepts through the registered mapping).

Every value in a vehicle YAML is a FIELD OBJECT carrying its own machine-readable classification:

    {value, units, class, source, ref, derivation, limitation, label}

    class        one of published | derived | modelling_assumption | study_policy
    published    requires `source` (a key of the YAML `sources:` block) and `ref` (page / table / figure)
    derived      requires `derivation` (the rule); the stored value is RE-DERIVED and must match
    modelling_assumption / study_policy   require `limitation` (why it is assumed / what it does not establish)

Unknown keys, unknown classes, missing provenance and malformed values fail closed (`ScenarioValidationError`, never
`assert`, so `python -O` keeps every gate). The complete published 1/3-octave spectra are the only spectral authority;
the eight 50-250 Hz model bands are SELECTED from them at load time (no stored duplicate) and every corner's published
OASPL must be reproduced by its own energy sum. The scenario hash covers the whole parsed YAML (spectra, frequencies,
OASPL, blanket state, corners, geometry, provenance, payload geometry, assumptions, claim semantics).

Downstream code gets an ordinary `scenarios.PhysicalScenario` (primary corner) from `to_physical_scenario`, so the FEM /
oracle interfaces are unchanged; `bind` / `check_binding` stop Falcon and Atlas records from being mixed.

The older YAMLs (`falcon9_2025_blanketed.yaml`, `falcon9_2025_unblanketed_replacement.yaml`) are NOT edited: their bytes
are pinned by frozen studies (cVAE v2.3-v2.5 manifests, the PHYS-GEN leakage barrier). They used the superseded 13.1 m
representative height; the corrected definitions live in new files.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from ..fem import scenarios as S
from ..provenance import schemas as SC

SCHEMA = SC.schema("vehicle-scenario")
CLASSES = ("published", "derived", "modelling_assumption", "study_policy")
FIELD_KEYS = {"value", "units", "class", "source", "ref", "derivation", "limitation", "label"}
LABELS = ("generic_cross_vehicle_model_assumption", "historical_untraced_assumption")
MODEL_BANDS_HZ = (50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0)
OASPL_TOL_MAX_DB = 1.0         # per-scenario tolerance must stay below the corner separation (0.3 dB Falcon, 3 dB Atlas)
DERIVED_TOL = 1e-9
IN_M = 0.0254
TOP_KEYS = {"schema", "name", "scenario_kind", "is_vehicle", "supersedes", "sources", "geometry", "spectrum",
            "source_conditions", "acoustics", "payload_geometry", "payload_model", "semantics"}
REQUIRED_TOP = TOP_KEYS - {"supersedes", "payload_geometry"}
PAYLOAD_RULES = ("equal_blocked_area_solid", "keep_in_annulus", "fill_fraction_solid")
PAYLOAD_ROLES = ("primary", "sensitivity")
PAYLOAD_SURFACES = {"rigid": (0.0, 0.0)}      # surface condition -> normalized admittance beta (real, imag)
IDENTITY_KEYS = ("scenario_name", "scenario_hash", "source_corner", "payload_case", "payload_geometry_sha256",
                 "payload_bc_sha256")
STAGES = ("finalist", "fem", "cfd_case", "cfd_result", "corrected_fem")
SCENARIO_KINDS = ("published_mpe_representative_vehicle_scenario", "historical_representative_vehicle_scenario")


class ScenarioValidationError(S.ScenarioValidationError):
    pass


class ScenarioMixError(RuntimeError):
    """A record bound to one scenario / source corner was used with another."""


def _err(msg):
    raise ScenarioValidationError(msg)


# =================================================================================================================
#  field objects
# =================================================================================================================
def _field(node, path, sources, numeric=True):
    if not isinstance(node, dict) or "value" not in node or "class" not in node:
        _err(f"{path}: must be a field object with at least 'value' and 'class'")
    extra = set(node) - FIELD_KEYS
    if extra:
        _err(f"{path}: unknown keys {sorted(extra)}")
    c = node["class"]
    if c not in CLASSES:
        _err(f"{path}: class {c!r} not in {CLASSES}")
    if "label" in node and node["label"] not in LABELS:
        _err(f"{path}: label {node['label']!r} not in {LABELS}")
    if c == "published":
        if node.get("source") not in sources:
            _err(f"{path}: published value needs 'source' naming an entry of sources:")
        if not str(node.get("ref", "")).strip():
            _err(f"{path}: published value needs 'ref' (page / table / figure)")
    elif c == "derived":
        if not str(node.get("derivation", "")).strip():
            _err(f"{path}: derived value needs 'derivation'")
    elif not str(node.get("limitation", "")).strip():
        _err(f"{path}: {c} value needs 'limitation'")
    v = node["value"]
    if numeric:
        vals = v if isinstance(v, list) else [v]
        if not vals or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                           for x in vals):
            _err(f"{path}: value must be finite number(s); got {v!r}")
    return v


def _num(d, key, path, sources):
    if key not in d:
        _err(f"{path}.{key}: required field missing")
    return _field(d[key], f"{path}.{key}", sources)


def _txt(d, key, path, sources):
    if key not in d:
        _err(f"{path}.{key}: required field missing")
    v = _field(d[key], f"{path}.{key}", sources, numeric=False)
    if not isinstance(v, str) or not v.strip():
        _err(f"{path}.{key}: value must be non-empty text")
    return v


def _check_derived(d, key, path, expected):
    got = d[key]["value"]
    if d[key]["class"] != "derived" or abs(float(got) - float(expected)) > DERIVED_TOL * max(1.0, abs(expected)):
        _err(f"{path}.{key}: stored {got!r} is not the derived value {expected!r}")


def _closed(d, path, allowed):
    if not isinstance(d, dict):
        _err(f"{path}: must be a mapping")
    extra = set(d) - set(allowed)
    if extra:
        _err(f"{path}: unknown keys {sorted(extra)}")


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def oaspl(levels_dB):
    return 10.0 * math.log10(sum(10.0 ** (float(L) / 10.0) for L in levels_dB))


# =================================================================================================================
#  Figure C-3 axisymmetric equivalent
# =================================================================================================================
def cube_volume_plan_area_in2(R, inner_face, offset_to_tip, half_height, edge_run):
    """Plan-view area (in^2) of one Figure C-3 keep-in volume: a flat inner face of length `inner_face`, two 45-degree
    side edges running `edge_run` outward to the arc end points (+-`half_height`), and an arc of radius `R` reaching
    `offset_to_tip` beyond the inner face, so its centre (the vehicle axis for four volumes per tier) lies
    R - offset_to_tip behind the face. The dimensions over-determine this: the arc end points must lie on R
    (checked by the loader)."""
    c = R - offset_to_tip                          # arc centre to inner face
    xe = edge_run                                  # arc end points relative to the inner face
    # trapezoid between the inner face and the chord through the arc end points
    trap = 0.5 * (inner_face + 2.0 * half_height) * xe
    d = c + xe                                     # centre-to-chord distance
    seg = R * R * math.acos(d / R) - d * math.sqrt(R * R - d * d)
    return trap + seg, c


def axisymmetric_equivalent(pg):
    """Annulus r_in..R_max over the stacked axial extent that preserves (1) blocked cross-sectional area, (2) maximum
    radial extent, (3) axial extent and (4) displaced volume of the published keep-in volumes."""
    dims = {k: pg["figure_c3_dimensions_in"][k]["value"] for k in pg["figure_c3_dimensions_in"]}
    a_vol, axis_to_face = cube_volume_plan_area_in2(dims["arc_radius"], dims["inner_face_length"],
                                                    dims["face_to_arc_tangent"], 0.5 * dims["chord_height"],
                                                    dims["face_to_arc_end"])
    n_per_tier = pg["volumes_per_tier"]["value"]
    tiers = pg["tiers_modelled"]["value"]
    A_b = n_per_tier * a_vol * IN_M ** 2
    R_max = dims["arc_radius"] * IN_M
    L = tiers * dims["tier_axial_extent"] * IN_M
    r_in = math.sqrt(R_max ** 2 - A_b / math.pi)
    return dict(name="axisymmetric_equivalent_of_published_constellation_keep_in_geometry", is_actual_cad=False,
                equations=["A_vol = trapezoid(inner face -> chord) + circular segment(R, chord)",
                           "A_b = volumes_per_tier * A_vol", "R_max = arc_radius", "L = tiers * tier_axial_extent",
                           "r_in = sqrt(R_max^2 - A_b / pi)", "V = A_b * L"],
                plan_area_one_volume_m2=a_vol * IN_M ** 2, axis_to_inner_face_m=axis_to_face * IN_M,
                blocked_area_m2=A_b, R_max_m=R_max, r_in_m=r_in, axial_extent_m=L, displaced_volume_m3=A_b * L,
                limitations=["keep-in envelope is a permitted volume, not the spacecraft outer mould line",
                             "envelope surface is not a payload surface area or acoustic impedance",
                             "axial position of the stack in the fairing is a modelling assumption",
                             "a continuous revolved annulus blocks propagation that the discrete stack does not "
                             "(used by the FEM only as the keep_in_annulus geometry-sensitivity case)"])


# =================================================================================================================
#  load + validate
# =================================================================================================================
class VehicleScenario:
    def __init__(self, raw, path=None):
        self.raw, self.path = raw, path
        self._validate()

    # ---- validation ------------------------------------------------------------------------------------------
    def _validate(self):
        d = self.raw
        _closed(d, "scenario", TOP_KEYS)
        miss = REQUIRED_TOP - set(d)
        if miss:
            _err(f"scenario: missing required blocks {sorted(miss)}")
        try:
            SC.accept_schema(d["schema"], "vehicle-scenario")
        except SC.SchemaError:
            _err(f"schema must be {SCHEMA!r}")
        if d["scenario_kind"] not in SCENARIO_KINDS:
            _err(f"scenario_kind must be one of {SCENARIO_KINDS}")
        if not isinstance(d["is_vehicle"], bool) or not d["is_vehicle"]:
            _err("is_vehicle must be boolean true")
        src = d["sources"]
        _closed(src, "sources", src.keys())
        for k, s in src.items():
            _closed(s, f"sources.{k}", ("title", "revision", "url", "publisher"))
            for f in ("title", "revision", "url"):
                if not str(s.get(f, "")).strip():
                    _err(f"sources.{k}.{f} must be non-empty")
        self.sources = src
        self._geometry(d["geometry"])
        self._spectrum(d["spectrum"], d["source_conditions"])
        self._acoustics(d["acoustics"])
        if "payload_geometry" in d:
            self._payload(d["payload_geometry"])
        self._payload_model(d["payload_model"])
        self._semantics(d["semantics"])

    def _geometry(self, g):
        _closed(g, "geometry", ("outer_diameter_m", "H_total_m", "R_fair_m", "H_barrel_m", "kind", "barrel_nose_rule"))
        D = _num(g, "outer_diameter_m", "geometry", self.sources)
        H = _num(g, "H_total_m", "geometry", self.sources)
        R = _num(g, "R_fair_m", "geometry", self.sources)
        Hb = _num(g, "H_barrel_m", "geometry", self.sources)
        _txt(g, "kind", "geometry", self.sources)
        _txt(g, "barrel_nose_rule", "geometry", self.sources)
        _check_derived(g, "R_fair_m", "geometry", 0.5 * D)
        if g["H_barrel_m"]["class"] != "modelling_assumption":
            _err("geometry.H_barrel_m must be a modelling_assumption (no published interior barrel/nose station)")
        self.geometry = S.FairingGeometry(R_fair=R, H_barrel=Hb, H_total=H,
                                          kind=g["kind"]["value"], provenance=g["barrel_nose_rule"]["value"])

    def _spectrum(self, sp, sc):
        _closed(sp, "spectrum", ("band_hz_full", "pressure_ref_uPa", "model_bands_hz", "oaspl_tolerance_dB"))
        f = _num(sp, "band_hz_full", "spectrum", self.sources)
        _num(sp, "pressure_ref_uPa", "spectrum", self.sources)
        mb = _num(sp, "model_bands_hz", "spectrum", self.sources)
        tol = _num(sp, "oaspl_tolerance_dB", "spectrum", self.sources)
        if sp["oaspl_tolerance_dB"]["class"] != "study_policy" or not 0 < tol < OASPL_TOL_MAX_DB:
            _err(f"spectrum.oaspl_tolerance_dB must be a study_policy in (0, {OASPL_TOL_MAX_DB})")
        if any(b <= a for a, b in zip(f, f[1:])):
            _err("spectrum.band_hz_full must be strictly increasing")
        if tuple(float(x) for x in mb) != MODEL_BANDS_HZ or sp["model_bands_hz"]["class"] != "study_policy":
            _err(f"spectrum.model_bands_hz must be the study_policy bands {MODEL_BANDS_HZ}")
        missing = [b for b in mb if b not in f]
        if missing:
            _err(f"model bands {missing} not in the published frequency grid")
        self.band_hz_full = [float(x) for x in f]
        self.idx = [self.band_hz_full.index(float(b)) for b in mb]
        _closed(sc, "source_conditions", ("primary", "secondary", "blanket_state", "statistic", "fill_factor",
                                         "fill_factor_range", "fill_factor_nominal"))
        self.corners = {}
        for role in ("primary", "secondary"):
            c = sc.get(role)
            _closed(c, f"source_conditions.{role}", ("id", "role_policy", "levels_dB", "oaspl_published_dB"))
            cid = _txt(c, "id", f"source_conditions.{role}", self.sources)
            _txt(c, "role_policy", f"source_conditions.{role}", self.sources)
            if c["role_policy"]["class"] != "study_policy":
                _err(f"source_conditions.{role}.role_policy must be study_policy")
            L = _num(c, "levels_dB", f"source_conditions.{role}", self.sources)
            P = _num(c, "oaspl_published_dB", f"source_conditions.{role}", self.sources)
            if len(L) != len(f):
                _err(f"source_conditions.{role}.levels_dB has {len(L)} values for {len(f)} frequencies")
            if c["levels_dB"]["class"] != "published":
                _err(f"source_conditions.{role}.levels_dB must be published")
            calc = oaspl(L)
            if abs(calc - float(P)) > tol:
                _err(f"source corner {cid!r}: energy sum {calc:.3f} dB does not reproduce published OASPL {P} dB "
                     f"(corner swapped or mistranscribed)")
            self.corners[role] = dict(id=cid, levels_full=[float(x) for x in L], oaspl_published=float(P),
                                      oaspl_computed=calc, model_levels=[float(L[i]) for i in self.idx])
        if self.corners["primary"]["id"] == self.corners["secondary"]["id"]:
            _err("primary and secondary source corners must differ")
        _txt(sc, "blanket_state", "source_conditions", self.sources)
        _txt(sc, "statistic", "source_conditions", self.sources)
        if "fill_factor" in sc:
            self.fill_nominal = _num(sc, "fill_factor", "source_conditions", self.sources)
        else:
            lo, hi = _num(sc, "fill_factor_range", "source_conditions", self.sources)
            self.fill_nominal = _num(sc, "fill_factor_nominal", "source_conditions", self.sources)
            if sc["fill_factor_nominal"]["class"] == "published" or not lo <= self.fill_nominal <= hi:
                _err("fill_factor_nominal must be an assumption inside the published fill_factor_range")

    def _acoustics(self, a):
        keys = ("alpha_bare_nominal", "alpha_bare_range", "A_other_assumed_m2", "kappa_nominal", "kappa_half_width",
                "kappa_eval_range", "Q_fixed", "payload_impedance", "gas_state")
        _closed(a, "acoustics", keys)
        v = {k: (_txt if k in ("payload_impedance", "gas_state") else _num)(a, k, "acoustics", self.sources)
             for k in keys}
        for k in ("alpha_bare_nominal", "alpha_bare_range", "A_other_assumed_m2", "Q_fixed", "payload_impedance",
                  "gas_state"):
            if a[k]["class"] != "modelling_assumption":
                _err(f"acoustics.{k} must be a modelling_assumption (not a vehicle property)")
        if a["kappa_half_width"]["class"] != "study_policy":
            _err("acoustics.kappa_half_width must be study_policy")
        A_wall = self.geometry.lined_wall_area
        _check_derived(a, "kappa_nominal", "acoustics", v["A_other_assumed_m2"] / A_wall)
        k0, hw = v["kappa_nominal"], v["kappa_half_width"]
        er = a["kappa_eval_range"]["value"]
        if a["kappa_eval_range"]["class"] != "derived" or max(abs(er[0] - (k0 - hw)), abs(er[1] - (k0 + hw))) > 1e-12:
            _err("acoustics.kappa_eval_range must be derived as kappa_nominal -+ kappa_half_width")
        self.acoustics = v

    def _payload(self, pg):
        _closed(pg, "payload_geometry", ("primary_figure", "requirement_link", "figure_c3_dimensions_in",
                                         "figure_c3_dimensions_mm", "volumes_per_tier", "tiers_modelled",
                                         "sensitivity_cases", "axisymmetric_equivalent"))
        _txt(pg, "primary_figure", "payload_geometry", self.sources)
        _txt(pg, "requirement_link", "payload_geometry", self.sources)
        din, dmm = pg["figure_c3_dimensions_in"], pg["figure_c3_dimensions_mm"]
        names = ("arc_radius", "chord_height", "inner_face_length", "face_to_arc_end", "face_to_arc_tangent",
                 "tier_axial_extent")
        _closed(din, "figure_c3_dimensions_in", names)
        _closed(dmm, "figure_c3_dimensions_mm", names)
        for n in names:
            vi = _num(din, n, "figure_c3_dimensions_in", self.sources)
            vm = _num(dmm, n, "figure_c3_dimensions_mm", self.sources)
            if din[n]["class"] != "published" or dmm[n]["class"] != "published":
                _err(f"payload_geometry.{n} must be published (transcribed from the figure)")
            if abs(vi * 25.4 - vm) > 0.005 * 25.4 + 0.006:     # inch values are printed rounded to 0.01 in
                _err(f"payload_geometry.{n}: {vi} in and {vm} mm disagree (transcription error)")
        g = {n: din[n]["value"] for n in names}
        d_end = math.hypot(g["arc_radius"] - g["face_to_arc_tangent"] + g["face_to_arc_end"], 0.5 * g["chord_height"])
        if abs(d_end - g["arc_radius"]) > 0.05 or abs(g["chord_height"] - g["inner_face_length"] - 2 * g["face_to_arc_end"]) > 0.05:
            _err(f"Figure C-3 dimensions are not mutually consistent (arc end at {d_end:.3f} in vs R "
                 f"{g['arc_radius']}; 45-degree edges); transcription error")
        _num(pg, "volumes_per_tier", "payload_geometry", self.sources)
        _num(pg, "tiers_modelled", "payload_geometry", self.sources)
        if pg["tiers_modelled"]["class"] != "modelling_assumption":
            _err("payload_geometry.tiers_modelled must be a modelling_assumption")
        _closed(pg.get("sensitivity_cases", {}), "sensitivity_cases", ("figure_c4", "figure_c5"))
        for k, c in pg.get("sensitivity_cases", {}).items():
            _txt(pg["sensitivity_cases"], k, "sensitivity_cases", self.sources)
        eq = axisymmetric_equivalent(pg)
        if pg.get("axisymmetric_equivalent", {}).get("value", {}).get("name") != eq["name"]:
            _err("payload_geometry.axisymmetric_equivalent must be the derived "
                 "'axisymmetric_equivalent_of_published_constellation_keep_in_geometry'")
        node = pg["axisymmetric_equivalent"]
        if node.get("class") != "derived" or not str(node.get("derivation", "")).strip():
            _err("payload_geometry.axisymmetric_equivalent must be derived with its derivation")
        if node["value"].get("is_actual_cad") is not False:
            _err("the axisymmetric equivalent cannot be labelled actual CAD")
        self.payload_equivalent = eq

    def _payload_model(self, pm):
        """FEM payload obstruction cases. Every radius / station is RE-DERIVED from its rule and must match; each
        case must fit inside the cylindrical barrel clear of the source plane; exactly one case is primary."""
        P = "payload_model"
        _closed(pm, P, ("axial_origin", "axial_placement_rule", "axial_extent_fraction_of_barrel", "axial_extent_m",
                        "z_base_m", "surface_condition", "surface_beta", "primary_case", "cases"))
        origin = _txt(pm, "axial_origin", P, self.sources)
        if _txt(pm, "axial_placement_rule", P, self.sources) != "centred_in_barrel":
            _err(f"{P}.axial_placement_rule must be 'centred_in_barrel'")
        g = self.geometry
        L = _num(pm, "axial_extent_m", P, self.sources)
        if hasattr(self, "payload_equivalent"):
            if "axial_extent_fraction_of_barrel" in pm:
                _err(f"{P}: a published keep-in stack fixes the axial extent; no barrel fraction allowed")
            _check_derived(pm, "axial_extent_m", P, self.payload_equivalent["axial_extent_m"])
        else:
            fr = _num(pm, "axial_extent_fraction_of_barrel", P, self.sources)
            if pm["axial_extent_fraction_of_barrel"]["class"] != "modelling_assumption" or not 0 < fr < 1:
                _err(f"{P}.axial_extent_fraction_of_barrel must be a modelling_assumption in (0, 1)")
            _check_derived(pm, "axial_extent_m", P, fr * g.H_barrel)
        zb = _num(pm, "z_base_m", P, self.sources)
        _check_derived(pm, "z_base_m", P, 0.5 * (g.H_barrel - L))
        surf = _txt(pm, "surface_condition", P, self.sources)
        beta = _num(pm, "surface_beta", P, self.sources)
        for k in ("axial_origin", "axial_placement_rule", "surface_condition", "surface_beta"):
            if pm[k]["class"] != "modelling_assumption":
                _err(f"{P}.{k} must be a modelling_assumption (no published payload placement or impedance)")
        if surf not in PAYLOAD_SURFACES or tuple(float(x) for x in beta) != PAYLOAD_SURFACES[surf]:
            _err(f"{P}: surface_condition/surface_beta must be one of {PAYLOAD_SURFACES}")
        primary = _txt(pm, "primary_case", P, self.sources)
        cases = pm["cases"]
        if not isinstance(cases, dict) or not cases:
            _err(f"{P}.cases must be a non-empty mapping")
        bc = dict(scenario_name=self.name, surface_condition=surf, beta_real=float(beta[0]), beta_imag=float(beta[1]))
        self.payload_bc = dict(bc, sha256=_sha(bc))
        self.payload_cases = {}
        fill_range = self.raw["source_conditions"].get("fill_factor_range", {}).get("value")
        for cid, c in cases.items():
            cp = f"{P}.cases.{cid}"
            _closed(c, cp, ("rule", "role", "fill", "r_in_m", "r_out_m"))
            rule, role = _txt(c, "rule", cp, self.sources), _txt(c, "role", cp, self.sources)
            if rule not in PAYLOAD_RULES or c["rule"]["class"] != "study_policy":
                _err(f"{cp}.rule must be a study_policy in {PAYLOAD_RULES}")
            if role not in PAYLOAD_ROLES or c["role"]["class"] != "study_policy":
                _err(f"{cp}.role must be a study_policy in {PAYLOAD_ROLES}")
            r_in, r_out = _num(c, "r_in_m", cp, self.sources), _num(c, "r_out_m", cp, self.sources)
            fill = None
            if rule == "fill_fraction_solid":
                if not fill_range:
                    _err(f"{cp}: fill_fraction_solid needs a published fill_factor_range")
                fill = _num(c, "fill", cp, self.sources)
                if c["fill"]["class"] != "derived" or not fill_range[0] <= fill <= fill_range[1]:
                    _err(f"{cp}.fill must be derived from (inside) the published fill_factor_range {fill_range}")
                exp_in, exp_out = 0.0, g.R_fair * math.sqrt(fill)
            else:
                if not hasattr(self, "payload_equivalent") or "fill" in c:
                    _err(f"{cp}: rule {rule!r} needs the published keep-in payload_geometry and no fill")
                e = self.payload_equivalent
                exp_in, exp_out = ((0.0, math.sqrt(e["blocked_area_m2"] / math.pi)) if rule == "equal_blocked_area_solid"
                                   else (e["r_in_m"], e["R_max_m"]))
            _check_derived(c, "r_in_m", cp, exp_in)
            _check_derived(c, "r_out_m", cp, exp_out)
            if not (0.0 <= r_in < r_out < g.R_fair and 0.0 < zb and zb + L <= g.H_barrel):
                _err(f"{cp}: payload r {r_in}..{r_out} z {zb}..{zb + L} must lie inside the barrel "
                     f"(R {g.R_fair}, H_barrel {g.H_barrel}) clear of the source plane")
            geo = dict(scenario_name=self.name, case=cid, rule=rule, axial_origin=origin,
                       shape="solid_coaxial_cylinder" if r_in == 0 else "coaxial_annulus",
                       r_in_m=float(r_in), r_out_m=float(r_out), z_base_m=float(zb), z_top_m=float(zb + L))
            ring = r_out ** 2 - r_in ** 2
            self.payload_cases[cid] = dict(
                geo, role=role, fill=fill, axial_extent_m=float(L), blocked_area_m2=math.pi * ring,
                displaced_volume_m3=math.pi * ring * L,
                surface_area_m2=2 * math.pi * (r_out + r_in) * L + 2 * math.pi * ring,
                fluid_volume_m3=g.cavity_volume - math.pi * ring * L,
                geometry_sha256=_sha(geo), bc_sha256=self.payload_bc["sha256"])
        roles = [c["role"] for c in self.payload_cases.values()]
        if roles.count("primary") != 1 or self.payload_cases.get(primary, {}).get("role") != "primary":
            _err(f"{P}: exactly one case must be primary and primary_case must name it")
        self.payload_primary = primary

    def _semantics(self, sm):
        allowed = ("id", "metric", "absolute_post_treatment_spl_allowed", "validated_retrofit_claim_allowed",
                   "blankets_modelled", "blanket_replacement_claim_allowed", "alpha_bare_source", "kappa_source",
                   "flight_validation_claim_allowed", "experimental_validation_claim_allowed",
                   "exact_vehicle_claim_allowed", "retrofit_certification_claim_allowed")
        _closed(sm, "semantics", allowed)
        for k in allowed[2:6] + allowed[8:]:
            if sm.get(k) is not False:
                _err(f"semantics.{k} must be false for a representative vehicle scenario")
        self.semantics = S.ClaimSemantics(**{k: sm[k] for k in allowed[:8]})

    # ---- public views ----------------------------------------------------------------------------------------
    @property
    def name(self):
        return self.raw["name"]

    @property
    def scenario_hash(self):
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def model_levels(self, corner_id):
        for c in self.corners.values():
            if c["id"] == corner_id:
                return list(c["model_levels"])
        raise ScenarioMixError(f"{self.name}: no source corner {corner_id!r}")

    def to_physical_scenario(self, corner_id=None):
        """The legacy `PhysicalScenario` view (8 model bands of one source corner; primary by default)."""
        c = self.corners["primary"] if corner_id is None else next(
            (x for x in self.corners.values() if x["id"] == corner_id), None)
        if c is None:
            raise ScenarioMixError(f"{self.name}: no source corner {corner_id!r}")
        sc = self.raw["source_conditions"]
        a = self.acoustics
        src = S.SourceConfig(name=self.name, revision=next(iter(self.sources.values()))["revision"],
                             url=next(iter(self.sources.values()))["url"], blankets=sc["blanket_state"]["value"],
                             fill_factor=float(self.fill_nominal), table_page=sc["statistic"]["value"],
                             range_site=c["id"])
        return S.PhysicalScenario(
            name=f"{self.name}:{c['id']}", geometry=self.geometry, source=src, semantics=self.semantics,
            spl_dB=tuple(c["model_levels"]), oaspl_dB=c["oaspl_published"],
            alpha_bare_nominal=a["alpha_bare_nominal"], alpha_bare_range=tuple(a["alpha_bare_range"]),
            kappa_nominal=a["kappa_nominal"], kappa_eval_range=tuple(a["kappa_eval_range"]),
            pressure_ref_uPa=float(self.raw["spectrum"]["pressure_ref_uPa"]["value"]), Q_fixed=a["Q_fixed"],
            band_hz=MODEL_BANDS_HZ, is_vehicle=True)

    def bind(self, record, corner_id):
        """Stamp a downstream (FEM / CFD) record with this scenario's identity and source corner."""
        self.model_levels(corner_id)
        return dict(record, scenario_name=self.name, scenario_hash=self.scenario_hash, source_corner=corner_id)

    def check_binding(self, record, corner_id=None):
        if record.get("scenario_name") != self.name or record.get("scenario_hash") != self.scenario_hash:
            raise ScenarioMixError(f"record bound to {record.get('scenario_name')!r} "
                                   f"{str(record.get('scenario_hash'))[:12]}, not {self.name!r} {self.scenario_hash[:12]}")
        if corner_id is not None and record.get("source_corner") != corner_id:
            raise ScenarioMixError(f"record source corner {record.get('source_corner')!r} != {corner_id!r}")
        return True

    # ---- pipeline identity (finalist -> FEM -> CFD case -> CFD result -> corrected FEM) -------------------------
    def identity(self, corner_id=None, payload_case=None):
        """The identity every vehicle-stage record carries: scenario, source corner, payload geometry + BC hashes."""
        corner_id = self.corners["primary"]["id"] if corner_id is None else corner_id
        self.model_levels(corner_id)
        case = self.payload_primary if payload_case is None else payload_case
        if case not in self.payload_cases:
            raise ScenarioMixError(f"{self.name}: no payload case {case!r}")
        pc = self.payload_cases[case]
        return dict(scenario_name=self.name, scenario_hash=self.scenario_hash, source_corner=corner_id,
                    payload_case=case, payload_geometry_sha256=pc["geometry_sha256"],
                    payload_bc_sha256=pc["bc_sha256"])

    def require_identity(self, record, corner_id=None, payload_case=None):
        """Fail closed unless `record` carries exactly this scenario's identity (current YAML hash, corner, payload)."""
        missing = [k for k in IDENTITY_KEYS if k not in record]
        if missing:
            raise ScenarioMixError(f"record lacks vehicle identity fields {missing} (legacy or unbound artifact)")
        exp = self.identity(record["source_corner"] if corner_id is None else corner_id,
                            record["payload_case"] if payload_case is None else payload_case)
        bad = {k: (record[k], exp[k]) for k in IDENTITY_KEYS if record[k] != exp[k]}
        if bad:
            raise ScenarioMixError(f"vehicle identity mismatch (stale, cross-vehicle or wrong case): "
                                   + "; ".join(f"{k} {str(a)[:16]!r} != {str(b)[:16]!r}" for k, (a, b) in bad.items()))
        return True


def record_sha256(record):
    return _sha(record)


def stamp(record, identity, stage, parent=None):
    """Bind a stage record to a vehicle identity and (for every stage after the first) to its parent record."""
    if stage not in STAGES:
        raise ScenarioMixError(f"unknown stage {stage!r}; expected one of {STAGES}")
    if set(IDENTITY_KEYS) - set(identity):
        raise ScenarioMixError("identity is incomplete")
    if parent is not None:
        _require_compatible(parent, identity)
    out = dict(record, **{k: identity[k] for k in IDENTITY_KEYS}, stage=stage)
    out["parent_sha256"] = None if parent is None else record_sha256(parent)
    return out


def _require_compatible(parent, identity):
    """A child must carry its parent's identity exactly -- except below a FINALIST, which covers every source corner
    and payload case it lists (`source_corners`, `payload_cases`), so each corner / payload FEM can descend from it."""
    keys = IDENTITY_KEYS
    if parent.get("stage") == "finalist" or ("source_corners" in parent and "payload_cases" in parent):
        keys = ("scenario_name", "scenario_hash")
        if identity["source_corner"] not in parent.get("source_corners", [parent.get("source_corner")]):
            raise ScenarioMixError(f"finalist does not cover source corner {identity['source_corner']!r}")
        pc = parent.get("payload_cases", {parent.get("payload_case"): dict(
            geometry_sha256=parent.get("payload_geometry_sha256"), bc_sha256=parent.get("payload_bc_sha256"))})
        want = pc.get(identity["payload_case"])
        if want != dict(geometry_sha256=identity["payload_geometry_sha256"], bc_sha256=identity["payload_bc_sha256"]):
            raise ScenarioMixError(f"finalist does not cover payload case {identity['payload_case']!r} with these hashes")
    for k in keys:
        if parent.get(k) != identity[k]:
            raise ScenarioMixError(f"parent {parent.get('stage')!r} carries {k} {str(parent.get(k))[:16]!r}, "
                                   f"not {str(identity[k])[:16]!r}")


def verify_lineage(scenario, records):
    """Fail closed unless `records` is a stage-ordered chain, every link carries the CURRENT identity of `scenario`
    and each record names its parent's exact hash."""
    if not records:
        raise ScenarioMixError("empty lineage")
    order = [r.get("stage") for r in records]
    if any(s not in STAGES for s in order) or [STAGES.index(s) for s in order] != sorted(STAGES.index(s) for s in order):
        raise ScenarioMixError(f"stages out of order: {order}")
    for i, r in enumerate(records):
        scenario.require_identity(r)                       # every record: CURRENT scenario hash + its own case
        if i:
            _require_compatible(records[i - 1], {k: r[k] for k in IDENTITY_KEYS})
        want = None if i == 0 else record_sha256(records[i - 1])
        if r.get("parent_sha256") != want:
            raise ScenarioMixError(f"{r.get('stage')!r}: parent hash does not match the preceding record")
    return True


def finalist_coverage(scenario):
    """Fields a finalist record carries so that every source corner / payload case FEM may descend from it."""
    return dict(source_corners=[c["id"] for c in scenario.corners.values()],
                payload_cases={k: dict(geometry_sha256=c["geometry_sha256"], bc_sha256=c["bc_sha256"])
                               for k, c in scenario.payload_cases.items()})


def load(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return VehicleScenario(raw, str(path))


def falcon9_unblanketed():
    from .. import paths
    return load(paths.CONFIGS / "falcon9_2025_unblanketed_v2.yaml")


def atlas_i_14ft():
    from .. import paths
    return load(paths.CONFIGS / "atlas_i_14ft_representative_unblanketed.yaml")
