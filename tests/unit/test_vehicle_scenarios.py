# -*- coding: utf-8 -*-
"""Falcon 9 (no blankets, corrected) and Atlas I 14-ft representative scenarios: transcription, energy sums,
derived geometry, payload equivalent, provenance classification, fail-closed validation, hashing, identity."""
import copy
import hashlib
import math
import subprocess
import sys

import pytest
import yaml

from acoustic_liner import paths
from acoustic_liner.vehicles import scenarios as V

FALCON_F = [31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
            3150, 4000, 5000, 6300, 8000, 10000]
FALCON_W = [120.5, 121.5, 122.0, 122.5, 123.5, 124.5, 127.7, 129.3, 129.8, 129.5, 128.0, 126.0, 124.0, 122.0, 119.5,
            117.8, 116.4, 114.5, 113.0, 111.3, 110.2, 109.0, 107.5, 106.0, 104.0, 102.0]
FALCON_E = [118.0, 119.5, 120.0, 120.0, 121.0, 123.3] + FALCON_W[6:]
ATLAS_F = [25, 32, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
           3150, 4000, 5000, 6300, 8000, 10000]
ATLAS_Q = [116, 120, 123.5, 124.5, 125.5, 126.5, 127, 128, 128, 128, 127.5, 127, 126, 123.5, 121.5, 119.5, 116.5,
           114.5, 113, 113, 111, 111, 110.5, 110, 110.5, 112.5, 113.5]
ATLAS_A = [x - 3 for x in ATLAS_Q]
OLD_YAML_SHA = {"falcon9_2025_blanketed.yaml": "e6a5d14d8555387edde9086947535a4d65484773662c259a17572af61a7ae9d7",
                "falcon9_2025_unblanketed_replacement.yaml":
                    "f4504d045a41040acddae20643434f391f1dd00818734a6f8ddedccfbea83d41"}


@pytest.fixture(scope="module")
def falcon():
    return V.falcon9_unblanketed()


@pytest.fixture(scope="module")
def atlas():
    return V.atlas_i_14ft()


def raw(name):
    return yaml.safe_load(open(paths.CONFIGS / name, encoding="utf-8"))


FALCON_YAML, ATLAS_YAML = "falcon9_2025_unblanketed_v2.yaml", "atlas_i_14ft_representative_unblanketed.yaml"


# 1-4 Falcon spectra, OASPL, model bands, height
def test_01_falcon_table_5_6_transcription(falcon):
    assert falcon.band_hz_full == FALCON_F
    assert falcon.corners["primary"]["id"] == "west_range" and falcon.corners["primary"]["levels_full"] == FALCON_W
    assert falcon.corners["secondary"]["id"] == "east_range" and falcon.corners["secondary"]["levels_full"] == FALCON_E


def test_02_falcon_published_and_recomputed_oaspl(falcon):
    w, e = falcon.corners["primary"], falcon.corners["secondary"]
    assert (w["oaspl_published"], e["oaspl_published"]) == (137.9, 137.6)
    assert w["oaspl_computed"] == pytest.approx(137.858439568, abs=1e-8)
    assert e["oaspl_computed"] == pytest.approx(137.561530063, abs=1e-8)


def test_03_falcon_model_bands_are_selected_from_full_table(falcon):
    assert falcon.model_levels("west_range") == [122.0, 122.5, 123.5, 124.5, 127.7, 129.3, 129.8, 129.5]
    assert falcon.model_levels("east_range") == [120.0, 120.0, 121.0, 123.3, 127.7, 129.3, 129.8, 129.5]
    assert V.oaspl(falcon.model_levels("west_range")) == pytest.approx(136.157662931, abs=1e-8)
    assert V.oaspl(falcon.model_levels("east_range")) == pytest.approx(135.827352000, abs=1e-8)
    assert "spl_dB" not in str(raw(FALCON_YAML)) and "band_dB" not in str(raw(FALCON_YAML))     # no stored duplicate


def test_04_falcon_height_is_guide_13_2_m(falcon):
    g = raw(FALCON_YAML)["geometry"]
    assert g["H_total_m"]["value"] == 13.2 and g["H_total_m"]["class"] == "published"
    assert falcon.geometry.H_total == 13.2 and falcon.geometry.R_fair == 2.6
    assert "/assets/media/" in raw(FALCON_YAML)["sources"]["falcon_ug"]["url"]


# 5 derived area / volume and kappa
def test_05_falcon_corrected_area_volume_and_kappa(falcon):
    assert falcon.geometry.lined_wall_area == pytest.approx(166.635908435117, abs=1e-9)
    assert falcon.geometry.cavity_volume == pytest.approx(188.302874865967, abs=1e-9)
    a = falcon.acoustics
    assert a["A_other_assumed_m2"] == pytest.approx(38.60083431799, abs=1e-9)
    assert a["kappa_nominal"] == pytest.approx(0.23164775636, abs=1e-11)
    assert a["kappa_eval_range"] == pytest.approx([a["kappa_nominal"] - 0.03, a["kappa_nominal"] + 0.03])
    p = falcon.to_physical_scenario()
    assert p.A_other_nominal == pytest.approx(38.60083431799, abs=1e-9)


# 6-8 payload geometry
def test_06_figure_c3_transcription(falcon):
    pg = raw(FALCON_YAML)["payload_geometry"]
    inch = {k: v["value"] for k, v in pg["figure_c3_dimensions_in"].items()}
    mm = {k: v["value"] for k, v in pg["figure_c3_dimensions_mm"].items()}
    assert inch == dict(arc_radius=87.00, chord_height=119.33, inner_face_length=49.69, face_to_arc_end=34.82,
                        face_to_arc_tangent=58.50, tier_axial_extent=46.00)
    assert mm == dict(arc_radius=2209.80, chord_height=3030.86, inner_face_length=1262.03, face_to_arc_end=884.42,
                      face_to_arc_tangent=1485.90, tier_axial_extent=1168.40)
    assert pg["volumes_per_tier"]["value"] == 4
    assert all("Figure C-3" in v["ref"] for v in pg["figure_c3_dimensions_in"].values())


def test_07_figure_c3_linked_to_no_blanket_mpe():
    pg = raw(FALCON_YAML)["payload_geometry"]
    assert "Figure C-3" in pg["primary_figure"]["value"] and "No Acoustic Blankets" in pg["primary_figure"]["value"]
    assert "Section 5.3.4.2" in pg["requirement_link"]["value"] and "4.1.6" in pg["requirement_link"]["ref"]
    assert set(pg["sensitivity_cases"]) == {"figure_c4", "figure_c5"}


def test_08_axisymmetric_equivalent_records_equations_and_is_not_cad(falcon):
    eq = falcon.payload_equivalent
    assert eq["name"] == "axisymmetric_equivalent_of_published_constellation_keep_in_geometry"
    assert eq["is_actual_cad"] is False and len(eq["equations"]) >= 5 and eq["limitations"]
    assert eq["R_max_m"] == pytest.approx(87 * 0.0254)
    assert math.pi * (eq["R_max_m"] ** 2 - eq["r_in_m"] ** 2) == pytest.approx(eq["blocked_area_m2"])
    assert eq["displaced_volume_m3"] == pytest.approx(eq["blocked_area_m2"] * eq["axial_extent_m"])
    d = raw(FALCON_YAML)
    d["payload_geometry"]["axisymmetric_equivalent"]["value"]["is_actual_cad"] = True
    with pytest.raises(V.ScenarioValidationError, match="CAD"):
        V.VehicleScenario(d)
    d = raw(FALCON_YAML)
    d["payload_geometry"]["figure_c3_dimensions_in"]["face_to_arc_end"]["value"] = 33.82   # inconsistent figure
    d["payload_geometry"]["figure_c3_dimensions_mm"]["face_to_arc_end"]["value"] = 859.02
    with pytest.raises(V.ScenarioValidationError, match="consistent"):
        V.VehicleScenario(d)


# 9-13 Atlas
def test_09_atlas_table_c3_transcription(atlas):
    assert atlas.band_hz_full == ATLAS_F
    assert atlas.corners["primary"]["id"] == "qualification" and atlas.corners["primary"]["levels_full"] == ATLAS_Q
    assert atlas.corners["secondary"]["id"] == "acceptance" and atlas.corners["secondary"]["levels_full"] == ATLAS_A


def test_10_atlas_published_and_recomputed_oaspl(atlas):
    q, a = atlas.corners["primary"], atlas.corners["secondary"]
    assert (q["oaspl_published"], a["oaspl_published"]) == (138, 135)
    assert q["oaspl_computed"] == pytest.approx(137.757931832, abs=1e-8)
    assert a["oaspl_computed"] == pytest.approx(134.757931832, abs=1e-8)


def test_11_atlas_model_bands_selected(atlas):
    assert atlas.model_levels("qualification") == [124.5, 125.5, 126.5, 127.0, 128.0, 128.0, 128.0, 127.5]
    assert atlas.model_levels("acceptance") == [121.5, 122.5, 123.5, 124.0, 125.0, 125.0, 125.0, 124.5]
    assert V.oaspl(atlas.model_levels("qualification")) == pytest.approx(136.063529702, abs=1e-8)
    assert V.oaspl(atlas.model_levels("acceptance")) == pytest.approx(133.063529702, abs=1e-8)


def test_12_atlas_geometry_conversions(atlas):
    g = raw(ATLAS_YAML)["geometry"]
    assert g["H_total_m"]["value"] == pytest.approx(473 * 0.0254) and g["outer_diameter_m"]["value"] == pytest.approx(165 * 0.0254)
    assert atlas.geometry.R_fair == pytest.approx(2.0955) and g["R_fair_m"]["class"] == "derived"
    assert "proxy" in g["R_fair_m"]["limitation"] and "4.3 m / 14 ft" in g["outer_diameter_m"]["ref"]
    assert g["H_barrel_m"]["class"] == "modelling_assumption" and "223.8" in g["H_barrel_m"]["limitation"]
    assert atlas.geometry.lined_wall_area == pytest.approx(118.73848754, abs=1e-7)
    assert atlas.geometry.cavity_volume == pytest.approx(107.524798036, abs=1e-7)


def test_13_atlas_fill_range_preserved(atlas):
    sc = raw(ATLAS_YAML)["source_conditions"]
    assert sc["fill_factor_range"]["value"] == [0.50, 0.60] and sc["fill_factor_range"]["class"] == "published"
    assert sc["fill_factor_nominal"]["value"] == 0.55 and sc["fill_factor_nominal"]["class"] == "modelling_assumption"
    d = raw(ATLAS_YAML)
    d["source_conditions"]["fill_factor_nominal"]["class"] = "published"
    d["source_conditions"]["fill_factor_nominal"].update(source="gevs", ref="x")
    with pytest.raises(V.ScenarioValidationError):
        V.VehicleScenario(d)


# 14-16 classification, fail closed, hashing
def _walk(node, path=""):
    if isinstance(node, dict) and "class" in node and "value" in node:
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}")


@pytest.mark.parametrize("name", [FALCON_YAML, ATLAS_YAML])
def test_14_every_field_is_classified(name):
    d = raw(name)
    fields = list(_walk({k: d[k] for k in ("geometry", "spectrum", "source_conditions", "acoustics",
                                           "payload_geometry") if k in d}))
    assert len(fields) >= 20
    assert {n["class"] for _, n in fields} <= set(V.CLASSES)
    assert {n["class"] for _, n in fields} == set(V.CLASSES)                      # all four classes in use
    atlas_generic = [p for p, n in _walk(raw(ATLAS_YAML)["acoustics"]) if n.get("label")
                     == "generic_cross_vehicle_model_assumption"]
    assert len(atlas_generic) >= 6


@pytest.mark.parametrize("mutate", [
    lambda d: d["geometry"]["H_total_m"].pop("ref"),
    lambda d: d["geometry"]["H_total_m"].update(source="unknown_doc"),
    lambda d: d["geometry"]["H_barrel_m"].pop("limitation"),
    lambda d: d["acoustics"]["kappa_nominal"].pop("derivation"),
    lambda d: d["geometry"]["H_total_m"].update(klass="published"),
    lambda d: d["geometry"]["H_total_m"].update({"class": "measured"}),
    lambda d: d["acoustics"].update(extra_field={"value": 1, "class": "study_policy", "limitation": "x"}),
    lambda d: d.update(unknown_block={}),
    lambda d: d["sources"]["falcon_ug"].update(url=""),
    lambda d: d["geometry"].update(R_fair_m=dict(d["geometry"]["R_fair_m"], value=2.7)),
    lambda d: d["acoustics"].update(kappa_nominal=dict(d["acoustics"]["kappa_nominal"], value=0.2327)),
    lambda d: d["semantics"].update(flight_validation_claim_allowed=True),
    lambda d: d["source_conditions"]["primary"]["levels_dB"]["value"].pop(),
])
def test_15_missing_or_malformed_provenance_fails_closed(mutate):
    d = raw(FALCON_YAML)
    mutate(d)
    with pytest.raises(V.ScenarioValidationError):
        V.VehicleScenario(d)


def test_16_any_field_change_changes_hash(falcon):
    base = falcon.scenario_hash
    for mutate in (lambda d: d["source_conditions"]["secondary"]["levels_dB"]["value"].__setitem__(-1, 102.1),
                   lambda d: d["geometry"]["H_barrel_m"].update(limitation="changed"),
                   lambda d: d["sources"]["falcon_ug"].update(title="changed"),
                   lambda d: d["payload_geometry"]["tiers_modelled"].update(value=4)):
        d = raw(FALCON_YAML)
        mutate(d)
        try:
            h = V.VehicleScenario(d).scenario_hash
        except V.ScenarioValidationError:
            continue                                                               # rejected: also not the same
        assert h != base
    assert V.VehicleScenario(raw(FALCON_YAML)).scenario_hash == base               # deterministic


# 17 python -O
def test_17_loads_under_python_O():
    code = ("from acoustic_liner.vehicles import scenarios as V; f=V.falcon9_unblanketed(); a=V.atlas_i_14ft(); "
            "import copy,yaml; d=copy.deepcopy(f.raw); d['geometry']['H_total_m'].pop('ref')\n"
            "try:\n V.VehicleScenario(d); print('NOGATE')\nexcept V.ScenarioValidationError: print('OK', f.scenario_hash, a.scenario_hash)")
    for flags in ([], ["-O"]):
        out = subprocess.run([sys.executable, *flags, "-c", code], capture_output=True, text=True,
                             env=dict(__import__("os").environ, PYTHONPATH=str(paths.ROOT / "src")))
        assert out.stdout.startswith("OK"), out.stdout + out.stderr


# 18 old frozen artifacts preserved
def test_18_old_yamls_unchanged_and_not_rebound(falcon):
    for name, sha in OLD_YAML_SHA.items():
        b = (paths.CONFIGS / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(b).hexdigest() == sha, f"{name} was modified"
    from acoustic_liner.fem import scenarios as S
    old = S.load_scenario(paths.CONFIGS / "falcon9_2025_unblanketed_replacement.yaml")
    assert old.geometry.H_total == 13.1 and old.name != falcon.name
    assert raw(FALCON_YAML)["supersedes"]["file"] == "configs/falcon9_2025_unblanketed_replacement.yaml"


# 19-20 identity
def test_19_falcon_and_atlas_records_cannot_mix(falcon, atlas):
    rec = falcon.bind(dict(kind="fem", robust_dOASPL=1.0), "west_range")
    assert falcon.check_binding(rec, "west_range")
    with pytest.raises(V.ScenarioMixError):
        atlas.check_binding(rec)
    with pytest.raises(V.ScenarioMixError):
        atlas.bind({}, "west_range")
    with pytest.raises(V.ScenarioMixError):
        falcon.check_binding(dict(rec, scenario_hash=atlas.scenario_hash))
    assert falcon.to_physical_scenario().name != atlas.to_physical_scenario().name


def test_20_source_corners_cannot_be_swapped(falcon, atlas):
    rec = falcon.bind({}, "west_range")
    with pytest.raises(V.ScenarioMixError):
        falcon.check_binding(rec, "east_range")
    for name, a, b in ((FALCON_YAML, "primary", "secondary"), (ATLAS_YAML, "primary", "secondary")):
        d = raw(name)
        sc = d["source_conditions"]
        sc[a]["levels_dB"], sc[b]["levels_dB"] = sc[b]["levels_dB"], sc[a]["levels_dB"]   # arrays swapped, labels kept
        with pytest.raises(V.ScenarioValidationError, match="OASPL"):
            V.VehicleScenario(d)
    assert falcon.to_physical_scenario("east_range").spl_dB != falcon.to_physical_scenario("west_range").spl_dB
