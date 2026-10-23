# -*- coding: utf-8 -*-
"""Operating-point rerun support rule (vehicle_cfd_rerun.bracket)."""
from acoustic_liner.vehicles import cfd_rerun as R

V, X, F = "valid", "needs_extend", "failed"


def test_adequate_when_neighbours_valid():
    b = R.bracket([("u2", 2.0, V), ("u6", 6.0, V), ("u10.8", 10.8, V), ("u12", 12.0, X)], 5.1, 7.5)
    assert b["status"] == "adequate" and b["needed"] == [] and (b["valid_below"], b["valid_above"]) == (2.0, 10.8)


def test_unconverged_neighbour_and_inside_case_are_needed():
    b = R.bracket([("u2", 2.0, V), ("u6", 6.0, X), ("u11.1", 11.1, X), ("u12", 12.0, V)], 5.7, 8.5)
    assert {n["name"] for n in b["needed"]} == {"u6", "u11.1"} and b["status"] == "needs_rerun"


def test_failed_neighbour_redundant_when_valid_case_within_tolerance():
    b = R.bracket([("u2", 2.0, V), ("u6", 6.0, V), ("u11.6", 11.6, F), ("u12", 12.0, V)], 5.2, 7.4)
    assert b["status"] == "adequate" and b["redundant"][0]["covered_by"] == "u12"


def test_far_valid_case_does_not_cover():
    b = R.bracket([("u2", 2.0, V), ("u6", 6.0, V), ("u12", 12.0, X), ("u20", 20.0, V)], 5.4, 7.8)
    assert [n["name"] for n in b["needed"]] == ["u12"]


def test_missing_side_is_not_adequate():
    b = R.bracket([("u2", 2.0, V), ("u6", 6.0, V)], 6.5, 9.0)
    assert b["missing_side"] == ["above"] and b["status"] == "needs_rerun"
