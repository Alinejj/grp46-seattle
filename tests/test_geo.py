"""Tests for common/geo.py.

Made-up shapes check the ray casting, then a few real Seattle spots check the MCPP file.
"""
import pytest

from common import geo

# 10x10 square with a 4x4 hole in the middle, [lon, lat] like geojson
SQUARE_WITH_HOLE = {
    "type": "Polygon",
    "coordinates": [
        [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
        [[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]],
    ],
}

# two separate squares, far apart
TWO_SQUARES = {
    "type": "MultiPolygon",
    "coordinates": [
        [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
        [[[20, 20], [22, 20], [22, 22], [20, 22], [20, 20]]],
    ],
}


# swap the real file for our made-up shapes
@pytest.fixture
def fake_areas(monkeypatch):
    areas = [geo._prepare("HOLEY", SQUARE_WITH_HOLE), geo._prepare("TWINS", TWO_SQUARES)]
    monkeypatch.setattr(geo, "_features", areas)


def test_square_with_hole(fake_areas):
    assert geo.find_neighbourhood(1, 1) == "HOLEY"        # in the square
    assert geo.find_neighbourhood(5, 5) is None           # in the hole
    assert geo.find_neighbourhood(8, 5) == "HOLEY"        # right of the hole, same latitude


def test_multipolygon_any_part(fake_areas):
    assert geo.find_neighbourhood(21, 21) == "TWINS"      # lat, lon order, both 21 anyway
    assert geo.find_neighbourhood(1, 30) is None


def test_outside_everything(fake_areas):
    assert geo.find_neighbourhood(-5, -5) is None
    assert geo.find_neighbourhood(15, 15) is None         # between the two twins


def test_ring_directly():
    ring = SQUARE_WITH_HOLE["coordinates"][0]
    assert geo._point_in_ring(5, 5, ring)
    assert not geo._point_in_ring(11, 5, ring)


def test_haversine_space_needle_to_pike_place():
    km = geo.haversine_km(47.6205, -122.3493, 47.6097, -122.3422)
    assert km == pytest.approx(1.3, abs=0.1)


def test_haversine_same_point_is_zero():
    assert geo.haversine_km(47.6, -122.3, 47.6, -122.3) == 0


# the real mcpp file from here down
@pytest.fixture
def real_areas(monkeypatch):
    monkeypatch.setattr(geo, "_features", None)
    if not geo.GEOJSON_PATH.exists():
        pytest.skip("data/geo/mcpp.geojson not downloaded")


def test_58_names_load(real_areas):
    names = [name for name, _, _ in geo._load()]
    assert len(names) == 58
    assert len(set(names)) == 58


@pytest.mark.parametrize("lat, lon, expected", [
    (47.6205, -122.3493, "QUEEN ANNE"),     # space needle
    (47.6553, -122.3035, "UNIVERSITY"),     # uw
    (47.6097, -122.3422, "BELLTOWN"),       # pike place
    (47.6101, -122.2015, None),             # bellevue, across the lake
])
def test_real_spots(real_areas, lat, lon, expected):
    assert geo.find_neighbourhood(lat, lon) == expected
