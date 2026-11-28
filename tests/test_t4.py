"""Tests for tasks/t4_hotels.py.

Hotels and rates are made up and swapped in, so nothing calls Xotelo.
"""
from datetime import date, timedelta

import pandas as pd
import pytest

from tasks import t4_hotels as t4

CHECK_IN = date.today() + timedelta(days=30)

# a few hotels around downtown, roughly where they really are
HOTELS = pd.DataFrame([
    {"hotel_key": "h1", "hotel": "Inn at the Market", "lat": 47.6098, "lon": -122.3415, "neighbourhood": "DOWNTOWN COMMERCIAL"},
    {"hotel_key": "h2", "hotel": "The Paramount Hotel", "lat": 47.6132, "lon": -122.3335, "neighbourhood": "DOWNTOWN COMMERCIAL"},
    {"hotel_key": "h3", "hotel": "Mediterranean Inn", "lat": 47.6231, "lon": -122.3569, "neighbourhood": "QUEEN ANNE"},
    {"hotel_key": "h4", "hotel": "Belltown Inn", "lat": 47.6150, "lon": -122.3460, "neighbourhood": "BELLTOWN"},
    {"hotel_key": "h5", "hotel": "University Inn", "lat": 47.6600, "lon": -122.3140, "neighbourhood": "UNIVERSITY"},
])

PRICES = {"h1": {"Booking.com": 260.0, "Agoda.com": 248.0}, "h2": {"Agoda.com": 103.0, "Trip.com": 132.0},
          "h3": {"Agoda.com": 86.0}, "h4": {"Booking.com": 150.0}, "h5": {"Agoda.com": 120.0}}


def _fake_rates(key, check_in, nights, live=True, refresh=False, guests=None):
    rates = PRICES[key]
    site = min(rates, key=rates.get)
    return {"hotel_key": key, "rates": rates, "cheapest_site": site, "cheapest_price": rates[site],
            "stay_total": rates[site] * nights, "retrieved_at": "2026-11-28T01:00:00+00:00",
            "source": "cached", "stale": False, "note": None}


# swap in the made-up hotels, prices and a fixed top 5
@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(t4, "load_hotels", lambda live=True: HOTELS.copy())
    monkeypatch.setattr(t4, "get_rates", _fake_rates)
    monkeypatch.setattr(t4, "_top5", lambda month: (["DOWNTOWN COMMERCIAL", "QUEEN ANNE"], False))
    monkeypatch.setattr(t4, "last_table", None)


def test_budget_filter_and_order(fake):
    found, meta = t4.search(CHECK_IN, 3, 2, 150, live=False)
    assert list(found["hotel_key"]) == ["h3", "h2", "h5", "h4"]     # 248 is over budget, cheapest first
    assert found["in_budget"].all()
    assert meta["n_hotels"] == 5 and meta["n_in_budget"] == 4
    assert meta["source"] == "cached"
    assert meta["cheapest_in_budget"]["hotel"] == "Mediterranean Inn"
    assert found.loc[found["hotel_key"] == "h2", "stay_total"].iloc[0] == 309.0
    assert {"agoda_com", "booking_com", "trip_com"} <= set(found.columns)


def test_outside_top5(fake):
    found, meta = t4.search(CHECK_IN, 3, 2, 150, live=False)
    flagged = dict(zip(found["hotel_key"], found["in_top5_incident_area"]))
    assert flagged == {"h3": True, "h2": True, "h5": False, "h4": False}
    assert meta["cheapest_in_budget_outside_top5"]["hotel"] == "University Inn"


def test_none_budget_means_no_limit(fake):
    found, _ = t4.search(CHECK_IN, 3, 2, None, live=False)
    assert len(found) == 5


def test_nearby_cheapest_first_within_km(fake):
    t4.search(CHECK_IN, 3, 2, 1000, live=False)
    near = t4.nearby("h1", km=2)
    assert "h1" not in set(near["hotel_key"])
    assert "h5" not in set(near["hotel_key"])          # university is ~6 km away
    assert list(near["cheapest_price"]) == sorted(near["cheapest_price"])
    assert list(near["hotel_key"]) == ["h3", "h2", "h4"]
    assert all(round(d, 2) == d for d in near["distance_km"])


def test_nearby_before_search_is_a_clear_error(fake):
    with pytest.raises(ValueError, match="search"):
        t4.nearby("h1")


@pytest.mark.parametrize("typed, expected", [
    ("inn at the market", "Inn at the Market"),     # exact, any case
    ("Paramount", "The Paramount Hotel"),           # substring
    ("Mediteranean Inn", "Mediterranean Inn"),      # typo
])
def test_name_matching(typed, expected):
    assert t4._match_hotel(typed, HOTELS)["hotel"] == expected


def test_no_match_lists_close_names():
    with pytest.raises(ValueError) as err:
        t4._match_hotel("Space Needle Lodge", HOTELS)
    assert "Closest:" in str(err.value)
    assert str(err.value).count(",") >= 4               # 5 names


def test_nearby_by_name(fake):
    near, meta = t4.nearby_by_name("paramount", CHECK_IN, 3, 2)
    assert meta["matched_hotel"] == "The Paramount Hotel"
    assert meta["matched_cheapest_price"] == 103.0
    assert "h2" not in set(near["hotel_key"])


def test_xotelo_down_uses_stale_cache_with_warning(fake, monkeypatch):
    monkeypatch.setattr(t4, "get_rates", lambda *a, **k: {**_fake_rates(*a, **k), "stale": True})
    _, meta = t4.search(CHECK_IN, 3, 2, 200, live=True)
    assert meta["warning"] == t4.DOWN_WARNING
    assert meta["source"] == "cached" and meta["age_hours"] > 0


def test_nothing_cached_falls_back_to_serpapi(fake, monkeypatch):
    dead = {"rates": {}, "note": "xotelo unreachable and nothing cached", "stale": False, "source": None,
            "retrieved_at": None}
    monkeypatch.setattr(t4, "get_rates", lambda *a, **k: dict(dead))
    s_hotels = HOTELS.head(1).copy()
    s_rates = {"h1": {"rates": {"Expedia": 140.0}, "cheapest_site": "Expedia", "cheapest_price": 140.0,
                      "retrieved_at": "2026-11-28T01:00:00+00:00", "source": "serpapi", "stale": False}}
    monkeypatch.setattr(t4, "_serpapi_rates", lambda *a: (s_hotels, s_rates))
    found, meta = t4.search(CHECK_IN, 3, 2, 200, live=True)
    assert meta["source"] == "serpapi"
    assert list(found["expedia"]) == [140.0]


def test_past_date_is_friendly(fake):
    with pytest.raises(ValueError, match="already passed"):
        t4.search(date.today() - timedelta(days=1), 3, 2, 200)
