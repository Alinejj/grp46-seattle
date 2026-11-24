"""Placeholder task functions with FAKE data, in the shapes the assistant expects.

The assistant uses a teammate's real function as soon as it exists in tasks/t2_outlook.py,
tasks/t4_hotels.py, tasks/t6_best_months.py or tasks/t8_places.py, and falls back to these
until then. Every source line from a stub says STUB, so fake numbers can't be mistaken for real ones.
Delete this file once all four real functions are in.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd


def outlook(month):
    """Task 2 shape: 15 rows, 5 per list."""
    rows = []
    for lst, names, base in [
        ("most", ["DOWNTOWN COMMERCIAL", "CAPITOL HILL", "NORTHGATE", "QUEEN ANNE", "SLU/CASCADE"], 420),
        ("largest_rise", ["ALKI", "FREMONT", "BALLARD SOUTH", "UNIVERSITY", "PIONEER SQUARE"], 120),
        ("fewest", ["COMMERCIAL HARBOR ISLAND", "COMMERCIAL DUWAMISH", "PIGEON POINT", "EASTLAKE - EAST", "MADISON PARK"], 6),
    ]:
        for i, name in enumerate(names):
            avg = round(base - i * base * 0.12, 1)
            rows.append({
                "list": lst, "neighbourhood": name, "avg_incidents": avg,
                "share_pct": round(avg / 3000 * 100, 2), "seasonal_ratio": round(1.3 - i * 0.05, 2),
                "persons": round(avg * 0.25, 1), "property": round(avg * 0.6, 1),
            })
    return pd.DataFrame(rows)


def search(check_in, nights, guests, max_price, live=True):
    """Task 4 shape: (hotels table, meta)."""
    hotels = [
        ("Hotel A", "BELLTOWN", 189.0, 175.0, False), ("Hotel B", "QUEEN ANNE", 149.0, 155.0, True),
        ("Hotel C", "FREMONT", 132.0, 128.0, False), ("Hotel D", "DOWNTOWN COMMERCIAL", 210.0, 199.0, True),
        ("Hotel E", "UNIVERSITY", 118.0, 121.0, False),
    ]
    rows = []
    for name, hood, booking, agoda, in_top5 in hotels:
        cheapest = min(booking, agoda)
        if cheapest <= max_price:
            rows.append({
                "hotel": name, "neighbourhood": hood, "booking_com": booking, "agoda": agoda,
                "cheapest_site": "Booking.com" if booking <= agoda else "Agoda",
                "cheapest_price": cheapest, "stay_total": round(cheapest * nights, 2),
                "in_top5_incident_area": in_top5,
            })
    retrieved = datetime.now(timezone.utc) - timedelta(hours=3)
    meta = {"source": "cached", "retrieved_at": retrieved.strftime("%Y-%m-%d %H:%M UTC"), "age_hours": 3.0}
    return pd.DataFrame(rows).sort_values("cheapest_price"), meta


def nearby_by_name(hotel_name, check_in, nights, guests, km=2):
    """Task 4 nearby shape: (hotels within km, meta)."""
    rows = [
        {"hotel": "Hotel C", "distance_km": 0.8, "cheapest_price": 128.0, "neighbourhood": "FREMONT"},
        {"hotel": "Hotel E", "distance_km": 1.6, "cheapest_price": 118.0, "neighbourhood": "UNIVERSITY"},
    ]
    retrieved = datetime.now(timezone.utc) - timedelta(hours=3)
    meta = {"source": "cached", "retrieved_at": retrieved.strftime("%Y-%m-%d %H:%M UTC"), "age_hours": 3.0}
    return pd.DataFrame(rows).sort_values("cheapest_price"), meta


def best_months():
    """Task 6 shape: 12 rows."""
    temps = [42.1, 44.3, 47.8, 51.9, 57.6, 62.4, 67.9, 68.3, 63.1, 54.2, 46.5, 41.8]
    incidents = [3010, 2790, 3105, 3050, 3200, 3150, 3320, 3290, 3120, 3080, 2950, 2900]
    rows = []
    for m, (t, inc) in enumerate(zip(temps, incidents), start=1):
        rows.append({"month": m, "avg_temp_f": t, "distance_from_70": round(abs(70 - t), 1), "avg_incidents": inc})
    df = pd.DataFrame(rows)
    df["score"] = (df["distance_from_70"].rank() + df["avg_incidents"].rank()).round(1)
    df["rank"] = df["score"].rank(method="min").astype(int)
    return df.sort_values("rank")


def places(month, budget=None):
    """Task 8 shape: (top 10 attractions, top 5 neighbourhoods)."""
    attractions = pd.DataFrame([
        {"name": f"Attraction {i}", "neighbourhood": hood, "has_wikipedia": i % 2 == 1,
         "area_avg_incidents": inc, "hotels_within_1km": h, "cheapest_nearby_price": p}
        for i, (hood, inc, h, p) in enumerate([
            ("SLU/CASCADE", 310.5, 6, 145.0), ("PIONEER SQUARE", 99.4, 4, 160.0), ("FREMONT", 123.6, 2, 128.0),
            ("QUEEN ANNE", 374.9, 5, 149.0), ("UNIVERSITY", 298.2, 3, 118.0), ("BALLARD SOUTH", 252.6, 2, 139.0),
            ("ALKI", 39.3, 1, 171.0), ("MAGNOLIA", 102.5, 0, None), ("BELLTOWN", 164.2, 7, 175.0),
            ("CAPITOL HILL", 460.5, 4, 152.0)], start=1)
    ])
    neighbourhoods = pd.DataFrame([
        {"neighbourhood": n, "attractions": a, "avg_incidents": inc, "hotels_in_budget": h, "median_price": p}
        for n, a, inc, h, p in [("FREMONT", 4, 123.6, 3, 131.0), ("PIONEER SQUARE", 5, 99.4, 4, 158.0),
                                ("BALLARD SOUTH", 3, 252.6, 3, 137.0), ("UNIVERSITY", 3, 298.2, 5, 122.0),
                                ("ALKI", 2, 39.3, 1, 171.0)]
    ])
    return attractions, neighbourhoods