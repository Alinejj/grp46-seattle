"""Task 4: Seattle hotels from Xotelo, tagged with their MCPP neighbourhood, plus nightly rates.

load_hotels pages through Xotelo's hotel list, get_rates gets every booking site's price.
Everything goes through common/cache.py so a second run doesn't hit Xotelo at all.
search() and nearby() come next and are used by the assistant (Task 5).
"""
from datetime import date, timedelta

import pandas as pd

from common import cache, geo

BASE_URL = "https://data.xotelo.com/api"
LOCATION_KEY = "g60878"          # tripadvisor's id for seattle
PAGE_SIZE = 30
TARGET_HOTELS = 70               # spec wants 50+, a bit over gives search() more to pick from
MIN_HOTELS = 50
MAX_PAGES = 10                   # list has ~300 hotels, never need more than this
MAX_DAYS_AHEAD = 330             # xotelo has nothing past about 11 months

try:
    from common.name_map import to_mcpp       # only exists if incident and mcpp names differ
except ImportError:
    to_mcpp = None


def _neighbourhood(lat, lon):
    name = geo.find_neighbourhood(lat, lon)
    return to_mcpp(name) if (name and to_mcpp) else name


# one row per hotel from a page of api/list, skipping anything without coordinates
def _parse_list(page):
    rows = []
    for h in page.get("list", []):
        spot = h.get("geo") or {}
        lat, lon = spot.get("latitude"), spot.get("longitude")
        if lat is None or lon is None:
            continue
        reviews = h.get("review_summary") or {}
        rows.append({
            "hotel_key": h["key"],
            "hotel": h["name"],
            "type": h.get("accommodation_type"),
            "rating": reviews.get("rating"),
            "reviews": reviews.get("count"),
            "lat": lat,
            "lon": lon,
            "url": h.get("url"),
        })
    return rows


# page through xotelo's seattle list until we have enough hotels with coordinates
def load_hotels(live=True):
    rows, retrieved = [], []
    for page_no in range(MAX_PAGES):
        params = {"location_key": LOCATION_KEY, "limit": PAGE_SIZE, "offset": page_no * PAGE_SIZE}
        out = cache.get_json("xotelo", f"{BASE_URL}/list", params, live=live)
        if out.get("error"):
            print(f"[t4] xotelo list error on page {page_no}: {out['error']}")
            break
        page = out.get("result") or {}
        rows += _parse_list(page)
        retrieved.append(out["_meta"]["retrieved_at"])
        total = page.get("total_count") or 0
        if len(rows) >= TARGET_HOTELS or (page_no + 1) * PAGE_SIZE >= total:
            break

    df = pd.DataFrame(rows).drop_duplicates("hotel_key").reset_index(drop=True)
    if len(df) < MIN_HOTELS:
        print(f"[t4] only {len(df)} hotels with coordinates, spec wants {MIN_HOTELS}+")
    if df.empty:
        return df
    df["neighbourhood"] = [_neighbourhood(lat, lon) for lat, lon in zip(df["lat"], df["lon"])]
    df["list_retrieved_at"] = min(retrieved)    # oldest page, so we never overstate freshness
    return df


def _check_dates(check_in, nights):
    if isinstance(check_in, str):
        try:
            check_in = date.fromisoformat(check_in)
        except ValueError:
            raise ValueError(f"'{check_in}' isn't a date I can read, try YYYY-MM-DD") from None
    nights = int(nights)
    if nights < 1:
        raise ValueError("the stay needs to be at least 1 night")
    today = date.today()
    if check_in <= today:
        raise ValueError(f"check-in {check_in} has already passed, pick a date after {today}")
    if (check_in - today).days > MAX_DAYS_AHEAD:
        last = today + timedelta(days=MAX_DAYS_AHEAD)
        raise ValueError(f"prices only go about {MAX_DAYS_AHEAD} days out, pick a check-in by {last}")
    return check_in, nights


def _empty_rates(hotel_key, check_in, check_out, nights, why, meta=None):
    meta = meta or {}
    return {
        "hotel_key": hotel_key, "check_in": str(check_in), "check_out": str(check_out), "nights": nights,
        "currency": None, "rates": {}, "cheapest_site": None, "cheapest_price": None, "stay_total": None,
        "retrieved_at": meta.get("retrieved_at"), "source": meta.get("source"),
        "stale": meta.get("stale", False), "note": why,
    }


# every site's nightly rate for one hotel and stay, plus the cheapest one
# xotelo does take guests, as `adults` (4 adults left only 1 site vs 4 sites for 1 adult)
# so guests is an optional extra on the end, only sent when given
def get_rates(hotel_key, check_in, nights, live=True, refresh=False, guests=None):
    check_in, nights = _check_dates(check_in, nights)
    check_out = check_in + timedelta(days=nights)
    params = {"hotel_key": hotel_key, "chk_in": check_in.isoformat(), "chk_out": check_out.isoformat()}
    if guests:
        params["adults"] = int(guests)

    try:
        out = cache.get_json("xotelo", f"{BASE_URL}/rates", params, live=live, refresh=refresh)
        # cache.py saves any 200, including xotelo's own error replies, so give a cached error one more go
        if out.get("error") and out["_meta"]["source"] == "cached" and live and not refresh:
            out = cache.get_json("xotelo", f"{BASE_URL}/rates", params, refresh=True)
    except RuntimeError as err:
        print(f"[t4] no rates for {hotel_key}: {err}")
        return _empty_rates(hotel_key, check_in, check_out, nights, "xotelo unreachable and nothing cached")

    meta = out["_meta"]
    if out.get("error"):
        return _empty_rates(hotel_key, check_in, check_out, nights, f"xotelo error: {out['error']}", meta)

    result = out.get("result") or {}
    # rate is per night, tax is always null for seattle so we leave it out
    rates = {r["name"]: float(r["rate"]) for r in result.get("rates") or [] if r.get("rate")}
    if not rates:
        return _empty_rates(hotel_key, check_in, check_out, nights, "no site has a price for these dates", meta)

    cheapest_site = min(rates, key=rates.get)
    return {
        "hotel_key": hotel_key, "check_in": str(check_in), "check_out": str(check_out), "nights": nights,
        "currency": result.get("currency"), "rates": rates,
        "cheapest_site": cheapest_site, "cheapest_price": rates[cheapest_site],
        "stay_total": round(rates[cheapest_site] * nights, 2),
        "retrieved_at": meta["retrieved_at"], "source": meta["source"], "stale": meta["stale"], "note": None,
    }


if __name__ == "__main__":
    hotels = load_hotels()
    print(f"[t4] {len(hotels)} hotels with coordinates")
    print(hotels["neighbourhood"].value_counts(dropna=False).to_string())

    check_in = date.today() + timedelta(days=30)
    for _, h in hotels.head(3).iterrows():
        r = get_rates(h["hotel_key"], check_in, 3)
        print(f"\n[t4] {h['hotel']} ({h['neighbourhood']}) {r['check_in']} -> {r['check_out']}")
        print(f"  rates: {r['rates']}")
        print(f"  cheapest: {r['cheapest_site']} ${r['cheapest_price']}/night, ${r['stay_total']} total, "
              f"{r['source']} at {r['retrieved_at']} {r['note'] or ''}")
