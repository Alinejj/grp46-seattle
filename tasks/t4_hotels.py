"""Task 4: Seattle hotels from Xotelo, tagged with their MCPP neighbourhood, priced per booking site.

search() finds hotels under a nightly budget and flags the top 5 reported-incident areas,
nearby() / nearby_by_name() list hotels within 2 km, cheapest first.
Everything goes through common/cache.py, so prices always say when they were retrieved.
Used by the assistant (Task 5) and Task 8.
"""
import difflib
import importlib
import math
import os
import re
from datetime import date, timedelta

import pandas as pd
from dotenv import load_dotenv

from common import cache, geo

BASE_URL = "https://data.xotelo.com/api"
LOCATION_KEY = "g60878"          # tripadvisor's id for seattle
PAGE_SIZE = 30
TARGET_HOTELS = 70               # spec wants 50+, a bit over gives search() more to pick from
MIN_HOTELS = 50
MAX_PAGES = 10                   # list has ~300 hotels, never need more than this
MAX_DAYS_AHEAD = 330             # xotelo has nothing past about 11 months
LIVE_LIMIT = 20                  # live search re-prices this many, keeps the demo around a minute
SERPAPI_URL = "https://serpapi.com/search.json"
DOWN_WARNING = "Xotelo isn't answering, showing cached prices"

last_table = None                # full priced table from the latest search, nearby() reads it
last_meta = None

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
        try:
            out = cache.get_json("xotelo", f"{BASE_URL}/list", params, live=live)
        except RuntimeError as err:
            print(f"[t4] couldn't get hotel list page {page_no}: {err}")
            break
        if out.get("error"):
            print(f"[t4] xotelo list error on page {page_no}: {out['error']}")
            break
        page = out.get("result") or {}
        rows += _parse_list(page)
        retrieved.append(out["_meta"]["retrieved_at"])
        total = page.get("total_count") or 0
        if len(rows) >= TARGET_HOTELS or (page_no + 1) * PAGE_SIZE >= total:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates("hotel_key").reset_index(drop=True)
    if len(df) < MIN_HOTELS:
        print(f"[t4] only {len(df)} hotels with coordinates, spec wants {MIN_HOTELS}+")
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


# "Booking.com" -> "booking_com", so every site is a normal column name
def _site_col(site):
    return re.sub(r"[^a-z0-9]+", "_", site.lower()).strip("_")


# neighbourhoods in member 1's "most reported incidents" list for that month
def _top5(month):
    try:
        outlook = importlib.import_module("tasks.t2_outlook").outlook
        used_stub = False
    except (ImportError, AttributeError):
        outlook = importlib.import_module("tasks.stubs").outlook
        used_stub = True
        print("[t4] USING STUB top 5 (tasks/t2_outlook.py isn't in yet)")
    df = outlook(month)
    if "list" in df.columns:          # stub shape, t2 might differ so fall back to the first 5 rows
        df = df[df["list"].astype(str).str.contains("most")]
    col = "neighbourhood" if "neighbourhood" in df.columns else "neighborhood"
    return list(df[col].head(5)), used_stub


# hotels + their rates -> one row per priced hotel, one column per booking site
def _build_table(hotels, rates_by_key, nights, max_price, top5):
    rows = []
    for _, h in hotels.iterrows():
        r = rates_by_key.get(h["hotel_key"])
        if not r or not r["rates"]:
            continue
        row = {"hotel": h["hotel"], "hotel_key": h["hotel_key"], "lat": h["lat"], "lon": h["lon"],
               "neighbourhood": h["neighbourhood"]}
        row.update({_site_col(site): price for site, price in r["rates"].items()})
        row.update({
            "cheapest_site": r["cheapest_site"], "cheapest_price": r["cheapest_price"],
            "stay_total": round(r["cheapest_price"] * nights, 2),
            "price_retrieved_at": cache.cache_age(r["retrieved_at"])[0],
        })
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["in_budget"] = df["cheapest_price"] <= max_price
    df["in_top5_incident_area"] = df["neighbourhood"].isin(top5)
    fixed = ["hotel", "hotel_key", "lat", "lon", "neighbourhood"]
    tail = ["cheapest_site", "cheapest_price", "stay_total", "in_budget", "in_top5_incident_area",
            "price_retrieved_at"]
    sites = sorted(c for c in df.columns if c not in fixed + tail)
    return df[fixed + sites + tail].sort_values("cheapest_price").reset_index(drop=True)


def _pick(df):
    if df.empty:
        return None
    r = df.iloc[0]
    return {"hotel": r["hotel"], "neighbourhood": r["neighbourhood"], "cheapest_site": r["cheapest_site"],
            "cheapest_price": float(r["cheapest_price"]), "stay_total": float(r["stay_total"])}


# google hotels through serpapi, only when xotelo is down and we've got nothing cached
def _serpapi_rates(check_in, check_out, guests):
    load_dotenv()
    key = os.getenv("SERPAPI_KEY")
    if not key:
        return None, None
    params = {"engine": "google_hotels", "q": "Seattle hotels", "check_in_date": str(check_in),
              "check_out_date": str(check_out), "adults": int(guests or 1), "currency": "USD",
              "gl": "us", "hl": "en", "api_key": key}     # api_key is dropped from the cache file by cache.py
    try:
        out = cache.get_json("serpapi", SERPAPI_URL, params)
    except RuntimeError as err:
        print(f"[t4] serpapi failed too: {err}")
        return None, None
    hotels, rates = [], {}
    for p in out.get("properties") or []:
        spot = p.get("gps_coordinates") or {}
        if spot.get("latitude") is None:
            continue
        hkey = f"serpapi-{p.get('property_token') or p['name']}"
        site_rates = {s["source"]: float(s["rate_per_night"]["extracted_lowest"])
                      for s in p.get("prices") or []
                      if s.get("source") and (s.get("rate_per_night") or {}).get("extracted_lowest")}
        lowest = (p.get("rate_per_night") or {}).get("extracted_lowest")
        if not site_rates and lowest:
            site_rates = {"Google Hotels": float(lowest)}
        if not site_rates:
            continue
        cheapest = min(site_rates, key=site_rates.get)
        hotels.append({"hotel_key": hkey, "hotel": p["name"], "lat": spot["latitude"], "lon": spot["longitude"],
                       "neighbourhood": _neighbourhood(spot["latitude"], spot["longitude"])})
        rates[hkey] = {"rates": site_rates, "cheapest_site": cheapest, "cheapest_price": site_rates[cheapest],
                       "retrieved_at": out["_meta"]["retrieved_at"], "source": "serpapi", "stale": False}
    return pd.DataFrame(hotels), rates


# hotels under max_price a night for the stay, cheapest first, plus a meta dict about the prices
def search(check_in, nights, guests, max_price, live=True):
    global last_table, last_meta
    check_in, nights = _check_dates(check_in, nights)
    check_out = check_in + timedelta(days=nights)
    max_price = math.inf if max_price is None else float(max_price)

    hotels = load_hotels(live=live)
    if live and not hotels.empty:
        hotels = hotels.head(LIVE_LIMIT)
    rates = {}
    for key in hotels.get("hotel_key", []):
        # live re-prices the first 20, otherwise reuse whatever's cached and only fetch the gaps
        rates[key] = get_rates(key, check_in, nights, live=live, refresh=live, guests=guests)

    got = [r for r in rates.values() if r["rates"]]
    failed = [r for r in rates.values() if r["note"] == "xotelo unreachable and nothing cached"]
    stale = [r for r in got if r["stale"]]
    xotelo_down = hotels.empty or (bool(failed or stale) and not any(r["source"] == "live" for r in got))
    warning = DOWN_WARNING if xotelo_down else None

    source = "live" if any(r["source"] == "live" for r in got) else "cached"
    if not got:
        s_hotels, s_rates = _serpapi_rates(check_in, check_out, guests)
        if s_rates:
            hotels, rates, got, source = s_hotels, s_rates, list(s_rates.values()), "serpapi"
            warning = "Xotelo isn't answering and nothing is cached, showing Google Hotels prices via SerpApi"
        elif xotelo_down:
            warning = "Xotelo isn't answering and no prices are cached for these dates"

    top5, top5_stub = _top5(check_in.month)
    table = _build_table(hotels, rates, nights, max_price, top5) if got else pd.DataFrame()

    # oldest price in the table is the honest "retrieved at"
    oldest = min((r["retrieved_at"] for r in got), default=None)
    retrieved_at, age_hours = cache.cache_age(oldest) if oldest else (None, None)

    in_budget = table[table["in_budget"]] if not table.empty else table
    outside = in_budget[~in_budget["in_top5_incident_area"]] if not in_budget.empty else in_budget
    meta = {
        "source": source, "retrieved_at": retrieved_at, "age_hours": age_hours,
        "n_hotels": len(table), "n_in_budget": len(in_budget), "warning": warning,
        "check_in": str(check_in), "check_out": str(check_out), "nights": nights, "guests": guests,
        "max_price": max_price,
        "cheapest_in_budget": _pick(in_budget),
        "cheapest_in_budget_outside_top5": _pick(outside),
        "top5_incident_areas": top5, "top5_from_stub": top5_stub,
        # xotelo prices for the guest count we send, so no per-room caveat needed
        "per_room_note": None if guests else "prices are per room, guest count wasn't given",
    }
    if warning:
        print(f"[t4] {warning}")
    last_table, last_meta = table, meta
    return in_budget.reset_index(drop=True), meta


# other hotels within km of this one, cheapest first, using the latest search prices
def nearby(hotel_key, km=2):
    if last_table is None or last_table.empty:
        raise ValueError("no prices yet, run search() first")
    here = last_table[last_table["hotel_key"] == hotel_key]
    if here.empty:                       # not priced in the last search, still need its location
        hotels = load_hotels(live=False)
        here = hotels[hotels["hotel_key"] == hotel_key] if not hotels.empty else hotels
    if here.empty:
        raise ValueError(f"don't know where hotel {hotel_key} is")
    lat, lon = here.iloc[0]["lat"], here.iloc[0]["lon"]

    others = last_table[last_table["hotel_key"] != hotel_key].copy()
    others["distance_km"] = [round(geo.haversine_km(lat, lon, a, b), 2) for a, b in zip(others["lat"], others["lon"])]
    others = others[others["distance_km"] <= km]
    cols = ["hotel", "hotel_key", "distance_km", "cheapest_price", "cheapest_site", "stay_total",
            "neighbourhood", "in_top5_incident_area", "price_retrieved_at"]
    return others.sort_values(["cheapest_price", "distance_km"])[cols].reset_index(drop=True)


# users type names, not keys: exact, then "contains", then closest spelling
def _match_hotel(name, hotels):
    wanted = name.lower().strip()
    names = list(hotels["hotel"])
    lower = [n.lower() for n in names]
    if wanted in lower:
        return hotels.iloc[lower.index(wanted)]
    contains = [i for i, n in enumerate(lower) if wanted in n]
    if contains:
        return hotels.iloc[min(contains, key=lambda i: len(lower[i]))]    # shortest is usually the one meant
    close = difflib.get_close_matches(wanted, lower, n=1, cutoff=0.6)
    if close:
        return hotels.iloc[lower.index(close[0])]
    guesses = difflib.get_close_matches(wanted, lower, n=5, cutoff=0)
    raise ValueError(f"couldn't find a hotel called '{name}'. Closest: "
                     + ", ".join(names[lower.index(g)] for g in guesses))


# nearby() for a hotel the user named, with prices for their stay
def nearby_by_name(hotel_name, check_in, nights, guests, km=2):
    hotels = load_hotels(live=False)
    if hotels.empty:
        raise ValueError("no hotel list available right now")
    match = _match_hotel(hotel_name, hotels)
    _, meta = search(check_in, nights, guests, max_price=float("inf"), live=False)
    near = nearby(match["hotel_key"], km=km)
    own = last_table[last_table["hotel_key"] == match["hotel_key"]]
    meta = {**meta, "matched_hotel": match["hotel"], "matched_hotel_key": match["hotel_key"],
            "matched_neighbourhood": match["neighbourhood"],
            "matched_cheapest_price": float(own.iloc[0]["cheapest_price"]) if not own.empty else None,
            "radius_km": km}
    return near, meta


if __name__ == "__main__":
    import time

    out_dir = cache.ROOT / "outputs"
    check_in = date.today() + timedelta(days=60)

    t0 = time.time()
    found, meta = search(check_in, 3, 2, 200, live=True)
    print(f"[t4] live search took {time.time() - t0:.0f}s, {meta['n_hotels']} priced, "
          f"{meta['n_in_budget']} under $200, source {meta['source']} at {meta['retrieved_at']}")
    print(found[["hotel", "neighbourhood", "cheapest_site", "cheapest_price", "in_top5_incident_area"]]
          .head(10).to_string(index=False))
    print("cheapest:", meta["cheapest_in_budget"])
    print("cheapest outside top 5:", meta["cheapest_in_budget_outside_top5"])
    last_table.to_csv(out_dir / "t4_hotels.csv", index=False)

    frames = []
    for name in ["Inn at the Market", "paramount"]:
        near, nmeta = nearby_by_name(name, check_in, 3, 2)
        print(f"\n[t4] '{name}' -> {nmeta['matched_hotel']}, {len(near)} hotels within 2 km")
        print(near[["hotel", "distance_km", "cheapest_price", "neighbourhood"]].head(5).to_string(index=False))
        frames.append(near.assign(near_hotel=nmeta["matched_hotel"]))
    nearby_df = pd.concat(frames, ignore_index=True)
    nearby_df[["near_hotel"] + [c for c in nearby_df.columns if c != "near_hotel"]].to_csv(
        out_dir / "t4_nearby.csv", index=False)

    cache.stats.update(api_calls=0, cache_hits=0)
    found, meta = search(check_in, 3, 2, 200, live=False)
    print(f"\n[t4] re-run live=False: source {meta['source']}, retrieved {meta['retrieved_at']}, "
          f"{meta['age_hours']:.2f} hours old, {meta['n_hotels']} priced, cache {cache.stats}")
    print("[t4] wrote outputs/t4_hotels.csv and outputs/t4_nearby.csv")
