"""Saves every API response to disk so we only hit each API once.

Xotelo, SerpAPI, Socrata, OSM and Wikipedia all go through get_json here.
Each response lands in cache/<service>/ with the time we got it.
Used by Tasks 1, 4, 6 and 8. The LLM has its own cache in common/llm.py.
"""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"

MIN_GAP_SECONDS = 1.0          # spec wants 1 s between calls to the same api
TIMEOUT_SECONDS = 30
RETRY_WAIT_SECONDS = 3         # give a busy server a moment before the retry
USER_AGENT = "SafeStay-CS4371-Group46"
SECRET_WORDS = ("key", "token", "secret", "password", "auth", "signature")   # param names we never save

_last_call_time = {}                        # when we last called each service
stats = {"api_calls": 0, "cache_hits": 0}   # task 7 counts these, same idea as llm.py


def _is_secret(name):
    return any(word in str(name).lower() for word in SECRET_WORDS)


# drop the keys and sort, so the same search always gives the same file
def _clean_params(params):
    return {k: params[k] for k in sorted(params or {}, key=str) if not _is_secret(k)}


# same url + params -> same file name
def _cache_path(service, url, params):
    key = json.dumps([url, _clean_params(params)], ensure_ascii=False, default=str)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / service / f"{digest}.json"


# wait a bit if we called this service less than a second ago
def _throttle(service):
    last = _last_call_time.get(service)
    if last is not None:
        wait = MIN_GAP_SECONDS - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
    _last_call_time[service] = time.time()


def _fetch(service, url, params, headers):
    all_headers = {"User-Agent": USER_AGENT, **(headers or {})}
    for attempt in range(2):
        _throttle(service)
        stats["api_calls"] += 1
        resp = requests.get(url, params=params, headers=all_headers, timeout=TIMEOUT_SECONDS)
        # 429 and 5xx are usually just a hiccup, so try once more
        if attempt == 0 and (resp.status_code == 429 or resp.status_code >= 500):
            print(f"[cache] {service} returned {resp.status_code}, retrying once")
            time.sleep(RETRY_WAIT_SECONDS)
            continue
        resp.raise_for_status()
        return resp.json()


def _with_meta(data, source, retrieved_at, stale=False):
    meta = {"source": source, "retrieved_at": retrieved_at, "stale": stale}
    if isinstance(data, dict):
        return {**data, "_meta": meta}
    return {"items": data, "_meta": meta}


# get json from the api, or from our saved copy if we already have one
# you always get a dict back with "_meta" = {source, retrieved_at, stale}
# lists come back as {"items": [...], "_meta": ...} so meta still has somewhere to go
# refresh=True skips the saved copy, live=False just means use the cache when we can
# if the api fails but we have an old copy, you get that with stale=True
# headers are never saved, that's where keys usually live
def get_json(service, url, params, live=True, headers=None, refresh=False):
    path = _cache_path(service, url, params)
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    if saved is not None and not refresh:
        stats["cache_hits"] += 1
        return _with_meta(saved["data"], "cached", saved["retrieved_at"])

    try:
        data = _fetch(service, url, params, headers)
    except (requests.RequestException, ValueError) as err:
        # don't print the full error, it can have the url with our key in it
        status = getattr(getattr(err, "response", None), "status_code", None)
        reason = f"{type(err).__name__}" + (f" {status}" if status else "")
        if saved is not None:
            print(f"[cache] {service} call failed ({reason}), using stale copy from {saved['retrieved_at']}")
            stats["cache_hits"] += 1
            return _with_meta(saved["data"], "cached", saved["retrieved_at"], stale=True)
        raise RuntimeError(f"[cache] {service} call failed ({reason}) and nothing is cached for it") from None

    retrieved_at = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "retrieved_at": retrieved_at,
        "url": url,
        "params": _clean_params(params),
        "data": data,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return _with_meta(data, "live", retrieved_at)


# turn a saved time into something readable, plus how many hours old it is
def cache_age(retrieved_at_iso):
    when = datetime.fromisoformat(retrieved_at_iso)
    if when.tzinfo is None:          # no timezone on it, just assume utc
        when = when.replace(tzinfo=timezone.utc)
    when = when.astimezone(timezone.utc)
    age_hours = (datetime.now(timezone.utc) - when).total_seconds() / 3600
    return when.strftime("%Y-%m-%d %H:%M UTC"), round(age_hours, 2)


if __name__ == "__main__":
    out = get_json("wikipedia", "https://en.wikipedia.org/api/rest_v1/page/summary/Space_Needle", {})
    print(out.get("title"), out["_meta"])
    out = get_json("wikipedia", "https://en.wikipedia.org/api/rest_v1/page/summary/Space_Needle", {})
    print("again:", out["_meta"], cache_age(out["_meta"]["retrieved_at"]))
