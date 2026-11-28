"""Tests for common/cache.py.

requests.get is swapped for a fake, so none of this goes online.
"""
import pytest
import requests

from common import cache

URL = "https://example.test/api"


class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload if payload is not None else {"ok": True}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} for url {URL}?key=SECRET", response=self)


# fresh temp cache folder each test, plus fake responses and no real sleeping
@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cache, "_last_call_time", {})
    monkeypatch.setattr(cache, "stats", {"api_calls": 0, "cache_hits": 0})
    sleeps = []
    monkeypatch.setattr(cache.time, "sleep", sleeps.append)
    state = {"responses": [], "calls": []}

    def fake_get(url, params=None, headers=None, timeout=None):
        state["calls"].append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        nxt = state["responses"].pop(0) if state["responses"] else FakeResponse()
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    monkeypatch.setattr(cache.requests, "get", fake_get)
    state["sleeps"] = sleeps
    state["dir"] = tmp_path
    return state


def test_second_call_is_cache_hit(fake):
    fake["responses"] = [FakeResponse(payload={"price": 142})]
    first = cache.get_json("xotelo", URL, {"hotel": "a"})
    second = cache.get_json("xotelo", URL, {"hotel": "a"})
    assert first["_meta"]["source"] == "live"
    assert second["_meta"]["source"] == "cached"
    assert second["price"] == 142
    assert len(fake["calls"]) == 1
    assert cache.stats == {"api_calls": 1, "cache_hits": 1}


def test_param_order_does_not_matter(fake):
    cache.get_json("xotelo", URL, {"a": 1, "b": 2})
    cache.get_json("xotelo", URL, {"b": 2, "a": 1})
    assert len(fake["calls"]) == 1


def test_user_agent_and_timeout_sent(fake):
    cache.get_json("osm", URL, {"q": "x"}, headers={"X-Extra": "1"})
    call = fake["calls"][0]
    assert call["headers"]["User-Agent"] == "SafeStay-CS4371-Group46"
    assert call["headers"]["X-Extra"] == "1"
    assert call["timeout"]


def test_throttle_waits_between_calls_to_same_service(fake, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(cache.time, "time", lambda: clock[0])
    cache.get_json("osm", URL, {"q": 1})
    clock[0] += 0.3
    cache.get_json("osm", URL, {"q": 2})
    assert fake["sleeps"] and fake["sleeps"][-1] == pytest.approx(0.7)


def test_throttle_is_per_service(fake, monkeypatch):
    monkeypatch.setattr(cache.time, "time", lambda: 1000.0)
    cache.get_json("osm", URL, {"q": 1})
    cache.get_json("wikipedia", URL, {"q": 1})
    assert fake["sleeps"] == []


def test_retries_once_on_500(fake):
    fake["responses"] = [FakeResponse(500), FakeResponse(payload={"ok": 1})]
    out = cache.get_json("xotelo", URL, {"h": 1})
    assert out["ok"] == 1
    assert len(fake["calls"]) == 2


def test_failure_falls_back_to_stale_cache(fake):
    fake["responses"] = [FakeResponse(payload={"price": 99})]
    cache.get_json("xotelo", URL, {"h": 1})
    fake["responses"] = [FakeResponse(503), FakeResponse(503)]
    out = cache.get_json("xotelo", URL, {"h": 1}, refresh=True)
    assert out["price"] == 99
    assert out["_meta"]["source"] == "cached"
    assert out["_meta"]["stale"] is True


def test_failure_with_no_cache_raises_clear_error(fake):
    fake["responses"] = [requests.ConnectionError(f"failed {URL}?api_key=SECRET")]
    with pytest.raises(RuntimeError) as err:
        cache.get_json("xotelo", URL, {"h": 1, "api_key": "SECRET"})
    assert "nothing is cached" in str(err.value)
    assert "SECRET" not in str(err.value)


def test_token_param_never_on_disk(fake):
    params = {"q": "seattle", "api_key": "SECRET123", "$$app_token": "SECRET456", "access_token": "SECRET789"}
    cache.get_json("socrata", URL, params, headers={"Authorization": "Bearer SECRETHDR"})
    files = list(fake["dir"].rglob("*"))
    for f in files:
        assert "SECRET" not in f.name
        if f.is_file():
            assert "SECRET" not in f.read_text(encoding="utf-8")
    assert fake["calls"][0]["params"]["api_key"] == "SECRET123"   # the api still gets it, we just don't save it


def test_token_change_hits_same_cache_file(fake):
    cache.get_json("serpapi", URL, {"q": "x", "api_key": "one"})
    cache.get_json("serpapi", URL, {"q": "x", "api_key": "two"})
    assert len(fake["calls"]) == 1


def test_list_response_is_wrapped(fake):
    fake["responses"] = [FakeResponse(payload=[{"a": 1}, {"a": 2}])]
    out = cache.get_json("socrata", URL, {"q": 1})
    assert out["items"] == [{"a": 1}, {"a": 2}]
    assert out["_meta"]["source"] == "live"
    assert cache.get_json("socrata", URL, {"q": 1})["items"] == [{"a": 1}, {"a": 2}]


def test_live_false_still_calls_when_nothing_cached(fake):
    out = cache.get_json("xotelo", URL, {"h": 1}, live=False)
    assert out["_meta"]["source"] == "live"
    assert len(fake["calls"]) == 1


def test_cache_age_format():
    text, hours = cache.cache_age("2026-01-02T03:04:05+00:00")
    assert text == "2026-01-02 03:04 UTC"
    assert isinstance(hours, float) and hours > 0
