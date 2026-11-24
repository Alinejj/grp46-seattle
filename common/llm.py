"""LLM client for SafeStay.

One HTTPS request per call, temperature 0, every response cached to disk.
Main model: openai/gpt-oss-120b on Groq. Second model (Task 3): Gemini Flash.
Backup if Groq fails: OpenRouter.
"""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache" / "llm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "openai/gpt-oss-120b",
        "key_env": "GROQ_API_KEY",
        "extra": {"reasoning_effort": "low", "max_completion_tokens": 8192},
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "openai/gpt-oss-120b:free",
        "key_env": "OPENROUTER_API_KEY",
    },
    "gemini": {
        "model": "gemini-3.5-flash",
        "key_env": "GEMINI_API_KEY",
    },
}

MIN_GAP_SECONDS = 1.0
_last_call_time = {}          # provider -> time of last real API call
fallback_events = []          # records every time Groq failed and we used OpenRouter


def _get_key(provider):
    key = os.getenv(PROVIDERS[provider]["key_env"])
    if not key:
        raise RuntimeError(f"{PROVIDERS[provider]['key_env']} is missing from .env")
    return key


def _throttle(provider):
    """Wait so calls to the same provider are at least 1 second apart."""
    last = _last_call_time.get(provider)
    if last is not None:
        wait = MIN_GAP_SECONDS - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
    _last_call_time[provider] = time.time()


def _cache_path(provider, model, system, user):
    """Same provider + model + messages -> same file name."""
    key = json.dumps([provider, model, system, user, 0], ensure_ascii=False)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / f"{provider}_{digest}.json"


def _call_openai_style(provider, system, user):
    """Groq and OpenRouter share the same request format."""
    cfg = PROVIDERS[provider]
    body = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
    }
    body.update(cfg.get("extra", {}))
    headers = {"Authorization": f"Bearer {_get_key(provider)}"}
    resp = requests.post(cfg["url"], headers=headers, json=body, timeout=90)
    if resp.status_code == 429:          # rate limited: wait once and retry
        time.sleep(5)
        resp = requests.post(cfg["url"], headers=headers, json=body, timeout=90)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_gemini(system, user):
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=_get_key("gemini"),
        http_options=types.HttpOptions(timeout=60_000),
    )
    resp = client.models.generate_content(
        model=PROVIDERS["gemini"]["model"],
        contents=user,
        config=types.GenerateContentConfig(system_instruction=system, temperature=0),
    )
    return resp.text


def chat(system, user, provider="groq", use_cache=True):
    """Send one system + user message and return the reply text."""
    model = PROVIDERS[provider]["model"]
    path = _cache_path(provider, model, system, user)

    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["reply"]

    _throttle(provider)
    try:
        if provider == "gemini":
            reply = _call_gemini(system, user)
        else:
            reply = _call_openai_style(provider, system, user)
    except Exception as err:
        if provider == "groq" and os.getenv("OPENROUTER_API_KEY"):
            fallback_events.append({
                "time": datetime.now(timezone.utc).isoformat(),
                "error": str(err)[:200],
            })
            print(f"[llm] Groq failed ({err}); falling back to OpenRouter")
            return chat(system, user, provider="openrouter", use_cache=use_cache)
        raise

    if not reply or not reply.strip():
        raise RuntimeError(f"{provider} returned an empty reply (likely hit the token limit)")
    
    path.write_text(json.dumps({
        "provider": provider,
        "model": model,
        "system": system,
        "user": user,
        "reply": reply,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return reply


def _parse_json(text):
    """Pull a JSON object out of a reply, even if wrapped in ```json fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise json.JSONDecodeError("No JSON object found", cleaned, 0)
    return json.loads(cleaned[start:end + 1])


def chat_json(system, user, required_keys, provider="groq", retries=1):
    """Ask for JSON, check it parses and has the required keys, retry once with the error."""
    prompt = user
    for attempt in range(retries + 1):
        text = chat(system, prompt, provider=provider)
        try:
            data = _parse_json(text)
            missing = [k for k in required_keys if k not in data]
            if not missing:
                return data
            error = f"missing keys {missing}"
        except json.JSONDecodeError as err:
            error = f"invalid JSON ({err.msg})"
        if attempt == retries:
            raise ValueError(f"LLM did not return valid JSON after {retries + 1} tries: {error}")
        prompt = (f"{user}\n\nYour previous reply was rejected: {error}. "
                  "Reply with a single JSON object only, no other text.")


if __name__ == "__main__":
    t0 = time.time()
    print("groq:", chat("You are terse.", "Say hi in exactly 3 words."))
    print(f"  first call {time.time() - t0:.2f}s")

    t0 = time.time()
    print("groq again:", chat("You are terse.", "Say hi in exactly 3 words."))
    print(f"  second call {time.time() - t0:.2f}s (should be ~0, from cache)")

    print("gemini:", chat("You are terse.", "Say hi in exactly 3 words.", provider="gemini"))

    data = chat_json(
        "Return only JSON.",
        'Extract fields from: "Seattle, 3 nights from 2027-07-10, 2 adults, under $180 a night". '
        'Keys: city, check_in, nights, guests, max_price_per_night.',
        required_keys=["city", "check_in", "nights", "guests", "max_price_per_night"],
    )
    print("chat_json:", data)