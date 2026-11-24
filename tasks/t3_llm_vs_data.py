"""Task 3: what LLMs claim about crime levels in Seattle, checked against the real data.

Part A (now): ask both models with no data attached, validate the JSON.
Part B (after Task 1 is ready): ask again with the monthly averages, compute metrics.

Run from the repo root:  python -m tasks.t3_llm_vs_data
"""
import json
import numbers
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from common.llm import PROVIDERS, chat_json

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
SOCRATA_CACHE = ROOT / "cache" / "socrata"
OUTPUTS.mkdir(exist_ok=True)
SOCRATA_CACHE.mkdir(parents=True, exist_ok=True)

INCIDENTS_URL = "https://data.seattle.gov/resource/tazs-3rd5.json"
NOT_NEIGHBOURHOODS = {"", "-", "UNKNOWN", "OOJ", "FK ERROR"}   # placeholder values in the data
MODELS = ["groq", "gemini"]
ANSWERS_FILE = OUTPUTS / "t3_llm_answers.json"


def now():
    return datetime.now(timezone.utc).isoformat()


#  neighbourhood list 

def get_neighbourhoods():
    """Every real neighbourhood with incidents in 2022-2024 (one Socrata query, cached)."""
    path = SOCRATA_CACHE / "t3_neighbourhood_list.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["neighbourhoods"]

    params = {
        "$select": "neighborhood, count(*) AS n",
        "$where": "offense_date between '2022-01-01T00:00:00' and '2024-12-31T23:59:59'",
        "$group": "neighborhood",
        "$order": "neighborhood",
        "$limit": 500,
    }
    headers = {}
    if os.getenv("SOCRATA_APP_TOKEN"):
        headers["X-App-Token"] = os.getenv("SOCRATA_APP_TOKEN")

    resp = requests.get(INCIDENTS_URL, params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    rows = resp.json()
    names = sorted(
        r["neighborhood"].strip()
        for r in rows
        if r.get("neighborhood", "").strip() not in NOT_NEIGHBOURHOODS
    )
    path.write_text(json.dumps({
        "url": resp.url,
        "retrieved_at": now(),
        "raw": rows,
        "neighbourhoods": names,
    }, indent=2), encoding="utf-8")
    return names


# prompts 

SYSTEM = """You estimate reported police incident levels for Seattle neighbourhoods.
Rules:
- Reply with one JSON object only, no other text.
- Use the neighbourhood names exactly as given, each one exactly once. Do not add, merge or rename any.
- rating is an integer from 1 (fewest reported incidents) to 5 (most reported incidents), relative to the other neighbourhoods in the list.
- incidents_per_month is the average number of incidents reported to the Seattle Police Department per month in that neighbourhood during 2022-2024, as a whole number."""

JSON_SHAPE = '{"neighbourhoods": [{"name": "...", "rating": 1, "incidents_per_month": 0}, ...]}'


def prompt_no_data(names):
    return (
        "Seattle Police Department neighbourhoods:\n"
        + "\n".join(names)
        + f"\n\nRate every neighbourhood above from memory. Return exactly this shape:\n{JSON_SHAPE}"
    )


#validation 

def validate(answer, names):
    """Sort each entry into valid / invalid, and list skipped and invented names."""
    expected = {n.upper(): n for n in names}
    valid, invalid, invented, seen = [], [], [], set()

    for item in answer.get("neighbourhoods", []):
        if not isinstance(item, dict):
            invalid.append({"item": item, "problem": "not an object"})
            continue
        raw_name = str(item.get("name", "")).strip()
        key = raw_name.upper()
        rating, est = item.get("rating"), item.get("incidents_per_month")

        if key not in expected:
            invented.append(raw_name)
            continue
        if key in seen:
            invalid.append({"item": item, "problem": "duplicate"})
            continue
        if (isinstance(rating, bool) or not isinstance(rating, numbers.Real)
                or rating != int(rating) or not 1 <= rating <= 5):
            invalid.append({"item": item, "problem": "rating not an integer 1-5"})
            continue
        if isinstance(est, bool) or not isinstance(est, numbers.Real) or est < 0:
            invalid.append({"item": item, "problem": "incidents_per_month not a number >= 0"})
            continue

        seen.add(key)
        valid.append({"name": expected[key], "rating": int(rating), "incidents_per_month": float(est)})

    skipped = [n for n in names if n.upper() not in seen]
    return {"valid": valid, "skipped": skipped, "invented": invented, "invalid": invalid}


# running and saving

def ask_models(user_prompt, names):
    results = {}
    for provider in MODELS:
        answer = chat_json(SYSTEM, user_prompt, required_keys=["neighbourhoods"], provider=provider)
        checked = validate(answer, names)
        results[provider] = {
            "model": PROVIDERS[provider]["model"],
            "run_at": now(),
            "raw_answer": answer,
            **checked,
        }
        print(f"  {provider:7} {PROVIDERS[provider]['model']:22} "
              f"valid {len(checked['valid']):3}  skipped {len(checked['skipped']):3}  "
              f"invented {len(checked['invented']):3}  invalid {len(checked['invalid']):3}")
    return results


def save(section, results):
    """Merge one section ("no_data" or "with_data") into t3_llm_answers.json."""
    data = json.loads(ANSWERS_FILE.read_text(encoding="utf-8")) if ANSWERS_FILE.exists() else {}
    data[section] = results
    ANSWERS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run_no_data():
    names = get_neighbourhoods()
    print(f"{len(names)} neighbourhoods")
    print("Asking without data:")
    save("no_data", ask_models(prompt_no_data(names), names))
    print(f"Saved to {ANSWERS_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    run_no_data()