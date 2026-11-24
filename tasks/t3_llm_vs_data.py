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
import sys
import numpy as np
import pandas as pd

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

#part B: ask with data, compare with real figures 

MONTHS_IN_RANGE = 36                                   # Jan 2022 - Dec 2024
T1_TABLE = OUTPUTS / "neighbourhood_month.parquet"     #  Task 1 output


def load_real_averages(use_standin=False):
    """Real average monthly incidents per neighbourhood.

    Divides by 36 months, not by the number of rows, so a month with zero
    incidents (and therefore no row) still counts.
    """
    if use_standin:
        cached = json.loads((SOCRATA_CACHE / "t3_neighbourhood_list.json").read_text(encoding="utf-8"))
        df = pd.DataFrame([
            {"neighbourhood": r["neighborhood"].strip(), "total": int(r["n"])}
            for r in cached["raw"]
            if r.get("neighborhood", "").strip() not in NOT_NEIGHBOURHOODS
        ])
        source = "STAND-IN: Socrata 2022-2024 counts / 36 months (re-run with the Task 1 table before submitting)"
    else:
        t1 = pd.read_parquet(T1_TABLE)
        df = t1.groupby("neighbourhood", as_index=False)["total"].sum()
        source = f"Task 1 table ({T1_TABLE.name})"
    df["real_monthly_avg"] = (df["total"] / MONTHS_IN_RANGE).round(2)
    return df[["neighbourhood", "real_monthly_avg"]], source


def prompt_with_data(real):
    csv_text = real.to_csv(index=False, header=["neighbourhood", "monthly_avg_incidents"])
    return (
        "Below are the actual average monthly reported incidents per Seattle neighbourhood, "
        "2022-2024, from Seattle Police Department data:\n\n"
        + csv_text
        + "\nUsing only these figures, rate every neighbourhood and give its incidents_per_month. "
        f"Return exactly this shape:\n{JSON_SHAPE}"
    )


def spearman(a, b):
    """Spearman rank correlation, written out: Pearson correlation of the ranks (ties averaged)."""
    ra = pd.Series(list(a)).rank(method="average")
    rb = pd.Series(list(b)).rank(method="average")
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def compute_metrics(valid, real):
    llm = pd.DataFrame(valid).rename(columns={
        "name": "neighbourhood", "rating": "llm_rating", "incidents_per_month": "llm_estimate"})
    merged = real.merge(llm, on="neighbourhood", how="inner")
    merged["real_rank"] = merged["real_monthly_avg"].rank(ascending=False, method="min").astype(int)
    merged["pct_error"] = ((merged["llm_estimate"] - merged["real_monthly_avg"]).abs()
                           / merged["real_monthly_avg"] * 100).round(1)

    real_top5 = set(real.nlargest(5, "real_monthly_avg")["neighbourhood"])
    llm_top5 = set(llm.sort_values(["llm_rating", "llm_estimate"], ascending=False)
                   .head(5)["neighbourhood"])

    metrics = {
        "compared": len(merged),
        "spearman_rating": round(spearman(merged["llm_rating"], merged["real_monthly_avg"]), 3),
        "spearman_estimate": round(spearman(merged["llm_estimate"], merged["real_monthly_avg"]), 3),
        "top5_overlap": len(real_top5 & llm_top5),
        "median_pct_error": round(float(merged["pct_error"].median()), 1),
    }
    return metrics, merged.sort_values("real_rank")


def run_compare(use_standin=False):
    names = get_neighbourhoods()
    real, source = load_real_averages(use_standin)
    print(f"Real figures: {source}")
    print("Asking with data:")
    save("with_data", ask_models(prompt_with_data(real), names))

    answers = json.loads(ANSWERS_FILE.read_text(encoding="utf-8"))
    comparison_parts = []
    lines = ["Task 3 metrics", f"Real figures: {source}", f"Generated: {now()}", ""]

    for mode in ["no_data", "with_data"]:
        for provider in MODELS:
            res = answers[mode][provider]
            metrics, merged = compute_metrics(res["valid"], real)
            merged.insert(0, "model", res["model"])
            merged.insert(0, "provider", provider)
            merged.insert(0, "mode", mode)
            comparison_parts.append(merged)

            line = (f"{mode:9} {provider:7} {res['model']:22} "
                    f"spearman(rating)={metrics['spearman_rating']:6} "
                    f"spearman(estimate)={metrics['spearman_estimate']:6} "
                    f"top5={metrics['top5_overlap']}/5 "
                    f"median_error={metrics['median_pct_error']}% "
                    f"skipped={len(res['skipped'])} invented={len(res['invented'])} "
                    f"invalid={len(res['invalid'])}")
            lines.append(line)
            if res["invented"]:
                lines.append(f"{'':18}invented names: {', '.join(res['invented'])}")
            if res["skipped"]:
                lines.append(f"{'':18}skipped names: {', '.join(res['skipped'])}")
            print("  " + line)

    pd.concat(comparison_parts).to_csv(OUTPUTS / "t3_comparison.csv", index=False)
    (OUTPUTS / "t3_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved outputs/t3_comparison.csv and outputs/t3_metrics.txt")

if __name__ == "__main__":
    if "compare" in sys.argv:
        run_compare(use_standin="--standin" in sys.argv)
    else:
        run_no_data()