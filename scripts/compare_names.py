"""Check the SPD incident neighbourhood names line up with the MCPP polygon names.

If they don't, Task 4 and Task 8 can't join a hotel's area to its incident counts.
Writes outputs/name_comparison.txt. Run: python -m scripts.compare_names
"""
import difflib
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))    # so plain `python scripts/compare_names.py` finds common/

from dotenv import load_dotenv

from common import cache, geo

PARQUET = ROOT / "outputs" / "neighbourhood_month.parquet"
OUT_FILE = ROOT / "outputs" / "name_comparison.txt"
SOCRATA_URL = "https://data.seattle.gov/resource/tazs-3rd5.json"
JUNK = {"-", "UNKNOWN", "OOJ", "FK ERROR"}     # from the readme data notes
NEAR_CUTOFF = 0.75                              # difflib ratio, lower just pairs random names


# upper case, trim, and treat runs of spaces and hyphens as one space
def _norm(name):
    return re.sub(r"[\s\-]+", " ", str(name).upper().strip()).strip()


# member 1's parquet if it's there, otherwise ask socrata for the distinct values
def _incident_names():
    if PARQUET.exists():
        import pandas as pd
        df = pd.read_parquet(PARQUET)
        col = "neighbourhood" if "neighbourhood" in df.columns else "neighborhood"
        print(f"[names] using {PARQUET.name}")
        return set(df[col].dropna().astype(str)), f"{PARQUET.name} column {col}"

    load_dotenv()
    token = os.getenv("SOCRATA_APP_TOKEN")
    params = {"$select": "neighborhood", "$group": "neighborhood", "$limit": 500}
    out = cache.get_json("socrata", SOCRATA_URL, params,
                         headers={"X-App-Token": token} if token else None)
    print(f"[names] parquet not there yet, using socrata ({out['_meta']['source']})")
    names = {row["neighborhood"] for row in out["items"] if row.get("neighborhood")}
    return names, f"Socrata tazs-3rd5 distinct neighborhood, retrieved {out['_meta']['retrieved_at']}"


def compare():
    mcpp = {name for name, _, _ in geo._load()}
    raw_incident, where = _incident_names()
    incident = {n for n in raw_incident if _norm(n) not in JUNK and _norm(n)}
    dropped = sorted(raw_incident - incident)

    mcpp_by_norm = {_norm(n): n for n in mcpp}
    inc_by_norm = {_norm(n): n for n in incident}

    exact = sorted(k for k in inc_by_norm if k in mcpp_by_norm)
    left_inc = sorted(k for k in inc_by_norm if k not in mcpp_by_norm)
    left_mcpp = sorted(k for k in mcpp_by_norm if k not in inc_by_norm)

    # pair up leftovers by similarity, each mcpp name used once
    near, taken = [], set()
    for k in left_inc:
        options = [m for m in left_mcpp if m not in taken]
        best = difflib.get_close_matches(k, options, n=1, cutoff=NEAR_CUTOFF)
        if best:
            ratio = difflib.SequenceMatcher(None, k, best[0]).ratio()
            near.append((inc_by_norm[k], mcpp_by_norm[best[0]], ratio))
            taken.add(best[0])
    near_inc = {_norm(a) for a, _, _ in near}
    only_inc = [inc_by_norm[k] for k in left_inc if k not in near_inc]
    only_mcpp = [mcpp_by_norm[k] for k in left_mcpp if k not in taken]

    # same after normalising but spelled differently, still needs a map entry
    spelled_diff = [(inc_by_norm[k], mcpp_by_norm[k]) for k in exact if inc_by_norm[k] != mcpp_by_norm[k]]

    lines = [
        "MCPP polygon names vs incident neighbourhood names",
        f"incident names from: {where}",
        f"mcpp names from: {geo.GEOJSON_PATH.relative_to(ROOT).as_posix()}",
        f"dropped as junk: {dropped or 'none'}",
        "",
        f"mcpp: {len(mcpp)}   incident (after junk): {len(incident)}",
        f"exact matches: {len(exact)}   near: {len(near)}   "
        f"only incident: {len(only_inc)}   only mcpp: {len(only_mcpp)}",
        "",
        f"## exact matches ({len(exact)})",
        *[f"  {mcpp_by_norm[k]}" + (f"   (incident writes it '{inc_by_norm[k]}')" if inc_by_norm[k] != mcpp_by_norm[k] else "")
          for k in exact],
        "",
        f"## near matches ({len(near)})",
        *[f"  {a}  ->  {b}   ({r:.2f})" for a, b, r in near],
        "",
        f"## only in incident data ({len(only_inc)})",
        *[f"  {n}" for n in only_inc],
        "",
        f"## only in mcpp polygons ({len(only_mcpp)})",
        *[f"  {n}" for n in only_mcpp],
    ]
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[5:7]))
    print(f"[names] wrote {OUT_FILE.relative_to(ROOT).as_posix()}")
    return {"near": near, "only_incident": only_inc, "only_mcpp": only_mcpp, "spelled_diff": spelled_diff}


if __name__ == "__main__":
    result = compare()
    if any(result.values()):
        print("[names] names differ, a name map is needed")
    else:
        print("[names] every name lines up, no map needed")
