"""Task 6: which months are best to visit Seattle, by temperature and reported incidents.

Temperature from data/city_temperature.csv (2020-2024), incidents from member 1's
outputs/neighbourhood_month.parquet (2022-2024). Both are min-max scaled and weighted.
Used by the assistant (Task 5) for "when should I go" questions.
"""
import calendar
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEMP_CSV = ROOT / "data" / "city_temperature.csv"
INCIDENTS_PARQUET = ROOT / "outputs" / "neighbourhood_month.parquet"
OUT_CSV = ROOT / "outputs" / "t6_best_months.csv"

COMFORT_F = 70
TEMP_YEARS = (2020, 2024)
INCIDENT_YEARS = (2022, 2024)
# weather is what a visitor actually feels day to day, citywide incidents barely move month to month
TEMP_WEIGHT = 0.6
INCIDENT_WEIGHT = 0.4


# seattle's average temperature per month over 2020-2024
def _monthly_temps():
    df = pd.read_csv(TEMP_CSV, low_memory=False)
    df = df[(df["City"] == "Seattle") & df["Year"].between(*TEMP_YEARS)]
    missing = (df["AvgTemperature"] == -99).sum()     # -99 is the dataset's "no reading"
    print(f"[t6] dropped {missing} days with AvgTemperature -99")
    df = df[df["AvgTemperature"] != -99]
    temps = df.groupby("Month")["AvgTemperature"].mean().rename("avg_temp_f")
    return temps.rename_axis("month").reset_index()


# citywide reported incidents per month, averaged over 2022-2024
def _monthly_incidents():
    if not INCIDENTS_PARQUET.exists():
        print("[t6] " + "!" * 60)
        print("[t6] USING STUB incidents, outputs/neighbourhood_month.parquet isn't there yet")
        print("[t6] " + "!" * 60)
        from tasks import stubs
        return stubs.best_months()[["month", "avg_incidents"]], True

    df = pd.read_parquet(INCIDENTS_PARQUET)
    df = df[df["year"].between(*INCIDENT_YEARS)]
    # sum the neighbourhoods first so each year-month is one citywide total
    per_month = df.groupby(["year", "month"])["total"].sum().reset_index()
    avg = per_month.groupby("month")["total"].mean().rename("avg_incidents")
    return avg.reset_index(), False


def _minmax(s):
    spread = s.max() - s.min()
    return (s - s.min()) / spread if spread else s * 0.0    # all equal -> everyone's best


# score and rank the 12 months, lower score is better
def _score(df, temp_weight):
    df = df.copy()
    df["score"] = (temp_weight * df["temp_norm"] + (1 - temp_weight) * df["incident_norm"]).round(4)
    df["rank"] = df["score"].rank(method="min").astype(int)
    return df.sort_values(["rank", "month"]).reset_index(drop=True)


# 12 months ranked best to worst for a visit
def best_months(temp_weight=TEMP_WEIGHT):
    temps = _monthly_temps()
    incidents, used_stub = _monthly_incidents()
    df = temps.merge(incidents, on="month", how="left")
    if len(df) != 12 or df["avg_incidents"].isna().any():
        print(f"[t6] expected 12 full months, got {len(df)} with {df['avg_incidents'].isna().sum()} missing incidents")

    df["month_name"] = df["month"].map(lambda m: calendar.month_name[m])
    df["avg_temp_f"] = df["avg_temp_f"].round(1)
    df["distance_from_70"] = (df["avg_temp_f"] - COMFORT_F).abs().round(1)
    df["avg_incidents"] = df["avg_incidents"].round(1)
    df["temp_norm"] = _minmax(df["distance_from_70"]).round(4)
    df["incident_norm"] = _minmax(df["avg_incidents"]).round(4)
    df = _score(df, temp_weight)
    df.attrs["incidents_from_stub"] = used_stub
    cols = ["month", "month_name", "avg_temp_f", "distance_from_70", "avg_incidents",
            "temp_norm", "incident_norm", "score", "rank"]
    return df[cols]


if __name__ == "__main__":
    ranked = best_months()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(OUT_CSV, index=False)
    print(ranked.to_string(index=False))
    top3 = list(ranked["month_name"].head(3))
    print(f"\n[t6] top 3 at {TEMP_WEIGHT}/{INCIDENT_WEIGHT}: {', '.join(top3)}")

    for w in (0.5, 0.7):
        other = list(_score(ranked, w)["month_name"].head(3))
        same = "same top 3" if other == top3 else ("same months, different order" if set(other) == set(top3)
                                                    else "top 3 changes")
        print(f"[t6] top 3 at {w}/{round(1 - w, 1)}: {', '.join(other)} ({same})")
    print(f"[t6] wrote {OUT_CSV.relative_to(ROOT).as_posix()}")
