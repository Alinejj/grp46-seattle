"""Task 7: run every test request through the assistant and score it against the hand labels.

Run from the repo root:
    python -m tests.t7_evaluate           (reuses cached LLM replies)
    python -m tests.t7_evaluate --fresh   (forces new API calls, for real timing and call counts)
"""
import csv
import json
import sys
import time
from datetime import date
from pathlib import Path

from common import llm
from tasks.t5_assistant import respond

ROOT = Path(__file__).resolve().parent.parent
TEST_SET = ROOT / "tests" / "t7_test_set.json"
OUT_CSV = ROOT / "outputs" / "t7_evaluation.csv"
GRADED_FIELDS = ["intent", "check_in", "nights", "guests", "max_price_per_night", "month"]


def same(got, want):
    """Compare one field. A list in the label means any of those values is fine."""
    options = want if isinstance(want, list) else [want]
    for option in options:
        if option is None and got in (None, ""):
            return True
        if isinstance(option, (int, float)) and isinstance(got, (int, float)) and float(got) == float(option):
            return True
        if option == got:
            return True
    return False


def run_case(case, today):
    state = {}
    llm.stats.update(api_calls=0, cache_hits=0)
    start = time.time()
    reply = ""
    for message in case["messages"]:
        reply, state, _sources = respond(message, state, today=today)
    seconds = time.time() - start

    behaviour = state.get("last_behaviour")
    fields = state.get("collected", {})
    expected = case["expected"]

    checked, wrong = 0, []
    for field in GRADED_FIELDS:
        if field in expected:
            checked += 1
            if not same(fields.get(field), expected[field]):
                wrong.append(f"{field}: got {fields.get(field)!r}, want {expected[field]!r}")

    return {
        "id": case["id"],
        "source": case["source"],
        "request": " | ".join(case["messages"]),
        "expected_behaviour": expected["behaviour"],
        "got_behaviour": behaviour,
        "behaviour_ok": behaviour in (expected["behaviour"] if isinstance(expected["behaviour"], list)
                                        else [expected["behaviour"]]),
        "fields_checked": checked,
        "fields_correct": checked - len(wrong),
        "wrong_fields": "; ".join(wrong),
        "number_check": state.get("last_result", {}).get("number_check", ""),
        "bad_numbers": ", ".join(state.get("last_result", {}).get("bad_numbers", [])),
        "seconds": round(seconds, 2),
        "api_calls": llm.stats["api_calls"],
        "cache_hits": llm.stats["cache_hits"],
        "reply": reply.replace("\n", " ")[:300],
        "extracted": json.dumps(state.get("last_extracted")),
    }


def summarise(rows):
    n = len(rows)
    checked = sum(r["fields_checked"] for r in rows)
    correct = sum(r["fields_correct"] for r in rows)

    def rate(subset):
        return f"{sum(r['behaviour_ok'] for r in subset)}/{len(subset)}" if subset else "n/a"

    follow = [r for r in rows if r["expected_behaviour"] == "follow_up"]
    decline = [r for r in rows if r["expected_behaviour"] == "decline"]
    answer = [r for r in rows if r["expected_behaviour"] == "answer"]
    checked_answers = [r for r in rows if r["number_check"] in ("pass", "pass_after_retry", "fail")]
    passed_answers = [r for r in checked_answers if r["number_check"] in ("pass", "pass_after_retry")]
    first_try = [r for r in checked_answers if r["number_check"] == "pass"]
    lines = [
        f"Requests:                      {n}",
        f"Fields correct:                {correct}/{checked} ({100 * correct / checked:.1f}%)",
        f"Requests with all fields right:{sum(r['fields_correct'] == r['fields_checked'] for r in rows):>3}/{n}",
        f"Correct behaviour overall:     {rate(rows)}",
        f"  asked a follow-up correctly: {rate(follow)}",
        f"  declined correctly:          {rate(decline)}",
        f"  answered when it should:     {rate(answer)}",
        f"Answers passing number check:  {len(passed_answers)}/{len(checked_answers)}"
        f" ({len(first_try)} on the first try)",
        f"Mean seconds per request:      {sum(r['seconds'] for r in rows) / n:.2f}",
        f"Mean API calls per request:    {sum(r['api_calls'] for r in rows) / n:.2f}",
    ]
    return "\n".join(lines)


def main():
    if "--fresh" in sys.argv:
        llm.READ_CACHE = False
    test_set = json.loads(TEST_SET.read_text(encoding="utf-8"))
    today = date.fromisoformat(test_set["today"])        # fixed date so relative dates stay labelled correctly

    rows = []
    for case in test_set["cases"]:
        row = run_case(case, today)
        rows.append(row)
        mark = "ok " if row["behaviour_ok"] and not row["wrong_fields"] else "BAD"
        print(f"{mark} {row['id']} behaviour {row['got_behaviour']}/{row['expected_behaviour']}"
              f"  fields {row['fields_correct']}/{row['fields_checked']}  {row['wrong_fields']}")

    OUT_CSV.parent.mkdir(exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + summarise(rows))
    print(f"\nSaved {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()