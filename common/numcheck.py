"""Number check: every number in an LLM answer must appear in the data we sent it.

Used in Tasks 2, 5 and 8. An answer that fails is not shown: the caller retries
or shows the table alone.
"""
import math
import numbers
import re

# Optional $, then either 1,234,567.89 style or plain 1234.5, then optional %
NUM_RE = re.compile(r"(\$)?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(\s?%)?")


def extract_numbers(text):
    """Every number in a piece of text, with how many decimals it was written with."""
    found = []
    for m in NUM_RE.finditer(text):
        digits = m.group(2).replace(",", "")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        found.append({
            "raw": m.group(0).strip(),
            "value": float(digits),
            "decimals": decimals,
            "percent": bool(m.group(3)),
        })
    return found


def _row_values(rows):
    """All numbers in the rows: numeric cells, plus numbers inside text cells (dates etc.)."""
    if hasattr(rows, "to_dict"):              # a pandas DataFrame
        rows = rows.to_dict("records")
    values = []
    for row in rows:
        cells = row.values() if isinstance(row, dict) else row
        for v in cells:
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, numbers.Real):   # also catches numpy int64/float64
                if not math.isnan(float(v)):
                    values.append(float(v))
            else:
                values.extend(n["value"] for n in extract_numbers(str(v)))
    return values


def _matches(num, allowed):
    """True if the number equals some allowed value at the precision it was written."""
    tolerance = 0.5 * 10 ** (-num["decimals"]) + 1e-9   # "412.7" accepts 412.65..412.75
    for a in allowed:
        candidates = [a, a * 100] if num["percent"] else [a]   # 0.2485 may be written 24.9%
        if any(abs(c - num["value"]) <= tolerance for c in candidates):
            return True
    return False


def check(text, rows, extra_allowed=()):
    """Return (passed, numbers_not_found).

    text:          the LLM's answer
    rows:          list of dicts, list of lists, or a DataFrame: exactly what was sent
    extra_allowed: numbers that legitimately come from the prompt, e.g. [5] for "top 5"
    """
    allowed = _row_values(rows) + [float(x) for x in extra_allowed]
    not_found = [n["raw"] for n in extract_numbers(text) if not _matches(n, allowed)]
    return len(not_found) == 0, not_found


if __name__ == "__main__":
    rows = [
        {"neighbourhood": "CAPITOL HILL", "avg_incidents": 412.67, "share": 0.0831, "seasonal_ratio": 1.12},
        {"neighbourhood": "DOWNTOWN COMMERCIAL", "avg_incidents": 1234.33, "share": 0.2485, "seasonal_ratio": 1.05},
    ]
    good = ("Downtown Commercial averaged 1,234.3 reported incidents in July "
            "(24.9% of the city total), and Capitol Hill 412.7, with a seasonal ratio of 1.12.")
    bad = "Capitol Hill saw 450 reported incidents, up 15% on June."

    print("good summary:", check(good, rows))
    print("bad summary: ", check(bad, rows))
    print("top 5, not allowed:", check("The top 5 areas ...", rows))
    print("top 5, allowed:    ", check("The top 5 areas ...", rows, extra_allowed=[5]))
    print("date in a row:     ", check("Check-in 2027-07-10 for 3 nights at $142.",
                                       [{"check_in": "2027-07-10", "nights": 3, "price": 142.0}]))