# SafeStay: Group 46 (Seattle), CS4371 Group Project 2

An LLM-assisted trip planner for Seattle. It combines SPD incident data (Spark), hotel prices (Xotelo),
temperatures (Homework 2/4), OpenStreetMap attractions, and an LLM that answers plain-English requests.
Due **December 5, 2026, 11:59 pm**.

## Setup

### 1. Clone and create a virtual environment

Run these from the repo root (the folder that contains `.git`). Keep the repo **outside OneDrive**.

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If PowerShell says scripts are disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

**WSL / Mac / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create and activate the venv in the same kind of terminal. A Windows venv won't work in WSL, and the reverse.
In VS Code: Ctrl+Shift+P → "Python: Select Interpreter" → pick `.venv`.

If you add a package, run `pip freeze > requirements.txt` and commit it.

### 2. API keys

Copy `.env.example` to `.env` and fill in the keys you need. **Never commit `.env`, and never paste a key
into code, chat, or a screenshot.** A key found in the repo costs 10 marks.

| Variable | Who needs it | Where to get it (free, no card) |
|---|---|---|
| `GROQ_API_KEY` | Everyone running the assistant or LLM summaries | console.groq.com → API Keys |
| `GEMINI_API_KEY` | Member 3 (Task 3) | aistudio.google.com → Get API key (use a personal Gmail; UTD accounts are blocked) |
| `OPENROUTER_API_KEY` | Optional backup if Groq is down | openrouter.ai → Keys |
| `SOCRATA_APP_TOKEN` | Member 1 (incident download) | data.seattle.gov → sign up → Developer Settings |
| `SERPAPI_KEY` | Member 2 (hotel backup) | serpapi.com free account |

Check your keys load:
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('groq', bool(os.getenv('GROQ_API_KEY')))"
```

## Repo layout

```
common/
  llm.py          LLM client: Groq / Gemini / OpenRouter, temperature 0, every reply cached  (Member 3)
  numcheck.py     checks every number in an LLM answer appears in the data sent             (Member 3)
  cache.py        cached GET for every other API                                            (Member 2, todo)
  geo.py          ray-casting point-in-polygon + haversine                                  (Member 2, todo)
tasks/
  t1_*.py, t2_outlook.py        Spark pipeline + monthly outlook                            (Member 1, todo)
  t3_llm_vs_data.py             LLM claims vs real data                                     (Member 3, done*)
  t4_hotels.py, t6_best_months.py                                                           (Member 2, todo)
  t5_assistant.py               the chat assistant's brain                                  (Member 3, done*)
  t8_places.py                  attractions + best neighbourhoods                           (Member 4, todo)
  stubs.py                      FAKE placeholder data until the real task functions exist
tests/
  t7_test_set.json, t7_evaluate.py   Task 7 evaluation                                      (Member 3, done*)
app.py                          Streamlit web app                                           (Member 4, todo)
cache/     every API response with its retrieval time (goes in the final ZIP)
outputs/   every t*_ output file the spec names
data/      city_temperature.csv (data/raw/ is gitignored)
```
\* done on stand-in or stub data; re-run once the real Task 1/2/4/6/8 functions land.

## Function contracts: please match these names exactly

The assistant (`tasks/t5_assistant.py`) imports these. Until a function exists, it falls back to
`tasks/stubs.py`, and its answers say **STUB DATA** in the sources. Column names inside the DataFrames
are up to you, because the LLM reads whatever comes back. Look at `tasks/stubs.py` for example shapes.

| Member | File | Function | Returns |
|---|---|---|---|
| 1 | `tasks/t2_outlook.py` | `outlook(month)` | DataFrame: top 5 by incidents, top 5 by seasonal rise, bottom 5, with persons/property split |
| 2 | `tasks/t4_hotels.py` | `search(check_in, nights, guests, max_price, live=True)` | `(DataFrame, meta)`, where meta = `{"source": "live" or "cached", "retrieved_at": "...", "age_hours": float}` |
| 2 | `tasks/t4_hotels.py` | `nearby_by_name(hotel_name, check_in, nights, guests, km=2)` | `(DataFrame, meta)`: hotels within km, cheapest first. Finds the hotel by name (users type names, not keys) |
| 2 | `tasks/t6_best_months.py` | `best_months()` | DataFrame, 12 rows |
| 4 | `tasks/t8_places.py` | `places(month, budget)` | `(top10_attractions DataFrame, top5_neighbourhoods DataFrame)`; budget may be `None` |

Don't run heavy work (Spark, downloads) at import time. Put it under `if __name__ == "__main__":`
or inside functions, or the assistant will trigger it on every import.

Member 1's Task 1 output: `outputs/neighbourhood_month.parquet` with columns
`neighbourhood, year, month, total, persons, property, other, top3_offences`.
Task 3 reads it, so please keep those names.

## Using the shared LLM code

```python
from common.llm import chat, chat_json
from common.numcheck import check

text = chat(system_prompt, user_message)                                  # Groq gpt-oss-120b, cached
data = chat_json(system_prompt, user_message, required_keys=["summary"])  # parsed + validated JSON
passed, bad_numbers = check(text, rows_you_sent)                          # Task 2 / 8 number check
```
- Responses are cached in `cache/llm/`, so an identical request never calls the API twice.
- `check()` accepts a DataFrame or a list of dicts. Pass exactly the rows you put in the prompt.
  If the prompt itself mentions a number (like "top 5"), add it: `check(text, rows, extra_allowed=[5])`.
- Ask the LLM to write numbers as digits, never as words. Spelled-out numbers aren't checked.

## Seattle data notes (checked Nov 24)

- Incidents: `data.seattle.gov/resource/tazs-3rd5.json`. The date field is `offense_date`, the neighbourhood
  field is `neighborhood`, and `nibrs_crime_against_category` gives PERSON / PROPERTY / SOCIETY directly.
- 2022–2024 has **58 real neighbourhoods** plus junk values to drop: `-`, `UNKNOWN`, `OOJ` (and `FK ERROR` in older years).
- The neighbourhood names must match the SPD MCPP GeoJSON names, or T4/T8 break. Members 1 and 2: compare early.
- `city_temperature.csv`: Seattle has 1,827 daily rows for 2020–2024, with no -99 placeholders.

## How to run what exists

```bash
python -m common.llm                              # smoke test: Groq, cache, Gemini, JSON
python -m common.numcheck                         # number-check self-test
python -m tasks.t3_llm_vs_data                    # Task 3 part A: ask both models without data
python -m tasks.t3_llm_vs_data compare --standin  # Task 3 part B on stand-in figures
python -m tasks.t3_llm_vs_data compare            # Task 3 part B on the real Task 1 table (needs pyarrow)
python -m tasks.t5_assistant                      # chat with the assistant in the terminal
python -m tests.t7_evaluate                       # Task 7 evaluation (add --fresh for real timings)
```

## Team rules (from the spec)

- Every API call goes through a cache, saved with its retrieval time. Wait at least 1 s between calls to the same service.
- No LangChain, LlamaIndex, CrewAI, AutoGen or similar. No paid services or sign-ups that ask for a card.
- Say "reported incidents", never "dangerous" or "safe". Never use demographic fields.
- Every number shown to a user comes from our data or an API. Prices show when they were retrieved.
