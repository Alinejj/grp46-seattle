"""Task 5 assistant: understand the request, ask when unsure, decline what we don't cover,
and answer from the task results with every number checked.

Member 4's Streamlit app calls respond(message, state) on every chat message.
Run from the repo root to chat in the terminal:  python -m tasks.t5_assistant
"""
import importlib
import json
import re
from datetime import date, timedelta

from common.llm import chat, chat_json
from common.numcheck import check

CITY = "Seattle"
MAX_DAYS_AHEAD = 330          # hotel prices only exist for dates that can still be booked (~11 months)

# What each kind of request needs before we can answer it
REQUIRED = {
    "outlook":    ["month"],                                                # Task 2
    "hotels":     ["check_in", "nights", "guests", "max_price_per_night"],  # Task 4
    "nearby":     ["hotel_name", "check_in", "nights", "guests"],           # Task 4 nearby
    "best_month": [],                                                       # Task 6
    "places":     ["month"],                                                # Task 8
}
QUESTIONS = {
    "month": "Which month are you planning to visit?",
    "check_in": "What date do you want to check in?",
    "nights": "How many nights will you stay?",
    "guests": "How many guests will there be?",
    "max_price_per_night": "What is your budget per night, in US dollars?",
    "hotel_name": "Which hotel should I search around?",
}
INTENT_LABELS = {
    "outlook": "show reported incidents by neighbourhood",
    "hotels": "search hotels",
    "nearby": "find hotels near that one",
    "best_month": "find the best month to visit",
    "places": "recommend places and neighbourhoods",
}
DECLINE = ("Sorry, I can only help with trips to Seattle: reported incidents by neighbourhood, "
           "hotel prices and nearby alternatives, the best month to visit, and places to see. "
           "{reason}")

HELP = ("What would you like help with? I can show which Seattle neighbourhoods had more reported "
        "incidents in a month, find and compare hotel prices, suggest the best month to visit, "
        "or recommend places to see and neighbourhoods to stay in.")

EXTRACT_SYSTEM = """You turn a traveller's message into JSON for a Seattle trip planner. Today is {today}.
Return one JSON object only, with exactly these keys:
- intent: what the NEW message asks for. One of "outlook" (reported incidents by area or neighbourhood), "hotels" (hotel prices, cheapest hotel, comparing booking sites, booking a stay), "nearby" (other or cheaper hotels near a hotel the user names), "best_month" (best time or month to visit), "places" (places to visit, things to do, best neighbourhoods to stay in), "follow_up" (the message only supplies details such as dates, nights, guests, budget or a month, and asks nothing new), "out_of_scope" (anything else).
- city: the city the user names, or null if they name none.
- check_in: check-in date as YYYY-MM-DD, or null. Resolve relative dates ("next weekend", "12 March") from today's date to the next future date.
- nights: whole number, or null.
- guests: whole number, or null ("two adults" = 2, "two of us" = 2, "me and my partner" = 2, "just me" = 1).
- max_price_per_night: number in US dollars, or null.
- month: number 1-12 if the user names or implies a month, or null.
- preferences: list of short strings for wishes such as "museums" or "quiet area", or [].
- hotel_name: the hotel the user names, or null.
Rules:
- Never guess a value the user did not give: use null.
- "Collected so far" holds details from earlier in the conversation. Keep those values unless the new message changes them. Do not let them decide the intent: a new question gets its own intent even if earlier details are still missing."""

EXTRACT_KEYS = ["intent", "city", "check_in", "nights", "guests",
                "max_price_per_night", "month", "preferences", "hotel_name"]


#  understanding the request 

def extract(message, collected, today):
    user = (f"Collected so far: {json.dumps(collected)}\n"
            f'New message: "{message}"')
    return chat_json(EXTRACT_SYSTEM.format(today=today.isoformat()), user, required_keys=EXTRACT_KEYS)


def _whole_number(value, low, high):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number == int(number) and low <= number <= high else None


def clean(raw, today, message=""):
    """Validate every field in code. A bad value becomes None (so we ask for it) plus a note saying why."""
    notes = []
    req = {}

    intent = raw.get("intent")
    req["intent"] = intent if intent in REQUIRED or intent in ("out_of_scope", "follow_up") else None

    city = raw.get("city")
    req["city"] = city.strip() if isinstance(city, str) and city.strip() else None

    req["check_in"] = None
    if raw.get("check_in"):
        try:
            d = date.fromisoformat(str(raw["check_in"]))
            # "May 3" with no year sometimes comes back as this year's (already past) date:
            # move it to next year if the user didn't type a year and it then fits the window
            if d <= today and not re.search(r"\b20\d\d\b", message):
                try:
                    next_year = d.replace(year=d.year + 1)
                    if today < next_year <= today + timedelta(days=MAX_DAYS_AHEAD):
                        d = next_year
                except ValueError:          # 29 February
                    pass
            if d <= today:
                notes.append("The check-in date has to be in the future.")
            elif d > today + timedelta(days=MAX_DAYS_AHEAD):
                notes.append("Hotel prices are only available for about the next 11 months.")
            else:
                req["check_in"] = d.isoformat()
        except ValueError:
            notes.append("I couldn't read that check-in date.")

    req["nights"] = _whole_number(raw.get("nights"), 1, 30)
    req["guests"] = _whole_number(raw.get("guests"), 1, 8)

    price = raw.get("max_price_per_night")
    try:
        price = float(price) if price is not None and not isinstance(price, bool) else None
    except (TypeError, ValueError):
        price = None
    req["max_price_per_night"] = price if price is not None and 0 < price <= 5000 else None

    req["month"] = _whole_number(raw.get("month"), 1, 12)
    if req["month"] is None and req["check_in"]:
        req["month"] = date.fromisoformat(req["check_in"]).month       # stay month = check-in month

    prefs = raw.get("preferences") or []
    req["preferences"] = ([str(p) for p in prefs if isinstance(p, (str, int, float))]
                          if isinstance(prefs, list) else [])

    hotel = raw.get("hotel_name")
    req["hotel_name"] = hotel.strip() if isinstance(hotel, str) and hotel.strip() else None
    return req, notes


def respond(message, state=None, today=None):
    """One chat turn. Returns (reply, new_state, sources).

    state["collected"]      details gathered so far; the app keeps state in st.session_state
    state["last_behaviour"] what this turn did: answer, follow_up, decline, help or error
    state["last_result"]    for answers: the tables shown and the number-check outcome
    """
    today = today or date.today()
    state = dict(state or {})
    state.pop("last_result", None)
    collected = dict(state.get("collected", {}))

    try:
        raw = extract(message, collected, today)
    except (ValueError, RuntimeError):
        state["last_behaviour"] = "error"
        return ("Sorry, I couldn't understand that. Could you rephrase it?", state, [])

    request, notes = clean(raw, today, message)

    # A message that only supplies details continues the earlier request.
    # With no earlier request, hotel details (dates, nights, guests, budget) mean a hotel search.
    if request["intent"] == "follow_up":
        hotel_details = any(request[f] is not None
                            for f in ("check_in", "nights", "guests", "max_price_per_night"))
        request["intent"] = collected.get("intent") or ("hotels" if hotel_details else None)

    # Merge: a new value replaces an old one; a missing value keeps the old one
    merged = dict(collected)
    for key, value in request.items():
        if value not in (None, []):
            merged[key] = value
    if request["check_in"] and request["check_in"] != collected.get("check_in"):
        merged["month"] = date.fromisoformat(request["check_in"]).month   # month follows a new check-in

    state["last_extracted"] = raw

    # Decline other cities and topics we don't cover (earlier details stay untouched)
    if request["intent"] == "out_of_scope":
        state["last_behaviour"] = "decline"
        return (DECLINE.format(reason="That request is outside what I cover."), state, [])
    if request["city"] and CITY.lower() not in request["city"].lower():
        state["last_behaviour"] = "decline"
        return (DECLINE.format(reason=f"I don't have data for {request['city']}."), state, [])
    if merged.get("intent") is None:
        state["last_behaviour"] = "help"
        return (HELP, state, [])

    merged["city"] = CITY
    intent = merged["intent"]
    missing = [f for f in REQUIRED[intent] if merged.get(f) in (None, "")]
    state["collected"] = merged

    if missing:
        state["pending"] = missing
        state["last_behaviour"] = "follow_up"
        lines = notes + [f"To {INTENT_LABELS[intent]}, I need a bit more:"]
        lines += [f"- {QUESTIONS[f]}" for f in missing]
        if intent in ("hotels", "nearby") and re.search(
                r"\b(book|reserve|reservation|pay|charge|card|visa|mastercard)\b", message.lower()):
            lines.insert(0, "I can't make reservations or take payment details, "
                            "but I can find and compare hotel prices for you.")
        return ("\n".join(lines), state, [])

    state["pending"] = []
    state["last_behaviour"] = "answer"
    return answer(message, intent, merged, state)


#  answering from the task results 

INCIDENT_SOURCE = "Reported incidents: Seattle Police Department crime data (data.seattle.gov), 2022-2024"
OSM_SOURCE = "Places: © OpenStreetMap contributors (ODbL), via the Overpass API"
TEMP_SOURCE = "Temperatures: city_temperature.csv (Homework 2 and 4), 2020-2024"
MAX_ROWS_SENT = 15            # keep the prompt small

ANSWER_SYSTEM = """You are SafeStay, a trip-planning assistant for Seattle. Answer the traveller's request in under 150 words, using ONLY the data provided.
Rules:
- Every number you write must appear in the data. You may round a number to fewer decimals, but never calculate a new one (no differences, sums, percentages or averages of your own).
- Write numbers as digits, never as words.
- Say "reported incidents", never "danger", "dangerous", "safe" or "safety". Reported incidents are not the same as risk to a visitor.
- If hotel prices are included, say when they were retrieved and whether they are live or cached.
- If the data cannot answer part of the request, say so instead of guessing.
- You cannot make reservations or take payments; you only find and compare prices.
- Plain text, no tables."""


def _task(module, name):
    """The teammate's real function if it exists yet, otherwise the stub. Returns (function, is_stub)."""
    try:
        return getattr(importlib.import_module(f"tasks.{module}"), name), False
    except (ImportError, AttributeError):
        return getattr(importlib.import_module("tasks.stubs"), name), True


def _price_source(meta):
    when = meta.get("retrieved_at", "unknown time")
    if meta.get("source") == "live":
        return f"Hotel prices: Xotelo, retrieved live at {when}"
    age = meta.get("age_hours")
    age_text = f", {age:.1f} hours old" if isinstance(age, (int, float)) else ""
    return f"Hotel prices: Xotelo, cached copy retrieved at {when}{age_text}"


def run_task(intent, req):
    """Run the task code for one request. Returns (tables, sources, context, used_stub)."""
    used_stub = False
    context = {"data_period": "2022-2024"}

    if intent == "outlook":
        fn, used_stub = _task("t2_outlook", "outlook")
        tables = {"neighbourhood_outlook": fn(req["month"])}
        sources = [INCIDENT_SOURCE]
        context["list_size"] = 5

    elif intent == "hotels":
        fn, used_stub = _task("t4_hotels", "search")
        hotels, meta = fn(req["check_in"], req["nights"], req["guests"], req["max_price_per_night"], live=False)
        tables = {"hotels_within_budget": hotels}
        sources = [_price_source(meta), INCIDENT_SOURCE]
        context.update(prices=meta)

    elif intent == "nearby":
        fn, used_stub = _task("t4_hotels", "nearby_by_name")
        hotels, meta = fn(req["hotel_name"], req["check_in"], req["nights"], req["guests"])
        tables = {"hotels_within_2_km": hotels}
        sources = [_price_source(meta)]
        context.update(prices=meta, radius_km=2)

    elif intent == "best_month":
        fn, used_stub = _task("t6_best_months", "best_months")
        tables = {"months_ranked": fn()}
        sources = [TEMP_SOURCE, INCIDENT_SOURCE]
        context.update(comfortable_temp_f=70, temperature_period="2020-2024", recommend_top=3)

    elif intent == "places":
        fn, used_stub = _task("t8_places", "places")
        attractions, neighbourhoods = fn(req["month"], req.get("max_price_per_night"))
        tables = {"top_attractions": attractions, "top_neighbourhoods": neighbourhoods}
        sources = [OSM_SOURCE, INCIDENT_SOURCE]
        context.update(top_attractions=10, top_neighbourhoods=5, hotel_radius_km=1)
    else:
        raise ValueError(f"unknown intent {intent}")

    if used_stub:
        sources.insert(0, "STUB DATA: placeholder numbers until the real task code is in")
    return tables, sources, context, used_stub


def _data_text(details, context, tables):
    parts = [f"Request details: {json.dumps(details)}", f"Context: {json.dumps(context)}"]
    for name, df in tables.items():
        parts.append(f"### {name}\n{df.head(MAX_ROWS_SENT).to_csv(index=False)}")
    return "\n\n".join(parts)


def _table_text(tables):
    """Fallback reply: the results as plain text tables."""
    return "\n\n".join(f"{name}:\n{df.head(MAX_ROWS_SENT).to_string(index=False)}"
                       for name, df in tables.items())


def answer(message, intent, req, state):
    """Run the task, have the LLM write the answer, and only show it if every number checks out."""
    details = {k: v for k, v in req.items() if k not in ("intent", "hotel_name") and v not in (None, [])}
    if req.get("hotel_name"):
        details["hotel_name"] = req["hotel_name"]

    try:
        tables, sources, context, used_stub = run_task(intent, req)
    except Exception as err:                       # a task failed: say so instead of crashing the chat
        state["last_result"] = {"intent": intent, "number_check": "task_error", "error": str(err)[:200]}
        return (f"Sorry, I couldn't get the results for that right now ({type(err).__name__}).", state, [])

    data = _data_text(details, context, tables)
    # Everything we sent is allowed in the answer: table rows, the request details and the context
    sent_rows = [details, context] + [r for df in tables.values()
                                      for r in df.head(MAX_ROWS_SENT).to_dict("records")]
    if "prices" in context:
        sent_rows.append(context["prices"])
        sent_rows += [list(df.columns) for df in tables.values()]   # column names were sent too (e.g. "in_top5_...")

    user = f'Traveller\'s request: "{message}"\n\n{data}'
    reply, status, bad = None, "fail", []
    try:
        draft = chat(ANSWER_SYSTEM, user)
        passed, bad = check(draft, sent_rows)
        if passed:
            reply, status = draft, "pass"
        else:
            retry_user = (f"{user}\n\nYour previous answer used numbers that are not in the data: "
                          f"{', '.join(bad)}. Rewrite it using only numbers that appear in the data.")
            draft = chat(ANSWER_SYSTEM, retry_user)
            passed, bad = check(draft, sent_rows)
            if passed:
                reply, status = draft, "pass_after_retry"
    except Exception:                              # LLM unavailable: fall through to the table
        pass

    if reply is None:
        reply = ("I couldn't write a summary where every number matched the data, "
                 "so here are the results directly:\n\n" + _table_text(tables))

    state["last_result"] = {
        "intent": intent,
        "number_check": status,
        "bad_numbers": bad if status == "fail" else [],
        "used_stub": used_stub,
        "tables": {name: df.to_dict("records") for name, df in tables.items()},
    }
    return reply, state, sources


if __name__ == "__main__":
    print("SafeStay assistant. Type 'new' to start over, 'quit' to exit.")
    state = {}
    while True:
        message = input("\nyou> ").strip()
        if message.lower() in ("quit", "exit"):
            break
        if message.lower() == "new":
            state = {}
            print("(new conversation)")
            continue
        reply, state, sources = respond(message, state)
        print(f"assistant> {reply}")
        for source in sources:
            print(f"  source: {source}")