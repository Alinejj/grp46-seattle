"""Task 5 assistant logic: understand the request, ask when unsure, decline what we don't cover.

Member 4's Streamlit app calls respond(message, state) on every chat message.
Run from the repo root to chat in the terminal:  python -m tasks.t5_assistant
"""
import json
from datetime import date, timedelta

from common.llm import chat_json

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
- guests: whole number, or null ("two adults" = 2).
- max_price_per_night: number in US dollars, or null.
- month: number 1-12 if the user names or implies a month, or null.
- preferences: list of short strings for wishes such as "museums" or "quiet area", or [].
- hotel_name: the hotel the user names, or null.
Rules:
- Never guess a value the user did not give: use null.
- "Collected so far" holds details from earlier in the conversation. Keep those values unless the new message changes them. Do not let them decide the intent: a new question gets its own intent even if earlier details are still missing."""

EXTRACT_KEYS = ["intent", "city", "check_in", "nights", "guests",
                "max_price_per_night", "month", "preferences", "hotel_name"]


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


def clean(raw, today):
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

    state["collected"] holds the details gathered so far; the app keeps it in st.session_state.
    """
    today = today or date.today()
    state = dict(state or {})
    collected = dict(state.get("collected", {}))

    try:
        raw = extract(message, collected, today)
    except (ValueError, RuntimeError):
        return ("Sorry, I couldn't understand that. Could you rephrase it?", state, [])

    request, notes = clean(raw, today)

    # A message that only supplies details continues the earlier request
    if request["intent"] == "follow_up":
        request["intent"] = collected.get("intent")

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
        return (DECLINE.format(reason="That request is outside what I cover."), state, [])
    if request["city"] and CITY.lower() not in request["city"].lower():
        return (DECLINE.format(reason=f"I don't have data for {request['city']}."), state, [])
    if merged.get("intent") is None:
        return (HELP, state, [])

    merged["city"] = CITY
    intent = merged["intent"]
    missing = [f for f in REQUIRED[intent] if merged.get(f) in (None, "")]
    state["collected"] = merged

    if missing:
        state["pending"] = missing
        lines = notes + [f"To {INTENT_LABELS[intent]}, I need a bit more:"]
        lines += [f"- {QUESTIONS[f]}" for f in missing]
        if intent == "hotels" and "book" in message.lower():
            lines.insert(0, "I can't make reservations, but I can find and compare hotel prices for you.")
        return ("\n".join(lines), state, [])

    state["pending"] = []
    return run_request(intent, merged, state)


def run_request(intent, request, state):
    """Placeholder until the task functions exist: shows what would be called."""
    args = {k: v for k, v in request.items() if k != "intent" and v not in (None, [])}
    return f"[ready] would run {intent} with {json.dumps(args)}", state, []


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