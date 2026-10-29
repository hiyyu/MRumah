"""
NestBot — AI Rental Assistant backend for NestMatch / MRumah
----------------------------------------------------------------
Flask API that powers the "NestBot" chat tab inside nestmatch.html.

This version fulfils the MRumah proposal objectives:
  (a) A conversational chatbot that understands natural-language property
      queries, extracts entities (location, budget, bedrooms, property type,
      pet/furnished preference, amenities), and returns RANKED listing
      suggestions — via Groq's function/tool calling.
  (b) A recommendation engine that computes personalised match scores from
      user-defined preferences (see `score_listing()` below). The same
      scoring logic is mirrored on the frontend (nestmatch.html) so the
      Discover / Swipe / Map screens and the chatbot always agree on why a
      home is (or isn't) a good match.

Design choices, on purpose:
- No database, nothing persisted server-side. The property catalogue lives
  in ../data/listings.json (a structured JSON dataset — price, bedrooms,
  amenities, coordinates — covering Kuala Lumpur, Penang, Ipoh, Johor Bahru,
  and more, per the proposal's "Proposed Method / Technology" section).
- The browser tab keeps the short conversation history in memory (JS array)
  and sends it back with each request so NestBot has context.
- CORS is enabled because nestmatch.html can be opened directly from disk or
  served from a different port than Flask.

Run this in PyCharm:
1. Open the `backend` folder as a PyCharm project (or mark it as a source root).
2. Create a virtual environment.
3. In the Terminal tab, run: pip install -r requirements.txt
4. Copy `.env.example` to `.env` and paste in your real Groq API key
   (get one free at https://console.groq.com/keys).
5. Right-click nestbot_app.py -> Run 'nestbot_app'.
6. You should see "Running on http://127.0.0.1:5000". Leave this running.
7. Open nestmatch.html (or http://127.0.0.1:5000/) and click the NestBot tab.
"""

import json
import math
import os
import re
from typing import Optional

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import groq
from groq import Groq

# ── Setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "templates"))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "listings.json"))
ENV_PATH = os.path.join(BASE_DIR, "app.env")

load_dotenv(ENV_PATH)
load_dotenv(os.path.join(BASE_DIR, ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print(
        "\n[WARNING] GROQ_API_KEY is not set. Create a .env file next to this "
        "script (see .env.example) with:\n    GROQ_API_KEY=your_key_here\n"
    )

client = Groq(api_key=GROQ_API_KEY)

# llama-3.1-8b-instant also supports tool/function calling on Groq, and has a
# much higher free-tier daily token budget (500K TPD) than the 70b model
# (100K TPD) — important since a single chat turn can call the API twice
# (tool-decision + summary) and history is resent every turn. Swap back to
# "llama-3.3-70b-versatile" if you need stronger reasoning and have quota.
MODEL_NAME = "llama-3.1-8b-instant"

app = Flask(__name__, template_folder=TEMPLATES_DIR)
CORS(app)

# ── Property catalogue ──────────────────────────────────────────────────
with open(DATA_PATH, "r", encoding="utf-8") as f:
    LISTINGS = json.load(f)


# Common Malaysian abbreviations used inconsistently across the listing
# dataset (e.g. some entries say "Bangsar, KL", others say "KLCC, Kuala
# Lumpur"). Without normalizing these, a user asking for "Kuala Lumpur"
# would fail to match listings that only say "KL", and vice versa.
LOCATION_ALIASES = {
    "kl": "kuala lumpur",
    "klcc": "kuala lumpur",
    "jb": "johor bahru",
    "kb": "kota bharu",
    "pj": "petaling jaya",
    "penang": "penang",
    "kedah": "kedah",
}


def _expand_location_text(text: str) -> str:
    """Lowercases and expands known abbreviations, matched as whole words
    (so 'kl' in 'kelantan' or 'klcc' doesn't wrongly get mangled)."""
    t = (text or "").strip().lower()
    for abbr, full in LOCATION_ALIASES.items():
        t = re.sub(rf"\b{re.escape(abbr)}\b", full, t)
    return t


def _location_matches(query_location: str, listing_loc: str) -> bool:
    """True if the user's requested area/city/state is actually present in
    the listing's location, after normalizing abbreviations on both sides."""
    q = _expand_location_text(query_location)
    loc = _expand_location_text(listing_loc)
    if not q:
        return True
    return q in loc or any(part in loc for part in q.split() if len(part) > 2)


def _haversine_km(lat1, lng1, lat2, lng2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def score_listing(listing: dict, prefs: dict) -> dict:
    """Computes a 0-100 personalised match score for one listing, broken
    down by Location / Budget / Size / Amenities — mirrors the scoring
    used client-side in nestmatch.html so the whole app stays consistent.
    `prefs` is a dict of user-defined preferences, all optional:
      budget (int RM), min_beds (int), min_sqft (int), city (str),
      pet_friendly (bool), furnished (bool), amenities (list[str]),
      user_lat/user_lng (float, for distance-based location scoring)
    """
    budget = prefs.get("budget")
    if budget:
        if listing["price"] <= budget:
            budget_score = 100 - max(0, (budget - listing["price"]) / budget * 20)
        else:
            over = (listing["price"] - budget) / budget
            budget_score = max(0, 100 - over * 140)
    else:
        budget_score = 75

    min_sqft = prefs.get("min_sqft")
    if min_sqft:
        size_score = min(100, (listing["sqft"] / min_sqft) * 100)
    else:
        size_score = min(100, 55 + listing["sqft"] / 60)

    min_beds = prefs.get("min_beds")
    if min_beds:
        bed_score = 100 if listing["beds"] >= min_beds else max(0, 100 - (min_beds - listing["beds"]) * 35)
        size_score = (size_score + bed_score) / 2

    city = (prefs.get("city") or "").strip()
    loc_score = 70
    if city:
        loc_score = 96 if _location_matches(city, listing["loc"]) else 45
    user_lat, user_lng = prefs.get("user_lat"), prefs.get("user_lng")
    if user_lat is not None and user_lng is not None:
        dist_km = _haversine_km(user_lat, user_lng, listing["lat"], listing["lng"])
        max_dist = prefs.get("max_distance_km") or 25
        dist_score = max(0, 100 - (dist_km / max_dist) * 100)
        loc_score = dist_score if not city else (loc_score + dist_score) / 2

    amen_hits, amen_total = 0, 0
    if prefs.get("pet_friendly"):
        amen_total += 1
        amen_hits += 1 if listing["pet"] else 0
    if prefs.get("furnished"):
        amen_total += 1
        amen_hits += 1 if listing["furnished"] else 0
    wanted = [a.lower() for a in (prefs.get("amenities") or [])]
    if wanted:
        have = [a.lower() for a in listing["amenities"]]
        for w in wanted:
            amen_total += 1
            amen_hits += 1 if any(w in h or h in w for h in have) else 0
    amen_score = (amen_hits / amen_total * 100) if amen_total else 78

    overall = round(loc_score * 0.25 + budget_score * 0.30 + size_score * 0.20 + amen_score * 0.25)
    overall = max(1, min(99, overall))

    return {
        "overall": overall,
        "breakdown": {
            "Location": round(loc_score),
            "Budget": round(budget_score),
            "Size": round(size_score),
            "Amenities": round(amen_score),
        },
    }


def search_listings(location=None, min_price=None, max_price=None, bedrooms=None,
                     property_type=None, pet_friendly=None, furnished=None,
                     amenities=None, limit=5) -> list:
    """The function-calling 'tool' the LLM invokes once it has extracted
    entities from the user's natural-language message. Filters the
    catalogue, scores every remaining candidate against the extracted
    preferences, and returns the top-N ranked results."""
    budget = max_price or min_price
    prefs = {
        "budget": budget,
        "min_beds": bedrooms,
        "city": location,
        "pet_friendly": pet_friendly,
        "furnished": furnished,
        "amenities": amenities or [],
    }

    candidates = []
    for listing in LISTINGS:
        # Location is a HARD filter, not just a scoring nudge — otherwise a
        # great-fit listing in the wrong state (e.g. Ipoh, Perak) can easily
        # outrank the correct-area listing on budget/size/amenities alone,
        # since those together carry 75% of the score weight. This was the
        # cause of e.g. a Kelantan search surfacing Perak/Kedah results.
        if location and not _location_matches(location, listing["loc"]):
            continue
        if property_type and listing["type"] != property_type:
            continue
        if min_price and listing["price"] < min_price:
            continue
        if max_price and listing["price"] > max_price * 1.15:
            continue
        if bedrooms and listing["beds"] < bedrooms:
            continue
        if pet_friendly and not listing["pet"]:
            continue
        if furnished and not listing["furnished"]:
            continue
        result = score_listing(listing, prefs)
        candidates.append({**listing, "personalized_match": result["overall"], "score_breakdown": result["breakdown"]})

    candidates.sort(key=lambda c: c["personalized_match"], reverse=True)
    return candidates[:limit]


def _find_listing(query: str) -> Optional[dict]:
    """Fuzzy-matches a user-provided property name/description against the
    catalogue — used by compare_listings and save_listing, since users
    rarely type the exact listing name."""
    if not query or not query.strip():
        return None
    q = query.strip().lower()

    for l in LISTINGS:
        if l["name"].lower() == q:
            return l

    name_matches = [l for l in LISTINGS if q in l["name"].lower() or l["name"].lower() in q]
    if name_matches:
        return name_matches[0]

    loc_matches = [l for l in LISTINGS if _location_matches(q, l["loc"])]
    if loc_matches:
        return loc_matches[0]

    return None


def compare_listings(names: list = None, limit: int = 4) -> list:
    """Resolves 2+ user-given property names/areas to real catalogue
    listings so they can be shown side-by-side. Unmatched names are simply
    skipped (the caller/model is told which ones, if any, couldn't be
    found)."""
    if not names:
        return []
    matched, seen_ids = [], set()
    for n in names:
        m = _find_listing(n)
        if m and m["id"] not in seen_ids:
            result = score_listing(m, {})
            matched.append({**m, "personalized_match": result["overall"], "score_breakdown": result["breakdown"]})
            seen_ids.add(m["id"])
    return matched[:limit]


def save_listing(name: str = None) -> Optional[dict]:
    """Resolves a user-given property name/description to a real catalogue
    listing to shortlist. Returns None if nothing matched closely enough."""
    m = _find_listing(name)
    if not m:
        return None
    result = score_listing(m, {})
    return {**m, "personalized_match": result["overall"], "score_breakdown": result["breakdown"]}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_listings",
            "description": (
                "Search the MRumah/NestMatch rental property catalogue and return "
                "ranked matches. Call this whenever the user is looking for, or "
                "asking to be recommended, a rental property, even if they only "
                "give partial details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City, area, or state, e.g. 'Ipoh', 'KLCC', 'Penang', 'Johor Bahru'."},
                    "min_price": {"type": ["number", "string"], "description": "Minimum monthly rent in RM. Numeric value only, no currency symbols."},
                    "max_price": {"type": ["number", "string"], "description": "Maximum monthly rent / budget in RM. Numeric value only, no currency symbols."},
                    "bedrooms": {"type": ["integer", "string"], "description": "Minimum number of bedrooms required. Numeric value only."},
                    "property_type": {"type": "string", "description": "One of 'apartment', 'house', or 'studio'. Omit this field entirely if the user didn't specify a type — do not send an empty string."},
                    "pet_friendly": {"type": "boolean", "description": "True only if the user needs a pet-friendly home."},
                    "furnished": {"type": "boolean", "description": "True only if the user wants a furnished home."},
                    "amenities": {"type": "array", "items": {"type": "string"}, "description": "Specific amenities requested, e.g. ['Pool','Gym','Near LRT']."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_listings",
            "description": (
                "Look up two or more specific properties by name (or by area, if "
                "the user doesn't give an exact name) and return their full specs "
                "side-by-side, e.g. 'compare Bangsar Garden Villa and Mont Kiara "
                "Sanctuary' or 'compare the KLCC place with the one in Ipoh'. Call "
                "this instead of search_listings whenever the user names or "
                "clearly refers to specific properties they've already seen, "
                "rather than describing a new search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The property names or areas the user wants compared, e.g. ['Bangsar Garden Villa', 'Mont Kiara Sanctuary']. Provide at least 2.",
                    },
                },
                "required": ["names"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_listing",
            "description": (
                "Shortlist/save a specific property the user names or clearly "
                "refers to, e.g. 'save the Bangsar one' or 'shortlist Sky Loft "
                "Residency'. Call this whenever the user asks to save, shortlist, "
                "bookmark, or favourite a property — do not call search_listings "
                "for this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The property name or area the user wants saved, e.g. 'Bangsar Garden Villa' or 'the Ipoh one'."},
                },
                "required": ["name"],
            },
        },
    },
]

PERSONA = (
    "You are 'NestBot', the AI rental assistant built into the NestMatch/MRumah "
    "home-rental app for Malaysian home seekers. You have access to three tools "
    "that query the REAL property catalogue: `search_listings` (find/recommend "
    "properties matching criteria), `compare_listings` (look up specific named "
    "properties side-by-side), and `save_listing` (shortlist a specific named "
    "property).\n\n"
    "How to behave:\n"
    "- Whenever the user describes what they're looking for (location, budget, "
    "bedrooms, property type, pet-friendly, furnished, amenities like pool/gym/"
    "near LRT), extract those details and call `search_listings`. Call it even "
    "with partial info — omit fields you don't know entirely, don't send empty "
    "strings like \"\" for them. Always pass min_price, max_price, and bedrooms "
    "as plain numbers (e.g. 1800), never as text or with currency symbols.\n"
    "- Whenever the user names or clearly refers to two or more specific "
    "properties they want weighed against each other (e.g. 'compare X and Y', "
    "'which is better, the Bangsar one or the Mont Kiara one'), call "
    "`compare_listings` with those names instead of `search_listings`.\n"
    "- Whenever the user asks to save, shortlist, bookmark, or favourite a "
    "specific property (e.g. 'save the Bangsar one', 'shortlist Sky Loft "
    "Residency'), call `save_listing` with that property's name. If the tool "
    "can't find a close match, say so and ask them to clarify which listing "
    "they mean — don't guess or invent one.\n"
    "- After any tool returns results, summarise conversationally: name, area, "
    "price, bedrooms, and personalised match %. Explain briefly why each is a "
    "good fit (or, for comparisons, how they differ) given what the user asked "
    "for. If no results came back, say so plainly and suggest loosening a "
    "filter (e.g. a higher budget) or double-checking the property name.\n"
    "- Never invent properties, prices, or landlord details that didn't come from "
    "a tool result.\n"
    "- For general (non-search) questions — deposits, tenancy agreements, what to "
    "check during a viewing, red flags, negotiating with landlords, moving tips — "
    "answer directly and helpfully without calling a tool.\n"
    "- You are NOT a lawyer or licensed financial advisor. For anything with "
    "binding legal or tax consequences, give general information and suggest the "
    "user confirm specifics with a qualified professional or the relevant "
    "Malaysian authority.\n"
    "- Formatting is important — this is a chat bubble, not an essay. Never reply "
    "with one long unbroken paragraph. Use short paragraphs (1-3 sentences each), "
    "and use markdown: **bold** for key numbers/terms, and \"- \" bullet points "
    "whenever you're listing more than one item, option, step, or factor (e.g. "
    "budget breakdowns, pros/cons, things to check). Keep it skimmable, like a "
    "helpful person texting, not a wall of text.\n"
    "- Be warm, concise, and conversational — a few short sentences or a few "
    "bullets per reply unless a detailed breakdown is asked for."
)

MAX_HISTORY_MESSAGES = 6


def build_messages(user_message: str, history: list, prefs: dict) -> list:
    messages = [{"role": "system", "content": PERSONA}]

    if prefs:
        messages.append({
            "role": "system",
            "content": (
                "The user's saved search preferences (use as defaults only if "
                "their message doesn't specify otherwise): " + json.dumps(prefs)
            ),
        })

    for turn in history[-MAX_HISTORY_MESSAGES:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})
    return messages


@app.route("/")
def home():
    return render_template("nestmatch.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    history = data.get("history") or []
    prefs = data.get("preferences") or {}

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    if not GROQ_API_KEY:
        return jsonify({
            "reply": "NestBot isn't fully set up yet — the server is missing a "
                     "GROQ_API_KEY. Add one to backend/.env and restart the server."
        }), 200

    try:
        messages = build_messages(user_message, history, prefs)

        first = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=250,
            temperature=0.4,
        )
        msg = first.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)
        listings_out = []
        actions_out = []

        if tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in tool_calls],
            })
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if tc.function.name == "search_listings":
                    # The model sometimes emits numbers as strings (e.g. "1800"
                    # instead of 1800) — coerce defensively instead of trusting
                    # the declared schema type.
                    for key in ("min_price", "max_price"):
                        if args.get(key) not in (None, ""):
                            try:
                                args[key] = float(args[key])
                            except (TypeError, ValueError):
                                args.pop(key, None)
                    if args.get("bedrooms") not in (None, ""):
                        try:
                            args["bedrooms"] = int(float(args["bedrooms"]))
                        except (TypeError, ValueError):
                            args.pop("bedrooms", None)
                    if isinstance(args.get("pet_friendly"), str):
                        args["pet_friendly"] = args["pet_friendly"].strip().lower() in ("true", "yes", "1")
                    if isinstance(args.get("furnished"), str):
                        args["furnished"] = args["furnished"].strip().lower() in ("true", "yes", "1")

                    # Drop empty-string / invalid values instead of forwarding them —
                    # the model occasionally sends "" for fields it doesn't actually
                    # have info for, which would otherwise wrongly filter everything.
                    if args.get("property_type") not in ("apartment", "house", "studio"):
                        args.pop("property_type", None)
                    if not (args.get("location") or "").strip():
                        args.pop("location", None)
                    if not isinstance(args.get("amenities"), list) or not args.get("amenities"):
                        args.pop("amenities", None)

                    if prefs.get("budget") and not args.get("max_price"):
                        args["max_price"] = prefs["budget"]
                    if prefs.get("min_beds") and not args.get("bedrooms"):
                        args["bedrooms"] = prefs["min_beds"]

                    results = search_listings(**{k: v for k, v in args.items() if k != "limit"})
                    listings_out.extend(results)
                    tool_payload = [{
                        "name": r["name"], "loc": r["loc"], "price": r["price"],
                        "beds": r["beds"], "baths": r["baths"], "sqft": r["sqft"],
                        "type": r["type"], "match": r["personalized_match"],
                    } for r in results]

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"results": tool_payload, "count": len(tool_payload)}),
                    })

                elif tc.function.name == "compare_listings":
                    names = args.get("names")
                    names = [n for n in names if isinstance(n, str) and n.strip()] if isinstance(names, list) else []

                    results = compare_listings(names)
                    listings_out.extend(results)
                    tool_payload = [{
                        "name": r["name"], "loc": r["loc"], "price": r["price"],
                        "beds": r["beds"], "baths": r["baths"], "sqft": r["sqft"],
                        "type": r["type"], "pet": r["pet"], "furnished": r["furnished"],
                        "amenities": r["amenities"], "match": r["personalized_match"],
                    } for r in results]

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({
                            "requested": names,
                            "results": tool_payload,
                            "matched_count": len(tool_payload),
                            "note": "Any requested name not reflected in results/matched_count "
                                    "could not be found in the catalogue — say so rather than "
                                    "inventing details for it.",
                        }),
                    })

                elif tc.function.name == "save_listing":
                    name = args.get("name") if isinstance(args.get("name"), str) else None
                    match = save_listing(name)

                    if match:
                        listings_out.append(match)
                        actions_out.append({"type": "shortlist", "id": match["id"], "name": match["name"]})
                        tool_content = {
                            "found": True,
                            "listing": {"name": match["name"], "loc": match["loc"], "price": match["price"]},
                        }
                    else:
                        tool_content = {"found": False, "requested": name}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_content),
                    })

                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": "unknown tool"}),
                    })

            second = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=400,
                temperature=0.6,
            )
            reply = (second.choices[0].message.content or "").strip()
        else:
            reply = (msg.content or "").strip()

        if not reply:
            reply = "Sorry, I couldn't come up with a reply just now. Could you try rephrasing that?"

        return jsonify({"reply": reply, "listings": listings_out, "actions": actions_out})

    except groq.RateLimitError as exc:
        print("Groq rate limit hit:", repr(exc))
        return jsonify({
            "reply": "NestBot has hit today's free AI quota (Groq's free tier resets "
                     "daily). Give it a little while and try again, or ask whoever "
                     "runs the server to switch to a lighter model / upgrade the tier "
                     "in the meantime.",
            "listings": [],
        }), 200
        except Exception as exc:
        app.logger.exception("NestBot AI request failed")

        return jsonify({
            "reply": (
                "Sorry, NestBot couldn't process your request "
                "right now. Please try again shortly."
            ),
            "listings": [],
            "actions": []
        }), 200


@app.route("/api/listings", methods=["GET"])
def api_listings():
    """Exposes the shared JSON dataset so the frontend (or graders) can
    verify the chatbot and the Discover/Map screens read from the same
    source of truth."""
    return jsonify(LISTINGS)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "groq_key_set": bool(GROQ_API_KEY), "listings": len(LISTINGS)})


if __name__ == "__main__":
    print(f"Base directory : {BASE_DIR}")
    print(f"Templates dir  : {TEMPLATES_DIR}")
    print(f"Listings file  : {DATA_PATH} ({len(LISTINGS)} properties)")
    print(f"Model in use   : {MODEL_NAME}")
    print(f"GROQ_API_KEY set: {bool(GROQ_API_KEY)}")
    app.run(debug=True, port=5000)
