"""
Builds picks.json from live odds.

Method (v1, market-consensus value finder):
  1. Pull moneyline (h2h) odds from every bookmaker for each upcoming game.
  2. Remove each book's margin (vig) to get that book's implied true probabilities.
  3. Average across books -> consensus probability (sharper than any single book).
  4. Find the best available price for each outcome.
  5. Expected value (EV) = consensus_prob * best_decimal_odds - 1.
  Flag outcomes where EV is positive after a minimum threshold.

This finds line-shopping value, not guaranteed winners.
Add your own sport-specific model later and blend it with the consensus.
"""
import json
import os
import sys
from datetime import datetime, timezone

import requests

API_KEY = os.environ.get("ODDS_API_KEY")
if not API_KEY:
    sys.exit("Missing ODDS_API_KEY environment variable.")

SPORTS = {
    "americanfootball_nfl": "NFL",
    "americanfootball_ncaaf": "College Football",
    "basketball_ncaab": "College Basketball",
    "basketball_nba": "NBA",
    "soccer_epl": "Premier League",
}

MIN_BOOKS = 4        # need enough books for a trustworthy consensus
MIN_EV = 0.02        # 2% expected value to show a pick
MAX_ODDS = 8.0       # skip extreme longshots


def tier(ev):
    if ev >= 0.05:
        return "Strong"
    if ev >= 0.03:
        return "Moderate"
    return "Lean"


def to_american(decimal):
    if decimal >= 2:
        return f"+{round((decimal - 1) * 100)}"
    return f"{round(-100 / (decimal - 1))}"


def devig(prices):
    """prices: {outcome: decimal odds} -> {outcome: true probability}"""
    implied = {k: 1 / v for k, v in prices.items() if v > 1}
    total = sum(implied.values())
    return {k: v / total for k, v in implied.items()} if total else {}


def analyze_event(event, sport_label):
    per_book_probs = {}
    best = {}  # outcome -> (decimal, book)
    for book in event.get("bookmakers", []):
        market = next((m for m in book.get("markets", []) if m["key"] == "h2h"), None)
        if not market:
            continue
        prices = {o["name"]: o["price"] for o in market["outcomes"]}
        probs = devig(prices)
        if not probs:
            continue
        per_book_probs[book["key"]] = probs
        for name, price in prices.items():
            if name not in best or price > best[name][0]:
                best[name] = (price, book["title"])

    if len(per_book_probs) < MIN_BOOKS:
        return []

    outcomes = set()
    for p in per_book_probs.values():
        outcomes.update(p.keys())

    picks = []
    for name in outcomes:
        vals = [p[name] for p in per_book_probs.values() if name in p]
        if len(vals) < MIN_BOOKS or name not in best:
            continue
        consensus = sum(vals) / len(vals)
        spread = max(vals) - min(vals)
        price, book_title = best[name]
        ev = consensus * price - 1
        if ev < MIN_EV or price > MAX_ODDS:
            continue
        picks.append({
            "sport": sport_label,
            "game": f'{event["away_team"]} at {event["home_team"]}',
            "start": event["commence_time"],
            "pick": name,
            "best_book": book_title,
            "best_odds_decimal": round(price, 3),
            "best_odds_american": to_american(price),
            "win_prob": round(consensus, 4),
            "ev": round(ev, 4),
            "books_used": len(vals),
            "book_disagreement": round(spread, 4),
            "tier": tier(ev),
            "reason": (
                f"{len(vals)} books put the true win chance near {consensus:.0%}. "
                f"{book_title} pays {to_american(price)}, which implies "
                f"{1 / price:.0%}."
            ),
        })
    return picks


def main():
    all_picks = []
    for key, label in SPORTS.items():
        url = f"https://api.the-odds-api.com/v4/sports/{key}/odds/"
        params = {
            "apiKey": API_KEY,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "decimal",
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"{label}: request failed ({e})")
            continue
        events = r.json()
        print(f"{label}: {len(events)} games, quota left {r.headers.get('x-requests-remaining')}")
        for ev in events:
            all_picks.extend(analyze_event(ev, label))

    all_picks.sort(key=lambda p: p["ev"], reverse=True)
    out = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "method": "Market consensus vs best available price (moneyline)",
        "picks": all_picks,
    }
    with open("picks.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(all_picks)} picks.")


if __name__ == "__main__":
    main()
