"""
Finds Kalshi sports contracts that look underpriced.

How it works:
  1. Pull sportsbook moneylines (The Odds API, US + EU books, Pinnacle weighted 3x).
  2. Strip each book's margin and average -> "fair" win probability.
  3. Pull open Kalshi game markets (public API, no key needed).
  4. For each outcome compare the fair probability to Kalshi's YES ask price.
       edge = fair probability - price - estimated fee
  5. Score confidence from edge size, how many books agree, and how tight
     Kalshi's bid/ask is. Keep picks that clear the minimum edge.
  6. Optional: if ANTHROPIC_API_KEY is set, Claude writes a one-line plain-English
     note for each top pick. Claude only restates the numbers; it does not predict.

This finds prices that look off versus the sportsbook market. It cannot
promise winners.
"""
import difflib
import json
import os
import re
import sys
from datetime import datetime, timezone

import requests

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
ODDS = "https://api.the-odds-api.com/v4/sports/{}/odds/"

# Odds API sport key -> (Kalshi series ticker, label)
SPORTS = {
    "americanfootball_nfl": ("KXNFLGAME", "NFL"),
    "americanfootball_ncaaf": ("KXNCAAFGAME", "College Football"),
    "basketball_ncaab": ("KXNCAAMBGAME", "College Basketball"),
    "basketball_nba": ("KXNBAGAME", "NBA"),
    "soccer_epl": ("KXEPLGAME", "Premier League"),
}

SHARP_WEIGHT = {"pinnacle": 3.0}   # sharpest book counts triple
MIN_BOOKS = 4
MIN_EDGE = 0.02        # 2 points after fees
FEE_RATE = 0.07        # estimated Kalshi taker fee: 7% * price * (1 - price)
STOP = {"fc", "afc", "cf", "ac", "the", "of", "university"}


# ---------- helpers ----------
def fee(c):
    return FEE_RATE * c * (1 - c)


def edge(p, c):
    return p - c - fee(c)


def max_price(p, margin=0.015):
    """Highest price (in dollars) that still leaves `margin` of edge."""
    for cents in range(99, 0, -1):
        if edge(p, cents / 100) >= margin:
            return cents / 100
    return None


def devig(prices):
    implied = {k: 1 / v for k, v in prices.items() if v > 1}
    total = sum(implied.values())
    return {k: v / total for k, v in implied.items()} if total else {}


def consensus(event):
    wsum, wtot, lo, hi, books = {}, {}, {}, {}, set()
    for b in event.get("bookmakers", []):
        mk = next((m for m in b.get("markets", []) if m["key"] == "h2h"), None)
        if not mk:
            continue
        probs = devig({o["name"]: o["price"] for o in mk["outcomes"]})
        if not probs:
            continue
        w = SHARP_WEIGHT.get(b["key"], 1.0)
        books.add(b["key"])
        for n, p in probs.items():
            wsum[n] = wsum.get(n, 0) + w * p
            wtot[n] = wtot.get(n, 0) + w
            lo[n] = min(lo.get(n, 1), p)
            hi[n] = max(hi.get(n, 0), p)
    if not wsum:
        return {}, 0, {}
    probs = {n: wsum[n] / wtot[n] for n in wsum}
    t = sum(probs.values())
    probs = {n: p / t for n, p in probs.items()}
    return probs, len(books), {n: hi[n] - lo[n] for n in hi}


def tokens(s):
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP}


def sim(a, b):
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if inter and inter == min(len(ta), len(tb)):
        return 1 + 0.01 * inter / max(len(ta), len(tb))
    r = difflib.SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio()
    return r if r >= 0.85 else 0.0


def kprice(m, name):
    """Kalshi price in dollars. Handles cents ints or *_dollars strings."""
    for key, scale in ((name + "_dollars", 1), (name, 0.01)):
        v = m.get(key)
        if v in (None, ""):
            continue
        try:
            x = float(v) * scale
        except (TypeError, ValueError):
            continue
        if x > 0:
            return x
    return None


def is_tie(m):
    sub = (m.get("yes_sub_title") or "").strip().lower()
    return sub in ("tie", "draw") or str(m.get("ticker", "")).upper().endswith("-TIE")


def match_markets(home, away, markets):
    """Return {odds outcome name: kalshi market} or {} if no confident match."""
    teams = [m for m in markets if not is_tie(m)]
    if len(teams) != 2:
        return {}
    k0, k1 = teams
    n0, n1 = k0.get("yes_sub_title", ""), k1.get("yes_sub_title", "")
    a = (sim(home, n0), sim(away, n1))
    b = (sim(home, n1), sim(away, n0))
    if min(a) > 0 and sum(a) >= sum(b):
        out = {home: k0, away: k1}
    elif min(b) > 0:
        out = {home: k1, away: k0}
    else:
        return {}
    tie = next((m for m in markets if is_tie(m)), None)
    if tie:
        out["Draw"] = tie
    return out


def score(edge_pts, books, spread, bid, ask, volume):
    s = min(edge_pts / 0.08, 1) * 40
    s += min(books / 8, 1) * 20
    s += max(0, 1 - spread / 0.08) * 20
    gap = (ask - bid) if bid else 0.06
    s += max(0, 1 - gap / 0.06) * 10
    s += min((volume or 0) / 5000, 1) * 10
    return round(s)


def level(s, e):
    if s >= 70 and e >= 0.04:
        return "High"
    if s >= 50 and e >= 0.03:
        return "Medium"
    return "Low"


# ---------- data pulls ----------
def kalshi_events(series):
    events, cursor = [], None
    for _ in range(10):
        params = {"series_ticker": series, "status": "open",
                  "with_nested_markets": "true", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(KALSHI + "/events", params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        events += j.get("events", [])
        cursor = j.get("cursor")
        if not cursor:
            break
    return events


def odds_events(sport_key, api_key):
    r = requests.get(ODDS.format(sport_key), params={
        "apiKey": api_key, "regions": "us,eu", "markets": "h2h", "oddsFormat": "decimal"
    }, timeout=30)
    r.raise_for_status()
    print(f"  odds quota left: {r.headers.get('x-requests-remaining')}")
    return r.json()


# ---------- core ----------
def find_picks(sport_label, series, odds_list, kalshi_list):
    picks = []
    for ev in odds_list:
        probs, nbooks, spreads = consensus(ev)
        if nbooks < MIN_BOOKS:
            continue
        home, away = ev["home_team"], ev["away_team"]
        for kev in kalshi_list:
            matched = match_markets(home, away, kev.get("markets", []))
            if not matched:
                continue
            for name, m in matched.items():
                if name not in probs:
                    continue
                ask, bid = kprice(m, "yes_ask"), kprice(m, "yes_bid")
                if not ask or ask >= 0.97 or ask < 0.05:
                    continue
                p = probs[name]
                e = edge(p, ask)
                if e < MIN_EDGE:
                    continue
                sc = score(e, nbooks, spreads.get(name, 0.1), bid, ask, m.get("volume"))
                lv = level(sc, e)
                picks.append({
                    "sport": sport_label,
                    "game": f"{away} at {home}",
                    "start": ev["commence_time"],
                    "side": name,
                    "kalshi_ticker": m.get("ticker"),
                    "series": series,
                    "price_cents": round(ask * 100),
                    "fair_prob": round(p, 4),
                    "edge": round(e, 4),
                    "fee_est": round(fee(ask), 4),
                    "max_price_cents": round((max_price(p) or ask) * 100),
                    "confidence": lv,
                    "score": sc,
                    "books_used": nbooks,
                    "book_spread": round(spreads.get(name, 0), 4),
                    "bid_cents": round(bid * 100) if bid else None,
                    "volume": m.get("volume"),
                    "reason": (
                        f"{nbooks} sportsbooks put {name}'s true win chance near {p:.0%}. "
                        f"Kalshi sells YES at {ask * 100:.0f} cents, which implies {ask:.0%}."
                    ),
                    "ai_note": None,
                })
            break  # one Kalshi event per game
    return picks


def add_ai_notes(picks):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or not picks:
        return
    top = picks[:12]
    data = [{
        "pick": p["side"], "game": p["game"], "sport": p["sport"],
        "kalshi_price_cents": p["price_cents"],
        "fair_win_prob_pct": round(p["fair_prob"] * 100, 1),
        "edge_points": round(p["edge"] * 100, 1),
        "books": p["books_used"],
        "book_disagreement_points": round(p["book_spread"] * 100, 1),
        "bid_cents": p["bid_cents"],
    } for p in top]
    prompt = (
        "You are helping explain prediction-market picks to a casual reader. For each item, "
        "write ONE plain-English sentence (max 30 words) on why the Kalshi price may be too low "
        "and what the main risk is. Use ONLY the numbers given. Do not invent injuries, stats, "
        "news or records. Return ONLY a JSON array of strings in the same order.\n\n"
        + json.dumps(data)
    )
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5"),
                  "max_tokens": 1500,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=90,
        )
        r.raise_for_status()
        text = r.json()["content"][0]["text"]
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        notes = json.loads(text)
        for p, n in zip(top, notes):
            p["ai_note"] = str(n)
        print(f"Claude notes added for {len(top)} picks.")
    except Exception as e:  # notes are optional
        print(f"Skipping Claude notes ({e})")


def main():
    odds_key = os.environ.get("ODDS_API_KEY")
    if not odds_key:
        sys.exit("Missing ODDS_API_KEY environment variable.")
    all_picks = []
    for sport_key, (series, label) in SPORTS.items():
        print(label)
        try:
            kal = kalshi_events(series)
            print(f"  Kalshi open events: {len(kal)}")
            if not kal:
                print(f"  (no open Kalshi events for {series}; the ticker may differ)")
                continue
            odds = odds_events(sport_key, odds_key)
            print(f"  sportsbook games: {len(odds)}")
        except requests.RequestException as e:
            print(f"  request failed: {e}")
            continue
        found = find_picks(label, series, odds, kal)
        print(f"  picks: {len(found)}")
        all_picks += found

    all_picks.sort(key=lambda p: (p["confidence"] != "High", p["confidence"] != "Medium", -p["score"]))
    add_ai_notes(all_picks)
    with open("picks.json", "w") as f:
        json.dump({
            "updated": datetime.now(timezone.utc).isoformat(),
            "method": "Sportsbook consensus vs Kalshi YES ask, after estimated fees",
            "picks": all_picks,
        }, f, indent=2)
    print(f"Wrote {len(all_picks)} picks.")


if __name__ == "__main__":
    main()
