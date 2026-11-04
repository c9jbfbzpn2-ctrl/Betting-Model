"""
Finds Kalshi sports contracts that look underpriced and likely to hit.

Bet types: moneyline, point spreads, and game totals.

How it works:
  1. Pull sportsbook lines from The Odds API (moneyline, spreads, totals).
  2. Strip each book's margin and average -> "fair" probability for each line.
  3. Pull open Kalshi markets (public API, no key needed).
  4. Match a Kalshi contract to a sportsbook line ONLY when the line is identical
     (for example Kalshi "Over 45.5" vs the books' Over 45.5). No guessing between lines.
  5. For each contract, test buying YES and (for spreads/totals) buying NO:
         edge = fair chance of winning - price paid - estimated fee
  6. Keep only bets with at least a 35% fair chance of hitting and a real edge.
  7. Rate confidence 1-5 and suggest units.
  8. Optional: Claude writes a one-line plain-English note if ANTHROPIC_API_KEY is set.

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import model as M
except Exception as e:                     # the picks must still work without the ratings
    M = None
    print(f"Ratings model unavailable ({e})")

MODEL = None                                # set in main()
# 0 = ratings are shown for information only. The NFL backtest found no benefit from blending,
# so raise this only after the live track record supports it.
MODEL_WEIGHT = float(os.environ.get("MODEL_WEIGHT", "0"))

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
ODDS = "https://api.the-odds-api.com/v4/sports/{}/odds/"
REGIONS = os.environ.get("ODDS_REGIONS", "us")   # each region costs credits; "us,eu" adds Pinnacle

# Odds API sport key -> (label, Kalshi moneyline, spread, total series, scoring word)
SPORTS = {
    "americanfootball_nfl": ("NFL", "KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL", "points"),
    "americanfootball_ncaaf": ("College Football", "KXNCAAFGAME", "KXNCAAFSPREAD", "KXNCAAFTOTAL", "points"),
    "basketball_ncaab": ("College Basketball", "KXNCAAMBGAME", "KXNCAAMBSPREAD", "KXNCAAMBTOTAL", "points"),
    "basketball_nba": ("NBA", "KXNBAGAME", "KXNBASPREAD", "KXNBATOTAL", "points"),
    "soccer_epl": ("Premier League", "KXEPLGAME", "KXEPLSPREAD", "KXEPLTOTAL", "goals"),
}

SHARP_WEIGHT = {"pinnacle": 3.0}
MIN_BOOKS = 4          # moneyline: books needed
MIN_BOOKS_LINE = 3     # spreads/totals: books posting that exact line
MIN_PROB = 0.35        # a bet must have at least a 35% fair chance of hitting
MIN_EDGE = 0.02        # 2 points after estimated fees = a real pick (1 to 5 stars)
WATCH_EDGE = 0.005     # 0.5 to 2 points = "Watch": priced about fairly, not a recommended bet
MAX_EDGE = 0.10        # a 10+ point gap on a liquid market is almost always a data error, not free money
MIN_PRICE = 0.15
MAX_PRICE = 0.95
MAX_GAP = 0.10         # skip contracts whose bid/ask are more than 10 cents apart
MAX_STALE = 0.15       # skip when the ask is this far from the last traded price (stale or odd data)
SKIPPED = {"inconsistent": 0, "stale": 0}
FEE_RATE = 0.07        # estimated Kalshi taker fee: 7% * price * (1 - price)
PER_GAME = 2           # keep at most this many picks per game (they are correlated)
STOP = {"fc", "afc", "cf", "ac", "the", "of", "university"}
MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
UNITS = {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 3.0}


# ---------- math ----------
def fee(c):
    return FEE_RATE * c * (1 - c)


def edge(p, c):
    return p - c - fee(c)


def max_price(p, margin=0.015):
    for cents in range(99, 0, -1):
        if edge(p, cents / 100) >= margin:
            return cents / 100
    return None


def devig(prices):
    implied = {k: 1 / v for k, v in prices.items() if v and v > 1}
    total = sum(implied.values())
    return {k: v / total for k, v in implied.items()} if total and len(implied) == len(prices) else {}


def score(edge_pts, books, spread, bid, ask, volume, prob):
    s = min(edge_pts / 0.08, 1) * 35
    s += min(books / 8, 1) * 15
    s += max(0, 1 - spread / 0.08) * 15
    gap = (ask - bid) if bid else MAX_GAP
    s += max(0, 1 - gap / 0.06) * 10
    s += min((volume or 0) / 5000, 1) * 10
    s += min(max((prob - MIN_PROB) / 0.40, 0), 1) * 15
    return round(s)


def blend(p_mkt, p_mod, sport):
    if p_mod is None or MODEL is None or MODEL_WEIGHT <= 0 or not MODEL.enabled(sport):
        return p_mkt
    return (1 - MODEL_WEIGHT) * p_mkt + MODEL_WEIGHT * p_mod


def stars(s, e):
    if s >= 80 and e >= 0.06:
        return 5
    if s >= 70 and e >= 0.04:
        return 4
    if s >= 60 and e >= 0.03:
        return 3
    if s >= 50 and e >= 0.025:
        return 2
    return 1


def label(st):
    return "High" if st >= 4 else "Medium" if st == 3 else "Low" if st >= 1 else "Watch"


# ---------- sportsbook consensus ----------
def consensus_h2h(event):
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


def consensus_lines(event, market_key):
    """{line: (prob, books, disagreement)}.
    spreads: line = home team's point, prob = home team covers.
    totals:  line = total, prob = Over hits."""
    home = event["home_team"]
    acc = {}
    for b in event.get("bookmakers", []):
        mk = next((m for m in b.get("markets", []) if m["key"] == market_key), None)
        if not mk:
            continue
        outs = mk["outcomes"]
        if market_key == "spreads":
            x = next((o for o in outs if o["name"] == home), None)
            y = next((o for o in outs if o["name"] != home), None)
        else:
            x = next((o for o in outs if o["name"].lower() == "over"), None)
            y = next((o for o in outs if o["name"].lower() == "under"), None)
        if not x or not y or x.get("point") is None:
            continue
        probs = devig({"x": x["price"], "y": y["price"]})
        if not probs:
            continue
        w = SHARP_WEIGHT.get(b["key"], 1.0)
        acc.setdefault(round(float(x["point"]), 1), []).append((w, probs["x"]))
    out = {}
    for pt, lst in acc.items():
        if len(lst) < MIN_BOOKS_LINE:
            continue
        wt = sum(w for w, _ in lst)
        p = sum(w * v for w, v in lst) / wt
        vals = [v for _, v in lst]
        out[pt] = (p, len(lst), max(vals) - min(vals))
    return out


# ---------- name matching ----------
def tokens(s):
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP]


def sim(a, b):
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)

    def tok_ok(x):
        return any(x == y or (len(x) <= 2 and y.startswith(x)) for y in big)

    if all(tok_ok(x) for x in small):
        return 1 + 0.01 * len(small) / len(big)
    if ta[-1] == tb[-1] and len(ta[-1]) >= 4:          # same nickname (Chiefs, Jets...)
        return 0.9
    r = difflib.SequenceMatcher(None, " ".join(sorted(ta)), " ".join(sorted(tb))).ratio()
    return r if r >= 0.85 else 0.0


def date_ok(kev, commence):
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", str(kev.get("event_ticker", "")))
    if not m or m.group(2) not in MONTHS:
        return True
    try:
        kd = datetime(2000 + int(m.group(1)), MONTHS[m.group(2)], int(m.group(3)))
        cd = datetime.fromisoformat(commence.replace("Z", "+00:00")).replace(tzinfo=None)
        return abs((kd - cd).days) <= 1
    except ValueError:
        return True


def title_matches(kev, home, away):
    title = (kev.get("title") or "").split(":")[0].strip()
    parts = re.split(r"\s+(?:at|vs\.?|v|@)\s+", title, flags=re.I)
    if len(parts) != 2:
        return False
    a, b = parts
    return (sim(home, a) > 0 and sim(away, b) > 0) or (sim(home, b) > 0 and sim(away, a) > 0)


# ---------- Kalshi helpers ----------
def kprice(m, name):
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


def quotes(m):
    """(yes_ask, yes_bid, no_ask, no_bid) in dollars, filling gaps from the other side."""
    ya, yb = kprice(m, "yes_ask"), kprice(m, "yes_bid")
    na, nb = kprice(m, "no_ask"), kprice(m, "no_bid")
    if ya is None and nb:
        ya = 1 - nb
    if na is None and yb:
        na = 1 - yb
    if yb is None and na:
        yb = 1 - na
    if nb is None and ya:
        nb = 1 - ya
    return ya, yb, na, nb


def is_tie(m):
    sub = (m.get("yes_sub_title") or "").strip().lower()
    return sub in ("tie", "draw") or str(m.get("ticker", "")).upper().endswith("-TIE")


def match_moneyline(home, away, markets):
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


def num(x):
    return f"{x:g}"


# ---------- building picks ----------
def evaluate(base, m, series, p_yes, bet_yes, bet_no, kind, books, spread, allow_no,
             market_yes=None, model_yes=None):
    """Return picks for buying YES and (optionally) NO on one Kalshi market."""
    ya, yb, na, nb = quotes(m)
    last = kprice(m, "last_price")
    out = []
    sides = [("YES", p_yes, ya, yb, bet_yes)]
    if allow_no:
        sides.append(("NO", 1 - p_yes, na, nb, bet_no))
    for action, p, ask, bid, bet in sides:
        if not ask or not bid:
            continue
        if p < MIN_PROB or ask < MIN_PRICE or ask > MAX_PRICE or (ask - bid) > MAX_GAP:
            continue
        if last is not None and abs(ask - (1 - last if action == "NO" else last)) > MAX_STALE:
            SKIPPED["stale"] += 1
            continue
        e = edge(p, ask)
        if e < WATCH_EDGE or e > MAX_EDGE:
            continue
        flip = action == "NO"
        mk = None if market_yes is None else (1 - market_yes if flip else market_yes)
        md = None if model_yes is None else (1 - model_yes if flip else model_yes)
        sc = score(e, books, spread, bid, ask, m.get("volume"), p)
        st = stars(sc, e) if e >= MIN_EDGE else 0
        out.append({
            **base,
            "type": kind,
            "bet": bet,
            "side": bet,
            "action": action,
            "kalshi_ticker": m.get("ticker"),
            "event_ticker": m.get("event_ticker") or str(m.get("ticker", "")).rsplit("-", 1)[0],
            "series": series,
            "price_cents": round(ask * 100),
            "fair_prob": round(p, 4),
            "market_prob": None if mk is None else round(mk, 4),
            "model_prob": None if md is None else round(md, 4),
            "edge": round(e, 4),
            "fee_est": round(fee(ask), 4),
            "max_price_cents": round((max_price(p) or ask) * 100),
            "stars": st,
            "units": UNITS[st],
            "confidence": label(st),
            "score": sc,
            "books_used": books,
            "book_spread": round(spread, 4),
            "bid_cents": round(bid * 100),
            "last_cents": None if last is None else round(last * 100),
            "volume": m.get("volume"),
            "reason": (
                f"{books} sportsbooks put the true chance of this near {p:.0%}. "
                f"Kalshi charges {ask * 100:.0f} cents for {action}, which implies {ask:.0%}."
            ),
            "ai_note": None,
        })
    return out


def find_picks(label_, word, odds_list, kal):
    """kal = {"ML": (series, events), "SPREAD": (...), "TOTAL": (...)}"""
    picks = []
    for ev in odds_list:
        home, away, start = ev["home_team"], ev["away_team"], ev["commence_time"]
        base = {"sport": label_, "game": f"{away} at {home}", "start": start}
        game_picks = []

        # moneyline
        probs, nbooks, spreads = consensus_h2h(ev)
        series, events = kal.get("ML", ("", []))
        pred = MODEL.predict(label_, home, away) if MODEL else None
        if nbooks >= MIN_BOOKS:
            for kev in events:
                if not date_ok(kev, start):
                    continue
                matched = match_moneyline(home, away, kev.get("markets", []))
                if not matched:
                    continue
                asks = [quotes(x)[0] for x in matched.values()]
                if any(a is None for a in asks) or not (0.97 <= sum(asks) <= 1.25):
                    SKIPPED["inconsistent"] += 1
                    print(f"  skipped {away} at {home}: Kalshi asks {[round(a, 2) if a else a for a in asks]} "
                          f"do not add up to about 100%")
                    continue
                for name, m in matched.items():
                    if name not in probs:
                        continue
                    bet = "Draw" if name == "Draw" else f"{name} to win"
                    pm = None
                    if pred:
                        pm = pred["p_draw"] if name == "Draw" else pred["p_home"] if name == home else pred["p_away"]
                    game_picks += evaluate(base, m, series, blend(probs[name], pm, label_), bet, bet,
                                           "Moneyline", nbooks, spreads.get(name, 0.1), False,
                                           probs[name], pm)
                break

        # spreads
        series, events = kal.get("SPREAD", ("", []))
        lines = consensus_lines(ev, "spreads") if events else {}
        for kev in events:
            if not date_ok(kev, start) or not title_matches(kev, home, away):
                continue
            for m in kev.get("markets", []):
                mt = re.match(r"\s*(.+?)\s+wins by over\s+([\d.]+)", m.get("yes_sub_title") or "", re.I)
                if not mt:
                    continue
                team, n = mt.group(1), round(float(mt.group(2)), 1)
                if sim(home, team) > 0 and sim(away, team) == 0:
                    is_home = True
                elif sim(away, team) > 0 and sim(home, team) == 0:
                    is_home = False
                else:
                    continue
                key = -n if is_home else n            # home team's point on the book line
                if key not in lines:
                    continue
                p_cover, nb, sp = lines[key]
                pm_cover = M.p_home_covers(pred, -key) if (pred and M) else None
                p_cov = blend(p_cover, pm_cover, label_)
                p_yes = p_cov if is_home else 1 - p_cov
                mk_yes = p_cover if is_home else 1 - p_cover
                pm_yes = None if pm_cover is None else (pm_cover if is_home else 1 - pm_cover)
                name, opp = (home, away) if is_home else (away, home)
                game_picks += evaluate(base, m, series, p_yes,
                                       f"{name} -{num(n)}", f"{opp} +{num(n)}",
                                       "Spread", nb, sp, True, mk_yes, pm_yes)

        # totals
        series, events = kal.get("TOTAL", ("", []))
        lines = consensus_lines(ev, "totals") if events else {}
        for kev in events:
            if not date_ok(kev, start) or not title_matches(kev, home, away):
                continue
            for m in kev.get("markets", []):
                mt = re.match(r"\s*over\s+([\d.]+)", m.get("yes_sub_title") or "", re.I)
                if not mt:
                    continue
                n = round(float(mt.group(1)), 1)
                if n not in lines:
                    continue
                p_over, nb, sp = lines[n]
                game_picks += evaluate(base, m, series, p_over,
                                       f"Over {num(n)} total {word}", f"Under {num(n)} total {word}",
                                       "Total", nb, sp, True)

        game_picks.sort(key=lambda p: -p["score"])
        picks += game_picks[:PER_GAME]
    return picks


# ---------- track record ----------
HISTORY_FILE = "history.json"
SETTLE_AFTER_HOURS = 3      # wait this long after the start before checking a result
GIVE_UP_DAYS = 10           # stop waiting on a result after this long
MAX_SETTLE_CALLS = 120


def load_history():
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def track(history, picks):
    """Log each new pick once, the first time it appears (price and stars as first seen)."""
    seen = {h["id"] for h in history}
    now = datetime.now(timezone.utc).isoformat()
    added = 0
    for p in picks:
        if p["stars"] < 1:                   # watchlist items are not recommended bets
            continue
        pid = f'{p["kalshi_ticker"]}:{p["action"]}'
        if pid in seen:
            continue
        seen.add(pid)
        added += 1
        history.append({
            "id": pid, "first_seen": now, "result": None,
            **{k: p.get(k) for k in ("sport", "type", "bet", "action", "game", "start",
                                     "price_cents", "fair_prob", "market_prob", "model_prob",
                                     "fee_est", "stars", "kalshi_ticker", "event_ticker")},
        })
    print(f"Tracking {added} new picks ({len(history)} total in history).")
    return history


def fetch_market(ticker, event_ticker):
    try:
        r = requests.get(f"{KALSHI}/markets/{ticker}", timeout=20)
        if r.status_code == 200:
            m = r.json().get("market")
            if m:
                return m
    except requests.RequestException:
        pass
    try:
        r = requests.get(f"{KALSHI}/events/{event_ticker}",
                         params={"with_nested_markets": "true"}, timeout=20)
        if r.status_code == 200:
            j = r.json()
            ms = j.get("markets") or (j.get("event") or {}).get("markets") or []
            return next((m for m in ms if m.get("ticker") == ticker), None)
    except requests.RequestException:
        pass
    return None


def settle(history):
    now = datetime.now(timezone.utc)
    calls = settled = 0
    for h in history:
        if h.get("result") is not None or calls >= MAX_SETTLE_CALLS:
            continue
        try:
            start = datetime.fromisoformat(h["start"].replace("Z", "+00:00"))
        except ValueError:
            continue
        age = now - start
        if age.total_seconds() < SETTLE_AFTER_HOURS * 3600:
            continue
        calls += 1
        m = fetch_market(h["kalshi_ticker"], h["event_ticker"])
        res = (m or {}).get("result")
        if res in ("yes", "no"):
            h["result"] = "won" if res == h["action"].lower() else "lost"
            settled += 1
        elif age.days >= GIVE_UP_DAYS:
            h["result"] = "void"
    print(f"Checked {calls} unsettled picks, settled {settled}.")


def summarize(rows):
    n = len(rows)
    if not n:
        return {"n": 0, "wins": 0}
    wins = sum(1 for r in rows if r["result"] == "won")
    stake = ret = 0.0
    for r in rows:
        cost = r["price_cents"] / 100 + r.get("fee_est", 0)
        stake += cost
        ret += (1 - cost) if r["result"] == "won" else -cost
    return {
        "n": n, "wins": wins,
        "hit_rate": round(wins / n, 4),
        "expected_rate": round(sum(r["fair_prob"] for r in rows) / n, 4),
        "roi": round(ret / stake, 4) if stake else 0,
    }


def record(history):
    done = [h for h in history if h.get("result") in ("won", "lost")]
    pending = [h for h in history if h.get("result") is None]
    return {
        "overall": summarize(done),
        "high_confidence": summarize([h for h in done if h["stars"] >= 4]),
        "by_stars": {str(i): summarize([h for h in done if h["stars"] == i]) for i in range(1, 6)},
        "by_type": {t: summarize([h for h in done if h["type"] == t]) for t in ("Moneyline", "Spread", "Total")},
        "pending": len(pending),
        "tracking_since": min((h["first_seen"] for h in history), default=None),
    }


def add_ai_notes(picks):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or not picks:
        return
    top = picks[:12]
    data = [{
        "bet": p["bet"], "type": p["type"], "game": p["game"], "sport": p["sport"],
        "kalshi_price_cents": p["price_cents"],
        "fair_chance_pct": round(p["fair_prob"] * 100, 1),
        "edge_points": round(p["edge"] * 100, 1),
        "books": p["books_used"],
        "book_disagreement_points": round(p["book_spread"] * 100, 1),
    } for p in top]
    prompt = (
        "You are helping explain prediction-market bets to a casual reader. For each item, "
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
        for p, n in zip(top, json.loads(text)):
            p["ai_note"] = str(n)
        print(f"Claude notes added for {len(top)} picks.")
    except Exception as e:
        print(f"Skipping Claude notes ({e})")


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
        "apiKey": api_key, "regions": REGIONS,
        "markets": "h2h,spreads,totals", "oddsFormat": "decimal"
    }, timeout=30)
    r.raise_for_status()
    print(f"  odds credits left: {r.headers.get('x-requests-remaining')}")
    return r.json()


def main():
    odds_key = os.environ.get("ODDS_API_KEY")
    if not odds_key:
        sys.exit("Missing ODDS_API_KEY environment variable.")
    global MODEL
    model_status = {}
    if M is not None and os.environ.get("SKIP_MODEL") != "1":
        try:
            print("Updating team ratings")
            state = M.update(M.load_state())
            M.save_state(state)
            MODEL = M.Model(state)
            model_status = MODEL.status
            for k, v in model_status.items():
                print(f"  {k}: checked {v['games_checked']} games, gain vs naive {v['gain_vs_baseline']:.1%}, "
                      f"{'usable' if v['enabled'] else 'not yet usable'}")
        except Exception as e:
            print(f"Ratings update failed ({e}); continuing without it")
    all_picks = []
    for sport_key, (label_, s_ml, s_sp, s_tot, word) in SPORTS.items():
        print(label_)
        kal = {}
        for kind, series in (("ML", s_ml), ("SPREAD", s_sp), ("TOTAL", s_tot)):
            try:
                evs = kalshi_events(series)
            except requests.RequestException as e:
                print(f"  {series}: request failed ({e})")
                evs = []
            kal[kind] = (series, evs)
            sample = (evs[0].get("title") if evs else "none")
            print(f"  {series}: {len(evs)} open events (sample title: {sample})")
        if not any(evs for _, evs in kal.values()):
            continue
        try:
            odds = odds_events(sport_key, odds_key)
        except requests.RequestException as e:
            print(f"  odds request failed: {e}")
            continue
        print(f"  sportsbook games: {len(odds)}")
        found = find_picks(label_, word, odds, kal)
        by_type = {}
        for p in found:
            by_type[p["type"]] = by_type.get(p["type"], 0) + 1
        print(f"  picks: {len(found)} {by_type}")
        all_picks += found

    all_picks.sort(key=lambda p: (-p["stars"], -p["score"]))
    add_ai_notes(all_picks)

    history = track(load_history(), all_picks)
    settle(history)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=1)

    with open("picks.json", "w") as f:
        json.dump({
            "updated": datetime.now(timezone.utc).isoformat(),
            "method": "Sportsbook consensus vs Kalshi price, after estimated fees; 35% minimum chance",
            "record": record(history),
            "model": {"weight": MODEL_WEIGHT, "sports": model_status},
            "picks": all_picks,
        }, f, indent=2)
    if MODEL and MODEL.unmatched:
        print("Team names the ratings could not match (model skipped for these): "
              + "; ".join(sorted(MODEL.unmatched)[:20]))
    print(f"Skipped for data checks: {SKIPPED}")
    print(f"Wrote {len(all_picks)} picks.")


if __name__ == "__main__":
    main()
