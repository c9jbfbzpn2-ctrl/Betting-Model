"""
Kalshi board: every moneyline, spread and total Kalshi lists, rated from team data.

No sportsbook odds are used. For each Kalshi game:
  1. Group the game's moneyline, spread and total markets by Kalshi's own event code
     (for example 26NOV07UKTENN), so there is no guessing between data sources.
  2. Work out the home and away team from the Kalshi tickers.
  3. Look both teams up in the ratings built from past results (model.py).
  4. Turn the ratings into a chance for each bet, compare it to Kalshi's price after
     an estimated fee, and give a confidence (1 to 5 stars = how likely the bet is to
     win, per the team data) and a verdict (Take, Lean, Pass).

Everything Kalshi lists is kept (very low to very high confidence). Teams the ratings
do not recognise are shown as "Unrated" with Kalshi's price only.

The ratings have not been shown to beat Kalshi's prices. Treat verdicts as a starting
point, and watch the track record on the page.
"""
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import requests
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import model as M
except Exception as e:                       # the board must still work without ratings
    M = None
    print(f"Ratings model unavailable ({e})")

MODEL = None
SCHED = {}                 # sport -> {(home, away): kickoff ISO time} from ESPN
ET = ZoneInfo("America/New_York")
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"

# label: (moneyline series, spread series, total series, scoring word)
SPORTS = {
    "NFL": ("KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL", "points"),
    "College Football": ("KXNCAAFGAME", "KXNCAAFSPREAD", "KXNCAAFTOTAL", "points"),
    "College Basketball": ("KXNCAAMBGAME", "KXNCAAMBSPREAD", "KXNCAAMBTOTAL", "points"),
    "NBA": ("KXNBAGAME", "KXNBASPREAD", "KXNBATOTAL", "points"),
    "Premier League": ("KXEPLGAME", "KXEPLSPREAD", "KXEPLTOTAL", "goals"),
}
SCHEMA = 3
FEE_RATE = 0.07            # estimated Kalshi taker fee: 7% * price * (1 - price)
MIN_ASK, MAX_ASK = 0.03, 0.97
MAX_GAP = 0.12             # skip contracts whose bid and ask are more than 12 cents apart
MAX_STALE = 0.20           # skip when the ask is far from the last traded price
TAKE_EDGE, LEAN_EDGE, CHECK_EDGE = 0.05, 0.02, 0.15
UNITS = {1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 3.0}
PER_TYPE = 2               # spreads/totals: keep the best this many per game
MAX_PICKS = 1500
SKIPPED = {"games_no_teams": 0, "stale_or_wide": 0, "started": 0}


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


def conf_stars(p):
    """1 to 5 = how likely the bet is to win, per the team data."""
    return 5 if p >= 0.75 else 4 if p >= 0.60 else 3 if p >= 0.45 else 2 if p >= 0.30 else 1


def verdict(e):
    if e is None:
        return "Unrated"
    if e > CHECK_EDGE:
        return "Check"            # a gap this big is usually our data being wrong, not a bargain
    return "Take" if e >= TAKE_EDGE else "Lean" if e >= LEAN_EDGE else "Pass"


def units_for(v, st):
    base = UNITS.get(st, 0)
    return base if v == "Take" else max(0.5, base / 2) if v == "Lean" else 0.0


# ---------- Kalshi parsing ----------
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


def suffix(m):
    return str(m.get("ticker", "")).rsplit("-", 1)[-1]


def live(m):
    return m.get("status") in (None, "", "active", "open")


def code_of(ev):
    et = ev.get("event_ticker", "")
    return et.split("-", 1)[1] if "-" in et else et


def parse_code(code):
    m = re.match(r"(\d{2})([A-Z]{3})(\d{2})(.*)", code)
    if not m:
        return None, ""
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    if m.group(2) not in months:
        return None, m.group(4)
    try:
        d = datetime(2000 + int(m.group(1)), months.index(m.group(2)) + 1, int(m.group(3)))
    except ValueError:
        return None, m.group(4)
    return d, m.group(4)


def game_start(ev, markets, day):
    for src in [ev] + markets:
        for k in ("strike_date", "occurrence_datetime"):
            v = src.get(k)
            if v:
                try:
                    return datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(timezone.utc), True
                except ValueError:
                    pass
    return (day.replace(hour=17, tzinfo=timezone.utc) if day else datetime.now(timezone.utc)), False


def _norm(s):
    s = (s or "").lower().replace("&", " and ").replace("'", "").replace(".", "")
    return [t for t in re.sub(r"[^a-z0-9 ]+", " ", s).split() if t not in ("fc", "afc", "the", "of", "university")]


def same_team(text, team):
    """Compare spread text with one of the two teams in the same game."""
    a, b = _norm(text), _norm(team["name"])
    if not a or not b:
        return 0
    if a == b:
        return 3
    if a[1:] == b[1:] and a[:1] == [team["abbr"].lower()]:
        return 2
    if a and (a[0] == team["abbr"].lower()) and a[1:] == b:
        return 2
    if set(a) <= set(b) or set(b) <= set(a):
        return 1
    return 0


def orient(ev, a, b, soccer):
    """Return (home, away, certain). US ticker order is away+home, soccer is home+away."""
    _, rest = parse_code(code_of(ev))
    first = second = None
    if rest == a["abbr"] + b["abbr"]:
        first, second = a, b
    elif rest == b["abbr"] + a["abbr"]:
        first, second = b, a
    certain = first is not None
    if not certain:
        title = ev.get("title") or ""
        mt = re.split(r"\s+(?:at|@)\s+", title, flags=re.I)
        if len(mt) == 2 and not soccer:
            l, r = mt
            if same_team(l, a) >= same_team(l, b):
                first, second = a, b
            else:
                first, second = b, a
            certain = True
        else:
            first, second = a, b
    return (first, second, certain) if soccer else (second, first, certain)


# ---------- building bets ----------
def bet_sides(m, p_yes, text_yes, text_no, allow_no):
    """Yield (action, text, p, ask, bid, last_for_side) for each side worth showing."""
    ya, yb, na, nb = quotes(m)
    last = kprice(m, "last_price")
    out = []
    sides = [("YES", p_yes, text_yes, ya, yb, last)]
    if allow_no:
        sides.append(("NO", None if p_yes is None else 1 - p_yes, text_no, na, nb,
                      None if last is None else 1 - last))
    for action, p, text, ask, bid, ref in sides:
        if not ask or not bid or ask < MIN_ASK or ask > MAX_ASK or (ask - bid) > MAX_GAP:
            if ask and bid:
                SKIPPED["stale_or_wide"] += 1
            continue
        if ref is not None and abs(ask - ref) > MAX_STALE:
            SKIPPED["stale_or_wide"] += 1
            continue
        out.append((action, text, p, ask, bid, ref))
    return out


def make_bet(base, m, series, kind, side, note):
    action, text, p, ask, bid, ref = side
    if p is not None:
        p = min(max(p, 0.02), 0.98)
    e = None if p is None else p - ask - fee(ask)
    v = verdict(e)
    st = conf_stars(p) if p is not None else 0
    shown = ref if ref is not None else (ask + bid) / 2
    reason = (f"Team ratings give this about a {p:.0%} chance. Kalshi charges {ask * 100:.0f} cents for {action}, "
              f"which implies {ask:.0%}. {note}" if p is not None else
              "No team ratings for these teams yet, so this shows Kalshi's price only.")
    return {
        **base, "type": kind, "bet": text, "side": text, "action": action,
        "kalshi_ticker": m.get("ticker"),
        "event_ticker": m.get("event_ticker") or str(m.get("ticker", "")).rsplit("-", 1)[0],
        "series": series,
        "price_cents": round(ask * 100), "bid_cents": round(bid * 100),
        "last_cents": None if ref is None else round(ref * 100),
        "market_prob": round(shown, 4),
        "fair_prob": None if p is None else round(p, 4),
        "model_prob": None if p is None else round(p, 4),
        "edge": None if e is None else round(e, 4),
        "fee_est": round(fee(ask), 4),
        "max_price_cents": None if p is None else round((max_price(p) or ask) * 100),
        "stars": st, "verdict": v, "units": units_for(v, st),
        "confidence": {0: "Unrated", 1: "Very low", 2: "Low", 3: "Medium", 4: "High", 5: "Very high"}[st],
        "volume": m.get("volume"), "reason": reason, "ai_note": None,
    }


def build_game(sport, series, word, ml_ev, sp_ev, tot_ev):
    soccer = sport == "Premier League"
    ml = [m for m in ml_ev.get("markets", []) if live(m)]
    teams = [{"name": m.get("yes_sub_title") or "", "abbr": suffix(m), "m": m} for m in ml if not is_tie(m)]
    tie = next((m for m in ml if is_tie(m)), None)
    if len(teams) != 2 or not all(t["name"] for t in teams):
        SKIPPED["games_no_teams"] += 1
        return []
    home, away, certain = orient(ml_ev, teams[0], teams[1], soccer)
    day, _ = parse_code(code_of(ml_ev))
    start, exact = game_start(ml_ev, ml, day)
    now = datetime.now(timezone.utc)
    if (exact and start < now) or (day and day < (now - timedelta(days=1)).replace(tzinfo=None)):
        SKIPPED["started"] += 1
        return []

    hk = MODEL.team(sport, home["name"], home["abbr"]) if MODEL else None
    ak = MODEL.team(sport, away["name"], away["abbr"]) if MODEL else None
    pred = MODEL.predict(sport, hk, ak) if (MODEL and hk and ak) else None
    ko = SCHED.get(sport, {}).get((hk, ak)) or SCHED.get(sport, {}).get((ak, hk))
    if ko:
        try:
            start = datetime.fromisoformat(ko.replace("Z", "+00:00")).astimezone(timezone.utc)
            exact = True
            if start < now:
                SKIPPED["started"] += 1
                return []
        except ValueError:
            pass
    note = (f"Ratings: {home['name']} {pred['rating_home']}, {away['name']} {pred['rating_away']} (an average team is about 1500)."
            if pred else "")
    if pred and not certain:
        note += " Home and away were inferred from Kalshi's listing."
    base = {"sport": sport, "game": f"{away['name']} at {home['name']}", "start": start.isoformat(),
            "start_exact": exact,
            "day": (start.astimezone(ET).date().isoformat() if exact else (day.date().isoformat() if day else start.astimezone(ET).date().isoformat())),
            "rated": bool(pred)}
    tp = pred["total"] if pred else None
    out = []

    # moneyline
    cands = []
    for t, side in ((home, "home"), (away, "away")):
        p = (pred["p_home"] if side == "home" else pred["p_away"]) if pred else None
        for s in bet_sides(t["m"], p, f"{t['name']} to win", "", False):
            cands.append(make_bet(base, t["m"], series[0], "Moneyline", s, note))
    if tie:
        for s in bet_sides(tie, pred["p_draw"] if pred else None, "Draw", "", False):
            cands.append(make_bet(base, tie, series[0], "Moneyline", s, note))
    out += cands

    # spreads
    cands = []
    for m in (sp_ev or {}).get("markets", []):
        if not live(m):
            continue
        mt = re.match(r"\s*(.+?)\s+wins by over\s+([\d.]+)", m.get("yes_sub_title") or "", re.I)
        if not mt:
            continue
        n = round(float(mt.group(2)), 1)
        letters = re.match(r"[A-Z]+", suffix(m))
        who = None
        if letters and letters.group(0) == home["abbr"] != away["abbr"]:
            who = "home"
        elif letters and letters.group(0) == away["abbr"] != home["abbr"]:
            who = "away"
        else:
            sh, sa = same_team(mt.group(1), home), same_team(mt.group(1), away)
            who = "home" if sh > sa else "away" if sa > sh else None
        if who is None:
            continue
        p = None
        if pred and soccer and tp:
            p = M.p_soccer_margin_over(tp, n, who)
        elif pred and pred.get("mu") is not None:
            ph = M.p_home_covers(pred, n if who == "home" else -n)
            p = None if ph is None else (ph if who == "home" else 1 - ph)
        name, opp = (home, away) if who == "home" else (away, home)
        for s in bet_sides(m, p, f"{name['name']} -{n:g}", f"{opp['name']} +{n:g}", True):
            cands.append(make_bet(base, m, series[1], "Spread", s, note))
    out += pick_best(cands)

    # totals
    cands = []
    for m in (tot_ev or {}).get("markets", []):
        if not live(m):
            continue
        mt = re.match(r"\s*over\s+([\d.]+)", m.get("yes_sub_title") or "", re.I)
        if not mt:
            continue
        n = round(float(mt.group(1)), 1)
        p = M.p_total_over(tp, n) if (tp and M) else None
        for s in bet_sides(m, p, f"Over {n:g} total {word}", f"Under {n:g} total {word}", True):
            cands.append(make_bet(base, m, series[2], "Total", s, note))
    out += pick_best(cands)
    return out


def pick_best(cands):
    """Keep the most relevant lines: best value when rated, otherwise nearest to a 50/50 price."""
    if any(c["edge"] is not None for c in cands):
        cands = [c for c in cands if c["edge"] is not None]
        cands.sort(key=lambda c: -c["edge"])
    else:
        cands.sort(key=lambda c: abs(c["market_prob"] - 0.5))
    return cands[:PER_TYPE]


# ---------- data pulls ----------
def kalshi_events(series):
    events, cursor = [], None
    for _ in range(10):
        params = {"series_ticker": series, "status": "open", "with_nested_markets": "true", "limit": 200}
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


# ---------- track record ----------
HISTORY_FILE = "history.json"
SETTLE_AFTER_HOURS = 3
GIVE_UP_DAYS = 10
MAX_SETTLE_CALLS = 120


def load_history():
    try:
        with open(HISTORY_FILE) as f:
            h = json.load(f)
    except (OSError, ValueError):
        return []
    keep = [x for x in h if x.get("schema") == SCHEMA]
    if len(keep) != len(h):
        print(f"Dropped {len(h) - len(keep)} old-format history entries (their prices came from the previous method).")
    return keep


def track(history, picks):
    """Log each recommended bet (Take or Lean) once, the first time it appears."""
    seen = {h["id"] for h in history}
    now = datetime.now(timezone.utc).isoformat()
    added = 0
    for p in picks:
        if p["verdict"] not in ("Take", "Lean"):
            continue
        pid = f'{p["kalshi_ticker"]}:{p["action"]}'
        if pid in seen:
            continue
        seen.add(pid)
        added += 1
        history.append({"id": pid, "first_seen": now, "result": None, "schema": SCHEMA,
                        **{k: p.get(k) for k in ("sport", "type", "bet", "action", "game", "start", "price_cents",
                                                 "fair_prob", "market_prob", "fee_est", "stars", "verdict",
                                                 "kalshi_ticker", "event_ticker")}})
    print(f"Tracking {added} new recommended bets ({len(history)} total in history).")
    return history


def fetch_market(ticker, event_ticker):
    try:
        r = requests.get(f"{KALSHI}/markets/{ticker}", timeout=20)
        if r.status_code == 200 and r.json().get("market"):
            return r.json()["market"]
    except requests.RequestException:
        pass
    try:
        r = requests.get(f"{KALSHI}/events/{event_ticker}", params={"with_nested_markets": "true"}, timeout=20)
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
        except (ValueError, AttributeError):
            continue
        age = now - start
        if age.total_seconds() < SETTLE_AFTER_HOURS * 3600:
            continue
        calls += 1
        res = (fetch_market(h["kalshi_ticker"], h["event_ticker"]) or {}).get("result")
        if res in ("yes", "no"):
            h["result"] = "won" if res == h["action"].lower() else "lost"
            settled += 1
        elif age.days >= GIVE_UP_DAYS:
            h["result"] = "void"
    print(f"Checked {calls} unsettled bets, settled {settled}.")


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
    return {"n": n, "wins": wins, "hit_rate": round(wins / n, 4),
            "expected_rate": round(sum(r["fair_prob"] or 0 for r in rows) / n, 4),
            "roi": round(ret / stake, 4) if stake else 0}


def record(history):
    done = [h for h in history if h.get("result") in ("won", "lost")]
    return {
        "overall": summarize(done),
        "high_confidence": summarize([h for h in done if h["stars"] >= 4]),
        "by_stars": {str(i): summarize([h for h in done if h["stars"] == i]) for i in range(1, 6)},
        "by_verdict": {v: summarize([h for h in done if h.get("verdict") == v]) for v in ("Take", "Lean")},
        "by_type": {t: summarize([h for h in done if h["type"] == t]) for t in ("Moneyline", "Spread", "Total")},
        "pending": len([h for h in history if h.get("result") is None]),
        "tracking_since": min((h["first_seen"] for h in history), default=None),
    }


def add_ai_notes(picks):
    key = os.environ.get("ANTHROPIC_API_KEY")
    top = [p for p in picks if p["verdict"] in ("Take", "Lean")][:12]
    if not key or not top:
        return
    data = [{"bet": p["bet"], "type": p["type"], "game": p["game"], "sport": p["sport"],
             "kalshi_price_cents": p["price_cents"], "team_data_chance_pct": round(p["model_prob"] * 100, 1),
             "gap_points": round(p["edge"] * 100, 1), "verdict": p["verdict"]} for p in top]
    prompt = ("You are helping explain prediction-market bets to a casual reader. For each item, write ONE "
              "plain-English sentence (max 30 words) on why the team data likes this bet at Kalshi's price and "
              "what the main risk is. Use ONLY the numbers given. Do not invent injuries, stats, news or "
              "records. Return ONLY a JSON array of strings in the same order.\n\n" + json.dumps(data))
    try:
        r = requests.post("https://api.anthropic.com/v1/messages",
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5"), "max_tokens": 1500,
                                "messages": [{"role": "user", "content": prompt}]}, timeout=90)
        r.raise_for_status()
        text = re.sub(r"^```(?:json)?|```$", "", r.json()["content"][0]["text"].strip(), flags=re.M).strip()
        for p, n in zip(top, json.loads(text)):
            p["ai_note"] = str(n)
        print(f"Claude notes added for {len(top)} bets.")
    except Exception as e:
        print(f"Skipping Claude notes ({e})")


# ---------- main ----------
def slim(ev):
    return {"event_ticker": ev.get("event_ticker"), "title": ev.get("title"), "strike_date": ev.get("strike_date"),
            "markets": [{k: m.get(k) for k in ("ticker", "yes_sub_title", "status", "yes_bid", "yes_ask",
                                                "yes_bid_dollars", "yes_ask_dollars", "no_bid", "no_ask",
                                                "last_price", "last_price_dollars", "volume")}
                        for m in ev.get("markets", [])[:3]]}


def main():
    global MODEL
    debug = {"updated": datetime.now(timezone.utc).isoformat(), "kalshi": {}}
    model_status = {}
    if M is not None and os.environ.get("SKIP_MODEL") != "1":
        try:
            print("Updating team ratings")
            state = M.update(M.load_state())
            M.save_state(state)
            MODEL = M.Model(state)
            model_status = MODEL.status
            for k, v in model_status.items():
                print(f"  {k}: {v['teams']} teams, checked {v['games_checked']} games, gain vs naive {v['gain_vs_baseline']:.1%}")
        except Exception as e:
            print(f"Ratings update failed ({e}); continuing without it")
            debug["ratings_error"] = str(e)
        debug["espn_errors"] = M.ERRORS[:25]
        debug["espn_style"] = M.MODE["name"]

    all_bets = []
    counts = {}
    for sport, (s_ml, s_sp, s_tot, word) in SPORTS.items():
        print(sport)
        data = {}
        for kind, series in (("ml", s_ml), ("sp", s_sp), ("tot", s_tot)):
            try:
                evs = kalshi_events(series)
            except requests.RequestException as e:
                print(f"  {series}: request failed ({e})")
                evs = []
            data[kind] = {code_of(e): e for e in evs}
            debug["kalshi"][series] = {"open_events": len(evs), "sample": slim(evs[0]) if evs else None}
            print(f"  {series}: {len(evs)} open events")
        if MODEL:
            try:
                SCHED[sport] = M.schedule(sport)
                print(f"  kickoff times found: {len(SCHED[sport])}")
            except Exception as e:
                print(f"  kickoff times unavailable ({e})")
        debug.setdefault("kickoff_times", {})[sport] = len(SCHED.get(sport, {}))
        found = []
        for code, ml_ev in data["ml"].items():
            found += build_game(sport, (s_ml, s_sp, s_tot), word, ml_ev, data["sp"].get(code), data["tot"].get(code))
        rated = sum(1 for b in found if b["verdict"] != "Unrated")
        print(f"  bets: {len(found)} ({rated} rated from team data)")
        counts[sport] = {"games": len(data["ml"]), "bets": len(found), "rated": rated}
        all_bets += found

    order = {"Take": 0, "Lean": 1, "Pass": 2, "Check": 3, "Unrated": 4}
    all_bets.sort(key=lambda b: (order[b["verdict"]], -(b["edge"] if b["edge"] is not None else -1)))
    all_bets = all_bets[:MAX_PICKS]
    add_ai_notes(all_bets)

    history = track(load_history(), all_bets)
    settle(history)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=1)
    with open("picks.json", "w") as f:
        json.dump({"updated": datetime.now(timezone.utc).isoformat(),
                   "method": "Kalshi prices vs chances from team ratings; no sportsbook odds",
                   "record": record(history), "model": {"sports": model_status}, "counts": counts,
                   "picks": all_bets}, f, indent=1)
    debug["skipped"] = SKIPPED
    debug["unmatched_teams"] = sorted(MODEL.unmatched)[:60] if MODEL else []
    with open("debug.json", "w") as f:
        json.dump(debug, f, indent=1)
    print(f"Skipped: {SKIPPED}")
    if MODEL and MODEL.unmatched:
        print("Team names the ratings could not match: " + "; ".join(sorted(MODEL.unmatched)[:20]))
    print(f"Wrote {len(all_bets)} bets.")


if __name__ == "__main__":
    main()
