"""
Team-strength ratings (Elo style) for NFL, college football, college basketball,
NBA and the Premier League.

What it does
  * Pulls completed game results from ESPN's public scoreboard feed.
  * Keeps a rating per team, updated after every game (margin of victory counts).
  * Turns the rating gap into a win chance (and, for points sports, a chance of
    covering a spread) using numbers fitted from the games it has seen.
  * Checks itself: before each game it records how well it predicted, and compares
    that to a naive baseline (just "home teams win X% of the time"). A sport's model
    is only used in the picks when it beats that baseline by a clear margin.

State is saved in model_state.json so each daily run only fetches new games.
The first run downloads about 14 months of results, which takes a few minutes.
"""
import json
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone

import requests

STATE_FILE = os.environ.get("MODEL_STATE_FILE", "model_state.json")
ESPN = "https://site.api.espn.com/apis/site/v2/sports/{}/scoreboard"
MEAN = 1500.0

CFG = {
    "NFL": dict(path="football/nfl", kind="margin", k=20, hfa=48, a0=2.0, b0=1 / 25,
                sigma0=13.5, chunk=7, lookback=430, new=1500, min_games=3, params={}),
    "College Football": dict(path="football/college-football", kind="margin", k=25, hfa=55,
                             a0=2.5, b0=1 / 22, sigma0=16.0, chunk=7, lookback=430,
                             new=1400, min_games=3, params={"groups": 80}),
    "NBA": dict(path="basketball/nba", kind="margin", k=20, hfa=85, a0=2.6, b0=1 / 28,
                sigma0=12.0, chunk=4, lookback=430, new=1500, min_games=5, params={}),
    "College Basketball": dict(path="basketball/mens-college-basketball", kind="margin", k=30,
                               hfa=70, a0=3.2, b0=1 / 24, sigma0=11.0, chunk=3, lookback=400,
                               new=1400, min_games=5, params={"groups": 50}),
    "Premier League": dict(path="soccer/eng.1", kind="soccer", k=22, hfa=65, chunk=7,
                           lookback=430, new=1500, min_games=5, params={}),
}
SOCCER_SCALE = 220.0     # rating points per unit of the ordered-logit score
SOCCER_DRAW = 0.56       # draw band width

# A sport's model is used only if, over at least MIN_VALID games, it beats the naive
# baseline's log loss by at least MIN_GAIN (fraction).
MIN_VALID = 250
MIN_GAIN = 0.015
BLEND_WEIGHT = 0.25


# ---------- maths ----------
def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def clamp(p, lo=0.02, hi=0.98):
    return min(max(p, lo), hi)


# ---------- state ----------
def new_sport():
    return {"ratings": {}, "games": {}, "last_date": None, "prev_date": None, "recent": {},
            "calib": {"n": 0, "sx": 0.0, "sy": 0.0, "sxx": 0.0, "sxy": 0.0, "syy": 0.0},
            "val": {"n": 0, "ll_model": 0.0, "ll_base": 0.0, "hw": 0, "dr": 0, "aw": 0}}


def load_state(path=None):
    try:
        with open(path or STATE_FILE) as f:
            s = json.load(f)
    except (OSError, ValueError):
        s = {}
    s.setdefault("sports", {})
    for name in CFG:
        s["sports"].setdefault(name, new_sport())
    return s


def save_state(state, path=None):
    out = json.loads(json.dumps(state))
    for st in out["sports"].values():
        st["ratings"] = {k: round(v, 1) for k, v in st["ratings"].items()}
    out["saved"] = datetime.now(timezone.utc).isoformat()
    with open(path or STATE_FILE, "w") as f:
        json.dump(out, f, separators=(",", ":"))


# ---------- prediction ----------
def calibration(st, cfg):
    c = st["calib"]
    n = c["n"]
    a, b, sigma = cfg["a0"], cfg["b0"], cfg["sigma0"]
    if n >= 150:
        den = n * c["sxx"] - c["sx"] ** 2
        if den > 0:
            bb = (n * c["sxy"] - c["sx"] * c["sy"]) / den
            aa = (c["sy"] - bb * c["sx"]) / n
            sse = c["syy"] - aa * c["sy"] - bb * c["sxy"]
            ss = math.sqrt(max(sse / (n - 2), 1.0))
            if 0.005 < bb < 0.2 and 0 <= aa < 8 and 3 < ss < 30:
                a, b, sigma = aa, bb, ss
    return a, b, sigma


def predict_raw(st, cfg, home, away, neutral=False):
    rh = st["ratings"].get(home, cfg["new"])
    ra = st["ratings"].get(away, cfg["new"])
    if cfg["kind"] == "soccer":
        z = (rh - ra + (0 if neutral else cfg["hfa"])) / SOCCER_SCALE
        ph, pa = sigmoid(z - SOCCER_DRAW), sigmoid(-z - SOCCER_DRAW)
        return {"p_home": ph, "p_away": pa, "p_draw": max(1 - ph - pa, 0.0), "mu": None, "sigma": None}
    a, b, sigma = calibration(st, cfg)
    mu = (0 if neutral else a) + b * (rh - ra)
    ph = phi(mu / sigma)
    return {"p_home": ph, "p_away": 1 - ph, "p_draw": 0.0, "mu": mu, "sigma": sigma}


def p_home_covers(pred, threshold):
    """Chance the home team wins by MORE than `threshold` points (negative = home may lose)."""
    if pred.get("mu") is None:
        return None
    return 1 - phi((threshold - pred["mu"]) / pred["sigma"])


# ---------- learning ----------
def process(st, cfg, g):
    """Return the prediction made before the game, then learn from the result.
    The prediction is None when either team is still too new to rate."""
    home, away = g["home"], g["away"]
    gd = datetime.strptime(g["date"], "%Y-%m-%d")
    prev = st.get("prev_date")
    if prev and (gd - datetime.strptime(prev, "%Y-%m-%d")).days > 100:      # new season
        for t in st["ratings"]:
            st["ratings"][t] = MEAN + (st["ratings"][t] - MEAN) * (2 / 3)
    if not prev or g["date"] > prev:
        st["prev_date"] = g["date"]

    neutral = g.get("neutral", False)
    ready = st["games"].get(home, 0) >= cfg["min_games"] and st["games"].get(away, 0) >= cfg["min_games"]
    pred = predict_raw(st, cfg, home, away, neutral)
    rh = st["ratings"].get(home, cfg["new"])
    ra = st["ratings"].get(away, cfg["new"])
    hs, as_ = g["hs"], g["as"]
    margin = hs - as_

    if cfg["kind"] == "soccer":
        out = "H" if margin > 0 else "A" if margin < 0 else "D"
        if ready:
            v = st["val"]
            p = {"H": pred["p_home"], "D": pred["p_draw"], "A": pred["p_away"]}[out]
            v["ll_model"] += -math.log(clamp(p, 0.01, 0.99))
            n = v["n"]
            base = {"H": (v["hw"] + 1) / (n + 3), "D": (v["dr"] + 1) / (n + 3), "A": (v["aw"] + 1) / (n + 3)}[out]
            v["ll_base"] += -math.log(base)
            v["n"] += 1
            v[{"H": "hw", "D": "dr", "A": "aw"}[out]] += 1
        x = rh - ra + (0 if neutral else cfg["hfa"])
        exp_h = pred["p_home"] + 0.5 * pred["p_draw"]
        res = 1.0 if out == "H" else 0.5 if out == "D" else 0.0
        mult = 1 + math.log(1 + abs(margin)) * 0.6
        delta = cfg["k"] * mult * (res - exp_h)
    else:
        if ready and margin != 0:
            v = st["val"]
            y = 1.0 if margin > 0 else 0.0
            p = clamp(pred["p_home"])
            v["ll_model"] += -(y * math.log(p) + (1 - y) * math.log(1 - p))
            n = v["n"]
            pb = (v["hw"] + 1) / (n + 2)
            v["ll_base"] += -(y * math.log(pb) + (1 - y) * math.log(1 - pb))
            v["n"] += 1
            v["hw"] += int(y)
        if not neutral and ready:
            c = st["calib"]
            x_ = rh - ra
            c["n"] += 1
            c["sx"] += x_
            c["sy"] += margin
            c["sxx"] += x_ * x_
            c["sxy"] += x_ * margin
            c["syy"] += margin * margin
        x = rh - ra + (0 if neutral else cfg["hfa"])
        exp_h = 1 / (1 + 10 ** (-x / 400))
        res = 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
        wdiff = x if margin > 0 else -x
        mult = math.log(abs(margin) + 1) * 2.2 / (wdiff * 0.001 + 2.2) if margin != 0 else 1.0
        delta = cfg["k"] * mult * (res - exp_h)

    st["ratings"][home] = rh + delta
    st["ratings"][away] = ra - delta
    st["games"][home] = st["games"].get(home, 0) + 1
    st["games"][away] = st["games"].get(away, 0) + 1
    return pred if ready else None


# ---------- ESPN fetching ----------
def get_json(url, params, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30, headers={"User-Agent": "picks-model/1.0"})
            if r.status_code == 200:
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(1.5 * (i + 1))
    return None


def parse_events(j):
    out = []
    for ev in j.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        status = (comp.get("status") or ev.get("status") or {}).get("type") or {}
        if not status.get("completed"):
            continue
        if ((ev.get("season") or {}).get("type")) == 1:          # preseason
            continue
        cs = comp.get("competitors") or []
        h = next((c for c in cs if c.get("homeAway") == "home"), None)
        a = next((c for c in cs if c.get("homeAway") == "away"), None)
        if not h or not a:
            continue
        try:
            hs, as_ = float(h["score"]), float(a["score"])
            out.append({"id": str(ev["id"]), "date": str(ev["date"])[:10],
                        "home": h["team"]["displayName"], "away": a["team"]["displayName"],
                        "hs": hs, "as": as_, "neutral": bool(comp.get("neutralSite"))})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def fetch_games(cfg, start, end):
    games, d = [], start
    while d <= end:
        e = min(d + timedelta(days=cfg["chunk"] - 1), end)
        dates = f"{d:%Y%m%d}" if d == e else f"{d:%Y%m%d}-{e:%Y%m%d}"
        j = get_json(ESPN.format(cfg["path"]), dict(cfg["params"], dates=dates, limit=1000))
        if j is None:
            return None
        games += parse_events(j)
        d = e + timedelta(days=1)
        time.sleep(0.25)
    return games


def update(state, today=None, fetch=fetch_games):
    today = today or datetime.now(timezone.utc).date()
    for name, cfg in CFG.items():
        st = state["sports"][name]
        start = (datetime.strptime(st["last_date"], "%Y-%m-%d").date() - timedelta(days=3)) if st["last_date"] \
            else today - timedelta(days=cfg["lookback"])
        games = fetch(cfg, start, today)
        if games is None:
            print(f"  {name}: ESPN request failed, keeping previous ratings")
            continue
        games.sort(key=lambda g: (g["date"], g["id"]))
        floor = start.isoformat() if st["last_date"] else "0000-00-00"
        new = [g for g in games if g["id"] not in st["recent"] and g["date"] >= floor]
        for g in new:
            process(st, cfg, g)
            st["recent"][g["id"]] = g["date"]
            if not st["last_date"] or g["date"] > st["last_date"]:
                st["last_date"] = g["date"]
        if st["last_date"]:
            # keep ids relative to the latest game, so a long gap (offseason) never causes re-counting
            cutoff = (datetime.strptime(st["last_date"], "%Y-%m-%d") - timedelta(days=12)).strftime("%Y-%m-%d")
            st["recent"] = {k: v for k, v in st["recent"].items() if v >= cutoff}
        print(f"  {name}: {len(new)} new games, {len(st['ratings'])} teams, last game {st['last_date']}")
    return state


# ---------- status and name matching ----------
def status(state):
    out = {}
    for name in CFG:
        v = state["sports"][name]["val"]
        gain = (1 - v["ll_model"] / v["ll_base"]) if v["n"] and v["ll_base"] else 0.0
        out[name] = {"games_checked": v["n"], "gain_vs_baseline": round(gain, 4),
                     "enabled": bool(v["n"] >= MIN_VALID and gain >= MIN_GAIN),
                     "teams": len(state["sports"][name]["ratings"])}
    return out


def norm(s):
    s = s.lower().replace("&", " and ").replace("'", "").replace(".", "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return [t for t in s.split() if t not in ("fc", "afc", "the", "of", "university")]


class Model:
    def __init__(self, state):
        self.state = state
        self.status = status(state)
        self._idx = {}
        self.unmatched = set()

    def enabled(self, sport):
        return self.status.get(sport, {}).get("enabled", False)

    def _find(self, sport, name):
        key = (sport, name)
        if key in self._idx:
            return self._idx[key]
        teams = list(self.state["sports"][sport]["ratings"])
        want = norm(name)
        found = None
        exact = [t for t in teams if norm(t) == want]
        if len(exact) == 1:
            found = exact[0]
        else:
            sub = [t for t in teams if want and norm(t) and
                   (set(want) <= set(norm(t)) or set(norm(t)) <= set(want))]
            if len(sub) == 1:
                found = sub[0]
            elif sport in ("NFL", "NBA", "Premier League"):
                nick = [t for t in teams if norm(t)[-1:] == want[-1:] and want]
                if len(nick) == 1:
                    found = nick[0]
        if not found:
            self.unmatched.add(f"{sport}: {name}")
        self._idx[key] = found
        return found

    def predict(self, sport, home, away, neutral=False):
        if sport not in CFG:
            return None
        h, a = self._find(sport, home), self._find(sport, away)
        if not h or not a or h == a:
            return None
        st, cfg = self.state["sports"][sport], CFG[sport]
        if st["games"].get(h, 0) < cfg["min_games"] or st["games"].get(a, 0) < cfg["min_games"]:
            return None
        return predict_raw(st, cfg, h, a, neutral)
