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
                sigma0=13.5, chunk=7, lookback=430, new=1500, min_games=3, params={}, alpha=0.15, tsig0=10.5),
    "College Football": dict(path="football/college-football", kind="margin", k=25, hfa=55,
                             a0=2.5, b0=1 / 22, sigma0=16.0, chunk=7, lookback=430,
                             new=1400, min_games=3, params={"groups": 80}, alpha=0.15, tsig0=14.0),
    "NBA": dict(path="basketball/nba", kind="margin", k=20, hfa=85, a0=2.6, b0=1 / 28,
                sigma0=12.0, chunk=4, lookback=430, new=1500, min_games=5, params={}, alpha=0.08, tsig0=17.5),
    "College Basketball": dict(path="basketball/mens-college-basketball", kind="margin", k=30,
                               hfa=70, a0=3.2, b0=1 / 24, sigma0=11.0, chunk=3, lookback=400,
                               new=1400, min_games=5, params={"groups": 50}, alpha=0.10, tsig0=12.5),
    "Premier League": dict(path="soccer/eng.1", kind="soccer", k=22, hfa=65, chunk=7,
                           lookback=430, new=1500, min_games=5, params={}, alpha=0.10),
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
            "meta": {}, "off": {}, "def": {}, "lg": {"s": 0.0, "n": 0},
            "tcal": {"n": 0, "se": 0.0, "se2": 0.0},
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
        for k, v in new_sport().items():           # states saved by older versions
            s["sports"][name].setdefault(k, v)
    return s


def save_state(state, path=None):
    out = json.loads(json.dumps(state))
    for st in out["sports"].values():
        st["ratings"] = {k: round(v, 1) for k, v in st["ratings"].items()}
        st["off"] = {k: round(v, 2) for k, v in st.get("off", {}).items()}
        st["def"] = {k: round(v, 2) for k, v in st.get("def", {}).items()}
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


def total_pred(st, cfg, home, away, raw=False):
    """Expected total points (or goals) from each team's recent scoring and conceding."""
    lg = st["lg"]
    if lg["n"] < 40:
        return None
    lgm = lg["s"] / lg["n"]
    oh, dh = st["off"].get(home, lgm), st["def"].get(home, lgm)
    oa, da = st["off"].get(away, lgm), st["def"].get(away, lgm)
    eh, ea = oh + da - lgm, oa + dh - lgm
    if cfg["kind"] == "soccer":
        return {"lam_h": max(0.2, eh + 0.15), "lam_a": max(0.2, ea - 0.15)}
    c = st["tcal"]
    mean, sigma = eh + ea, cfg["tsig0"]
    if c["n"] >= 100 and not raw:
        bias = c["se"] / c["n"]
        var = c["se2"] / c["n"] - bias * bias
        if 3 < math.sqrt(max(var, 0)) < 40:
            mean, sigma = mean + bias, math.sqrt(var)
    return {"mean": mean, "sigma": sigma}


def poisson_pmf(lam, kmax=16):
    out, p = [], math.exp(-lam)
    for k in range(kmax + 1):
        out.append(p)
        p *= lam / (k + 1)
    return out


def p_total_over(tp, line):
    if tp is None:
        return None
    if "lam_h" in tp:
        pm = poisson_pmf(tp["lam_h"] + tp["lam_a"], 20)
        return max(0.0, 1 - sum(pm[: int(math.floor(line)) + 1]))
    return 1 - phi((line - tp["mean"]) / tp["sigma"])


def p_soccer_margin_over(tp, threshold, side):
    """Chance `side` ("home"/"away") wins by more than `threshold` goals."""
    ph, pa = poisson_pmf(tp["lam_h"]), poisson_pmf(tp["lam_a"])
    tot = 0.0
    for h, x in enumerate(ph):
        for a, y in enumerate(pa):
            m = (h - a) if side == "home" else (a - h)
            if m > threshold:
                tot += x * y
    return tot


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

    # scoring model (used for totals)
    tp = total_pred(st, cfg, home, away, raw=True) if ready else None      # raw: calibrate against the uncorrected estimate
    if tp is not None and "mean" in tp:
        err = (hs + as_) - tp["mean"]
        c = st["tcal"]
        c["n"] += 1
        c["se"] += err
        c["se2"] += err * err
    lg = st["lg"]
    lgm = lg["s"] / lg["n"] if lg["n"] else (hs + as_) / 2
    al = cfg["alpha"]
    for team, pf, pa_ in ((home, hs, as_), (away, as_, hs)):
        o, d = st["off"].get(team, lgm), st["def"].get(team, lgm)
        st["off"][team] = o + al * (pf - o)
        st["def"][team] = d + al * (pa_ - d)
    lg["s"] += hs + as_
    lg["n"] += 2

    st["ratings"][home] = rh + delta
    st["ratings"][away] = ra - delta
    if g.get("hmeta"):
        st["meta"][home] = g["hmeta"]
        st["meta"][away] = g["ameta"]
    st["games"][home] = st["games"].get(home, 0) + 1
    st["games"][away] = st["games"].get(away, 0) + 1
    return pred if ready else None


# ---------- ESPN fetching ----------
ERRORS = []     # recent fetch problems, saved to debug.json so they can be read later


UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def get_json(url, params, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30, headers={"User-Agent": UA, "Accept": "application/json"})
            if r.status_code == 200:
                return r.json()
            note = f"HTTP {r.status_code}: {r.text[:120]!r}"
        except (requests.RequestException, ValueError) as e:
            note = f"{type(e).__name__}: {str(e)[:120]}"
        if len(ERRORS) < 25:
            ERRORS.append({"url": url, "params": {k: str(v) for k, v in params.items()}, "error": note})
        if i < tries - 1:
            time.sleep(1.5 * (i + 1))
    return None


def _completed(ev, comp):
    for node in (ev.get("status"), comp.get("status")):
        t = (node or {}).get("type") or {}
        if t.get("completed") is True or t.get("state") == "post" or "FINAL" in str(t.get("name", "")).upper():
            return True
    return False


def parse_events(j):
    out = []
    for ev in j.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        if not _completed(ev, comp):
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
            ht, at = h["team"], a["team"]

            def meta(t):
                return {"loc": t.get("location"), "name": t.get("name"), "abbr": t.get("abbreviation"),
                        "short": t.get("shortDisplayName")}
            out.append({"id": str(ev["id"]), "date": str(ev["date"])[:10],
                        "home": ht["displayName"], "away": at["displayName"],
                        "hs": hs, "as": as_, "neutral": bool(comp.get("neutralSite")),
                        "hmeta": meta(ht), "ameta": meta(at)})
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ESPN has answered some date styles with HTTP 400, so several request styles are tried
# and the first one that works is remembered for the rest of the run.
VARIANTS = ["range+limit", "range", "day"]
MODE = {"name": None}


def _chunk(cfg, d, e, variant, tries):
    if variant == "day":
        spans = [(d + timedelta(days=i),) * 2 for i in range((e - d).days + 1)]
    else:
        spans = [(d, e)]
    games = []
    for a, b in spans:
        dates = f"{a:%Y%m%d}" if a == b else f"{a:%Y%m%d}-{b:%Y%m%d}"
        params = dict(cfg["params"], dates=dates)
        if variant == "range+limit":
            params["limit"] = 1000
        j = get_json(ESPN.format(cfg["path"]), params, tries=tries)
        if j is None:
            return None
        games += parse_events(j)
        if variant == "day":
            time.sleep(0.1)
    return games


def fetch_chunk(cfg, d, e):
    if MODE["name"]:
        return _chunk(cfg, d, e, MODE["name"], 3)
    for v in VARIANTS:
        g = _chunk(cfg, d, e, v, 1)
        if g is not None:
            MODE["name"] = v
            print(f"  ESPN request style that works: {v}")
            return g
    return None


def fetch_games(cfg, start, end):
    games, d = [], start
    while d <= end:
        e = min(d + timedelta(days=cfg["chunk"] - 1), end)
        g = fetch_chunk(cfg, d, e)
        if g is None:
            return None
        games += g
        d = e + timedelta(days=1)
        time.sleep(0.2)
    return games


def schedule(sport, today=None, days=8):
    """Upcoming kickoff times from ESPN: {(home displayName, away displayName): ISO time}."""
    cfg = CFG[sport]
    today = today or datetime.now(timezone.utc).date()
    out = {}
    for i in range(days):
        d = today + timedelta(days=i)
        j = get_json(ESPN.format(cfg["path"]), dict(cfg["params"], dates=f"{d:%Y%m%d}"), tries=2)
        for ev in (j or {}).get("events", []):
            try:
                comp = (ev.get("competitions") or [{}])[0]
                cs = comp.get("competitors") or []
                h = next(c for c in cs if c.get("homeAway") == "home")
                a = next(c for c in cs if c.get("homeAway") == "away")
                out[(h["team"]["displayName"], a["team"]["displayName"])] = str(ev["date"])
            except (StopIteration, KeyError, TypeError):
                continue
        time.sleep(0.15)
    return out


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
    s = (s or "").lower().replace("&", " and ").replace("'", "").replace(".", "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(t for t in s.split() if t not in ("fc", "afc", "the", "of", "university"))


POOL_KINDS = ("NFL", "NBA", "Premier League")      # leagues where a nickname alone is unambiguous


class Model:
    def __init__(self, state):
        self.state = state
        self.status = status(state)
        self._keys = {}
        self._cache = {}
        self.unmatched = set()

    def enabled(self, sport):
        return self.status.get(sport, {}).get("enabled", False)

    def _build(self, sport):
        st = self.state["sports"][sport]
        owners = {}
        for team in st["ratings"]:
            m = st["meta"].get(team, {})
            keys = {norm(team), norm(m.get("short")), norm(m.get("abbr")), norm(m.get("loc"))}
            nick = norm(m.get("name"))
            if m.get("abbr") and nick:
                keys.add(norm(m["abbr"]) + " " + nick)
            if sport in POOL_KINDS:
                if nick:
                    keys.add(nick)
                if m.get("loc") and nick:
                    keys.add(norm(m["loc"]) + " " + nick[0])         # "Los Angeles C"
            for k in keys:
                if k:
                    owners.setdefault(k, set()).add(team)
        # a key shared by two teams is ambiguous, so it is never used
        self._keys[sport] = {k: next(iter(v)) for k, v in owners.items() if len(v) == 1}

    def team(self, sport, name, abbr=None):
        """Exact matches only. 'Kentucky' never matches 'Western Kentucky'."""
        key = (sport, name, abbr)
        if key in self._cache:
            return self._cache[key]
        if sport not in self._keys:
            self._build(sport)
        keys = self._keys[sport]
        found = keys.get(norm(name)) or (keys.get(norm(abbr)) if abbr else None)
        if not found:
            self.unmatched.add(f"{sport}: {name}")
        self._cache[key] = found
        return found

    def ready(self, sport, home, away):
        st, cfg = self.state["sports"][sport], CFG[sport]
        return (home in st["ratings"] and away in st["ratings"] and home != away
                and st["games"].get(home, 0) >= cfg["min_games"] and st["games"].get(away, 0) >= cfg["min_games"])

    def predict(self, sport, home, away, neutral=False):
        """home/away are rating keys from team()."""
        if sport not in CFG or not self.ready(sport, home, away):
            return None
        st, cfg = self.state["sports"][sport], CFG[sport]
        pred = predict_raw(st, cfg, home, away, neutral)
        pred["total"] = total_pred(st, cfg, home, away)
        pred["rating_home"] = round(st["ratings"][home])
        pred["rating_away"] = round(st["ratings"][away])
        return pred
