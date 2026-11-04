"""
Backtest the rating model against real NFL closing spreads (nflverse data).
Run:  python scripts/backtest_nfl.py
Walk-forward: every game is predicted using only games played before it.
"""
import csv
import io
import math
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(__file__))
import model as M

URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
FIRST_TEST_SEASON = 2008
WEIGHTS = (0.0, 0.1, 0.25, 0.4, 0.6)


def main():
    rows = list(csv.DictReader(io.StringIO(requests.get(URL, timeout=60).text)))
    games = []
    for r in rows:
        try:
            games.append({"id": r["game_id"], "date": r["gameday"], "season": int(r["season"]),
                          "home": r["home_team"], "away": r["away_team"],
                          "hs": float(r["home_score"]), "as": float(r["away_score"]),
                          "neutral": r.get("location") == "Neutral",
                          "line": float(r["spread_line"]) if r.get("spread_line") not in ("", "NA", None) else None})
        except (ValueError, KeyError):
            continue            # unplayed games have no score
    games.sort(key=lambda g: (g["date"], g["id"]))
    cfg, st = M.CFG["NFL"], M.new_sport()
    res = []
    for g in games:
        pred = M.process(st, cfg, g)
        if pred and g["season"] >= FIRST_TEST_SEASON and g["line"] is not None:
            res.append((g, pred))
    n = len(res)
    print(f"Games tested: {n} (seasons {FIRST_TEST_SEASON}+), all predicted before being played")
    print(f"Model vs naive baseline, win-probability log loss: gain {M.status({'sports': {'NFL': st, **{k: M.new_sport() for k in M.CFG if k != 'NFL'}}})['NFL']['gain_vs_baseline']:.2%}")

    print("\nError in predicting the final margin (lower is better):")
    for w in WEIGHTS:
        se = ae = 0.0
        for g, p in res:
            est = (1 - w) * g["line"] + w * p["mu"]
            e = (g["hs"] - g["as"]) - est
            se += e * e
            ae += abs(e)
        tag = "  <- sportsbook line only" if w == 0 else ""
        print(f"  {int(w*100):>3}% model weight: RMSE {math.sqrt(se/n):.3f}  MAE {ae/n:.3f}{tag}")

    print("\nBetting the model's side against the closing spread:")
    for gap in (1.0, 2.0, 3.0):
        w = l = 0
        for g, p in res:
            d = p["mu"] - g["line"]
            if abs(d) < gap:
                continue
            m = g["hs"] - g["as"]
            cover = m - g["line"]
            if cover == 0:
                continue
            won = (cover > 0) == (d > 0)
            w += won
            l += (not won)
        t = w + l
        print(f"  gap >= {gap:.0f} pts: {w}-{l} ({w/t:.1%} over {t} bets; breakeven at -110 is 52.4%)" if t else f"  gap >= {gap}: none")


if __name__ == "__main__":
    main()
