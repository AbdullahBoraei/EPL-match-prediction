"""Leakage-free form features for match prediction.

Every feature for a match is computed ONLY from matches played strictly
before it. The key trick is `.shift(1)` before every rolling window: it
moves each team's history down one row, so the window for match N covers
matches ..., N-2, N-1 — never N itself.
"""

import pandas as pd

# One row per TEAM per match (so each match appears twice: once from the
# home team's perspective, once from the away team's).
POINTS_HOME = {"H": 3, "D": 1, "A": 0}
POINTS_AWAY = {"A": 3, "D": 1, "H": 0}

# Features computed per team, then prefixed with home_/away_ when merged back.
TEAM_FEATURES = [
    "ppg_last5",        # points per game, last 5 matches (recent form)
    "ppg_last38",       # points per game, last ~season (underlying strength)
    "gf_last5",         # goals scored per game, last 5
    "ga_last5",         # goals conceded per game, last 5
    "sot_last5",        # shots on target per game, last 5 (form beyond goals)
    "ppg_venue_last5",  # points per game in last 5 matches AT THIS VENUE
    "rest_days",        # days since the team's previous match
]

FEATURE_COLS = [f"home_{f}" for f in TEAM_FEATURES] + [f"away_{f}" for f in TEAM_FEATURES]


def _team_match_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Reshape the match table to one row per team per match."""
    home = pd.DataFrame({
        "match_id": matches.index,
        "date": matches["date"],
        "team": matches["home_team"],
        "venue": "H",
        "points": matches["result"].map(POINTS_HOME),
        "gf": matches["home_goals"],
        "ga": matches["away_goals"],
        "sot": matches["home_shots_on_target"],
    })
    away = pd.DataFrame({
        "match_id": matches.index,
        "date": matches["date"],
        "team": matches["away_team"],
        "venue": "A",
        "points": matches["result"].map(POINTS_AWAY),
        "gf": matches["away_goals"],
        "ga": matches["home_goals"],
        "sot": matches["away_shots_on_target"],
    })
    return pd.concat([home, away], ignore_index=True).sort_values(["team", "date"])


def _rolling_past_mean(grouped, col: str, window: int, min_periods: int) -> pd.Series:
    """Mean of `col` over the previous `window` rows — current row excluded."""
    return grouped[col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean()
    )


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Return `matches` with home_*/away_* form feature columns added.

    Rows where a team has too little history (start of the dataset, or a
    newly promoted club's first matches) contain NaNs — the caller decides
    whether to drop them.
    """
    long = _team_match_long(matches)

    by_team = long.groupby("team")
    long["ppg_last5"] = _rolling_past_mean(by_team, "points", 5, 3)
    long["ppg_last38"] = _rolling_past_mean(by_team, "points", 38, 10)
    long["gf_last5"] = _rolling_past_mean(by_team, "gf", 5, 3)
    long["ga_last5"] = _rolling_past_mean(by_team, "ga", 5, 3)
    long["sot_last5"] = _rolling_past_mean(by_team, "sot", 5, 3)
    # cap at 30: beyond that it's an off-season break, not extra freshness
    long["rest_days"] = by_team["date"].diff().dt.days.clip(upper=30)

    by_team_venue = long.groupby(["team", "venue"])
    long["ppg_venue_last5"] = _rolling_past_mean(by_team_venue, "points", 5, 3)

    out = matches.copy()
    for venue, prefix in [("H", "home"), ("A", "away")]:
        side = long[long["venue"] == venue].set_index("match_id")[TEAM_FEATURES]
        side.columns = [f"{prefix}_{c}" for c in side.columns]
        out = out.join(side)
    return out


def latest_team_state(matches: pd.DataFrame) -> pd.DataFrame:
    """Each team's CURRENT form — for predicting a hypothetical next fixture.

    Same windows as build_features, but ending at the team's most recent
    match: for a future fixture, "the previous 5 matches" are simply the
    last 5 played, so no shift is needed. Also returns the team's last five
    results as a W/D/L string for display.
    """
    long = _team_match_long(matches)
    states = {}
    for team, g in long.groupby("team"):
        g = g.sort_values("date")

        def tail_mean(frame, col, window, min_periods):
            s = frame[col].tail(window)
            return s.mean() if len(s) >= min_periods else float("nan")

        states[team] = {
            "ppg_last5": tail_mean(g, "points", 5, 3),
            "ppg_last38": tail_mean(g, "points", 38, 10),
            "gf_last5": tail_mean(g, "gf", 5, 3),
            "ga_last5": tail_mean(g, "ga", 5, 3),
            "sot_last5": tail_mean(g, "sot", 5, 3),
            "ppg_home_last5": tail_mean(g[g["venue"] == "H"], "points", 5, 3),
            "ppg_away_last5": tail_mean(g[g["venue"] == "A"], "points", 5, 3),
            "last_match": g["date"].max(),
            "last5_results": "".join(
                {3: "W", 1: "D", 0: "L"}[p] for p in g["points"].tail(5)
            ),
        }
    return pd.DataFrame(states).T


def fixture_features(home_state: pd.Series, away_state: pd.Series, rest_days: float = 7.0) -> pd.DataFrame:
    """Assemble one model-ready feature row for a hypothetical fixture."""
    row = {}
    for prefix, state, venue_col in [
        ("home", home_state, "ppg_home_last5"),
        ("away", away_state, "ppg_away_last5"),
    ]:
        row[f"{prefix}_ppg_last5"] = state["ppg_last5"]
        row[f"{prefix}_ppg_last38"] = state["ppg_last38"]
        row[f"{prefix}_gf_last5"] = state["gf_last5"]
        row[f"{prefix}_ga_last5"] = state["ga_last5"]
        row[f"{prefix}_sot_last5"] = state["sot_last5"]
        row[f"{prefix}_ppg_venue_last5"] = state[venue_col]
        row[f"{prefix}_rest_days"] = rest_days
    return pd.DataFrame([row])[FEATURE_COLS]
