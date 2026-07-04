"""Download and clean Premier League match data from football-data.co.uk.

The raw files have one row per match and 100+ columns (statistics plus odds
from many bookmakers). We keep only the columns the project needs and give
them readable names.
"""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

# Season codes as used in football-data.co.uk URLs: "1516" = 2015-16, etc.
SEASONS = [
    "1516", "1617", "1718", "1819", "1920",
    "2021", "2122", "2223", "2324", "2425", "2526",
]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Raw column -> readable name. FT = full time, H/A = home/away,
# B365* = Bet365 odds recorded before kick-off.
COLUMNS = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",  # H = home win, D = draw, A = away win
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "B365H": "odds_home",
    "B365D": "odds_draw",
    "B365A": "odds_away",
}


def download_raw(force: bool = False) -> None:
    """Download each season's CSV into data/raw, skipping files that exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        target = RAW_DIR / f"E0_{season}.csv"
        if target.exists() and not force:
            continue
        urlretrieve(BASE_URL.format(season=season), target)
        print(f"downloaded {target.name}")


def season_label(code: str) -> str:
    """'1516' -> '2015-16'."""
    return f"20{code[:2]}-{code[2:]}"


def build_match_table() -> pd.DataFrame:
    """Combine all raw season files into one clean, chronologically sorted table."""
    frames = []
    for season in SEASONS:
        df = pd.read_csv(RAW_DIR / f"E0_{season}.csv")
        df = df[list(COLUMNS)].rename(columns=COLUMNS)
        df["season"] = season_label(season)
        frames.append(df)

    matches = pd.concat(frames, ignore_index=True)

    # Dates are day-first ("13/08/15" or "13/08/2015" depending on the season).
    matches["date"] = pd.to_datetime(matches["date"], dayfirst=True, format="mixed")

    matches = matches.sort_values("date").reset_index(drop=True)
    return matches


def save_match_table(matches: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "matches.csv"
    matches.to_csv(out, index=False)
    return out


def load_match_table() -> pd.DataFrame:
    """Load the clean match table produced by notebook 01."""
    return pd.read_csv(PROCESSED_DIR / "matches.csv", parse_dates=["date"])


def add_implied_probs(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert bookmaker odds to outcome probabilities.

    Decimal odds of 2.0 imply a probability of 1/2.0 = 50%. But a bookmaker's
    three implied probabilities sum to slightly MORE than 1 — the excess is
    their profit margin ("overround"). Dividing by the sum removes the margin,
    leaving the bookmaker's actual probability estimates.
    """
    inv = 1.0 / matches[["odds_home", "odds_draw", "odds_away"]]
    overround = inv.sum(axis=1)
    out = matches.copy()
    out["book_prob_home"] = inv["odds_home"] / overround
    out["book_prob_draw"] = inv["odds_draw"] / overround
    out["book_prob_away"] = inv["odds_away"] / overround
    out["overround"] = overround
    return out
