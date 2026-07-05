"""Export one season's fixtures with model probabilities to CSV.

This is the bridge to the C++ Monte Carlo engine (epl-monte-carlo):
Python owns modeling, C++ owns simulation, and this file defines the
contract between them. One row per fixture:

    match_id, date, home_team, away_team, p_home, p_draw, p_away, actual_result

`actual_result` (H/D/A) is included so the simulation's output can be
validated against what really happened.

Run from the project root:
    .venv/bin/python scripts/export_fixtures.py --season 2025-26
"""

import argparse
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import load_match_table
from src.features import FEATURE_COLS, build_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--out", default=None,
                        help="output path (default: data/exports/fixtures_<season>.csv)")
    args = parser.parse_args()

    matches = load_match_table()
    if args.season not in set(matches["season"]):
        raise SystemExit(f"season {args.season!r} not in data "
                         f"(have {sorted(set(matches['season']))})")

    artifact = joblib.load(ROOT / "models" / "match_predictor.joblib")
    model = artifact["model"]

    # Features for ALL matches (rolling windows need the full history),
    # then keep only the season we're exporting.
    features = build_features(matches)
    season = features[features["season"] == args.season].copy()

    # Rows with NaN features (a promoted club's first matches, before its
    # rolling windows have enough history) can't go through the model.
    # The simulation still needs all 380 fixtures, so those few rows fall
    # back to the league's long-run base rates computed from every match
    # BEFORE this season -- the honest "we know nothing about this team
    # yet" prior. The fallback is flagged so downstream analysis can see it.
    predictable = season[FEATURE_COLS].notna().all(axis=1)
    base = matches[matches["season"] < args.season]["result"].value_counts(normalize=True)

    for col in ("p_home", "p_draw", "p_away"):
        season[col] = 0.0
    proba = model.predict_proba(season.loc[predictable, FEATURE_COLS])
    for cls, col in zip(model.classes_, proba.T):
        season.loc[predictable, f"p_{'home' if cls == 'H' else 'draw' if cls == 'D' else 'away'}"] = col
    season.loc[~predictable, ["p_home", "p_draw", "p_away"]] = [base["H"], base["D"], base["A"]]
    season["fallback"] = (~predictable).astype(int)

    out = season.sort_values("date").reset_index(drop=True)
    out.index.name = "match_id"
    out_path = Path(args.out) if args.out else ROOT / "data" / "exports" / f"fixtures_{args.season.replace('-', '_')}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out[["date", "home_team", "away_team", "p_home", "p_draw", "p_away",
         "result", "fallback"]].rename(columns={"result": "actual_result"}).to_csv(out_path)

    n_fallback = int(out["fallback"].sum())
    print(f"wrote {len(out)} fixtures to {out_path}")
    print(f"model-predicted: {len(out) - n_fallback}, base-rate fallback: {n_fallback}")
    print(f"probability rows sum to 1.0: "
          f"{bool(((out.p_home + out.p_draw + out.p_away) - 1.0).abs().max() < 1e-9)}")


if __name__ == "__main__":
    main()
