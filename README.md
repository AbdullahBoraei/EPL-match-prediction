# Can simple stats beat the bookmakers?

**Predicting Premier League match outcomes from 11 seasons of data — and measuring honestly how far a weekend model gets against Bet365.**

An end-to-end data science project: data collection, cleaning, exploratory analysis, leakage-free feature engineering, model selection, evaluation beyond accuracy, and an interactive [Streamlit app](#try-the-app).

## TL;DR

Trained on 2015-16 → 2023-24, tested once on two fully unseen seasons (2024-25, 2025-26 — 750 matches):

| Contestant | Accuracy | Log loss |
|---|---|---|
| Naive baseline (class frequencies) | 41.9% | 1.082 |
| **Logistic regression (this project)** | **50.9%** | **1.018** |
| Gradient boosting, tuned | 48.5% | 1.041 |
| Bookmaker (Bet365, margin removed) | 51.6% | 0.995 |

**No, we don't beat the bookmakers — and finding out precisely *by how much* we lose is the point.** Fourteen rolling form features and a linear model close most of the gap between "predict the most common outcome" and an institution with money on the line. The last percentage point is where the hard information lives: injuries, lineups, motivation.

## The question

Football prediction is a perfect setting for honest ML evaluation, because the benchmark bets back. Bookmaker odds encode probability estimates backed by real money, so instead of reporting an accuracy number in a vacuum, every result here is bracketed between a naive floor and a professional ceiling.

**Data:** [football-data.co.uk](https://www.football-data.co.uk/englandm.php) — 4,180 matches (11 seasons × 380), with results, match statistics, and Bet365 pre-match odds.

## What the data revealed (notebook 02)

**Home advantage is real, large — and partly made of crowd noise.** Home teams win ~44% of matches overall. But in 2020-21, the season played in empty stadiums, home wins fell to 37.9% and *away* wins rose to 40.3% — the only season in the dataset where playing at home was a disadvantage. A natural experiment hiding in plain sight:

![Match outcomes by season](figures/outcomes_by_season.png)

**The bookmakers' favourite wins only 54.9% of the time, and their probabilities are almost perfectly calibrated.** When Bet365 says 60%, it happens ~60% of the time. This reset my expectations before training anything: if the professionals top out near 55%, a weekend project will not hit 70%, and any model that claims to is leaking data.

**The draw has *never* been the favourite — 0 matches out of 4,180.** Even in dead-even matchups, draws happen only ~30% of the time, below the 1/3 threshold needed to ever be the single most likely outcome:

![Draw rate by matchup evenness](figures/draw_rate_by_evenness.png)

This predicted, before any modeling, that a most-likely-outcome classifier would never predict a draw — and explained it as a property of football rather than a bug.

## Features without leakage (notebook 03)

At prediction time you know a team's *history*, nothing else. Each side gets seven rolling features — recent form (points/game over last 5), underlying strength (points/game over last 38), goals for/against, shots on target, venue-specific form, rest days — computed with `.shift(1)` before every rolling window so a match's features come only from strictly earlier matches. The notebook verifies this with a hand-computed spot check, not just a claim.

The in-match statistics (shots, corners) appear **only** inside these historical windows. Using them directly is the classic way to build a football model that looks brilliant and is useless.

## Model selection: the simple model won

Candidates were compared with time-series cross-validation *inside the training years only* — the test seasons played no role in any decision:

- Logistic regression: CV log loss **0.981**
- Gradient boosting (tuned over a small grid): CV log loss 0.995

CV picked the linear model, and the test set later confirmed it. With ~3,200 training matches and 14 features that already summarize the relevant history, a flexible model mostly finds noise. "Use the fanciest model" lost to "use the right-sized model."

## Evaluation beyond accuracy

**The confusion matrix shows the promised empty draw column** — 0 draws predicted in 750 matches, exactly as the EDA said it must be:

![Confusion matrix](figures/confusion_matrix.png)

**The model's probabilities are honest.** Its calibration curve tracks the diagonal nearly as well as the bookmaker's, and confidence is informative: accuracy is ~46% on toss-ups but **69% when the model puts one outcome above 70%**.

![Model calibration](figures/model_calibration.png)

**The worst errors are irreducible.** The ten biggest misses are shock results — Arsenal losing at home to West Ham at odds of 1.27, promoted Ipswich beating Chelsea — that the bookmakers also priced as near-impossible.

**What did it learn?** The coefficients say season-long strength (`ppg_last38`) dominates recent 5-match form: *"form is temporary, class is permanent"* is measurable.

## What surprised me

1. **Away wins beat home wins in the empty-stadium season.** I expected home advantage to shrink without crowds, not invert.
2. **The draw is structurally unpredictable** — not "hard", but *never* the rational top pick, for my model and for the bookmakers alike.
3. **The tuned gradient boosting model lost to plain logistic regression** on both CV and test. I expected a small win for boosting.
4. **How close 14 transparent features get to Bet365** — 0.7 accuracy points. The gap is real, but far smaller than I assumed.

## What I'd do differently with more time

- **Player-level data** (injuries, lineups, transfers) — almost certainly where the remaining bookmaker edge lives.
- **Elo-style ratings** instead of raw rolling points, which handle promoted teams and opponent strength more gracefully.
- **A betting simulation**: the model's probabilities occasionally disagree with the odds — would a value-betting strategy have made or lost money after the ~4.5% margin? (My calibration curve says: probably lost, which is exactly why it's worth showing.)
- More leagues, to test whether the conclusions transfer.

## Try the app

```bash
git clone https://github.com/AbdullahBoraei/EPL-match-prediction.git
cd EPL-match-prediction
python -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Pick any two current Premier League teams and get win/draw/loss probabilities with each team's current form — plus a plain-language section on how much to trust the numbers.

## Repository guide

```
notebooks/01_data_and_cleaning.ipynb        download, clean, sanity-check 4,180 matches
notebooks/02_eda.ipynb                      home advantage, scoring, the bookmaker benchmark
notebooks/03_modeling_and_evaluation.ipynb  features, model selection, honest evaluation
src/                                        shared logic (data, features) used by notebooks AND app
app/streamlit_app.py                        interactive predictor
data/raw/                                   original season CSVs (never edited)
models/                                     trained model artifact
```

To reproduce from scratch: run the three notebooks in order (each is self-contained and re-downloads/re-builds what it needs).

---

*Data: [football-data.co.uk](https://www.football-data.co.uk) (free historical data). Educational project — not betting advice.*
