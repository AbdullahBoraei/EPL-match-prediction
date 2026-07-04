"""Interactive EPL match predictor.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data import load_match_table
from src.features import fixture_features, latest_team_state

st.set_page_config(page_title="EPL Match Predictor", page_icon="⚽", layout="centered")

RESULT_DOTS = {"W": "🟢", "D": "⚪", "L": "🔴"}


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "match_predictor.joblib")


@st.cache_data
def load_states():
    matches = load_match_table()
    state = latest_team_state(matches)
    current_teams = sorted(matches.loc[matches["season"] == matches["season"].max(), "home_team"].unique())
    return state, current_teams


artifact = load_model()
state, teams = load_states()

st.title("⚽ EPL Match Predictor")
st.caption(
    f"Logistic regression over rolling form features · trained on matches through "
    f"{artifact['trained_through']} · test accuracy {artifact['test_metrics']['accuracy']:.1%} "
    f"(bookmakers: 51.6% on the same matches)"
)

col_home, col_away = st.columns(2)
home = col_home.selectbox("Home team", teams, index=teams.index("Arsenal"))
away = col_away.selectbox("Away team", teams, index=teams.index("Liverpool"))

if home == away:
    st.warning("Pick two different teams.")
    st.stop()

X = fixture_features(state.loc[home], state.loc[away])
model = artifact["model"]
proba = dict(zip(model.classes_, model.predict_proba(X)[0]))

st.subheader(f"{home} (home) vs {away}")

m1, m2, m3 = st.columns(3)
m1.metric(f"{home} win", f"{proba['H']:.0%}")
m2.metric("Draw", f"{proba['D']:.0%}")
m3.metric(f"{away} win", f"{proba['A']:.0%}")

chart_data = pd.DataFrame(
    {"probability": [proba["H"], proba["D"], proba["A"]]},
    index=[f"{home} win", "Draw", f"{away} win"],
)
st.bar_chart(chart_data, horizontal=True, height=180)

st.divider()

st.subheader("Current form")
f1, f2 = st.columns(2)
for col, team in [(f1, home), (f2, away)]:
    s = state.loc[team]
    dots = " ".join(RESULT_DOTS[c] for c in s["last5_results"])
    col.markdown(f"**{team}**")
    col.markdown(f"last 5: {dots} (oldest → newest)")
    col.markdown(
        f"- points/game, last 5: **{s['ppg_last5']:.2f}**\n"
        f"- points/game, last 38: **{s['ppg_last38']:.2f}**\n"
        f"- goals for / against (last 5): **{s['gf_last5']:.1f} / {s['ga_last5']:.1f}**\n"
        f"- last match: {pd.Timestamp(s['last_match']).date()}"
    )

with st.expander("How much should you trust this?"):
    st.markdown(
        """
        Honest answer: **football is mostly noise.** On two full unseen seasons this model
        is right **50.9%** of the time — clearly better than always predicting a home win
        (~42%), and within one point of the bookmakers (51.6%), whose odds are backed by
        real money and vastly more information.

        Things to know when reading the numbers:

        - **The favourite here will never be the draw.** Draws never exceed ~1/3 probability
          (that's football, not a bug — even bookmakers have *never* priced a draw as the
          favourite in the 4,180 matches studied). The draw probability shown is still meaningful.
        - **Confidence is informative:** when this model puts one outcome above 70%, it's
          right about 69% of the time; on toss-ups it's ~46%.
        - The model knows **team form only** — no injuries, suspensions, transfers, or cup
          fixtures. Both teams are assumed to have a normal week of rest.

        Built as a portfolio project — see the notebooks in the
        [GitHub repo](https://github.com/AbdullahBoraei/EPL-match-prediction) for the full
        analysis. Educational only; not betting advice.
        """
    )
