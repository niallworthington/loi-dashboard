import streamlit as st
import pandas as pd
import numpy as np
import math

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="LOI 2026 Analytics",
    layout="wide"
)

st.title("🏆 League of Ireland 2026 Dashboard - NiallW Football")

# ---------------------------------------------------
# MODEL CONSTANTS
# ---------------------------------------------------
LEAGUE_AVG_GOALS_HOME = 1.389
LEAGUE_AVG_GOALS_AWAY = 1.05
ITERATIONS = 1000

# ---------------------------------------------------
# DATA LOADING & UNIQUE TAGGING
# ---------------------------------------------------
@st.cache_data
def load_data():
    results_raw = pd.read_csv("current_results.csv")
    fixtures_raw = pd.read_csv("fixtures.csv")
    ratings = pd.read_csv("ratings.csv")

    for df in [results_raw, fixtures_raw, ratings]:
        for col in ['Home', 'Away', 'Club', 'Team']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

    # Tag played results
    played_results = results_raw.dropna(subset=['HomeGoals', 'AwayGoals']).copy()
    played_results['FixtureTag'] = played_results['Home'] + " v " + played_results['Away']
    played_results['Instance'] = played_results.groupby(['Home', 'Away']).cumcount() + 1
    played_results['UniqueID'] = played_results['FixtureTag'] + " " + played_results['Instance'].astype(str)

    # Tag fixtures — offset instance by how many times that pairing already appears in played results
    fixtures_raw['FixtureTag'] = fixtures_raw['Home'] + " v " + fixtures_raw['Away']
    fixture_counts = fixtures_raw.groupby(['Home', 'Away']).cumcount() + 1
    played_max = played_results.groupby(['Home', 'Away'])['Instance'].max().rename('PlayedMax')
    fixtures_raw = fixtures_raw.join(played_max, on=['Home', 'Away'])
    fixtures_raw['PlayedMax'] = fixtures_raw['PlayedMax'].fillna(0)
    fixtures_raw['Instance'] = fixture_counts + fixtures_raw['PlayedMax']
    fixtures_raw.drop(columns='PlayedMax', inplace=True)
    fixtures_raw['UniqueID'] = fixtures_raw['FixtureTag'] + " " + fixtures_raw['Instance'].astype(str)

    return played_results, fixtures_raw, ratings

# ---------------------------------------------------
# POISSON FUNCTIONS
# ---------------------------------------------------
def poisson(k, lam):
    return (lam**k) * np.exp(-lam) / math.factorial(k)

def match_prediction(home, away, ratings):
    h = ratings[ratings.Club == home].iloc[0]
    a = ratings[ratings.Club == away].iloc[0]

    lam_h = LEAGUE_AVG_GOALS_HOME * (h.AttackRating / a.DefenceRating)
    lam_a = LEAGUE_AVG_GOALS_AWAY * (a.AttackRating / h.DefenceRating)

    home_win, draw, away_win = 0, 0, 0
    score_probs = {}

    for i in range(7):
        for j in range(7):
            p = poisson(i, lam_h) * poisson(j, lam_a)
            score_probs[f"{i} - {j}"] = p
            if i > j: home_win += p
            elif i == j: draw += p
            else: away_win += p

    top_5 = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:5]

    return home_win, draw, away_win, lam_h, lam_a, top_5

# ---------------------------------------------------
# TABLE & SIMULATION
# ---------------------------------------------------
def calculate_current_table(df):
    df["HomeWin"] = np.where(df.HomeGoals > df.AwayGoals, 3, 0)
    df["Draw"] = np.where(df.HomeGoals == df.AwayGoals, 1, 0)
    df["AwayWin"] = np.where(df.AwayGoals > df.HomeGoals, 3, 0)
    h_stats = df.groupby("Home").agg(
        HW=("HomeWin", lambda x: (x==3).sum()),
        HD=("Draw", "sum"),
        HGF=("HomeGoals", "sum"),
        HGA=("AwayGoals", "sum"),
        HP=("Home", "count")
    )
    a_stats = df.groupby("Away").agg(
        AW=("AwayWin", lambda x: (x==3).sum()),
        AD=("Draw", "sum"),
        AGF=("AwayGoals", "sum"),
        AGA=("HomeGoals", "sum"),
        AP=("Away", "count")
    )
    table = pd.merge(h_stats, a_stats, left_index=True, right_index=True, how="outer").fillna(0)
    table["Team"] = table.index
    table["Points"] = (table["HW"] * 3) + (table["AW"] * 3) + table["HD"] + table["AD"]
    table["GD"] = (table["HGF"] + table["AGF"]) - (table["HGA"] + table["AGA"])
    table["Played"] = table["HP"] + table["AP"]
    return table[["Team", "Played", "Points", "GD"]].sort_values(
        ["Points", "GD"], ascending=False
    ).reset_index(drop=True)

@st.cache_data
def run_simulation(ratings, remaining_fixtures, current_pts_dict):
    team_names = ratings.Club.tolist()
    team_map = {t: i for i, t in enumerate(team_names)}
    start_pts = np.array([current_pts_dict.get(team, 0) for team in team_names])
    final_points = np.zeros((ITERATIONS, len(ratings)))

    for i in range(ITERATIONS):
        pts = start_pts.copy()
        for _, m in remaining_fixtures.iterrows():
            h_idx, a_idx = team_map[m.Home], team_map[m.Away]
            lam_h = LEAGUE_AVG_GOALS_HOME * (ratings.iloc[h_idx].AttackRating / ratings.iloc[a_idx].DefenceRating)
            lam_a = LEAGUE_AVG_GOALS_AWAY * (ratings.iloc[a_idx].AttackRating / ratings.iloc[h_idx].DefenceRating)
            g_h, g_a = np.random.poisson(lam_h), np.random.poisson(lam_a)
            if g_h > g_a: pts[h_idx] += 3
            elif g_h == g_a: pts[h_idx] += 1; pts[a_idx] += 1
            else: pts[a_idx] += 3
        final_points[i] = pts

    probs = pd.DataFrame(index=ratings.Club)
    positions = np.zeros_like(final_points)
    for i in range(ITERATIONS):
        order = np.argsort(-final_points[i])
        for pos, team in enumerate(order):
            positions[i, team] = pos + 1

    probs["Title %"] = (positions == 1).mean(axis=0) * 100
    probs["Europe %"] = (positions <= 3).mean(axis=0) * 100
    probs["Relegation %"] = (positions <= 9).mean(axis=0) * 100
    proj = ratings.copy()
    proj["Current"] = start_pts
    proj["Forecasted Total"] = final_points.mean(axis=0)
    return proj.sort_values("Forecasted Total", ascending=False), probs

# ---------------------------------------------------
# MAIN APP FLOW
# ---------------------------------------------------
played_results, fixtures, ratings = load_data()

if "manual_results" not in st.session_state:
    st.session_state.manual_results = []

manual_ids = set([m['UniqueID'] for m in st.session_state.manual_results])
played_ids = set(played_results['UniqueID'])
finished_ids = played_ids.union(manual_ids)
remaining_to_simulate = fixtures[~fixtures['UniqueID'].isin(finished_ids)]

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("🛠️ What-If Scenarios")
with st.sidebar.expander("Add Manual Result"):
    if not remaining_to_simulate.empty:
        selected_tag = st.selectbox("Select Fixture", remaining_to_simulate['UniqueID'].tolist())
        row = remaining_to_simulate[remaining_to_simulate['UniqueID'] == selected_tag].iloc[0]
        c1, c2 = st.columns(2)
        h_score = c1.number_input("Home Goals", 0, 10, 0)
        a_score = c2.number_input("Away Goals", 0, 10, 0)
        if st.button("Add to Standings"):
            st.session_state.manual_results.append({
                "Home": row.Home,
                "Away": row.Away,
                "HomeGoals": h_score,
                "AwayGoals": a_score,
                "UniqueID": row.UniqueID
            })
            st.rerun()

if st.session_state.manual_results and st.sidebar.button("Clear Overrides"):
    st.session_state.manual_results = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.write(f"✅ Played: {len(played_results)}")
st.sidebar.write(f"🛠️ Manual: {len(st.session_state.manual_results)}")
st.sidebar.write(f"🔮 Remaining: {len(remaining_to_simulate)}")
st.sidebar.write(f"📐 Total: {len(played_results) + len(st.session_state.manual_results) + len(remaining_to_simulate)} / 180")

# ---------------------------------------------------
# STANDINGS
# ---------------------------------------------------
manual_df = pd.DataFrame(st.session_state.manual_results) if st.session_state.manual_results else pd.DataFrame(columns=played_results.columns)
combined_df = pd.concat([played_results, manual_df], ignore_index=True)
current_table = calculate_current_table(combined_df)
st.subheader("🏁 Current Standings")
st.dataframe(current_table, use_container_width=True)

# ---------------------------------------------------
# SIMULATION
# ---------------------------------------------------
if st.button("Run Season Simulation", type="primary"):
    pts_dict = current_table.set_index("Team")["Points"].to_dict()
    proj, probs = run_simulation(ratings, remaining_to_simulate, pts_dict)
    st.subheader("🏆 Probabilities & Projection")
    st.dataframe(probs.round(1), use_container_width=True)
    st.dataframe(proj[["Club", "Current", "Forecasted Total"]].round(1), use_container_width=True)

# ---------------------------------------------------
# MATCH PREDICTOR
# ---------------------------------------------------
st.markdown("---")
st.subheader("⚽ Match Predictor")
teams = sorted(ratings.Club.tolist())
col1, col2 = st.columns(2)
h_p = col1.selectbox("Home Team", teams)
a_p = col2.selectbox("Away Team", teams, index=1)

if st.button("Predict"):
    hw, d, aw, xh, xa, top_scores = match_prediction(h_p, a_p, ratings)

    # Show probabilities as text instead of bar chart
    st.markdown("**Result Probabilities**")
    st.write(f"{h_p} win: **{hw*100:.1f}%**")
    st.write(f"Draw: **{d*100:.1f}%**")
    st.write(f"{a_p} win: **{aw*100:.1f}%**")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Expected Goals (xG)**")
        st.write(f"{h_p}: **{xh:.2f}**")
        st.write(f"{a_p}: **{xa:.2f}**")
    with c2:
        st.markdown("**Top 5 Most Likely Scores**")
        for score, prob in top_scores:
            st.write(f"👉 **{score}**: {prob:.1%}")
