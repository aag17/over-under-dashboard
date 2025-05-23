#!/usr/bin/env python3
"""
analysis_dashboard.py

Streamlit app to show % Over vs Under by book, with filters on date, day-of-week,
prop, team, home/away, percent-change OM/UM, expected value (EV) insights, and
book accuracy metrics to evaluate bookmaker calibration.
Also allows toggling between original Over/Under counts and EV-filtered counts.
"""


# 1) Pull in Inter from Google Fonts
import streamlit as ststrea

import os, glob
from datetime import date

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("⚡️ App starting up")

st.set_page_config(
    page_title="Over vs Under Dashboard",
    layout="wide"
)

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
      /* make the entire app use Inter */
      html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif !important;
      }
      /* restyle Streamlit buttons to have softer corners */
      .stButton>button {
        border-radius: 8px;
      }
      /* tweak AgGrid header to use our primaryColor bg + white text */
      .ag-theme-streamlit .ag-header-cell-label {
        background-color: #F37B20 !important;
        color: #FFFFFF !important;
      }
      /* make the grid cells a bit darker text */
      .ag-theme-streamlit .ag-cell {
        color: #333333 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

import psutil, os, streamlit as st

@st.cache_resource # or st.cache_resource in 1.18+
def get_process():
    return psutil.Process(os.getpid())

def log_memory(tag: str):
    proc = get_process()
    mb = proc.memory_info().rss / 1024**2
    st.sidebar.text(f"{tag}: {mb:.0f} MB")

def log_memory(tag=""):
    proc = get_process()
    rss = proc.memory_info().rss / 1024**2
    st.sidebar.write(f"{tag} Memory: {rss:.1f} MB")

# Then early in your app:
log_memory("Start")
# after loading df:

# inside loops or callbacks:
# ─── Page config ───────────────────────────────────────────────────────────

st.title("Over vs Under % by Book & EV Insights")
log_memory("Start")
# ─── Load data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    here = ""
    files = glob.glob(os.path.join(here, "feature_*.csv"))
    if not files:
        raise FileNotFoundError(f"No files matching {here}/feature_*.csv")
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        book = os.path.basename(fp).split("_")[1]
        df["book"] = book
        dfs.append(df)
    log_memory("After load")
    return pd.concat(dfs, ignore_index=True)
    
try:
    df = load_data()
    log_memory("After load")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# ─── Precompute columns ────────────────────────────────────────────────────
df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
if not set(["home","away"]).issubset(df.columns):
    st.error("Data must include 'home' and 'away' columns.")
    st.stop()
# drop essential missing
df = df.dropna(subset=["game_date","outcome","player_type","pct_change_om","pct_change_um","team","prop","home","away"])
df["day_of_week"] = df["game_date"].dt.day_name()
df = df[df["outcome"].isin(["over","under","push"])]

# ─── Sidebar filters ───────────────────────────────────────────────────────
st.sidebar.header("Filters")
# Book
books = sorted(df["book"].unique())
books_opts = ["All"] + books
sel_books = st.sidebar.multiselect("Book", books_opts, default=["All"])
if "All" in sel_books:
    sel_books = books
# Player Type
ptypes = sorted(df["player_type"].unique())
ptypes_opts = ["All"] + ptypes
sel_ptypes = st.sidebar.multiselect("Player Type", ptypes_opts, default=["All"])
if "All" in sel_ptypes:
    sel_ptypes = ptypes
# Date Range
min_d, max_d = df["game_date"].dt.date.min(), df["game_date"].dt.date.max()
start_date, end_date = st.sidebar.date_input(
    "Date Range",
    value=[min_d, max_d],
    min_value=min_d,
    max_value=max_d
)
# Location
action = st.sidebar.radio("Location", ["All","Home","Away"], index=0)
# Team
teams = sorted(df["team"].unique())
teams_opts = ["All"] + teams
sel_teams = st.sidebar.multiselect("Team", teams_opts, default=["All"])
if "All" in sel_teams:
    sel_teams = teams

# Day of Week
dows = sorted(df["day_of_week"].unique())
dows_opts = ["All"] + dows
sel_dows = st.sidebar.multiselect("Day of Week", dows_opts, default=["All"])
if "All" in sel_dows:
    sel_dows = dows
# Prop
props = sorted(df["prop"].unique())
props_opts = ["All"] + props
sel_props = st.sidebar.multiselect("Prop", props_opts, default=["All"])
if "All" in sel_props:
    sel_props = props
# Percent Change OM/UM
pom_min, pom_max = float(df["pct_change_om"].min()), float(df["pct_change_om"].max())
sel_pom = st.sidebar.slider("Pct Change OM (%)", pom_min, pom_max, (pom_min,pom_max))
pum_min, pum_max = float(df["pct_change_um"].min()), float(df["pct_change_um"].max())
sel_pum = st.sidebar.slider("Pct Change UM (%)", pum_min, pum_max, (pum_min,pum_max))

# EV controls
st.sidebar.header("EV Insights and Chart Options")
# Define or extract multiplier column
if 'multiplier' not in df.columns:
    if 'line_value_book' in df.columns:
        df['multiplier'] = df['line_value_book']
    else:

        
        df['multiplier'] = df['line_value']
# PrizePicks uses profit-only odds (1× profit); convert to decimal odds (stake + profit)
mask_pp = df['book'] == 'prizepicks'
df.loc[mask_pp, 'multiplier'] = df.loc[mask_pp, 'multiplier'] + 1.0
# Probability basis
prob_basis = st.sidebar.selectbox("Probability Basis", ['empirical_prob','implied_prob'], index=0)
# EV threshold
ev_thresh = st.sidebar.slider("Minimum EV", -1.0, 2.0, 0.0, 0.01)
# Chart data mode
data_mode = st.sidebar.radio("Chart Mode", ['All Props'], index=0)

# ─── Apply filters ─────────────────────────────────────────────────────────
mask = (
    df['book'].isin(sel_books) &
    df['player_type'].isin(sel_ptypes) &
    df['game_date'].dt.date.between(start_date,end_date) &
    df['team'].isin(sel_teams) &
    df['day_of_week'].isin(sel_dows) &
    df['prop'].isin(sel_props) &
    df['pct_change_om'].between(*sel_pom) &
    df['pct_change_um'].between(*sel_pum)
)
if action=='Home': mask &= df['team']==df['home']
elif action=='Away': mask &= df['team']==df['away']

df_f = df[mask]
if df_f.empty:
    st.warning("No data matches filters.")
    st.stop()

# ─── Compute EV & probabilities ─────────────────────────────────────────────
df_f['hit'] = (df_f['outcome']=='over').astype(int)
df_f['empirical_prob'] = df_f.groupby('prop')['hit'].transform('mean')
df_f['implied_prob'] = 1/df_f['multiplier']
df_f['p_win'] = df_f[prob_basis]
df_f['EV'] = df_f['multiplier'] * df_f['p_win'] - 1
# Profitable subset
prof = df_f[df_f['EV']>=ev_thresh]

# Select dataset for chart
if data_mode == 'Profitable Props':
    df_chart = prof
else:
    df_chart = df_f

# ─── Cumulative Over % Line Chart ──────────────────────────────────────────
# 1) Filter to only Over/Under outcomes and extract date
df_roll = df_chart[df_chart['outcome'].isin(['over','under'])].copy()
df_roll['date'] = df_roll['game_date'].dt.date

# 2) Sort and flag “over”
df_roll = df_roll.sort_values(['book','date'])
df_roll['is_over'] = (df_roll['outcome'] == 'over').astype(int)

# 3) Aggregate daily counts per book
daily = (
    df_roll
    .groupby(['date','book'])
    .agg(daily_over=('is_over','sum'),
         daily_total=('outcome','count'))
    .reset_index()
)

# 4) Compute cumulative sums per book
daily['cum_over']  = daily.groupby('book')['daily_over'].cumsum()
daily['cum_total'] = daily.groupby('book')['daily_total'].cumsum()

# 5) Calculate cumulative percentage
daily['cum_over_pct'] = daily['cum_over'] / daily['cum_total'] * 100

# 6) Plot with Plotly Express
import plotly.express as px

fig = px.line(
    daily,
    x='date',
    y='cum_over_pct',
    color='book',
    markers=True,
    labels={
        'date': 'Date',
        'cum_over_pct': 'Cumulative % Over',
        'book': 'Book'
    },
    title='Cumulative Over % by Book'
)

# figure out the actual min/max so we can “zoom in” a bit
y_min = daily['cum_over_pct'].min() - 2
y_max = daily['cum_over_pct'].max() + 2
# clamp to [0,100]
y_min = max(y_min, 0)
y_max = min(y_max, 100)

fig.update_layout(
    yaxis=dict(
        ticksuffix='%',
        range=[y_min, y_max]
    )
)

fig.update_layout(
  yaxis=dict(ticksuffix='%'),
  hovermode='x unified',
  autosize=True
)
fig.update_xaxes(autorange=True)
fig.update_yaxes(autorange=True)


st.plotly_chart(fig, use_container_width=True)

# ─── Top Combinations for This Weekday, by Location, with Book EV ─────────
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

# 1) Build base df without the sidebar Location filter
mask_base = (
    df['book'].isin(sel_books) &
    df['player_type'].isin(sel_ptypes) &
    df['game_date'].dt.date.between(start_date, end_date) &
    df['team'].isin(sel_teams) &
    df['day_of_week'].isin(sel_dows) &
    df['prop'].isin(sel_props) &
    df['pct_change_om'].between(*sel_pom) &
    df['pct_change_um'].between(*sel_pum)
)
df_base = df[mask_base]

# 2) Filter to today’s weekday & only Over/Under
today_dow = date.today().strftime('%A')
st.subheader(f'Top Combinations for {today_dow}s (All / Home / Away, N ≥ 50)')

combo_base = df_base[
    (df_base['day_of_week'] == today_dow) &
    (df_base['outcome'].isin(['over','under']))
]

# 3) Slice into All/Home/Away, compute counts, avg odds & Book EV
slices = []
for loc in ['All','Home','Away']:
    log_memory("Inside loop")
    if loc == 'All':
        df_loc = combo_base
    elif loc == 'Home':
        df_loc = combo_base[combo_base['team'] == combo_base['home']]
    else:  # Away
        df_loc = combo_base[combo_base['team'] == combo_base['away']]

    if df_loc.empty:
        continue

    # group and compute count + average multiplier
    tbl = (
        df_loc
        .groupby(['book','prop','outcome'])
        .agg(
            count          = ('outcome','size'),
            avg_multiplier = ('multiplier','mean')
        )
        .reset_index()
    )

    # total sample size per book+prop
    totals = (
        df_loc
        .groupby(['book','prop'])['outcome']
        .count()
        .reset_index(name='total')
    )

    # merge in totals, compute pct and Book EV
    tbl = tbl.merge(totals, on=['book','prop'])
    tbl['pct']     = tbl['count'] / tbl['total'] * 100
    #tbl['Book EV'] = tbl['avg_multiplier'] * (tbl['count']/tbl['total']) - 1

    tbl['Location'] = loc
    slices.append(tbl)

# 4) Combine, filter N ≥ 50 and sort by pct (or you could sort by Book EV)
top_combo = (
    pd.concat(slices, ignore_index=True)
      .query('total >= 50')
      .sort_values('pct', ascending=False)
      .reset_index(drop=True)
)
top_combo = top_combo.drop(columns=['avg_multiplier'])

# 5) Render via AG-Grid
grid_df = top_combo.rename(columns={
    'book':          'Book',
    'prop':          'Prop',
    'outcome':       'Outcome',
    'count':         'Count',
    'total':         'Total',
    'pct':           'Percent',
    'avg_multiplier':'Avg Odds',
    'Book EV':       'Book EV',
    'Location':      'Location'
})

gb = GridOptionsBuilder.from_dataframe(grid_df)
gb.configure_selection('single')
grid_opts = gb.build()

grid_resp = AgGrid(
    grid_df,
    gridOptions=grid_opts,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    fit_columns_on_grid_load=True
)



# 5) On selection, filter & draw the cumulative chart

import pandas as pd  # if you haven’t already

# … after AgGrid …
log_memory("after AgGrid ")

# 1) grab whatever Streamlit returned (might be None, list, or DataFrame)
selected = grid_resp.get('selected_rows')

# 2) normalize to a plain list
if selected is None:
    selected = []
elif isinstance(selected, pd.DataFrame):
    selected = selected.to_dict('records')

# 3) now a simple truth‐test works
if not selected:
    st.info("👉 Click a row above to view its cumulative chart.")
else:
    sel = selected[0]
    # … your existing chart logic …
    sel_book    = sel['Book']
    sel_prop    = sel['Prop']
    sel_outcome = sel['Outcome']
    sel_loc     = sel['Location']

    # filter df_base to the selected combo
    # … inside your “else:” after grabbing sel_book/sel_prop/sel_outcome/sel_loc …

# 1) Filter for Book+Prop (keep both Over & Under), then apply location
    df_sel = df_base[
    (df_base['book'] == sel_book) &
    (df_base['prop'] == sel_prop) &
    (df_base['outcome'].isin(['over','under'])) &
    (df_base['day_of_week'] == today_dow)
    ].copy()
    if sel_loc == 'Home':
        df_sel = df_sel[df_sel['team'] == df_sel['home']]
    elif sel_loc == 'Away':
        df_sel = df_sel[df_sel['team'] == df_sel['away']]

    # 2) Mark hits vs misses
    df_sel = df_sel.sort_values('game_date')
    df_sel['date']   = df_sel['game_date'].dt.date
    df_sel['is_hit'] = (df_sel['outcome'] == sel_outcome).astype(int)

    # 3) Daily aggregation: sum of hits, count of all O/U
    daily = (
        df_sel
        .groupby('date')
        .agg(
            daily_hit   = ('is_hit',    'sum'),
            daily_total = ('outcome',   'count')
        )
        .reset_index()
    )

    # 4) Cumulative sums & pct
    daily['cum_hit']     = daily['daily_hit'].cumsum()
    daily['cum_total']   = daily['daily_total'].cumsum()
    daily['cum_hit_pct'] = daily['cum_hit'] / daily['cum_total'] * 100

    # 4b) Compute the actual daily percentage
    daily['daily_pct'] = daily['daily_hit'] / daily['daily_total'] * 100

    # 5) Plot both Daily % and Cumulative % on the same chart
    import plotly.graph_objects as go

    
    fig = go.Figure()

    # Daily percentage trace (showing daily_total as sample size)
    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['daily_pct'],
        mode='lines+markers',
        name='Daily %',
        customdata=daily['daily_total'],               # pass daily sample size
        hovertemplate=(
            'Date: %{x}<br>'
            'Daily %: %{y:.1f}%<br>'
            'Sample size: %{customdata}<extra></extra>'
        )
    ))

    # Cumulative percentage trace (showing cum_total as sample size)
    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['cum_hit_pct'],
        mode='lines+markers',
        name='Cumulative %',
        customdata=daily['cum_total'],                 # pass cumulative sample size
        hovertemplate=(
            'Date: %{x}<br>'
            'Cumulative %: %{y:.1f}%<br>'
            'Cumulative sample: %{customdata}<extra></extra>'
        )
    ))

    fig.update_layout(
    hovermode='x unified',
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    autosize=True
    )

    fig.update_layout(
        title=f"{sel_book} – {sel_prop} ({sel_loc})",
        xaxis_title='Date',
        yaxis=dict(
            title='Percentage',
            ticksuffix='%',
        ),
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    fig.update_xaxes(autorange=True)
    fig.update_yaxes(autorange=True)

    st.plotly_chart(fig, use_container_width=True)


with st.expander("❓ What are Top Combinations?"):
    st.write("This table shows the most frequent prop+outcome combinations for each book among all filtered props, restricted to those with at least 50 occurrences.")
    st.write("Columns: Book, Prop, Outcome, Count (number of Over or Under hits), Total (sample size), Percent (hit rate).")
# ─── Book Accuracy Metrics (static table + interactive over‐time chart) ───
st.subheader('Book Accuracy Metrics')

# 1) Restrict to OU only and compute normalized implied probs & hit flags
acc_df = df_f[df_f['outcome'].isin(['over','under'])].copy()
acc_df['imp_raw'] = 1.0 / acc_df['multiplier']
acc_df['imp_sum'] = acc_df.groupby(['book','prop'])['imp_raw'].transform('sum')
acc_df['implied_prob_norm'] = acc_df['imp_raw'] / acc_df['imp_sum']
acc_df['hit'] = (acc_df['outcome']=='over').astype(int)
# errors
acc_df['se'] = (acc_df['implied_prob_norm'] - acc_df['hit'])**2
acc_df['ae'] = (acc_df['implied_prob_norm'] - acc_df['hit']).abs()

# 2) Aggregate overall per book
book_accuracy = (
    acc_df
    .groupby('book')
    .agg(
        brier_score    = ('se',     'mean'),
        mae            = ('ae',     'mean'),
        implied_avg    = ('implied_prob_norm','mean'),
        empirical_avg  = ('hit',    'mean'),
        sample_size    = ('hit',    'count')
    )
    .reset_index()
)
# display static table
st.dataframe(book_accuracy.rename(columns={
    'brier_score':'Brier Score (MSE)',
    'mae':'Mean Abs Error',
    'implied_avg':'Avg Implied Prob',
    'empirical_avg':'Empirical Hit Rate',
    'sample_size':'N'
}))

with st.expander("❓ What do the accuracy metrics mean?"):
    st.write("- **Brier Score (MSE):** Mean squared error between implied probability and actual outcomes.")
    st.write("- **Mean Abs Error (MAE):** Average absolute error between implied probability and actual outcomes.")
    st.write("- **Avg Implied Prob:** Average market-implied probability (normalized, juice-free).")
    st.write("- **Empirical Hit Rate:** Fraction of props that went Over.")
    st.write("- **N:** Number of Over/Under props.")

# ─── Prepare daily accuracy DataFrame ──────────────────────────────────────
acc_df['date'] = acc_df['game_date'].dt.date
daily_accuracy = (
    acc_df
    .groupby(['date','book'])
    .agg(
        **{
          'Brier Score (MSE)'   : ('se',     'mean'),
          'Mean Abs Error'      : ('ae',     'mean'),
          'Avg Implied Prob'    : ('implied_prob_norm','mean'),
          'Empirical Hit Rate'  : ('hit',    'mean')
        }
    )
    .reset_index()
)

# ─── Interactive metric selector + time‐series plot ────────────────────────
metric = st.selectbox(
    "Show accuracy metric over time:",
    ["Brier Score (MSE)", "Mean Abs Error", "Avg Implied Prob", "Empirical Hit Rate"]
)

fig = px.line(
    daily_accuracy,
    x='date',
    y=metric,
    color='book',
    markers=True,
    labels={'date':'Date', metric:metric, 'book':'Book'},
    title=f"{metric} Over Time by Book"
)
fig.update_layout(
  title=f"{metric} Over Time by Book",
  autosize=True
)
fig.update_xaxes(autorange=True)
fig.update_yaxes(autorange=True)

if metric in ["Avg Implied Prob","Empirical Hit Rate"]:
    # probability metrics
    fig.update_layout(yaxis=dict(ticksuffix='%', range=[0,1]))
else:
    # non‐probability metrics (Brier, MAE) — auto‐zoom to daily range
    y_max = daily_accuracy[metric].max() * 1.1
    

st.plotly_chart(fig, use_container_width=True)
