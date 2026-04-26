import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Padel ML Dashboard",
    page_icon="🎾",
    layout="wide"
)

# ══════════════════════════════════════════════════════════
# GREEN THEME CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stButton > button {
        background: linear-gradient(135deg, #2e7d32, #1b5e20) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: bold !important;
        font-size: 16px !important;
        width: 100% !important;
        padding: 12px !important;
        box-shadow: 0 4px 15px rgba(46,125,50,0.3) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #388e3c, #2e7d32) !important;
        box-shadow: 0 6px 20px rgba(46,125,50,0.5) !important;
        transform: translateY(-2px) !important;
    }
    [data-testid="stSidebar"] {
        border-right: 3px solid #2e7d32;
    }
    [data-testid="metric-container"] {
        border-left: 4px solid #2e7d32 !important;
        border-radius: 10px !important;
    }
    [data-testid="stSlider"] > div > div > div > div {
        background: #2e7d32 !important;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: #2e7d32 !important;
        box-shadow: 0 0 0 2px rgba(46,125,50,0.3) !important;
    }
    hr {
        border-color: #2e7d32 !important;
        opacity: 0.3;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    return pd.read_csv("players_clean.csv")

@st.cache_data
def load_raw():
    return pd.read_csv("players_data.csv")

@st.cache_data
def load_timeseries():
    df = pd.read_csv("player_timeseries.csv")
    df['full_date'] = pd.to_datetime(df['full_date'])
    return df

df       = load_data()
df_raw   = load_raw()
df_ts    = load_timeseries()

# ══════════════════════════════════════════════════════════
# TRAIN MODELS
# ══════════════════════════════════════════════════════════
@st.cache_resource
def train_models(df):
    drop_clf = ['is_top_player', 'ranking_position',
                'ranking_points', 'performance_score']
    drop_reg = ['contract_value_eur', 'sponsorship_value_annual_eur',
                'is_top_player', 'performance_score']

    X_clf = df.drop(columns=drop_clf)
    y_clf = df['is_top_player']
    X_reg = df.drop(columns=drop_reg)
    y_reg = df['contract_value_eur']

    sc_clf = StandardScaler()
    sc_reg = StandardScaler()
    X_clf_sc = sc_clf.fit_transform(X_clf)
    X_reg_sc = sc_reg.fit_transform(X_reg)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_clf_sc, y_clf)

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_reg_sc, y_reg)

    return clf, reg, sc_clf, sc_reg, X_clf, X_reg

clf, reg, sc_clf, sc_reg, X_clf, X_reg = train_models(df)

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🎾 Padel ML")
    st.markdown("**Machine Learning Dashboard**")
    st.markdown("---")

    page = st.radio("📌 Navigation", [
        "🏠 Overview",
        "🔍 Player Search",
        "📊 Player Stats",
        "🤖 Top Player Predictor",
        "💰 Contract Predictor",
        "🔵 Player Clusters",
        "📈 Time Series"
    ])

    st.markdown("---")
    st.success(f"📋 **{df.shape[0]} players** · {df.shape[1]} features")

# ══════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🎾 Padel ML Dashboard")
    st.markdown("##### Complete Machine Learning Analysis of Padel Players")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Players",     df.shape[0])
    col2.metric("⭐ Top 20 Players",    int(df['is_top_player'].sum()))
    col3.metric("💰 Avg Contract (€)",  f"{df['contract_value_eur'].mean():,.0f}")
    col4.metric("📱 Avg Instagram (M)", f"{df['instagram_followers_millions'].mean():.2f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏆 Top 10 Players by Contract Value")
        top10 = df.nlargest(10, 'contract_value_eur')[[
            'contract_value_eur', 'total_titles',
            'instagram_followers_millions', 'is_top_player'
        ]].reset_index(drop=True)
        top10.index += 1
        st.dataframe(top10.style.format({
            'contract_value_eur'          : '€{:,.0f}',
            'instagram_followers_millions': '{:.2f}M'
        }), use_container_width=True)

    with col2:
        st.markdown("#### 📊 ML Project Summary")
        summary = pd.DataFrame({
            'Section'   : ['C — Classification', 'D — Regression',
                           'E — Clustering', 'F — Time Series'],
            'Task'      : ['Predict Top 20', 'Predict Contract €',
                           'Player Profiles', 'Ranking Evolution'],
            'Best Model': ['Random Forest', 'Random Forest',
                           'Hierarchical', 'ARIMA / GB'],
            'Score'     : ['F1 = 1.0', 'R² = 1.0',
                           'Silhouette = 0.596', 'MAE based']
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🔥 Correlation Heatmap")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    fig = px.imshow(df[num_cols].corr(),
                    color_continuous_scale='Greens',
                    title="Feature Correlation Matrix",
                    height=500)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 2 — PLAYER SEARCH
# ══════════════════════════════════════════════════════════
elif page == "🔍 Player Search":
    st.title("🔍 Player Search")
    st.markdown("Search and explore any player's full profile")
    st.markdown("---")

    search = st.text_input("Search player by name",
                           placeholder="e.g. Facundo, Juan, Denis...")

    if search:
        results = df_raw[
            df_raw['player_name'].str.contains(search, case=False, na=False)
        ]
        if len(results) == 0:
            st.warning(f"No player found with name '{search}'")
        else:
            st.success(f"Found {len(results)} player(s)")
            for _, player in results.iterrows():
                with st.expander(f"🎾 {player['player_name']}", expanded=True):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("**👤 Profile**")
                        st.write(f"🌍 Gender: `{player.get('gender','N/A')}`")
                        st.write(f"✋ Hand: `{player.get('dominant_hand','N/A')}`")
                        st.write(f"📏 Height: `{player.get('height_cm','N/A')} cm`")
                        st.write(f"✅ Active: `{player.get('is_active','N/A')}`")

                    with col2:
                        st.markdown("**🏅 Performance**")
                        st.write(f"🎯 Ranking: `#{player.get('ranking','N/A')}`")
                        st.write(f"🏆 Titles: `{player.get('total_titles','N/A')}`")
                        st.write(f"🎾 Win Rate: `{player.get('win_rate_finals','N/A')}%`")
                        st.write(f"📈 Points: `{player.get('ranking_points','N/A')}`")

                    with col3:
                        st.markdown("**📱 Social Media**")
                        st.write(f"📸 Instagram: `{player.get('instagram_followers_millions','N/A')}M`")
                        st.write(f"🎵 TikTok: `{player.get('tiktok_followers_millions','N/A')}M`")
                        st.write(f"🐦 Twitter: `{player.get('twitter_followers_thousands','N/A')}K`")
                        st.write(f"▶️ YouTube: `{player.get('youtube_subscribers_thousands','N/A')}K`")

                    st.markdown("---")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("💰 Contract",
                              f"€{float(player.get('contract_value_eur',0)):,.0f}")
                    c2.metric("💼 Sponsorship",
                              f"€{float(player.get('sponsorship_value_annual_eur',0)):,.0f}")
                    c3.metric("🤝 Brand", str(player.get('brand_name','N/A')))

            # ── Player Ranking Timeline ──
            ts_player = df_ts[
                df_ts['player_name'].str.contains(search, case=False, na=False)
            ]
            if len(ts_player) > 1:
                st.markdown("---")
                st.markdown("#### 📈 Ranking Evolution Over Time")
                fig = px.line(ts_player.sort_values('full_date'),
                              x='full_date', y='ranking_position',
                              color='player_name',
                              title="Ranking Position Over Time",
                              color_discrete_sequence=['#2e7d32'])
                fig.update_yaxes(autorange='reversed')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👆 Type a player name above to search")
        cols_show = ['player_name', 'gender', 'dominant_hand',
                     'height_cm', 'ranking', 'total_titles',
                     'contract_value_eur', 'brand_name']
        available = [c for c in cols_show if c in df_raw.columns]
        st.dataframe(df_raw[available], use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
# PAGE 3 — PLAYER STATS
# ══════════════════════════════════════════════════════════
elif page == "📊 Player Stats":
    st.title("📊 Player Statistics")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x='ranking_position', nbins=15,
                           color_discrete_sequence=['#2e7d32'],
                           title="Player Ranking Distribution")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x='contract_value_eur', nbins=15,
                           color_discrete_sequence=['#388e3c'],
                           title="Contract Value Distribution (€)")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df,
                         x='total_social_followers',
                         y='contract_value_eur',
                         color='is_top_player',
                         size='total_titles',
                         title="Social Followers vs Contract Value",
                         color_discrete_map={0: '#81c784', 1: '#1b5e20'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(df,
                         x='total_titles',
                         y='win_rate_finals',
                         color='is_top_player',
                         size='contract_value_eur',
                         title="Titles vs Win Rate in Finals",
                         color_discrete_map={0: '#81c784', 1: '#1b5e20'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 Full Dataset")
    st.dataframe(df, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 4 — TOP PLAYER PREDICTOR
# ══════════════════════════════════════════════════════════
elif page == "🤖 Top Player Predictor":
    st.title("🤖 Top Player Predictor")
    st.markdown("Predict if a player will be in the **Top 20**")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🏅 Performance")
        total_titles    = st.slider("Total Titles", 0, 200, 20)
        win_rate_finals = st.slider("Win Rate Finals (%)", 0.0, 100.0, 50.0)
        total_finals    = st.slider("Total Finals", 0, 100, 10)
        ranking_change  = st.slider("Ranking Change", -50, 50, 0)

    with col2:
        st.markdown("#### 📱 Social Media")
        instagram = st.slider("Instagram (M)", 0.0, 5.0, 0.5)
        tiktok    = st.slider("TikTok (M)", 0.0, 5.0, 0.3)
        twitter   = st.slider("Twitter (K)", 0.0, 500.0, 50.0)
        youtube   = st.slider("YouTube (K)", 0.0, 500.0, 50.0)

    with col3:
        st.markdown("#### 💼 Profile")
        yearly_rackets  = st.slider("Yearly Rackets", 0, 50, 12)
        engagement_rate = st.slider("Engagement Rate (%)", 0.0, 20.0, 3.0)
        height_cm       = st.slider("Height (cm)", 160, 210, 180)
        gender          = st.selectbox("Gender", [0, 1],
                                        format_func=lambda x: "Male" if x==0 else "Female")

    st.markdown("---")
    if st.button("🔮 Predict Now", use_container_width=True):
        total_social = instagram + tiktok + twitter/1000 + youtube/1000
        input_data   = pd.DataFrame([{col: 0 for col in X_clf.columns}])
        input_data['total_titles']                  = total_titles
        input_data['win_rate_finals']               = win_rate_finals
        input_data['total_finals']                  = total_finals
        input_data['ranking_change']                = ranking_change
        input_data['instagram_followers_millions']  = instagram
        input_data['tiktok_followers_millions']     = tiktok
        input_data['twitter_followers_thousands']   = twitter
        input_data['youtube_subscribers_thousands'] = youtube
        input_data['yearly_rackets']                = yearly_rackets
        input_data['engagement_rate_percent']       = engagement_rate
        input_data['height_cm']                     = height_cm
        input_data['gender']                        = gender
        input_data['total_social_followers']        = total_social

        input_sc    = sc_clf.transform(input_data)
        prediction  = clf.predict(input_sc)[0]
        probability = clf.predict_proba(input_sc)[0][1]

        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.success(f"⭐ TOP 20 PLAYER — Confidence: {probability:.1%}")
            else:
                st.error(f"📉 NOT Top 20 — Confidence: {1-probability:.1%}")

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Top Player Probability (%)"},
                gauge={
                    'axis'     : {'range': [0, 100]},
                    'bar'      : {'color': "#2e7d32"},
                    'bgcolor'  : "rgba(0,0,0,0)",
                    'steps'    : [
                        {'range': [0,  40], 'color': 'rgba(198,40,40,0.15)'},
                        {'range': [40, 70], 'color': 'rgba(245,127,23,0.15)'},
                        {'range': [70,100], 'color': 'rgba(27,94,32,0.15)'}
                    ],
                    'threshold': {'line': {'color': '#1b5e20', 'width': 4},
                                  'thickness': 0.75, 'value': 50}
                }
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=300)
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 5 — CONTRACT PREDICTOR
# ══════════════════════════════════════════════════════════
elif page == "💰 Contract Predictor":
    st.title("💰 Contract Value Predictor")
    st.markdown("Predict a player's **sponsorship contract value** in €")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🏅 Performance")
        r_titles  = st.slider("Total Titles", 0, 200, 20, key='r1')
        r_winrate = st.slider("Win Rate Finals (%)", 0.0, 100.0, 50.0, key='r2')
        r_finals  = st.slider("Total Finals", 0, 100, 10, key='r3')
        r_change  = st.slider("Ranking Change", -50, 50, 0, key='r4')

    with col2:
        st.markdown("#### 📱 Social Media")
        r_instagram = st.slider("Instagram (M)", 0.0, 5.0, 0.5, key='r5')
        r_tiktok    = st.slider("TikTok (M)", 0.0, 5.0, 0.3, key='r6')
        r_twitter   = st.slider("Twitter (K)", 0.0, 500.0, 50.0, key='r7')
        r_youtube   = st.slider("YouTube (K)", 0.0, 500.0, 50.0, key='r8')

    with col3:
        st.markdown("#### 💼 Profile")
        r_rackets = st.slider("Yearly Rackets", 0, 50, 12, key='r9')
        r_engage  = st.slider("Engagement Rate (%)", 0.0, 20.0, 3.0, key='r10')
        r_height  = st.slider("Height (cm)", 160, 210, 180, key='r11')
        r_gender  = st.selectbox("Gender", [0, 1],
                                  format_func=lambda x: "Male" if x==0 else "Female",
                                  key='r12')

    st.markdown("---")
    if st.button("💰 Predict Contract Value", use_container_width=True):
        r_social = r_instagram + r_tiktok + r_twitter/1000 + r_youtube/1000
        input_r  = pd.DataFrame([{col: 0 for col in X_reg.columns}])
        input_r['total_titles']                  = r_titles
        input_r['win_rate_finals']               = r_winrate
        input_r['total_finals']                  = r_finals
        input_r['ranking_change']                = r_change
        input_r['instagram_followers_millions']  = r_instagram
        input_r['tiktok_followers_millions']     = r_tiktok
        input_r['twitter_followers_thousands']   = r_twitter
        input_r['youtube_subscribers_thousands'] = r_youtube
        input_r['yearly_rackets']                = r_rackets
        input_r['engagement_rate_percent']       = r_engage
        input_r['height_cm']                     = r_height
        input_r['gender']                        = r_gender
        input_r['total_social_followers']        = r_social

        input_r_sc      = sc_reg.transform(input_r)
        predicted_value = reg.predict(input_r_sc)[0]

        if predicted_value >= 200000:
            tier = "⭐ Elite — Premium sponsorship tier"
        elif predicted_value >= 100000:
            tier = "🥈 Mid-tier — Standard sponsorship"
        else:
            tier = "🥉 Developing — Entry sponsorship"

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"💰 Predicted: **€{predicted_value:,.0f}**")
            st.info(f"Tier: {tier}")
            st.metric("vs Average",
                      f"€{predicted_value:,.0f}",
                      f"€{predicted_value - df['contract_value_eur'].mean():,.0f}")

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_value,
                delta={'reference': df['contract_value_eur'].mean(),
                       'valueformat': ',.0f'},
                title={'text': "Contract Value (€)"},
                number={'valueformat': ',.0f'},
                gauge={
                    'axis'   : {'range': [0, df['contract_value_eur'].max()]},
                    'bar'    : {'color': "#2e7d32"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'steps'  : [
                        {'range': [0, 100000],
                         'color': 'rgba(74,20,140,0.15)'},
                        {'range': [100000, 200000],
                         'color': 'rgba(13,71,161,0.15)'},
                        {'range': [200000, df['contract_value_eur'].max()],
                         'color': 'rgba(27,94,32,0.15)'}
                    ]
                }
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=300)
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 6 — CLUSTERS
# ══════════════════════════════════════════════════════════
elif page == "🔵 Player Clusters":
    st.title("🔵 Player Clusters")
    st.markdown("Unsupervised grouping of players into **natural profiles**")
    st.markdown("---")

    cluster_features = [
        'ranking_position', 'total_titles', 'win_rate_finals',
        'contract_value_eur', 'instagram_followers_millions',
        'tiktok_followers_millions', 'engagement_rate_percent',
        'total_social_followers', 'sponsorship_value_annual_eur'
    ]

    X_cl    = df[cluster_features].copy()
    sc_cl   = StandardScaler()
    X_cl_sc = sc_cl.fit_transform(X_cl)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = km.fit_predict(X_cl_sc)

    pca       = PCA(n_components=2, random_state=42)
    X_pca     = pca.fit_transform(X_cl_sc)
    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]

    cluster_labels = {0: "🥈 Mid-tier", 1: "🥉 Developing", 2: "🥇 Elite"}
    df['cluster_label'] = df['cluster'].map(cluster_labels)

    col1, col2, col3 = st.columns(3)
    for col, label in zip([col1, col2, col3],
                           ['🥇 Elite', '🥈 Mid-tier', '🥉 Developing']):
        mask  = df['cluster_label'] == label
        count = mask.sum()
        avg   = df[mask]['contract_value_eur'].mean() if count > 0 else 0
        col.metric(label, f"{count} players", f"Avg €{avg:,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(df, x='PC1', y='PC2',
                         color='cluster_label',
                         size='contract_value_eur',
                         title="Player Clusters (PCA 2D)",
                         color_discrete_map={
                             '🥇 Elite'     : '#1b5e20',
                             '🥈 Mid-tier'  : '#66bb6a',
                             '🥉 Developing': '#c8e6c9'
                         })
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        profile = df.groupby('cluster_label')[[
            'contract_value_eur', 'total_titles',
            'total_social_followers', 'win_rate_finals'
        ]].mean().reset_index()

        fig = px.bar(profile.melt(id_vars='cluster_label'),
                     x='variable', y='value',
                     color='cluster_label',
                     barmode='group',
                     title="Average Profile per Cluster",
                     color_discrete_map={
                         '🥇 Elite'     : '#1b5e20',
                         '🥈 Mid-tier'  : '#66bb6a',
                         '🥉 Developing': '#c8e6c9'
                     })
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 Cluster Summary Table")
    summary_cl = df.groupby('cluster_label')[cluster_features].mean().round(1)
    st.dataframe(summary_cl.style.background_gradient(cmap='Greens'),
                 use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 7 — TIME SERIES
# ══════════════════════════════════════════════════════════
elif page == "📈 Time Series":
    st.title("📈 Time Series Analysis")
    st.markdown("Player ranking evolution over time — ARIMA vs Gradient Boosting")
    st.markdown("---")

    # ── Player selector ──
    players_available = sorted(df_ts['player_name'].dropna().unique())
    selected_player   = st.selectbox("🎾 Select a Player", ["All Players"] + list(players_available))

    if selected_player == "All Players":
        ts_data = df_ts.copy()
        title_suffix = "All Players"
    else:
        ts_data = df_ts[df_ts['player_name'] == selected_player].copy()
        title_suffix = selected_player

    ts_data = ts_data.sort_values('full_date')

    # ── Monthly aggregation ──
    ts_data['year_month'] = ts_data['full_date'].dt.to_period('M')
    monthly = ts_data.groupby('year_month').agg(
        avg_ranking   = ('ranking_position',           'mean'),
        avg_contract  = ('contract_value_eur',         'mean'),
        avg_instagram = ('instagram_followers_millions','mean'),
        avg_engagement= ('engagement_rate_percent',    'mean'),
        player_count  = ('player_name',                'count')
    ).reset_index()
    monthly['year_month_dt'] = monthly['year_month'].dt.to_timestamp()
    monthly = monthly.sort_values('year_month_dt').reset_index(drop=True)

    st.markdown(f"**Periods available: {len(monthly)} | Players: {ts_data['player_name'].nunique()}**")

    # ── KPIs ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 Date Range",
                f"{ts_data['full_date'].min().strftime('%Y-%m')} → {ts_data['full_date'].max().strftime('%Y-%m')}")
    col2.metric("📊 Avg Ranking",   f"{monthly['avg_ranking'].mean():.1f}")
    col3.metric("💰 Avg Contract",  f"€{monthly['avg_contract'].mean():,.0f}")
    col4.metric("📱 Avg Instagram", f"{monthly['avg_instagram'].mean():.2f}M")

    st.markdown("---")

    # ── Full time series chart ──
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(monthly, x='year_month_dt', y='avg_ranking',
                      title=f"Avg Ranking Over Time — {title_suffix}",
                      color_discrete_sequence=['#2e7d32'])
        fig.update_yaxes(autorange='reversed')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          xaxis_title="Date",
                          yaxis_title="Avg Ranking Position")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(monthly, x='year_month_dt', y='avg_contract',
                      title=f"Avg Contract Value Over Time — {title_suffix}",
                      color_discrete_sequence=['#e65100'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          xaxis_title="Date",
                          yaxis_title="Contract Value (€)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Stationarity Test ──
    st.markdown("#### 🔬 Stationarity Test (ADF)")
    series = monthly['avg_ranking'].dropna()

    if len(series) >= 5:
        adf_result = adfuller(series)
        col1, col2, col3 = st.columns(3)
        col1.metric("ADF Statistic", f"{adf_result[0]:.4f}")
        col2.metric("p-value",       f"{adf_result[1]:.4f}")
        col3.metric("Stationary?",
                    "✅ Yes" if adf_result[1] < 0.05 else "❌ No")
    else:
        st.warning("⚠️ Not enough data points for stationarity test")

    st.markdown("---")

    # ── Models ──
    if len(monthly) >= 6:
        split     = int(len(monthly) * 0.8)
        train     = monthly[:split]
        test      = monthly[split:]

        st.markdown(f"**Train: {len(train)} periods | Test: {len(test)} periods**")

        # ARIMA
        try:
            arima_model  = ARIMA(train['avg_ranking'], order=(1,1,1))
            arima_fit    = arima_model.fit()
            arima_pred   = np.array(arima_fit.forecast(steps=len(test)))
            mae_arima    = mean_absolute_error(test['avg_ranking'], arima_pred)
            rmse_arima   = np.sqrt(mean_squared_error(test['avg_ranking'], arima_pred))
            mape_arima   = np.mean(np.abs((test['avg_ranking'].values - arima_pred) /
                                          (test['avg_ranking'].values + 1e-10))) * 100
            arima_ok     = True
        except:
            arima_ok     = False
            mae_arima = rmse_arima = mape_arima = 0
            arima_pred = np.full(len(test), train['avg_ranking'].mean())

        # Gradient Boosting
        def make_features(df):
            df = df.copy()
            df['month']        = df['year_month_dt'].dt.month
            df['quarter']      = df['year_month_dt'].dt.quarter
            df['year']         = df['year_month_dt'].dt.year
            df['lag_1']        = df['avg_ranking'].shift(1)
            df['lag_2']        = df['avg_ranking'].shift(2)
            df['rolling_mean'] = df['avg_ranking'].shift(1).rolling(3).mean()
            return df.dropna()

        feat_cols   = ['month','quarter','year','lag_1','lag_2','rolling_mean']
        monthly_f   = make_features(monthly)
        split_f     = int(len(monthly_f) * 0.8)
        train_f     = monthly_f[:split_f]
        test_f      = monthly_f[split_f:]

        if len(train_f) >= 3 and len(test_f) >= 1:
            gb       = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb.fit(train_f[feat_cols], train_f['avg_ranking'])
            gb_pred  = gb.predict(test_f[feat_cols])
            mae_gb   = mean_absolute_error(test_f['avg_ranking'], gb_pred)
            rmse_gb  = np.sqrt(mean_squared_error(test_f['avg_ranking'], gb_pred))
            mape_gb  = np.mean(np.abs((test_f['avg_ranking'].values - gb_pred) /
                                       (test_f['avg_ranking'].values + 1e-10))) * 100
            gb_ok    = True
        else:
            gb_ok    = False
            gb_pred  = np.array([])
            mae_gb = rmse_gb = mape_gb = 0

        # ── Model charts ──
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📉 ARIMA Forecast")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train['year_month_dt'],
                                     y=train['avg_ranking'],
                                     name='Train', line=dict(color='#2e7d32', width=2)))
            fig.add_trace(go.Scatter(x=test['year_month_dt'],
                                     y=test['avg_ranking'],
                                     name='Actual', line=dict(color='#1565c0', width=2)))
            fig.add_trace(go.Scatter(x=test['year_month_dt'],
                                     y=arima_pred,
                                     name='ARIMA', line=dict(color='#e65100',
                                     width=2, dash='dash')))
            fig.update_yaxes(autorange='reversed')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              title="ARIMA — Forecast vs Actual")
            st.plotly_chart(fig, use_container_width=True)
            if arima_ok:
                st.metric("ARIMA MAE",  f"{mae_arima:.2f}")
                st.metric("ARIMA RMSE", f"{rmse_arima:.2f}")
                st.metric("ARIMA MAPE", f"{mape_arima:.1f}%")

        with col2:
            st.markdown("#### 🌲 Gradient Boosting Forecast")
            if gb_ok:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train_f['year_month_dt'],
                                         y=train_f['avg_ranking'],
                                         name='Train', line=dict(color='#2e7d32', width=2)))
                fig.add_trace(go.Scatter(x=test_f['year_month_dt'],
                                         y=test_f['avg_ranking'],
                                         name='Actual', line=dict(color='#1565c0', width=2)))
                fig.add_trace(go.Scatter(x=test_f['year_month_dt'],
                                         y=gb_pred,
                                         name='GB Pred', line=dict(color='#6a1b9a',
                                         width=2, dash='dash')))
                fig.update_yaxes(autorange='reversed')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)',
                                  title="Gradient Boosting — Forecast vs Actual")
                st.plotly_chart(fig, use_container_width=True)
                st.metric("GB MAE",  f"{mae_gb:.2f}")
                st.metric("GB RMSE", f"{rmse_gb:.2f}")
                st.metric("GB MAPE", f"{mape_gb:.1f}%")
            else:
                st.warning("⚠️ Not enough data for Gradient Boosting")

        # ── Metrics comparison ──
        st.markdown("---")
        st.markdown("#### 📊 Model Comparison")
        col1, col2, col3 = st.columns(3)
        col1.metric("Metric", "MAE / RMSE / MAPE")
        col2.metric("ARIMA",
                    f"{mae_arima:.2f} / {rmse_arima:.2f} / {mape_arima:.1f}%")
        col3.metric("Gradient Boosting",
                    f"{mae_gb:.2f} / {rmse_gb:.2f} / {mape_gb:.1f}%"
                    if gb_ok else "N/A")

        best_ts = "ARIMA" if mae_arima <= mae_gb or not gb_ok else "Gradient Boosting"
        st.success(f"🏆 Best Time Series Model: **{best_ts}**")

        # ── Future forecast ──
        st.markdown("---")
        st.markdown("#### 🔮 Future Forecast — Next 6 Months")
        try:
            arima_full  = ARIMA(monthly['avg_ranking'], order=(1,1,1))
            arima_full_fit = arima_full.fit()
            future_pred = np.array(arima_full_fit.forecast(steps=6))
            future_dates= pd.date_range(
                start=monthly['year_month_dt'].max(),
                periods=7, freq='MS')[1:]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly['year_month_dt'],
                y=monthly['avg_ranking'],
                name='Historical',
                line=dict(color='#2e7d32', width=2),
                mode='lines+markers'
            ))
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_pred,
                name='Forecast',
                line=dict(color='#e65100', width=2, dash='dash'),
                mode='lines+markers'
            ))
            fig.add_vline(x=monthly['year_month_dt'].max(),
                          line_dash="dot", line_color="gray")
            fig.update_yaxes(autorange='reversed')
            fig.update_layout(
                title=f"ARIMA 6-Month Ranking Forecast — {title_suffix}",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Date",
                yaxis_title="Predicted Ranking"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Predicted rankings:**")
            forecast_df = pd.DataFrame({
                'Month'           : [d.strftime('%Y-%m') for d in future_dates],
                'Predicted Ranking': [f"{p:.1f}" for p in future_pred]
            })
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Future forecast not available: {e}")
    else:
        st.warning("⚠️ Not enough time periods to run models. Select 'All Players' for more data.")

    # ── Instagram over time ──
    st.markdown("---")
    st.markdown("#### 📱 Social Media Growth Over Time")
    fig = px.line(monthly, x='year_month_dt', y='avg_instagram',
                  title=f"Avg Instagram Followers — {title_suffix}",
                  color_discrete_sequence=['#6a1b9a'])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)