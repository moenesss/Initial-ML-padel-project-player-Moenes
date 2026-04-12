import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🎾 Padel ML Dashboard",
    page_icon="🎾",
    layout="wide"
)

# ══════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("players_clean.csv")
    return df

df = load_data()

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
st.sidebar.image("https://img.icons8.com/emoji/96/tennis.png", width=80)
st.sidebar.title("🎾 Padel ML Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio("📌 Navigation", [
    "🏠 Overview",
    "📊 Player Stats",
    "🤖 Top Player Predictor",
    "💰 Contract Predictor",
    "🔵 Player Clusters"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"📋 **Dataset:** {df.shape[0]} players × {df.shape[1]} features")

# ══════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🎾 Padel ML Dashboard")
    st.markdown("### Complete Machine Learning Analysis of Padel Players")
    st.markdown("---")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Players", df.shape[0])
    col2.metric("⭐ Top 20 Players", int(df['is_top_player'].sum()))
    col3.metric("💰 Avg Contract (€)",
                f"{df['contract_value_eur'].mean():,.0f}")
    col4.metric("📱 Avg Instagram (M)",
                f"{df['instagram_followers_millions'].mean():.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏆 Top 10 Players by Contract Value")
        top10 = df.nlargest(10, 'contract_value_eur')[
            ['contract_value_eur', 'total_titles',
             'instagram_followers_millions', 'is_top_player']
        ].reset_index(drop=True)
        top10.index += 1
        st.dataframe(top10.style.format({
            'contract_value_eur': '€{:,.0f}',
            'instagram_followers_millions': '{:.2f}M'
        }), use_container_width=True)

    with col2:
        st.markdown("#### 📊 Project ML Summary")
        summary = pd.DataFrame({
            'Section': ['Classification', 'Regression', 'Clustering'],
            'Task': ['Predict Top 20', 'Predict Contract €', 'Player Profiles'],
            'Best Model': ['Random Forest', 'Random Forest', 'Hierarchical'],
            'Score': ['F1 = 1.0', 'R² = 1.0', 'Silhouette = 0.596']
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🔥 Correlation Heatmap")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()
    fig = px.imshow(corr, color_continuous_scale='RdBu_r',
                    title="Feature Correlation Matrix",
                    height=500)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 2 — PLAYER STATS
# ══════════════════════════════════════════════════════════
elif page == "📊 Player Stats":
    st.title("📊 Player Statistics")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Ranking Distribution")
        fig = px.histogram(df, x='ranking_position', nbins=15,
                           color_discrete_sequence=['steelblue'],
                           title="Player Ranking Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 💰 Contract Value Distribution")
        fig = px.histogram(df, x='contract_value_eur', nbins=15,
                           color_discrete_sequence=['seagreen'],
                           title="Contract Value Distribution (€)")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📱 Social Media vs Contract Value")
        fig = px.scatter(df,
                         x='total_social_followers',
                         y='contract_value_eur',
                         color='is_top_player',
                         size='total_titles',
                         title="Social Followers vs Contract Value",
                         labels={
                             'total_social_followers': 'Total Social Followers (M)',
                             'contract_value_eur': 'Contract Value (€)',
                             'is_top_player': 'Top Player'
                         },
                         color_discrete_map={0: 'steelblue', 1: 'gold'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 🏆 Titles vs Win Rate")
        fig = px.scatter(df,
                         x='total_titles',
                         y='win_rate_finals',
                         color='is_top_player',
                         size='contract_value_eur',
                         title="Titles vs Win Rate in Finals",
                         color_discrete_map={0: 'steelblue', 1: 'gold'})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 Full Dataset")
    st.dataframe(df, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 3 — TOP PLAYER PREDICTOR
# ══════════════════════════════════════════════════════════
elif page == "🤖 Top Player Predictor":
    st.title("🤖 Top Player Predictor")
    st.markdown("Predict if a player will be in the **Top 20** ranking")
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
        instagram = st.slider("Instagram Followers (M)", 0.0, 5.0, 0.5)
        tiktok    = st.slider("TikTok Followers (M)", 0.0, 5.0, 0.3)
        twitter   = st.slider("Twitter Followers (K)", 0.0, 500.0, 50.0)
        youtube   = st.slider("YouTube Subscribers (K)", 0.0, 500.0, 50.0)

    with col3:
        st.markdown("#### 💼 Contract & Profile")
        yearly_rackets  = st.slider("Yearly Rackets", 0, 50, 12)
        engagement_rate = st.slider("Engagement Rate (%)", 0.0, 20.0, 3.0)
        height_cm       = st.slider("Height (cm)", 160, 210, 180)
        gender          = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==0 else "Female")

    st.markdown("---")

    if st.button("🔮 Predict Top Player", use_container_width=True):
        total_social = instagram + tiktok + twitter/1000 + youtube/1000

        input_data = pd.DataFrame([{
            col: 0 for col in X_clf.columns
        }])

        input_data['total_titles']               = total_titles
        input_data['win_rate_finals']            = win_rate_finals
        input_data['total_finals']               = total_finals
        input_data['ranking_change']             = ranking_change
        input_data['instagram_followers_millions'] = instagram
        input_data['tiktok_followers_millions']  = tiktok
        input_data['twitter_followers_thousands'] = twitter
        input_data['youtube_subscribers_thousands'] = youtube
        input_data['yearly_rackets']             = yearly_rackets
        input_data['engagement_rate_percent']    = engagement_rate
        input_data['height_cm']                  = height_cm
        input_data['gender']                     = gender
        input_data['total_social_followers']     = total_social

        input_sc = sc_clf.transform(input_data)
        prediction = clf.predict(input_sc)[0]
        probability = clf.predict_proba(input_sc)[0][1]

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.success(f"⭐ TOP 20 PLAYER — Confidence: {probability:.1%}")
            else:
                st.warning(f"📉 NOT Top 20 — Confidence: {1-probability:.1%}")

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Top Player Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "gold"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 4 — CONTRACT PREDICTOR
# ══════════════════════════════════════════════════════════
elif page == "💰 Contract Predictor":
    st.title("💰 Contract Value Predictor")
    st.markdown("Predict a player's **sponsorship contract value** in €")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🏅 Performance")
        r_total_titles    = st.slider("Total Titles", 0, 200, 20, key='r1')
        r_win_rate        = st.slider("Win Rate Finals (%)", 0.0, 100.0, 50.0, key='r2')
        r_total_finals    = st.slider("Total Finals", 0, 100, 10, key='r3')
        r_ranking_change  = st.slider("Ranking Change", -50, 50, 0, key='r4')

    with col2:
        st.markdown("#### 📱 Social Media")
        r_instagram = st.slider("Instagram Followers (M)", 0.0, 5.0, 0.5, key='r5')
        r_tiktok    = st.slider("TikTok Followers (M)", 0.0, 5.0, 0.3, key='r6')
        r_twitter   = st.slider("Twitter Followers (K)", 0.0, 500.0, 50.0, key='r7')
        r_youtube   = st.slider("YouTube Subscribers (K)", 0.0, 500.0, 50.0, key='r8')

    with col3:
        st.markdown("#### 💼 Profile")
        r_yearly_rackets  = st.slider("Yearly Rackets", 0, 50, 12, key='r9')
        r_engagement      = st.slider("Engagement Rate (%)", 0.0, 20.0, 3.0, key='r10')
        r_height          = st.slider("Height (cm)", 160, 210, 180, key='r11')
        r_gender          = st.selectbox("Gender", [0, 1],
                                          format_func=lambda x: "Male" if x==0 else "Female",
                                          key='r12')

    st.markdown("---")

    if st.button("💰 Predict Contract Value", use_container_width=True):
        r_total_social = r_instagram + r_tiktok + r_twitter/1000 + r_youtube/1000

        input_r = pd.DataFrame([{col: 0 for col in X_reg.columns}])
        input_r['total_titles']                  = r_total_titles
        input_r['win_rate_finals']               = r_win_rate
        input_r['total_finals']                  = r_total_finals
        input_r['ranking_change']                = r_ranking_change
        input_r['instagram_followers_millions']  = r_instagram
        input_r['tiktok_followers_millions']     = r_tiktok
        input_r['twitter_followers_thousands']   = r_twitter
        input_r['youtube_subscribers_thousands'] = r_youtube
        input_r['yearly_rackets']                = r_yearly_rackets
        input_r['engagement_rate_percent']       = r_engagement
        input_r['height_cm']                     = r_height
        input_r['gender']                        = r_gender
        input_r['total_social_followers']        = r_total_social

        input_r_sc = sc_reg.transform(input_r)
        predicted_value = reg.predict(input_r_sc)[0]

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"💰 Predicted Contract Value: **€{predicted_value:,.0f}**")

            if predicted_value >= 200000:
                tier = "⭐ Elite — Premium sponsorship tier"
            elif predicted_value >= 100000:
                tier = "🥈 Mid-tier — Standard sponsorship"
            else:
                tier = "🥉 Developing — Entry sponsorship"
            st.info(f"Player Tier: {tier}")

        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_value,
                delta={'reference': df['contract_value_eur'].mean()},
                title={'text': "Contract Value (€)"},
                gauge={
                    'axis': {'range': [0, df['contract_value_eur'].max()]},
                    'bar': {'color': "seagreen"},
                    'steps': [
                        {'range': [0, 100000], 'color': "lightgray"},
                        {'range': [100000, 200000], 'color': "lightyellow"},
                        {'range': [200000, df['contract_value_eur'].max()],
                         'color': "lightgreen"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 5 — CLUSTERS
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

    X_cl = df[cluster_features].copy()
    sc_cl = StandardScaler()
    X_cl_sc = sc_cl.fit_transform(X_cl)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = km.fit_predict(X_cl_sc)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_cl_sc)
    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]

    cluster_labels = {0: "Mid-tier", 1: "Developing", 2: "Elite"}
    df['cluster_label'] = df['cluster'].map(cluster_labels)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🗺️ PCA 2D — Player Clusters")
        fig = px.scatter(df, x='PC1', y='PC2',
                         color='cluster_label',
                         size='contract_value_eur',
                         title="Player Clusters (PCA 2D)",
                         color_discrete_map={
                             'Elite': 'gold',
                             'Mid-tier': 'steelblue',
                             'Developing': 'lightgray'
                         })
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📊 Cluster Profile Comparison")
        profile = df.groupby('cluster_label')[
            ['contract_value_eur', 'total_titles',
             'total_social_followers', 'win_rate_finals']
        ].mean().reset_index()

        fig = px.bar(profile.melt(id_vars='cluster_label'),
                     x='variable', y='value',
                     color='cluster_label',
                     barmode='group',
                     title="Average Profile per Cluster",
                     color_discrete_map={
                         'Elite': 'gold',
                         'Mid-tier': 'steelblue',
                         'Developing': 'lightgray'
                     })
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📋 Cluster Summary")
    summary_cl = df.groupby('cluster_label')[cluster_features].mean().round(1)
    st.dataframe(summary_cl.style.background_gradient(cmap='YlOrRd'),
                 use_container_width=True)

    col1, col2, col3 = st.columns(3)
    for i, (label, color) in enumerate(
            zip(['Elite', 'Mid-tier', 'Developing'],
                ['🥇', '🥈', '🥉'])):
        count = (df['cluster_label'] == label).sum()
        avg_contract = df[df['cluster_label'] == label]['contract_value_eur'].mean()
        [col1, col2, col3][i].metric(
            f"{color} {label}",
            f"{count} players",
            f"Avg €{avg_contract:,.0f}"
        )