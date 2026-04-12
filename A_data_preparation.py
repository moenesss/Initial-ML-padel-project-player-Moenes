import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load data ──────────────────────────────────────────────
df = pd.read_csv("players_data.csv")
print("✅ Data loaded:", df.shape)

# ══════════════════════════════════════════════════════════
# 1. OVERVIEW
# ══════════════════════════════════════════════════════════
print("\n── Data Types ──")
print(df.dtypes)

print("\n── Missing Values ──")
print(df.isnull().sum())

print("\n── Basic Stats ──")
print(df.describe())

# ══════════════════════════════════════════════════════════
# 2. DROP USELESS COLUMNS
# ══════════════════════════════════════════════════════════
cols_to_drop = ['player_stat_id', 'player_id', 'date_id',
                'tour_id', 'brand_id', 'product_id',
                'nationality_location_id']
df.drop(columns=cols_to_drop, inplace=True)
print("\n✅ Dropped ID columns. Remaining:", df.shape[1], "columns")

# ══════════════════════════════════════════════════════════
# 3. HANDLE MISSING VALUES
# ══════════════════════════════════════════════════════════
# Numeric columns → fill with median
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical columns → fill with mode
cat_cols = df.select_dtypes(include=['object', 'str']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("✅ Missing values handled")
print(df.isnull().sum().sum(), "missing values remaining")

# ══════════════════════════════════════════════════════════
# 4. OUTLIERS — Visualize with boxplots
# ══════════════════════════════════════════════════════════
key_cols = ['ranking_position', 'contract_value_eur',
            'instagram_followers_millions', 'win_rate_finals',
            'total_titles', 'sponsorship_value_annual_eur']

plt.figure(figsize=(14, 8))
for i, col in enumerate(key_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=df[col], color='skyblue')
    plt.title(col)
plt.tight_layout()
plt.savefig("outliers_boxplot.png")
plt.show()
print("✅ Outliers boxplot saved")

# ══════════════════════════════════════════════════════════
# 5. ENCODING
# ══════════════════════════════════════════════════════════
print("\n── Categorical columns to encode ──")
print(df[cat_cols].nunique())

# Label encoding for binary columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

binary_cols = ['gender', 'dominant_hand', 'is_active', 'sponsorship_type']
for col in binary_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# One-hot encoding for multi-category columns
multi_cols = ['brand_name', 'market_position']
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

print("✅ Encoding done. Shape:", df.shape)

# ══════════════════════════════════════════════════════════
# 6. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════
# Total social media followers
df['total_social_followers'] = (
    df['instagram_followers_millions'] +
    df['tiktok_followers_millions'] +
    df['twitter_followers_thousands'] / 1000 +
    df['youtube_subscribers_thousands'] / 1000
)

# Performance score
df['performance_score'] = (
    df['total_titles'] * 10 +
    df['win_rate_finals'] * 5 -
    df['ranking_position'] * 0.1
)

# Is top player (top 20)
df['is_top_player'] = (df['ranking_position'] <= 20).astype(int)

print("✅ New features created:")
print("   - total_social_followers")
print("   - performance_score")
print("   - is_top_player")

# ══════════════════════════════════════════════════════════
# 7. SCALING
# ══════════════════════════════════════════════════════════
from sklearn.preprocessing import StandardScaler

scale_cols = ['ranking_position', 'ranking_points', 'contract_value_eur',
              'instagram_followers_millions', 'win_rate_finals',
              'total_titles', 'sponsorship_value_annual_eur',
              'total_social_followers', 'performance_score', 'height_cm']

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])

print("✅ Scaling done")

# ══════════════════════════════════════════════════════════
# 8. SAVE CLEAN DATA
# ══════════════════════════════════════════════════════════
df.to_csv("players_clean.csv", index=False)
df_scaled.to_csv("players_scaled.csv", index=False)
print("\n✅ Saved: players_clean.csv and players_scaled.csv")
print("Final shape:", df.shape)