from db_connection import get_data

query = """
    SELECT 
        fps.*,
        p.gender,
        p.dominant_hand,
        p.height_cm,
        p.nationality_location_id,
        p.is_active,
        b.brand_name,
        b.market_position,
        b.market_share_percent
    FROM fact_player_stats fps
    LEFT JOIN dim_player p ON fps.player_id = p.player_id
    LEFT JOIN dim_brand b ON fps.brand_id = b.brand_id
"""

df = get_data(query)
print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())
print("\nColumns:", list(df.columns))

# Save to CSV
df.to_csv("players_data.csv", index=False)
print("\n✅ Saved to players_data.csv")