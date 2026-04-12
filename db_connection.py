from sqlalchemy import create_engine
import pandas as pd

DB_USER = "postgres"
DB_PASSWORD = "."
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "dw_padel"

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

def get_data(query):
    with engine.connect() as conn:
        return pd.read_sql(query, conn)

if __name__ == "__main__":
    df = get_data("SELECT * FROM dim_player LIMIT 5")
    print("✅ Connexion réussie !")
    print(df)