import psycopg2
import psycopg2.extras as extras
import pandas as pd 
from datetime import datetime, timedelta 

conn = psycopg2.connect(database="app_db", user="admin", password="admin", host="localhost", port=5433)
query = """
    CREATE TABLE IF NOT EXISTS eur_usd_raw(
        id SERIAL PRIMARY KEY,
        datetime DATE NOT NULL UNIQUE,
        open NUMERIC,
        high NUMERIC,
        low NUMERIC,
        close NUMERIC
    );
"""
cur = conn.cursor()
cur.execute(query)
print("Table created successfully.")
conn.commit()


df = pd.read_csv("eur_usd_forex_data.csv")
print(df)

cols = ("datetime", "open", "high", "low", "close")
values = [tuple(x) for x in df[list(cols)].to_numpy()]
query = f"""
    INSERT INTO eur_usd_raw ({', '.join(cols)}) VALUES %s
    ON CONFLICT (datetime) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close;    
"""

extras.execute_values(cur, query, values)
print(f"{len(df)} record(s) added to the database.")

conn.commit()
conn.close()
