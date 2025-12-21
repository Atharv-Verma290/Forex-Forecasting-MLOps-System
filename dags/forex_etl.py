from datetime import datetime, timedelta
from airflow.sdk import dag, task 
from data_ingestion import TwelveDataIngestor #type: ignore
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import psycopg2
load_dotenv()



default_args = {
    'owner': 'atharv',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

@dag(dag_id="forex_pipeline_dag", default_args=default_args)
def forex_etl_pipeline():

    @task
    def extract_data():
        data_ingestor = TwelveDataIngestor()
        extracted_data = data_ingestor.ingest(symbol="EUR/USD")

        print("Adding extracted_data to database")
        record = extracted_data[0]
        
        try: 
            conn = psycopg2.connect(database="app_db", user="admin", password="admin", host="app-postgres", port="5432")
            print("Database connected successfully.")
            
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS eur_usd_raw(
                id SERIAL PRIMARY KEY,
                datetime DATE NOT NULL UNIQUE,
                open NUMERIC,
                high NUMERIC,
                low NUMERIC,
                close NUMERIC
            );
            """)            
            print("Table created successfully.")

            query = """
            INSERT INTO eur_usd_raw (datetime, open, high, low, close) VALUES
            (%s, %s, %s, %s, %s)
            ON CONFLICT (datetime) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close;    
            """

            values = (
                record["datetime"],
                record["open"],
                record["high"],
                record["low"],
                record["close"]
            )

            cur.execute(query, values)
            print("Record added to the database.")
            conn.commit()
            conn.close()
        
        except Exception as e:
            print(e)
        
        print(record)
        return "EUR_USD_raw"

    @task
    def transform_data(raw_table):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        supabase: Client = create_client(url, key)

        raw_data = supabase.table(raw_table).select("*").order("datetime", desc=True).execute()
        raw_records = raw_data.data
        print("Successfully fetched raw_data from supabase")
        print(f"Latest new record from supabase: {raw_records[0]}")

    @task 
    def load_data():
        pass
    
    raw_table = extract_data()
    # transform_data(raw_table=raw_table)

etl_pipeline = forex_etl_pipeline()