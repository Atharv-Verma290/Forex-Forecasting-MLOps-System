from datetime import datetime, timedelta
from airflow.sdk import dag, task 
from data_ingestion import TwelveDataIngestor #type: ignore
from supabase import create_client, Client
from dotenv import load_dotenv
import os
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
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        supabase: Client = create_client(url, key)

        data_ingestor = TwelveDataIngestor()
        extracted_data = data_ingestor.ingest(symbol="EUR/USD")

        print("Adding extracted_data to database")
        record = extracted_data[0]
        extracted_data = supabase.table("EUR_USD_raw").upsert(
            {
                "datetime": record["datetime"],
                "open": float(record["open"]),
                "high": float(record["high"]),
                "low": float(record["low"]),
                "close": float(record["close"]),
            },
            on_conflict="datetime"
        ).execute()
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
    transform_data(raw_table=raw_table)

etl_pipeline = forex_etl_pipeline()