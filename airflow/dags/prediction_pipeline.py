from datetime import datetime, timedelta, timezone
from airflow.sdk import dag, task 
from airflow.datasets import Dataset
from dotenv import load_dotenv
from src.utility import next_forex_trading_day, SQLTableBuilder, PredictionTableStrategy  #type: ignore
from sqlalchemy import create_engine, text
import psycopg2.extras as extras
import pandas as pd
import requests
load_dotenv()

EUR_USD_FINAL_DATASET = Dataset("postgres://app-postgres:5432/app_db/public/eur_usd_final")
CONNECTION_URL = "postgresql://admin:admin@app-postgres:5432/app_db"

default_args = {
    'owner': 'atharv',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

@dag(
        dag_id="forex_prediction_pipeline",
        default_args=default_args,
        start_date=datetime(2026, 1, 5),
        schedule=[EUR_USD_FINAL_DATASET],
        catchup=False
    )
def forex_prediction_pipeline():

    @task
    def predict_tomorrow():
        API_URL = "http://model-service:8000/predict_forex"
        try:
            engine = create_engine(CONNECTION_URL)
            print("Database connected successfully")
            query = """SELECT * FROM eur_usd_final ORDER BY datetime DESC LIMIT 1;"""
            input_df = pd.read_sql(query, engine)
            print("Input data retrieved: ", input_df)

            payload = {
                "input_data": input_df.to_json()
            }
            response = requests.post(API_URL, json=payload)
            data = response.json()
            print("Got the model service response: ")
            print(data)

        except Exception as e:
            print(e)
        
        return data 
    
    @task 
    def store_predictions(data):
        engine = create_engine(CONNECTION_URL)

        table_builder = SQLTableBuilder(PredictionTableStrategy(tablename="eur_usd_predictions"))
        creation_query = table_builder.get_create_query()
        with engine.connect() as conn:
            conn.execute(text(creation_query))
        print("Table created successfully.")

        record = {
            "feature_date": datetime.strptime(data["datetime"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc),
            "prediction_date": next_forex_trading_day(datetime.strptime(data["datetime"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)),
            "predicted_direction": data["prediction"],
            "model_name": data["model_name"],
            "model_version": data["model_version"]
        }

        insertion_query, values = table_builder.get_insert_query(data=record)
        raw_conn = engine.raw_connection()
        cur = raw_conn.cursor()
        extras.execute_values(cur, insertion_query, values)
        print("Record(s) added to the database.")
        raw_conn.commit()
        raw_conn.close()

        
    predictions = predict_tomorrow()
    store_predictions(predictions)
    
prediction_pipeline = forex_prediction_pipeline()

