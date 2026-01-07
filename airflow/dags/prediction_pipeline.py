from datetime import datetime, timedelta
from airflow.sdk import dag, task 
from dotenv import load_dotenv
import os
import psycopg2
import psycopg2.extras as extras
import pandas as pd
import numpy as np
import requests
load_dotenv()

default_args = {
    'owner': 'atharv',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

@dag(
        dag_id="forex_prediction_pipeline",
        default_args=default_args,
        start_date=datetime(2026, 1, 5)
    )
def forex_prediction_pipeline():

    @task
    def predict_tomorrow():
        API_URL = "http://model-service:8000/predict_forex"
        try:
            conn = psycopg2.connect(database="app_db", user="admin", password="admin", host="app-postgres", port="5432")
            print("Database connected successfully")
            query = """SELECT * FROM eur_usd_final ORDER BY datetime DESC LIMIT 1;"""
            input_df = pd.read_sql(query, conn)
            conn.close()
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
        conn = psycopg2.connect(database="app_db", user="admin", password="admin", host="app-postgres", port="5432")
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS eur_usd_predictions (
                id SERIAL PRIMARY KEY,
                feature_date DATE NOT NULL UNIQUE,
                prediction_date DATE,
                predicted_direction INTEGER,
                model_name TEXT,
                model_version TEXT
            );
        """)
        print("Table created successfully.")
        
        feature_date = datetime.strptime(data["datetime"], "%Y-%m-%d %H:%M:%S").date()
        prediction_date = feature_date + timedelta(days=1)
        predicted_direction = data["prediction"]
        model_name = "Random forest"
        model_version = "1"
        
        cols = ("feature_date", "prediction_date", "predicted_direction", "model_name", "model_version")
        values = [(feature_date, prediction_date, predicted_direction, model_name, model_version)]
        query = f"""
            INSERT INTO eur_usd_predictions ({', '.join(list(cols))}) VALUES %s
            ON CONFLICT (feature_date) DO UPDATE SET
                prediction_date = EXCLUDED.prediction_date,
                predicted_direction = EXCLUDED.predicted_direction,
                model_name = EXCLUDED.model_name,
                model_version = EXCLUDED.model_version;
        """
        extras.execute_values(cur, query, values)
        print("Record(s) added to the database.")
        conn.commit()
        conn.close()

        
    predictions = predict_tomorrow()
    store_predictions(predictions)
    
prediction_pipeline = forex_prediction_pipeline()

