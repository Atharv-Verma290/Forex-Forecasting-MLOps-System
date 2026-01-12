from datetime import datetime, timedelta, timezone
from airflow.sdk import dag, task 
from dotenv import load_dotenv
from data_preprocessing import ForexDataPreProcessing #type: ignore
from utility import SQLTableBuilder, TrainTestTableStrategy #type: ignore
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
        dag_id="forex_training_pipeline",
        default_args=default_args,
        start_date=datetime(2026, 1, 12),
        schedule='@weekly',
    )
def forex_prediction_pipeline():

    @task
    def data_preprocessing():
        try:
            conn = psycopg2.connect(database="app_db", user="admin", password="admin", host="app-postgres", port="5432")
            print("Database connected successfully")
            query = """SELECT * FROM eur_usd_final ORDER BY datetime DESC;"""
            input_df = pd.read_sql(query, conn)
            
            print("Input data retrieved: ", input_df)

            preprocessor = ForexDataPreProcessing(input_df)
            train_df, test_df = preprocessor.preprocess_data()

            # create train table
            table_builder = SQLTableBuilder(TrainTestTableStrategy(tablename="eur_usd_train"))
            creation_query = table_builder.get_create_query(df=train_df)
            cur = conn.cursor()
            cur.execute(creation_query)
            conn.commit()
            insertion_query, values = table_builder.get_insert_query(data=train_df)
            extras.execute_values(cur, insertion_query, values)
            conn.commit()

            # create test table
            table_builder.set_strategy(TrainTestTableStrategy(tablename="eur_usd_test"))
            creation_query = table_builder.get_create_query(df=test_df)
            cur.execute(creation_query)
            conn.commit()
            insertion_query, values = table_builder.get_insert_query(data=test_df)
            extras.execute_values(cur, insertion_query, values)
            conn.commit()

            conn.close()

        except Exception as e:
            print(e)
        
    data_preprocessing()
    
prediction_pipeline = forex_prediction_pipeline()

