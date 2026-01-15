from datetime import datetime, timedelta
from airflow.sdk import dag, task
from airflow.datasets import Dataset
from data_ingestion import TwelveDataIngestor #type: ignore
from data_transformation import ForexDataTransformation #type: ignore
from utility import SQLTableBuilder, RawTableStrategy, StagingTableStrategy, FinalTableStrategy #type: ignore
from dotenv import load_dotenv
import os
import psycopg2
import psycopg2.extras as extras
import pandas as pd
import numpy as np
load_dotenv()

EUR_USD_FINAL_DATASET = Dataset("postgres://app-postgres:5432/app_db/public/eur_usd_final")

default_args = {
    'owner': 'atharv',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

@dag(
        dag_id="forex_pipeline_dag", 
        default_args=default_args,
        start_date=datetime(2025, 12, 23)
    )
def forex_etl_pipeline():

    @task
    def extract_data():
        data_ingestor = TwelveDataIngestor()
        extracted_data = data_ingestor.ingest(symbol="EUR/USD")

        print("Adding extracted_data to database")
        try: 
            conn = psycopg2.connect(database="app_db", user="admin", password="admin", host="app-postgres", port="5432")
            print("Database connected successfully.")
            
            table_builder = SQLTableBuilder(RawTableStrategy(tablename="eur_usd_raw"))
            creation_query = table_builder.get_create_query()
            cur = conn.cursor()
            cur.execute(creation_query)            
            print("Table created successfully.")

            insertion_query, values = table_builder.get_insert_query(data=extracted_data)
            extras.execute_values(cur, insertion_query, values)
            print("Record(s) added to the database.")
            conn.commit()
            conn.close()
        
        except Exception as e:
            print(e)
        
        return "eur_usd_raw"

    @task
    def transform_data(raw_table):
        conn = psycopg2.connect(database="app_db", user="admin", password="admin", host="app-postgres", port="5432")
        print("Database connected successfully.")

        raw_data = pd.read_sql(
            f"SELECT * FROM {raw_table} ORDER BY datetime DESC;",
            conn
        )
        
        transformer = ForexDataTransformation(raw_data)
        transformed_data = transformer.apply_transformation()
        
        cur = conn.cursor()
        staging_table = "eur_usd_staging"
        table_builder = SQLTableBuilder(StagingTableStrategy(tablename="eur_usd_staging"))
        creation_query = table_builder.get_create_query(df=transformed_data)
        cur.execute(creation_query)
        conn.commit()

        insertion_query, values = table_builder.get_insert_query(data=transformed_data)
        try:
            extras.execute_values(cur, insertion_query, values)
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            conn.rollback()
            cur.close()
            return 1
        
        print("transformed data into staging.")
        conn.commit()
        cur.close()
        conn.close()

        return staging_table

    @task(outlets=[EUR_USD_FINAL_DATASET]) 
    def load_data(staging_table):
        conn = psycopg2.connect(database="app_db", user="admin", password="admin", host="app-postgres", port="5432")

        table_builder = SQLTableBuilder(FinalTableStrategy(tablename="eur_usd_final", staging_tablename=staging_table))
        creation_query = table_builder.get_create_query()

        cur = conn.cursor()
        cur.execute(creation_query)
        conn.commit()
        cur.close()
        conn.close()
        print("Final data added.")

    
    raw_table = extract_data()
    staging_table = transform_data(raw_table=raw_table)
    load_data(staging_table)

etl_pipeline = forex_etl_pipeline()