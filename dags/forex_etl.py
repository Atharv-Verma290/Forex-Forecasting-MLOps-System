from datetime import datetime, timedelta
from airflow.sdk import dag, task 
from data_ingestion import TwelveDataIngestor #type: ignore
from data_transformation import ForexDataTransformation #type: ignore
from dotenv import load_dotenv
import os
import psycopg2
import psycopg2.extras as extras
import pandas as pd
import numpy as np
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

            cols = ("datetime", "open", "high", "low", "close")
            values = [
                (rec["datetime"], rec["open"], rec["high"], rec["low"], rec["close"])
                for rec in extracted_data
            ]
            query = f"""
            INSERT INTO eur_usd_raw ({', '.join(cols)}) VALUES %s
            ON CONFLICT (datetime) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close;    
            """
            extras.execute_values(cur, query, values)
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
        cur = conn.cursor()

        query=f"""
        SELECT * FROM {raw_table};
        """
        cur.execute(query)
        raw_data = cur.fetchall()

        raw_data = pd.DataFrame(raw_data, columns=['id', 'datetime', 'open', 'high', 'low', 'close'])
        
        transformer = ForexDataTransformation(raw_data)
        transformed_data = transformer.apply_transformation()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS eur_usd_features(
            id SERIAL PRIMARY KEY,
            datetime DATE NOT NULL UNIQUE,
            open NUMERIC,
            high NUMERIC,
            low NUMERIC,
            close NUMERIC,
            close_ratio_2 NUMERIC,
            close_ratio_5 NUMERIC,
            close_ratio_60 NUMERIC,
            close_ratio_250 NUMERIC,
            close_ratio_1000 NUMERIC
            );
        """)
        conn.commit()

        tuples = [tuple(x) for x in transformed_data.to_numpy()]
        cols = ', '.join(list(transformed_data.columns))

        query = "INSERT INTO eur_usd_features (%s) VALUES %%s" % (cols)
        try:
            extras.execute_values(cur, query, tuples)
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            conn.rollback()
            cur.close()
            return 1
        
        print("The transformed_data inserted in db")
        cur.close()
        conn.close()

        
    @task 
    def load_data():
        pass
    
    raw_table = extract_data()
    transform_data(raw_table=raw_table)

etl_pipeline = forex_etl_pipeline()