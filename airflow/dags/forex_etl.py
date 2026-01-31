from datetime import datetime, timedelta
from airflow.sdk import dag, task
from airflow.datasets import Dataset
from src.data_ingestion import TwelveDataIngestor
from src.data_transformation import ForexDataTransformation 
from src.utility import SQLTableBuilder, RawTableStrategy, StagingTableStrategy, FinalTableStrategy 
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import psycopg2.extras as extras
import pandas as pd

EUR_USD_FINAL_DATASET = Dataset("postgres://app-postgres:5432/app_db/public/eur_usd_final")
CONNECTION_URL = "postgresql://admin:admin@app-postgres:5432/app_db"

default_args = {
    'owner': 'atharv',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

@dag(
        dag_id="forex_etl_pipeline", 
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
            engine = create_engine(CONNECTION_URL)
            print("Database connected successfully.")
            
            table_builder = SQLTableBuilder(RawTableStrategy(tablename="eur_usd_raw"))
            creation_query = table_builder.get_create_query()
            with engine.connect() as conn:
                conn.execute(text(creation_query))
                conn.commit()
            print("Table created successfully.")

            insertion_query, values = table_builder.get_insert_query(data=extracted_data)
            raw_conn = engine.raw_connection()
            cur = raw_conn.cursor()
            extras.execute_values(cur, insertion_query, values)
            raw_conn.commit()
            print("Record(s) added to the database.")
            raw_conn.close()            
        
        except Exception as e:
            print(e)
        
        return "eur_usd_raw"

    @task
    def transform_data(raw_table):
        engine = create_engine(CONNECTION_URL)
        print("Database connected successfully.")

        raw_data = pd.read_sql(
            f"SELECT * FROM {raw_table} ORDER BY datetime DESC;",
            engine
        )

        transformer = ForexDataTransformation(raw_data)
        transformed_data = transformer.apply_transformation()

        staging_table = "eur_usd_staging"
        table_builder = SQLTableBuilder(StagingTableStrategy(tablename=staging_table))
        creation_query = table_builder.get_create_query(df=transformed_data)
        with engine.connect() as conn:
            conn.execute(text(creation_query))
            conn.commit()

        insertion_query, values = table_builder.get_insert_query(data=transformed_data)
        try:
            raw_conn = engine.raw_connection()
            cur = raw_conn.cursor()
            extras.execute_values(cur, insertion_query, values)
            raw_conn.commit()
            print("transformed data into staging.")

        except (Exception, SQLAlchemyError) as error:
            print(f"Error: {error}")
            if 'raw_conn' in locals():
                raw_conn.rollback()
            return 1
        finally:
            if 'raw_conn' in locals():
                raw_conn.close()

        return staging_table

    @task(outlets=[EUR_USD_FINAL_DATASET]) 
    def load_data(staging_table):
        engine = create_engine(CONNECTION_URL)
        print("Database connected successfully.")

        table_builder = SQLTableBuilder(FinalTableStrategy(tablename="eur_usd_final", staging_tablename=staging_table))
        creation_query = table_builder.get_create_query()

        with engine.connect() as conn:
            conn.execute(text(creation_query))
            conn.commit()

        print("Final data added.")
        

    
    raw_table = extract_data()
    staging_table = transform_data(raw_table=raw_table)
    load_data(staging_table)

etl_pipeline = forex_etl_pipeline()