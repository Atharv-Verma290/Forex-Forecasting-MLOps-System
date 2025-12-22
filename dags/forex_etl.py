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
        # cur = conn.cursor()

        # query=f"""
        # SELECT * FROM {raw_table};
        # """
        # cur.execute(query)
        # raw_data = cur.fetchall()
        # raw_data = pd.DataFrame(raw_data, columns=['id', 'datetime', 'open', 'high', 'low', 'close'])

        raw_data = pd.read_sql(
            f"SELECT * FROM {raw_table} ORDER BY datetime DESC;",
            conn
        )
        
        transformer = ForexDataTransformation(raw_data)
        transformed_data = transformer.apply_transformation()
        
        cur = conn.cursor()
        staging_table = "eur_usd_staging"

        cur.execute(f"DROP TABLE IF EXISTS {staging_table};")

        columns_sql = []
        for col, dtype in transformed_data.dtypes.items():
            if col == "id":
                sql_type = "SERIAL PRIMARY KEY"
            elif col == "datetime":
                sql_type = "DATE NOT NULL UNIQUE"
            elif "float" in str(dtype):
                sql_type = "NUMERIC"
            elif "int" in str(dtype):
                sql_type = "INTEGER"
            else:
                sql_type = "TEXT"
            columns_sql.append(f"{col} {sql_type}")
        
        create_sql = f"""
            CREATE TABLE {staging_table} (
                {', '.join(columns_sql)}
            );
        """
        cur.execute(create_sql)
        conn.commit()

        # buffer = StringIO()
        # df_features.to_csv(buffer, index=False, header=False)
        # buffer.seek(0)

        # cur.copy_expert(
        #     f"COPY {staging_table} FROM STDIN WITH CSV",
        #     buffer
        # )
        # conn.commit()

        tuples = [tuple(x) for x in transformed_data.to_numpy()]
        cols = ', '.join(list(transformed_data.columns))
        insertion_query = "INSERT INTO %s (%s) VALUES %%s" % (staging_table, cols)
        try:
            extras.execute_values(cur, insertion_query, tuples)
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

    @task 
    def load_data(staging_table):
        conn = psycopg2.connect(database="app_db", user="admin", password="admin", host="app-postgres", port="5432")
        final_table = "eur_usd_final"
        swap_query = f"""
            BEGIN;
            DROP TABLE IF EXISTS {final_table};
            ALTER TABLE {staging_table} RENAME TO {final_table};
            COMMIT;
        """
        cur = conn.cursor()
        cur.execute(swap_query)
        conn.commit()
        cur.close()
        conn.close()
        print("Final data added.")

    
    raw_table = extract_data()
    staging_table = transform_data(raw_table=raw_table)
    load_data(staging_table)

etl_pipeline = forex_etl_pipeline()