from datetime import datetime, timedelta, timezone
from airflow.sdk import dag, task 
from dotenv import load_dotenv
from data_preprocessing import ForexDataPreProcessing #type: ignore
from utility import SQLTableBuilder, TrainTestTableStrategy #type: ignore
from orchestrator import TrainingOrchestrator #type: ignore
from model_tuning import OptunaModelTuner #type: ignore
from model_promotion import ModelPromotionManager #type: ignore
import psycopg2
import psycopg2.extras as extras
import pandas as pd

load_dotenv()


default_args = {
    'owner': 'atharv',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

@dag(
        dag_id="forex_training_pipeline",
        default_args=default_args,
        start_date=datetime(2026, 1, 12)
    )
def forex_prediction_pipeline():

    @task(multiple_outputs=True)
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

            return {"train_dataset": "eur_usd_train", "test_dataset": "eur_usd_test"}

        except Exception as e:
            print(e)

    @task(multiple_outputs=True)
    def model_training(datasets):
        train_data = datasets["train_dataset"]
        conn = psycopg2.connect(database="app_db", user="admin", password="admin", host="app-postgres", port="5432")
        print("Database connected successfully")
        query = f"SELECT * FROM {train_data} ORDER BY datetime DESC;"
        train_df = pd.read_sql(query, conn)
        
        orchestrator = TrainingOrchestrator(train_df)
        best_model_dict = orchestrator.run()
        print(best_model_dict)
        return best_model_dict
        
    @task(multiple_outputs=True)
    def hyperparameter_tuning(datasets, best_model_data):
        train_data = datasets["train_dataset"]
        conn = psycopg2.connect(database="app_db", user="admin", password="admin", host="app-postgres", port="5432")
        print("Database connected successfully")
        query = f"SELECT * FROM {train_data} ORDER BY datetime DESC;"
        train_df = pd.read_sql(query, conn)

        tuner = OptunaModelTuner(train_df, best_model_data)
        tuned_model_data = tuner.start_tuning()
        print("Tuned model report: ")
        print(tuned_model_data)
        return tuned_model_data
    
    @task(multiple_outputs=True)
    def train_challenger(datasets, tuned_model_data):
        train_data = datasets["train_dataset"]
        conn = psycopg2.connect(database="app_db", user="admin", password="admin", host="app-postgres", port="5432")
        print("Database connected successfully")
        query = f"SELECT * FROM {train_data} ORDER BY datetime DESC;"
        train_df = pd.read_sql(query, conn)

        orchestrator = TrainingOrchestrator(train_df)
        challenger_data = orchestrator.train_challenger(tuned_model_data)
        
        print(challenger_data)
        return challenger_data

    @task 
    def model_promotion(datasets, challenger_data):
        test_data = datasets["test_dataset"]
        conn = psycopg2.connect(database="app_db", user="admin", password="admin", host="app-postgres", port="5432")
        print("Database connected successfully")
        query = f"SELECT * FROM {test_data} ORDER BY datetime DESC;"
        test_df = pd.read_sql(query, conn)
        
        manager = ModelPromotionManager(model_name=challenger_data["name"], test_df=test_df)
        result = manager.promote_if_better()
        print(result)


    datasets = data_preprocessing()
    best_model_data = model_training(datasets)
    tuned_model_data = hyperparameter_tuning(datasets, best_model_data)
    challenger_data = train_challenger(datasets, tuned_model_data)
    model_promotion(datasets, challenger_data)
    
prediction_pipeline = forex_prediction_pipeline()

