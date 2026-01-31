from datetime import datetime, timedelta
from airflow.sdk import dag, task 
from dotenv import load_dotenv
from src.data_preprocessing import ForexDataPreProcessing 
from src.utility import SQLTableBuilder, TrainTestTableStrategy 
from src.orchestrator import TrainingOrchestrator 
from src.model_tuning import OptunaModelTuner 
from src.model_promotion import ModelPromotionManager 
import psycopg2.extras as extras
import pandas as pd
from sqlalchemy import create_engine, text 

load_dotenv()

CONNECTION_URL = "postgresql://admin:admin@app-postgres:5432/app_db"

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
    """
    Model training pipeline dag. Trains new models and compare with production model and promote the best model.
    """
    @task(multiple_outputs=True)
    def data_preprocessing():
        """
        Orchestrates the preparation and storage of training and testing datasets.

        Args:
            None

        Returns:
            dict: A mapping of the created table names (e.g., {'train_dataset': 'eur_usd_train', ...}) 
                or None if an error occurs.
        """
        try:
            engine = create_engine(CONNECTION_URL)
            print("Database connected successfully")
            query = """SELECT * FROM eur_usd_final ORDER BY datetime DESC;"""
            input_df = pd.read_sql(query, engine)
            print("Input data retrieved: ", input_df)

            preprocessor = ForexDataPreProcessing(input_df)
            train_df, test_df = preprocessor.preprocess_data()

            # create train table
            table_builder = SQLTableBuilder(TrainTestTableStrategy(tablename="eur_usd_train"))
            creation_query = table_builder.get_create_query(df=train_df)
            with engine.connect() as conn:
                conn.execute(text(creation_query))
                conn.commit()
            insertion_query, values = table_builder.get_insert_query(data=train_df)
            raw_conn = engine.raw_connection()
            cur = raw_conn.cursor()
            extras.execute_values(cur, insertion_query, values)
            raw_conn.commit()
            raw_conn.close()

            # create test table
            table_builder.set_strategy(TrainTestTableStrategy(tablename="eur_usd_test"))
            creation_query = table_builder.get_create_query(df=test_df)
            with engine.connect() as conn:
                conn.execute(text(creation_query))
                conn.commit()
            insertion_query, values = table_builder.get_insert_query(data=test_df)
            raw_conn = engine.raw_connection()
            cur = raw_conn.cursor()
            extras.execute_values(cur, insertion_query, values)
            raw_conn.commit()
            raw_conn.close()

            return {"train_dataset": "eur_usd_train", "test_dataset": "eur_usd_test"}

        except Exception as e:
            print(e)

    @task(multiple_outputs=True)
    def model_training(datasets):
        """
        Loads training data from the database and executes the model training pipeline.

        Args:
            datasets (dict): A dictionary containing the 'train_dataset' table name.

        Returns:
            dict: A dictionary containing the best model's performance metrics, 
                parameters, and metadata.
        """
        train_data = datasets["train_dataset"]
        engine = create_engine(CONNECTION_URL)
        print("Database connected successfully")
        query = f"SELECT * FROM {train_data} ORDER BY datetime DESC;"
        train_df = pd.read_sql(query, engine)

        orchestrator = TrainingOrchestrator(train_df)
        best_model_dict = orchestrator.run()
        print(best_model_dict)
        return best_model_dict
        
    @task(multiple_outputs=True)
    def hyperparameter_tuning(datasets, best_model_data):
        """
        Performs hyperparameter optimization on the selected model.

        Args:
            datasets (dict): Dictionary containing the 'train_dataset' table name.
            best_model_data (dict): Metadata and configuration of the model selected 
                                    for tuning.

        Returns:
            dict: A report containing the optimized hyperparameters and updated 
                model metrics.
        """
        train_data = datasets["train_dataset"]
        engine = create_engine(CONNECTION_URL)
        print("Database connected successfully")
        query = f"SELECT * FROM {train_data} ORDER BY datetime DESC;"
        train_df = pd.read_sql(query, engine)

        tuner = OptunaModelTuner(train_df, best_model_data)
        tuned_model_data = tuner.start_tuning()
        print("Tuned model report: ")
        print(tuned_model_data)
        return tuned_model_data
    
    @task(multiple_outputs=True)
    def train_challenger(datasets, tuned_model_data):
        """
        Trains a final 'challenger' model using the optimized hyperparameters.

        Args:
            datasets (dict): Dictionary containing the 'train_dataset' table name.
            tuned_model_data (dict): The optimized hyperparameter configuration 
                                    retrieved from the tuning phase.

        Returns:
            dict: A dictionary containing the challenger model's path, 
                performance metrics, and metadata.
        """
        train_data = datasets["train_dataset"]
        engine = create_engine(CONNECTION_URL)
        query = f"SELECT * FROM {train_data} ORDER BY datetime DESC;"
        train_df = pd.read_sql(query, engine)

        orchestrator = TrainingOrchestrator(train_df)
        challenger_data = orchestrator.train_challenger(tuned_model_data)
        
        print(challenger_data)
        return challenger_data
    
    @task 
    def model_promotion(datasets, challenger_data):
        """
        Evaluates the challenger model against the current champion and promotes if superior.

        Args:
            datasets (dict): Dictionary containing the 'test_dataset' table name.
            challenger_data (dict): Metadata and performance metrics for the candidate model.

        Returns:
            None: Updates the model registry or production alias via ModelPromotionManager.
        """
        test_data = datasets["test_dataset"]
        engine = create_engine(CONNECTION_URL)
        print("Database connected successfully")
        query = f"SELECT * FROM {test_data} ORDER BY datetime DESC;"
        test_df = pd.read_sql(query, engine)
        
        manager = ModelPromotionManager(model_name=challenger_data["name"], test_df=test_df)
        result = manager.promote_if_better()
        print(result)


    datasets = data_preprocessing()
    best_model_data = model_training(datasets)
    tuned_model_data = hyperparameter_tuning(datasets, best_model_data)
    challenger_data = train_challenger(datasets, tuned_model_data)
    model_promotion(datasets, challenger_data)
    
prediction_pipeline = forex_prediction_pipeline()

