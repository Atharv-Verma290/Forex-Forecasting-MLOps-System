import numpy as np
import pandas as pd
import mlflow
import os
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

from src.model_factory import ModelFactory
from src.utility import cross_validate_model, get_predictors

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5080"))


class TrainingOrchestrator():
    """
    Coordinates the machine learning lifecycle from data sorting to model selection and MLflow registration.

    Args:
        df (pd.DataFrame): The raw training dataset from the database.

    Methods:
        run: Executes the initial screening of multiple candidate models via cross-validation.
        train_challenger: Re-trains the selected model with optimized parameters and logs it to MLflow.
        reorder_data: Normalizes the DataFrame index and removes database-specific columns.
        train_models: Benchmarks a list of model architectures using PR-AUC metrics.
        select_best: Identifies the top-performing model from the benchmark report.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df 

    def run(self):
        """
        Orchestrates the selection of the best base architecture.

        Returns:
            dict: Configuration of the top-performing model.
        """
        train_df = self.reorder_data(self.df)
        model_report = self.train_models(train_df)
        best_model = self.select_best(model_report)
        return best_model
    
    def train_challenger(self, model_data: dict):
        """
        Trains a final model using optimized hyperparameters and registers it as a 'challenger' in MLflow.

        This method takes the results of the tuning phase, builds a fresh model instance, 
        fits it on the training data, and logs the artifacts, parameters, and signatures 
        to the MLflow Model Registry.

        Args:
            model_data (dict): A dictionary containing model configurations, including:
                - 'name': The model architecture name.
                - 'model_type': The factory type string (e.g., 'xgboost_classifier').
                - 'best_hyperparameters': Dictionary of tuned parameters.
                - 'best_train_pr_auc_score': The optimization metric from the tuning phase.

        Returns:
            dict: Updated model_data containing the 'run_id' and the new 'model_version' 
                from the MLflow Registry.
        """
        train_df = self.reorder_data(self.df)

        predictors = get_predictors(train_df)
        X = train_df[predictors]
        y = train_df["target"]

        classifier = ModelFactory.get_model(model_data["name"], model_data["model_type"])
        classifier.build_model(model_data["best_hyperparameters"])
        classifier.fit(X, y)

        registered_model_name = "eur_usd_direction_model"
        signature = infer_signature(X, y)

        with mlflow.start_run(run_name="train_challenger") as run:
            mlflow.log_params(model_data["best_hyperparameters"])
            mlflow.log_param("model_type", model_data["model_type"])
            mlflow.log_metric("train_pr_auc_score", model_data["best_train_pr_auc_score"])

            classifier.log_model(signature=signature, registered_model_name=registered_model_name, input_example=X.head())

            run_id = run.info.run_id

        client = MlflowClient()
        versions = client.get_latest_versions(registered_model_name, stages=["None"])
        challenger_version = max(v.version for v in versions)

        client.set_model_version_tag(
            name=registered_model_name,
            version=challenger_version,
            key="candidate",
            value="challenger"
        )

        model_data["name"] = registered_model_name
        model_data["model_version"] = challenger_version,
        model_data["run_id"] = run_id

        return model_data

    def reorder_data(self, df: pd.DataFrame):
        """
        Standardizes the training DataFrame by setting the time index and removing database IDs.

        Args:
            df (pd.DataFrame): The raw training data retrieved from the SQL database.

        Returns:
            pd.DataFrame: A cleaned DataFrame sorted chronologically and ready for 
                        feature/target extraction.
        """
        indexed_df = df.set_index("datetime")
        sorted_df = indexed_df.sort_index(ascending=True)
        ordered_df = sorted_df.drop(columns=["id"])
        return ordered_df

    def train_models(self, df):
        """
        Benchmarks multiple model architectures using cross-validation.

        This method iterates through a predefined list of candidate models, 
        initializes them via the ModelFactory, and evaluates their performance 
        using the Area Under the Precision-Recall Curve (PR-AUC).

        Args:
            df (pd.DataFrame): The training dataset containing features and the 'target' column.

        Returns:
            list: A list of dictionaries, each enriched with the mean cross-validation score.
        """
        model_list = [
            {
                'name': 'random forest model',
                'model_type': 'random_forest'
            },
            {
                'name': 'logistic regression',
                'model_type': 'logistic_regression'
            },
            {
                'name': 'xgboost model',
                'model_type': 'xgboost_classifier'
            }
        ]

        predictors = get_predictors(df)
        X = df[predictors]
        y = df["target"]

        for model in model_list:
            classifier = ModelFactory.get_model(**model)
            classifier.build_model()

            avg_score, std_score = cross_validate_model(classifier, X, y)

            print(f"Avg CV PR-AUC Score for {model["name"]}: {avg_score:.4f} ± {std_score:.4f}")
            model["cv_pr_auc_score"] = round(float(avg_score), 4)

        return model_list

    def select_best(self, model_report: dict):
        """
        Identifies the top-performing model architecture based on cross-validation scores.

        Args:
            model_report (list): A list of dictionaries containing model configurations 
                                and their corresponding 'cv_pr_auc_score'.

        Returns:
            dict: The configuration dictionary of the model with the highest PR-AUC.
        """
        best_model = model_report[0]
        for model in model_report:
            if model["cv_pr_auc_score"] > best_model["cv_pr_auc_score"]:
                best_model = model 
        return best_model


if __name__ == "__main__":
    df = pd.read_csv("train_sample.csv")
    df["id"] = 100
    
    orchestrator = TrainingOrchestrator(df=df)
    results = orchestrator.run()

    print("Output results: ")
    print(results)