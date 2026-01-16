import numpy as np
import pandas as pd
import mlflow
import os
from mlflow.tracking import MlflowClient

from model_factory import ModelFactory
from utility import cross_validate_model

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5080"))


class TrainingOrchestrator():
    def __init__(self, df: pd.DataFrame):
        self.df = df 

    def run(self):
        train_df = self.reorder_data(self.df)
        model_report = self.train_models(train_df)
        best_model = self.select_best(model_report)
        return best_model
    
    def train_challenger(self, model_data: dict):
        train_df = self.reorder_data(self.df)

        predictors = train_df.columns.drop(["target", "tomorrow"])
        X = train_df[predictors]
        y = train_df["target"]

        classifier = ModelFactory.get_model(model_data["name"], model_data["model_type"])
        classifier.build_model(model_data["best_hyperparameters"])
        classifier.fit(X, y)

        registered_model_name = "eur_usd_direction_model"

        with mlflow.start_run(run_name="train_challenger") as run:
            mlflow.log_params(model_data["best_hyperparameters"])
            mlflow.log_param("model_type", model_data["model_type"])
            mlflow.log_metric("train_precision_score", model_data["best_train_precision_score"])

            classifier.log_model(registered_model_name)

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
        indexed_df = df.set_index("datetime")
        sorted_df = indexed_df.sort_index(ascending=True)
        ordered_df = sorted_df.drop(columns=["id"])
        return ordered_df

    def train_models(self, df):
        model_list = [
            {
                'name': 'random forest model',
                'model_type': 'random_forest'
            },
            {
                'name': 'logistic regression',
                'model_type': 'logistic_regression'
            }
        ]

        predictors = df.columns.drop(["target", "tomorrow"])
        X = df[predictors]
        y = df["target"]

        for model in model_list:
            classifier = ModelFactory.get_model(**model)
            classifier.build_model()

            avg_score, std_score = cross_validate_model(classifier, X, y)

            print(f"Avg CV Precision Score for {model["name"]}: {avg_score:.4f} ± {std_score:.4f}")
            model["cv_precision_score"] = round(float(avg_score), 4)

        return model_list

    def select_best(self, model_report: dict):
        best_model = model_report[0]
        for model in model_report:
            if model["cv_precision_score"] > best_model["cv_precision_score"]:
                best_model = model 
        return best_model


if __name__ == "__main__":
    df = pd.read_csv("train_sample.csv")
    df["id"] = 100
    
    orchestrator = TrainingOrchestrator(df=df)
    results = orchestrator.run()

    print("Output results: ")
    print(results)