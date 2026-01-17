import pandas as pd
import mlflow
import os
from mlflow.tracking import MlflowClient
from mlflow.data import from_pandas
from sklearn.metrics import precision_score

from utility import evaluate

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5080"))

class ModelPromotionManager:
    def __init__(self, model_name: str, test_df: pd.DataFrame, metric_fn=precision_score):
        self.model_name = model_name
        self.test_df = test_df
        self.metric_fn = metric_fn
        self.client = MlflowClient()

    def promote_if_better(self):
        X_test, y_test = self.prepare_test_data(self.test_df)
        challenger = self.load_challenger_model()
        production = self.load_production_model()

        with mlflow.start_run(run_name="model_promotion_evaluation"):
            challenger_score = evaluate(challenger["model"], X_test, y_test)
            mlflow.log_metric("challenger_test_precision_score", challenger_score)

            test_df = X_test.copy()
            test_df["target"] = y_test 

            test_dataset = from_pandas(
                test_df,
                source="eur_usd_test"
            )
            mlflow.log_input(
                test_dataset,
                context="testing"
            )

            if production:
                production_score = evaluate(production["model"], X_test, y_test)
                mlflow.log_metric("production_test_precision_score", production_score)
            else:
                production_score = None 
                mlflow.log_param("production_model_exists", False)

            mlflow.log_param("challenger_version", challenger["model_version"])
            if production:
                mlflow.log_param("production_version", production["model_version"])

            print(f"Challenger score: {challenger_score}")
            print(f"Production score: {production_score}")

            decision = (
                "promote_challenger" 
                if production_score is None or challenger_score > production_score
                else "retain_production"
            )
            mlflow.log_param("promotion_decision", decision)

        if decision == "promote_challenger":
            self.promote_challenger(challenger, production)
            return f"challenger promoted '{challenger["model_name"]}'"
        else:
            self.archive_challenger(challenger)
            return "production model retained"    

    def prepare_test_data(self, df: pd.DataFrame):
        indexed_df = df.set_index("datetime")
        sorted_df = indexed_df.sort_index(ascending=True)
        ordered_df = sorted_df.drop(columns=["id"])
        X = ordered_df.drop(columns=["tomorrow", "target"])
        y = ordered_df["target"]
        return X, y

    def load_model_by_version(self, version: str):
        model_uri = f"models:/{self.model_name}/{version}"
        return mlflow.pyfunc.load_model(model_uri)
    
    def load_challenger_model(self):
        versions = self.client.search_model_versions(f"name='{self.model_name}'")

        challengers = [
            v for v in versions
            if v.current_stage == "None"
            and v.tags.get("candidate") == "challenger"
        ]
        if not challengers:
            raise RuntimeError("No challenger model found")
        
        challenger = max(challengers, key=lambda v: int(v.version))

        model = self.load_model_by_version(challenger.version)

        return {
            "model": model,
            "model_name": self.model_name,
            "model_version": challenger.version,
            "run_id": challenger.run_id
        }

    def load_production_model(self):
        prod_versions = self.client.get_latest_versions(name=self.model_name, stages=["Production"])

        if not prod_versions:
            print(f"[INFO] No Production model found for '{self.model_name}'.")
            return None
        
        production = prod_versions[0]

        model = self.load_model_by_version(production.version)
        return {
            "model": model,
            "model_name": self.model_name,
            "model_version": production.version,
            "run_id": production.run_id
        } 
    
    def promote_challenger(self, challenger, production):
        if production:
            self.client.transition_model_version_stage(
                name=production["model_name"],
                version=production["model_version"],
                stage="Archived"
            )

        self.client.transition_model_version_stage(
            name=challenger["model_name"],
            version=challenger["model_version"],
            stage="Production"
        )

        self.client.set_model_version_tag(
            name=challenger["model_name"],
            version=challenger["model_version"],
            key="candidate",
            value="champion"
        )
        print(f"Promoted version {challenger["model_version"]} to Production")

    def archive_challenger(self, challenger):
        self.client.transition_model_version_stage(
            name=challenger["model_name"],
            version=challenger["model_version"],
            stage="Archived"
        )
        print(f"Archived challenger version {challenger["model_version"]}")

