import numpy as np
import pandas as pd

from model_factory import ModelFactory
from utility import cross_validate_model


class TrainingOrchestrator():
    def __init__(self, df):
        self.df = df 

    def run(self):
        train_df = self.reorder_data(self.df)
        model_report = self.train_models(train_df)
        best_model = self.select_best(model_report)
        return best_model
    
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