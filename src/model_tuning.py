from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import optuna
from model_factory import ModelFactory
from utility import suggest_params, cross_validate_model

class ModelTunerTemplate(ABC):
    def __init__(self, df: pd.DataFrame, model_data: dict):
        self.df = df
        self.model_data = model_data

    def start_tuning(self):
        X, y = self.prepare_data(self.df)
        output_report = self.tune_model(X, y, self.model_data)
        return output_report

    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass 

    @abstractmethod
    def tune_model(self, X: pd.DataFrame, y: pd.DataFrame, model_data: dict) -> dict:
        pass


class OptunaModelTuner(ModelTunerTemplate):
    def prepare_data(self, df):
        clean_df = df.set_index("datetime").sort_index(ascending=True)
        X = clean_df.drop(columns=["id", "tomorrow", "target"])
        y = clean_df["target"]
        return X, y
    
    def tune_model(self, X, y, model_data):
        model = ModelFactory.get_model(name=model_data["name"], model_type=model_data["model_type"])
        param_space = model.param_space

        def objective_function(trial):
            params = suggest_params(trial, param_space)
            model.build_model(params)

            mean_score, std_score = cross_validate_model(model, X, y)
            return mean_score
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective_function, n_trials=50)

        model_data["best_train_pr_auc_score"] = study.best_trial.value
        model_data["best_hyperparameters"] = study.best_trial.params

        return model_data

   
if __name__=="__main__":
    df = pd.read_csv("train_sample.csv")
    df["id"] = 100
    model_data = {"name": "random forest model", 
                  "model_type": "random_forest", 
                  "cv_pr_auc_score": 0.506}

    tuner = OptunaModelTuner(df, model_data)
    tuned_model_data = tuner.start_tuning()
    print(tuned_model_data)
