from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import optuna
from src.model_factory import ModelFactory
from src.utility import suggest_params, cross_validate_model, get_predictors

class ModelTunerTemplate(ABC):
    """
    An abstract base class providing a standardized workflow for hyperparameter optimization.

    This template ensures that data preparation consistently precedes the tuning process,
    facilitating reproducible experiments across different model architectures.

    Args:
        df (pd.DataFrame): The training dataset.
        model_data (dict): Metadata and configuration for the model to be tuned.

    Methods:
        start_tuning: Orchestrates the data preparation and optimization loop.
        prepare_data: Abstract method to define feature/target extraction logic.
        tune_model: Abstract method to define the specific optimization strategy (e.g., Optuna, GridSearch).
    """
    def __init__(self, df: pd.DataFrame, model_data: dict):
        self.df = df
        self.model_data = model_data

    def start_tuning(self):
        """
        Executes the tuning pipeline.

        Returns:
            dict: A report containing optimized parameters and resulting performance metrics.
        """
        X, y = self.prepare_data(self.df)
        output_report = self.tune_model(X, y, self.model_data)
        return output_report

    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separates the input DataFrame into features (X) and target (y).
        """
        pass 

    @abstractmethod
    def tune_model(self, X: pd.DataFrame, y: pd.DataFrame, model_data: dict) -> dict:
        """
        Contains the specific logic for hyperparameter search.
        """
        pass


class OptunaModelTuner(ModelTunerTemplate):
    """
    Automates hyperparameter optimization using the Tree-structured Parzen Estimator (TPE) algorithm.

    This class leverages Optuna to explore the model's defined parameter space, 
    evaluating candidate configurations through time-series cross-validation 
    to maximize the PR-AUC score.

    Methods:
        prepare_data: Cleans, sorts, and extracts features/targets from the raw DataFrame.
        tune_model: Defines the objective function and orchestrates the Optuna study.
    """
    def prepare_data(self, df):
        """
        Prepares the training set for the tuning process.

        Args:
            df (pd.DataFrame): Raw input data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features (X) and target (y).
        """
        clean_df = df.set_index("datetime").sort_index(ascending=True)
        predictors = get_predictors(clean_df)
        X = clean_df[predictors]
        y = clean_df["target"]
        return X, y
    
    def tune_model(self, X, y, model_data):
        """
        Runs an Optuna study to find the optimal hyperparameters.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training labels.
            model_data (dict): Dictionary containing model type and metadata.

        Returns:
            dict: The model_data dictionary updated with 'best_hyperparameters' 
                  and 'best_train_pr_auc_score'.
        """
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
