from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import mlflow.sklearn


class Model(ABC):
    """
    Standard interface for all model implementations. 
    Ensures consistent interaction between the model factory, 
    tuning orchestrators, and the MLflow registry.

    Attributes:
        name (str): Unique model identifier.
        model_type (str): Task type (e.g., classification).
        model (object): The internal machine learning estimator.

    Methods:
        build_model: Configures the estimator with specific hyperparameters.
        fit: Trains the estimator on input features and labels.
        predict: Outputs discrete class predictions.
        predict_proba: Outputs class probability estimates.
        log_model: Handles MLflow serialization and versioning.
    """
    def __init__(self, name, model_type):
        self.name = name
        self.model_type = model_type
        self.model = None

    @abstractmethod
    def build_model(self, params: dict | None = None):
        """Instantiates the estimator with provided or default parameters."""
        pass 

    @abstractmethod
    def fit(self, X, y):
        """Trains the model on the provided dataset."""
        pass 

    @abstractmethod
    def predict(self, X):
        """Predicts the target labels for the input features."""
        pass 

    @abstractmethod
    def predict_proba(self, X):
        """Predicts class probabilities, required for scoring metrics like PR-AUC."""
        pass

    @abstractmethod
    def log_model(self, signature, registered_model_name: str, input_example):
        """Saves the model and its metadata to the MLflow Model Registry."""
        pass



class RandomForestModel(Model):
    """
    Random Forest implementation of the Model interface.

    Uses an ensemble of decision trees to provide robust classification, 
    particularly effective for non-linear relationships in financial data.

    Methods:
        build_model(params): Instantiates the classifier.
        fit(X, y): Trains the ensemble.
        predict(X): Returns class labels.
        predict_proba(X): Returns probability estimates.
        log_model(signature, name, example): Saves to MLflow.
    """
    param_space = {
        'n_estimators': ('int', 50, 200),
        'max_depth': ('int', 5, 15),
        'min_samples_split': ('int', 2, 5)
    }

    default_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_split': 2
    }

    def build_model(self, params: dict | None = None):
        """
        Instantiates the RandomForestClassifier.

        Args:
            params (dict, optional): Hyperparameters. Defaults to class default_params.
        """
        params = params or self.default_params
        self.model = RandomForestClassifier(**params)
    
    def fit(self, X, y):
        """
        Trains the model.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Target labels.
        """
        self.model.fit(X, y)
    
    def predict(self, X):
        """
        Predicts discrete class labels.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Predicted 0 or 1 labels.
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predicts class probabilities.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Probability array (n_samples, n_classes).
        """
        return self.model.predict_proba(X)
    
    def log_model(self, signature, registered_model_name, input_example):
        """
        Logs and registers the model to MLflow.

        Args:
            signature (ModelSignature): MLflow model schema.
            registered_model_name (str): Name for the Model Registry.
            input_example (pd.DataFrame): Data sample for schema validation.
        """
        mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name
        )
    


class LogisticRegressionModel(Model):
    """
    Logistic Regression implementation of the Model interface.

    A linear model for classification that provides a baseline for predictive
    performance and model interpretability.

    Methods:
        build_model(params): Instantiates the LogisticRegression estimator.
        fit(X, y): Trains the linear classifier on the dataset.
        predict(X): Returns binary class predictions.
        predict_proba(X): Returns probability scores for class membership.
        log_model(signature, name, example): Saves the model to MLflow.
    """
    param_space = {
        'C': ('float', 1e-4, 10.0),       
        'max_iter': ('int', 100, 500)   
    }
    
    def build_model(self, params: dict | None = None):
        """
        Instantiates the LogisticRegression estimator.

        Args:
            params (dict, optional): Hyperparameters (e.g., 'C' for regularization).
        """
        if params:
            self.model = LogisticRegression(**params)
        else:
            self.model = LogisticRegression()

    def fit(self, X, y):
        """
        Trains the linear model.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Target labels.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predicts binary class labels.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Predicted 0 or 1 labels.
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predicts class probabilities.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Probability array (n_samples, n_classes).
        """
        return self.model.predict_proba(X)
    
    def log_model(self, signature, registered_model_name, input_example):
        """
        Logs and registers the model to MLflow.

        Args:
            signature (ModelSignature): MLflow model schema.
            registered_model_name (str): Name for the Model Registry.
            input_example (pd.DataFrame): Data sample for schema validation.
        """
        mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name
        )


class XGBoostModel(Model):
    """
    XGBoost implementation of the Model interface.

    An optimized gradient boosting library designed for high efficiency and 
    computational speed. It incorporates L1/L2 regularization to prevent 
    overfitting and handles missing values natively.

    Methods:
        build_model(params): Configures the XGBClassifier.
        fit(X, y): Sequentially trains trees to minimize loss.
        predict(X): Returns binary labels (0 or 1).
        predict_proba(X): Returns class probabilities.
        log_model(signature, name, example): Saves the model using the XGBoost flavor.
    """
    param_space = {
        'n_estimators': ('int', 50, 300),         
        'max_depth': ('int', 3, 6),               
        'learning_rate': ('float', 0.005, 0.1),    
        'subsample': ('float', 0.5, 1.0),         
        'colsample_bytree': ('float', 0.5, 1.0),   
        'gamma': ('float', 0.0, 0.5),                  
        'min_child_weight': ('int', 1, 10),
        'reg_alpha': ('float', 0.0, 0.1),
        'reg_lambda': ('float', 0.0, 2.0)
    }

    default_params = {
        "n_estimators": 200,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }

    def build_model(self, params: dict | None = None):
        """
        Instantiates the XGBClassifier.

        Args:
            params (dict, optional): Hyperparameters. Defaults to class default_params.
        """
        params = params or self.default_params
        self.model = xgb.XGBClassifier(**params)

    def fit(self, X, y):
        """
        Trains the XGBoost model.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Target labels.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predicts binary class labels.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Predicted 0 or 1 labels.
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predicts class probabilities.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Probability array (n_samples, n_classes).
        """
        return self.model.predict_proba(X)
    
    def log_model(self, signature, registered_model_name, input_example):
        """
        Logs and registers the model to MLflow.

        Args:
            signature (ModelSignature): MLflow model schema.
            registered_model_name (str): Name for the Model Registry.
            input_example (pd.DataFrame): Data sample for schema validation.
        """
        mlflow.xgboost.log_model(
            xgb_model=self.model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name
        )
        

class ModelFactory:
    """
    Creational factory for instantiating model wrappers.

    Centralizes object creation logic to provide a consistent interface 
    regardless of the underlying machine learning library.

    Methods:
        get_model(name, model_type): Returns a concrete Model instance.
    """
    @staticmethod
    def get_model(name: str, model_type: str):
        """
        Factory method to retrieve a specific model implementation.

        Args:
            name (str): The identifier to assign to the model instance.
            model_type (str): The type of model to create (e.g., 'random_forest', 
                             'logistic_regression', 'xgboost_classifier').

        Returns:
            Model: A concrete instance of a class inheriting from the Model ABC.
            None: If the requested model_type is unsupported.
        """
        if model_type == "random_forest":
            model_instance = RandomForestModel(name=name, model_type=model_type)
            
        elif model_type == "logistic_regression":
            model_instance = LogisticRegressionModel(name=name, model_type=model_type)

        elif model_type == "xgboost_classifier":
            model_instance = XGBoostModel(name=name, model_type=model_type)

        else:
            print(f"Unsupported model_type: {model_type}")
            return 

        return model_instance