from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class Model(ABC):
    def __init__(self, name, model_type):
        self.name = name
        self.model_type = model_type
        self.model = None

    @abstractmethod
    def build_model(self):
        pass 

    @abstractmethod
    def fit(self, X, y):
        pass 

    @abstractmethod
    def predict(self, X):
        pass 

class RandomForestModel(Model):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5]
    }

    def build_model(self):
        self.model = RandomForestClassifier()
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    

class LogisticRegressionModel(Model):
    def build_model(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class ModelFactory:
    @staticmethod
    def get_model(name: str, model_type: str):
        if model_type == "random_forest":
            model_instance = RandomForestModel(name=name, model_type=model_type)
            
        elif model_type == "logistic_regression":
            model_instance = LogisticRegressionModel(name=name, model_type=model_type)

        else:
            print(f"Unsupported model_type: {model_type}")
            return 

        model_instance.build_model()
        return model_instance