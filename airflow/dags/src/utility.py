from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

def next_forex_trading_day(feature_date_utc: datetime) -> datetime.date:
    """
    Calculates the next valid forex trading date from a given timestamp.

    Args:
        feature_date_utc (datetime): The UTC datetime of the features/observation.

    Returns:
        datetime.date: The date of the next trading session (skips Sundays).
    """
    weekday = feature_date_utc.weekday() # Monday=0, Sunday=6

    # If Saturday, jump 2 days to Monday. 
    # Otherwise, just move to the next calendar day.
    days_to_add = 2 if weekday == 5 else 1
    
    return (feature_date_utc + timedelta(days=days_to_add)).date()


def cross_validate_model(classifier, X, y):
    """
    Performs Time-Series Cross-Validation to evaluate model stability over time.

    Using a walk-forward approach, this function splits the data into sequential 
    folds, ensuring no future data is used to predict the past. It calculates 
    the Precision-Recall AUC for each fold.

    Args:
        classifier: A model instance from the ModelFactory with a fit/predict_proba interface.
        X (pd.DataFrame): Feature set.
        y (pd.Series): Binary target labels.

    Returns:
        Tuple[float, float]: The mean and standard deviation of the PR-AUC scores.
    """
    tscv = TimeSeriesSplit(
        n_splits=5,
        test_size=250,
        gap=1
        )
    pr_auc_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        classifier.fit(X=X_tr, y=y_tr)
        probs = classifier.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, probs)
        pr_auc_scores.append(pr_auc)

    return np.mean(pr_auc_scores), np.std(pr_auc_scores)


def suggest_params(trial, param_space: dict):
    """
    Maps a static parameter configuration to Optuna trial suggestion methods.

    This helper bridges the gap between a declarative dictionary (the search space) 
    and the Optuna Trial object, allowing for dynamic hyperparameter sampling.

    Args:
        trial (optuna.trial.Trial): The current Optuna trial object.
        param_space (dict): A dictionary mapping parameter names to specifications 
                            (e.g., {'n_estimators': ['int', 10, 100]}).

    Returns:
        dict: A dictionary of sampled hyperparameters for the current trial.
    """
    params = {}
    for name, spec in param_space.items():
        kind = spec[0]

        if kind == 'int':
            _, low, high = spec
            params[name] = trial.suggest_int(name, low, high)

        elif kind == "float":
            _, low, high = spec
            params[name] = trial.suggest_float(name, low, high)
        
        elif kind == "categorical":
            _, choices = spec
            params[name] = trial.suggest_categorical(name, choices)

        else:
            raise ValueError(f"Unknown param type: {kind}")
    
    return params


def get_predictors(df: pd.DataFrame):
    """
    Identifies the feature set (X) by excluding target variables and raw price data.

    This function implements a "blacklist" approach to feature selection, 
    automatically removing high-cardinality raw prices and target-leaking columns 
    to ensure the model only trains on engineered signals.

    Args:
        df (pd.DataFrame): The preprocessed dataset.

    Returns:
        list: A list of column names to be used as model inputs.
    """
    vol_cols = [c for c in df.columns if 'vol_absmean_' in c]
    hl_mean_cols = [c for c in df.columns if 'hl_range_mean_' in c]
    cols_to_drop = [
        "open", "high", "low", "close", 
        "tomorrow", "target", 
        "log_return", "abs_log_return", 
        "hl_range", "large_move"
    ] + vol_cols + hl_mean_cols
    predictors = df.columns.drop(cols_to_drop, errors='ignore')
    return list(predictors)


def evaluate(model, X_test, y_test):
    """
    Computes the Precision-Recall AUC score for a model on the test dataset.

    This function handles both classifier instances that return probability 
    estimates and those that might return multi-column arrays, ensuring the 
    positive class probability is isolated for scoring.

    Args:
        model: The trained estimator (must support .predict() or .predict_proba()).
        X_test (pd.DataFrame): The feature matrix for testing.
        y_test (pd.Series): The true binary labels.

    Returns:
        float: The Average Precision (PR-AUC) score.
    """
    probs = model.predict(X_test)
    if len(probs.shape) > 1 and probs.shape[1] > 1:
        probs = probs[:, 1]
    return average_precision_score(y_test, probs)

########################
# SQL Query generators
########################
class TableStrategy(ABC):
    @abstractmethod
    def generate_create_query(self, df: pd.DataFrame = None) -> str:
        pass
    
    @abstractmethod
    def generate_insert_query(self, data: Any) -> Tuple[str, List[Tuple]]:
        pass

    def _format_data(self, data: Union[pd.DataFrame, Dict, List, Tuple]) -> Tuple[str, List[Tuple]]:
        # Case 1: Pandas DataFrame (Staging / TrainTest)
        if isinstance(data, pd.DataFrame):
            cols = data.columns.tolist()
            rows = [tuple(x) for x in data.to_numpy()]

        # Case 2: Tuple or List of dicts (Raw)
        elif isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], dict):
            cols = list(data[0].keys())
            rows = [tuple(d.values()) for d in data]

        # Case 3: Single Dictionary (Prediction)
        elif isinstance(data, dict):
            cols = list(data.keys())
            rows = [tuple(data.values())]

        else:
            ValueError(f"Unsupported data format: {type(data)}")

        return cols, rows


class RawTableStrategy(TableStrategy):
    """
    Strategy for managing raw forex data tables with UPSERT logic.

    Args:
        tablename (str): The name of the raw data table.

    Methods:
        generate_create_query(df): Returns a static SQL CREATE TABLE statement.
        generate_insert_query(data): Returns an INSERT query with ON CONFLICT resolution.
    """
    def __init__(self, tablename: str):
        self.tablename = tablename 

    def generate_create_query(self, df: pd.DataFrame = None) -> str:
        """
        Generates the SQL statement for the raw data schema.

        Args:
            df (pd.DataFrame, optional): Unused; schema is predefined.

        Returns:
            str: SQL CREATE TABLE IF NOT EXISTS statement.
        """
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.tablename}(
            id SERIAL PRIMARY KEY,
            datetime DATE NOT NULL UNIQUE,
            open NUMERIC,
            high NUMERIC,
            low NUMERIC,
            close NUMERIC
        );
        """
        return query
    
    def generate_insert_query(self, data: Any) -> Tuple[str, List[Tuple]]:
        """
        Generates a parameterized UPSERT query for raw data.

        Args:
            data (Any): The raw data to be formatted and inserted.

        Returns:
            Tuple[str, List[Tuple]]: SQL INSERT string and the formatted values.
        """
        cols, vals = self._format_data(data)
        query = f"""
        INSERT INTO {self.tablename} ({', '.join(cols)}) 
        VALUES %s
        ON CONFLICT (datetime) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close;
        """
        return query, vals


class StagingTableStrategy(TableStrategy):
    """
    Strategy for creating temporary staging tables based on DataFrame schema.

    Args:
        tablename (str): The name of the staging table.

    Methods:
        generate_create_query(df): Dynamically generates a DROP and CREATE statement.
        generate_insert_query(data): Returns a standard bulk INSERT statement.
    """
    def __init__(self, tablename: str):
        self.tablename = tablename

    def generate_create_query(self, df: pd.DataFrame = None) -> str:
        """
        Generates a SQL schema by mapping Pandas dtypes to PostgreSQL types.

        Args:
            df (pd.DataFrame): The DataFrame used to infer column names and types.

        Returns:
            str: SQL script to drop the existing table and create a new one.
        """
        columns_sql = []
        for col, dtype in df.dtypes.items():
            if col == "id":
                sql_type = "SERIAL PRIMARY KEY"
            elif col == "datetime":
                sql_type = "DATE NOT NULL UNIQUE"
            elif "float" in str(dtype):
                sql_type = "NUMERIC"
            elif "int" in str(dtype):
                sql_type = "INTEGER"
            else:
                sql_type = "TEXT"
            columns_sql.append(f"{col} {sql_type}")
        
        query = f"""
        DROP TABLE IF EXISTS {self.tablename};
        CREATE TABLE {self.tablename} (
            {', '.join(columns_sql)}
        );
        """
        return query
        
    def generate_insert_query(self, data: Any) -> Tuple[str, List[Tuple]]:
        """
        Generates a standard parameterized INSERT query for staging data.

        Args:
            data (Any): The transformed data to be formatted and inserted.

        Returns:
            Tuple[str, List[Tuple]]: SQL INSERT string and the formatted values.
        """
        cols, vals = self._format_data(data)
        query = f"""
        INSERT INTO {self.tablename} ({', '.join(cols)})
        VALUES %s;
        """     
        return query, vals


class FinalTableStrategy(TableStrategy):
    """
    Strategy for promoting a staging table to a final production table using table renaming.

    Args:
        tablename (str): The name of the final production table.
        staging_tablename (str): The name of the staging table to be promoted.

    Methods:
        generate_create_query(df): Returns SQL to atomically swap staging to production.
        generate_insert_query(data): Returns None as data is moved via table renaming.
    """
    def __init__(self, tablename: str, staging_tablename: str):
        self.tablename = tablename
        self.staging_tablename = staging_tablename

    def generate_create_query(self, df: pd.DataFrame = None) -> str:
        """
        Generates an atomic transaction to replace the final table with the staging table.

        Args:
            df (pd.DataFrame, optional): Unused; schema is inherited from the staging table.

        Returns:
            str: SQL block containing DROP, RENAME, and transaction control.
        """
        query = f"""
        BEGIN;
        DROP TABLE IF EXISTS {self.tablename};
        ALTER TABLE {self.staging_tablename} RENAME TO {self.tablename};
        COMMIT;
        """
        return query
    
    def generate_insert_query(self, data: Any) -> Tuple[str, List[Tuple]]:
        """
        Inherited placeholder; no insert query is needed for this strategy.

        Args:
            data (Any): Unused.

        Returns:
            None
        """
        pass
    

class TrainTestTableStrategy(TableStrategy):
    """
    Strategy for creating and populating tables used for machine learning training and testing.

    Args:
        tablename (str): The name of the target train or test table.

    Methods:
        generate_create_query(df): Dynamically generates a DROP and CREATE statement based on DataFrame schema.
        generate_insert_query(data): Returns a bulk INSERT statement for the split dataset.
    """
    def __init__(self, tablename: str):
        self.tablename = tablename

    def generate_create_query(self, df: pd.DataFrame = None) -> str:
        """
        Infers SQL types from the DataFrame to create a fresh train/test table.

        Args:
            df (pd.DataFrame): The DataFrame containing features and labels to map.

        Returns:
            str: SQL script to drop and recreate the table.
        """
        columns_sql = []
        for col, dtype in df.dtypes.items():
            if col == "id":
                sql_type = "SERIAL PRIMARY KEY"
            elif col == "datetime":
                sql_type = "DATE NOT NULL UNIQUE"
            elif "float" in str(dtype):
                sql_type = "NUMERIC"
            elif "int" in str(dtype):
                sql_type = "INTEGER"
            else:
                sql_type = "TEXT"
            columns_sql.append(f"{col} {sql_type}")
        
        query = f"""
        DROP TABLE IF EXISTS {self.tablename};
        CREATE TABLE {self.tablename} (
            {', '.join(columns_sql)}
        );
        """
        return query
    
    def generate_insert_query(self, data: Any) -> Tuple[str, List[Tuple]]:
        """
        Generates a parameterized INSERT query for the training or testing records.

        Args:
            data (Any): The split dataset (DataFrame or list) to be inserted.

        Returns:
            Tuple[str, List[Tuple]]: SQL INSERT string and the formatted values.
        """
        cols, vals = self._format_data(data)
        query = f"""
        INSERT INTO {self.tablename} ({', '.join(cols)})
        VALUES %s;
        """
        return query, vals


class PredictionTableStrategy(TableStrategy):
    """
    Strategy for managing model inference results with UPSERT logic.

    Args:
        tablename (str): The name of the table storing model predictions.

    Methods:
        generate_create_query(df): Returns a static SQL CREATE TABLE statement for predictions.
        generate_insert_query(data): Returns an UPSERT query to handle model re-runs.
    """
    def __init__(self, tablename: str):
        self.tablename = tablename

    def generate_create_query(self, df: pd.DataFrame = None) -> str:
        """
        Generates the schema for storing model outputs and metadata.

        Args:
            df (pd.DataFrame, optional): Unused; schema is predefined.

        Returns:
            str: SQL CREATE TABLE IF NOT EXISTS statement.
        """
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.tablename}(
            id SERIAL PRIMARY KEY,
            feature_date DATE NOT NULL UNIQUE,
            prediction_date DATE,
            predicted_direction INTEGER,
            model_name TEXT,
            model_version TEXT
        );
        """
        return query
    
    def generate_insert_query(self, data: Any) -> Tuple[str, List[Tuple]]:
        """
        Generates a parameterized UPSERT query to update existing predictions.

        Args:
            data (Any): The prediction records to be formatted and inserted.

        Returns:
            Tuple[str, List[Tuple]]: SQL INSERT string and the formatted values.
        """
        cols, vals = self._format_data(data)
        query = f"""
        INSERT INTO {self.tablename} ({', '.join(cols)})
        VALUES %s
        ON CONFLICT (feature_date) DO UPDATE SET
            prediction_date = EXCLUDED.prediction_date,
            predicted_direction = EXCLUDED.predicted_direction,
            model_name = EXCLUDED.model_name,
            model_version = EXCLUDED.model_version;
        """
        return query, vals


class SQLTableBuilder:
    """
    A context class for generating SQL queries using different table strategies.

    Args:
        strategy (TableStrategy): The initial strategy to use for query generation.

    Methods:
        set_strategy(strategy): Dynamically switches the current table strategy.
        get_create_query(df): Generates the SQL CREATE TABLE statement.
        get_insert_query(data): Generates the SQL INSERT statement and associated values.
    """
    def __init__(self, strategy: TableStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: TableStrategy):
        self._strategy = strategy

    def get_create_query(self, df: pd.DataFrame = None) -> str:
        """
        Generates a SQL CREATE TABLE query.

        Args:
            df (pd.DataFrame, optional): Dataframe used to infer schema for dynamic tables.

        Returns:
            str: The generated SQL CREATE statement.
        """
        return self._strategy.generate_create_query(df)
    
    def get_insert_query(self, data: Any) -> Tuple[str, List[Tuple]]:
        """
        Generates a SQL INSERT query and formats the data for execution.

        Args:
            data (Any): The dataset to be inserted (usually a DataFrame or list of dicts).

        Returns:
            Tuple[str, List[Tuple]]: A tuple containing the parameterized SQL string 
                                    and the list of records to insert.
        """
        return self._strategy.generate_insert_query(data)



if __name__ == "__main__":
    df = pd.read_csv("eur_usd_forex_data.csv")
    table_builder = SQLTableBuilder(PredictionTableStrategy(tablename="eur_usd_predictions"))
    prediction_query = table_builder.get_create_query()
    print("Prediction Query: ")
    print(prediction_query)

    table_builder.set_strategy(TrainTestTableStrategy(tablename="eur_usd_train"))
    training_query = table_builder.get_create_query(df)
    print("Train Query")
    print(training_query)