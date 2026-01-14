from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.model_selection import TimeSeriesSplit

def next_forex_trading_day(feature_date_utc: datetime) -> datetime.date:
    """
    Returns the next day, unless it's Saturday, 
    in which case it jumps to Monday.
    """
    weekday = feature_date_utc.weekday() # Monday=0, Sunday=6

    # If Saturday, jump 2 days to Monday. 
    # Otherwise, just move to the next calendar day.
    days_to_add = 2 if weekday == 5 else 1
    
    return (feature_date_utc + timedelta(days=1)).date()


def cross_validate_model(classifier, X, y):
    tscv = TimeSeriesSplit(
        n_splits=5,
        test_size=300,
        gap=1
        )
    precision_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        classifier.fit(X=X_tr, y=y_tr)
        preds = classifier.predict(X_val)
        fold_score = precision_score(y_val, preds, zero_division=0)
        precision_scores.append(fold_score)

    return np.mean(precision_scores), np.std(precision_scores)


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
    def __init__(self, tablename: str):
        self.tablename = tablename 

    def generate_create_query(self, df: pd.DataFrame = None) -> str:
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
    def __init__(self, tablename: str):
        self.tablename = tablename

    def generate_create_query(self, df: pd.DataFrame = None) -> str:
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
        cols, vals = self._format_data(data)
        query = f"""
        INSERT INTO {self.tablename} ({', '.join(cols)})
        VALUES %s;
        """     
        return query, vals


class FinalTableStrategy(TableStrategy):
    def __init__(self, tablename: str, staging_tablename: str):
        self.tablename = tablename
        self.staging_tablename = staging_tablename

    def generate_create_query(self, df: pd.DataFrame = None) -> str:
        query = f"""
        BEGIN;
        DROP TABLE IF EXISTS {self.tablename};
        ALTER TABLE {self.staging_tablename} RENAME TO {self.tablename};
        COMMIT;
        """
        return query
    
    def generate_insert_query(self, data: Any) -> Tuple[str, List[Tuple]]:
        pass
    

class TrainTestTableStrategy(TableStrategy):
    def __init__(self, tablename: str):
        self.tablename = tablename

    def generate_create_query(self, df: pd.DataFrame = None) -> str:
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
        cols, vals = self._format_data(data)
        query = f"""
        INSERT INTO {self.tablename} ({', '.join(cols)})
        VALUES %s;
        """
        return query, vals


class PredictionTableStrategy(TableStrategy):
    def __init__(self, tablename: str):
        self.tablename = tablename

    def generate_create_query(self, df: pd.DataFrame = None) -> str:
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
    def __init__(self, strategy: TableStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: TableStrategy):
        self._strategy = strategy

    def get_create_query(self, df: pd.DataFrame = None) -> str:
        return self._strategy.generate_create_query(df)
    
    def get_insert_query(self, data: Any) -> Tuple[str, List[Tuple]]:
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