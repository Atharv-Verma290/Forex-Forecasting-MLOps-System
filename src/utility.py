from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import pandas as pd

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
    

# SQL Query generators
class TableStrategy(ABC):
    @abstractmethod
    def generate_query(self, df: pd.DataFrame = None) -> str:
        pass


class RawTableStrategy(TableStrategy):
    def __init__(self, tablename: str):
        self.tablename = tablename 

    def generate_query(self, df: pd.DataFrame = None) -> str:
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
    

class StagingTableStrategy(TableStrategy):
    def __init__(self, tablename: str):
        self.tablename = tablename

    def generate_query(self, df: pd.DataFrame = None) -> str:
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
        

class FinalTableStrategy(TableStrategy):
    def __init__(self, tablename: str, staging_tablename: str):
        self.tablename = tablename
        self.staging_tablename = staging_tablename

    def generate_query(self, df: pd.DataFrame = None) -> str:
        query = f"""
            BEGIN;
            DROP TABLE IF EXISTS {self.tablename};
            ALTER TABLE {self.staging_tablename} RENAME TO {self.tablename};
            COMMIT;
        """
        return query
    

class TrainTestTableStrategy(TableStrategy):
    def __init__(self, tablename: str):
        self.tablename = tablename

    def generate_query(self, df: pd.DataFrame = None) -> str:
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
            CREATE TABLE IF NOT EXISTS {self.tablename} (
                {', '.join(columns_sql)}
            );
        """
        return query


class PredictionTableStrategy(TableStrategy):
    def __init__(self, tablename: str):
        self.tablename = tablename

    def generate_query(self, df: pd.DataFrame = None) -> str:
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


class SQLTableBuilder:
    def __init__(self, strategy: TableStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: TableStrategy):
        self._strategy = strategy

    def get_query(self, df: pd.DataFrame = None) -> str:
        return self._strategy.generate_query(df)



if __name__ == "__main__":
    df = pd.read_csv("eur_usd_forex_data.csv")
    table_builder = SQLTableBuilder(PredictionTableStrategy(tablename="eur_usd_predictions"))
    prediction_query = table_builder.get_query()
    print("Prediction Query: ")
    print(prediction_query)

    table_builder.set_strategy(TrainTestTableStrategy(tablename="eur_usd_train"))
    training_query = table_builder.get_query(df)
    print("Train Query")
    print(training_query)