import pandas as pd
import numpy as np
import os
from abc import ABC, abstractmethod

class DataTransformationTemplate(ABC):
    def __init__(self, df):
        self.df = df

    def apply_transformation(self) -> pd.DataFrame:
        raw_df = self.reorder_data(self.df)
        transformed_df = self.derive_indicators(raw_df)
        cleaned_df = self.clean_data(transformed_df)
        return cleaned_df

    @abstractmethod
    def reorder_data(self, df):
        pass 

    @abstractmethod
    def derive_indicators(self, df):
        pass 

    @abstractmethod
    def clean_data(self, df):
        pass 


class ForexDataTransformation(DataTransformationTemplate):
    def reorder_data(self, df):

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")
        
        numeric_cols = ['open', 'high', 'low', 'close']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)

        df = df.sort_values(by='datetime', ascending=True)
        return df

    def derive_indicators(self, df):
        df = df.copy()
        horizons = [2, 5, 60, 250, 1000]
        for horizon in horizons:
            rolling_averages = df[['close']].rolling(horizon).mean()
            ratio_column = f"close_ratio_{horizon}"
            df[ratio_column] = df["close"] / rolling_averages["close"]
        return df
    
    def clean_data(self, df):
        df = df.dropna()
        df = df.drop_duplicates()
        return df
    

if __name__ == "__main__": 
    raw_df = pd.read_csv("eur_usd_forex_data.csv")
    print("Initial raw dataframe: ")
    print(raw_df.head())
    print(raw_df.info())

    transformer = ForexDataTransformation(raw_df)
    transformed_df = transformer.apply_transformation()
    print("Transformed dataframe: ")
    print(transformed_df.tail())