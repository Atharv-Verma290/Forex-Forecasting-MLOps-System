import pandas as pd
import numpy as np
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
    def reorder_data(self, df: pd.DataFrame):

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")
        
        numeric_cols = ['open', 'high', 'low', 'close']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)

        df = df.set_index('datetime')
        df = df.sort_index(ascending=True)
        return df

    def derive_indicators(self, df: pd.DataFrame):
        df = df.copy()
        df['log_return'] = np.log(df['close'])
        df['abs_log_return'] = df['log_return'].abs()
        df['hl_range'] = np.log(df['high'] / df['low'])

        windows = [5, 10, 20, 60]
        for w in windows:
            df[f'vol_std_{w}'] = df['log_return'].rolling(w).std()
            df[f'vol_absmean_{w}'] = df['abs_log_return'].rolling(w).mean()
            df[f'hl_range_mean_{w}'] = df['hl_range'].rolling(w).mean()

        df['vol_ratio_5_20'] = df['vol_std_5'] / df['vol_std_20']
        df['vol_ratio_10_60'] = df['vol_std_10'] / df['vol_std_60']

        df['vol_20_change_5'] = df['vol_std_20'] - df['vol_std_20'].shift(5)
        df['vol_20_pct_change_5'] = df['vol_std_20'].pct_change(5)

        window = 80
        df['vol_20_zscore'] = (
            df['vol_std_20'] - df['vol_std_20'].rolling(window).mean()
        ) / df['vol_std_20'].rolling(window).std()

        k = 2

        df['large_move'] = (df['abs_log_return'] > k * df['vol_std_20']).astype(int)
        df['large_move_count_20'] = df['large_move'].rolling(20).sum()
        return df
    
    def clean_data(self, df: pd.DataFrame):
        df = df.dropna()
        df = df.drop_duplicates()
        df = df.reset_index()
        return df
    

if __name__ == "__main__": 
    raw_df = pd.read_csv("forex_sample_data.csv")
    print("Initial raw dataframe: ")
    print(raw_df.head())
    print(raw_df.info())

    transformer = ForexDataTransformation(raw_df)
    transformed_df = transformer.apply_transformation()
    print("Transformed dataframe: ")
    print(transformed_df.tail())