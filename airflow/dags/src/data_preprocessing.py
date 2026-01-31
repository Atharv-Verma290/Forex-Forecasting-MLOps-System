from abc import ABC, abstractmethod
import pandas as pd

class DataPreProcessingTemplate(ABC):
    def __init__(self, df):
        self.df = df

    def preprocess_data(self):
        processed_df = self.derive_target(self.df)
        train_df, test_df = self.split_dataset(processed_df)
        return train_df, test_df

    @abstractmethod
    def derive_target(self, df):
        pass 
    
    @abstractmethod
    def split_dataset(self, df):
        pass 


class ForexDataPreProcessing(DataPreProcessingTemplate):
    def derive_target(self, df: pd.DataFrame):
        df["tomorrow"] = df["close"].shift(1)
        df = df.dropna()
        df["target"] = (df["close"] < df["tomorrow"]).astype(int)
        return df
    
    def split_dataset(self, df: pd.DataFrame):
        test_df = df.iloc[:500]
        train_df = df.iloc[500:]

        return train_df, test_df
    

if __name__ == "__main__":
    df = pd.read_csv("eur_usd_forex_data.csv")
    print("Input dataset: ")
    print(df.head())
    print(df.info())

    processor = ForexDataPreProcessing(df)
    train_df, test_df = processor.preprocess_data()

    print("Train dataset: ")
    print(train_df.head())
    print(train_df.shape)

    print("Test datset: ")
    print(test_df.head())
    print(test_df.shape)