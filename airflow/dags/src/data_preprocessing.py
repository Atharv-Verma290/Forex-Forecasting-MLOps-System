from abc import ABC, abstractmethod
import pandas as pd

class DataPreProcessingTemplate(ABC):
    """
    An abstract base class defining the workflow for preparing datasets for machine learning.

    Args:
        df (pd.DataFrame): The input dataset (typically from the 'final' production table).

    Methods:
        preprocess_data: Orchestrates the target derivation and data splitting sequence.
        derive_target: Abstract method to define the label/y-variable.
        split_dataset: Abstract method to divide data into training and evaluation sets.
    """
    def __init__(self, df):
        self.df = df

    def preprocess_data(self):
        """
        Executes the preprocessing pipeline in a fixed order.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the (train_df, test_df).
        """
        processed_df = self.derive_target(self.df)
        train_df, test_df = self.split_dataset(processed_df)
        return train_df, test_df

    @abstractmethod
    def derive_target(self, df):
        """
        Logic to create the ground truth labels. Implementation required by subclass.
        """
        pass 
    
    @abstractmethod
    def split_dataset(self, df):
        """
        Logic to partition data (e.g., Random split or Time-series split). 
        Implementation required by subclass.
        """
        pass 


class ForexDataPreProcessing(DataPreProcessingTemplate):
    """
    Concrete implementation of preprocessing logic for Forex binary classification.

    Methods:
        derive_target: Creates a binary target indicating if the price will drop tomorrow.
        split_dataset: Partitions the data into training and testing sets using index slicing.
    """
    def derive_target(self, df: pd.DataFrame):
        """
        Calculates the 'target' column based on the next day's closing price.

        Args:
            df (pd.DataFrame): The feature-enriched price data.

        Returns:
            pd.DataFrame: DataFrame with the 'target' binary label.
        """
        df["tomorrow"] = df["close"].shift(1)
        df = df.dropna()
        df["target"] = (df["close"] < df["tomorrow"]).astype(int)
        return df
    
    def split_dataset(self, df: pd.DataFrame):
        """
        Splits data into train and test sets based on row position.

        Args:
            df (pd.DataFrame): The labeled dataset.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
        """
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