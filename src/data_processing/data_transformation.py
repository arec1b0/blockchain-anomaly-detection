"""
data_transformation.py

This module handles data transformation tasks such as converting timestamps and normalizing numeric data.
It prepares the cleaned data for analysis by ensuring all values are in the correct format and scale.

Adheres to the Single Responsibility Principle (SRP) by focusing solely on data transformation.
"""

import pandas as pd
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class DataTransformer:
    """
    DataTransformer class is responsible for transforming transaction data for analysis.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the DataTransformer with the provided DataFrame.

        :param df: Pandas DataFrame containing transaction data.
        """
        self.df = df

    def convert_timestamp(self):
        """
        Converts the UNIX timestamp column ('timeStamp') from seconds to human-readable datetime format.

        :return: DataFrame with converted timestamps.
        """
        self.df['timeStamp'] = pd.to_datetime(self.df['timeStamp'], unit='s', errors='coerce')
        logger.info("Converted UNIX timestamps to human-readable datetime format.")
        return self.df

    def normalize_column(self, column_name):
        """
        Normalizes the specified numeric column using min-max scaling.

        :param column_name: The name of the column to normalize.
        :return: DataFrame with normalized column values.
        """
        if column_name in self.df.columns:
            min_val = self.df[column_name].min()
            max_val = self.df[column_name].max()
            self.df[column_name] = (self.df[column_name] - min_val) / (max_val - min_val)
            logger.info(f"Normalized column '{column_name}' using min-max scaling.")
        else:
            logger.error(f"Column '{column_name}' not found in the data.")
            raise KeyError(f"Column '{column_name}' is missing.")
        return self.df

    def transform_data(self):
        """
        Applies all transformations: converts timestamps and normalizes numeric columns.

        :return: Fully transformed DataFrame.
        """
        self.convert_timestamp()
        self.normalize_column('value')  # Example: Normalizing the 'value' column
        logger.info("Data transformation process completed successfully.")
        return self.df
