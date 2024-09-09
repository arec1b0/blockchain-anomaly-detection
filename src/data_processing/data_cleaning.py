"""
data_cleaning.py

This module handles data cleaning operations such as removing duplicates, handling missing values,
and filtering invalid transactions. It prepares the dataset for further analysis and ensures data consistency.

Adheres to the Single Responsibility Principle (SRP) by focusing solely on data cleaning.
"""

import pandas as pd
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class DataCleaner:
    """
    DataCleaner class handles cleaning and preprocessing of the transaction data.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the DataCleaner with the provided DataFrame.

        :param df: Pandas DataFrame containing transaction data.
        """
        self.df = df

    def remove_duplicates(self):
        """
        Removes duplicate rows from the DataFrame.

        :return: Cleaned DataFrame without duplicates.
        """
        initial_count = len(self.df)
        self.df.drop_duplicates(inplace=True)
        final_count = len(self.df)
        logger.info(f"Removed {initial_count - final_count} duplicate rows.")
        return self.df

    def handle_missing_values(self):
        """
        Handles missing values by filling them with default values (0 for numerical columns).

        :return: Cleaned DataFrame with no missing values.
        """
        self.df.fillna(0, inplace=True)
        logger.info("Handled missing values by filling with default value 0.")
        return self.df

    def filter_invalid_transactions(self):
        """
        Filters out invalid transactions, such as those with zero or negative value.

        :return: DataFrame with valid transactions only.
        """
        self.df['value'] = pd.to_numeric(self.df['value'], errors='coerce')  # Convert values to numeric
        initial_count = len(self.df)
        self.df = self.df[self.df['value'] > 0]  # Keep only transactions with positive values
        final_count = len(self.df)
        logger.info(f"Filtered out {initial_count - final_count} invalid transactions with zero or negative value.")
        return self.df

    def clean_data(self):
        """
        Executes all cleaning steps: removes duplicates, handles missing values, and filters invalid transactions.

        :return: Fully cleaned DataFrame.
        """
        self.remove_duplicates()
        self.handle_missing_values()
        self.filter_invalid_transactions()
        logger.info("Data cleaning process completed successfully.")
        return self.df
