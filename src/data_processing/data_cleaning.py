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
        :raises ValueError: If df is None, not a DataFrame, or empty.
        :raises TypeError: If df is not a pandas DataFrame.
        """
        if df is None:
            raise ValueError("DataFrame cannot be None")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        if len(df) == 0:
            raise ValueError("DataFrame cannot be empty")

        self.df = df.copy()  # Create a copy to avoid modifying the original
        logger.info(f"DataCleaner initialized with {len(self.df)} rows")

    def remove_duplicates(self):
        """
        Removes duplicate rows from the DataFrame.

        :return: Cleaned DataFrame without duplicates.
        :raises RuntimeError: If an error occurs during duplicate removal.
        """
        try:
            initial_count = len(self.df)
            self.df.drop_duplicates(inplace=True)
            final_count = len(self.df)
            logger.info(f"Removed {initial_count - final_count} duplicate rows.")
            return self.df
        except Exception as e:
            logger.error(f"Error removing duplicates: {e}")
            raise RuntimeError(f"Failed to remove duplicates: {e}") from e

    def handle_missing_values(self, fill_value=0):
        """
        Handles missing values by filling them with default values (0 for numerical columns).

        :param fill_value: Value to use for filling missing values (default: 0).
        :return: Cleaned DataFrame with no missing values.
        :raises RuntimeError: If an error occurs during missing value handling.
        """
        try:
            self.df.fillna(fill_value, inplace=True)
            logger.info(f"Handled missing values by filling with value {fill_value}.")
            return self.df
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            raise RuntimeError(f"Failed to handle missing values: {e}") from e

    def filter_invalid_transactions(self):
        """
        Filters out invalid transactions, such as those with zero or negative value.

        :return: DataFrame with valid transactions only.
        :raises ValueError: If 'value' column doesn't exist.
        :raises RuntimeError: If an error occurs during filtering.
        """
        try:
            if 'value' not in self.df.columns:
                raise ValueError("DataFrame must contain a 'value' column")

            self.df['value'] = pd.to_numeric(self.df['value'], errors='coerce')  # Convert values to numeric
            initial_count = len(self.df)
            self.df = self.df[self.df['value'] > 0]  # Keep only transactions with positive values
            final_count = len(self.df)
            logger.info(f"Filtered out {initial_count - final_count} invalid transactions with zero or negative value.")

            if len(self.df) == 0:
                logger.warning("All transactions were filtered out as invalid")

            return self.df
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error filtering invalid transactions: {e}")
            raise RuntimeError(f"Failed to filter invalid transactions: {e}") from e

    def clean_data(self):
        """
        Executes all cleaning steps: removes duplicates, handles missing values, and filters invalid transactions.

        :return: Fully cleaned DataFrame.
        :raises RuntimeError: If the cleaning process fails.
        """
        try:
            self.remove_duplicates()
            self.handle_missing_values()
            self.filter_invalid_transactions()
            logger.info("Data cleaning process completed successfully.")
            return self.df
        except Exception as e:
            logger.error(f"Data cleaning process failed: {e}")
            raise RuntimeError(f"Data cleaning failed: {e}") from e
