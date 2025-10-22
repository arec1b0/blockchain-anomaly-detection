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
        :raises ValueError: If df is None or empty.
        :raises TypeError: If df is not a pandas DataFrame.
        """
        if df is None:
            raise ValueError("DataFrame cannot be None")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        if len(df) == 0:
            raise ValueError("DataFrame cannot be empty")

        self.df = df.copy()
        logger.info(f"DataTransformer initialized with {len(self.df)} rows")

    def convert_timestamp(self, column_name='timeStamp'):
        """
        Converts the UNIX timestamp column from seconds to human-readable datetime format.

        :param column_name: Name of the timestamp column (default: 'timeStamp').
        :return: DataFrame with converted timestamps.
        :raises ValueError: If the timestamp column doesn't exist.
        :raises RuntimeError: If conversion fails.
        """
        try:
            if column_name not in self.df.columns:
                raise ValueError(f"Column '{column_name}' not found in DataFrame")

            self.df[column_name] = pd.to_datetime(self.df[column_name], unit='s', errors='coerce')

            # Check for any failed conversions
            failed_count = self.df[column_name].isna().sum()
            if failed_count > 0:
                logger.warning(f"{failed_count} timestamps could not be converted")

            logger.info(f"Converted UNIX timestamps to datetime format in column '{column_name}'")
            return self.df
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error converting timestamps: {e}")
            raise RuntimeError(f"Failed to convert timestamps: {e}") from e

    def normalize_column(self, column_name):
        """
        Normalizes the specified numeric column using min-max scaling.

        :param column_name: The name of the column to normalize.
        :return: DataFrame with normalized column values.
        :raises ValueError: If column doesn't exist or contains non-numeric data.
        :raises RuntimeError: If normalization fails.
        """
        try:
            if not column_name:
                raise ValueError("Column name cannot be empty")

            if column_name not in self.df.columns:
                raise ValueError(f"Column '{column_name}' not found in DataFrame")

            # Convert to numeric if possible
            self.df[column_name] = pd.to_numeric(self.df[column_name], errors='coerce')

            # Check if column has numeric data
            if self.df[column_name].isna().all():
                raise ValueError(f"Column '{column_name}' contains no numeric data")

            min_val = self.df[column_name].min()
            max_val = self.df[column_name].max()

            # Handle case where all values are the same
            if min_val == max_val:
                logger.warning(f"Column '{column_name}' has constant value, setting to 0")
                self.df[column_name] = 0
            else:
                self.df[column_name] = (self.df[column_name] - min_val) / (max_val - min_val)

            logger.info(f"Normalized column '{column_name}' using min-max scaling")
            return self.df
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error normalizing column '{column_name}': {e}")
            raise RuntimeError(f"Failed to normalize column: {e}") from e

    def transform_data(self):
        """
        Applies all transformations: converts timestamps and normalizes numeric columns.

        :return: Fully transformed DataFrame.
        :raises RuntimeError: If transformation process fails.
        """
        try:
            self.convert_timestamp()
            if 'value' in self.df.columns:
                self.normalize_column('value')
            else:
                logger.warning("'value' column not found, skipping normalization")
            logger.info("Data transformation process completed successfully")
            return self.df
        except Exception as e:
            logger.error(f"Data transformation process failed: {e}")
            raise RuntimeError(f"Data transformation failed: {e}") from e
