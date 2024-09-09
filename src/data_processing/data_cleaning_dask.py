"""
data_cleaning_dask.py

This module provides data cleaning operations using Dask for parallel and distributed data processing.
It is designed to handle large datasets efficiently by utilizing Dask's parallelization capabilities.

Adheres to the Single Responsibility Principle (SRP) by focusing solely on data cleaning using Dask.
"""

import dask.dataframe as dd
from dask.distributed import Client
from src.utils.logger import get_logger

# Initialize Dask client and logger
client = Client()
logger = get_logger(__name__)


class DataCleanerDask:
    """
    DataCleanerDask class provides parallel data cleaning operations using Dask.
    """

    def __init__(self, df):
        """
        Initializes the DataCleanerDask with the provided Dask DataFrame.

        :param df: Dask DataFrame containing transaction data.
        """
        self.df = dd.from_pandas(df, npartitions=4)

    def remove_duplicates(self):
        """
        Removes duplicate rows from the Dask DataFrame in parallel.

        :return: Dask DataFrame without duplicates.
        """
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        final_count = len(self.df)
        logger.info(f"Removed {initial_count - final_count} duplicate rows using Dask.")
        return self.df

    def handle_missing_values(self):
        """
        Handles missing values in the Dask DataFrame by filling them with default values (0).

        :return: Cleaned Dask DataFrame with no missing values.
        """
        self.df = self.df.fillna(0)
        logger.info("Handled missing values using Dask by filling with default value 0.")
        return self.df

    def filter_invalid_transactions(self):
        """
        Filters out invalid transactions in the Dask DataFrame, such as those with zero or negative value.

        :return: Dask DataFrame with valid transactions only.
        """
        self.df['value'] = dd.to_numeric(self.df['value'], errors='coerce')
        self.df = self.df[self.df['value'] > 0]
        logger.info("Filtered out invalid transactions using Dask.")
        return self.df

    def clean_data(self):
        """
        Executes all cleaning steps in parallel using Dask: removes duplicates, handles missing values,
        and filters invalid transactions.

        :return: Fully cleaned Dask DataFrame.
        """
        self.remove_duplicates()
        self.handle_missing_values()
        self.filter_invalid_transactions()
        logger.info("Data cleaning process completed successfully using Dask.")
        return self.df.compute()  # Trigger computation of Dask DataFrame
