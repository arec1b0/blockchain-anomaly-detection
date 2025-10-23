"""
data_cleaning_dask.py

This module provides data cleaning operations using Dask for parallel and distributed data processing.
It is designed to handle large datasets efficiently by utilizing Dask's parallelization capabilities.

Adheres to the Single Responsibility Principle (SRP) by focusing solely on data cleaning using Dask.
"""

import dask.dataframe as dd
from dask.distributed import Client
from typing import Optional
from src.utils.logger import get_logger
from src.utils.config import get_config

# Initialize logger
logger = get_logger(__name__)

# Global Dask client instance
_dask_client = None


def get_dask_client() -> Client:
    """
    Get or create a Dask client instance.

    This function initializes a Dask client with the configuration specified
    in the environment variables. If a client has already been initialized,
    it returns the existing client.

    Returns:
        Client: A Dask distributed Client instance.
    """
    global _dask_client
    if _dask_client is None:
        config = get_config()
        try:
            _dask_client = Client(
                n_workers=config.DASK_N_WORKERS,
                threads_per_worker=config.DASK_THREADS_PER_WORKER,
                memory_limit=config.DASK_MEMORY_LIMIT,
            )
            logger.info(f"Dask client initialized with {config.DASK_N_WORKERS} workers")
        except Exception as e:
            logger.error(f"Failed to initialize Dask client: {e}")
            raise
    return _dask_client


def close_dask_client():
    """
    Close the Dask client if it exists.
    """
    global _dask_client
    if _dask_client is not None:
        _dask_client.close()
        _dask_client = None
        logger.info("Dask client closed")


class DataCleanerDask:
    """
    Provides parallel data cleaning operations using Dask.

    This class is designed to handle large datasets efficiently by leveraging
    Dask for parallel and distributed data processing.
    """

    def __init__(self, df, npartitions: Optional[int] = None, client: Optional[Client] = None):
        """
        Initializes the DataCleanerDask with the provided DataFrame.

        Args:
            df (pd.DataFrame): The Pandas DataFrame containing transaction data.
            npartitions (Optional[int]): The number of partitions for the Dask DataFrame.
                Defaults to 4.
            client (Optional[Client]): An optional Dask client instance. If not provided,
                a global client will be used.

        Raises:
            ValueError: If the input DataFrame is None or empty.
        """
        if df is None or len(df) == 0:
            raise ValueError("DataFrame cannot be None or empty")

        self.client = client if client is not None else get_dask_client()
        self.npartitions = npartitions or 4
        self.df = dd.from_pandas(df, npartitions=self.npartitions)
        logger.info(f"DataCleanerDask initialized with {self.npartitions} partitions")

    def remove_duplicates(self):
        """
        Removes duplicate rows from the Dask DataFrame in parallel.

        Returns:
            dd.DataFrame: A Dask DataFrame with duplicate rows removed.
        """
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        final_count = len(self.df)
        logger.info(f"Removed {initial_count - final_count} duplicate rows using Dask.")
        return self.df

    def handle_missing_values(self):
        """
        Handles missing values in the Dask DataFrame by filling them with a default value (0).

        Returns:
            dd.DataFrame: A Dask DataFrame with missing values handled.
        """
        self.df = self.df.fillna(0)
        logger.info("Handled missing values using Dask by filling with default value 0.")
        return self.df

    def filter_invalid_transactions(self):
        """
        Filters out invalid transactions from the Dask DataFrame.

        This method removes transactions with zero or negative values.

        Returns:
            dd.DataFrame: A Dask DataFrame with invalid transactions filtered out.
        """
        self.df['value'] = dd.to_numeric(self.df['value'], errors='coerce')
        self.df = self.df[self.df['value'] > 0]
        logger.info("Filtered out invalid transactions using Dask.")
        return self.df

    def clean_data(self):
        """
        Executes all data cleaning steps in parallel using Dask.

        This method performs the following cleaning operations:
        1. Removes duplicate rows.
        2. Handles missing values.
        3. Filters invalid transactions.

        Returns:
            pd.DataFrame: A fully cleaned Pandas DataFrame.
        """
        self.remove_duplicates()
        self.handle_missing_values()
        self.filter_invalid_transactions()
        logger.info("Data cleaning process completed successfully using Dask.")
        return self.df.compute()  # Trigger computation of Dask DataFrame
