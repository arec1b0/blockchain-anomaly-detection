"""
Data Processing Module

This module provides data cleaning and transformation utilities
for blockchain transaction data, with support for both Pandas and Dask.
"""

from src.data_processing.data_cleaning import DataCleaner
from src.data_processing.data_transformation import DataTransformer

# Try to import Dask components, but make them optional
try:
    from src.data_processing.data_cleaning_dask import (
        DataCleanerDask,
        get_dask_client,
        close_dask_client
    )
    _DASK_AVAILABLE = True
    __all__ = [
        'DataCleaner',
        'DataCleanerDask',
        'DataTransformer',
        'get_dask_client',
        'close_dask_client'
    ]
except (ImportError, TypeError) as e:
    # Dask is not available or incompatible with current Pandas version
    _DASK_AVAILABLE = False
    DataCleanerDask = None
    get_dask_client = None
    close_dask_client = None
    __all__ = [
        'DataCleaner',
        'DataTransformer',
    ]
