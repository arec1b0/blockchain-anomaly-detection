import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, Mock
import sys

# Create mock modules for Dask to avoid compatibility issues
mock_dd = MagicMock()
mock_client = MagicMock()

# Mock dask modules before importing the module under test
sys.modules['dask.dataframe'] = mock_dd
sys.modules['dask.distributed'] = mock_client

# Now import the module
from src.data_processing.data_cleaning_dask import DataCleanerDask


@pytest.fixture
def sample_data():
    """Create sample transaction data for testing"""
    data = {
        'value': [100, 200, None, 0, 500, 600, 700, 100],  # Includes duplicates, None, and invalid (0)
        'gas': [21000, 21000, 21000, None, 21000, 21000, 21000, 21000],
        'gasPrice': [50, 50, 50, 50, 50, 50, None, 50],
        'hash': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'a']  # 'a' is duplicate
    }
    df = pd.DataFrame(data)
    # Add actual duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df


@pytest.fixture
def clean_data():
    """Create clean sample data without issues"""
    data = {
        'value': [100, 200, 300, 400],
        'gas': [21000, 21000, 21000, 21000],
        'gasPrice': [50, 60, 70, 80]
    }
    return pd.DataFrame(data)


class TestDataCleanerDask:
    """Tests for DataCleanerDask class

    Note: These tests use mocked Dask to avoid Dask/Pandas compatibility issues.
    The tests verify the structure and behavior of the DataCleanerDask class.
    """

    def test_init(self, sample_data):
        """Test DataCleanerDask initialization"""
        # Configure mock to return a mock Dask DataFrame
        mock_dask_df = MagicMock()
        mock_dd.from_pandas.return_value = mock_dask_df

        cleaner = DataCleanerDask(sample_data)

        # Verify from_pandas was called with correct arguments
        mock_dd.from_pandas.assert_called_once_with(sample_data, npartitions=4)
        assert cleaner.df == mock_dask_df

    def test_remove_duplicates(self, sample_data):
        """Test removal of duplicate rows"""
        # Setup mock Dask DataFrame
        mock_dask_df = MagicMock()
        mock_dask_df.__len__.return_value = 100  # Initial count
        mock_dask_df_after = MagicMock()
        mock_dask_df_after.__len__.return_value = 90  # After deduplication
        mock_dask_df.drop_duplicates.return_value = mock_dask_df_after
        mock_dd.from_pandas.return_value = mock_dask_df

        cleaner = DataCleanerDask(sample_data)
        result = cleaner.remove_duplicates()

        # Verify drop_duplicates was called
        mock_dask_df.drop_duplicates.assert_called_once()
        assert result == mock_dask_df_after

    def test_handle_missing_values(self, sample_data):
        """Test handling of missing values"""
        # Setup mock
        mock_dask_df = MagicMock()
        mock_dask_df_after = MagicMock()
        mock_dask_df.fillna.return_value = mock_dask_df_after
        mock_dd.from_pandas.return_value = mock_dask_df

        cleaner = DataCleanerDask(sample_data)
        result = cleaner.handle_missing_values()

        # Verify fillna was called with 0
        mock_dask_df.fillna.assert_called_once_with(0)
        assert result == mock_dask_df_after

    def test_filter_invalid_transactions(self, sample_data):
        """Test filtering of invalid transactions"""
        # Setup mock with indexing support
        mock_dask_df = MagicMock()
        mock_value_column = MagicMock()
        mock_dd.from_pandas.return_value = mock_dask_df
        mock_dd.to_numeric.return_value = mock_value_column

        # Mock the __setitem__ and __getitem__ for df['value']
        mock_dask_df.__getitem__.return_value = mock_value_column
        mock_filtered_df = MagicMock()

        # Mock the comparison operation
        comparison_result = MagicMock()
        mock_value_column.__gt__.return_value = comparison_result
        mock_dask_df.__getitem__.side_effect = lambda x: comparison_result if isinstance(x, MagicMock) else mock_value_column

        cleaner = DataCleanerDask(sample_data)
        result = cleaner.filter_invalid_transactions()

        # Verify to_numeric was called
        assert mock_dd.to_numeric.called

    def test_clean_data_full_pipeline(self, sample_data):
        """Test the complete cleaning pipeline"""
        # Setup comprehensive mock
        mock_dask_df = MagicMock()
        computed_df = pd.DataFrame({
            'value': [100, 200, 300],
            'gas': [21000, 21000, 21000],
            'gasPrice': [50, 60, 70]
        })

        # Configure the mock chain
        mock_dd.from_pandas.return_value = mock_dask_df
        mock_dask_df.drop_duplicates.return_value = mock_dask_df
        mock_dask_df.fillna.return_value = mock_dask_df
        mock_dask_df.__getitem__.return_value = mock_dask_df
        mock_dask_df.__setitem__ = MagicMock()
        mock_dd.to_numeric.return_value = mock_dask_df
        mock_dask_df.__gt__.return_value = mock_dask_df
        mock_dask_df.compute.return_value = computed_df

        cleaner = DataCleanerDask(sample_data)
        result = cleaner.clean_data()

        # Verify compute was called
        assert mock_dask_df.compute.called

        # Result should be the computed DataFrame
        assert result is computed_df

    def test_clean_data_returns_pandas_dataframe(self, clean_data):
        """Test that clean_data returns a computed pandas DataFrame"""
        # Setup mock
        mock_dask_df = MagicMock()
        expected_result = pd.DataFrame({'value': [100, 200, 300]})

        # Create a mock for the value column that supports comparison
        mock_value_column = MagicMock()
        mock_comparison_result = MagicMock()
        mock_value_column.__gt__ = MagicMock(return_value=mock_comparison_result)

        mock_dd.from_pandas.return_value = mock_dask_df
        mock_dask_df.drop_duplicates.return_value = mock_dask_df
        mock_dask_df.fillna.return_value = mock_dask_df
        mock_dask_df.__getitem__.return_value = mock_value_column
        mock_dask_df.__setitem__ = MagicMock()
        mock_dd.to_numeric.return_value = mock_value_column
        # Mock the filtering operation df[comparison]
        mock_dask_df.__getitem__.side_effect = lambda x: mock_value_column if x == 'value' else mock_dask_df
        mock_dask_df.compute.return_value = expected_result

        cleaner = DataCleanerDask(clean_data)
        result = cleaner.clean_data()

        # Verify result is what compute returns
        assert result is expected_result
        assert isinstance(result, pd.DataFrame)


# Integration-style test using real pandas operations to verify logic
class TestDataCleanerDaskLogic:
    """Test the actual cleaning logic without Dask"""

    def test_cleaning_logic_with_pandas(self):
        """Verify the cleaning logic works correctly using pandas directly"""
        # Create test data
        data = {
            'value': [100, 200, None, 0, 500, 600, 700, 100],
            'gas': [21000, 21000, 21000, None, 21000, 21000, 21000, 21000],
        }
        df = pd.DataFrame(data)
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # Add duplicate

        # Apply the same logic as DataCleanerDask
        # 1. Remove duplicates
        df = df.drop_duplicates()

        # 2. Handle missing values
        df = df.fillna(0)

        # 3. Filter invalid transactions
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df[df['value'] > 0]

        # Verify results
        assert df.duplicated().sum() == 0  # No duplicates
        assert df.isna().sum().sum() == 0  # No missing values (in remaining data)
        assert all(df['value'] > 0)  # Only valid transactions
        assert len(df) < 9  # Should have removed some rows
