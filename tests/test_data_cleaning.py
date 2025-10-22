import pytest
import pandas as pd
import numpy as np
from src.data_processing.data_cleaning import DataCleaner


@pytest.fixture
def sample_data():
    """Create sample transaction data for testing"""
    data = {
        'value': [100, 200, None, 0, 500, 600, 700],
        'gas': [21000, 21000, 21000, None, 21000, 21000, 21000],
        'gasPrice': [50, 50, 50, 50, 50, 50, None]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_duplicates():
    """Create sample data with duplicate rows"""
    data = {
        'value': [100, 200, 100, 300],
        'gas': [21000, 22000, 21000, 23000],
        'hash': ['a', 'b', 'a', 'c']
    }
    df = pd.DataFrame(data)
    # Add exact duplicate row
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


@pytest.fixture
def clean_data():
    """Create clean sample data without issues"""
    data = {
        'value': [100, 200, 300, 400],
        'gas': [21000, 21000, 21000, 21000],
        'gasPrice': [50, 60, 70, 80]
    }
    return pd.DataFrame(data)


class TestDataCleanerInitialization:
    """Tests for DataCleaner initialization and validation"""

    def test_init_with_none_dataframe(self):
        """Test that DataCleaner raises ValueError when df is None"""
        with pytest.raises(ValueError) as exc_info:
            DataCleaner(None)
        assert "DataFrame cannot be None" in str(exc_info.value)

    def test_init_with_empty_dataframe(self):
        """Test that DataCleaner raises ValueError when df is empty"""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError) as exc_info:
            DataCleaner(empty_df)
        assert "DataFrame cannot be empty" in str(exc_info.value)

    def test_init_with_wrong_type(self):
        """Test that DataCleaner raises TypeError when df is not a DataFrame"""
        with pytest.raises(TypeError) as exc_info:
            DataCleaner([1, 2, 3])
        assert "Expected pandas DataFrame" in str(exc_info.value)

        with pytest.raises(TypeError) as exc_info:
            DataCleaner("not a dataframe")
        assert "Expected pandas DataFrame" in str(exc_info.value)

        with pytest.raises(TypeError) as exc_info:
            DataCleaner({'key': 'value'})
        assert "Expected pandas DataFrame" in str(exc_info.value)

    def test_init_with_valid_dataframe(self, sample_data):
        """Test successful initialization with valid DataFrame"""
        cleaner = DataCleaner(sample_data)
        assert cleaner.df is not None
        assert isinstance(cleaner.df, pd.DataFrame)
        assert len(cleaner.df) == len(sample_data)

    def test_init_creates_copy_of_dataframe(self, sample_data):
        """Test that DataCleaner creates a copy of the input DataFrame"""
        cleaner = DataCleaner(sample_data)
        assert cleaner.df is not sample_data
        # Modify cleaner's df
        cleaner.df.loc[0, 'value'] = 999
        # Original should be unchanged
        assert sample_data.loc[0, 'value'] != 999


class TestRemoveDuplicates:
    """Tests for remove_duplicates method"""

    def test_remove_duplicates(self, sample_data):
        """Test basic duplicate removal"""
        cleaner = DataCleaner(sample_data)
        cleaned_data = cleaner.remove_duplicates()

        assert len(cleaned_data) == len(sample_data), "Duplicate removal failed."

    def test_remove_duplicates_with_actual_duplicates(self, sample_data_with_duplicates):
        """Test removal of actual duplicate rows"""
        initial_len = len(sample_data_with_duplicates)
        cleaner = DataCleaner(sample_data_with_duplicates)
        cleaned_data = cleaner.remove_duplicates()

        assert len(cleaned_data) < initial_len
        assert cleaned_data.duplicated().sum() == 0

    def test_remove_duplicates_no_duplicates(self, clean_data):
        """Test duplicate removal when there are no duplicates"""
        cleaner = DataCleaner(clean_data)
        initial_len = len(clean_data)
        cleaned_data = cleaner.remove_duplicates()

        assert len(cleaned_data) == initial_len

    def test_remove_duplicates_error_handling(self):
        """Test error handling in remove_duplicates"""
        # Create a DataFrame that might cause issues
        df = pd.DataFrame({'value': [1, 2, 3]})
        cleaner = DataCleaner(df)

        # Mock the drop_duplicates to raise an exception
        with pytest.raises(RuntimeError) as exc_info:
            cleaner.df = None  # This will cause an error
            cleaner.remove_duplicates()
        assert "Failed to remove duplicates" in str(exc_info.value)


class TestHandleMissingValues:
    """Tests for handle_missing_values method"""

    def test_handle_missing_values(self, sample_data):
        """Test basic missing value handling"""
        cleaner = DataCleaner(sample_data)
        cleaned_data = cleaner.handle_missing_values()

        assert cleaned_data.isna().sum().sum() == 0, "Missing value handling failed."

    def test_handle_missing_values_with_custom_fill(self, sample_data):
        """Test missing value handling with custom fill value"""
        cleaner = DataCleaner(sample_data)
        cleaned_data = cleaner.handle_missing_values(fill_value=-1)

        assert cleaned_data.isna().sum().sum() == 0
        # Check that missing values were filled with -1
        assert -1 in cleaned_data.values

    def test_handle_missing_values_no_missing(self, clean_data):
        """Test missing value handling when there are no missing values"""
        cleaner = DataCleaner(clean_data)
        cleaned_data = cleaner.handle_missing_values()

        assert cleaned_data.isna().sum().sum() == 0
        assert len(cleaned_data) == len(clean_data)

    def test_handle_missing_values_all_missing_column(self):
        """Test handling of column with all missing values"""
        df = pd.DataFrame({
            'value': [None, None, None],
            'gas': [1, 2, 3]
        })
        cleaner = DataCleaner(df)
        cleaned_data = cleaner.handle_missing_values(fill_value=0)

        assert cleaned_data.isna().sum().sum() == 0
        assert all(cleaned_data['value'] == 0)

    def test_handle_missing_values_error_handling(self):
        """Test error handling in handle_missing_values"""
        df = pd.DataFrame({'value': [1, 2, 3]})
        cleaner = DataCleaner(df)

        # Mock fillna to raise an exception
        with pytest.raises(RuntimeError) as exc_info:
            cleaner.df = None
            cleaner.handle_missing_values()
        assert "Failed to handle missing values" in str(exc_info.value)


class TestFilterInvalidTransactions:
    """Tests for filter_invalid_transactions method"""

    def test_filter_invalid_transactions(self, sample_data):
        """Test basic invalid transaction filtering"""
        cleaner = DataCleaner(sample_data)
        cleaned_data = cleaner.filter_invalid_transactions()

        assert all(cleaned_data['value'] > 0), "Invalid transactions not filtered correctly."

    def test_filter_invalid_transactions_removes_zero_values(self):
        """Test that zero values are filtered out"""
        df = pd.DataFrame({
            'value': [100, 0, 200, 0, 300],
            'gas': [21000, 21000, 21000, 21000, 21000]
        })
        cleaner = DataCleaner(df)
        cleaned_data = cleaner.filter_invalid_transactions()

        assert len(cleaned_data) == 3
        assert 0 not in cleaned_data['value'].values

    def test_filter_invalid_transactions_removes_negative_values(self):
        """Test that negative values are filtered out"""
        df = pd.DataFrame({
            'value': [100, -50, 200, -100, 300],
            'gas': [21000, 21000, 21000, 21000, 21000]
        })
        cleaner = DataCleaner(df)
        cleaned_data = cleaner.filter_invalid_transactions()

        assert len(cleaned_data) == 3
        assert all(cleaned_data['value'] > 0)

    def test_filter_invalid_transactions_missing_value_column(self):
        """Test that ValueError is raised when 'value' column is missing"""
        df = pd.DataFrame({
            'price': [100, 200, 300],
            'gas': [21000, 21000, 21000]
        })
        cleaner = DataCleaner(df)

        with pytest.raises(ValueError) as exc_info:
            cleaner.filter_invalid_transactions()
        assert "must contain a 'value' column" in str(exc_info.value)

    def test_filter_invalid_transactions_all_invalid(self):
        """Test filtering when all transactions are invalid"""
        df = pd.DataFrame({
            'value': [0, -10, -20, 0],
            'gas': [21000, 21000, 21000, 21000]
        })
        cleaner = DataCleaner(df)
        cleaned_data = cleaner.filter_invalid_transactions()

        assert len(cleaned_data) == 0

    def test_filter_invalid_transactions_converts_to_numeric(self):
        """Test that non-numeric values are converted properly"""
        df = pd.DataFrame({
            'value': ['100', '200', 'invalid', '300'],
            'gas': [21000, 21000, 21000, 21000]
        })
        cleaner = DataCleaner(df)
        cleaned_data = cleaner.filter_invalid_transactions()

        # 'invalid' should be converted to NaN and then filtered out
        assert len(cleaned_data) == 3
        assert all(cleaned_data['value'] > 0)

    def test_filter_invalid_transactions_error_handling(self):
        """Test error handling in filter_invalid_transactions"""
        df = pd.DataFrame({'value': [1, 2, 3]})
        cleaner = DataCleaner(df)

        # Set df to None to cause an error
        with pytest.raises(RuntimeError) as exc_info:
            cleaner.df = None
            cleaner.filter_invalid_transactions()
        assert "Failed to filter invalid transactions" in str(exc_info.value)


class TestCleanData:
    """Tests for clean_data method (full pipeline)"""

    def test_clean_data_full_pipeline(self, sample_data):
        """Test complete cleaning pipeline"""
        cleaner = DataCleaner(sample_data)
        cleaned_data = cleaner.clean_data()

        # Should have no duplicates
        assert cleaned_data.duplicated().sum() == 0
        # Should have no missing values
        assert cleaned_data.isna().sum().sum() == 0
        # Should have only valid transactions
        assert all(cleaned_data['value'] > 0)

    def test_clean_data_with_all_issues(self):
        """Test cleaning data with duplicates, missing values, and invalid transactions"""
        df = pd.DataFrame({
            'value': [100, 200, None, 0, -50, 100, 300],
            'gas': [21000, 21000, None, 21000, 21000, 21000, 21000]
        })
        # Add duplicate
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

        cleaner = DataCleaner(df)
        cleaned_data = cleaner.clean_data()

        assert cleaned_data.duplicated().sum() == 0
        assert cleaned_data.isna().sum().sum() == 0
        assert all(cleaned_data['value'] > 0)
        assert len(cleaned_data) < len(df)

    def test_clean_data_already_clean(self, clean_data):
        """Test cleaning already clean data"""
        cleaner = DataCleaner(clean_data)
        initial_len = len(clean_data)
        cleaned_data = cleaner.clean_data()

        assert len(cleaned_data) == initial_len
        assert cleaned_data.duplicated().sum() == 0
        assert cleaned_data.isna().sum().sum() == 0
        assert all(cleaned_data['value'] > 0)

    def test_clean_data_error_handling(self):
        """Test error handling in clean_data pipeline"""
        df = pd.DataFrame({'other_column': [1, 2, 3]})
        cleaner = DataCleaner(df)

        # Should fail because 'value' column is missing
        with pytest.raises(RuntimeError) as exc_info:
            cleaner.clean_data()
        assert "Data cleaning failed" in str(exc_info.value)


class TestDataCleanerEdgeCases:
    """Tests for edge cases and special scenarios"""

    def test_single_row_dataframe(self):
        """Test cleaning a DataFrame with only one row"""
        df = pd.DataFrame({'value': [100], 'gas': [21000]})
        cleaner = DataCleaner(df)
        cleaned_data = cleaner.clean_data()

        assert len(cleaned_data) == 1
        assert cleaned_data['value'].iloc[0] == 100

    def test_large_dataframe(self):
        """Test cleaning a large DataFrame"""
        df = pd.DataFrame({
            'value': list(range(1, 10001)),
            'gas': [21000] * 10000
        })
        cleaner = DataCleaner(df)
        cleaned_data = cleaner.clean_data()

        assert len(cleaned_data) == 10000
        assert all(cleaned_data['value'] > 0)

    def test_dataframe_with_special_values(self):
        """Test handling of special numeric values"""
        df = pd.DataFrame({
            'value': [100, np.inf, 200, -np.inf, 300, np.nan],
            'gas': [21000, 21000, 21000, 21000, 21000, 21000]
        })
        cleaner = DataCleaner(df)
        cleaned_data = cleaner.clean_data()

        # -inf should be filtered (it's <= 0), NaN is filled with 0 and then filtered
        # np.inf is > 0, so it will remain unless explicitly handled
        # Current implementation only filters value <= 0, so inf remains
        assert all(cleaned_data['value'] > 0)
        # Check that -inf and NaN were filtered/handled
        assert len(cleaned_data) == 4  # 100, inf, 200, 300 remain
