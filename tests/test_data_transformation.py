import pytest
import pandas as pd
import numpy as np
from src.data_processing.data_transformation import DataTransformer


@pytest.fixture
def sample_data():
    """Create sample transaction data for testing"""
    data = {
        'timeStamp': [1609459200, 1609545600, 1609632000, 1609718400],  # UNIX timestamps
        'value': [100, 200, 300, 400],
        'gas': [21000, 21000, 21000, 21000],
        'gasPrice': [50, 60, 70, 80]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_invalid_timestamp():
    """Create sample data with invalid timestamps"""
    data = {
        'timeStamp': [1609459200, 'invalid', 1609632000],
        'value': [100, 200, 300],
    }
    return pd.DataFrame(data)


class TestDataTransformer:
    """Tests for DataTransformer class"""

    def test_init(self, sample_data):
        """Test DataTransformer initialization"""
        transformer = DataTransformer(sample_data)
        assert transformer.df is not None
        assert len(transformer.df) == 4

    def test_convert_timestamp(self, sample_data):
        """Test conversion of UNIX timestamps to datetime"""
        transformer = DataTransformer(sample_data)
        result = transformer.convert_timestamp()

        # Check that the column is now datetime type
        assert pd.api.types.is_datetime64_any_dtype(result['timeStamp'])

        # Check specific values
        expected_dates = pd.Series(pd.to_datetime([1609459200, 1609545600, 1609632000, 1609718400], unit='s'))
        expected_dates.index = result['timeStamp'].index  # Match the index
        pd.testing.assert_series_equal(
            result['timeStamp'],
            expected_dates,
            check_names=False
        )

    def test_convert_timestamp_with_invalid_values(self, sample_data_with_invalid_timestamp):
        """Test that invalid timestamps are handled with coerce"""
        transformer = DataTransformer(sample_data_with_invalid_timestamp)
        result = transformer.convert_timestamp()

        # Check that invalid values are converted to NaT
        assert pd.api.types.is_datetime64_any_dtype(result['timeStamp'])
        assert pd.isna(result['timeStamp'].iloc[1])  # Invalid value should be NaT

    def test_normalize_column(self, sample_data):
        """Test min-max normalization of a column"""
        transformer = DataTransformer(sample_data)
        result = transformer.normalize_column('value')

        # Check that normalization is correct (min=0, max=1)
        assert result['value'].min() == 0.0
        assert result['value'].max() == 1.0

        # Check specific normalized values
        # Original: [100, 200, 300, 400] -> Normalized: [0, 0.333, 0.667, 1.0]
        expected = np.array([0.0, 0.333333, 0.666667, 1.0])
        np.testing.assert_array_almost_equal(result['value'].values, expected, decimal=5)

    def test_normalize_column_with_single_value(self):
        """Test normalization when all values are the same"""
        data = pd.DataFrame({'value': [100, 100, 100]})
        transformer = DataTransformer(data)

        # When min == max, the implementation sets all values to 0
        result = transformer.normalize_column('value')
        assert all(result['value'] == 0)

    def test_normalize_column_nonexistent(self, sample_data):
        """Test that ValueError is raised for nonexistent column"""
        transformer = DataTransformer(sample_data)

        with pytest.raises(ValueError) as exc_info:
            transformer.normalize_column('nonexistent_column')

        assert "not found in DataFrame" in str(exc_info.value)

    def test_normalize_multiple_columns(self, sample_data):
        """Test normalization of multiple columns"""
        transformer = DataTransformer(sample_data)
        transformer.normalize_column('value')
        result = transformer.normalize_column('gasPrice')

        # Both columns should be normalized
        assert result['value'].min() == 0.0
        assert result['value'].max() == 1.0
        assert result['gasPrice'].min() == 0.0
        assert result['gasPrice'].max() == 1.0

    def test_transform_data(self, sample_data):
        """Test complete transformation pipeline"""
        transformer = DataTransformer(sample_data)
        result = transformer.transform_data()

        # Check timestamp conversion
        assert pd.api.types.is_datetime64_any_dtype(result['timeStamp'])

        # Check value normalization (transform_data normalizes 'value' column)
        assert result['value'].min() == 0.0
        assert result['value'].max() == 1.0

    def test_transform_data_maintains_other_columns(self, sample_data):
        """Test that transformation doesn't affect non-transformed columns"""
        transformer = DataTransformer(sample_data)
        original_gas = sample_data['gas'].copy()
        result = transformer.transform_data()

        # Gas column should remain unchanged
        pd.testing.assert_series_equal(result['gas'], original_gas, check_names=False)

    def test_chaining_operations(self, sample_data):
        """Test that operations can be chained"""
        transformer = DataTransformer(sample_data)

        # Chain operations by calling them sequentially
        transformer.convert_timestamp()
        transformer.normalize_column('value')
        result = transformer.normalize_column('gasPrice')

        # Verify all transformations applied
        assert pd.api.types.is_datetime64_any_dtype(result['timeStamp'])
        assert result['value'].min() == 0.0
        assert result['gasPrice'].min() == 0.0


class TestDataTransformerInitialization:
    """Tests for DataTransformer initialization and validation"""

    def test_init_with_none_dataframe(self):
        """Test that DataTransformer raises ValueError when df is None"""
        with pytest.raises(ValueError) as exc_info:
            DataTransformer(None)
        assert "DataFrame cannot be None" in str(exc_info.value)

    def test_init_with_empty_dataframe(self):
        """Test that DataTransformer raises ValueError when df is empty"""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError) as exc_info:
            DataTransformer(empty_df)
        assert "DataFrame cannot be empty" in str(exc_info.value)

    def test_init_with_wrong_type(self):
        """Test that DataTransformer raises TypeError when df is not a DataFrame"""
        with pytest.raises(TypeError) as exc_info:
            DataTransformer([1, 2, 3])
        assert "Expected pandas DataFrame" in str(exc_info.value)

        with pytest.raises(TypeError) as exc_info:
            DataTransformer("not a dataframe")
        assert "Expected pandas DataFrame" in str(exc_info.value)

        with pytest.raises(TypeError) as exc_info:
            DataTransformer({'key': 'value'})
        assert "Expected pandas DataFrame" in str(exc_info.value)

    def test_init_with_valid_dataframe(self, sample_data):
        """Test successful initialization with valid DataFrame"""
        transformer = DataTransformer(sample_data)
        assert transformer.df is not None
        assert isinstance(transformer.df, pd.DataFrame)
        assert len(transformer.df) == len(sample_data)

    def test_init_creates_copy_of_dataframe(self, sample_data):
        """Test that DataTransformer creates a copy of the input DataFrame"""
        transformer = DataTransformer(sample_data)
        assert transformer.df is not sample_data
        # Modify transformer's df
        transformer.df.loc[0, 'value'] = 999
        # Original should be unchanged
        assert sample_data.loc[0, 'value'] != 999


class TestConvertTimestamp:
    """Tests for convert_timestamp method"""

    def test_convert_timestamp_missing_column(self):
        """Test that ValueError is raised when timestamp column is missing"""
        df = pd.DataFrame({
            'value': [100, 200, 300],
            'gas': [21000, 21000, 21000]
        })
        transformer = DataTransformer(df)

        with pytest.raises(ValueError) as exc_info:
            transformer.convert_timestamp()
        assert "not found in DataFrame" in str(exc_info.value)

    def test_convert_timestamp_custom_column_name(self):
        """Test conversion with custom column name"""
        df = pd.DataFrame({
            'custom_time': [1609459200, 1609545600, 1609632000],
            'value': [100, 200, 300]
        })
        transformer = DataTransformer(df)
        result = transformer.convert_timestamp(column_name='custom_time')

        assert pd.api.types.is_datetime64_any_dtype(result['custom_time'])

    def test_convert_timestamp_missing_custom_column(self):
        """Test error when custom column name doesn't exist"""
        df = pd.DataFrame({
            'timeStamp': [1609459200, 1609545600],
            'value': [100, 200]
        })
        transformer = DataTransformer(df)

        with pytest.raises(ValueError) as exc_info:
            transformer.convert_timestamp(column_name='wrong_column')
        assert "not found in DataFrame" in str(exc_info.value)

    def test_convert_timestamp_all_invalid(self):
        """Test conversion when all timestamps are invalid"""
        df = pd.DataFrame({
            'timeStamp': ['invalid1', 'invalid2', 'invalid3'],
            'value': [100, 200, 300]
        })
        transformer = DataTransformer(df)
        result = transformer.convert_timestamp()

        # All should be NaT
        assert result['timeStamp'].isna().sum() == 3

    def test_convert_timestamp_error_handling(self):
        """Test error handling in convert_timestamp"""
        df = pd.DataFrame({'value': [1, 2, 3]})
        transformer = DataTransformer(df)

        with pytest.raises(RuntimeError) as exc_info:
            transformer.df = None
            transformer.convert_timestamp()
        assert "Failed to convert timestamps" in str(exc_info.value)


class TestNormalizeColumn:
    """Tests for normalize_column method"""

    def test_normalize_column_empty_name(self):
        """Test that ValueError is raised when column name is empty"""
        df = pd.DataFrame({'value': [100, 200, 300]})
        transformer = DataTransformer(df)

        with pytest.raises(ValueError) as exc_info:
            transformer.normalize_column('')
        assert "Column name cannot be empty" in str(exc_info.value)

    def test_normalize_column_none_name(self):
        """Test that ValueError is raised when column name is None"""
        df = pd.DataFrame({'value': [100, 200, 300]})
        transformer = DataTransformer(df)

        with pytest.raises(ValueError) as exc_info:
            transformer.normalize_column(None)
        assert "Column name cannot be empty" in str(exc_info.value)

    def test_normalize_column_non_numeric(self):
        """Test normalization of non-numeric column"""
        df = pd.DataFrame({
            'text': ['a', 'b', 'c'],
            'value': [100, 200, 300]
        })
        transformer = DataTransformer(df)

        with pytest.raises(ValueError) as exc_info:
            transformer.normalize_column('text')
        assert "contains no numeric data" in str(exc_info.value)

    def test_normalize_column_with_nan_values(self):
        """Test normalization with NaN values"""
        df = pd.DataFrame({
            'value': [100, np.nan, 200, 300],
            'gas': [21000, 21000, 21000, 21000]
        })
        transformer = DataTransformer(df)
        result = transformer.normalize_column('value')

        # NaN should be preserved in normalized output
        assert result['value'].isna().sum() == 1
        # Other values should be normalized
        assert result['value'].min() >= 0.0
        assert result['value'].max() <= 1.0

    def test_normalize_column_with_negative_values(self):
        """Test normalization with negative values"""
        df = pd.DataFrame({
            'value': [-100, 0, 100, 200],
            'gas': [21000, 21000, 21000, 21000]
        })
        transformer = DataTransformer(df)
        result = transformer.normalize_column('value')

        # Should still normalize to [0, 1]
        assert result['value'].min() == 0.0
        assert result['value'].max() == 1.0

    def test_normalize_column_with_mixed_types(self):
        """Test normalization when column has mixed types"""
        df = pd.DataFrame({
            'value': ['100', '200', 'text', '300'],
            'gas': [21000, 21000, 21000, 21000]
        })
        transformer = DataTransformer(df)
        result = transformer.normalize_column('value')

        # Non-numeric values should be converted to NaN
        assert result['value'].isna().sum() >= 1

    def test_normalize_column_error_handling(self):
        """Test error handling in normalize_column"""
        df = pd.DataFrame({'value': [1, 2, 3]})
        transformer = DataTransformer(df)

        with pytest.raises(RuntimeError) as exc_info:
            transformer.df = None
            transformer.normalize_column('value')
        assert "Failed to normalize column" in str(exc_info.value)


class TestTransformData:
    """Tests for transform_data method (full pipeline)"""

    def test_transform_data_without_value_column(self):
        """Test transform_data when value column is missing"""
        df = pd.DataFrame({
            'timeStamp': [1609459200, 1609545600],
            'gas': [21000, 21000]
        })
        transformer = DataTransformer(df)
        result = transformer.transform_data()

        # Should still convert timestamp
        assert pd.api.types.is_datetime64_any_dtype(result['timeStamp'])
        # Should not fail, just skip normalization

    def test_transform_data_without_timestamp_column(self):
        """Test transform_data when timestamp column is missing"""
        df = pd.DataFrame({
            'value': [100, 200, 300],
            'gas': [21000, 21000, 21000]
        })
        transformer = DataTransformer(df)

        # Should fail because convert_timestamp requires timeStamp column
        with pytest.raises(RuntimeError) as exc_info:
            transformer.transform_data()
        assert "Data transformation failed" in str(exc_info.value)

    def test_transform_data_error_handling(self):
        """Test error handling in transform_data pipeline"""
        df = pd.DataFrame({'other': [1, 2, 3]})
        transformer = DataTransformer(df)

        with pytest.raises(RuntimeError) as exc_info:
            transformer.transform_data()
        assert "Data transformation failed" in str(exc_info.value)


class TestDataTransformerEdgeCases:
    """Tests for edge cases and special scenarios"""

    def test_single_row_dataframe(self):
        """Test transformation of a DataFrame with only one row"""
        df = pd.DataFrame({
            'timeStamp': [1609459200],
            'value': [100]
        })
        transformer = DataTransformer(df)
        result = transformer.transform_data()

        assert len(result) == 1
        assert pd.api.types.is_datetime64_any_dtype(result['timeStamp'])
        # Single value should normalize to 0
        assert result['value'].iloc[0] == 0

    def test_large_dataframe(self):
        """Test transformation of a large DataFrame"""
        df = pd.DataFrame({
            'timeStamp': [1609459200 + i for i in range(10000)],
            'value': list(range(1, 10001)),
            'gas': [21000] * 10000
        })
        transformer = DataTransformer(df)
        result = transformer.transform_data()

        assert len(result) == 10000
        assert pd.api.types.is_datetime64_any_dtype(result['timeStamp'])
        assert result['value'].min() == 0.0
        assert result['value'].max() == 1.0

    def test_dataframe_with_special_timestamp_values(self):
        """Test handling of special timestamp values"""
        df = pd.DataFrame({
            'timeStamp': [0, 1609459200, 2147483647],  # epoch, normal, max 32-bit
            'value': [100, 200, 300]
        })
        transformer = DataTransformer(df)
        result = transformer.convert_timestamp()

        assert pd.api.types.is_datetime64_any_dtype(result['timeStamp'])
        # All should be valid datetime objects
        assert result['timeStamp'].notna().all()
