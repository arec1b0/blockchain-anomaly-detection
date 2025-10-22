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

        # When min == max, division by zero occurs, resulting in NaN
        # This is expected behavior for min-max normalization
        result = transformer.normalize_column('value')
        assert np.all(np.isnan(result['value']))

    def test_normalize_column_nonexistent(self, sample_data):
        """Test that KeyError is raised for nonexistent column"""
        transformer = DataTransformer(sample_data)

        with pytest.raises(KeyError) as exc_info:
            transformer.normalize_column('nonexistent_column')

        assert "nonexistent_column" in str(exc_info.value)

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
