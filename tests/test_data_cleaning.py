import pytest
import pandas as pd
from src.data_processing.data_cleaning import DataCleaner


@pytest.fixture
def sample_data():
    data = {
        'value': [100, 200, None, 0, 500, 600, 700],
        'gas': [21000, 21000, 21000, None, 21000, 21000, 21000],
        'gasPrice': [50, 50, 50, 50, 50, 50, None]
    }
    return pd.DataFrame(data)


def test_remove_duplicates(sample_data):
    cleaner = DataCleaner(sample_data)
    cleaned_data = cleaner.remove_duplicates()

    assert len(cleaned_data) == len(sample_data), "Duplicate removal failed."


def test_handle_missing_values(sample_data):
    cleaner = DataCleaner(sample_data)
    cleaned_data = cleaner.handle_missing_values()

    assert cleaned_data.isna().sum().sum() == 0, "Missing value handling failed."


def test_filter_invalid_transactions(sample_data):
    cleaner = DataCleaner(sample_data)
    cleaned_data = cleaner.filter_invalid_transactions()

    assert all(cleaned_data['value'] > 0), "Invalid transactions not filtered correctly."
