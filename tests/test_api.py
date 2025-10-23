import pytest
import os
from src.api.etherscan_api import EtherscanAPI
from unittest.mock import patch


@pytest.fixture
def api():
    api_key = "dummy_key"
    return EtherscanAPI(api_key=api_key)


@patch('src.api.etherscan_api.requests.get')
def test_get_transactions(mock_get, api):
    # Mock the API response
    mock_get.return_value.json.return_value = {
        "status": "1",
        "message": "OK",
        "result": [
            {'value': '100', 'gas': '21000', 'gasPrice': '20'},
            {'value': '200', 'gas': '22000', 'gasPrice': '25'},
        ]
    }
    mock_get.return_value.status_code = 200

    address = "0xde0b295669a9fd93d5f28d9ec85e40f4cb697bae"
    transactions = api.get_transactions(address)

    assert transactions is not None, "API call failed to retrieve transactions."
    assert isinstance(transactions, list), "API result is not a list."
    assert len(transactions) > 0, "No transactions retrieved."
