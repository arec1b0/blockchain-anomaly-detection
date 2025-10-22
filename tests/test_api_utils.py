import pytest
import time
from unittest.mock import Mock, patch
from requests.exceptions import HTTPError, Timeout
from src.api.api_utils import handle_api_rate_limit, validate_response, retry_request


class TestHandleApiRateLimit:
    """Tests for handle_api_rate_limit function"""

    def test_rate_limit_with_retry_after_header(self):
        """Test that rate limiting pauses execution when status code is 429"""
        response = Mock()
        response.status_code = 429
        response.headers = {'Retry-After': '2'}

        start_time = time.time()
        handle_api_rate_limit(response)
        elapsed_time = time.time() - start_time

        assert elapsed_time >= 2, "Should wait for Retry-After duration"

    def test_rate_limit_without_retry_after_header(self):
        """Test default 60 second wait when Retry-After header is missing"""
        response = Mock()
        response.status_code = 429
        response.headers = {}

        # We'll patch time.sleep to avoid actually waiting 60 seconds
        with patch('time.sleep') as mock_sleep:
            handle_api_rate_limit(response)
            mock_sleep.assert_called_once_with(60)

    def test_no_rate_limit(self):
        """Test that no pause occurs when status code is not 429"""
        response = Mock()
        response.status_code = 200

        start_time = time.time()
        handle_api_rate_limit(response)
        elapsed_time = time.time() - start_time

        assert elapsed_time < 0.1, "Should not wait when no rate limit"


class TestValidateResponse:
    """Tests for validate_response function"""

    def test_successful_response(self):
        """Test validation of successful response"""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {'status': '1', 'result': []}

        result = validate_response(response)

        assert result == {'status': '1', 'result': []}
        response.raise_for_status.assert_called_once()

    def test_http_error_response(self):
        """Test that HTTPError is raised for error responses"""
        response = Mock()
        response.status_code = 404
        response.raise_for_status.side_effect = HTTPError("404 Not Found")

        with pytest.raises(HTTPError):
            validate_response(response)

    def test_json_decode_error(self):
        """Test that exception is raised when JSON parsing fails"""
        response = Mock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("Invalid JSON")

        with pytest.raises(ValueError):
            validate_response(response)


class TestRetryRequest:
    """Tests for retry_request function"""

    def test_successful_request_first_attempt(self):
        """Test successful request on first attempt"""
        mock_func = Mock(return_value="success")

        result = retry_request(mock_func, arg1="test", kwarg1="value")

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_on_http_error(self):
        """Test retry logic when HTTPError occurs"""
        mock_func = Mock()
        mock_func.side_effect = [
            HTTPError("500 Server Error"),
            HTTPError("500 Server Error"),
            "success"
        ]

        with patch('time.sleep'):  # Avoid actual waiting during test
            result = retry_request(mock_func)

        assert result == "success"
        assert mock_func.call_count == 3

    def test_retry_on_timeout(self):
        """Test retry logic when Timeout occurs"""
        mock_func = Mock()
        mock_func.side_effect = [
            Timeout("Request timeout"),
            "success"
        ]

        with patch('time.sleep'):
            result = retry_request(mock_func)

        assert result == "success"
        assert mock_func.call_count == 2

    def test_max_retries_exceeded(self):
        """Test that exception is raised after max retries exceeded"""
        mock_func = Mock()
        mock_func.__name__ = "test_function"  # Add __name__ attribute to Mock
        mock_func.side_effect = HTTPError("500 Server Error")

        with patch('time.sleep'):
            with pytest.raises(Exception) as exc_info:
                retry_request(mock_func)

        assert "Max retries exceeded" in str(exc_info.value)

    def test_unexpected_error_raises_immediately(self):
        """Test that unexpected errors are raised immediately without retry"""
        mock_func = Mock()
        mock_func.side_effect = ValueError("Unexpected error")

        with pytest.raises(ValueError):
            retry_request(mock_func)

        assert mock_func.call_count == 1

    def test_retry_with_args_and_kwargs(self):
        """Test that arguments are properly passed through retries"""
        mock_func = Mock()
        mock_func.side_effect = [HTTPError("Error"), "success"]

        with patch('time.sleep'):
            result = retry_request(mock_func, "arg1", "arg2", kwarg1="value1", kwarg2="value2")

        assert result == "success"
        # Verify all calls received the same arguments
        for call in mock_func.call_args_list:
            assert call[0] == ("arg1", "arg2")
            assert call[1] == {"kwarg1": "value1", "kwarg2": "value2"}
