"""
test_config.py

Comprehensive tests for the config module including environment variable loading,
validation, and configuration management.
"""

import pytest
import os
from unittest.mock import patch
from src.utils.config import Config, get_config


class TestConfigInitialization:
    """Tests for Config class initialization and environment variable loading"""

    def test_config_default_values(self):
        """Test that config initializes with default values when env vars are not set"""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()

            # API Configuration defaults
            assert config.API_KEY == ""
            assert config.BASE_URL == "https://api.etherscan.io/api"
            assert config.ETHERSCAN_ADDRESS == ""

            # Request Configuration defaults
            assert config.REQUEST_TIMEOUT == 10
            assert config.MAX_RETRIES == 3
            assert config.RETRY_BACKOFF == 2

            # Environment Settings defaults
            assert config.ENVIRONMENT == "development"
            assert config.LOG_LEVEL == "INFO"

            # Dask Configuration defaults
            assert config.USE_DASK is False
            assert config.DASK_N_WORKERS == 4
            assert config.DASK_THREADS_PER_WORKER == 1
            assert config.DASK_MEMORY_LIMIT == "auto"

            # Sentry Configuration defaults
            assert config.SENTRY_DSN == ""
            assert config.SENTRY_ENABLED is False
            assert config.SENTRY_ENVIRONMENT == "development"
            assert config.SENTRY_TRACES_SAMPLE_RATE == 1.0

    def test_config_loads_from_environment_variables(self):
        """Test that config correctly loads values from environment variables"""
        env_vars = {
            'ETHERSCAN_API_KEY': 'test_api_key_12345',
            'ETHERSCAN_BASE_URL': 'https://custom.api.com',
            'ETHERSCAN_ADDRESS': '0x1234567890abcdef',
            'REQUEST_TIMEOUT': '30',
            'MAX_RETRIES': '5',
            'RETRY_BACKOFF': '3',
            'ENVIRONMENT': 'production',
            'LOG_LEVEL': 'DEBUG',
            'USE_DASK': 'true',
            'DASK_N_WORKERS': '8',
            'DASK_THREADS_PER_WORKER': '2',
            'DASK_MEMORY_LIMIT': '4GB',
            'SENTRY_DSN': 'https://sentry.io/12345',
            'SENTRY_ENABLED': 'true',
            'SENTRY_ENVIRONMENT': 'staging',
            'SENTRY_TRACES_SAMPLE_RATE': '0.5',
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = Config()

            assert config.API_KEY == 'test_api_key_12345'
            assert config.BASE_URL == 'https://custom.api.com'
            assert config.ETHERSCAN_ADDRESS == '0x1234567890abcdef'
            assert config.REQUEST_TIMEOUT == 30
            assert config.MAX_RETRIES == 5
            assert config.RETRY_BACKOFF == 3
            assert config.ENVIRONMENT == 'production'
            assert config.LOG_LEVEL == 'DEBUG'
            assert config.USE_DASK is True
            assert config.DASK_N_WORKERS == 8
            assert config.DASK_THREADS_PER_WORKER == 2
            assert config.DASK_MEMORY_LIMIT == '4GB'
            assert config.SENTRY_DSN == 'https://sentry.io/12345'
            assert config.SENTRY_ENABLED is True
            assert config.SENTRY_ENVIRONMENT == 'staging'
            assert config.SENTRY_TRACES_SAMPLE_RATE == 0.5

    def test_config_boolean_parsing_variations(self):
        """Test that boolean environment variables are parsed correctly"""
        # Test USE_DASK variations
        for true_value in ['true', 'True', 'TRUE', 'TrUe']:
            with patch.dict(os.environ, {'USE_DASK': true_value}, clear=True):
                config = Config()
                assert config.USE_DASK is True

        for false_value in ['false', 'False', 'FALSE', '', '0', 'no']:
            with patch.dict(os.environ, {'USE_DASK': false_value}, clear=True):
                config = Config()
                assert config.USE_DASK is False

        # Test SENTRY_ENABLED variations
        for true_value in ['true', 'True', 'TRUE']:
            with patch.dict(os.environ, {'SENTRY_ENABLED': true_value}, clear=True):
                config = Config()
                assert config.SENTRY_ENABLED is True

    def test_config_integer_parsing(self):
        """Test that integer environment variables are parsed correctly"""
        with patch.dict(os.environ, {
            'REQUEST_TIMEOUT': '100',
            'MAX_RETRIES': '10',
            'RETRY_BACKOFF': '5',
            'DASK_N_WORKERS': '16',
            'DASK_THREADS_PER_WORKER': '4',
        }, clear=True):
            config = Config()
            assert config.REQUEST_TIMEOUT == 100
            assert config.MAX_RETRIES == 10
            assert config.RETRY_BACKOFF == 5
            assert config.DASK_N_WORKERS == 16
            assert config.DASK_THREADS_PER_WORKER == 4

    def test_config_float_parsing(self):
        """Test that float environment variables are parsed correctly"""
        test_cases = [
            ('0.0', 0.0),
            ('0.25', 0.25),
            ('0.5', 0.5),
            ('1.0', 1.0),
            ('1.5', 1.5),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {'SENTRY_TRACES_SAMPLE_RATE': env_value}, clear=True):
                config = Config()
                assert config.SENTRY_TRACES_SAMPLE_RATE == expected

    def test_config_sentry_environment_inherits_from_environment(self):
        """Test that SENTRY_ENVIRONMENT defaults to ENVIRONMENT value"""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}, clear=True):
            config = Config()
            assert config.SENTRY_ENVIRONMENT == 'production'


class TestConfigValidation:
    """Tests for Config.validate() method"""

    def test_validate_success_with_api_key(self):
        """Test validation succeeds when API_KEY is set"""
        with patch.dict(os.environ, {'ETHERSCAN_API_KEY': 'valid_key'}, clear=True):
            config = Config()
            assert config.validate() is True

    def test_validate_fails_without_api_key(self):
        """Test validation fails when API_KEY is missing"""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            with pytest.raises(ValueError) as exc_info:
                config.validate()
            assert "ETHERSCAN_API_KEY is required" in str(exc_info.value)

    def test_validate_fails_with_invalid_timeout(self):
        """Test validation fails when REQUEST_TIMEOUT is non-positive"""
        with patch.dict(os.environ, {
            'ETHERSCAN_API_KEY': 'valid_key',
            'REQUEST_TIMEOUT': '0'
        }, clear=True):
            config = Config()
            with pytest.raises(ValueError) as exc_info:
                config.validate()
            assert "REQUEST_TIMEOUT must be positive" in str(exc_info.value)

        with patch.dict(os.environ, {
            'ETHERSCAN_API_KEY': 'valid_key',
            'REQUEST_TIMEOUT': '-1'
        }, clear=True):
            config = Config()
            with pytest.raises(ValueError) as exc_info:
                config.validate()
            assert "REQUEST_TIMEOUT must be positive" in str(exc_info.value)

    def test_validate_fails_with_negative_retries(self):
        """Test validation fails when MAX_RETRIES is negative"""
        with patch.dict(os.environ, {
            'ETHERSCAN_API_KEY': 'valid_key',
            'MAX_RETRIES': '-1'
        }, clear=True):
            config = Config()
            with pytest.raises(ValueError) as exc_info:
                config.validate()
            assert "MAX_RETRIES must be non-negative" in str(exc_info.value)

    def test_validate_allows_zero_retries(self):
        """Test validation succeeds with MAX_RETRIES=0"""
        with patch.dict(os.environ, {
            'ETHERSCAN_API_KEY': 'valid_key',
            'MAX_RETRIES': '0'
        }, clear=True):
            config = Config()
            assert config.validate() is True

    def test_validate_fails_with_invalid_backoff(self):
        """Test validation fails when RETRY_BACKOFF is non-positive"""
        with patch.dict(os.environ, {
            'ETHERSCAN_API_KEY': 'valid_key',
            'RETRY_BACKOFF': '0'
        }, clear=True):
            config = Config()
            with pytest.raises(ValueError) as exc_info:
                config.validate()
            assert "RETRY_BACKOFF must be positive" in str(exc_info.value)

        with patch.dict(os.environ, {
            'ETHERSCAN_API_KEY': 'valid_key',
            'RETRY_BACKOFF': '-2'
        }, clear=True):
            config = Config()
            with pytest.raises(ValueError) as exc_info:
                config.validate()
            assert "RETRY_BACKOFF must be positive" in str(exc_info.value)

    def test_validate_fails_when_sentry_enabled_without_dsn(self):
        """Test validation fails when Sentry is enabled but DSN is not set"""
        with patch.dict(os.environ, {
            'ETHERSCAN_API_KEY': 'valid_key',
            'SENTRY_ENABLED': 'true',
            'SENTRY_DSN': ''
        }, clear=True):
            config = Config()
            with pytest.raises(ValueError) as exc_info:
                config.validate()
            assert "SENTRY_DSN is required when SENTRY_ENABLED is true" in str(exc_info.value)

    def test_validate_succeeds_when_sentry_enabled_with_dsn(self):
        """Test validation succeeds when Sentry is enabled with DSN"""
        with patch.dict(os.environ, {
            'ETHERSCAN_API_KEY': 'valid_key',
            'SENTRY_ENABLED': 'true',
            'SENTRY_DSN': 'https://sentry.io/12345'
        }, clear=True):
            config = Config()
            assert config.validate() is True

    def test_validate_succeeds_when_sentry_disabled_without_dsn(self):
        """Test validation succeeds when Sentry is disabled even without DSN"""
        with patch.dict(os.environ, {
            'ETHERSCAN_API_KEY': 'valid_key',
            'SENTRY_ENABLED': 'false',
            'SENTRY_DSN': ''
        }, clear=True):
            config = Config()
            assert config.validate() is True

    def test_validate_multiple_errors(self):
        """Test validation collects and reports multiple errors"""
        with patch.dict(os.environ, {
            'ETHERSCAN_API_KEY': '',  # Missing
            'REQUEST_TIMEOUT': '0',   # Invalid
            'MAX_RETRIES': '-1',      # Invalid
            'RETRY_BACKOFF': '-5',    # Invalid
            'SENTRY_ENABLED': 'true',
            'SENTRY_DSN': ''          # Missing when enabled
        }, clear=True):
            config = Config()
            with pytest.raises(ValueError) as exc_info:
                config.validate()

            error_message = str(exc_info.value)
            assert "ETHERSCAN_API_KEY is required" in error_message
            assert "REQUEST_TIMEOUT must be positive" in error_message
            assert "MAX_RETRIES must be non-negative" in error_message
            assert "RETRY_BACKOFF must be positive" in error_message
            assert "SENTRY_DSN is required" in error_message


class TestConfigToDict:
    """Tests for Config.to_dict() method"""

    def test_to_dict_returns_all_config_values(self):
        """Test that to_dict returns all configuration values"""
        with patch.dict(os.environ, {
            'ETHERSCAN_API_KEY': 'test_key',
            'REQUEST_TIMEOUT': '20',
        }, clear=True):
            config = Config()
            config_dict = config.to_dict()

            assert isinstance(config_dict, dict)
            assert 'API_KEY' in config_dict
            assert config_dict['API_KEY'] == 'test_key'
            assert 'REQUEST_TIMEOUT' in config_dict
            assert config_dict['REQUEST_TIMEOUT'] == 20

    def test_to_dict_excludes_private_attributes(self):
        """Test that to_dict excludes private attributes starting with underscore"""
        config = Config()
        config._private_attr = "should_not_appear"
        config_dict = config.to_dict()

        assert '_private_attr' not in config_dict

    def test_to_dict_contains_expected_keys(self):
        """Test that to_dict contains all expected configuration keys"""
        config = Config()
        config_dict = config.to_dict()

        expected_keys = [
            'API_KEY', 'BASE_URL', 'ETHERSCAN_ADDRESS',
            'REQUEST_TIMEOUT', 'MAX_RETRIES', 'RETRY_BACKOFF',
            'ENVIRONMENT', 'LOG_LEVEL',
            'USE_DASK', 'DASK_N_WORKERS', 'DASK_THREADS_PER_WORKER', 'DASK_MEMORY_LIMIT',
            'SENTRY_DSN', 'SENTRY_ENABLED', 'SENTRY_ENVIRONMENT', 'SENTRY_TRACES_SAMPLE_RATE'
        ]

        for key in expected_keys:
            assert key in config_dict


class TestGetConfig:
    """Tests for get_config() function"""

    def test_get_config_returns_config_instance(self):
        """Test that get_config returns a Config instance"""
        config = get_config()
        assert isinstance(config, Config)

    def test_get_config_returns_singleton(self):
        """Test that get_config returns the same instance on multiple calls"""
        # Clear the global config instance first
        import src.utils.config as config_module
        config_module._config = None

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_get_config_creates_instance_on_first_call(self):
        """Test that get_config creates instance on first call"""
        import src.utils.config as config_module
        config_module._config = None

        assert config_module._config is None
        config = get_config()
        assert config is not None
        assert config_module._config is config


class TestConfigEdgeCases:
    """Tests for edge cases and error handling"""

    def test_config_with_empty_string_values(self):
        """Test config handles empty string environment variables"""
        with patch.dict(os.environ, {
            'ETHERSCAN_API_KEY': '',
            'ETHERSCAN_BASE_URL': '',
        }, clear=True):
            config = Config()
            assert config.API_KEY == ''
            assert config.BASE_URL == ''

    def test_config_with_whitespace_values(self):
        """Test config handles whitespace in environment variables"""
        with patch.dict(os.environ, {
            'ETHERSCAN_API_KEY': '  test_key  ',
        }, clear=True):
            config = Config()
            # Note: os.getenv doesn't strip whitespace by default
            assert config.API_KEY == '  test_key  '

    def test_config_numeric_parsing_with_invalid_values(self):
        """Test that invalid numeric values raise ValueError during initialization"""
        # Invalid integer
        with patch.dict(os.environ, {'REQUEST_TIMEOUT': 'not_a_number'}, clear=True):
            with pytest.raises(ValueError):
                Config()

        # Invalid float
        with patch.dict(os.environ, {'SENTRY_TRACES_SAMPLE_RATE': 'invalid'}, clear=True):
            with pytest.raises(ValueError):
                Config()

    def test_config_with_special_characters(self):
        """Test config handles special characters in values"""
        special_key = 'test!@#$%^&*()_+-=[]{}|;:,.<>?'
        with patch.dict(os.environ, {'ETHERSCAN_API_KEY': special_key}, clear=True):
            config = Config()
            assert config.API_KEY == special_key

    def test_config_with_very_large_numbers(self):
        """Test config handles very large numbers"""
        with patch.dict(os.environ, {
            'ETHERSCAN_API_KEY': 'valid_key',
            'REQUEST_TIMEOUT': '999999',
            'DASK_N_WORKERS': '1000',
        }, clear=True):
            config = Config()
            assert config.REQUEST_TIMEOUT == 999999
            assert config.DASK_N_WORKERS == 1000


class TestBackwardCompatibility:
    """Tests for backward compatibility with module-level variables"""

    def test_module_level_variables_exist(self):
        """Test that module-level variables are exposed for backward compatibility"""
        from src.utils.config import (
            API_KEY, BASE_URL, REQUEST_TIMEOUT, MAX_RETRIES,
            RETRY_BACKOFF, ENVIRONMENT, LOG_LEVEL
        )

        # These should be defined (even if empty/default values)
        assert API_KEY is not None or API_KEY == ""
        assert BASE_URL is not None
        assert REQUEST_TIMEOUT is not None
        assert MAX_RETRIES is not None
        assert RETRY_BACKOFF is not None
        assert ENVIRONMENT is not None
        assert LOG_LEVEL is not None
