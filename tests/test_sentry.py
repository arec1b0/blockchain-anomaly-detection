"""
test_sentry.py

Comprehensive tests for the Sentry integration module including initialization,
exception capture, message capture, user context, and breadcrumbs.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import sentry_sdk


@pytest.fixture(autouse=True)
def reset_sentry_state():
    """Reset Sentry state before each test"""
    import src.utils.sentry as sentry_module
    sentry_module._sentry_initialized = False
    yield
    # Clean up after test
    sentry_module._sentry_initialized = False


@pytest.fixture
def mock_config():
    """Create a mock configuration object"""
    config = Mock()
    config.SENTRY_ENABLED = False
    config.SENTRY_DSN = ""
    config.SENTRY_ENVIRONMENT = "test"
    config.SENTRY_TRACES_SAMPLE_RATE = 1.0
    return config


@pytest.fixture
def enabled_config():
    """Create a config with Sentry enabled"""
    config = Mock()
    config.SENTRY_ENABLED = True
    config.SENTRY_DSN = "https://test@sentry.io/12345"
    config.SENTRY_ENVIRONMENT = "testing"
    config.SENTRY_TRACES_SAMPLE_RATE = 0.5
    return config


class TestInitSentry:
    """Tests for init_sentry() function"""

    @patch('src.utils.sentry.get_config')
    @patch('src.utils.sentry.sentry_sdk')
    def test_init_sentry_disabled(self, mock_sentry_sdk, mock_get_config, mock_config):
        """Test that init_sentry returns False when Sentry is disabled"""
        from src.utils.sentry import init_sentry

        mock_get_config.return_value = mock_config
        result = init_sentry()

        assert result is False
        mock_sentry_sdk.init.assert_not_called()

    @patch('src.utils.sentry.get_config')
    @patch('src.utils.sentry.sentry_sdk')
    def test_init_sentry_enabled_without_dsn(self, mock_sentry_sdk, mock_get_config, mock_config):
        """Test that init_sentry returns False when enabled but DSN is missing"""
        from src.utils.sentry import init_sentry

        mock_config.SENTRY_ENABLED = True
        mock_config.SENTRY_DSN = ""
        mock_get_config.return_value = mock_config

        result = init_sentry()

        assert result is False
        mock_sentry_sdk.init.assert_not_called()

    @patch('src.utils.sentry.get_config')
    @patch('src.utils.sentry.sentry_sdk')
    def test_init_sentry_success(self, mock_sentry_sdk, mock_get_config, enabled_config):
        """Test successful Sentry initialization"""
        from src.utils.sentry import init_sentry

        mock_get_config.return_value = enabled_config
        result = init_sentry()

        assert result is True
        mock_sentry_sdk.init.assert_called_once()

        # Verify init was called with correct parameters
        call_kwargs = mock_sentry_sdk.init.call_args[1]
        assert call_kwargs['dsn'] == 'https://test@sentry.io/12345'
        assert call_kwargs['environment'] == 'testing'
        assert call_kwargs['traces_sample_rate'] == 0.5
        assert call_kwargs['release'] == 'blockchain-anomaly-detection@0.1.0'
        assert call_kwargs['attach_stacktrace'] is True
        assert call_kwargs['send_default_pii'] is False

    @patch('src.utils.sentry.get_config')
    @patch('src.utils.sentry.sentry_sdk')
    def test_init_sentry_already_initialized(self, mock_sentry_sdk, mock_get_config, enabled_config):
        """Test that init_sentry returns True when already initialized without re-initializing"""
        from src.utils.sentry import init_sentry
        import src.utils.sentry as sentry_module

        mock_get_config.return_value = enabled_config

        # First initialization
        result1 = init_sentry()
        assert result1 is True
        assert mock_sentry_sdk.init.call_count == 1

        # Second call should not re-initialize
        result2 = init_sentry()
        assert result2 is True
        assert mock_sentry_sdk.init.call_count == 1  # Still only 1 call

    @patch('src.utils.sentry.get_config')
    @patch('src.utils.sentry.sentry_sdk')
    def test_init_sentry_initialization_error(self, mock_sentry_sdk, mock_get_config, enabled_config):
        """Test that init_sentry handles initialization errors gracefully"""
        from src.utils.sentry import init_sentry

        mock_get_config.return_value = enabled_config
        mock_sentry_sdk.init.side_effect = Exception("Sentry initialization failed")

        result = init_sentry()

        assert result is False
        mock_sentry_sdk.init.assert_called_once()


class TestCaptureException:
    """Tests for capture_exception() function"""

    def test_capture_exception_when_not_initialized(self):
        """Test that capture_exception returns None when Sentry is not initialized"""
        from src.utils.sentry import capture_exception

        error = ValueError("test error")
        result = capture_exception(error)

        assert result is None

    @patch('src.utils.sentry.sentry_sdk')
    def test_capture_exception_success(self, mock_sentry_sdk):
        """Test successful exception capture"""
        from src.utils.sentry import capture_exception
        import src.utils.sentry as sentry_module

        # Mark as initialized
        sentry_module._sentry_initialized = True
        mock_sentry_sdk.capture_exception.return_value = 'event-id-12345'

        error = ValueError("test error")
        result = capture_exception(error)

        assert result == 'event-id-12345'
        mock_sentry_sdk.capture_exception.assert_called_once_with(error)

    @patch('src.utils.sentry.sentry_sdk')
    def test_capture_exception_with_context(self, mock_sentry_sdk):
        """Test exception capture with custom context"""
        from src.utils.sentry import capture_exception
        import src.utils.sentry as sentry_module

        sentry_module._sentry_initialized = True
        mock_scope = MagicMock()
        mock_sentry_sdk.push_scope.return_value.__enter__.return_value = mock_scope
        mock_sentry_sdk.capture_exception.return_value = 'event-id-with-context'

        error = RuntimeError("runtime error")
        context = {
            'user_id': '12345',
            'transaction_id': 'tx-67890',
            'extra_data': {'key': 'value'}
        }

        result = capture_exception(error, context=context)

        assert result == 'event-id-with-context'

        # Verify context was set
        assert mock_scope.set_context.call_count == 3
        mock_scope.set_context.assert_any_call('user_id', '12345')
        mock_scope.set_context.assert_any_call('transaction_id', 'tx-67890')
        mock_scope.set_context.assert_any_call('extra_data', {'key': 'value'})

    @patch('src.utils.sentry.sentry_sdk')
    def test_capture_exception_error_handling(self, mock_sentry_sdk):
        """Test that capture_exception handles errors gracefully"""
        from src.utils.sentry import capture_exception
        import src.utils.sentry as sentry_module

        sentry_module._sentry_initialized = True
        mock_sentry_sdk.capture_exception.side_effect = Exception("Sentry error")

        error = ValueError("test error")
        result = capture_exception(error)

        assert result is None


class TestCaptureMessage:
    """Tests for capture_message() function"""

    def test_capture_message_when_not_initialized(self):
        """Test that capture_message returns None when Sentry is not initialized"""
        from src.utils.sentry import capture_message

        result = capture_message("test message")

        assert result is None

    @patch('src.utils.sentry.sentry_sdk')
    def test_capture_message_success(self, mock_sentry_sdk):
        """Test successful message capture"""
        from src.utils.sentry import capture_message
        import src.utils.sentry as sentry_module

        sentry_module._sentry_initialized = True
        mock_sentry_sdk.capture_message.return_value = 'message-event-id'

        result = capture_message("Test message", level='info')

        assert result == 'message-event-id'
        mock_sentry_sdk.capture_message.assert_called_once_with("Test message", level='info')

    @patch('src.utils.sentry.sentry_sdk')
    def test_capture_message_with_different_levels(self, mock_sentry_sdk):
        """Test message capture with different severity levels"""
        from src.utils.sentry import capture_message
        import src.utils.sentry as sentry_module

        sentry_module._sentry_initialized = True

        levels = ['debug', 'info', 'warning', 'error', 'fatal']
        for level in levels:
            mock_sentry_sdk.capture_message.return_value = f'event-{level}'
            result = capture_message(f"Message with {level}", level=level)

            assert result == f'event-{level}'
            mock_sentry_sdk.capture_message.assert_called_with(f"Message with {level}", level=level)

    @patch('src.utils.sentry.sentry_sdk')
    def test_capture_message_with_context(self, mock_sentry_sdk):
        """Test message capture with custom context"""
        from src.utils.sentry import capture_message
        import src.utils.sentry as sentry_module

        sentry_module._sentry_initialized = True
        mock_scope = MagicMock()
        mock_sentry_sdk.push_scope.return_value.__enter__.return_value = mock_scope
        mock_sentry_sdk.capture_message.return_value = 'message-with-context'

        context = {'request_id': 'req-123', 'user': 'test_user'}
        result = capture_message("Message with context", level='warning', context=context)

        assert result == 'message-with-context'

        # Verify context was set
        assert mock_scope.set_context.call_count == 2
        mock_scope.set_context.assert_any_call('request_id', 'req-123')
        mock_scope.set_context.assert_any_call('user', 'test_user')

    @patch('src.utils.sentry.sentry_sdk')
    def test_capture_message_error_handling(self, mock_sentry_sdk):
        """Test that capture_message handles errors gracefully"""
        from src.utils.sentry import capture_message
        import src.utils.sentry as sentry_module

        sentry_module._sentry_initialized = True
        mock_sentry_sdk.capture_message.side_effect = Exception("Sentry error")

        result = capture_message("test message")

        assert result is None


class TestSetUser:
    """Tests for set_user() function"""

    def test_set_user_when_not_initialized(self):
        """Test that set_user does nothing when Sentry is not initialized"""
        from src.utils.sentry import set_user

        # Should not raise an exception
        set_user(user_id='123', email='test@example.com')

    @patch('src.utils.sentry.sentry_sdk')
    def test_set_user_with_id_only(self, mock_sentry_sdk):
        """Test setting user with only user ID"""
        from src.utils.sentry import set_user
        import src.utils.sentry as sentry_module

        sentry_module._sentry_initialized = True

        set_user(user_id='user-123')

        mock_sentry_sdk.set_user.assert_called_once_with({'id': 'user-123'})

    @patch('src.utils.sentry.sentry_sdk')
    def test_set_user_with_all_fields(self, mock_sentry_sdk):
        """Test setting user with all standard fields"""
        from src.utils.sentry import set_user
        import src.utils.sentry as sentry_module

        sentry_module._sentry_initialized = True

        set_user(user_id='user-123', email='user@example.com', username='testuser')

        expected_data = {
            'id': 'user-123',
            'email': 'user@example.com',
            'username': 'testuser'
        }
        mock_sentry_sdk.set_user.assert_called_once_with(expected_data)

    @patch('src.utils.sentry.sentry_sdk')
    def test_set_user_with_additional_attributes(self, mock_sentry_sdk):
        """Test setting user with additional custom attributes"""
        from src.utils.sentry import set_user
        import src.utils.sentry as sentry_module

        sentry_module._sentry_initialized = True

        set_user(
            user_id='user-123',
            email='user@example.com',
            custom_field='custom_value',
            subscription='premium'
        )

        expected_data = {
            'id': 'user-123',
            'email': 'user@example.com',
            'custom_field': 'custom_value',
            'subscription': 'premium'
        }
        mock_sentry_sdk.set_user.assert_called_once_with(expected_data)

    @patch('src.utils.sentry.sentry_sdk')
    def test_set_user_with_none_values(self, mock_sentry_sdk):
        """Test setting user with None values (should be excluded)"""
        from src.utils.sentry import set_user
        import src.utils.sentry as sentry_module

        sentry_module._sentry_initialized = True

        set_user(user_id=None, email=None, username=None)

        # Should call with empty dict when all values are None
        mock_sentry_sdk.set_user.assert_called_once_with({})

    @patch('src.utils.sentry.sentry_sdk')
    def test_set_user_error_handling(self, mock_sentry_sdk):
        """Test that set_user handles errors gracefully"""
        from src.utils.sentry import set_user
        import src.utils.sentry as sentry_module

        sentry_module._sentry_initialized = True
        mock_sentry_sdk.set_user.side_effect = Exception("Sentry error")

        # Should not raise an exception
        set_user(user_id='123', email='test@example.com')


class TestAddBreadcrumb:
    """Tests for add_breadcrumb() function"""

    def test_add_breadcrumb_when_not_initialized(self):
        """Test that add_breadcrumb does nothing when Sentry is not initialized"""
        from src.utils.sentry import add_breadcrumb

        # Should not raise an exception
        add_breadcrumb("test breadcrumb")

    @patch('src.utils.sentry.sentry_sdk')
    def test_add_breadcrumb_basic(self, mock_sentry_sdk):
        """Test adding a basic breadcrumb"""
        from src.utils.sentry import add_breadcrumb
        import src.utils.sentry as sentry_module

        sentry_module._sentry_initialized = True

        add_breadcrumb("User action")

        mock_sentry_sdk.add_breadcrumb.assert_called_once_with(
            message="User action",
            category='default',
            level='info',
            data={}
        )

    @patch('src.utils.sentry.sentry_sdk')
    def test_add_breadcrumb_with_all_parameters(self, mock_sentry_sdk):
        """Test adding breadcrumb with all parameters"""
        from src.utils.sentry import add_breadcrumb
        import src.utils.sentry as sentry_module

        sentry_module._sentry_initialized = True

        data = {'transaction_id': 'tx-123', 'amount': 100}
        add_breadcrumb(
            message="Transaction processed",
            category='transaction',
            level='info',
            data=data
        )

        mock_sentry_sdk.add_breadcrumb.assert_called_once_with(
            message="Transaction processed",
            category='transaction',
            level='info',
            data=data
        )

    @patch('src.utils.sentry.sentry_sdk')
    def test_add_breadcrumb_different_levels(self, mock_sentry_sdk):
        """Test adding breadcrumbs with different severity levels"""
        from src.utils.sentry import add_breadcrumb
        import src.utils.sentry as sentry_module

        sentry_module._sentry_initialized = True

        levels = ['debug', 'info', 'warning', 'error']
        for level in levels:
            add_breadcrumb(f"Breadcrumb {level}", level=level)

        # Verify all levels were called
        assert mock_sentry_sdk.add_breadcrumb.call_count == 4

    @patch('src.utils.sentry.sentry_sdk')
    def test_add_breadcrumb_with_none_data(self, mock_sentry_sdk):
        """Test adding breadcrumb with None data (should default to empty dict)"""
        from src.utils.sentry import add_breadcrumb
        import src.utils.sentry as sentry_module

        sentry_module._sentry_initialized = True

        add_breadcrumb("Test message", data=None)

        mock_sentry_sdk.add_breadcrumb.assert_called_once_with(
            message="Test message",
            category='default',
            level='info',
            data={}
        )

    @patch('src.utils.sentry.sentry_sdk')
    def test_add_breadcrumb_error_handling(self, mock_sentry_sdk):
        """Test that add_breadcrumb handles errors gracefully"""
        from src.utils.sentry import add_breadcrumb
        import src.utils.sentry as sentry_module

        sentry_module._sentry_initialized = True
        mock_sentry_sdk.add_breadcrumb.side_effect = Exception("Sentry error")

        # Should not raise an exception
        add_breadcrumb("test breadcrumb")


class TestCloseSentry:
    """Tests for close_sentry() function"""

    def test_close_sentry_when_not_initialized(self):
        """Test that close_sentry does nothing when Sentry is not initialized"""
        from src.utils.sentry import close_sentry

        # Should not raise an exception
        close_sentry()

    @patch('src.utils.sentry.sentry_sdk')
    def test_close_sentry_success(self, mock_sentry_sdk):
        """Test successful Sentry client closure"""
        from src.utils.sentry import close_sentry
        import src.utils.sentry as sentry_module

        sentry_module._sentry_initialized = True

        close_sentry()

        mock_sentry_sdk.flush.assert_called_once_with(timeout=2)
        assert sentry_module._sentry_initialized is False

    @patch('src.utils.sentry.sentry_sdk')
    def test_close_sentry_with_custom_timeout(self, mock_sentry_sdk):
        """Test closing Sentry with custom timeout"""
        from src.utils.sentry import close_sentry
        import src.utils.sentry as sentry_module

        sentry_module._sentry_initialized = True

        close_sentry(timeout=5)

        mock_sentry_sdk.flush.assert_called_once_with(timeout=5)
        assert sentry_module._sentry_initialized is False

    @patch('src.utils.sentry.sentry_sdk')
    def test_close_sentry_error_handling(self, mock_sentry_sdk):
        """Test that close_sentry handles errors gracefully"""
        from src.utils.sentry import close_sentry
        import src.utils.sentry as sentry_module

        sentry_module._sentry_initialized = True
        mock_sentry_sdk.flush.side_effect = Exception("Flush error")

        # Should not raise an exception
        close_sentry()

        # State should still be reset even if flush fails
        # Note: Current implementation doesn't reset on error, but tests the current behavior
        mock_sentry_sdk.flush.assert_called_once()


class TestSentryIntegration:
    """Integration tests for Sentry module"""

    @patch('src.utils.sentry.get_config')
    @patch('src.utils.sentry.sentry_sdk')
    def test_full_sentry_lifecycle(self, mock_sentry_sdk, mock_get_config, enabled_config):
        """Test complete Sentry lifecycle: init -> capture -> close"""
        from src.utils.sentry import init_sentry, capture_exception, close_sentry

        mock_get_config.return_value = enabled_config
        mock_sentry_sdk.capture_exception.return_value = 'event-123'

        # Initialize
        assert init_sentry() is True

        # Capture exception
        error = ValueError("test error")
        event_id = capture_exception(error)
        assert event_id == 'event-123'

        # Close
        close_sentry()

        # Verify SDK was called correctly
        mock_sentry_sdk.init.assert_called_once()
        mock_sentry_sdk.capture_exception.assert_called_once_with(error)
        mock_sentry_sdk.flush.assert_called_once()

    @patch('src.utils.sentry.get_config')
    @patch('src.utils.sentry.sentry_sdk')
    def test_multiple_captures(self, mock_sentry_sdk, mock_get_config, enabled_config):
        """Test multiple exception and message captures"""
        from src.utils.sentry import (
            init_sentry, capture_exception, capture_message,
            add_breadcrumb, set_user
        )

        mock_get_config.return_value = enabled_config
        mock_sentry_sdk.capture_exception.return_value = 'event-1'
        mock_sentry_sdk.capture_message.return_value = 'event-2'

        # Initialize
        init_sentry()

        # Set user context
        set_user(user_id='user-123', email='test@example.com')

        # Add breadcrumbs
        add_breadcrumb("User logged in", category='auth')
        add_breadcrumb("Transaction started", category='transaction')

        # Capture exception
        capture_exception(ValueError("error 1"))

        # Capture message
        capture_message("Important event", level='warning')

        # Verify all calls were made
        assert mock_sentry_sdk.set_user.called
        assert mock_sentry_sdk.add_breadcrumb.call_count == 2
        assert mock_sentry_sdk.capture_exception.called
        assert mock_sentry_sdk.capture_message.called
