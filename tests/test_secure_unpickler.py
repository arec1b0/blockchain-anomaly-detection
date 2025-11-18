"""
Tests for secure model deserialization.
"""

import os
import pickle
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from sklearn.ensemble import IsolationForest
import numpy as np

from src.ml.security.secure_unpickler import (
    SecureModelLoader,
    RestrictedUnpickler,
    ModelIntegrityError
)


class TestRestrictedUnpickler:
    """Test RestrictedUnpickler class."""

    def test_allows_safe_sklearn_model(self):
        """Test that safe sklearn models can be unpickled."""
        # Create a simple IsolationForest model
        model = IsolationForest(n_estimators=10, random_state=42)
        X = np.random.randn(100, 5)
        model.fit(X)

        # Pickle and unpickle with RestrictedUnpickler
        import io
        buffer = io.BytesIO()
        pickle.dump(model, buffer)
        buffer.seek(0)

        unpickler = RestrictedUnpickler(buffer)
        loaded_model = unpickler.load()

        assert isinstance(loaded_model, IsolationForest)
        assert loaded_model.n_estimators == 10

    def test_allows_numpy_arrays(self):
        """Test that numpy arrays can be unpickled."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])

        import io
        buffer = io.BytesIO()
        pickle.dump(arr, buffer)
        buffer.seek(0)

        unpickler = RestrictedUnpickler(buffer)
        loaded_arr = unpickler.load()

        assert np.array_equal(loaded_arr, arr)

    def test_blocks_unsafe_classes(self):
        """Test that unsafe classes are blocked."""
        # Create a malicious class
        class MaliciousClass:
            def __reduce__(self):
                import os
                return (os.system, ('echo pwned',))

        obj = MaliciousClass()

        import io
        buffer = io.BytesIO()
        pickle.dump(obj, buffer)
        buffer.seek(0)

        unpickler = RestrictedUnpickler(buffer)

        with pytest.raises(pickle.UnpicklingError) as exc_info:
            unpickler.load()

        assert "not allowed" in str(exc_info.value).lower()

    def test_add_safe_class(self):
        """Test adding custom safe classes."""
        # Add a custom class to whitelist
        SecureModelLoader.add_safe_class('__main__', 'CustomModel')

        # Verify it was added
        assert '__main__.CustomModel' in RestrictedUnpickler.SAFE_CLASSES

    def test_add_safe_module(self):
        """Test adding custom safe modules."""
        # Add a custom module to whitelist
        SecureModelLoader.add_safe_module('mypackage')

        # Verify it was added
        assert 'mypackage' in RestrictedUnpickler.SAFE_MODULES


class TestSecureModelLoader:
    """Test SecureModelLoader class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing."""
        model = IsolationForest(n_estimators=10, random_state=42)
        X = np.random.randn(100, 5)
        model.fit(X)
        return model

    @pytest.fixture
    def sample_model_file(self, temp_dir, sample_model):
        """Create a sample model file."""
        model_file = os.path.join(temp_dir, 'model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(sample_model, f)
        return model_file

    def test_load_model_without_verification(self, sample_model_file):
        """Test loading model with verification skipped."""
        loader = SecureModelLoader()

        model = loader.load_model(
            model_file=sample_model_file,
            skip_verification=True
        )

        assert isinstance(model, IsolationForest)

    def test_calculate_checksum(self, sample_model_file):
        """Test checksum calculation."""
        loader = SecureModelLoader()
        checksum = loader._calculate_checksum(sample_model_file)

        # Checksum should be 64 character hex string (SHA256)
        assert len(checksum) == 64
        assert all(c in '0123456789abcdef' for c in checksum)

    def test_verify_checksum_valid(self, sample_model_file):
        """Test checksum verification with valid checksum."""
        loader = SecureModelLoader()
        expected_checksum = loader._calculate_checksum(sample_model_file)

        # Should not raise exception
        result = loader._verify_checksum(sample_model_file, expected_checksum)
        assert result is True

    def test_verify_checksum_invalid(self, sample_model_file):
        """Test checksum verification with invalid checksum."""
        loader = SecureModelLoader()
        invalid_checksum = 'a' * 64  # Wrong checksum

        with pytest.raises(ModelIntegrityError) as exc_info:
            loader._verify_checksum(sample_model_file, invalid_checksum)

        assert "checksum verification failed" in str(exc_info.value).lower()

    @patch.dict(os.environ, {'MODEL_VERIFY_SIGNATURE': 'true', 'MODEL_SIGNATURE_KEY': 'test-secret-key'})
    def test_calculate_signature(self, sample_model_file):
        """Test HMAC signature calculation."""
        loader = SecureModelLoader()

        with open(sample_model_file, 'rb') as f:
            data = f.read()

        signature = loader._calculate_signature(data)

        # Signature should be 64 character hex string (HMAC-SHA256)
        assert len(signature) == 64
        assert all(c in '0123456789abcdef' for c in signature)

    @patch.dict(os.environ, {'MODEL_VERIFY_SIGNATURE': 'true', 'MODEL_SIGNATURE_KEY': 'test-secret-key'})
    def test_verify_signature_valid(self, sample_model_file):
        """Test signature verification with valid signature."""
        loader = SecureModelLoader()

        with open(sample_model_file, 'rb') as f:
            data = f.read()

        expected_signature = loader._calculate_signature(data)

        # Should not raise exception
        result = loader._verify_signature(sample_model_file, expected_signature)
        assert result is True

    @patch.dict(os.environ, {'MODEL_VERIFY_SIGNATURE': 'true', 'MODEL_SIGNATURE_KEY': 'test-secret-key'})
    def test_verify_signature_invalid(self, sample_model_file):
        """Test signature verification with invalid signature."""
        loader = SecureModelLoader()
        invalid_signature = 'a' * 64  # Wrong signature

        with pytest.raises(ModelIntegrityError) as exc_info:
            loader._verify_signature(sample_model_file, invalid_signature)

        assert "signature verification failed" in str(exc_info.value).lower()

    @patch.dict(os.environ, {'MODEL_VERIFY_CHECKSUM': 'true'})
    def test_load_model_with_checksum_verification(self, temp_dir, sample_model_file):
        """Test loading model with checksum verification."""
        loader = SecureModelLoader()

        # Generate metadata with checksum
        metadata = loader.generate_metadata(sample_model_file)
        metadata_file = os.path.join(temp_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        # Load model with verification
        model = loader.load_model(
            model_file=sample_model_file,
            metadata_file=metadata_file
        )

        assert isinstance(model, IsolationForest)

    @patch.dict(os.environ, {'MODEL_VERIFY_CHECKSUM': 'true'})
    def test_load_model_with_invalid_checksum(self, temp_dir, sample_model_file):
        """Test loading model with invalid checksum."""
        loader = SecureModelLoader()

        # Create metadata with wrong checksum
        metadata = {'checksum': 'a' * 64}
        metadata_file = os.path.join(temp_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        # Should raise ModelIntegrityError
        with pytest.raises(ModelIntegrityError):
            loader.load_model(
                model_file=sample_model_file,
                metadata_file=metadata_file
            )

    @patch.dict(os.environ, {
        'MODEL_VERIFY_CHECKSUM': 'true',
        'MODEL_VERIFY_SIGNATURE': 'true',
        'MODEL_SIGNATURE_KEY': 'test-secret-key'
    })
    def test_load_model_with_full_verification(self, temp_dir, sample_model_file):
        """Test loading model with checksum and signature verification."""
        loader = SecureModelLoader()

        # Generate metadata with checksum and signature
        metadata = loader.generate_metadata(sample_model_file)
        metadata_file = os.path.join(temp_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        # Load model with full verification
        model = loader.load_model(
            model_file=sample_model_file,
            metadata_file=metadata_file
        )

        assert isinstance(model, IsolationForest)

    @patch.dict(os.environ, {'MODEL_TRUSTED_PATHS': '/trusted/path1,/trusted/path2'})
    def test_validate_trusted_path_allowed(self):
        """Test trusted path validation for allowed paths."""
        loader = SecureModelLoader()

        # Mock trusted paths
        loader.trusted_paths = [Path('/trusted/path1'), Path('/trusted/path2')]

        # Test allowed path
        assert loader._validate_trusted_path('/trusted/path1/models/model.pkl') is True
        assert loader._validate_trusted_path('/trusted/path2/model.pkl') is True

    @patch.dict(os.environ, {'MODEL_TRUSTED_PATHS': '/trusted/path'})
    def test_validate_trusted_path_blocked(self):
        """Test trusted path validation for blocked paths."""
        loader = SecureModelLoader()

        # Mock trusted paths
        loader.trusted_paths = [Path('/trusted/path')]

        # Test blocked path
        assert loader._validate_trusted_path('/untrusted/model.pkl') is False

    @patch.dict(os.environ, {'MODEL_TRUSTED_PATHS': '/trusted/path'})
    def test_load_model_untrusted_path(self, temp_dir, sample_model_file):
        """Test loading model from untrusted path."""
        loader = SecureModelLoader()
        loader.trusted_paths = [Path('/trusted/path')]

        # Should raise ModelIntegrityError
        with pytest.raises(ModelIntegrityError) as exc_info:
            loader.load_model(
                model_file=sample_model_file,
                skip_verification=False
            )

        assert "not in trusted storage path" in str(exc_info.value).lower()

    def test_load_model_file_not_found(self):
        """Test loading non-existent model file."""
        loader = SecureModelLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_model(
                model_file='/nonexistent/model.pkl',
                skip_verification=True
            )

    def test_generate_metadata(self, sample_model_file):
        """Test metadata generation."""
        loader = SecureModelLoader()

        metadata = loader.generate_metadata(sample_model_file)

        # Should have checksum
        assert 'checksum' in metadata
        assert 'checksum_algorithm' in metadata
        assert metadata['checksum_algorithm'] == 'sha256'
        assert len(metadata['checksum']) == 64

    @patch.dict(os.environ, {'MODEL_VERIFY_SIGNATURE': 'true', 'MODEL_SIGNATURE_KEY': 'test-secret'})
    def test_generate_metadata_with_signature(self, sample_model_file):
        """Test metadata generation with signature."""
        loader = SecureModelLoader()

        metadata = loader.generate_metadata(sample_model_file)

        # Should have checksum and signature
        assert 'checksum' in metadata
        assert 'signature' in metadata
        assert 'signature_algorithm' in metadata
        assert metadata['signature_algorithm'] == 'hmac-sha256'
        assert len(metadata['signature']) == 64

    def test_generate_metadata_with_existing(self, sample_model_file):
        """Test metadata generation with existing metadata."""
        loader = SecureModelLoader()

        existing_metadata = {
            'model_type': 'IsolationForest',
            'version': '1.0.0',
            'trained_at': '2025-01-01T00:00:00'
        }

        metadata = loader.generate_metadata(sample_model_file, existing_metadata)

        # Should preserve existing metadata
        assert metadata['model_type'] == 'IsolationForest'
        assert metadata['version'] == '1.0.0'
        assert metadata['trained_at'] == '2025-01-01T00:00:00'

        # Should add security fields
        assert 'checksum' in metadata
        assert 'checksum_algorithm' in metadata


class TestSecureModelLoaderConfiguration:
    """Test SecureModelLoader configuration."""

    @patch.dict(os.environ, {'MODEL_VERIFY_CHECKSUM': 'true'})
    def test_checksum_verification_enabled(self):
        """Test that checksum verification can be enabled."""
        loader = SecureModelLoader()
        assert loader.verify_checksum is True

    @patch.dict(os.environ, {'MODEL_VERIFY_CHECKSUM': 'false'})
    def test_checksum_verification_disabled(self):
        """Test that checksum verification can be disabled."""
        loader = SecureModelLoader()
        assert loader.verify_checksum is False

    @patch.dict(os.environ, {'MODEL_VERIFY_SIGNATURE': 'true', 'MODEL_SIGNATURE_KEY': 'test-key'})
    def test_signature_verification_enabled(self):
        """Test that signature verification can be enabled."""
        loader = SecureModelLoader()
        assert loader.verify_signature is True
        assert loader.signature_key == 'test-key'

    @patch.dict(os.environ, {'MODEL_VERIFY_SIGNATURE': 'true'})
    def test_signature_verification_without_key(self):
        """Test that signature verification is disabled without key."""
        loader = SecureModelLoader()
        assert loader.verify_signature is False

    @patch.dict(os.environ, {'MODEL_TRUSTED_PATHS': '/path1,/path2,/path3'})
    def test_trusted_paths_configuration(self):
        """Test trusted paths configuration."""
        loader = SecureModelLoader()
        assert len(loader.trusted_paths) == 3
        assert any(str(p).endswith('path1') for p in loader.trusted_paths)
        assert any(str(p).endswith('path2') for p in loader.trusted_paths)
        assert any(str(p).endswith('path3') for p in loader.trusted_paths)


class TestSecureModelLoaderIntegration:
    """Integration tests for SecureModelLoader."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def complete_model_package(self, temp_dir):
        """Create a complete model package with metadata."""
        # Create model
        model = IsolationForest(n_estimators=10, random_state=42)
        X = np.random.randn(100, 5)
        model.fit(X)

        # Save model
        model_file = os.path.join(temp_dir, 'model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)

        # Generate and save metadata
        loader = SecureModelLoader()
        metadata = loader.generate_metadata(model_file, {
            'model_type': 'IsolationForest',
            'version': '1.0.0',
            'hyperparameters': {'n_estimators': 10, 'random_state': 42}
        })

        metadata_file = os.path.join(temp_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        return model_file, metadata_file

    @patch.dict(os.environ, {'MODEL_VERIFY_CHECKSUM': 'true'})
    def test_complete_workflow(self, complete_model_package):
        """Test complete save and load workflow."""
        model_file, metadata_file = complete_model_package
        loader = SecureModelLoader()

        # Load model with verification
        model = loader.load_model(
            model_file=model_file,
            metadata_file=metadata_file
        )

        # Verify model works
        assert isinstance(model, IsolationForest)
        X_test = np.random.randn(10, 5)
        predictions = model.predict(X_test)
        assert len(predictions) == 10

    @patch.dict(os.environ, {'MODEL_VERIFY_CHECKSUM': 'true'})
    def test_tampered_model_detection(self, complete_model_package, temp_dir):
        """Test detection of tampered model files."""
        model_file, metadata_file = complete_model_package
        loader = SecureModelLoader()

        # Tamper with model file
        with open(model_file, 'ab') as f:
            f.write(b'malicious data')

        # Should detect tampering
        with pytest.raises(ModelIntegrityError) as exc_info:
            loader.load_model(
                model_file=model_file,
                metadata_file=metadata_file
            )

        assert "checksum verification failed" in str(exc_info.value).lower()
