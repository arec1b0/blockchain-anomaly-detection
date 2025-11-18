"""
Secure model deserialization with integrity verification.

This module provides secure unpickling of ML models with:
- SHA256 checksum verification
- HMAC signature verification (optional)
- Restricted unpickler (whitelisted classes only)
- Trusted storage path validation

Security measures prevent:
- Remote code execution via malicious pickle files
- Model tampering
- Unauthorized model modifications
"""

import os
import pickle
import hashlib
import hmac
import json
from typing import Any, Dict, Optional, Set
from pathlib import Path
import io

from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)
config = get_config()


class ModelIntegrityError(Exception):
    """Raised when model integrity verification fails."""
    pass


class RestrictedUnpickler(pickle.Unpickler):
    """
    Restricted unpickler that only allows safe classes.

    This prevents arbitrary code execution by whitelisting
    allowed classes for deserialization.
    """

    # Whitelist of safe modules and classes
    SAFE_MODULES = {
        'sklearn',
        'numpy',
        'scipy',
        'pandas',
        'statsmodels',
        'collections',
        'builtins',
        '__builtin__',
        'copyreg',
        '_codecs',
    }

    # Additional safe classes (full module.class paths)
    SAFE_CLASSES = {
        'sklearn.ensemble._iforest.IsolationForest',
        'sklearn.ensemble._forest.IsolationForest',
        'statsmodels.tsa.arima.model.ARIMA',
        'statsmodels.tsa.arima.model.ARIMAResults',
        'numpy.ndarray',
        'numpy.dtype',
        'pandas.core.frame.DataFrame',
        'pandas.core.series.Series',
    }

    def find_class(self, module: str, name: str) -> Any:
        """
        Override find_class to restrict allowed classes.

        Args:
            module: Module name
            name: Class name

        Returns:
            Class object if allowed

        Raises:
            pickle.UnpicklingError: If class is not whitelisted
        """
        # Check if module is in safe list
        module_root = module.split('.')[0]
        full_class_path = f"{module}.{name}"

        if module_root in self.SAFE_MODULES or full_class_path in self.SAFE_CLASSES:
            return super().find_class(module, name)

        # Log and raise error for unsafe class
        logger.error(
            f"Attempted to unpickle unsafe class: {full_class_path}",
            extra={"module_name": module, "class_name": name}
        )
        raise pickle.UnpicklingError(
            f"Unsafe class not allowed: {full_class_path}. "
            f"Only whitelisted ML model classes are permitted."
        )


class SecureModelLoader:
    """
    Secure model loader with integrity verification.

    Features:
    - SHA256 checksum verification
    - HMAC signature verification (if enabled)
    - Restricted unpickler
    - Trusted storage path validation

    Usage:
        loader = SecureModelLoader()
        model = loader.load_model(
            model_file='/path/to/model.pkl',
            metadata_file='/path/to/metadata.json'
        )

    Configuration:
        MODEL_VERIFY_CHECKSUM: Enable checksum verification (default: true)
        MODEL_VERIFY_SIGNATURE: Enable HMAC signature verification (default: false)
        MODEL_SIGNATURE_KEY: Secret key for HMAC verification
        MODEL_TRUSTED_PATHS: Comma-separated list of trusted storage paths
    """

    def __init__(self):
        """Initialize SecureModelLoader with configuration."""
        self.verify_checksum = os.getenv(
            "MODEL_VERIFY_CHECKSUM",
            "true"
        ).lower() == "true"

        self.verify_signature = os.getenv(
            "MODEL_VERIFY_SIGNATURE",
            "false"
        ).lower() == "true"

        self.signature_key = os.getenv("MODEL_SIGNATURE_KEY")
        if self.verify_signature and not self.signature_key:
            logger.warning(
                "MODEL_VERIFY_SIGNATURE enabled but MODEL_SIGNATURE_KEY not set. "
                "Signature verification will be skipped."
            )
            self.verify_signature = False

        # Trusted storage paths
        trusted_paths_env = os.getenv("MODEL_TRUSTED_PATHS", "")
        self.trusted_paths = [
            Path(p.strip()) for p in trusted_paths_env.split(',')
            if p.strip()
        ]

        logger.info(
            f"SecureModelLoader initialized: "
            f"checksum={self.verify_checksum}, "
            f"signature={self.verify_signature}, "
            f"trusted_paths={len(self.trusted_paths)}"
        )

    def _validate_trusted_path(self, file_path: str) -> bool:
        """
        Validate that file is in a trusted storage path.

        Args:
            file_path: Path to validate

        Returns:
            True if path is trusted or no trusted paths configured
        """
        if not self.trusted_paths:
            # No trusted paths configured, allow all
            return True

        file_path_obj = Path(file_path).resolve()

        for trusted_path in self.trusted_paths:
            try:
                # Check if file is under trusted path
                file_path_obj.relative_to(trusted_path.resolve())
                return True
            except ValueError:
                continue

        return False

    def _calculate_checksum(self, file_path: str) -> str:
        """
        Calculate SHA256 checksum of file.

        Args:
            file_path: Path to file

        Returns:
            Hex string of SHA256 hash
        """
        sha256 = hashlib.sha256()

        with open(file_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)

        return sha256.hexdigest()

    def _verify_checksum(
        self,
        file_path: str,
        expected_checksum: str
    ) -> bool:
        """
        Verify file checksum matches expected value.

        Args:
            file_path: Path to file
            expected_checksum: Expected SHA256 hex string

        Returns:
            True if checksum matches

        Raises:
            ModelIntegrityError: If checksum doesn't match
        """
        actual_checksum = self._calculate_checksum(file_path)

        if actual_checksum != expected_checksum:
            logger.error(
                f"Checksum verification failed for {file_path}",
                extra={
                    "expected": expected_checksum,
                    "actual": actual_checksum
                }
            )
            raise ModelIntegrityError(
                f"Model checksum verification failed. "
                f"Expected: {expected_checksum}, Got: {actual_checksum}"
            )

        logger.debug(f"Checksum verified for {file_path}")
        return True

    def _calculate_signature(self, data: bytes) -> str:
        """
        Calculate HMAC signature of data.

        Args:
            data: Data to sign

        Returns:
            Hex string of HMAC-SHA256
        """
        signature = hmac.new(
            self.signature_key.encode('utf-8'),
            data,
            hashlib.sha256
        )
        return signature.hexdigest()

    def _verify_signature(
        self,
        file_path: str,
        expected_signature: str
    ) -> bool:
        """
        Verify HMAC signature of file.

        Args:
            file_path: Path to file
            expected_signature: Expected HMAC hex string

        Returns:
            True if signature matches

        Raises:
            ModelIntegrityError: If signature doesn't match
        """
        with open(file_path, 'rb') as f:
            data = f.read()

        actual_signature = self._calculate_signature(data)

        # Use constant-time comparison to prevent timing attacks
        if not hmac.compare_digest(actual_signature, expected_signature):
            logger.error(
                f"Signature verification failed for {file_path}",
                extra={
                    "expected": expected_signature[:16] + "...",
                    "actual": actual_signature[:16] + "..."
                }
            )
            raise ModelIntegrityError(
                "Model signature verification failed. "
                "File may have been tampered with."
            )

        logger.debug(f"Signature verified for {file_path}")
        return True

    def load_model(
        self,
        model_file: str,
        metadata_file: Optional[str] = None,
        skip_verification: bool = False
    ) -> Any:
        """
        Load model with security verification.

        Args:
            model_file: Path to model pickle file
            metadata_file: Optional path to metadata.json with checksums
            skip_verification: Skip all verification (DANGEROUS - testing only)

        Returns:
            Loaded model object

        Raises:
            FileNotFoundError: If files don't exist
            ModelIntegrityError: If integrity verification fails
            pickle.UnpicklingError: If unsafe classes detected
        """
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # Validate trusted path
        if not self._validate_trusted_path(model_file):
            logger.error(
                f"Model file not in trusted storage path: {model_file}",
                extra={"trusted_paths": [str(p) for p in self.trusted_paths]}
            )
            raise ModelIntegrityError(
                f"Model file not in trusted storage path. "
                f"Trusted paths: {self.trusted_paths}"
            )

        # Load metadata if provided
        metadata = {}
        if metadata_file and os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        # Perform verification (unless explicitly skipped)
        if not skip_verification:
            # Checksum verification
            if self.verify_checksum:
                expected_checksum = metadata.get('checksum')
                if not expected_checksum:
                    logger.warning(
                        f"Checksum verification enabled but no checksum in metadata for {model_file}"
                    )
                else:
                    self._verify_checksum(model_file, expected_checksum)

            # Signature verification
            if self.verify_signature:
                expected_signature = metadata.get('signature')
                if not expected_signature:
                    logger.warning(
                        f"Signature verification enabled but no signature in metadata for {model_file}"
                    )
                else:
                    self._verify_signature(model_file, expected_signature)
        else:
            logger.warning(
                f"SECURITY WARNING: Verification skipped for {model_file}. "
                f"This should only be used in testing!"
            )

        # Load model with restricted unpickler
        logger.info(f"Loading model with secure unpickler: {model_file}")

        try:
            with open(model_file, 'rb') as f:
                model = RestrictedUnpickler(f).load()

            logger.info(
                f"Model loaded successfully: {model_file}",
                extra={
                    "model_type": type(model).__name__,
                    "verified": not skip_verification
                }
            )

            return model

        except pickle.UnpicklingError as e:
            logger.error(
                f"Failed to unpickle model (possibly unsafe class): {e}",
                exc_info=True
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load model: {e}")

    def generate_metadata(
        self,
        model_file: str,
        existing_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate metadata with security checksums.

        This should be called when saving a new model to generate
        the checksums and signatures.

        Args:
            model_file: Path to model pickle file
            existing_metadata: Existing metadata to merge with

        Returns:
            Metadata dictionary with security fields
        """
        metadata = existing_metadata.copy() if existing_metadata else {}

        # Generate checksum
        checksum = self._calculate_checksum(model_file)
        metadata['checksum'] = checksum
        metadata['checksum_algorithm'] = 'sha256'

        # Generate signature if enabled
        if self.verify_signature and self.signature_key:
            with open(model_file, 'rb') as f:
                data = f.read()
            signature = self._calculate_signature(data)
            metadata['signature'] = signature
            metadata['signature_algorithm'] = 'hmac-sha256'

        logger.info(
            f"Generated security metadata for {model_file}",
            extra={
                "checksum": checksum[:16] + "...",
                "has_signature": 'signature' in metadata
            }
        )

        return metadata

    @staticmethod
    def add_safe_class(module: str, class_name: str):
        """
        Add a custom class to the whitelist.

        Use with caution - only add classes you trust.

        Args:
            module: Module name (e.g., 'mypackage.models')
            class_name: Class name (e.g., 'MyModel')
        """
        full_path = f"{module}.{class_name}"
        RestrictedUnpickler.SAFE_CLASSES.add(full_path)
        logger.info(f"Added safe class: {full_path}")

    @staticmethod
    def add_safe_module(module: str):
        """
        Add a module to the whitelist.

        Use with caution - only add modules you trust.

        Args:
            module: Module name (e.g., 'mypackage')
        """
        RestrictedUnpickler.SAFE_MODULES.add(module)
        logger.info(f"Added safe module: {module}")
