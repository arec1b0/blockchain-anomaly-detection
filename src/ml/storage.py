"""
Model storage for S3/GCS integration.

This module handles uploading and downloading ML model artifacts to/from cloud storage.
"""

import os
import shutil
from typing import Optional
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)
config = get_config()


class ModelStorage:
    """
    Handles model artifact storage to S3/GCS.

    Supports both cloud storage (S3/GCS) and local filesystem for development.
    """

    def __init__(self):
        """Initialize model storage with configuration."""
        self.storage_type = os.getenv("MODEL_STORAGE_TYPE", "local").lower()
        self.storage_path = os.getenv("MODEL_STORAGE_PATH", "./models")
        self.s3_bucket = os.getenv("S3_BUCKET", "")
        self.gcs_bucket = os.getenv("GCS_BUCKET", "")
        
        # Ensure local storage directory exists
        if self.storage_type == "local":
            Path(self.storage_path).mkdir(parents=True, exist_ok=True)

    def upload_model(
        self,
        local_dir: str,
        model_name: str,
        version: str
    ) -> str:
        """
        Upload model artifacts to storage.

        Args:
            local_dir: Local directory containing model artifacts
            model_name: Name of the model
            version: Version string (e.g., "20240101-120000" or "1.0.0")

        Returns:
            Storage path where model was saved
        """
        if self.storage_type == "s3":
            return self._upload_to_s3(local_dir, model_name, version)
        elif self.storage_type == "gcs":
            return self._upload_to_gcs(local_dir, model_name, version)
        else:
            return self._upload_to_local(local_dir, model_name, version)

    def download_model(
        self,
        storage_path: str,
        local_dir: str
    ) -> str:
        """
        Download model artifacts from storage.

        Args:
            storage_path: Storage path of the model
            local_dir: Local directory to download to

        Returns:
            Path to downloaded model directory
        """
        if storage_path.startswith("s3://"):
            return self._download_from_s3(storage_path, local_dir)
        elif storage_path.startswith("gs://"):
            return self._download_from_gcs(storage_path, local_dir)
        else:
            return self._download_from_local(storage_path, local_dir)

    def _upload_to_local(
        self,
        local_dir: str,
        model_name: str,
        version: str
    ) -> str:
        """Upload to local filesystem."""
        target_dir = os.path.join(self.storage_path, model_name, version)
        Path(target_dir).mkdir(parents=True, exist_ok=True)

        # Copy all files from local_dir to target_dir
        for item in os.listdir(local_dir):
            src = os.path.join(local_dir, item)
            dst = os.path.join(target_dir, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            elif os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)

        logger.info(f"Model uploaded to local storage: {target_dir}")
        return target_dir

    def _download_from_local(
        self,
        storage_path: str,
        local_dir: str
    ) -> str:
        """Download from local filesystem."""
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        if not os.path.exists(storage_path):
            raise FileNotFoundError(f"Model not found at: {storage_path}")

        # Copy all files from storage_path to local_dir
        for item in os.listdir(storage_path):
            src = os.path.join(storage_path, item)
            dst = os.path.join(local_dir, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            elif os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)

        logger.info(f"Model downloaded from local storage: {storage_path} -> {local_dir}")
        return local_dir

    def _upload_to_s3(
        self,
        local_dir: str,
        model_name: str,
        version: str
    ) -> str:
        """Upload to S3 (requires boto3)."""
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            logger.error("boto3 not installed. Install with: pip install boto3")
            raise ImportError("boto3 is required for S3 storage")

        if not self.s3_bucket:
            raise ValueError("S3_BUCKET environment variable not set")

        s3_client = boto3.client('s3')
        s3_path = f"s3://{self.s3_bucket}/models/{model_name}/{version}/"

        # Upload all files
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = f"models/{model_name}/{version}/{relative_path}".replace("\\", "/")

                try:
                    s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                    logger.debug(f"Uploaded {local_path} to s3://{self.s3_bucket}/{s3_key}")
                except ClientError as e:
                    logger.error(f"Failed to upload {local_path} to S3: {e}")
                    raise

        logger.info(f"Model uploaded to S3: {s3_path}")
        return s3_path

    def _download_from_s3(
        self,
        storage_path: str,
        local_dir: str
    ) -> str:
        """Download from S3 (requires boto3)."""
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            logger.error("boto3 not installed. Install with: pip install boto3")
            raise ImportError("boto3 is required for S3 storage")

        # Parse S3 path: s3://bucket/path/to/model/
        if not storage_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {storage_path}")

        parts = storage_path[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        s3_client = boto3.client('s3')
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        # List and download all objects with prefix
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

            for page in pages:
                if 'Contents' not in page:
                    continue
                for obj in page['Contents']:
                    key = obj['Key']
                    relative_path = key[len(prefix):].lstrip('/')
                    local_path = os.path.join(local_dir, relative_path)

                    # Create directory if needed
                    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

                    s3_client.download_file(bucket, key, local_path)
                    logger.debug(f"Downloaded s3://{bucket}/{key} to {local_path}")

        except ClientError as e:
            logger.error(f"Failed to download from S3: {e}")
            raise

        logger.info(f"Model downloaded from S3: {storage_path} -> {local_dir}")
        return local_dir

    def _upload_to_gcs(
        self,
        local_dir: str,
        model_name: str,
        version: str
    ) -> str:
        """Upload to GCS (requires google-cloud-storage)."""
        try:
            from google.cloud import storage
        except ImportError:
            logger.error("google-cloud-storage not installed. Install with: pip install google-cloud-storage")
            raise ImportError("google-cloud-storage is required for GCS storage")

        if not self.gcs_bucket:
            raise ValueError("GCS_BUCKET environment variable not set")

        client = storage.Client()
        bucket = client.bucket(self.gcs_bucket)
        gcs_path = f"gs://{self.gcs_bucket}/models/{model_name}/{version}/"

        # Upload all files
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                blob_name = f"models/{model_name}/{version}/{relative_path}".replace("\\", "/")

                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_path)
                logger.debug(f"Uploaded {local_path} to gs://{self.gcs_bucket}/{blob_name}")

        logger.info(f"Model uploaded to GCS: {gcs_path}")
        return gcs_path

    def _download_from_gcs(
        self,
        storage_path: str,
        local_dir: str
    ) -> str:
        """Download from GCS (requires google-cloud-storage)."""
        try:
            from google.cloud import storage
        except ImportError:
            logger.error("google-cloud-storage not installed. Install with: pip install google-cloud-storage")
            raise ImportError("google-cloud-storage is required for GCS storage")

        # Parse GCS path: gs://bucket/path/to/model/
        if not storage_path.startswith("gs://"):
            raise ValueError(f"Invalid GCS path: {storage_path}")

        parts = storage_path[5:].split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        # List and download all blobs with prefix
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            relative_path = blob.name[len(prefix):].lstrip('/')
            if not relative_path:
                continue

            local_path = os.path.join(local_dir, relative_path)
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            blob.download_to_filename(local_path)
            logger.debug(f"Downloaded gs://{bucket_name}/{blob.name} to {local_path}")

        logger.info(f"Model downloaded from GCS: {storage_path} -> {local_dir}")
        return local_dir

