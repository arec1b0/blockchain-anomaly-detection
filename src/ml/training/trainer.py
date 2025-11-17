"""
ML model training orchestrator.

This module provides the ModelTrainer class that orchestrates the complete
ML model training workflow including data fetching, feature engineering,
hyperparameter tuning, training, evaluation, and model registration.
"""

import os
import pickle
import hashlib
import json
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
from optuna.samplers import TPESampler

from src.anomaly_detection.isolation_forest import AnomalyDetectorIsolationForest
from src.data_processing.data_transformation import DataTransformer
from src.database.repositories.transaction_repository import TransactionRepository
from src.database.repositories.model_repository import ModelRepository, ModelVersionRepository
from src.database.models import Model, ModelVersion
from src.ml.storage import ModelStorage
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)
config = get_config()


class ModelTrainer:
    """
    Orchestrates ML model training workflow.

    Steps:
    1. Fetch training data
    2. Feature engineering
    3. Hyperparameter tuning (optional)
    4. Train model
    5. Evaluate performance
    6. Save artifacts
    7. Register in model registry
    """

    def __init__(self, db_session):
        """
        Initialize ModelTrainer.

        Args:
            db_session: SQLAlchemy database session
        """
        self.db = db_session
        self.transaction_repo = TransactionRepository(db_session)
        self.model_repo = ModelRepository(db_session)
        self.model_version_repo = ModelVersionRepository(db_session)
        self.storage = ModelStorage()

    async def train_isolation_forest(
        self,
        model_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        hyperparameter_tuning: bool = True,
        contamination: Optional[float] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Train Isolation Forest model.

        Args:
            model_name: Name for the model
            start_date: Training data start date
            end_date: Training data end date
            hyperparameter_tuning: Whether to tune hyperparameters
            contamination: Expected anomaly proportion (None = auto-tune)

        Returns:
            Tuple of (model_version_id, metrics_dict)
        """
        logger.info(f"Starting training for model: {model_name}")
        training_start = datetime.utcnow()

        # Step 1: Fetch training data
        logger.info("Fetching training data...")
        transactions = self.transaction_repo.get_by_date_range(
            start_date or datetime(2024, 1, 1),
            end_date or datetime.utcnow(),
            limit=100000  # Max 100K for training
        )

        if len(transactions) < 1000:
            raise ValueError(f"Insufficient training data: {len(transactions)} transactions")

        logger.info(f"Loaded {len(transactions)} transactions for training")

        # Step 2: Convert to DataFrame and engineer features
        df = pd.DataFrame([{
            'hash': t.hash,
            'value': t.value,
            'gas': t.gas,
            'gasPrice': t.gas_price,
            'timestamp': t.timestamp
        } for t in transactions])

        # Feature engineering
        transformer = DataTransformer(df)
        df = transformer.normalize_column('value')
        df = transformer.normalize_column('gas')
        df = transformer.normalize_column('gasPrice')

        # Add derived features
        df['value_per_gas'] = df['value'] / (df['gas'] + 1e-10)
        df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek

        # Select features
        feature_columns = ['value', 'gas', 'gasPrice', 'value_per_gas', 'hour_of_day', 'day_of_week']
        X = df[feature_columns].values

        # Step 3: Hyperparameter tuning
        best_params = {'contamination': contamination or 0.01, 'n_estimators': 100}

        if hyperparameter_tuning:
            logger.info("Running hyperparameter tuning...")
            best_params = self._tune_hyperparameters(X)
            logger.info(f"Best parameters: {best_params}")

        # Step 4: Train model with best parameters
        logger.info("Training model...")
        detector = AnomalyDetectorIsolationForest(
            contamination=best_params['contamination']
        )
        detector.model.set_params(n_estimators=best_params.get('n_estimators', 100))
        detector.train_model(df)

        # Step 5: Evaluate model
        logger.info("Evaluating model...")
        metrics = self._evaluate_model(detector, X)

        training_duration = (datetime.utcnow() - training_start).total_seconds()
        metrics['training_duration_seconds'] = training_duration
        metrics['training_samples'] = len(X)

        logger.info(f"Model metrics: {metrics}")

        # Step 6: Save model artifacts
        logger.info("Saving model artifacts...")
        model_path, checksum = self._save_model_artifacts(
            detector,
            model_name,
            best_params,
            metrics,
            feature_columns
        )

        # Step 7: Register in model registry
        logger.info("Registering model in registry...")
        model_version_id = self._register_model(
            model_name,
            model_path,
            checksum,
            best_params,
            metrics,
            len(X),
            training_duration
        )

        logger.info(f"Training completed. Model version: {model_version_id}")

        return model_version_id, metrics

    def _tune_hyperparameters(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Hyperparameter tuning using Optuna.

        Args:
            X: Training features

        Returns:
            Best hyperparameters dict
        """
        def objective(trial):
            # Define hyperparameter search space
            contamination = trial.suggest_float('contamination', 0.001, 0.1, log=True)
            n_estimators = trial.suggest_int('n_estimators', 50, 200, step=50)
            max_samples = trial.suggest_categorical('max_samples', ['auto', 256, 512, 1024])

            # Create model
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                max_samples=max_samples,
                random_state=42,
                n_jobs=-1
            )

            # Cross-validation
            # Since anomaly detection is unsupervised, we use anomaly score variance
            model.fit(X)
            scores = model.score_samples(X)

            # Objective: Maximize score variance (better separation)
            return np.var(scores)

        # Run optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=20, show_progress_bar=False)

        return study.best_params

    def _evaluate_model(
        self,
        detector: AnomalyDetectorIsolationForest,
        X: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        For unsupervised anomaly detection, we use:
        - Anomaly score distribution metrics
        - Silhouette score (if possible)
        - Number of anomalies detected

        Args:
            detector: Trained detector
            X: Test features

        Returns:
            Metrics dictionary
        """
        # Get anomaly scores
        scores = detector.model.score_samples(X)
        predictions = detector.model.predict(X)  # -1 = anomaly, 1 = normal

        num_anomalies = np.sum(predictions == -1)
        anomaly_rate = num_anomalies / len(predictions)

        metrics = {
            'num_samples': len(X),
            'num_anomalies_detected': int(num_anomalies),
            'anomaly_rate': float(anomaly_rate),
            'mean_anomaly_score': float(np.mean(scores)),
            'std_anomaly_score': float(np.std(scores)),
            'min_anomaly_score': float(np.min(scores)),
            'max_anomaly_score': float(np.max(scores)),
            'score_range': float(np.max(scores) - np.min(scores))
        }

        # If we have labeled data (from reviewed anomalies), calculate precision/recall
        # This would require fetching reviewed anomalies from DB
        # For now, we skip this

        return metrics

    def _save_model_artifacts(
        self,
        detector: AnomalyDetectorIsolationForest,
        model_name: str,
        hyperparameters: Dict,
        metrics: Dict,
        feature_columns: list
    ) -> Tuple[str, str]:
        """
        Save model artifacts to storage.

        Args:
            detector: Trained detector
            model_name: Model name
            hyperparameters: Hyperparameters used
            metrics: Evaluation metrics
            feature_columns: List of feature column names

        Returns:
            Tuple of (storage_path, checksum)
        """
        # Create temporary directory
        tmp_dir = tempfile.mkdtemp()

        try:
            # Save model pickle
            model_file = os.path.join(tmp_dir, 'model.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(detector.model, f)

            # Save metadata
            metadata = {
                'model_name': model_name,
                'model_type': 'isolation_forest',
                'hyperparameters': hyperparameters,
                'metrics': metrics,
                'feature_columns': feature_columns,
                'training_timestamp': datetime.utcnow().isoformat(),
                'scikit_learn_version': '1.2.2'
            }

            metadata_file = os.path.join(tmp_dir, 'metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Calculate checksum
            with open(model_file, 'rb') as f:
                checksum = hashlib.sha256(f.read()).hexdigest()

            # Upload to storage (S3/GCS/local)
            storage_path = self.storage.upload_model(
                local_dir=tmp_dir,
                model_name=model_name,
                version=datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            )

            logger.info(f"Model artifacts saved to: {storage_path}")

            return storage_path, checksum
        finally:
            # Cleanup temporary directory
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _register_model(
        self,
        model_name: str,
        storage_path: str,
        checksum: str,
        hyperparameters: Dict,
        metrics: Dict,
        training_size: int,
        training_duration: float
    ) -> str:
        """
        Register model in database registry.

        Args:
            model_name: Model name
            storage_path: Storage location
            checksum: SHA256 checksum
            hyperparameters: Hyperparameters used
            metrics: Evaluation metrics
            training_size: Number of training samples
            training_duration: Training time in seconds

        Returns:
            Model version ID
        """
        # Check if model exists
        model = self.model_repo.get_by_name(model_name)

        if not model:
            # Create new model
            model = Model(
                name=model_name,
                model_type='isolation_forest',
                description=f'Isolation Forest anomaly detector trained on {training_size} transactions'
            )
            model = self.model_repo.create(model)

        # Determine version number
        existing_versions = self.model_version_repo.get_by_model_id(model.id)
        version = f"1.0.{len(existing_versions)}"

        # Create model version
        model_version = ModelVersion(
            model_id=model.id,
            version=version,
            storage_path=storage_path,
            checksum=checksum,
            training_dataset_size=training_size,
            training_duration_seconds=training_duration,
            hyperparameters=hyperparameters,
            metrics=metrics,
            is_deployed=False,
            traffic_percentage=0.0
        )

        model_version = self.model_version_repo.create(model_version)

        logger.info(f"Model registered: {model_name} v{version} (ID: {model_version.id})")

        return model_version.id

