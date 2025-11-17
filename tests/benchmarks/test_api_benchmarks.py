"""
Performance benchmarks for API endpoints.

Requires pytest-benchmark:
    pip install pytest-benchmark

Usage:
    pytest tests/benchmarks/ --benchmark-only
    pytest tests/benchmarks/ --benchmark-only --benchmark-json=results.json
    pytest tests/benchmarks/ --benchmark-only --benchmark-histogram
"""

import pytest
import pandas as pd
import numpy as np
from src.anomaly_detection.isolation_forest import AnomalyDetectorIsolationForest
from src.data_processing.data_transformation import DataTransformer
from src.streaming.stream_processor import StreamProcessor


class TestAnomalyDetectionBenchmarks:
    """Benchmarks for anomaly detection algorithms."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample transaction data for benchmarking."""
        np.random.seed(42)
        return pd.DataFrame({
            'hash': [f'0x{i:064x}' for i in range(1000)],
            'value': np.random.lognormal(10, 2, 1000),
            'gas': np.random.randint(21000, 100000, 1000),
            'gasPrice': np.random.lognormal(3, 1, 1000),
        })

    @pytest.fixture
    def detector(self, sample_data):
        """Create and train a detector."""
        detector = AnomalyDetectorIsolationForest(contamination=0.01)
        detector.train_model(sample_data)
        return detector

    def test_isolation_forest_training(self, benchmark, sample_data):
        """Benchmark Isolation Forest training time."""
        def train():
            detector = AnomalyDetectorIsolationForest(contamination=0.01)
            detector.train_model(sample_data)
            return detector

        result = benchmark(train)
        assert result is not None

    def test_isolation_forest_prediction_single(self, benchmark, detector, sample_data):
        """Benchmark single prediction time."""
        single_row = sample_data.iloc[[0]]

        result = benchmark(detector.detect_anomalies, single_row)
        assert len(result) == 1

    def test_isolation_forest_prediction_batch_10(self, benchmark, detector, sample_data):
        """Benchmark batch prediction (10 transactions)."""
        batch = sample_data.iloc[:10]

        result = benchmark(detector.detect_anomalies, batch)
        assert len(result) == 10

    def test_isolation_forest_prediction_batch_100(self, benchmark, detector, sample_data):
        """Benchmark batch prediction (100 transactions)."""
        batch = sample_data.iloc[:100]

        result = benchmark(detector.detect_anomalies, batch)
        assert len(result) == 100

    def test_isolation_forest_prediction_batch_1000(self, benchmark, detector, sample_data):
        """Benchmark batch prediction (1000 transactions)."""
        result = benchmark(detector.detect_anomalies, sample_data)
        assert len(result) == 1000


class TestDataTransformationBenchmarks:
    """Benchmarks for data transformation operations."""

    @pytest.fixture
    def transformer(self):
        """Create data transformer."""
        return DataTransformer()

    @pytest.fixture
    def large_data(self):
        """Generate large dataset for benchmarking."""
        np.random.seed(42)
        return pd.DataFrame({
            'value': np.random.lognormal(10, 2, 10000),
            'gas': np.random.randint(21000, 100000, 10000),
            'gasPrice': np.random.lognormal(3, 1, 10000),
        })

    def test_normalize_column(self, benchmark, transformer, large_data):
        """Benchmark column normalization."""
        result = benchmark(transformer.normalize_column, large_data, 'value')
        assert 'value_normalized' in result.columns

    def test_normalize_multiple_columns(self, benchmark, transformer, large_data):
        """Benchmark normalizing multiple columns."""
        def normalize_all():
            data = large_data.copy()
            data = transformer.normalize_column(data, 'value')
            data = transformer.normalize_column(data, 'gas')
            data = transformer.normalize_column(data, 'gasPrice')
            return data

        result = benchmark(normalize_all)
        assert 'value_normalized' in result.columns


class TestStreamProcessorBenchmarks:
    """Benchmarks for stream processing operations."""

    @pytest.fixture
    def processor(self):
        """Create stream processor."""
        return StreamProcessor(batch_size=100)

    @pytest.fixture
    def transaction(self):
        """Generate a single transaction."""
        return {
            'hash': '0x' + '0' * 64,
            'value': 100.0,
            'gas': 21000.0,
            'gasPrice': 20.0,
            'blockNumber': 1000000,
            'from': '0x' + '0' * 40,
            'to': '0x' + '1' * 40,
            'nonce': 1
        }

    def test_process_single_transaction(self, benchmark, processor, transaction):
        """Benchmark processing a single transaction."""
        result = benchmark(processor.process_transaction, transaction)
        # Result may be None if buffer not full
        assert result is None or isinstance(result, dict)

    def test_transform_transaction(self, benchmark, processor, transaction):
        """Benchmark transaction transformation."""
        result = benchmark(processor._transform_transaction, transaction)
        assert result is not None
        assert 'value' in result


class TestCacheBenchmarks:
    """Benchmarks for caching operations."""

    @pytest.fixture
    def cache_layer(self):
        """Create cache layer (mocked if Redis not available)."""
        try:
            from src.cache.cache_layer import CacheLayer
            from src.cache.redis_client import get_redis_client

            redis_client = get_redis_client()
            if redis_client and redis_client.ping():
                return CacheLayer(redis_client)
        except:
            pass

        # Mock cache if Redis not available
        import pytest
        pytest.skip("Redis not available for benchmarking")

    def test_cache_set(self, benchmark, cache_layer):
        """Benchmark cache set operation."""
        def cache_set():
            cache_layer.cache_prediction(
                transaction_hash='0x' + '0' * 64,
                prediction={'is_anomaly': False, 'confidence': 0.95}
            )

        benchmark(cache_set)

    def test_cache_get(self, benchmark, cache_layer):
        """Benchmark cache get operation."""
        # Pre-populate cache
        cache_layer.cache_prediction(
            transaction_hash='0x' + '0' * 64,
            prediction={'is_anomaly': False, 'confidence': 0.95}
        )

        result = benchmark(cache_layer.get_prediction, '0x' + '0' * 64)
        assert result is not None


@pytest.mark.parametrize("batch_size", [1, 10, 50, 100, 500, 1000])
def test_prediction_scaling(benchmark, batch_size):
    """
    Test how prediction time scales with batch size.

    This helps identify if there are any non-linear scaling issues.
    """
    np.random.seed(42)
    data = pd.DataFrame({
        'value': np.random.lognormal(10, 2, batch_size),
        'gas': np.random.randint(21000, 100000, batch_size),
        'gasPrice': np.random.lognormal(3, 1, batch_size),
    })

    detector = AnomalyDetectorIsolationForest(contamination=0.01)
    detector.train_model(data)

    result = benchmark(detector.detect_anomalies, data)
    assert len(result) == batch_size


# Benchmark hooks for custom reporting
@pytest.fixture(scope="session", autouse=True)
def benchmark_config(request):
    """Configure benchmark settings."""
    # Ensure we're running enough rounds for statistical significance
    request.config.option.benchmark_min_rounds = 5
    request.config.option.benchmark_warmup = True
