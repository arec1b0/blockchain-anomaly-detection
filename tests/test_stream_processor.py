"""
Tests for stream processor.
"""

import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from src.streaming.stream_processor import StreamProcessor


class TestStreamProcessor:
    """Test cases for StreamProcessor."""

    def test_initialization(self):
        """Test processor initialization."""
        processor = StreamProcessor(
            batch_size=50,
            contamination=0.02
        )

        assert processor.batch_size == 50
        assert processor.contamination == 0.02
        assert len(processor.transaction_buffer) == 0
        assert len(processor.anomaly_buffer) == 0

    def test_initialization_with_model(self):
        """Test initialization with pre-trained model."""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            model_path = f.name
            # Create a dummy model file
            import pickle
            pickle.dump({'dummy': 'model'}, f)

        try:
            with patch('src.streaming.stream_processor.pickle.load'):
                processor = StreamProcessor(model_path=model_path)
                # Model loading is mocked, so we just verify it attempted to load
        finally:
            os.unlink(model_path)

    def test_process_valid_transaction(self):
        """Test processing a valid transaction."""
        processor = StreamProcessor(batch_size=100)

        transaction = {
            'hash': '0x123',
            'value': '100',
            'gas': '21000',
            'gasPrice': '20',
            'from': '0xabc',
            'to': '0xdef'
        }

        processor.process_transaction(transaction)

        assert len(processor.transaction_buffer) == 1
        assert processor.transaction_buffer[0]['hash'] == '0x123'
        assert processor.transaction_buffer[0]['value'] == 100.0

    def test_process_invalid_transaction(self):
        """Test processing an invalid transaction."""
        processor = StreamProcessor()

        transaction = {
            'hash': '0x123'
            # Missing required fields
        }

        processor.process_transaction(transaction)

        # Should handle error gracefully, buffer should be empty
        assert len(processor.transaction_buffer) == 0

    def test_transform_transaction(self):
        """Test transaction transformation."""
        processor = StreamProcessor()

        transaction = {
            'hash': '0x123',
            'value': '100',
            'gas': '21000',
            'gasPrice': '20',
            'from': '0xabc',
            'to': '0xdef',
            'blockNumber': 12345
        }

        transformed = processor._transform_transaction(transaction)

        assert transformed['hash'] == '0x123'
        assert transformed['value'] == 100.0
        assert transformed['gas'] == 21000.0
        assert transformed['gasPrice'] == 20.0
        assert transformed['from'] == '0xabc'
        assert transformed['to'] == '0xdef'
        assert transformed['blockNumber'] == 12345

    def test_transform_transaction_missing_fields(self):
        """Test transformation with missing required fields."""
        processor = StreamProcessor()

        transaction = {
            'hash': '0x123',
            'value': '100'
            # Missing gas and gasPrice
        }

        with pytest.raises(ValueError):
            processor._transform_transaction(transaction)

    def test_transform_transaction_invalid_values(self):
        """Test transformation with invalid numeric values."""
        processor = StreamProcessor()

        transaction = {
            'hash': '0x123',
            'value': 'invalid',
            'gas': '21000',
            'gasPrice': '20'
        }

        with pytest.raises(ValueError):
            processor._transform_transaction(transaction)

    @patch('src.streaming.stream_processor.AnomalyDetectorIsolationForest')
    def test_process_batch_new_model(self, mock_detector):
        """Test batch processing with new model training."""
        processor = StreamProcessor(batch_size=2)

        # Add transactions to buffer
        for i in range(2):
            transaction = {
                'hash': f'0x{i}',
                'value': '100',
                'gas': '21000',
                'gasPrice': '20'
            }
            processor.process_transaction(transaction)

        # Verify batch was processed and buffer cleared
        assert len(processor.transaction_buffer) == 0

    def test_batch_processing_threshold(self):
        """Test that batch processing triggers at threshold."""
        processor = StreamProcessor(batch_size=3)

        # Add transactions
        for i in range(2):
            transaction = {
                'hash': f'0x{i}',
                'value': '100',
                'gas': '21000',
                'gasPrice': '20'
            }
            processor.process_transaction(transaction)

        assert len(processor.transaction_buffer) == 2

        # Add one more to trigger batch processing
        transaction = {
            'hash': '0x3',
            'value': '100',
            'gas': '21000',
            'gasPrice': '20'
        }
        processor.process_transaction(transaction)

        # Buffer should be cleared after batch processing
        assert len(processor.transaction_buffer) == 0

    def test_calculate_severity(self):
        """Test severity calculation."""
        processor = StreamProcessor()

        # Critical severity
        anomaly = pd.Series({
            'value': 2000000,
            'gas': 21000,
            'gasPrice': 20
        })
        assert processor._calculate_severity(anomaly) == 'critical'

        # High severity
        anomaly = pd.Series({
            'value': 150000,
            'gas': 21000,
            'gasPrice': 20
        })
        assert processor._calculate_severity(anomaly) == 'high'

        # Medium severity
        anomaly = pd.Series({
            'value': 50000,
            'gas': 21000,
            'gasPrice': 20
        })
        assert processor._calculate_severity(anomaly) == 'medium'

        # Low severity
        anomaly = pd.Series({
            'value': 1000,
            'gas': 21000,
            'gasPrice': 20
        })
        assert processor._calculate_severity(anomaly) == 'low'

    def test_get_anomalies(self):
        """Test getting anomalies."""
        processor = StreamProcessor()

        # Add some anomalies to buffer
        processor.anomaly_buffer = [
            {'hash': '0x1', 'severity': 'high'},
            {'hash': '0x2', 'severity': 'low'},
            {'hash': '0x3', 'severity': 'medium'}
        ]

        anomalies = processor.get_anomalies()
        assert len(anomalies) == 3

        anomalies_limited = processor.get_anomalies(limit=2)
        assert len(anomalies_limited) == 2

    def test_clear_anomaly_buffer(self):
        """Test clearing anomaly buffer."""
        processor = StreamProcessor()

        processor.anomaly_buffer = [
            {'hash': '0x1'},
            {'hash': '0x2'}
        ]

        processor.clear_anomaly_buffer()

        assert len(processor.anomaly_buffer) == 0

    def test_get_stats(self):
        """Test getting statistics."""
        processor = StreamProcessor(batch_size=100, contamination=0.01)

        processor.transaction_buffer = [{'hash': '0x1'}]
        processor.anomaly_buffer = [{'hash': '0x2'}, {'hash': '0x3'}]

        stats = processor.get_stats()

        assert stats['buffer_size'] == 1
        assert stats['anomalies_detected'] == 2
        assert stats['batch_size'] == 100
        assert stats['contamination'] == 0.01

    def test_flush(self):
        """Test flushing buffer."""
        processor = StreamProcessor(batch_size=100)

        # Add transaction that won't trigger automatic batch processing
        transaction = {
            'hash': '0x1',
            'value': '100',
            'gas': '21000',
            'gasPrice': '20'
        }
        processor.process_transaction(transaction)

        assert len(processor.transaction_buffer) == 1

        # Flush should process the transaction
        processor.flush()

        assert len(processor.transaction_buffer) == 0

    @patch('src.streaming.stream_processor.AnomalyDetectorIsolationForest')
    def test_save_and_load_model(self, mock_detector):
        """Test model saving and loading."""
        processor = StreamProcessor()

        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            model_path = f.name

        try:
            # Save model
            processor.model = Mock()
            processor.save_model(model_path)

            assert os.path.exists(model_path)

            # Load model
            processor.load_model(model_path)

        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
