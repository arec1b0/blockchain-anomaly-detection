"""
Tests for Kafka consumer service.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from kafka.errors import KafkaError

from src.streaming.kafka_consumer import KafkaConsumerService


class TestKafkaConsumerService:
    """Test cases for KafkaConsumerService."""

    def test_initialization(self):
        """Test consumer initialization."""
        consumer = KafkaConsumerService(
            bootstrap_servers="localhost:9092",
            topic="test-topic",
            group_id="test-group"
        )

        assert consumer.bootstrap_servers == "localhost:9092"
        assert consumer.topic == "test-topic"
        assert consumer.group_id == "test-group"
        assert consumer.consumer is None
        assert not consumer.is_running

    @patch('src.streaming.kafka_consumer.KafkaConsumer')
    def test_connect_success(self, mock_kafka_consumer):
        """Test successful connection to Kafka."""
        consumer = KafkaConsumerService()
        consumer.connect()

        assert consumer.consumer is not None
        mock_kafka_consumer.assert_called_once()
        consumer.consumer.subscribe.assert_called_once_with([consumer.topic])

    @patch('src.streaming.kafka_consumer.KafkaConsumer')
    def test_connect_failure(self, mock_kafka_consumer):
        """Test connection failure."""
        mock_kafka_consumer.side_effect = KafkaError("Connection failed")

        consumer = KafkaConsumerService()

        with pytest.raises(KafkaError):
            consumer.connect()

    @patch('src.streaming.kafka_consumer.KafkaConsumer')
    def test_disconnect(self, mock_kafka_consumer):
        """Test disconnect."""
        consumer = KafkaConsumerService()
        consumer.connect()
        consumer.disconnect()

        consumer.consumer.close.assert_called_once()

    @patch('src.streaming.kafka_consumer.KafkaConsumer')
    def test_consume_messages(self, mock_kafka_consumer):
        """Test message consumption."""
        # Create mock messages
        mock_record = Mock()
        mock_record.value = {"hash": "0x123", "value": 100, "gas": 21000, "gasPrice": 20}

        mock_partition = Mock()
        mock_consumer_instance = mock_kafka_consumer.return_value
        mock_consumer_instance.poll.side_effect = [
            {mock_partition: [mock_record]},
            {}  # Empty poll to end the loop
        ]

        consumer = KafkaConsumerService()
        consumer.connect()

        messages_received = []

        def callback(message):
            messages_received.append(message)
            consumer.stop()  # Stop after first message

        consumer.consume(callback, timeout_ms=100)

        assert len(messages_received) == 1
        assert messages_received[0]["hash"] == "0x123"

    @patch('src.streaming.kafka_consumer.KafkaConsumer')
    def test_consume_with_error(self, mock_kafka_consumer):
        """Test consumption with processing error."""
        mock_record = Mock()
        mock_record.value = {"invalid": "data"}

        mock_partition = Mock()
        mock_consumer_instance = mock_kafka_consumer.return_value
        mock_consumer_instance.poll.side_effect = [
            {mock_partition: [mock_record]},
            {}
        ]

        consumer = KafkaConsumerService()
        consumer.connect()

        def callback(message):
            consumer.stop()
            raise ValueError("Processing error")

        # Should not raise, error should be logged
        consumer.consume(callback, timeout_ms=100)

    @patch('src.streaming.kafka_consumer.KafkaConsumer')
    def test_consume_max_messages(self, mock_kafka_consumer):
        """Test consuming maximum number of messages."""
        mock_records = [Mock(value={"id": i}) for i in range(10)]
        mock_partition = Mock()

        mock_consumer_instance = mock_kafka_consumer.return_value
        mock_consumer_instance.poll.return_value = {mock_partition: mock_records}

        consumer = KafkaConsumerService()
        consumer.connect()

        messages_received = []

        def callback(message):
            messages_received.append(message)

        consumer.consume(callback, timeout_ms=100, max_messages=5)

        assert len(messages_received) == 5

    @patch('src.streaming.kafka_consumer.KafkaConsumer')
    def test_get_consumer_lag(self, mock_kafka_consumer):
        """Test getting consumer lag."""
        mock_partition = Mock()
        mock_partition.partition = 0
        mock_partition.topic = "test-topic"

        mock_consumer_instance = mock_kafka_consumer.return_value
        mock_consumer_instance.assignment.return_value = [mock_partition]
        mock_consumer_instance.position.return_value = 100
        mock_consumer_instance.end_offsets.return_value = {mock_partition: 150}

        consumer = KafkaConsumerService()
        consumer.connect()

        lag = consumer.get_consumer_lag()

        assert lag[0] == 50  # 150 - 100

    @patch('src.streaming.kafka_consumer.KafkaConsumer')
    def test_seek_to_beginning(self, mock_kafka_consumer):
        """Test seeking to beginning."""
        consumer = KafkaConsumerService()
        consumer.connect()
        consumer.seek_to_beginning()

        consumer.consumer.seek_to_beginning.assert_called_once()

    @patch('src.streaming.kafka_consumer.KafkaConsumer')
    def test_seek_to_end(self, mock_kafka_consumer):
        """Test seeking to end."""
        consumer = KafkaConsumerService()
        consumer.connect()
        consumer.seek_to_end()

        consumer.consumer.seek_to_end.assert_called_once()

    def test_consume_without_connect(self):
        """Test consuming without connecting first."""
        consumer = KafkaConsumerService()

        with pytest.raises(RuntimeError):
            consumer.consume(lambda x: None)

    @patch('src.streaming.kafka_consumer.KafkaConsumer')
    def test_stop(self, mock_kafka_consumer):
        """Test stopping consumer."""
        consumer = KafkaConsumerService()
        consumer.connect()
        consumer.is_running = True

        consumer.stop()

        assert not consumer.is_running
