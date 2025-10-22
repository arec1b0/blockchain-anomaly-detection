"""
Kafka consumer service for streaming blockchain transaction data.
"""

import json
import logging
from typing import Callable, Dict, Any, Optional
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from prometheus_client import Counter, Histogram, Gauge
import time

logger = logging.getLogger(__name__)

# Prometheus metrics
kafka_messages_consumed = Counter(
    'kafka_messages_consumed_total',
    'Total number of Kafka messages consumed',
    ['topic', 'status']
)

kafka_processing_duration = Histogram(
    'kafka_message_processing_duration_seconds',
    'Time spent processing Kafka messages',
    ['topic']
)

kafka_consumer_lag = Gauge(
    'kafka_consumer_lag',
    'Current consumer lag',
    ['topic', 'partition']
)

kafka_errors = Counter(
    'kafka_errors_total',
    'Total number of Kafka errors',
    ['error_type']
)


class KafkaConsumerService:
    """
    Service for consuming blockchain transaction data from Kafka topics.

    Attributes:
        bootstrap_servers (str): Kafka bootstrap servers
        topic (str): Kafka topic to consume from
        group_id (str): Consumer group ID
        consumer (KafkaConsumer): Kafka consumer instance
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "blockchain-transactions",
        group_id: str = "anomaly-detection-group",
        auto_offset_reset: str = "latest",
        enable_auto_commit: bool = True,
        auto_commit_interval_ms: int = 5000,
        max_poll_records: int = 500
    ):
        """
        Initialize Kafka consumer service.

        Args:
            bootstrap_servers: Kafka broker addresses
            topic: Topic to consume messages from
            group_id: Consumer group identifier
            auto_offset_reset: Where to start reading messages
            enable_auto_commit: Whether to auto-commit offsets
            auto_commit_interval_ms: Auto-commit interval in milliseconds
            max_poll_records: Maximum records per poll
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.consumer: Optional[KafkaConsumer] = None
        self.is_running = False

        self.config = {
            'bootstrap_servers': bootstrap_servers.split(','),
            'group_id': group_id,
            'auto_offset_reset': auto_offset_reset,
            'enable_auto_commit': enable_auto_commit,
            'auto_commit_interval_ms': auto_commit_interval_ms,
            'max_poll_records': max_poll_records,
            'value_deserializer': lambda m: json.loads(m.decode('utf-8')),
            'key_deserializer': lambda m: m.decode('utf-8') if m else None,
        }

        logger.info(f"Kafka consumer initialized for topic: {topic}")

    def connect(self) -> None:
        """
        Establish connection to Kafka broker and subscribe to topic.

        Raises:
            KafkaError: If connection fails
        """
        try:
            self.consumer = KafkaConsumer(**self.config)
            self.consumer.subscribe([self.topic])
            logger.info(f"Connected to Kafka and subscribed to topic: {self.topic}")
        except KafkaError as e:
            kafka_errors.labels(error_type='connection_error').inc()
            logger.error(f"Failed to connect to Kafka: {e}")
            raise

    def disconnect(self) -> None:
        """Close Kafka consumer connection."""
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer disconnected")

    def consume(
        self,
        callback: Callable[[Dict[str, Any]], None],
        timeout_ms: int = 1000,
        max_messages: Optional[int] = None
    ) -> None:
        """
        Start consuming messages from Kafka topic.

        Args:
            callback: Function to process each message
            timeout_ms: Poll timeout in milliseconds
            max_messages: Maximum number of messages to process (None for infinite)
        """
        if not self.consumer:
            raise RuntimeError("Consumer not connected. Call connect() first.")

        self.is_running = True
        message_count = 0

        logger.info(f"Starting message consumption from topic: {self.topic}")

        try:
            while self.is_running:
                # Poll for messages
                messages = self.consumer.poll(timeout_ms=timeout_ms)

                if not messages:
                    continue

                for topic_partition, records in messages.items():
                    for record in records:
                        start_time = time.time()

                        try:
                            # Process message
                            callback(record.value)

                            # Record metrics
                            processing_time = time.time() - start_time
                            kafka_messages_consumed.labels(
                                topic=self.topic,
                                status='success'
                            ).inc()
                            kafka_processing_duration.labels(
                                topic=self.topic
                            ).observe(processing_time)

                            message_count += 1

                            # Check if we've reached max messages
                            if max_messages and message_count >= max_messages:
                                logger.info(f"Reached maximum message count: {max_messages}")
                                self.is_running = False
                                break

                        except Exception as e:
                            kafka_messages_consumed.labels(
                                topic=self.topic,
                                status='error'
                            ).inc()
                            kafka_errors.labels(error_type='processing_error').inc()
                            logger.error(f"Error processing message: {e}", exc_info=True)

                    if not self.is_running:
                        break

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping consumer...")
        except Exception as e:
            kafka_errors.labels(error_type='consumer_error').inc()
            logger.error(f"Error in consumer loop: {e}", exc_info=True)
            raise
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop consuming messages."""
        self.is_running = False
        logger.info("Kafka consumer stopped")

    def get_consumer_lag(self) -> Dict[str, int]:
        """
        Get current consumer lag for all partitions.

        Returns:
            Dictionary mapping partition to lag
        """
        if not self.consumer:
            return {}

        lag_info = {}

        try:
            # Get assigned partitions
            partitions = self.consumer.assignment()

            for partition in partitions:
                # Get current position
                current_position = self.consumer.position(partition)

                # Get high water mark (latest offset)
                end_offsets = self.consumer.end_offsets([partition])
                high_water_mark = end_offsets[partition]

                # Calculate lag
                lag = high_water_mark - current_position
                lag_info[partition.partition] = lag

                # Update Prometheus metric
                kafka_consumer_lag.labels(
                    topic=partition.topic,
                    partition=partition.partition
                ).set(lag)

        except Exception as e:
            logger.error(f"Error calculating consumer lag: {e}")

        return lag_info

    def seek_to_beginning(self) -> None:
        """Reset consumer to beginning of all partitions."""
        if self.consumer:
            self.consumer.seek_to_beginning()
            logger.info("Consumer seeked to beginning")

    def seek_to_end(self) -> None:
        """Move consumer to end of all partitions."""
        if self.consumer:
            self.consumer.seek_to_end()
            logger.info("Consumer seeked to end")
