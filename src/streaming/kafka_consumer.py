"""
Kafka consumer service for streaming blockchain transaction data.

This module provides a service for consuming blockchain transaction data from
Kafka topics. It includes functionalities for connecting to Kafka, consuming
messages, and handling consumer lag.
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
    'Total number of Kafka messages consumed.',
    ['topic', 'status']
)

kafka_processing_duration = Histogram(
    'kafka_message_processing_duration_seconds',
    'Time spent processing Kafka messages in seconds.',
    ['topic']
)

kafka_consumer_lag = Gauge(
    'kafka_consumer_lag',
    'Current consumer lag.',
    ['topic', 'partition']
)

kafka_errors = Counter(
    'kafka_errors_total',
    'Total number of Kafka errors.',
    ['error_type']
)


class KafkaConsumerService:
    """
    Service for consuming blockchain transaction data from Kafka topics.

    Attributes:
        bootstrap_servers (str): A comma-separated list of Kafka bootstrap servers.
        topic (str): The Kafka topic to consume messages from.
        group_id (str): The consumer group ID.
        consumer (Optional[KafkaConsumer]): The Kafka consumer instance.
        is_running (bool): A flag indicating if the consumer is running.
        config (Dict[str, Any]): A dictionary of Kafka consumer configurations.
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
        Initializes the KafkaConsumerService.

        Args:
            bootstrap_servers (str): A comma-separated list of Kafka broker addresses.
                Defaults to "localhost:9092".
            topic (str): The topic to consume messages from. Defaults to "blockchain-transactions".
            group_id (str): The consumer group identifier. Defaults to "anomaly-detection-group".
            auto_offset_reset (str): The position to start reading messages from.
                Defaults to "latest".
            enable_auto_commit (bool): Whether to auto-commit offsets. Defaults to True.
            auto_commit_interval_ms (int): The auto-commit interval in milliseconds.
                Defaults to 5000.
            max_poll_records (int): The maximum number of records per poll. Defaults to 500.
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
        Establishes a connection to the Kafka broker and subscribes to the topic.

        Raises:
            KafkaError: If the connection fails.
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
        """
        Closes the Kafka consumer connection.
        """
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
        Starts consuming messages from the Kafka topic.

        Args:
            callback (Callable[[Dict[str, Any]], None]): The function to process each message.
            timeout_ms (int): The poll timeout in milliseconds. Defaults to 1000.
            max_messages (Optional[int]): The maximum number of messages to process.
                If None, the consumer will run indefinitely. Defaults to None.

        Raises:
            RuntimeError: If the consumer is not connected.
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
        """
        Stops consuming messages.
        """
        self.is_running = False
        logger.info("Kafka consumer stopped")

    def get_consumer_lag(self) -> Dict[str, int]:
        """
        Gets the current consumer lag for all partitions.

        Returns:
            Dict[str, int]: A dictionary mapping each partition to its lag.
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
        """
        Resets the consumer to the beginning of all partitions.
        """
        if self.consumer:
            self.consumer.seek_to_beginning()
            logger.info("Consumer seeked to beginning")

    def seek_to_end(self) -> None:
        """
        Moves the consumer to the end of all partitions.
        """
        if self.consumer:
            self.consumer.seek_to_end()
            logger.info("Consumer seeked to end")
