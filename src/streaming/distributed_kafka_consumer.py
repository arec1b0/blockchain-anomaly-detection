"""
Distributed Kafka consumer with thread pool for parallel message processing.

This module provides an enhanced Kafka consumer service that supports:
- Parallel message processing using thread pool
- Distributed consumer group coordination
- Automatic partition rebalancing
- Graceful shutdown with message completion
- Enhanced monitoring and metrics
"""

import json
import logging
import threading
import signal
import sys
from typing import Callable, Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Empty
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

kafka_threads_active = Gauge(
    'kafka_consumer_threads_active',
    'Number of active consumer threads.'
)

kafka_queue_size = Gauge(
    'kafka_consumer_queue_size',
    'Size of the message processing queue.'
)


class DistributedKafkaConsumer:
    """
    Distributed Kafka consumer with parallel processing capabilities.

    Features:
    - Thread pool for concurrent message processing
    - Consumer group coordination for horizontal scaling
    - Automatic partition rebalancing
    - Graceful shutdown handling
    - Batch processing support
    - Enhanced error handling and retry logic

    Attributes:
        bootstrap_servers (str): Kafka bootstrap servers.
        topic (str): Topic to consume from.
        group_id (str): Consumer group ID for distributed consumption.
        consumer (Optional[KafkaConsumer]): The Kafka consumer instance.
        executor (ThreadPoolExecutor): Thread pool for message processing.
        is_running (bool): Flag indicating if consumer is active.
        processing_queue (Queue): Queue for messages awaiting processing.
        config (Dict[str, Any]): Kafka consumer configuration.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "blockchain-transactions",
        group_id: str = "anomaly-detection-group",
        auto_offset_reset: str = "latest",
        enable_auto_commit: bool = False,  # Manual commit for reliability
        max_poll_records: int = 500,
        num_worker_threads: int = 4,
        max_queue_size: int = 1000,
        session_timeout_ms: int = 30000,
        max_poll_interval_ms: int = 300000
    ):
        """
        Initializes the DistributedKafkaConsumer.

        Args:
            bootstrap_servers (str): Kafka broker addresses.
            topic (str): Topic to consume from.
            group_id (str): Consumer group ID.
            auto_offset_reset (str): Where to start reading messages.
            enable_auto_commit (bool): Whether to auto-commit offsets.
            max_poll_records (int): Max records per poll.
            num_worker_threads (int): Number of worker threads.
            max_queue_size (int): Maximum processing queue size.
            session_timeout_ms (int): Session timeout in milliseconds.
            max_poll_interval_ms (int): Max time between polls in milliseconds.
        """
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.consumer: Optional[KafkaConsumer] = None
        self.is_running = False
        self.num_worker_threads = num_worker_threads
        self.max_queue_size = max_queue_size

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(
            max_workers=num_worker_threads,
            thread_name_prefix='kafka-worker'
        )

        # Queue for buffering messages before processing
        self.processing_queue: Queue = Queue(maxsize=max_queue_size)

        # Track active futures for graceful shutdown
        self.active_futures: List[Future] = []
        self.futures_lock = threading.Lock()

        # Kafka consumer configuration
        self.config = {
            'bootstrap_servers': bootstrap_servers.split(','),
            'group_id': group_id,
            'auto_offset_reset': auto_offset_reset,
            'enable_auto_commit': enable_auto_commit,
            'max_poll_records': max_poll_records,
            'session_timeout_ms': session_timeout_ms,
            'max_poll_interval_ms': max_poll_interval_ms,
            'value_deserializer': lambda m: json.loads(m.decode('utf-8')),
            'key_deserializer': lambda m: m.decode('utf-8') if m else None,
            # Rebalance listener for logging
            'consumer_timeout_ms': 1000
        }

        # Offset tracking for manual commits
        self.pending_offsets: Dict[Any, int] = {}
        self.offset_lock = threading.Lock()

        # Graceful shutdown
        self._shutdown_event = threading.Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(
            f"DistributedKafkaConsumer initialized: topic={topic}, "
            f"group={group_id}, workers={num_worker_threads}"
        )

    def _signal_handler(self, signum, frame):
        """Handles shutdown signals for graceful termination."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()

    def connect(self) -> None:
        """
        Establishes connection to Kafka and subscribes to topic.

        Raises:
            KafkaError: If connection fails.
        """
        try:
            self.consumer = KafkaConsumer(**self.config)
            self.consumer.subscribe([self.topic])
            logger.info(
                f"Connected to Kafka: topic={self.topic}, "
                f"group={self.group_id}, "
                f"partitions={len(self.consumer.partitions_for_topic(self.topic) or [])}"
            )
        except KafkaError as e:
            kafka_errors.labels(error_type='connection_error').inc()
            logger.error(f"Failed to connect to Kafka: {e}")
            raise

    def disconnect(self) -> None:
        """Closes Kafka consumer connection."""
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer disconnected")

    def _process_message_wrapper(
        self,
        message_data: Dict[str, Any],
        callback: Callable,
        topic_partition: Any,
        offset: int
    ) -> None:
        """
        Wrapper for processing messages with error handling and metrics.

        Args:
            message_data (Dict[str, Any]): The message to process.
            callback (Callable): The processing function.
            topic_partition: The topic partition this message is from.
            offset (int): The message offset.
        """
        kafka_threads_active.inc()
        start_time = time.time()

        try:
            # Process message
            callback(message_data)

            # Record success metrics
            processing_time = time.time() - start_time
            kafka_messages_consumed.labels(
                topic=self.topic,
                status='success'
            ).inc()
            kafka_processing_duration.labels(
                topic=self.topic
            ).observe(processing_time)

            # Mark offset for commit
            with self.offset_lock:
                self.pending_offsets[topic_partition] = offset + 1

        except Exception as e:
            kafka_messages_consumed.labels(
                topic=self.topic,
                status='error'
            ).inc()
            kafka_errors.labels(error_type='processing_error').inc()
            logger.error(
                f"Error processing message at offset {offset}: {e}",
                exc_info=True
            )
        finally:
            kafka_threads_active.dec()

    def consume(
        self,
        callback: Callable[[Dict[str, Any]], None],
        timeout_ms: int = 1000,
        max_messages: Optional[int] = None,
        commit_interval: int = 100
    ) -> None:
        """
        Starts consuming messages with parallel processing.

        Args:
            callback (Callable): Function to process each message.
            timeout_ms (int): Poll timeout in milliseconds.
            max_messages (Optional[int]): Max messages to process (None = infinite).
            commit_interval (int): Commit offsets every N messages.

        Raises:
            RuntimeError: If consumer not connected.
        """
        if not self.consumer:
            raise RuntimeError("Consumer not connected. Call connect() first.")

        self.is_running = True
        message_count = 0
        messages_since_commit = 0

        logger.info(
            f"Starting distributed message consumption: "
            f"workers={self.num_worker_threads}, queue_size={self.max_queue_size}"
        )

        try:
            while self.is_running and not self._shutdown_event.is_set():
                # Poll for messages
                messages = self.consumer.poll(timeout_ms=timeout_ms)

                if not messages:
                    # Update queue size metric
                    kafka_queue_size.set(self.processing_queue.qsize())
                    continue

                for topic_partition, records in messages.items():
                    for record in records:
                        # Check if we should stop
                        if not self.is_running or self._shutdown_event.is_set():
                            break

                        # Submit to thread pool for processing
                        future = self.executor.submit(
                            self._process_message_wrapper,
                            record.value,
                            callback,
                            topic_partition,
                            record.offset
                        )

                        # Track future for graceful shutdown
                        with self.futures_lock:
                            self.active_futures.append(future)
                            # Clean up completed futures
                            self.active_futures = [
                                f for f in self.active_futures if not f.done()
                            ]

                        message_count += 1
                        messages_since_commit += 1

                        # Commit offsets periodically
                        if messages_since_commit >= commit_interval:
                            self._commit_offsets()
                            messages_since_commit = 0

                        # Check max messages
                        if max_messages and message_count >= max_messages:
                            logger.info(f"Reached max messages: {max_messages}")
                            self.is_running = False
                            break

                    if not self.is_running:
                        break

                # Update metrics
                kafka_queue_size.set(self.processing_queue.qsize())
                self._update_lag_metrics()

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            kafka_errors.labels(error_type='consumer_error').inc()
            logger.error(f"Error in consumer loop: {e}", exc_info=True)
            raise
        finally:
            self._graceful_shutdown()

    def _commit_offsets(self) -> None:
        """Commits pending offsets to Kafka."""
        with self.offset_lock:
            if not self.pending_offsets:
                return

            try:
                # Build offset dict for commit
                offsets = {
                    tp: offset
                    for tp, offset in self.pending_offsets.items()
                }

                # Commit
                self.consumer.commit(offsets=offsets)
                logger.debug(f"Committed offsets for {len(offsets)} partitions")

                # Clear pending offsets
                self.pending_offsets.clear()

            except Exception as e:
                logger.error(f"Error committing offsets: {e}")
                kafka_errors.labels(error_type='commit_error').inc()

    def _graceful_shutdown(self) -> None:
        """Performs graceful shutdown, waiting for in-flight messages."""
        logger.info("Initiating graceful shutdown...")

        # Stop accepting new messages
        self.stop()

        # Wait for active futures to complete
        logger.info(f"Waiting for {len(self.active_futures)} active tasks to complete...")
        with self.futures_lock:
            for future in self.active_futures:
                try:
                    future.result(timeout=30)  # 30 second timeout per task
                except Exception as e:
                    logger.error(f"Error waiting for task completion: {e}")

        # Commit final offsets
        self._commit_offsets()

        # Shutdown thread pool
        self.executor.shutdown(wait=True, cancel_futures=False)
        logger.info("Thread pool shutdown complete")

        # Disconnect from Kafka
        self.disconnect()
        logger.info("Graceful shutdown complete")

    def stop(self) -> None:
        """Stops consuming messages."""
        self.is_running = False
        self._shutdown_event.set()
        logger.info("Consumer stop requested")

    def _update_lag_metrics(self) -> None:
        """Updates consumer lag metrics."""
        if not self.consumer:
            return

        try:
            partitions = self.consumer.assignment()
            for partition in partitions:
                current_position = self.consumer.position(partition)
                end_offsets = self.consumer.end_offsets([partition])
                high_water_mark = end_offsets[partition]
                lag = high_water_mark - current_position

                kafka_consumer_lag.labels(
                    topic=partition.topic,
                    partition=partition.partition
                ).set(lag)

        except Exception as e:
            logger.debug(f"Error updating lag metrics: {e}")

    def get_consumer_lag(self) -> Dict[int, int]:
        """
        Gets current consumer lag for all partitions.

        Returns:
            Dict[int, int]: Map of partition -> lag.
        """
        if not self.consumer:
            return {}

        lag_info = {}
        try:
            partitions = self.consumer.assignment()
            for partition in partitions:
                current_position = self.consumer.position(partition)
                end_offsets = self.consumer.end_offsets([partition])
                lag = end_offsets[partition] - current_position
                lag_info[partition.partition] = lag
        except Exception as e:
            logger.error(f"Error calculating lag: {e}")

        return lag_info

    def get_stats(self) -> Dict[str, Any]:
        """
        Gets consumer statistics.

        Returns:
            Dict[str, Any]: Consumer stats including thread pool info.
        """
        with self.futures_lock:
            active_tasks = len([f for f in self.active_futures if not f.done()])

        return {
            'is_running': self.is_running,
            'topic': self.topic,
            'group_id': self.group_id,
            'num_worker_threads': self.num_worker_threads,
            'active_tasks': active_tasks,
            'queue_size': self.processing_queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'consumer_lag': self.get_consumer_lag()
        }

    def seek_to_beginning(self) -> None:
        """Resets consumer to beginning of all partitions."""
        if self.consumer:
            self.consumer.seek_to_beginning()
            logger.info("Consumer seeked to beginning")

    def seek_to_end(self) -> None:
        """Moves consumer to end of all partitions."""
        if self.consumer:
            self.consumer.seek_to_end()
            logger.info("Consumer seeked to end")
