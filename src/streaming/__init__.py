"""
Streaming module for real-time blockchain anomaly detection.
"""

from .kafka_consumer import KafkaConsumerService
from .stream_processor import StreamProcessor

__all__ = ["KafkaConsumerService", "StreamProcessor"]
