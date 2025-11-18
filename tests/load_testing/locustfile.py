"""
Load testing suite using Locust.

Target: 10,000 RPS sustained throughput
Target: <200ms p95 latency

Usage:
    # Install locust
    pip install locust

    # Run load test
    locust -f tests/load_testing/locustfile.py --host=http://localhost:8000

    # Headless mode
    locust -f tests/load_testing/locustfile.py \
        --host=http://localhost:8000 \
        --users 1000 \
        --spawn-rate 100 \
        --run-time 5m \
        --headless

    # Target: 10K RPS
    locust -f tests/load_testing/locustfile.py \
        --host=http://localhost:8000 \
        --users 5000 \
        --spawn-rate 500 \
        --run-time 10m \
        --headless
"""

from locust import HttpUser, task, between, events
import json
import random
import time
from datetime import datetime, timedelta


class AnomalyDetectionUser(HttpUser):
    """
    Simulated user for load testing the anomaly detection API.

    Simulates realistic user behavior with:
    - Authentication
    - Predictions
    - Anomaly queries
    - Batch operations
    """

    # Wait between 1-3 seconds between tasks
    wait_time = between(1, 3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token = None
        self.transaction_hashes = []

    def on_start(self):
        """Called when a user starts. Perform authentication."""
        # For load testing, we might skip real auth or use a test token
        # In a real scenario, you'd authenticate here
        self.token = "test-token"  # Replace with actual auth

        # Generate sample transaction hashes
        self.transaction_hashes = [
            f"0x{random.randint(0, 2**256):064x}"
            for _ in range(100)
        ]

    @task(10)
    def predict_single_transaction(self):
        """
        Test single transaction prediction endpoint.

        Weight: 10 (most common operation)
        """
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        # Generate realistic transaction data
        payload = {
            "hash": random.choice(self.transaction_hashes),
            "value": random.uniform(0.1, 1000.0),
            "gas": random.uniform(21000, 100000),
            "gasPrice": random.uniform(10, 100)
        }

        with self.client.post(
            "/api/v1/predict",
            json=payload,
            headers=headers,
            catch_response=True,
            name="/api/v1/predict [single]"
        ) as response:
            if response.status_code == 200:
                # Verify response structure
                data = response.json()
                if "is_anomaly" in data and "confidence" in data:
                    response.success()
                else:
                    response.failure("Invalid response structure")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(5)
    def batch_predict(self):
        """
        Test batch prediction endpoint.

        Weight: 5 (less common, but important for throughput)
        """
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        # Generate batch of 50 transactions
        transactions = [
            {
                "hash": f"0x{random.randint(0, 2**256):064x}",
                "value": random.uniform(0.1, 1000.0),
                "gas": random.uniform(21000, 100000),
                "gasPrice": random.uniform(10, 100),
                "timestamp": datetime.utcnow().isoformat()
            }
            for _ in range(50)
        ]

        payload = {"transactions": transactions}

        with self.client.post(
            "/api/v1/batch/predict",
            json=payload,
            headers=headers,
            catch_response=True,
            name="/api/v1/batch/predict [50]"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("total_processed") == 50:
                    response.success()
                else:
                    response.failure("Incorrect number of predictions")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(3)
    def get_anomalies(self):
        """
        Test anomaly list endpoint.

        Weight: 3 (moderate usage)
        """
        headers = {
            "Authorization": f"Bearer {self.token}"
        }

        params = {
            "limit": random.choice([10, 50, 100]),
            "severity": random.choice([None, "high", "medium", "low"])
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        with self.client.get(
            "/api/v1/anomalies",
            params=params,
            headers=headers,
            catch_response=True,
            name="/api/v1/anomalies"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "anomalies" in data:
                    response.success()
                else:
                    response.failure("Invalid response structure")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(2)
    def health_check(self):
        """
        Test health check endpoint.

        Weight: 2 (monitoring/health checks)
        """
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("Service not healthy")
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def get_models(self):
        """
        Test model list endpoint.

        Weight: 1 (less frequent)
        """
        headers = {
            "Authorization": f"Bearer {self.token}"
        }

        with self.client.get(
            "/api/v1/models",
            headers=headers,
            catch_response=True,
            name="/api/v1/models"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "models" in data:
                    response.success()
                else:
                    response.failure("Invalid response structure")
            else:
                response.failure(f"Status code: {response.status_code}")


class HighThroughputUser(HttpUser):
    """
    High-throughput user focused on prediction endpoints.

    Used for stress testing and maximum RPS scenarios.
    """

    # Minimal wait time for maximum throughput
    wait_time = between(0.1, 0.5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token = "test-token"

    @task
    def rapid_predictions(self):
        """Rapid-fire predictions for stress testing."""
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        payload = {
            "hash": f"0x{random.randint(0, 2**256):064x}",
            "value": random.uniform(0.1, 1000.0),
            "gas": 21000,
            "gasPrice": 20.0
        }

        self.client.post(
            "/api/v1/predict",
            json=payload,
            headers=headers,
            name="/api/v1/predict [rapid]"
        )


# ============================================================================
# Event Hooks for Metrics Collection
# ============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    print("=" * 60)
    print("LOAD TEST STARTING")
    print("=" * 60)
    print(f"Target host: {environment.host}")
    print(f"Target: 10,000 RPS sustained throughput")
    print(f"Target: <200ms p95 latency")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops."""
    print("=" * 60)
    print("LOAD TEST COMPLETED")
    print("=" * 60)

    # Print summary statistics
    stats = environment.stats
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Failure rate: {stats.total.fail_ratio:.2%}")
    print(f"RPS: {stats.total.total_rps:.0f}")
    print(f"Average response time: {stats.total.avg_response_time:.0f}ms")
    print(f"Median response time: {stats.total.median_response_time:.0f}ms")
    print(f"95th percentile: {stats.total.get_response_time_percentile(0.95):.0f}ms")
    print(f"99th percentile: {stats.total.get_response_time_percentile(0.99):.0f}ms")
    print(f"Max response time: {stats.total.max_response_time:.0f}ms")

    print("=" * 60)

    # Check if targets met
    p95 = stats.total.get_response_time_percentile(0.95)
    rps = stats.total.total_rps

    print("\nTARGET ASSESSMENT:")
    print(f"RPS Target (10K): {'✅ PASS' if rps >= 10000 else '❌ FAIL'} ({rps:.0f})")
    print(f"P95 Latency Target (<200ms): {'✅ PASS' if p95 < 200 else '❌ FAIL'} ({p95:.0f}ms)")
    print(f"Failure Rate Target (<1%): {'✅ PASS' if stats.total.fail_ratio < 0.01 else '❌ FAIL'} ({stats.total.fail_ratio:.2%})")
    print("=" * 60)
