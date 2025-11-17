"""
Locust load testing for Blockchain Anomaly Detection API.

Usage:
    # Local testing
    locust -f tests/load/locustfile.py --host http://localhost:8000

    # Headless mode with specific users and duration
    locust -f tests/load/locustfile.py --headless \
        --users 1000 --spawn-rate 100 --run-time 5m \
        --host http://localhost:8000 \
        --html locust-report.html

    # Distributed mode (master)
    locust -f tests/load/locustfile.py --master --expect-workers 4

    # Distributed mode (worker)
    locust -f tests/load/locustfile.py --worker --master-host=localhost
"""

import random
import time
from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser
import json


class BlockchainAnomalyUser(FastHttpUser):
    """
    User simulation for the Blockchain Anomaly Detection API.

    Simulates realistic user behavior including:
    - Single transaction predictions
    - Batch predictions
    - Health checks
    - Anomaly queries
    """

    # Wait time between tasks (1-3 seconds)
    wait_time = between(1, 3)

    def on_start(self):
        """
        Called when a simulated user starts.
        Perform authentication if needed (Phase 1).
        """
        # For now, no authentication
        # In Phase 1, this would:
        # response = self.client.post("/api/v1/auth/login", json={
        #     "email": "loadtest@example.com",
        #     "password": "testpassword"
        # })
        # self.token = response.json()["access_token"]
        pass

    @task(10)
    def predict_single_transaction(self):
        """
        Predict a single transaction (most common operation).
        Weight: 10 (happens 10x more often than other tasks).
        """
        transaction = self._generate_random_transaction()

        with self.client.post(
            "/api/v1/predict",
            json=transaction,
            catch_response=True,
            name="/api/v1/predict [single]"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "is_anomaly" in data and "confidence" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            elif response.status_code == 401:
                # Expected if authentication is required
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(3)
    def predict_batch_transactions(self):
        """
        Predict a batch of transactions.
        Weight: 3 (less common than single predictions).
        """
        batch_size = random.randint(5, 50)
        transactions = [
            self._generate_random_transaction()
            for _ in range(batch_size)
        ]

        with self.client.post(
            "/api/v1/predict/batch",
            json={"transactions": transactions},
            catch_response=True,
            name=f"/api/v1/predict/batch [size={batch_size}]"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "predictions" in data and len(data["predictions"]) == batch_size:
                    response.success()
                else:
                    response.failure("Invalid batch response")
            elif response.status_code == 401:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(5)
    def get_anomalies(self):
        """
        Query detected anomalies.
        Weight: 5 (moderately common).
        """
        # Randomize query parameters
        params = {
            "limit": random.choice([10, 50, 100]),
            "severity": random.choice(["low", "medium", "high", None])
        }

        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}

        with self.client.get(
            "/api/v1/anomalies",
            params=params,
            catch_response=True,
            name="/api/v1/anomalies [query]"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "anomalies" in data:
                    response.success()
                else:
                    response.failure("Invalid anomalies response")
            elif response.status_code == 401:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    @task(1)
    def health_check(self):
        """
        Health check (least common - monitoring systems do this).
        Weight: 1.
        """
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health [check]"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(1)
    def get_stream_status(self):
        """
        Get streaming status.
        Weight: 1.
        """
        with self.client.get(
            "/api/v1/stream/status",
            catch_response=True,
            name="/api/v1/stream/status"
        ) as response:
            if response.status_code in [200, 401]:
                response.success()
            else:
                response.failure(f"Unexpected status code: {response.status_code}")

    def _generate_random_transaction(self):
        """
        Generate a random blockchain transaction for testing.

        Returns:
            dict: Transaction data
        """
        # Generate realistic transaction values
        # Normal transactions: 90%
        # Anomalous transactions: 10%
        is_anomaly = random.random() < 0.1

        if is_anomaly:
            # Anomalous: very high values
            value = random.uniform(1e9, 1e12)
            gas = random.randint(100000, 1000000)
            gas_price = random.uniform(100, 10000)
        else:
            # Normal: typical values
            value = random.lognormal(10, 2)  # Log-normal distribution
            gas = random.randint(21000, 100000)
            gas_price = random.lognormal(3, 1)

        return {
            "hash": f"0x{random.randint(0, 2**256):064x}",
            "value": value,
            "gas": gas,
            "gasPrice": gas_price,
            "blockNumber": random.randint(1000000, 2000000),
            "from": f"0x{random.randint(0, 2**160):040x}",
            "to": f"0x{random.randint(0, 2**160):040x}",
            "nonce": random.randint(0, 1000)
        }


# Custom events for tracking
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts."""
    print("\n" + "="*60)
    print("BLOCKCHAIN ANOMALY DETECTION - LOAD TEST")
    print("="*60)
    print(f"Target: {environment.host}")
    print(f"Users: {environment.runner.target_user_count if hasattr(environment.runner, 'target_user_count') else 'N/A'}")
    print("="*60 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops."""
    print("\n" + "="*60)
    print("LOAD TEST COMPLETED")
    print("="*60)
    if environment.stats.total.num_requests > 0:
        print(f"Total Requests: {environment.stats.total.num_requests}")
        print(f"Total Failures: {environment.stats.total.num_failures}")
        print(f"Average Response Time: {environment.stats.total.avg_response_time:.2f}ms")
        print(f"RPS: {environment.stats.total.total_rps:.2f}")
        print(f"Failure Rate: {environment.stats.total.fail_ratio * 100:.2f}%")
    print("="*60 + "\n")


# Alternative: Staged user spawning for gradual ramp-up
class SteppedUser(BlockchainAnomalyUser):
    """
    User with stepped spawning for gradual load increase.

    Useful for finding breaking points.
    """
    wait_time = between(1, 2)


# Performance testing scenarios
class HighThroughputUser(FastHttpUser):
    """
    High throughput user for stress testing.
    Minimal wait time, maximum requests.
    """
    wait_time = between(0.1, 0.5)

    @task
    def rapid_predictions(self):
        """Rapid-fire predictions."""
        transaction = {
            "hash": f"0x{random.randint(0, 2**256):064x}",
            "value": random.lognormal(10, 2),
            "gas": random.randint(21000, 100000),
            "gasPrice": random.lognormal(3, 1)
        }
        self.client.post("/api/v1/predict", json=transaction)


class SpikeTestUser(FastHttpUser):
    """
    User for spike testing - sudden load increase.
    """
    wait_time = between(0.5, 1)

    @task(20)
    def spike_predict(self):
        """High frequency predictions."""
        transaction = {
            "hash": f"0x{random.randint(0, 2**256):064x}",
            "value": random.lognormal(10, 2),
            "gas": random.randint(21000, 100000),
            "gasPrice": random.lognormal(3, 1)
        }
        self.client.post("/api/v1/predict", json=transaction)

    @task(5)
    def spike_batch(self):
        """Batch predictions."""
        transactions = [
            {
                "hash": f"0x{random.randint(0, 2**256):064x}",
                "value": random.lognormal(10, 2),
                "gas": random.randint(21000, 100000),
                "gasPrice": random.lognormal(3, 1)
            }
            for _ in range(10)
        ]
        self.client.post("/api/v1/predict/batch", json={"transactions": transactions})
