# Load Testing Suite

This directory contains load testing scripts for the Blockchain Anomaly Detection API.

## Prerequisites

```bash
pip install locust
```

## Quick Start

### 1. Start the API Server

```bash
uvicorn src.api_server.app:app --host 0.0.0.0 --port 8000
```

### 2. Run Load Test (Interactive)

```bash
locust -f tests/load_testing/locustfile.py --host=http://localhost:8000
```

Then open http://localhost:8089 in your browser and configure:
- Number of users: 1000
- Spawn rate: 100
- Host: http://localhost:8000

### 3. Run Load Test (Headless)

```bash
locust -f tests/load_testing/locustfile.py \
    --host=http://localhost:8000 \
    --users 1000 \
    --spawn-rate 100 \
    --run-time 5m \
    --headless
```

## Performance Targets

### Primary Targets
- **Throughput**: 10,000 RPS sustained
- **Latency (P95)**: <200ms
- **Error Rate**: <1%

### Secondary Targets
- **Latency (P50)**: <100ms
- **Latency (P99)**: <500ms
- **Memory**: Stable under load
- **CPU**: <80% utilization

## Test Scenarios

### 1. Baseline Test (1K RPS)

```bash
locust -f tests/load_testing/locustfile.py \
    --host=http://localhost:8000 \
    --users 200 \
    --spawn-rate 50 \
    --run-time 3m \
    --headless
```

### 2. Target Test (10K RPS)

```bash
locust -f tests/load_testing/locustfile.py \
    --host=http://localhost:8000 \
    --users 5000 \
    --spawn-rate 500 \
    --run-time 10m \
    --headless
```

### 3. Stress Test (Maximum)

```bash
locust -f tests/load_testing/locustfile.py \
    --host=http://localhost:8000 \
    --users 10000 \
    --spawn-rate 1000 \
    --run-time 10m \
    --headless
```

### 4. Spike Test

```bash
# Quick ramp-up to test burst handling
locust -f tests/load_testing/locustfile.py \
    --host=http://localhost:8000 \
    --users 5000 \
    --spawn-rate 2500 \
    --run-time 2m \
    --headless
```

## User Types

### AnomalyDetectionUser
Simulates realistic user behavior:
- 50% single predictions
- 25% batch predictions
- 15% anomaly queries
- 10% health checks

### HighThroughputUser
Optimized for maximum RPS:
- 100% rapid predictions
- Minimal wait time
- Used for stress testing

## Monitoring During Tests

### 1. Prometheus Metrics

```bash
# View metrics
curl http://localhost:8000/metrics
```

Key metrics to watch:
- `http_request_duration_seconds`
- `http_requests_total`
- `cache_hit_rate`
- `kafka_consumer_lag`

### 2. System Metrics

```bash
# CPU and memory
htop

# Network
iftop

# Disk I/O
iotop
```

### 3. Application Logs

```bash
tail -f logs/app.log
```

## Results Interpretation

### Example Output

```
Total requests: 600000
Total failures: 120
Failure rate: 0.02%
RPS: 10053
Average response time: 95ms
Median response time: 87ms
95th percentile: 185ms
99th percentile: 420ms
Max response time: 1250ms

TARGET ASSESSMENT:
RPS Target (10K): ✅ PASS (10053)
P95 Latency Target (<200ms): ✅ PASS (185ms)
Failure Rate Target (<1%): ✅ PASS (0.02%)
```

### What to Look For

**Good Signs:**
- Stable RPS throughout test
- P95 latency <200ms
- Failure rate <1%
- Flat memory usage
- No error spikes

**Warning Signs:**
- Increasing latency over time (memory leak)
- High failure rate (errors, timeouts)
- Decreasing RPS (saturation)
- CPU >90% (need scaling)

## Optimization Tips

If targets not met:

### 1. Increase Workers

```bash
uvicorn src.api_server.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 8
```

### 2. Enable Caching

Ensure Redis is running and `REDIS_ENABLED=true`

### 3. Use Distributed Consumer

Set `KAFKA_NUM_WORKER_THREADS=8` or higher

### 4. Database Connection Pool

Increase pool size in database config:
```python
pool_size=50
max_overflow=20
```

### 5. Enable GZip Compression

Already enabled in performance middleware

### 6. Horizontal Scaling

Deploy multiple API instances behind load balancer

## Continuous Performance Testing

### CI/CD Integration

```yaml
# .github/workflows/performance-test.yml
name: Performance Test

on:
  push:
    branches: [main, develop]

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Start services
        run: docker-compose up -d
      - name: Run load test
        run: |
          pip install locust
          locust -f tests/load_testing/locustfile.py \
            --host=http://localhost:8000 \
            --users 1000 \
            --spawn-rate 100 \
            --run-time 3m \
            --headless \
            --csv=results
      - name: Check results
        run: python tests/load_testing/check_results.py results_stats.csv
```

## Troubleshooting

### Problem: Low RPS (<1000)

**Possible causes:**
- Database bottleneck
- No connection pooling
- Synchronous I/O
- No caching

**Solutions:**
- Enable Redis caching
- Use connection pooling
- Switch to async database driver
- Add indexes to frequent queries

### Problem: High Latency (>500ms P95)

**Possible causes:**
- Slow database queries
- Cache misses
- Complex computations
- Network latency

**Solutions:**
- Optimize queries with EXPLAIN
- Warm cache before test
- Profile code with cProfile
- Use CDN for static assets

### Problem: Increasing Memory

**Possible causes:**
- Memory leak
- Large objects in memory
- No garbage collection
- Unbounded cache

**Solutions:**
- Use memory profiler
- Implement TTL for cache
- Limit buffer sizes
- Force GC periodically

## Advanced Testing

### Custom Test Scenarios

Create custom test files for specific scenarios:

```python
# tests/load_testing/custom_scenario.py
from locust import HttpUser, task, between

class MyCustomUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def my_scenario(self):
        # Custom test logic
        pass
```

### Distributed Load Testing

Run Locust in distributed mode:

```bash
# Master
locust -f locustfile.py --master

# Workers (on multiple machines)
locust -f locustfile.py --worker --master-host=192.168.1.100
```

## References

- [Locust Documentation](https://docs.locust.io/)
- [Performance Testing Best Practices](https://martinfowler.com/articles/practical-test-pyramid.html)
- [Prometheus Metrics](https://prometheus.io/docs/practices/naming/)
