# Optimization Quick Start Guide

**Quick reference for implementing optimizations from the comprehensive plan**

## Priority 1: Immediate Performance Wins (Do First)

### 1. Stream Processing - Vectorized Batching
**File**: `src/streaming/stream_processor.py`
**Impact**: 83% latency reduction (300ms → 50ms)

Replace DataFrame creation with numpy arrays:
```python
# Before
df = pd.DataFrame(transactions)
results = self.model.predict(df[['value', 'gas', 'gasPrice']])

# After
self._batch_buffer = np.zeros((BATCH_SIZE, 3), dtype=np.float64)
for i, tx in enumerate(transactions):
    self._batch_buffer[i] = [tx['value'], tx['gas'], tx['gasPrice']]
results = self.model.predict(self._batch_buffer[:len(transactions)])
```

### 2. Kafka Consumer - Thread Pool
**File**: `src/streaming/kafka_consumer.py`
**Impact**: 10x throughput (500 → 5000 TPS)

```python
from concurrent.futures import ThreadPoolExecutor

class KafkaConsumerService:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)

    def consume(self, callback):
        futures = []
        for message in self.consumer:
            future = self.executor.submit(callback, message.value)
            futures.append(future)
            if len(futures) >= 100:
                concurrent.futures.wait(futures)
                futures.clear()
```

### 3. Model Persistence
**Files**: `src/anomaly_detection/model_manager.py` (new), `src/streaming/stream_processor.py`
**Impact**: 160x cold start improvement (8s → 50ms)

```python
import joblib

# Save model
joblib.dump(model, "./models/default_model.pkl")

# Load model at startup
self.model = joblib.load("./models/default_model.pkl")
```

### 4. Bounded Anomaly Buffer
**Files**: `src/utils/circular_buffer.py` (new), `src/streaming/stream_processor.py`
**Impact**: Prevents memory leaks

```python
from collections import deque

class TTLCircularBuffer:
    def __init__(self, max_size=10000):
        self._buffer = deque(maxlen=max_size)

    def append(self, item):
        self._buffer.append(item)
```

## Priority 2: Scalability Improvements

### 5. Async Batch API
**File**: `src/api_server/app.py`
**Impact**: 85% latency reduction (10s → 1.5s for 100 items)

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.post("/api/v1/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    chunk_size = 100
    chunks = [request.transactions[i:i+chunk_size]
              for i in range(0, len(request.transactions), chunk_size)]

    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(executor, process_chunk, chunk)
             for chunk in chunks]

    results = await asyncio.gather(*tasks)
    return flatten(results)
```

### 6. Feature Engineering
**File**: `src/anomaly_detection/feature_engineering.py` (new)
**Impact**: 7% accuracy improvement (85% → 92%)

```python
# Add ratio features
df['gas_price_per_unit'] = df['gasPrice'] / (df['gas'] + 1)
df['value_per_gas'] = df['value'] / (df['gas'] + 1)

# Add rolling statistics
df['value_rolling_mean'] = df['value'].rolling(10).mean()
df['value_zscore'] = (df['value'] - df['value_rolling_mean']) / df['value'].rolling(10).std()

# Add time features
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
```

### 7. Online Normalization
**File**: `src/data_processing/online_normalization.py` (new)
**Impact**: Constant memory usage for normalization

```python
class OnlineNormalizer:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0

    def update(self, value):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (value - self.mean)

    def transform(self, value):
        std = np.sqrt(self.M2 / self.n) if self.n > 1 else 1
        return (value - self.mean) / std
```

## Priority 3: Production Readiness

### 8. Docker Optimization
**File**: `docker/Dockerfile.optimized`
**Impact**: 73% image size reduction (1.5GB → 400MB)

```dockerfile
# Multi-stage build
FROM python:3.10-slim as builder
RUN python -m venv /opt/venv
COPY requirements.txt .
RUN /opt/venv/bin/pip install -r requirements.txt

FROM python:3.10-slim
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
USER appuser
CMD ["uvicorn", "src.api_server.app:app", "--workers", "4"]
```

### 9. Type Hints
**Files**: All Python files
**Impact**: Better IDE support, catch bugs early

```python
from typing import Dict, List, Optional, Tuple
import pandas as pd

def detect_anomalies(
    data: pd.DataFrame,
    features: List[str]
) -> Tuple[List[int], List[float]]:
    ...
```

### 10. Error Handling
**File**: `src/utils/exceptions.py` (new)
**Impact**: Consistent error responses

```python
class BlockchainAnomalyError(Exception):
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}

class ModelNotFoundError(BlockchainAnomalyError):
    pass
```

## Implementation Checklist

### Week 1-2: Critical Performance
- [ ] Vectorized batch processing
- [ ] Thread pool for Kafka
- [ ] Model persistence
- [ ] Bounded buffer
- [ ] Async batch API
- [ ] Basic feature engineering

### Week 3-4: Scalability
- [ ] Advanced features (30+)
- [ ] Online normalization
- [ ] Distributed consumer
- [ ] Redis caching
- [ ] Docker optimization

### Week 5-6: Code Quality
- [ ] Type hints (100%)
- [ ] Custom exceptions
- [ ] Error handlers
- [ ] Mypy strict mode
- [ ] Performance profiling

### Week 7-8: Advanced
- [ ] Async Kafka (aiokafka)
- [ ] Model versioning
- [ ] Kubernetes deployment
- [ ] Production monitoring

## Testing Strategy

### Performance Testing
```bash
# Load test API
ab -n 10000 -c 100 http://localhost:8000/api/v1/predict

# Kafka throughput test
kafka-producer-perf-test --topic blockchain-transactions \
  --num-records 10000 --record-size 1000 \
  --throughput 5000 --producer-props bootstrap.servers=localhost:9092

# Memory profiling
python -m memory_profiler src/main.py
```

### Validation
```bash
# Run tests
pytest --cov=src --cov-report=html

# Type checking
mypy src/

# Performance regression
pytest tests/test_performance.py --benchmark
```

## Key Metrics to Track

| Metric | Baseline | Target | Command |
|--------|----------|--------|---------|
| Stream latency | 300ms | <50ms | Check Grafana |
| API latency | 150ms | <100ms | `/metrics` endpoint |
| Throughput | 500 TPS | 10k TPS | Kafka consumer lag |
| Memory | 512MB | 256MB | `docker stats` |
| Image size | 1.5GB | <500MB | `docker images` |

## Quick Commands

```bash
# Start optimized stack
docker-compose up -d --scale api=3

# Monitor performance
docker stats
curl http://localhost:8000/metrics

# Test throughput
python scripts/load_test.py --tps 5000

# Profile code
python -m cProfile -o profile.stats src/main.py
python -m pstats profile.stats

# Check memory
valgrind --tool=massif python src/main.py
```

## Dependencies to Add

```txt
# requirements.txt additions
aiokafka==0.8.1          # Async Kafka
redis==5.0.1             # Distributed cache
memory-profiler==0.61.0  # Memory profiling
line-profiler==4.1.1     # Line profiling
```

## References

- Full plan: [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md)
- API docs: [API.md](API.md)
- Architecture: [README.md](../README.md)

## Support

For questions or issues during implementation:
1. Check the full optimization plan
2. Review existing tests for patterns
3. Run benchmarks before and after changes
4. Create feature flags for risky changes

---

**Last Updated**: 2025-10-27
**Version**: 1.0
