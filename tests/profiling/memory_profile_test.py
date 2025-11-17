"""
Memory profiling for blockchain anomaly detection system.

Usage:
    python -m memory_profiler tests/profiling/memory_profile_test.py

    # Or with matplotlib for visualization:
    mprof run tests/profiling/memory_profile_test.py
    mprof plot
"""

from memory_profiler import profile
import pandas as pd
import numpy as np
from src.anomaly_detection.isolation_forest import AnomalyDetectorIsolationForest


@profile
def test_large_dataset_memory():
    """Profile memory usage with large dataset."""
    # Generate large dataset (100K transactions)
    print("Generating 100K transactions...")
    data = pd.DataFrame({
        'value': np.random.lognormal(10, 2, 100000),
        'gas': np.random.randint(21000, 100000, 100000),
        'gasPrice': np.random.lognormal(3, 1, 100000),
    })

    print(f"Dataset size: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # Train model
    print("Training model...")
    detector = AnomalyDetectorIsolationForest(contamination=0.01)
    detector.train_model(data)

    # Detect anomalies
    print("Detecting anomalies...")
    results = detector.detect_anomalies(data)

    print(f"Found {len(results[results['anomaly'] == -1])} anomalies")


@profile
def test_batch_processing_memory():
    """Profile memory usage during batch processing."""
    print("Testing batch processing memory...")

    # Process in batches to test memory management
    batch_size = 10000
    total_samples = 50000

    detector = None

    for i in range(0, total_samples, batch_size):
        print(f"Processing batch {i // batch_size + 1}...")

        batch = pd.DataFrame({
            'value': np.random.lognormal(10, 2, batch_size),
            'gas': np.random.randint(21000, 100000, batch_size),
            'gasPrice': np.random.lognormal(3, 1, batch_size),
        })

        if detector is None:
            detector = AnomalyDetectorIsolationForest(contamination=0.01)
            detector.train_model(batch)
        else:
            results = detector.detect_anomalies(batch)
            anomalies = len(results[results['anomaly'] == -1])
            print(f"  Found {anomalies} anomalies in batch")

        # Clear batch to free memory
        del batch


if __name__ == '__main__':
    print("="*60)
    print("MEMORY PROFILING - LARGE DATASET TEST")
    print("="*60)
    test_large_dataset_memory()

    print("\n" + "="*60)
    print("MEMORY PROFILING - BATCH PROCESSING TEST")
    print("="*60)
    test_batch_processing_memory()

    print("\n" + "="*60)
    print("PROFILING COMPLETE")
    print("="*60)
