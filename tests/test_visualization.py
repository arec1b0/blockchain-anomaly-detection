import pytest
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend for testing
from src.visualization.visualization import DataVisualizer


@pytest.fixture
def sample_data():
    data = {
        'timeStamp': pd.date_range(start='2023-01-01', periods=7, freq='D'),
        'value': [100, 200, 150, 300, 500, 600, 700],
        'anomaly': ['normal', 'normal', 'anomaly', 'normal', 'normal', 'anomaly', 'normal']
    }
    return pd.DataFrame(data)


def test_plot_time_series(sample_data):
    visualizer = DataVisualizer(sample_data)

    plt.figure()
    visualizer.plot_time_series()
    plt.close()


def test_plot_anomalies(sample_data):
    visualizer = DataVisualizer(sample_data)

    plt.figure()
    visualizer.plot_anomalies()
    plt.close()


def test_plot_distribution(sample_data):
    visualizer = DataVisualizer(sample_data)

    plt.figure()
    visualizer.plot_distribution('value')
    plt.close()
