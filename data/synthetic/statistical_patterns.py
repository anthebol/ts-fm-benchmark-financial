import numpy as np

from .base import TimeSeriesGenerator


class VaryingMeanGenerator(TimeSeriesGenerator):
    """Generator for time series with different mean levels"""

    def __init__(self, mean_level="high", seasonal=True, **kwargs):
        """
        Initialize varying mean generator

        Args:
            mean_level (str): Mean level ('high', 'low', or 'varying')
            seasonal (bool): Whether to add seasonal component
        """
        super().__init__(**kwargs)
        self.mean_level = mean_level
        self.seasonal = seasonal

    def generate(self):
        """Generate time series with different mean levels"""
        t = np.linspace(0, self.seq_length - 1, self.seq_length)
        data = np.zeros((self.num_samples, self.seq_length))

        for i in range(self.num_samples):
            # Base seasonal component if enabled
            if self.seasonal:
                base = np.sin(2 * np.pi * 0.05 * t)
            else:
                base = np.zeros_like(t)

            # Add mean level
            if self.mean_level == "high":
                mean = 10.0
            elif self.mean_level == "low":
                mean = -10.0
            else:  # varying
                mean = 5 * np.sin(2 * np.pi * 0.01 * t)

            noise = np.random.normal(0, 0.1, self.seq_length)
            data[i] = base + mean + noise

        return data


class VaryingVolatilityGenerator(TimeSeriesGenerator):
    """Generator for time series with different volatility patterns"""

    def __init__(self, volatility_type="high", **kwargs):
        """
        Initialize varying volatility generator

        Args:
            volatility_type (str): Type of volatility ('high', 'low', or 'varying')
        """
        super().__init__(**kwargs)
        self.volatility_type = volatility_type

    def generate(self):
        """Generate time series with different volatility patterns"""
        t = np.linspace(0, self.seq_length - 1, self.seq_length)
        data = np.zeros((self.num_samples, self.seq_length))

        for i in range(self.num_samples):
            # Base signal
            base = np.sin(2 * np.pi * 0.05 * t)

            # Generate volatility pattern
            if self.volatility_type == "high":
                std = 2.0
            elif self.volatility_type == "low":
                std = 0.5
            else:  # varying
                std = 1 + np.sin(2 * np.pi * 0.01 * t)

            noise = np.random.normal(0, 1, self.seq_length) * std
            data[i] = base + noise

        return data
