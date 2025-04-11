import numpy as np

from .base import TimeSeriesGenerator


class TrendGenerator(TimeSeriesGenerator):
    """Generator for time series with different trends"""

    def __init__(
        self,
        trend_type="linear",
        seasonal_amplitude=1.0,
        trend_coefficient=0.01,
        **kwargs
    ):
        """
        Initialize trend generator

        Args:
            trend_type (str): Type of trend ('linear' or 'exponential')
            seasonal_amplitude (float): Amplitude of seasonal component
            trend_coefficient (float): Steepness of trend
        """
        super().__init__(**kwargs)
        self.trend_type = trend_type
        self.seasonal_amplitude = seasonal_amplitude
        self.trend_coefficient = trend_coefficient

    def generate(self):
        """Generate trended time series data"""
        t = np.linspace(0, self.seq_length - 1, self.seq_length)
        data = np.zeros((self.num_samples, self.seq_length))

        for i in range(self.num_samples):
            # Generate seasonal component
            seasonal = self.seasonal_amplitude * np.sin(2 * np.pi * 0.05 * t)

            # Add trend
            if self.trend_type == "linear":
                trend = self.trend_coefficient * t
            else:  # exponential
                trend = np.exp(self.trend_coefficient * t) - 1

            # Combine with small random variations
            noise = np.random.normal(0, 0.1, self.seq_length)
            data[i] = trend + seasonal + noise

        return data


class ChangingPatternGenerator(TimeSeriesGenerator):
    """Generator for time series with changing patterns"""

    def __init__(self, change_points=2, **kwargs):
        """
        Initialize changing pattern generator

        Args:
            change_points (int): Number of pattern changes in the sequence
        """
        super().__init__(**kwargs)
        self.change_points = change_points

        # Calculate segment points based on change_points
        segment_length = self.seq_length // (self.change_points + 1)
        self.segment_points = [
            i * segment_length for i in range(1, self.change_points + 1)
        ]

    def generate(self):
        """Generate time series with changing patterns"""
        data = np.zeros((self.num_samples, self.seq_length))

        for i in range(self.num_samples):
            # Divide sequence into segments
            segment_length = self.seq_length // (self.change_points + 1)

            for j in range(self.change_points + 1):
                start_idx = j * segment_length
                end_idx = (
                    start_idx + segment_length
                    if j < self.change_points
                    else self.seq_length
                )

                # Different pattern for each segment
                t = np.linspace(0, end_idx - start_idx - 1, end_idx - start_idx)
                pattern_type = np.random.choice(["sine", "linear", "constant"])

                if pattern_type == "sine":
                    freq = np.random.uniform(0.05, 0.1)
                    amp = np.random.uniform(0.5, 1.5)
                    segment = amp * np.sin(2 * np.pi * freq * t)
                elif pattern_type == "linear":
                    slope = np.random.uniform(-0.1, 0.1)
                    segment = slope * t
                else:  # constant
                    level = np.random.uniform(-1, 1)
                    segment = level * np.ones_like(t)

                # Add noise
                noise = np.random.normal(0, 0.1, len(segment))
                data[i, start_idx:end_idx] = segment + noise

        return data


class ARNonstationaryGenerator:
    """Generator for non-stationary time series using AR process with varying weights"""

    def __init__(self, seq_length=900, num_samples=1000, seed=42, noise_variance=16):
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.seed = seed
        self.noise_variance = noise_variance

        # Default segment points and AR coefficients
        self.segment_points = [0, 300, 600, 900]
        self.ar_coefficients = [
            [0.8, -0.5],  # First segment (Stable)
            [0.6, -0.2],  # Second segment (Stable)
            [0.4, -0.3],  # Third segment (Stable)
        ]

    def generate_ar_sequence(self):
        """
        Generate a non-stationary signal with varying AR coefficients.
        Returns:
            np.ndarray: Generated non-stationary signal.
        """
        # Initialize signal and input
        x = np.random.normal(0, 1, self.seq_length)
        q = np.random.normal(0, np.sqrt(self.noise_variance), self.seq_length)
        y = np.zeros(self.seq_length)

        # Generate signal for each segment
        for seg_idx in range(len(self.segment_points) - 1):
            start = self.segment_points[seg_idx]
            end = self.segment_points[seg_idx + 1]
            w = self.ar_coefficients[seg_idx]

            for n in range(start, end):
                if n >= 2:
                    y[n] = w[0] * x[n] + w[1] * x[n - 1] + q[n]
                elif n == 1:
                    y[n] = w[0] * x[n] + q[n]
                else:  # n == 0
                    y[n] = q[n]

        return y

    def _generate_time_features(self):
        """
        Generate time features for the sequence.
        Returns:
            np.ndarray: Time features of shape (seq_length, feature_dim)
        """
        time_index = np.arange(self.seq_length) / self.seq_length
        # Example time features: sine and cosine of normalized time index
        time_features = np.stack(
            [np.sin(2 * np.pi * time_index), np.cos(2 * np.pi * time_index)], axis=1
        )  # Shape: (seq_length, 2)
        return time_features

    def generate(self):
        """Generate non-stationary time series with varying AR coefficients and time features"""
        data = np.zeros((self.num_samples, self.seq_length))
        time_features = np.zeros(
            (self.num_samples, self.seq_length, 2)
        )  # Assuming 2 time features

        for i in range(self.num_samples):
            # Generate each sample using the AR sequence
            data[i] = self.generate_ar_sequence()
            # Generate time features for this sample
            time_features[i] = self._generate_time_features()

        return data, time_features
