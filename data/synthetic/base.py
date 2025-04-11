from abc import ABC, abstractmethod

import numpy as np


class TimeSeriesGenerator(ABC):
    """Base class for all time series generators"""

    def __init__(self, seq_length=200, num_samples=1000, seed=42):
        """
        Initialize the generator

        Args:
            seq_length (int): Length of each time series
            num_samples (int): Number of time series to generate
            seed (int): Random seed for reproducibility
        """
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.seed = seed
        np.random.seed(seed)

    @abstractmethod
    def generate(self):
        """Generate synthetic time series data"""
        pass

    def add_noise(self, data, noise_level=0.1):
        """
        Add Gaussian noise to data

        Args:
            data (np.ndarray): Input data
            noise_level (float): Standard deviation of noise

        Returns:
            np.ndarray: Data with added noise
        """
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise
