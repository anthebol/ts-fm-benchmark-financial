import numpy as np

from .base import TimeSeriesGenerator


class NoiseGenerator(TimeSeriesGenerator):
    """Generator for different types of noise"""

    def __init__(
        self,
        noise_type="gaussian",
        noise_params=None,
        **kwargs,
    ):
        """
        Initialize noise generator

        Args:
            noise_type (str): Type of noise ('gaussian', 'pink', 'brownian', 'heavy_tailed')
            noise_params (dict): Parameters for noise generation
        """
        super().__init__(**kwargs)
        self.noise_type = noise_type
        self.noise_params = noise_params or {}

    def generate_pink_noise(self):
        """Generate pink noise using 1/f power spectrum"""
        f = np.fft.fftfreq(self.seq_length)
        f[0] = float("inf")  # Avoid division by zero
        power_spectrum = 1 / np.abs(f)
        power_spectrum[0] = 0

        data = np.zeros((self.num_samples, self.seq_length))
        for i in range(self.num_samples):
            phases = np.random.uniform(0, 2 * np.pi, self.seq_length // 2 + 1)
            phases = np.concatenate([phases, -phases[-2:0:-1]])
            spectrum = np.sqrt(power_spectrum) * np.exp(1j * phases)
            data[i] = np.real(np.fft.ifft(spectrum))

        return data

    def generate_brownian_noise(self):
        """Generate Brownian noise (random walk)"""
        data = np.zeros((self.num_samples, self.seq_length))
        for i in range(self.num_samples):
            data[i] = np.cumsum(np.random.normal(0, 0.1, self.seq_length))
        return data

    def generate_heavy_tailed(self):
        """Generate heavy-tailed noise using Student's t-distribution"""
        # Default degrees of freedom (lower = heavier tails)
        df = self.noise_params.get("df", 3)
        # Default scale parameter
        scale = self.noise_params.get("scale", 1.0)

        data = np.zeros((self.num_samples, self.seq_length))
        for i in range(self.num_samples):
            # Generate using Student's t-distribution
            data[i] = scale * np.random.standard_t(df, size=self.seq_length)

        return data

    def generate(self):
        """Generate noise based on specified type"""
        if self.noise_type == "gaussian":
            std = self.noise_params.get("std", 1.0)
            return np.random.normal(0, std, (self.num_samples, self.seq_length))

        elif self.noise_type == "pink":
            return self.generate_pink_noise()

        elif self.noise_type == "brownian":
            return self.generate_brownian_noise()

        elif self.noise_type == "heavy_tailed":
            return self.generate_heavy_tailed()

        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")


class MixedNoiseGenerator(TimeSeriesGenerator):
    """Generator for time series with mixed noise types"""

    def __init__(self, base_signal="sine", **kwargs):
        """
        Initialize mixed noise generator

        Args:
            base_signal (str): Type of base signal ('sine' or 'constant')
        """
        super().__init__(**kwargs)
        self.base_signal = base_signal

    def generate(self):
        """Generate time series with mixed noise types"""
        t = np.linspace(0, self.seq_length - 1, self.seq_length)
        data = np.zeros((self.num_samples, self.seq_length))

        for i in range(self.num_samples):
            # Generate base signal
            if self.base_signal == "sine":
                base = np.sin(2 * np.pi * 0.05 * t)
            else:
                base = np.ones_like(t)

            # Add mixture of noise types
            gaussian = np.random.normal(0, 0.2, self.seq_length)
            pink = (
                NoiseGenerator(
                    noise_type="pink", seq_length=self.seq_length
                ).generate()[0]
                * 0.3
            )
            brownian = (
                NoiseGenerator(
                    noise_type="brownian", seq_length=self.seq_length
                ).generate()[0]
                * 0.1
            )
            heavy_tailed = (
                NoiseGenerator(
                    noise_type="heavy_tailed", seq_length=self.seq_length
                ).generate()[0]
                * 0.1
            )

            data[i] = base + gaussian + pink + brownian + heavy_tailed

        return data
