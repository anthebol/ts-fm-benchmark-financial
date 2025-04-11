import numpy as np

from .base import TimeSeriesGenerator


class SineWaveGenerator(TimeSeriesGenerator):
    """Generator for pure sine wave data"""

    def __init__(self, freq=0.1, amplitude=1.0, **kwargs):
        """
        Initialize sine wave generator

        Args:
            freq (float): Frequency of sine wave
            amplitude (float): Amplitude of sine wave
        """
        super().__init__(**kwargs)
        self.freq = freq
        self.amplitude = amplitude

    def generate(self):
        """Generate sine wave data"""
        t = np.linspace(0, self.seq_length - 1, self.seq_length)
        base_signal = self.amplitude * np.sin(2 * np.pi * self.freq * t)

        # Generate multiple samples with slight random variations
        data = np.zeros((self.num_samples, self.seq_length))
        for i in range(self.num_samples):
            # Add small random phase shift and amplitude variation
            phase_shift = np.random.uniform(0, 2 * np.pi)
            amp_variation = np.random.uniform(0.9, 1.1)
            data[i] = (
                amp_variation
                * self.amplitude
                * np.sin(2 * np.pi * self.freq * t + phase_shift)
            )

        return data


class CompositeSineGenerator(TimeSeriesGenerator):
    """Generator for composite sine wave data"""

    def __init__(
        self, frequencies=[0.1, 0.05, 0.02], amplitudes=[1.0, 0.5, 0.25], **kwargs
    ):
        """
        Initialize composite sine wave generator

        Args:
            frequencies (list): List of frequencies for each component
            amplitudes (list): List of amplitudes for each component
        """
        super().__init__(**kwargs)
        self.frequencies = frequencies
        self.amplitudes = amplitudes

    def generate(self):
        """Generate composite sine wave data"""
        t = np.linspace(0, self.seq_length - 1, self.seq_length)
        data = np.zeros((self.num_samples, self.seq_length))

        for i in range(self.num_samples):
            signal = np.zeros(self.seq_length)
            for freq, amp in zip(self.frequencies, self.amplitudes):
                # Add random phase shift for each component
                phase_shift = np.random.uniform(0, 2 * np.pi)
                signal += amp * np.sin(2 * np.pi * freq * t + phase_shift)

            # Add small random amplitude variation
            amp_variation = np.random.uniform(0.9, 1.1)
            data[i] = amp_variation * signal

        return data
