import numpy as np

from .base import TimeSeriesGenerator


class VolatilityClusteringGenerator(TimeSeriesGenerator):
    """Generator for time series with volatility clustering (GARCH-like process)"""

    def __init__(self, alpha=0.05, beta=0.85, omega=0.01, mean_return=0.0001, **kwargs):
        """
        Initialize volatility clustering generator

        Args:
            alpha (float): GARCH parameter for impact of past squared returns
            beta (float): GARCH parameter for impact of past volatility
            omega (float): GARCH parameter for base volatility
            mean_return (float): Mean daily return
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.omega = omega
        self.mean_return = mean_return

    def generate(self):
        """Generate time series with volatility clustering"""
        data = np.zeros((self.num_samples, self.seq_length))

        for i in range(self.num_samples):
            # Set a different seed for each sample to ensure diversity
            np.random.seed(self.seed + i * 100)

            # Initialize arrays
            prices = np.ones(self.seq_length) * 100
            returns = np.zeros(self.seq_length)
            volatility = np.zeros(self.seq_length)

            # Set initial volatility
            volatility[0] = 0.01

            # Generate returns
            for t in range(1, self.seq_length):
                # Generate random innovation
                z = np.random.normal(0, 1)

                # Generate return for this period with volatility capped for stability
                vol = min(0.05, np.sqrt(volatility[t - 1]))  # Cap the volatility
                returns[t] = self.mean_return + vol * z

                # Apply hard limits to returns to prevent explosive behavior
                returns[t] = max(min(returns[t], 0.05), -0.05)

                # Update volatility with stability constraints
                volatility[t] = (
                    self.omega
                    + self.alpha * (returns[t - 1] - self.mean_return) ** 2
                    + self.beta * volatility[t - 1]
                )
                volatility[t] = min(volatility[t], 0.0025)  # Cap squared volatility

                # Update price
                prices[t] = prices[t - 1] * (1 + returns[t])

            data[i] = prices

        return data


class JumpDiffusionGenerator(TimeSeriesGenerator):
    """Generator for jump-diffusion processes commonly used in financial markets"""

    def __init__(
        self,
        jump_frequency=0.01,
        jump_size_mean=0,
        jump_size_std=0.02,
        drift=0.0002,
        diffusion_std=0.01,
        **kwargs
    ):
        """
        Initialize jump-diffusion generator

        Args:
            jump_frequency (float): Probability of a jump occurring at each time step
            jump_size_mean (float): Mean of jump size
            jump_size_std (float): Standard deviation of jump size
            drift (float): Small drift parameter (daily expected return)
            diffusion_std (float): Volatility of the diffusion process
        """
        super().__init__(**kwargs)
        self.jump_frequency = jump_frequency
        self.jump_size_mean = jump_size_mean
        self.jump_size_std = jump_size_std
        self.drift = drift
        self.diffusion_std = diffusion_std

    def generate(self):
        """Generate jump-diffusion process"""
        data = np.zeros((self.num_samples, self.seq_length))

        for i in range(self.num_samples):
            # Set a different seed for each sample
            np.random.seed(self.seed + i * 100)

            # Initialize price series
            prices = np.zeros(self.seq_length)
            prices[0] = 100  # Start at 100

            # Generate jump-diffusion process
            for t in range(1, self.seq_length):
                # Diffusion component (log-normal)
                diffusion = self.drift + np.random.normal(0, self.diffusion_std)

                # Jump component
                jump = 0
                if np.random.rand() < self.jump_frequency:
                    # Make jump sign different for each sample to create diversity
                    jump_sign = 1 if (i % 2 == 0) else -1
                    jump = jump_sign * abs(
                        np.random.normal(self.jump_size_mean, self.jump_size_std)
                    )

                # Calculate return with caps to prevent extreme values
                total_return = diffusion + jump
                total_return = max(min(total_return, 0.05), -0.05)  # Cap at ±5%

                # Update price
                prices[t] = prices[t - 1] * (1 + total_return)

            data[i] = prices

        return data


class MultiSeasonalGenerator(TimeSeriesGenerator):
    """Generator for time series with multiple seasonal components"""

    def __init__(
        self,
        seasons=[5, 20, 60],
        seasonal_amplitudes=[0.005, 0.01, 0.003],
        base_volatility=0.005,
        drift=0.0001,
        **kwargs
    ):
        """
        Initialize multi-seasonal generator

        Args:
            seasons (list): List of seasonal periods (e.g., [5, 20, 60] for weekly, monthly, quarterly)
            seasonal_amplitudes (list): Amplitude for each seasonal component
            base_volatility (float): Base volatility level
            drift (float): Small drift parameter
        """
        super().__init__(**kwargs)
        self.seasons = seasons
        self.seasonal_amplitudes = seasonal_amplitudes
        self.base_volatility = base_volatility
        self.drift = drift

    def generate(self):
        """Generate multi-seasonal time series"""
        data = np.zeros((self.num_samples, self.seq_length))
        t = np.arange(self.seq_length)

        for i in range(self.num_samples):
            # Set different seed for each sample
            np.random.seed(self.seed + i * 100)

            # Initialize with a base level
            prices = np.ones(self.seq_length) * 100

            # Add seasonal components with phase shift for each sample
            for j, (period, amplitude) in enumerate(
                zip(self.seasons, self.seasonal_amplitudes)
            ):
                # Different phase for each sample and each seasonal component
                phase = 2 * np.pi * (i * 0.1 + j * 0.2)
                seasonality = amplitude * np.sin(2 * np.pi * t / period + phase)
                prices = prices * (1 + seasonality)

            # Add random walk component with drift (with controlled volatility)
            returns = np.random.normal(
                self.drift, self.base_volatility, self.seq_length
            )
            returns = np.clip(returns, -0.02, 0.02)  # Clip extreme returns

            # Apply returns
            for t in range(1, self.seq_length):
                prices[t] = prices[t] * (1 + returns[t])

            data[i] = prices

        return data


class LongMemoryGenerator(TimeSeriesGenerator):
    """Generator for time series with long memory (approximated by AR process)"""

    def __init__(
        self, persistence=0.95, volatility=0.005, mean_return=0.0001, **kwargs
    ):
        """
        Initialize long memory generator

        Args:
            persistence (float): AR coefficient for persistence (0 < persistence < 1)
            volatility (float): Base volatility level
            mean_return (float): Mean return
        """
        super().__init__(**kwargs)
        self.persistence = persistence
        self.volatility = volatility
        self.mean_return = mean_return

    def generate(self):
        """Generate long memory time series"""
        data = np.zeros((self.num_samples, self.seq_length))

        for i in range(self.num_samples):
            # Set different seed for each sample
            np.random.seed(self.seed + i * 100)

            # Initialize prices and returns
            prices = np.ones(self.seq_length) * 100
            returns = np.zeros(self.seq_length)

            # Shift the initial return level for each sample to create diversity
            returns[0] = self.mean_return + (i - 2) * 0.001

            # Generate returns with high autocorrelation (AR process)
            for t in range(1, self.seq_length):
                # AR(1) process with high persistence to approximate long memory
                returns[t] = (
                    self.mean_return
                    + self.persistence * (returns[t - 1] - self.mean_return)
                    + np.random.normal(0, self.volatility)
                )

                # Ensure stability
                returns[t] = np.clip(returns[t], -0.02, 0.02)

                # Update price
                prices[t] = prices[t - 1] * (1 + returns[t])

            data[i] = prices

        return data


class MomentumMeanReversionGenerator(TimeSeriesGenerator):
    """Generator for series alternating between momentum and mean-reversion regimes"""

    def __init__(
        self,
        regime_duration=30,
        momentum_strength=0.003,
        reversion_strength=0.003,
        volatility=0.01,
        **kwargs
    ):
        """
        Initialize generator for momentum/mean-reversion regimes

        Args:
            regime_duration (int): Average duration of each regime
            momentum_strength (float): Strength of momentum effect
            reversion_strength (float): Strength of mean reversion effect
            volatility (float): Base volatility level
        """
        super().__init__(**kwargs)
        self.regime_duration = regime_duration
        self.momentum_strength = momentum_strength
        self.reversion_strength = reversion_strength
        self.volatility = volatility

    def generate(self):
        """Generate series with alternating momentum and mean-reversion regimes"""
        data = np.zeros((self.num_samples, self.seq_length))

        for i in range(self.num_samples):
            # Set different seed for each sample
            np.random.seed(self.seed + i * 100)

            # Initialize price series
            prices = np.zeros(self.seq_length)
            prices[0] = 100

            # Initialize returns
            returns = np.zeros(self.seq_length)

            # Determine regime changes - shift start point for each sample
            regimes = np.zeros(self.seq_length)
            current_regime = i % 2  # Start with different regimes for diversity

            # Create regime blocks with varying durations
            t = 0
            while t < self.seq_length:
                # Randomize duration slightly for each regime block
                duration = self.regime_duration + np.random.randint(-5, 6)
                duration = min(duration, self.seq_length - t)

                regimes[t : t + duration] = current_regime
                current_regime = 1 - current_regime  # Switch regime
                t += duration

            # Generate returns based on regimes
            for t in range(1, self.seq_length):
                # Base return (random with small drift)
                rand_return = np.random.normal(0.0001, self.volatility)

                if regimes[t] == 0:  # Momentum regime
                    if t > 1:
                        # Return follows the sign of previous return
                        momentum_effect = self.momentum_strength * np.sign(
                            returns[t - 1]
                        )
                        returns[t] = rand_return + momentum_effect
                    else:
                        returns[t] = rand_return
                else:  # Mean-reversion regime
                    if t > 1:
                        # Return tends to reverse the previous return
                        reversion_effect = -self.reversion_strength * np.sign(
                            returns[t - 1]
                        )
                        returns[t] = rand_return + reversion_effect
                    else:
                        returns[t] = rand_return

                # Ensure stability
                returns[t] = np.clip(returns[t], -0.03, 0.03)

                # Update price
                prices[t] = prices[t - 1] * (1 + returns[t])

            data[i] = prices

        return data


class RealisticMarketGenerator(TimeSeriesGenerator):
    """Generator for realistic market-like behavior by combining multiple effects"""

    def __init__(
        self,
        trend_strength=0.0001,
        vol_cluster_strength=0.8,
        jump_frequency=0.005,
        seasonality_strength=0.005,
        long_memory_strength=0.1,
        **kwargs
    ):
        """
        Initialize realistic market generator

        Args:
            trend_strength (float): Strength of the trend component
            vol_cluster_strength (float): Strength of volatility clustering
            jump_frequency (float): Frequency of jumps
            seasonality_strength (float): Strength of seasonal components
            long_memory_strength (float): Strength of long memory component
        """
        super().__init__(**kwargs)
        self.trend_strength = trend_strength
        self.vol_cluster_strength = vol_cluster_strength
        self.jump_frequency = jump_frequency
        self.seasonality_strength = seasonality_strength
        self.long_memory_strength = long_memory_strength

    def generate(self):
        """Generate realistic market time series"""
        data = np.zeros((self.num_samples, self.seq_length))

        for i in range(self.num_samples):
            # Set different seed for each sample
            np.random.seed(self.seed + i * 100)

            # Initialize price series
            prices = np.ones(self.seq_length) * 100
            returns = np.zeros(self.seq_length)

            # Initialize volatility process (GARCH-like)
            volatility = np.ones(self.seq_length) * 0.0001  # Initial squared volatility

            # Create different trends for each sample
            trend_direction = 1 if i % 2 == 0 else -1
            trend_factor = trend_direction * self.trend_strength * (1 + 0.2 * i)

            # Add seasonality with phase shift per sample
            t = np.arange(self.seq_length)
            phase_shift = i * 0.5
            seasonality = self.seasonality_strength * (
                0.5 * np.sin(2 * np.pi * t / 5 + phase_shift)  # Weekly
                + 0.3 * np.sin(2 * np.pi * t / 20 + phase_shift * 2)  # Monthly
                + 0.2 * np.sin(2 * np.pi * t / 60 + phase_shift * 3)  # Quarterly
            )

            # Generate time series
            for t in range(1, self.seq_length):
                # Update volatility using GARCH(1,1) process with caps
                volatility[t] = min(
                    0.001,
                    0.00001
                    + 0.05 * returns[t - 1] ** 2
                    + self.vol_cluster_strength * volatility[t - 1],
                )

                # Generate random innovation
                z = np.random.normal(0, 1)

                # Base return
                returns[t] = trend_factor + np.sqrt(volatility[t]) * z

                # Add jump component with different jump characteristics per sample
                if np.random.rand() < self.jump_frequency:
                    jump_sign = (
                        1 if (np.random.rand() > 0.5 - 0.1 * i) else -1
                    )  # Bias jumps differently per sample
                    returns[t] += jump_sign * 0.01 * (1 + 0.1 * i)

                # Add seasonal component
                if t < len(seasonality):
                    returns[t] += seasonality[t]

                # Add long memory component (approximated by AR(1) with high persistence)
                if t > 1:
                    returns[t] += self.long_memory_strength * 0.9 * returns[t - 1]

                # Ensure stability with return caps
                returns[t] = np.clip(returns[t], -0.03, 0.03)

                # Update price
                prices[t] = prices[t - 1] * (1 + returns[t])

            data[i] = prices

        return data
