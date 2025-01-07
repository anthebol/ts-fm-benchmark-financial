# experiment_runner.py
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utils.metrics import MASE, metric
from utils.normalizer import Normalizer


@dataclass
class ExperimentConfig:
    """Configuration for different frequency experiments"""

    daily_config = {
        "context_windows": [96, 192, 336],
        "short_term_windows": [48, 64],
        "long_term_windows": [96, 192, 336],
    }

    weekly_config = {
        "context_windows": [12, 24, 48],
        "short_term_windows": [12, 24],
        "long_term_windows": [48, 64, 96],
    }

    monthly_config = {
        "context_windows": [6, 12, 24],
        "short_term_windows": [3, 6],
        "long_term_windows": [12, 24],
    }


class ContextPredictionWindowEvaluator:
    """
    Generic class to run time series forecasting experiments at different frequencies
    """

    def __init__(self, model, forecast_fn, config=None):
        """
        Initialize the experiment runner

        Args:
            model: Any forecasting model that can generate predictions
            forecast_fn: Function that takes (model, context, prediction_window) and returns
                        (low_forecast, median_forecast, high_forecast)
        """
        self.config = (
            config if config else ExperimentConfig()
        )  # Use passed config or default
        self.normalizer = Normalizer(norm_type="standardization")
        self.model = model
        self.forecast_fn = forecast_fn

    def _generate_forecast(
        self, context: np.ndarray, prediction_window: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecast using the provided model and forecast function"""
        context_normalized = self.normalizer.normalize(
            pd.DataFrame(context, columns=["Close"])
        ).values.flatten()

        low, median, high = self.forecast_fn(
            self.model, context_normalized, prediction_window
        )

        # Denormalize predictions
        low_denorm = self.normalizer.denormalize(
            pd.DataFrame(low, columns=["Close"])
        ).values.flatten()
        median_denorm = self.normalizer.denormalize(
            pd.DataFrame(median, columns=["Close"])
        ).values.flatten()
        high_denorm = self.normalizer.denormalize(
            pd.DataFrame(high, columns=["Close"])
        ).values.flatten()

        return low_denorm, median_denorm, high_denorm

    def _calculate_metrics(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        insample: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        actual = np.array(actual).flatten()
        predicted = np.array(predicted).flatten()

        # Calculate basic metrics
        mae, mse, rmse, mape, mspe = metric(predicted, actual)

        # Calculate MASE separately
        mase = MASE(predicted, actual, insample) if insample is not None else None

        # Calculate SMAPE
        smape = 200 * np.mean(
            np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual))
        )

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape * 100,  # Convert to percentage
            "SMAPE": smape,
            "MASE": mase,
        }

    def _run_experiment(
        self, data: pd.DataFrame, context_window: int, prediction_window: int
    ) -> Dict[str, float]:
        """Run a single experiment"""
        # Prepare context and actual values
        context = data["Close"].values[
            -context_window - prediction_window : -prediction_window
        ]
        actual_values = data["Close"].values[-prediction_window:]
        insample_data = data["Close"].values[
            -context_window - prediction_window : -prediction_window
        ]

        # Generate forecast
        _, median_forecast, _ = self._generate_forecast(context, prediction_window)

        # Calculate metrics
        return self._calculate_metrics(actual_values, median_forecast, insample_data)

    def run_frequency_experiments(
        self, data: pd.DataFrame, frequency: str
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Run experiments for a specific frequency

        Args:
            data: DataFrame with 'Close' prices
            frequency: One of 'daily', 'weekly', or 'monthly'
        """
        config = getattr(self.config, f"{frequency}_config")
        results = {"short_term": {}, "long_term": {}}

        # Run experiments for each context window
        for context_window in config["context_windows"]:
            # Short-term experiments
            short_term_results = {}
            for pred_window in config["short_term_windows"]:
                metrics = self._run_experiment(data, context_window, pred_window)
                short_term_results[f"pred_{pred_window}"] = metrics
            results["short_term"][f"context_{context_window}"] = short_term_results

            # Long-term experiments
            long_term_results = {}
            for pred_window in config["long_term_windows"]:
                metrics = self._run_experiment(data, context_window, pred_window)
                long_term_results[f"pred_{pred_window}"] = metrics
            results["long_term"][f"context_{context_window}"] = long_term_results

        return results

    def run_frequency_experiments_context_cannot_be_snaller_than_prediction(
        self, data: pd.DataFrame, frequency: str
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Run experiments for a specific frequency

        Args:
            data: DataFrame with 'Close' prices
            frequency: One of 'daily', 'weekly', or 'monthly'
        """
        config = getattr(self.config, f"{frequency}_config")
        results = {"short_term": {}, "long_term": {}}

        # Run experiments for each context window
        for context_window in config["context_windows"]:
            # Validate context and prediction windows
            if any(
                pred_window >= context_window
                for pred_window in config["short_term_windows"]
            ):
                print(f"Skipping invalid combination: context_window={context_window}")
                continue

            # Short-term experiments
            short_term_results = {}
            for pred_window in config["short_term_windows"]:
                if pred_window >= context_window:
                    print(
                        f"Skipping invalid combination: context_window={context_window}, prediction_window={pred_window}"
                    )
                    continue
                metrics = self._run_experiment(data, context_window, pred_window)
                short_term_results[f"pred_{pred_window}"] = metrics
            results["short_term"][f"context_{context_window}"] = short_term_results

            # Long-term experiments
            long_term_results = {}
            for pred_window in config["long_term_windows"]:
                if pred_window >= context_window:
                    print(
                        f"Skipping invalid combination: context_window={context_window}, prediction_window={pred_window}"
                    )
                    continue
                metrics = self._run_experiment(data, context_window, pred_window)
                long_term_results[f"pred_{pred_window}"] = metrics
            results["long_term"][f"context_{context_window}"] = long_term_results

        return results

    def format_results(self, results: Dict) -> pd.DataFrame:
        """Format results into a pandas DataFrame"""
        rows = []
        for horizon in ["short_term", "long_term"]:
            for context_key, context_results in results[horizon].items():
                context_window = int(context_key.split("_")[1])
                for pred_key, metrics in context_results.items():
                    pred_window = int(pred_key.split("_")[1])
                    for metric_name, value in metrics.items():
                        rows.append(
                            {
                                "Horizon": horizon,
                                "Context Window": context_window,
                                "Prediction Window": pred_window,
                                "Metric": metric_name,
                                "Value": value,
                            }
                        )

        return pd.DataFrame(rows)


class LogReturnsExperiments(ContextPredictionWindowEvaluator):
    """Extension of TimeSeriesExperiments for log returns"""

    def _run_experiment(
        self, data: pd.DataFrame, context_window: int, prediction_window: int
    ):
        """Run a single experiment using log returns"""
        # Prepare context and actual values using LogReturn column
        context = data["LogReturn"].values[
            -context_window - prediction_window : -prediction_window
        ]
        actual_values = data["LogReturn"].values[-prediction_window:]
        insample_data = data["LogReturn"].values[
            -context_window - prediction_window : -prediction_window
        ]

        # Generate forecast
        _, median_forecast, _ = self._generate_forecast(context, prediction_window)

        # Calculate metrics
        return self._calculate_metrics(actual_values, median_forecast, insample_data)

    def _generate_forecast(self, context: np.ndarray, prediction_window: int):
        """Generate forecast using log returns"""
        # Normalize log returns
        context_normalized = self.normalizer.normalize(
            pd.DataFrame(context, columns=["LogReturn"])
        ).values.flatten()

        low, median, high = self.forecast_fn(
            self.model, context_normalized, prediction_window
        )

        # Denormalize predictions
        low_denorm = self.normalizer.denormalize(
            pd.DataFrame(low, columns=["LogReturn"])
        ).values.flatten()
        median_denorm = self.normalizer.denormalize(
            pd.DataFrame(median, columns=["LogReturn"])
        ).values.flatten()
        high_denorm = self.normalizer.denormalize(
            pd.DataFrame(high, columns=["LogReturn"])
        ).values.flatten()

        return low_denorm, median_denorm, high_denorm


class NormalizedReturnExperiments(ContextPredictionWindowEvaluator):
    """Extension of TimeSeriesExperiments for log returns"""

    def _run_experiment(
        self, data: pd.DataFrame, context_window: int, prediction_window: int
    ):
        """Run a single experiment using log returns"""
        # Prepare context and actual values using LogReturn column
        context = data["NormalizedPrice"].values[
            -context_window - prediction_window : -prediction_window
        ]
        actual_values = data["NormalizedPrice"].values[-prediction_window:]
        insample_data = data["NormalizedPrice"].values[
            -context_window - prediction_window : -prediction_window
        ]

        # Generate forecast
        _, median_forecast, _ = self._generate_forecast(context, prediction_window)

        # Calculate metrics
        return self._calculate_metrics(actual_values, median_forecast, insample_data)

    def _generate_forecast(self, context: np.ndarray, prediction_window: int):
        """Generate forecast using log returns"""
        # Normalize log returns
        context_normalized = self.normalizer.normalize(
            pd.DataFrame(context, columns=["NormalizedPrice"])
        ).values.flatten()

        low, median, high = self.forecast_fn(
            self.model, context_normalized, prediction_window
        )

        # Denormalize predictions
        low_denorm = self.normalizer.denormalize(
            pd.DataFrame(low, columns=["NormalizedPrice"])
        ).values.flatten()
        median_denorm = self.normalizer.denormalize(
            pd.DataFrame(median, columns=["NormalizedPrice"])
        ).values.flatten()
        high_denorm = self.normalizer.denormalize(
            pd.DataFrame(high, columns=["NormalizedPrice"])
        ).values.flatten()

        return low_denorm, median_denorm, high_denorm
