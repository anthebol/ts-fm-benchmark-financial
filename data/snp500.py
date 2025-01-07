import numpy as np
import pandas as pd
import yfinance as yf

from utils.normalizer import Normalizer

# Define the ticker symbol and date range
ticker = "^GSPC"
start_date = "2004-12-23"
end_date = "2024-12-23"


def prepare_dataset(df):
    """
    Prepare dataset with various transformations

    Args:
        df: DataFrame with at least Close prices

    Returns:
        DataFrame with additional columns for different transformations
    """
    # Create a copy to avoid modifying the original
    data = df.copy()

    # Reset index if Date is in index
    if isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index()

    # Simple returns
    data["Return"] = data["Close"].pct_change()

    # Log returns
    data["LogReturn"] = np.log(data["Close"]).diff()

    # Normalized prices using standardization
    normalizer = Normalizer(norm_type="standardization")
    data["NormalizedPrice"] = normalizer.normalize(data[["Close"]])

    # Normalized prices using min-max scaling
    normalizer_minmax = Normalizer(norm_type="minmax")
    data["NormalizedPriceMinMax"] = normalizer_minmax.normalize(data[["Close"]])

    # Store normalization parameters for later use
    data.attrs["price_mean"] = normalizer.mean
    data.attrs["price_std"] = normalizer.std
    data.attrs["price_min"] = normalizer_minmax.min_val
    data.attrs["price_max"] = normalizer_minmax.max_val

    # Drop NaN values created by differencing
    data = data.dropna()

    return data


# Download and prepare data at different frequencies
# Daily data
snp500_daily_raw = yf.download(ticker, start=start_date, end=end_date, interval="1d")
snp500_daily = prepare_dataset(snp500_daily_raw)

# Weekly data
snp500_weekly_raw = yf.download(ticker, start=start_date, end=end_date, interval="1wk")
snp500_weekly = prepare_dataset(snp500_weekly_raw)

# Monthly data
snp500_monthly_raw = yf.download(ticker, start=start_date, end=end_date, interval="1mo")
snp500_monthly = prepare_dataset(snp500_monthly_raw)
