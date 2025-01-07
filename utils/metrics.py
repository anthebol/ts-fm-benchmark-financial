import numpy as np


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    result[np.isnan(result)] = 0.0
    result[np.isinf(result)] = 0.0
    return result


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def MASE(pred, true, insample, freq=1):
    """
    Calculate Mean Absolute Scaled Error (MASE)

    Args:
        pred: Predicted values
        true: True values for the prediction period
        insample: Historical/in-sample values used for scaling
        freq: Seasonal period (default=1 for non-seasonal time series)

    Returns:
        MASE value
    """
    if len(insample.shape) == 1:
        insample = insample.reshape(-1, 1)

    # Calculate the mean absolute error of the naive forecast on the training data
    masep = np.mean(np.abs(insample[freq:] - insample[:-freq]), axis=0)

    # Calculate MASE
    return np.mean(divide_no_nan(np.abs(true - pred), masep))


def metric(pred, true, insample=None, freq=1):
    """
    Calculate multiple error metrics

    Args:
        pred: Predicted values
        true: True values
        insample: Optional in-sample values for MASE calculation
        freq: Frequency for MASE calculation (default=1)

    Returns:
        mae, mse, rmse, mape, mspe, mase (if insample provided)
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    if insample is not None:
        mase = MASE(pred, true, insample, freq)
        return mae, mse, rmse, mape, mspe, mase

    return mae, mse, rmse, mape, mspe


def calculate_metrics(actual, predicted, insample=None):

    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()

    # Calculate basic metrics
    mae, mse, rmse, mape, mspe = metric(predicted, actual)

    # Calculate SMAPE
    smape = 200 * np.mean(
        np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual))
    )

    # Calculate MASE
    if insample is not None:
        naive_forecast = insample[:-1]
        naive_target = insample[1:]
        naive_mae = np.mean(np.abs(naive_target - naive_forecast))
        mase = mae / naive_mae if naive_mae != 0 else np.nan
    else:
        mase = np.nan

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape * 100,
        "SMAPE": smape,
        "MASE": mase if not np.isnan(mase) else None,
    }

    return metrics
