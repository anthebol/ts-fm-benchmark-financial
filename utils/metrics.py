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


def z_normalized_MAE(pred, true):
    mae = np.mean(np.abs(true - pred))
    std_true = np.std(true)
    return mae / std_true


def z_normalized_MSE(pred, true):
    mse = np.mean((true - pred) ** 2)
    std_true = np.std(true)
    return mse / (std_true**2)


def z_normalized_RMSE(pred, true):
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    std_true = np.std(true)
    return rmse / std_true


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


def calculate_directional_accuracy(
    actual, predicted, historical_last_point=None, min_change_threshold=0.001
):
    """
    Calculate directional accuracy with minimum change threshold.
    Includes special handling for the first point by using historical_last_point if provided.
    """
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()

    # Ensure lengths match
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted arrays must have the same length.")

    if len(actual) <= 1:
        raise ValueError("Need at least two points to calculate directional changes.")

    # If historical last point is provided, include it for first point direction calculation
    if historical_last_point is not None:
        # Convert historical_last_point to a scalar if it's an array or other sequence
        if hasattr(historical_last_point, "__len__") and not isinstance(
            historical_last_point, str
        ):
            historical_last_point = (
                historical_last_point.item()
                if hasattr(historical_last_point, "item")
                else historical_last_point[0]
            )

        # Create extended arrays with historical last point as a scalar
        extended_actual = np.concatenate([[historical_last_point], actual])
        extended_predicted = np.concatenate([[historical_last_point], predicted])

        # Compute changes including the transition from historical to forecast
        actual_diff = extended_actual[1:] - extended_actual[:-1]
        pred_diff = extended_predicted[1:] - extended_predicted[:-1]

        # Calculate percentage changes using appropriate bases
        actual_pct = divide_no_nan(actual_diff, extended_actual[:-1])
        pred_pct = divide_no_nan(pred_diff, extended_predicted[:-1])
    else:
        # Standard calculation without historical point - will exclude first point direction
        actual_diff = actual[1:] - actual[:-1]
        pred_diff = predicted[1:] - predicted[:-1]  # Compare within predicted series

        # Calculate percentage changes
        actual_pct = divide_no_nan(actual_diff, actual[:-1])
        pred_pct = divide_no_nan(pred_diff, predicted[:-1])

    # Determine direction masks for actual and predicted
    actual_up = actual_pct > min_change_threshold
    actual_down = actual_pct < -min_change_threshold
    actual_flat = np.abs(actual_pct) <= min_change_threshold

    pred_up = pred_pct > min_change_threshold
    pred_down = pred_pct < -min_change_threshold

    # Correct directions occur when both actual and predicted agree on up/down
    correct_direction = (actual_up & pred_up) | (actual_down & pred_down)

    # Valid movements are where actual is not flat
    valid_movements = ~actual_flat

    if np.sum(valid_movements) == 0:
        return np.nan  # Avoid division by zero if all movements are flat

    accuracy = np.mean(correct_direction[valid_movements]) * 100
    return accuracy


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
    z_normalized_mae = z_normalized_MAE(pred, true)
    z_normalized_mse = z_normalized_MSE(pred, true)
    z_normalized_rmse = z_normalized_RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    if insample is not None:
        mase = MASE(pred, true, insample, freq)
        return mae, mse, rmse, mape, mspe, mase

    return (
        mae,
        mse,
        rmse,
        z_normalized_mae,
        z_normalized_mse,
        z_normalized_rmse,
        mape,
        mspe,
    )


def calculate_metrics(
    actual,
    predicted,
    insample=None,
    historical_last_point=None,
    directional_accuracy=False,
):
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()

    # Calculate basic metrics
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)

    # Calculate MAPE and SMAPE
    nonzero_actual = actual != 0
    if np.any(nonzero_actual):
        mape = (
            np.mean(
                np.abs(
                    (actual[nonzero_actual] - predicted[nonzero_actual])
                    / actual[nonzero_actual]
                )
            )
            * 100
        )
    else:
        mape = np.nan

    smape = 200 * np.mean(
        np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual))
    )

    # Calculate MASE (simplified version)
    if insample is not None and len(insample) > 1:
        # Use one-step naive forecast for scaling
        naive_forecast = insample[:-1]
        naive_target = insample[1:]
        naive_mae = np.mean(np.abs(naive_target - naive_forecast))

        # Avoid division by zero
        if naive_mae > 0:
            mase = mae / naive_mae
        else:
            mase = np.nan
    else:
        mase = np.nan

    # Calculate directional accuracy
    if directional_accuracy == True:
        if historical_last_point is not None:
            dir_acc = calculate_directional_accuracy(
                actual, predicted, historical_last_point
            )
        else:
            dir_acc = calculate_directional_accuracy(actual, predicted)
    else:
        pass

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Z-NormalizedMAE": mae / np.std(actual) if np.std(actual) > 0 else np.nan,
        "Z-NormalizedMSE": (
            mse / (np.std(actual) ** 2) if np.std(actual) > 0 else np.nan
        ),
        "Z-NormalizedRMSE": rmse / np.std(actual) if np.std(actual) > 0 else np.nan,
        "MAPE": mape,
        "SMAPE": smape,
        "MASE": mase,
        "DirectionalAccuracy": dir_acc if directional_accuracy else np.nan,
    }
