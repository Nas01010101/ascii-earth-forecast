"""Next-day forecast of daily mean cloud cover and temperature: one ridge regression per latitude band.

Each cell is predicted from its own last three days and its four neighbours yesterday,
all as departures from that cell's mean over the earlier days. Nothing from the target
day or later enters the training set, the cell means or the inputs.
"""
import numpy as np

from .fetch import LATS

MIN_TRAIN_DAYS = 14
RIDGE = 1.0
FEATURES = ("yesterday", "2 days ago", "3 days ago", "north", "south", "east", "west")
# Tropics, mid-latitudes and poles have different prevailing winds, so each gets its own weights.
BANDS = (np.abs(LATS) < 30, (np.abs(LATS) >= 30) & (np.abs(LATS) < 60), np.abs(LATS) >= 60)
AREA = np.cos(np.radians(LATS))[:, None]


def _inputs(anomaly, day):
    """Feature cube (features, lat, lon) for predicting `day`. North of the top row repeats the top row."""
    last = anomaly[day - 1]
    return np.stack([
        last, anomaly[day - 2], anomaly[day - 3],
        np.vstack([last[1:], last[-1:]]), np.vstack([last[:1], last[:-1]]),
        np.roll(last, -1, axis=1), np.roll(last, 1, axis=1),
    ])


def _solve(x, y, weight):
    xw = x * weight[:, None]
    return np.linalg.solve(xw.T @ x + RIDGE * np.eye(x.shape[1]), xw.T @ y)


def predict(series, day):
    """Forecasts series[day] from series[:day]. Returns (forecast, trailing mean)."""
    if day - 3 < MIN_TRAIN_DAYS:
        raise ValueError(f"needs {MIN_TRAIN_DAYS + 3} days before the target, got {day}")
    climatology = series[:day].mean(axis=0)
    anomaly = series[:day] - climatology
    forecast = climatology.copy()
    inputs = _inputs(anomaly, day)
    n_features = len(FEATURES)
    train = np.stack([_inputs(anomaly, d) for d in range(3, day)])
    target = anomaly[3:day]
    for band in BANDS:
        x = train[:, :, band].transpose(0, 2, 3, 1).reshape(-1, n_features)
        y = target[:, band].ravel()
        area = np.broadcast_to(AREA[band], target[:, band].shape[1:])
        w = _solve(x, y, np.tile(area.ravel(), len(target)))
        forecast[band] += inputs[:, band].transpose(1, 2, 0) @ w
    return forecast, climatology


def mae(a, b):
    """Mean absolute error, weighted by the area each latitude row covers."""
    return float((np.abs(a - b) * AREA).sum() / (AREA.sum() * a.shape[1]))


def score(actual, forecast, persistence, climatology):
    return {
        "model": round(mae(actual, forecast), 2),
        "persistence": round(mae(actual, persistence), 2),
        "trailing_mean": round(mae(actual, climatology), 2),
    }
