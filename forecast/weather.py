"""Next-day forecast of daily mean cloud cover and temperature: one ridge regression per latitude band.

Each cell is predicted from its own last three days and its four neighbours yesterday,
all as departures from that cell's mean over the earlier days. Nothing from the target
day or later enters the training set, the cell means or the inputs.
"""
import numpy as np

from .fetch import LATS, LONS

MIN_TRAIN_DAYS = 14
RIDGE = 1.0
FEATURES = ("yesterday", "2 days ago", "3 days ago", "north", "south", "east", "west")
# Tropics, mid-latitudes and poles have different prevailing winds, so each gets its own weights.
BANDS = (np.abs(LATS) < 30, (np.abs(LATS) >= 30) & (np.abs(LATS) < 60), np.abs(LATS) >= 60)
BAND_NAMES = ("tropics", "mid-latitudes", "poles")
# One fixed cell per band, so the page can walk a real number through the weights.
CELLS = (("near Singapore", 3.75, 101.25), ("near Montreal", 48.75, -71.25), ("near Tromso", 71.25, 18.75))
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


def training_rows(anomaly, band):
    """Flat (inputs, target, area weight) rows of one band, over every day that has three days before it."""
    train = np.stack([_inputs(anomaly, d) for d in range(3, len(anomaly))])
    target = anomaly[3:]
    area = np.broadcast_to(AREA[band], target[:, band].shape[1:])
    x = train[:, :, band].transpose(0, 2, 3, 1).reshape(-1, len(FEATURES))
    return x, target[:, band].ravel(), np.tile(area.ravel(), len(target))


def fit(series, day):
    """Fits series[:day]. Returns (cell means, departures, per band {weights, loss, rmse}).

    Each band minimises sum(area * (y - x @ w)^2) + RIDGE * w @ w, which _solve does in one step.
    """
    if day - 3 < MIN_TRAIN_DAYS:
        raise ValueError(f"needs {MIN_TRAIN_DAYS + 3} days before the target, got {day}")
    climatology = series[:day].mean(axis=0)
    anomaly = series[:day] - climatology
    fits = []
    for band in BANDS:
        x, y, area = training_rows(anomaly, band)
        w = _solve(x, y, area)
        squared = float((area * (y - x @ w) ** 2).sum())
        fits.append({"weights": w, "loss": squared + RIDGE * float(w @ w), "rmse": (squared / float(area.sum())) ** 0.5})
    return climatology, anomaly, fits


def predict(series, day):
    """Forecasts series[day] from series[:day]. Returns (forecast, trailing mean)."""
    climatology, anomaly, fits = fit(series, day)
    forecast = climatology.copy()
    inputs = _inputs(anomaly, day)
    for band, fitted in zip(BANDS, fits):
        forecast[band] += inputs[:, band].transpose(1, 2, 0) @ fitted["weights"]
    return forecast, climatology


def _cell(lat, lon):
    return int(np.abs(LATS - lat).argmin()), int(np.abs(LONS - lon).argmin())


def at_places(cloud, temp):
    """A forecast's cloud and temperature at each named cell."""
    return [{"place": place, "cloud": round(float(cloud[_cell(lat, lon)]), 1), "temp": round(float(temp[_cell(lat, lon)]), 1)}
            for place, lat, lon in CELLS]


def explain(series, day, limits):
    """The fit behind one forecast, per band: weights, training loss and one worked cell."""
    climatology, anomaly, fits = fit(series, day)
    inputs = _inputs(anomaly, day)
    bands = []
    for name, fitted, (place, lat, lon) in zip(BAND_NAMES, fits, CELLS):
        i, j = _cell(lat, lon)
        x, w = inputs[:, i, j], fitted["weights"]
        raw = float(climatology[i, j] + x @ w)
        predicted = float(np.clip(raw, *limits))
        actual = float(series[day, i, j])
        bands.append({
            "name": name,
            "weights": [round(float(v), 3) for v in w],
            "loss": round(fitted["loss"]),
            "rmse": round(fitted["rmse"], 2),
            "cell": {
                "place": place, "lat": lat, "lon": lon,
                "inputs": [round(float(v), 1) for v in x],
                "mean": round(float(climatology[i, j]), 1),
                "change": round(float(x @ w), 1),
                "predicted": round(predicted, 1),
                "actual": round(actual, 1),
                "error": round(abs(predicted - actual), 1),
            },
        })
    return bands


def mae(a, b):
    """Mean absolute error, weighted by the area each latitude row covers."""
    return float((np.abs(a - b) * AREA).sum() / (AREA.sum() * a.shape[1]))


def mse(a, b):
    """Mean squared error with the same area weights, which is the data term the ridge minimises."""
    return float(((a - b) ** 2 * AREA).sum() / (AREA.sum() * a.shape[1]))


def score(actual, forecast, persistence, climatology):
    return {
        "model": round(mae(actual, forecast), 2),
        "persistence": round(mae(actual, persistence), 2),
        "trailing_mean": round(mae(actual, climatology), 2),
    }
