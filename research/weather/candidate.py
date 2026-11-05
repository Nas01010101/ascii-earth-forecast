"""Candidates the judge can run. Each gets only the fields before the target day, 6-hourly, oldest first."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]

from forecast import cloudnet, weather  # noqa: E402

from common import STEPS_PER_DAY, daily  # noqa: E402
from features import Spec, one_day, spec_of  # noqa: E402
from fit_linear import apply  # noqa: E402
from nets import UNet  # noqa: E402

RIDGE_DAYS = 60  # run.HISTORY_DAYS: the production ridge never sees more than this


class Repeat:
    def cloud(self, fields: dict, when) -> np.ndarray:
        return daily(fields["cloud"][-STEPS_PER_DAY:])[0]


class Ridge:
    def cloud(self, fields: dict, when) -> np.ndarray:
        series = daily(fields["cloud"][-RIDGE_DAYS * STEPS_PER_DAY:])
        return weather.predict(series, len(series))[0]


class Climatology:
    """The base forecast alone: the day-of-year mean plus the last 30 days' mean departure."""

    def __init__(self, stats: dict):
        self.stats = stats

    def cloud(self, fields: dict, when) -> np.ndarray:
        return one_day(fields, when, Spec(), self.stats)[1][0].numpy()


class Net:
    def __init__(self, saved: dict):
        self.spec = spec_of(saved)
        self.stats = saved["stats"]
        self.net = UNet(self.spec.channels, saved["width"], saved["place"])
        self.net.load_state_dict(saved["state"])
        self.net.eval()

    @torch.no_grad()
    def forecast(self, fields: dict, when) -> np.ndarray:
        x, base = one_day(fields, when, self.spec, self.stats)
        return (base + self.net(x) * float(self.stats[f"std_{self.spec.target}"]))[0].numpy()

    cloud = forecast


class Linear:
    def __init__(self, saved: dict):
        self.spec = Spec(tuple(str(v) for v in saved["variables"]), int(saved["steps"]), int(saved["lags"]))
        self.stats, self.radius = saved, int(saved["radius"])
        self.weights, self.group = torch.from_numpy(saved["weights"]), torch.from_numpy(saved["group"])

    @torch.no_grad()
    def cloud(self, fields: dict, when) -> np.ndarray:
        x, base = one_day(fields, when, self.spec, self.stats)
        return apply(x, base, self.weights, self.group, self.spec, self.radius, float(self.stats["std_cloud"]))[0].numpy()


class Frozen:
    """The file the daily job ships, run by the daily job's own numpy code."""

    def __init__(self, path: str):
        self.saved = cloudnet.load(path)

    def cloud(self, fields: dict, when) -> np.ndarray:
        keep = cloudnet.DAYS_NEEDED * STEPS_PER_DAY
        return cloudnet.predict(np.stack([fields[n][-keep:] for n in cloudnet.VARIABLES]), when, self.saved)


KINDS = {"repeat": Repeat, "ridge": Ridge}


def load(path: str):
    """A candidate file names its kind: .json for the baselines, .pt for a network, .npz for a linear model."""
    if path.endswith(".pt"):
        return Net(torch.load(path, map_location="cpu", weights_only=False))
    if path.endswith(".npz"):
        saved = dict(np.load(path))
        if "place" in saved:
            return Frozen(path)
        return Linear(saved) if str(saved.get("kind")) == "linear" else Climatology(saved)
    return KINDS[json.loads(Path(path).read_text())["kind"]]()
