"""Turns 6-hourly fields into model inputs, for training batches and for one forecast at a time.

Everything is a departure from the training years' day-of-year mean, divided by that variable's scale. The base
forecast of the target variable is that mean on the target day plus the last 30 days' mean departure, and a model
predicts the difference.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import torch

from climatology import doy_of
from common import STEPS_PER_DAY

STEP = 7.5
LATS = np.arange(-90 + STEP / 2, 90, STEP)
TRAILS = ("cloud", "temp")
TRAIL_DAYS = 30


@dataclass(frozen=True)
class Spec:
    variables: tuple = ("cloud", "temp")
    steps: int = 0  # how many 6-hourly steps before the target, per variable
    lags: int = 3  # how many daily means before the target, per variable
    target: str = "cloud"  # the variable whose next daily mean is forecast

    @property
    def dynamic(self) -> int:
        return len(self.variables) * (self.steps + self.lags) + len(TRAILS)

    @property
    def channels(self) -> int:
        return self.dynamic + 7  # land, height, sin lat, cos lat, doy sin, doy cos, target day's mean of the target

    @property
    def days_needed(self) -> int:
        return max(TRAIL_DAYS, self.lags, self.steps // STEPS_PER_DAY + 1)


class Frames:
    """All of one continuous stretch of fields. Day d's batch reads only steps before 00 UTC of day d."""

    def __init__(self, fields: dict, first: date, spec: Spec, stats: dict, device="cpu"):
        days = len(fields["cloud"]) // STEPS_PER_DAY
        self.spec, self.device = spec, device
        # One extra entry, because the day after the last stored one is a valid target.
        self.doy = np.array([doy_of(first + timedelta(days=d)) for d in range(days + 1)])
        step_doy = np.repeat(self.doy[:-1], STEPS_PER_DAY)
        self.anomaly, self.daily, self.sums = {}, {}, {}
        self.target_mean = torch.from_numpy(np.asarray(stats[f"clim_{spec.target}"])).to(device)
        self.target_std = float(stats[f"std_{spec.target}"])
        # Pressure's day-of-year mean is near 1000 hPa, 117 of its scales from zero, so that input is centred.
        # Cloud keeps its uncentred input, because the checkpoints already trained read it that way.
        self.target_centre = 0.0 if spec.target == "cloud" else float(np.asarray(stats[f"clim_{spec.target}"]).mean())
        summed = set(TRAILS) | {spec.target}
        for name in set(spec.variables) | summed:
            scaled = (np.asarray(fields[name]) - np.asarray(stats[f"clim_{name}"])[step_doy]) / float(stats[f"std_{name}"])
            tensor = torch.from_numpy(scaled.astype(np.float32)).to(device)
            self.daily[name] = tensor.reshape(days, STEPS_PER_DAY, 24, 48).mean(1)
            if spec.steps and name in spec.variables:
                self.anomaly[name] = tensor
            if name in summed:
                self.sums[name] = torch.cat([torch.zeros(1, 24, 48, dtype=torch.float64, device=device), self.daily[name].double().cumsum(0)])
        lat = np.radians(LATS)[:, None] * np.ones((24, 48))
        fixed = np.stack([np.asarray(stats["land_sea_mask"]), np.asarray(stats["geopotential_at_surface"]), np.sin(lat), np.cos(lat)])
        self.fixed = torch.from_numpy(fixed.astype(np.float32)).to(device)

    def first_day(self) -> int:
        return self.spec.days_needed

    def batch(self, days: torch.Tensor) -> tuple:
        """(inputs (B, C, 24, 48), base forecast in the target's unit (B, 24, 48)) for these target days."""
        days = days.to(self.device)
        size = len(days)
        parts = []
        for name in self.spec.variables:
            if self.spec.steps:
                back = torch.arange(1, self.spec.steps + 1, device=self.device)
                parts.append(self.anomaly[name][days[:, None] * STEPS_PER_DAY - back])
            if self.spec.lags:
                back = torch.arange(1, self.spec.lags + 1, device=self.device)
                parts.append(self.daily[name][days[:, None] - back])
        trails = {n: ((self.sums[n][days] - self.sums[n][days - TRAIL_DAYS]) / TRAIL_DAYS).float() for n in self.sums}
        parts.append(torch.stack([trails[n] for n in TRAILS], 1))
        target_doy = torch.from_numpy(self.doy[days.cpu().numpy()]).to(self.device)
        mean = self.target_mean[target_doy]
        angle = (2 * np.pi * target_doy / 366).float()
        seasons = torch.stack([torch.sin(angle), torch.cos(angle)], 1)[:, :, None, None].expand(size, 2, 24, 48)
        parts += [self.fixed.expand(size, -1, -1, -1), seasons, (mean[:, None] - self.target_centre) / self.target_std]
        return torch.cat(parts, 1), mean + trails[self.spec.target] * self.target_std

    def truth(self, days: torch.Tensor) -> torch.Tensor:
        """The target variable's daily mean on these days, in its own unit."""
        days = days.to(self.device)
        doy = torch.from_numpy(self.doy[days.cpu().numpy()]).to(self.device)
        return self.daily[self.spec.target][days] * self.target_std + self.target_mean[doy]

    def limited(self, forecast: torch.Tensor) -> torch.Tensor:
        """Cover cannot leave 0..100, and the other targets have no limit the daily job applies."""
        return forecast.clamp(0, 100) if self.spec.target == "cloud" else forecast


def spec_of(saved: dict) -> Spec:
    """The Spec a checkpoint was trained with. Checkpoints older than `target` forecast cloud."""
    held = saved["spec"]
    return Spec(tuple(held["variables"]), held["steps"], held["lags"], held.get("target", "cloud"))


def one_day(fields: dict, when: date, spec: Spec, stats: dict) -> tuple:
    """The inputs of one forecast, built from the view's last days only."""
    keep = spec.days_needed * STEPS_PER_DAY
    # Frames counts days from the cloud field, whatever the target.
    names = set(spec.variables) | set(TRAILS) | {spec.target, "cloud"}
    frames = Frames({n: fields[n][-keep:] for n in names}, when - timedelta(days=spec.days_needed), spec, stats)
    return frames.batch(torch.tensor([spec.days_needed]))
