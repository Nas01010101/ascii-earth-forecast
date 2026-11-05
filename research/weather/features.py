"""Turns 6-hourly fields into model inputs, for training batches and for one forecast at a time.

Everything is a departure from the training years' day-of-year mean, divided by that variable's scale. The base
forecast is that mean on the target day plus the last 30 days' mean departure, and a model predicts the difference.
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

    @property
    def dynamic(self) -> int:
        return len(self.variables) * (self.steps + self.lags) + len(TRAILS)

    @property
    def channels(self) -> int:
        return self.dynamic + 7  # land, height, sin lat, cos lat, doy sin, doy cos, target day's mean cloud

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
        self.cloud_mean = torch.from_numpy(np.asarray(stats["clim_cloud"])).to(device)
        self.cloud_std = float(stats["std_cloud"])
        for name in set(spec.variables) | set(TRAILS):
            scaled = (np.asarray(fields[name]) - np.asarray(stats[f"clim_{name}"])[step_doy]) / float(stats[f"std_{name}"])
            tensor = torch.from_numpy(scaled.astype(np.float32)).to(device)
            self.daily[name] = tensor.reshape(days, STEPS_PER_DAY, 24, 48).mean(1)
            if spec.steps and name in spec.variables:
                self.anomaly[name] = tensor
            if name in TRAILS:
                self.sums[name] = torch.cat([torch.zeros(1, 24, 48, dtype=torch.float64, device=device), self.daily[name].double().cumsum(0)])
        lat = np.radians(LATS)[:, None] * np.ones((24, 48))
        fixed = np.stack([np.asarray(stats["land_sea_mask"]), np.asarray(stats["geopotential_at_surface"]), np.sin(lat), np.cos(lat)])
        self.fixed = torch.from_numpy(fixed.astype(np.float32)).to(device)

    def first_day(self) -> int:
        return self.spec.days_needed

    def batch(self, days: torch.Tensor) -> tuple:
        """(inputs (B, C, 24, 48), base forecast in percent (B, 24, 48)) for these target days."""
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
        trails = {n: ((self.sums[n][days] - self.sums[n][days - TRAIL_DAYS]) / TRAIL_DAYS).float() for n in TRAILS}
        parts.append(torch.stack([trails[n] for n in TRAILS], 1))
        target_doy = torch.from_numpy(self.doy[days.cpu().numpy()]).to(self.device)
        mean = self.cloud_mean[target_doy]
        angle = (2 * np.pi * target_doy / 366).float()
        seasons = torch.stack([torch.sin(angle), torch.cos(angle)], 1)[:, :, None, None].expand(size, 2, 24, 48)
        parts += [self.fixed.expand(size, -1, -1, -1), seasons, mean[:, None] / self.cloud_std]
        return torch.cat(parts, 1), mean + trails["cloud"] * self.cloud_std

    def truth(self, days: torch.Tensor) -> torch.Tensor:
        """The target day's mean cloud in percent."""
        days = days.to(self.device)
        return self.daily["cloud"][days] * self.cloud_std + self.cloud_mean[torch.from_numpy(self.doy[days.cpu().numpy()]).to(self.device)]


def one_day(fields: dict, when: date, spec: Spec, stats: dict) -> tuple:
    """The inputs of one forecast, built from the view's last days only."""
    keep = spec.days_needed * STEPS_PER_DAY
    names = set(spec.variables) | set(TRAILS)
    frames = Frames({n: fields[n][-keep:] for n in names}, when - timedelta(days=spec.days_needed), spec, stats)
    return frames.batch(torch.tensor([spec.days_needed]))
