# ascii-earth-forecast

Daily weather and satellite position forecasts, scored against the past week. The output drives the clouds, rain and satellites on the ASCII planet at [nas01010101.github.io](https://nas01010101.github.io) and its [flow page](https://nas01010101.github.io/flow.html).

![ASCII planet with satellites](docs/planet.png)

## Data flow

```
weather                                satellites

Open-Meteo forecast API                CelesTrak active elements
1,152 cells on a 7.5 degree grid       one CSV snapshot a day
  |                                      |
  v                                      v
validate                               validate
ranges, shape, under 1% missing        header, row count, drops rows
                                       SGP4 cannot read
  |                                      |
  v                                      v
preprocess                             preprocess
7 fields: UTC daily means and          each object's element sets, oldest
6-hourly steps                         first, and 45 inputs from them
  |                                      |
  v                                      v
predict                                predict
cloud: U-Net reading 30 days, frozen   SGP4 from elements 1, 3 and 7 days
temperature: ridge regression, refit   old, plus a residual network: 6
daily; cloud too if days are missing   blocks, 256 wide, mean of 3, frozen
  |                                      |
  v                                      v
actual                                 actual
the stored daily means, once the day   SGP4 from the first newer set past
has finished                           the horizon, at its epoch
  |                                      |
  v                                      v
score                                  score
kept forecasts, last 28 days, area-    median distance in km, against
weighted squared error, against        SGP4 alone
repeating the day before
```

## Scoring

- **Weather.** Each run keeps that day's forecast of cloud cover and 2 m temperature in `forecasts.npz`, and `weather.losses` scores the kept forecasts of the last 28 finished days against what happened and against repeating the day before. The loss is the area-weighted squared error. A row has `issued: true` when the forecast was kept on the day it was made and `issued: false` when it was replayed from stored days because no forecast was kept then, and `net: true` when the network made its cloud map rather than the ridge. A replay uses only earlier days, and a kept forecast is never rewritten.
- **Satellites.** Each day, `satellites.losses` scores the forecasts that started 1, 3 and 7 days earlier: the median distance in km of SGP4 alone and of SGP4 with the network's correction. A forecast reads only the element sets held when it started. A row has `issued: true` when it was scored on its own day and `issued: false` when it was replayed from stored snapshots after a missed run.

"Actual" is a public reference with its own error, which these scores do not include. For weather it is Open-Meteo's model output for past hours, not station readings. For satellites it is SGP4 run from the newest published elements, not a tracking measurement.

## Weather network

`forecast/cloudnet.py` runs a U-Net with numpy from `forecast/cloud_weights.npz`: 5,114,817 weights, stored as 16-bit floats. It reads 7 fields over the past 30 days (cloud, temperature, rain, pressure, wind east and north, water vapour) and predicts the day's mean cloud cover. Training and evaluation code is in `research/weather`, on ERA5 from WeatherBench 2: trained on 1979 to 2018, checked on 2019 and 2020, tested on a sealed 2021 and 2022.

| Test | Cloud loss | Below the ridge |
|---|---|---|
| Sealed ERA5 holdout, 2021 and 2022 | | 48.48% (standard error 0.38) |
| 28 stored Open-Meteo days, 2026-10-07 to 2026-11-03, network | 374.3 | 18.0% |
| Same days, ridge | 456.3 | |
| Same days, repeating the day before | 647.0 | |
| Same days, Open-Meteo's own forecast made a day earlier | 172.1 | |

The Open-Meteo forecast is a reference and not a fair bar: it is scored against output of the same system. The gap between 48% and 18% is not fully explained. The stored days are point values at 7.5 degree cells and the network trained on ERA5's 1.5 degree area means, and the Open-Meteo fields have a wider spread than ERA5's (cloud standard deviation 1.36 times ERA5's), with no sign of a unit error. `weather.losses` is the test of the kept forecasts.

## Satellite network

`forecast/drift.py` runs the network with numpy from `forecast/drift_weights.npz`. Training and evaluation code is in `research/sat`, on public element sets from 2017 to 2025.

| Test, with weights trained through 2024 on the 2025 catalogue | 15-day improvement over SGP4, mean of objects' medians | 95% interval |
|---|---|---|
| Sealed holdout, every published set | 30.21% | 24.85 to 34.46 |
| Same year, sets as a once-a-day job holds them | 28.8% | 24.25 to 32.39 |

On the three objects of [Liu et al. 2021](https://doi.org/10.26077/aspb-yk57), a network from the same recipe trained on 2017 and 2018 improved the 15-day miss by 21.04% (13.23 to 27.54) from 2019 to mid 2020. The mean of that paper's 15-day table is 29.3%, and it describes no separate test period. The catalogue numbers are a different test and do not compare with it.

`research/sat/replay_check.py` runs the daily job's scorer on 47 weekly days of 2025, on an eighth of the catalogue, with the weights trained through 2024:

| Days ahead | Days the network beat SGP4 | Median miss, SGP4 | Median miss, network |
|---|---|---|---|
| 1 | 32 of 47 | 0.77 km | 0.70 km |
| 3 | 45 of 47 | 2.95 km | 2.33 km |
| 7 | 46 of 47 | 11.34 km | 6.76 km |

Truth is SGP4 from a later published element set, not tracking data. Manoeuvres are not filtered out. The weights in this repository were refit on every labelled year through 2025, so they have no held-out test of their own; `satellites.losses` is theirs.

## Output

A GitHub Actions job runs once a day and force-pushes the `data` branch, which holds a single commit:

| File | Contents |
|---|---|
| `forecast.json` | per-source status, today's ASCII map, its forecast at three cells and the model that made it (`weather.today`), the last finished day's map (`weather.last_scored`), the loss of past forecasts (`weather.losses`), and the satellite loss (`satellites.losses`) |
| `planet.bin` | cloud and rain grids every 3 hours for today and tomorrow, read by the planet |
| `sats.bin` | satellite elements for the browser |
| `history.npz` | up to 60 days of daily means |
| `steps.npz` | 38 days of the 7 fields, every 6 hours, read by the network |
| `forecasts.npz` | every forecast kept so far, by day, with its `issued` and `net` flags |
| `elements/` | up to 38 daily element snapshots |

If one source fails, its last good section stays in `forecast.json` and `status` records the failure and the date of that section. This covers any exception in a source. The job exits non-zero only when both sources fail. Stored days are never rewritten; a day left out for missing data is requested again on the next run. Design decisions: [docs/design.md](docs/design.md).

## Run

```sh
pip install ".[test]"
python -m pytest
python -m forecast data
```

`python -m forecast data` writes to `./data`. The first run downloads 45 days of weather, about 3,900 billed Open-Meteo calls at 500 a minute.

## Sources

- Weather data by [Open-Meteo](https://open-meteo.com/), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The free API is for non-commercial use.
- Orbital elements by [CelesTrak](https://celestrak.org/NORAD/elements/).

Code: MIT.
