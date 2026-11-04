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
UTC daily means, departures            each object's element sets, oldest
from each cell's trailing mean         first, and 45 inputs from them
  |                                      |
  v                                      v
train                                  predict
ridge regression per latitude band     SGP4 from elements 1, 3 and 7 days
                                       old, plus a residual network: 6
                                       blocks, 256 wide, mean of 3, frozen
  |                                      |
  v                                      v
predict                                actual
today, from the days through           SGP4 from the first newer set past
yesterday                              the horizon, at its epoch
  |                                      |
  v                                      v
score                                  score
last 7 days, area-weighted MAE,        median distance in km, against
against persistence and trailing mean  SGP4 alone
```

## Scoring

- **Weather.** Cloud cover and 2 m temperature. Each scored day is predicted by a model trained only on earlier days, and is predicted again on each run from the stored days, so the table is a backtest of the method and not a record of maps published earlier. Baselines: yesterday's value (persistence) and the mean of the earlier days.
- **Past forecasts.** Each run keeps that day's forecast in `forecasts.npz`, and `weather.losses` scores the kept forecasts of the last 28 finished days against what happened. The loss is the area-weighted squared error, the data term the ridge minimises. A row has `issued: true` when the forecast was kept on the day it was made and `issued: false` when it was replayed from stored days because no forecast was kept then. A replay uses only earlier days, and a kept forecast is never rewritten.
- **Satellites.** Each day, `satellites.losses` scores the forecasts that started 1, 3 and 7 days earlier: the median distance in km of SGP4 alone and of SGP4 with the network's correction. A forecast reads only the element sets held when it started. A row has `issued: true` when it was scored on its own day and `issued: false` when it was replayed from stored snapshots after a missed run.

"Actual" is a public reference with its own error, which these scores do not include. For weather it is Open-Meteo's model output for past hours, not station readings. For satellites it is SGP4 run from the newest published elements, not a tracking measurement.

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
| `forecast.json` | scores, ASCII maps, per-source status, the weights and loss of each band (`weather.model`), today's forecast globally and at three cells (`weather.today`), the loss of past forecasts (`weather.losses`), and the satellite loss (`satellites.losses`) |
| `planet.bin` | cloud and rain grids every 3 hours for today and tomorrow, read by the planet |
| `sats.bin` | satellite elements for the browser |
| `history.npz` | up to 60 days of daily means |
| `forecasts.npz` | every forecast kept so far, by day, with its `issued` flag |
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
