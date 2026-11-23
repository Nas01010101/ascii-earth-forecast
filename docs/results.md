# Results

"Actual" is a public reference with its own error, which these scores do not include. For weather it is Open-Meteo's model output for past hours, not station readings. For satellites it is SGP4 run from the newest published elements, not a tracking measurement.

## Weather networks

`forecast/cloudnet.py` runs a U-Net with numpy from `forecast/cloud_weights.npz`: 5,114,817 weights, stored as 16-bit floats. It reads 7 fields over the past 30 days (cloud, temperature, rain, pressure, wind east and north, water vapour) and predicts the day's mean cloud cover. Training and evaluation code is in `research/weather`, on ERA5 from WeatherBench 2: trained on 1979 to 2018, checked on 2019 and 2020, tested on a sealed 2021 and 2022.

| Test | Cloud loss | Below the ridge |
|---|---|---|
| Sealed ERA5 holdout, 2021 and 2022 | | 48.48% (standard error 0.38) |
| 28 stored Open-Meteo days, 2026-10-07 to 2026-11-03, network | 374.3 | 18.0% |
| Same days, ridge | 456.3 | |
| Same days, repeating the day before | 647.0 | |
| Same days, Open-Meteo's own forecast made a day earlier | 172.1 | |

The Open-Meteo forecast is a reference and not a fair bar: it is scored against output of the same system. The gap between 48% and 18% is not fully explained. The stored days are point values at 7.5 degree cells and the network trained on ERA5's 1.5 degree area means, and the Open-Meteo fields have a wider spread than ERA5's (cloud standard deviation 1.36 times ERA5's), with no sign of a unit error. `weather.losses` is the test of the kept forecasts.

`forecast/temp_weights.npz` and `forecast/pressure_weights.npz` hold three U-Nets each, 1,294,049 weights per network, reading the same 7 fields over the same 30 days. Each was trained on ERA5, then fine-tuned on 30 Open-Meteo days from 2026-07-18, stopping on the next 15. A forecast is the mean of the three. The weights are the ones scored below, not refit on the test days.

Each run also fetches ECMWF's [AIFS Single](https://huggingface.co/ecmwf/aifs-single-1.0), a machine-learned global forecast, for today's cloud cover, temperature and pressure through Open-Meteo, and stores its daily means in `aifs.npz` with the first value of a day kept. It is published next to the networks' forecast and scored on the same days. AIFS steps every 6 hours; Open-Meteo interpolates the hours between.

| Test against the ridge fed 24-hour means | Temperature | Pressure |
|---|---|---|
| Sealed Open-Meteo days, 2026-10-07 to 2026-11-03 | 61.42% below | 62.99% below |
| Selection days, 2026-09-01 to 2026-10-06 | 64.36% below | 63.60% below |
| Selection days, Open-Meteo's own forecast made a day earlier | 89.72% below | not fetched |

The mean of both fields on the sealed days is 62.20% below the ridge, standard error 0.75.

## Satellite network

`forecast/drift.py` runs the network with numpy from `forecast/drift_weights.npz`. Training and evaluation code is in `research/sat`, on public element sets from 2017 to 2025. The shipped weights were fit on 2020 to 2025, on the objects whose catalogue number modulo 64 falls outside 16 to 31, three quarters of the catalogue (`research/sat/final_fit.py`, `train_net.py`).

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

Truth is SGP4 from a later published element set, not tracking data. Manoeuvres are not filtered out. The weights in this repository were refit on 2020 to 2025, so no year is held out for them; `satellites.losses` is their test.
