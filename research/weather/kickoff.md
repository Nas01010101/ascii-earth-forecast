# Next-day global cloud forecast on the daily job's grid

## Scoring
- Percent lower area-weighted squared error of next-day UTC daily mean cloud cover than the production ridge in forecast/weather.py, walk-forward on its 60 trailing days, one number per calendar quarter (source: score.py)
- A candidate reads 6-hourly fields ending at 18 UTC the day before its target, and its output is clipped to 0..100 as production does (source: score.py)
- A network is adopted only at 2 percent or more below the ridge on every seed, on the ERA5 holdout and on the daily job's 28 stored days

## Test construction
- ERA5 from WeatherBench 2, bilinear from 1.5 degrees to the 48x24 points. Training 1979-2018, validation targets 2019-2020, holdout targets 2021-2022. The holdout seals the fields, because inputs of later days are labels of earlier days (source: split.py)

## Designer
- exposes: next-day cloud holds advection and synoptic signal that a per-cell model can only partly reach
- obvious: the production ridge, one set of weights per latitude band over a cell and its 4 neighbours, reaches one 7.5 degree cell, about 800 km at the equator, while weather systems move about 1000 km a day
- hypothesis: a wide receptive field with pressure and wind learns advection and beats a wider linear model
- hypothesis: 6-hourly inputs give most of the gain, because 18 UTC is closer to the target than yesterday's daily mean
- hypothesis: 40 years of training gives a linear model much of the gain with no network at all
- hypothesis: the gain on ERA5 does not carry to Open-Meteo, whose fields come from different models

## Test conditions
- ERA5 2021-2022, 8 calendar quarters, candidate against walk-forward ridge (rebuild: split.py)
- The daily job's last 28 stored Open-Meteo days, ridge 456.3 and repeat-yesterday 647.0, used to accept or refuse a pick and never to choose one (rebuild: replay_days.py)

## Bar
- bar: 2 (source: https://github.com/Nas01010101/ascii-earth-forecast/blob/main/forecast/weather.py, date: 2026-11-04, direction: max)
