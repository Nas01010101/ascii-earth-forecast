# Next-day temperature and pressure on the job's own source, round 3

## Scoring
- Mean over temperature and pressure of the percent lower area-weighted squared error of the next UTC day's mean than the walk-forward ridge, on 4-day parts of the labelled days (source: fields_score.py)
- Open-Meteo's temperature forecast made a day earlier is reported beside it on the same days, and no candidate reads it (source: nwp_loss.py)

## Test construction
- Open-Meteo days 2026-06-18 to 2026-11-03. Lead-in to 2026-07-17, fit targets to 2026-08-31, select targets 2026-09-01 to 2026-10-06, test targets 2026-10-07 to 2026-11-03, the dates of the cloud round (source: field_labels.py)
- Temperature truth is the 24-hour mean. Pressure truth is the mean of the 4 samples at 00, 06, 12 and 18 UTC, because the source gives no 24-hour mean (source: field_labels.py)
- Test labels are sealed, but test_x.npz holds later days' fields, which are labels of earlier days. Candidates read it only through inputs_of, which stops at 18 UTC the day before (source: openmeteo_split.py)
- Cloud on these test days was seen in rounds 1 and 2, and ERA5 2021-2022 cloud in round 1. Temperature and pressure scores on them have not been computed

## Designer
- exposes: a network trained on ERA5 1.5 degree means reads Open-Meteo point values, which cost the cloud network most of its lead over the ridge
- obvious: the ridge fitted to the last 60 days, which has no knowledge of how weather systems move
- hypothesis: temperature and pressure carry over from ERA5 better than cloud, because they vary less at small scale
- hypothesis: pressure gains more than temperature, because a one-day forecast of a moving system is what the network can learn and the ridge cannot
- hypothesis: fine-tuning on the 45 fit days closes the shift for these fields as it did for cloud

## Test conditions
- 28 days from 2026-10-07 to 2026-11-03, seven 4-day parts, against the walk-forward ridge (rebuild: field_labels.py)

## Bar
- bar: 0.0 (source: https://github.com/Nas01010101/ascii-earth-forecast/blob/main/forecast/weather.py, date: 2026-11-05, direction: max)
