# Next-day cloud on the job's own source, round 2

## Scoring
- Percent lower area-weighted squared error of next-day UTC daily mean cloud than the walk-forward ridge, on 4-day parts of the labelled days, clipped to 0..100 (source: parts_score.py)
- Open-Meteo's forecast made a day earlier scored 172.1 on the same 28 days, 62.3 percent below the ridge. A candidate that reads that forecast is compared with this number and reported as a correction of Open-Meteo, not as an independent forecast (source: nwp_loss.py)

## Test construction
- Open-Meteo days 2026-06-18 to 2026-11-03. Lead-in to 2026-07-17, fit targets to 2026-08-31, select targets 2026-09-01 to 2026-10-06, test targets 2026-10-07 to 2026-11-03. Truth on test days is the job's stored daily means (source: openmeteo_split.py)
- Test labels are sealed, but test_x.npz holds later days' fields, which are labels of earlier days. Candidates read it only through inputs_of, which stops at 18 UTC the day before (source: openmeteo_split.py)
- The test days were seen before this round: ids 14, 15 and 16 of round 1 scored 374.3, 378.6 and 380.5 on them, and the pick of round 1 was accepted on them, not chosen. No candidate here was fitted on them, and the seasons differ between fit and test
- The field download for 2026-06-18 to 2026-09-06 ran outside the ledger, before this file existed (rebuild: fetch_openmeteo.py)

## Designer
- exposes: a network trained on ERA5 1.5 degree means sees Open-Meteo point values with 1.36 times the spread, and its lead over the ridge drops from 48 to 18 percent
- obvious: raising the network's capacity on ERA5, which round 1 showed does not carry over
- hypothesis: an output calibration per latitude fitted on Open-Meteo days closes most of the gap
- hypothesis: rescaling inputs to Open-Meteo's own spread closes more of it than calibrating outputs
- hypothesis: fine-tuning on 45 days overfits a 5 million parameter network and loses to a linear calibration
- hypothesis: a linear correction of Open-Meteo's own forecast beats it by shrinking and smoothing, because squared error rewards a forecast that hedges
- hypothesis: adding the network to that correction adds nothing once Open-Meteo's forecast is an input

## Test conditions
- 28 stored days from 2026-10-07 to 2026-11-03, seven 4-day parts, against the walk-forward ridge (rebuild: parts_score.py)

## Bar
- bar: 17.97 (source: https://github.com/Nas01010101/ascii-earth-forecast/blob/main/forecast/weather.py, date: 2026-11-04, direction: max)
