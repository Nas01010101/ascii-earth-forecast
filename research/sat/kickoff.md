# Satellite position correction on top of SGP4

## Scoring
- Percent less mean 3D position error than raw SGP4 at 15 days, per object, averaged over objects with at least 5 samples, printed for 5 disjoint object groups (source: score.py)

## Test construction
- Element sets with t0 in 2025 from the CC0 Kaggle history. Truth is SGP4 at the epoch of the first element set at or after t0 + 15 days, within 1 day. Fit rows have truth ending before 2024-12-31 (source: split.py)
- Features use element sets at or before t0 and space weather observed the day before t0 (source: build_samples.py)

## Designer
- exposes: SGP4 from a fixed element set drifts along track as drag and manoeuvres go unmodelled, and that drift grows with element age
- obvious: a global linear fit on element age, ndot and bstar, which is the correction already in forecast/orbits.py
- hypothesis: an object's own recent hindcast error at 1, 3, 7 and 15 days carries most of the correctable signal, so a simple per-object extrapolation already beats the global linear fit
- hypothesis: a deep network gains over gradient boosting only by combining that history with orbit regime and space weather, and a deep network that does not beat boosting on the holdout is not adopted
- hypothesis: measuring along-track error as an arc instead of a flat component removes a false radial error at large drift and lowers the miss for any model

## Test conditions
- t0 in 2025 with horizons 1, 3, 7, 15 and 30 days, all catalogue objects with public element sets, primary horizon 15 days (rebuild: build_samples.py)
- Raw SGP4 on objects 27845, 20774 and 23975, June 2017 to June 2020, 15 and 30 days, compared with the published raw errors (rebuild: positive_control.py)

## Bar
- bar: 29.3 (source: https://doi.org/10.26077/aspb-yk57, date: 2026-11-04, direction: max)
