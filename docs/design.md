# Design

Decisions a reader cannot get from the code, and what each one gives up.

| Decision | Why | Cost |
|---|---|---|
| Open-Meteo forecast API with `past_days` | One endpoint for history, today and tomorrow; no key | Past hours are model output, not observations |
| 7.5 degree grid | 1,152 cells fit the free limits (600 calls a minute, 5,000 an hour, 10,000 a day) | No regional detail; the planet draws edges inside each cell with noise |
| Ridge per latitude band, minimising `sum a_i (delta - x'w)^2 + lambda \|w\|^2` with `a_i = cos(lat)`, `lambda = 1` | Closed form on numpy, weights can be printed | May lose to persistence on a given week; the page says so when it does |
| Anomalies from a trailing mean | Removes each cell's geographic mean, so bands share one model shape | A mean of up to 60 days lags the season; the first 17 days are not scored |
| Scores re-run on each update | Needs only the stored days and cannot leak later data | Scores the method, not the exact map shown the day before; once the 60-day window is full the oldest day differs |
| Stored days never rewritten | A day used to score another keeps its value | Later revisions by Open-Meteo are ignored |
| CelesTrak `GROUP=active` CSV | Current elements with no account | No history before the first run, so the satellite loss starts when a snapshot is a day old; a 7-day score needs 8 snapshots |
| Network weights trained offline and committed | The daily job needs only numpy, on a CPU, in minutes | No learning from new data until someone retrains; the published loss shows any decay |
| Network trained on the public catalogue, run on the active list | The catalogue gives 9 years of history; the active list has none | CelesTrak's active list leaves out debris, so the published loss may differ from the research numbers |
| Past forecasts kept in `forecasts.npz`, first entry of a day wins | A published forecast can be scored later without predicting again | Days before the archive began are replays, marked `issued: false`; the file grows by one day of grids per run |
| Output on a one-commit `data` branch | Repository stays small; the site reads raw files with no server | No version history of past runs, apart from `forecasts.npz` |
| One failure per source | A CelesTrak outage does not stop the weather | The page can show sections of different ages, and says so |

## Assumptions not yet checked against a live run

- GitHub runner IP ranges are not rate limited by Open-Meteo or CelesTrak, which count requests per address.
- Calling the workflow enable endpoint resets the 60-day inactivity timer. The push to `data` may also count; neither is documented.
