# Design

Decisions a reader cannot get from the code, and what each one gives up.

| Decision | Why | Cost |
|---|---|---|
| Open-Meteo forecast API with `past_days` | One endpoint for history, today and tomorrow; no key | Past hours are model output, not observations |
| 7.5 degree grid | 1,152 cells fit the free limits (600 calls a minute, 5,000 an hour, 10,000 a day) | No regional detail; the planet draws edges inside each cell with noise |
| U-Net for cloud cover, trained offline on ERA5 and committed | ERA5 gives 40 years to train on, and the network loses less than the ridge on every test so far | 18% below the ridge on Open-Meteo days against 48% on ERA5, whose fields differ from the job's; Open-Meteo's own forecast scores lower than the network on its own output |
| Network reads 6-hourly fields kept in `steps.npz` | Sub-daily steps took the validation score from 24% to 37% below the ridge, and all 7 fields took it to 46%; width 64 added 2 more | 2.6 MB more on the `data` branch, and the first network run must fetch 39 past days, about 450 seconds |
| Ridge kept for temperature and as the cloud fallback | Closed form on numpy, needs only the stored daily means; a forecast is still kept when a 6-hourly day is missing, with `net: false` | Temperature has no network; a chart can mix two models, which it marks |
| Past forecasts scored instead of re-run | A score then reads what was published, not what the method would say today | The first network rows appear a day after the first network forecast |
| Stored days never rewritten | A day used to score another keeps its value | Later revisions by Open-Meteo are ignored |
| CelesTrak `GROUP=active` CSV | Current elements with no account | No history before the first run, so the satellite loss starts when a snapshot is a day old; a 7-day score needs 8 snapshots |
| Network weights trained offline and committed | The daily job needs only numpy, on a CPU, in minutes | No learning from new data until someone retrains; the published loss shows any decay |
| Network trained on the public catalogue, run on the active list | The catalogue gives 9 years of history; the active list has none | CelesTrak's active list leaves out debris, so the published loss may differ from the research numbers |
| Past forecasts kept in `forecasts.npz`, first entry of a day wins | A published forecast can be scored later without predicting again; a ridge forecast stays a ridge forecast after the network arrives | Days before the archive began are replays, marked `issued: false`; the file grows by one day of grids per run |
| Output on a one-commit `data` branch | Repository stays small; the site reads raw files with no server | No version history of past runs, apart from `forecasts.npz` |
| Network corrections only with two days of held sets and a lookback | `research/sat/history_check.py` on 2025: with no lookback the median miss at 3 days is 4.97 km against SGP4's 3.49; with two days held it is 3.33 | The published score equals SGP4 for the first two days of a new data branch |
| `corrections.bin` holds 1, 3, 7 and 15 day corrections per object | The viewer shows a network position for any later time without running the network in the browser | About 32 bytes per object, 16 s a day on two threads |
| Two scheduled runs a day, each skips what is done | GitHub can drop a scheduled run; a failed source is retried without fetching the other again | A code change made after the day's run first shows in the next day's run |
| `forecast.check` before publishing | A file the site cannot read never replaces the last good one | A failed check leaves the data branch a day old |
| One failure per source | A CelesTrak outage does not stop the weather | The page can show sections of different ages, and says so |

## Assumptions not yet checked against a live run

- GitHub runner IP ranges are not rate limited by Open-Meteo or CelesTrak, which count requests per address.
- Calling the workflow enable endpoint resets the 60-day inactivity timer. The push to `data` may also count; neither is documented.
