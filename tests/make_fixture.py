"""Writes a synthetic data directory for browser tests of the site. Its scores mean nothing; never publish it."""
import sys, gzip, json
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
import numpy as np
from forecast import fetch, run
root = Path(sys.argv[1]); today = datetime.now(timezone.utc).date()
lat, lon = np.meshgrid(fetch.LATS, fetch.LONS, indexing='ij')
def hourly(past):
    start = np.datetime64(today - timedelta(days=past), 'h')
    hours = start + np.arange((past + 2) * 24)
    t = np.arange(hours.size)[:, None, None]; rng = np.random.default_rng(past)
    wave = np.sin(np.radians(lon * 3 + lat * 2) + t / 40)
    cloud = np.clip(50 + 40 * wave + rng.normal(0, 8, (hours.size, *lat.shape)), 0, 100).astype(np.float32)
    temp = (28 * np.cos(np.radians(lat)) - 8 + 4 * wave + rng.normal(0, 1, cloud.shape)).astype(np.float32)
    precip = np.maximum(0, wave - .4).astype(np.float32) * 2
    return hours, dict(cloud=cloud, temp=temp, precip=precip)
fetch.weather = hourly
text = open('tests/active-sample.csv').read()
fetch.elements = lambda: text
for d in range(3, 0, -1):
    (root / 'elements').mkdir(exist_ok=True)
    (root / 'elements' / f'{(today - timedelta(days=d)).isoformat()}.csv.gz').write_bytes(gzip.compress(text.encode()))
print(run.run(root, today, {'weather': run.update_weather, 'satellites': run.update_satellites}))
print(sorted(p.name for p in root.iterdir()), (root / 'planet.bin').stat().st_size, (root / 'forecast.json').stat().st_size)
print(json.dumps(json.loads((root / 'forecast.json').read_text())['weather']['week']))
