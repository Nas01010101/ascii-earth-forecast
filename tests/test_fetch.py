import json

import numpy as np
import pytest

from forecast import fetch

SHAPE = (2, 3)


def payloads(cloud=50, hours=("2026-11-02T00:00", "2026-11-02T01:00")):
    hourly = {"time": list(hours), "cloud_cover": [cloud] * len(hours),
              "temperature_2m": [10.0] * len(hours), "precipitation": [0.0] * len(hours)}
    return [{"hourly": hourly} for _ in range(SHAPE[0] * SHAPE[1])]


def test_parses_into_time_lat_lon():
    hours, fields = fetch.parse_weather(payloads(), SHAPE)
    assert hours.size == 2
    assert fields["cloud"].shape == (2, *SHAPE)


def test_null_becomes_nan():
    _, fields = fetch.parse_weather(payloads(cloud=None), SHAPE)
    assert np.isnan(fields["cloud"]).all()


@pytest.mark.parametrize("bad", [
    payloads(cloud=140),
    payloads()[:-1],
    [{"hourly": {}}] * 6,
    [{"error": True, "reason": "limit"}] * 6,
])
def test_rejects_bad_payloads(bad):
    with pytest.raises(fetch.SourceError):
        fetch.parse_weather(bad, SHAPE)


def test_ragged_rows_are_rejected():
    bad = payloads()
    bad[3] = {"hourly": {**bad[3]["hourly"], "cloud_cover": [1]}}
    with pytest.raises(fetch.SourceError):
        fetch.parse_weather(bad, SHAPE)


def test_grid_is_1152_cells_in_chunks_within_budget(monkeypatch):
    points = fetch.cells()
    assert len(points) == 1152
    calls, waits = [], []
    body = lambda n: json.dumps(payloads()[:1] * n).encode()
    monkeypatch.setattr(fetch, "get", lambda url: (calls.append(url), body(url.split("latitude=")[1].split("&")[0].count("%2C") + 1))[1])
    monkeypatch.setattr(fetch, "parse_weather", lambda p, shape: len(p))
    assert fetch.weather(45, pause=waits.append) == 1152
    assert len(calls) == 12
    billed = 1152 * 46 / 14
    assert billed < 5000
    # Paced under the per-minute limit.
    assert billed / (sum(waits) / 60) <= 600 * 1152 / 1100


def test_elements_rejects_the_repeat_download_notice(monkeypatch):
    monkeypatch.setattr(fetch, "get", lambda url, tries=4: b"GP data has not updated since your last successful download")
    with pytest.raises(fetch.SourceError):
        fetch.elements()


def test_get_stops_on_client_error_and_retries_on_server_error(monkeypatch):
    import urllib.error
    tries = []
    def opener(code):
        def fail(request, timeout):
            tries.append(code)
            raise urllib.error.HTTPError("u", code, "x", {}, None)
        return fail
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    monkeypatch.setattr(fetch.urllib.request, "urlopen", opener(400))
    with pytest.raises(fetch.SourceError):
        fetch.get("http://x")
    assert len(tries) == 1
    tries.clear()
    monkeypatch.setattr(fetch.urllib.request, "urlopen", opener(503))
    with pytest.raises(fetch.SourceError):
        fetch.get("http://x")
    assert len(tries) == 4


def test_get_retries_dropped_connections(monkeypatch):
    import http.client
    tries = []
    def drop(request, timeout):
        tries.append(1)
        raise http.client.RemoteDisconnected("closed")
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    monkeypatch.setattr(fetch.urllib.request, "urlopen", drop)
    with pytest.raises(fetch.SourceError):
        fetch.get("http://x")
    assert len(tries) == 4


def test_get_turns_a_truncated_body_into_a_source_error(monkeypatch):
    import http.client
    class Truncated:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            raise http.client.IncompleteRead(b"partial")
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda request, timeout: Truncated())
    with pytest.raises(fetch.SourceError):
        fetch.get("http://x")


def test_elements_never_retries_because_celestrak_firewalls_repeat_failures(monkeypatch):
    import urllib.error
    tries = []
    def fail(request, timeout):
        tries.append(1)
        raise urllib.error.HTTPError("u", 503, "x", {}, None)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    monkeypatch.setattr(fetch.urllib.request, "urlopen", fail)
    with pytest.raises(fetch.SourceError):
        fetch.elements()
    assert len(tries) == 1


def test_locations_with_different_hours_are_rejected():
    bad = payloads()
    bad[3] = {"hourly": {**bad[3]["hourly"], "time": ["2026-11-03T00:00", "2026-11-03T01:00"]}}
    with pytest.raises(fetch.SourceError):
        fetch.parse_weather(bad, SHAPE)
