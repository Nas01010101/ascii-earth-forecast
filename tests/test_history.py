from datetime import timedelta

import numpy as np

from forecast import fetch, run
from synthetic import TODAY, hourly


def test_only_complete_days_before_today_enter_history():
    hours, fields = hourly(3)
    history = {}
    run.add_days(history, hours[5:], {k: v[5:] for k, v in fields.items()}, TODAY)
    # The first day lost five hours and today and tomorrow are not complete past days.
    assert sorted(history) == ["2026-11-01", "2026-11-02"]


def test_history_is_trimmed_to_the_window():
    hours, fields = hourly(run.HISTORY_DAYS + 5)
    history = {}
    run.add_days(history, hours, fields, TODAY)
    assert len(history) == run.HISTORY_DAYS


def test_a_day_with_too_many_missing_cells_is_skipped_not_fatal():
    hours, fields = hourly(2)
    fields["cloud"][:24] = np.nan
    history = {}
    run.add_days(history, hours, fields, TODAY)
    assert sorted(history) == ["2026-11-02"]


def test_one_null_hour_does_not_lose_the_cell():
    hours, fields = hourly(1)
    fields["cloud"][3] = np.nan
    history = {}
    run.add_days(history, hours, fields, TODAY)
    assert not np.isnan(history["2026-11-02"][0]).any()


def test_stored_days_are_never_rewritten():
    hours, fields = hourly(1)
    history = {"2026-11-02": ("kept", "kept")}
    run.add_days(history, hours, fields, TODAY)
    assert history["2026-11-02"] == ("kept", "kept")


def past(days):
    return {(TODAY - timedelta(days=d)).isoformat(): None for d in range(1, days + 1)}


def test_fetch_asks_only_for_missing_days_and_heals_holes():
    steps = past(run.STEP_DAYS)
    assert run.days_to_fetch({}, steps, TODAY) == fetch.BOOTSTRAP_DAYS
    full = past(fetch.BOOTSTRAP_DAYS)
    assert run.days_to_fetch(full, steps, TODAY) == 0
    del full[(TODAY - timedelta(days=1)).isoformat()]
    assert run.days_to_fetch(full, steps, TODAY) == 1
    del full[(TODAY - timedelta(days=5)).isoformat()]
    assert run.days_to_fetch(full, steps, TODAY) == 5


def test_a_missing_step_day_also_fetches_the_day_before_for_its_rain():
    full = past(fetch.BOOTSTRAP_DAYS)
    assert run.days_to_fetch(full, {}, TODAY) == run.STEP_DAYS + 1
    steps = past(run.STEP_DAYS)
    del steps[(TODAY - timedelta(days=3)).isoformat()]
    assert run.days_to_fetch(full, steps, TODAY) == 4


def test_a_gap_stops_the_consecutive_run():
    history = {d: None for d in ("2026-11-02", "2026-11-01", "2026-10-30")}
    assert run.consecutive(history, TODAY) == ["2026-11-01", "2026-11-02"]


def test_a_rare_missing_cell_is_filled_in_a_stored_step():
    hours, fields = hourly(2)
    fields["water"][hours == np.datetime64(TODAY - timedelta(days=1), "h"), 3, 4] = np.nan
    steps = {}
    run.add_steps(steps, hours, fields, TODAY)
    assert sorted(steps) == [(TODAY - timedelta(days=1)).isoformat()]
    assert not np.isnan(steps[(TODAY - timedelta(days=1)).isoformat()]).any()


def test_stored_steps_are_never_rewritten_and_the_window_is_trimmed():
    hours, fields = hourly(run.STEP_DAYS + 5)
    old = (TODAY - timedelta(days=run.STEP_DAYS + 3)).isoformat()
    held = (TODAY - timedelta(days=2)).isoformat()
    steps = {old: "old", held: "kept"}
    run.add_steps(steps, hours, fields, TODAY)
    assert old not in steps and steps[held] == "kept"
    assert len(steps) == run.STEP_DAYS


def test_steps_round_trip_through_the_data_directory(tmp_path):
    values = np.arange(7 * 4 * 2 * 2, dtype=np.float32).reshape(7, 4, 2, 2)
    run.save_steps(tmp_path, {"2026-11-02": values, "2026-11-01": values + 1})
    saved = run.load_steps(tmp_path)
    assert sorted(saved) == ["2026-11-01", "2026-11-02"]
    assert np.array_equal(saved["2026-11-01"], values + 1)
    assert not list(tmp_path.glob("*.tmp*"))
