import json
import struct

import numpy as np
import pytest

from forecast import check


def good(tmp_path):
    (tmp_path / "forecast.json").write_text('{"weather": {"loss": 1.5}}')
    (tmp_path / "planet.bin").write_bytes(struct.pack("<dHHHH", 0.0, 2, 3, 24, 48) + bytes(2 * 2 * 24 * 48))
    (tmp_path / "sats.bin").write_bytes(bytes(8 + 32 * 3))
    (tmp_path / "corrections.bin").write_bytes(struct.pack("<10s2xII4H", b"2026-11-05", 2, 4, 1, 3, 7, 15) + bytes(2 * (4 + 24 + 4)))
    np.savez(tmp_path / "history.npz", days=np.array(["2026-11-05"]))
    return tmp_path


def test_files_the_site_can_read_pass(tmp_path):
    assert check.main([str(good(tmp_path))]) == 0


def test_a_first_run_with_only_a_report_passes(tmp_path):
    (tmp_path / "forecast.json").write_text("{}")
    assert check.main([str(tmp_path)]) == 0


def test_a_missing_report_fails(tmp_path):
    assert check.main([str(tmp_path)]) == 1


@pytest.mark.parametrize("name, cut", [("planet.bin", 1), ("sats.bin", 1), ("corrections.bin", 1), ("history.npz", 10)])
def test_a_cut_file_fails(tmp_path, name, cut):
    good(tmp_path)
    path = tmp_path / name
    path.write_bytes(path.read_bytes()[:-cut])
    assert check.main([str(tmp_path)]) == 1


def test_json_that_is_not_json_fails(tmp_path):
    good(tmp_path)
    (tmp_path / "forecast.json").write_text('{"weather": ')
    assert check.main([str(tmp_path)]) == 1


def test_nan_in_the_report_fails(tmp_path):
    good(tmp_path)
    (tmp_path / "forecast.json").write_text(json.dumps({"a": {"b": [1, float("nan")]}}))
    assert check.main([str(tmp_path)]) == 1


def test_an_empty_correction_file_passes(tmp_path):
    good(tmp_path)
    (tmp_path / "corrections.bin").write_bytes(struct.pack("<10s2xII4H", b"2026-11-05", 0, 4, 1, 3, 7, 15))
    assert check.main([str(tmp_path)]) == 0
