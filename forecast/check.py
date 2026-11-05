"""Refuses to publish a data folder whose files the site could not read. usage: python -m forecast.check DATA_DIR"""
import json
import math
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np


class BadData(Exception):
    pass


def _no_nan(value, where="forecast.json"):
    if isinstance(value, float) and not math.isfinite(value):
        raise BadData(f"{where} holds {value}")
    if isinstance(value, dict):
        for key, item in value.items():
            _no_nan(item, f"{where}.{key}")
    elif isinstance(value, list):
        for item in value:
            _no_nan(item, where)


def _report(path):
    _no_nan(json.loads(path.read_text()))


def _planet(path):
    data = path.read_bytes()
    _, frames, _, rows, cols = struct.unpack_from("<dHHHH", data)
    want = struct.calcsize("<dHHHH") + 2 * frames * rows * cols
    if not frames or len(data) != want:
        raise BadData(f"planet.bin is {len(data)} bytes, its header says {want}")


def _sats(path):
    size = path.stat().st_size
    if size <= 8 or (size - 8) % 32:
        raise BadData(f"sats.bin is {size} bytes, not 8 plus 32 per satellite")


def _corrections(path):
    data = path.read_bytes()
    head = "<10s2xII4H"
    _, n, count, *_ = struct.unpack_from(head, data)
    want = struct.calcsize(head) + n * (4 + 6 * count + 4)
    if len(data) != want:
        raise BadData(f"corrections.bin is {len(data)} bytes, its header says {want}")


CHECKS = {"forecast.json": _report, "planet.bin": _planet, "sats.bin": _sats, "corrections.bin": _corrections}
REQUIRED = {"forecast.json"}


def check(root):
    """Raises BadData on the first file the site could not read. Files a failed source never wrote are allowed."""
    root = Path(root)
    for name, read in CHECKS.items():
        path = root / name
        if not path.exists():
            if name in REQUIRED:
                raise BadData(f"{name} is missing")
            continue
        try:
            read(path)
        except (ValueError, struct.error) as error:
            raise BadData(f"{name}: {error}") from error
    for path in root.glob("*.npz"):
        try:
            with np.load(path) as stored:
                for key in stored.files:
                    stored[key]
        except (ValueError, OSError, EOFError, zipfile.BadZipFile) as error:
            raise BadData(f"{path.name}: {error}") from error


def main(argv):
    try:
        check(argv[0])
    except BadData as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
