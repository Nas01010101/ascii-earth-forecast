import pytest

from forecast import fetch


def _offline():
    raise fetch.SourceError("aifs: no network in tests")


@pytest.fixture(autouse=True)
def no_aifs_download(request, monkeypatch):
    """The daily job's tests never reach Open-Meteo for AIFS; test_fetch and test_aifs call the download themselves."""
    if not request.module.__name__.endswith(("test_fetch", "test_aifs")):
        monkeypatch.setattr(fetch, "aifs", _offline)
