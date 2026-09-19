# ABOUTME: Shared pytest fixtures for the JAFF test suite
# ABOUTME: Central output silencing + a make_network factory over tmp_path

import json
import logging
import os
from pathlib import Path

import pytest

from jaff import Network

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    """Silence the JAFF banner and logger so test output stays clean.

    ``Network`` prints an MOTD banner and logs INFO/WARNING lines while
    loading.  Muting both here replaces the per-test
    ``with patch("builtins.print"):`` wrapper that used to surround nearly
    every case in the old suite.

    ``logging.disable`` (not ``setLevel``) is used because every ``Network``
    construction re-runs ``JaffLogger.__init__``, which resets the "JAFF"
    logger back to ``INFO`` — a per-logger level would be clobbered mid-test.
    """
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture
def fixtures_dir() -> Path:
    """Absolute path to the bundled test-fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def make_network(tmp_path):
    """Factory building a :class:`~jaff.Network` from inline reaction lines.

    Writes *lines* to a ``.dat`` file under the test's ``tmp_path`` and loads
    it, replacing the ``NamedTemporaryFile`` + ``try/finally os.unlink`` dance
    repeated across the old suite.

    The returned callable accepts:

    lines : str | Iterable[str]
        Network body.  A string is written verbatim; any other iterable is
        joined with newlines.
    name : str, optional
        Filename to use (its stem becomes the network label).  Defaults to a
        unique ``network_<n>.dat``.
    **kwargs
        Forwarded to :class:`~jaff.Network` (e.g. ``duplicate_policy``,
        ``errors``).
    """
    counter = {"n": 0}

    def _make(lines, *, name=None, **kwargs):
        if not isinstance(lines, str):
            lines = "\n".join(lines) + "\n"
        counter["n"] += 1
        path = tmp_path / (name or f"network_{counter['n']}.dat")
        path.write_text(lines, encoding="utf-8")
        return Network(str(path), **kwargs)

    return _make


@pytest.fixture
def sample_network(fixtures_dir):
    """A :class:`~jaff.Network` loaded from the bundled ``sample_kida.dat``."""
    return Network(str(fixtures_dir / "sample_kida.dat"))


_SNAP_DIR = Path(__file__).parent / "snapshots" / "parser_output"


class _SnapshotStore:
    def __init__(self, update):
        self._update = update

    def check(self, key, value):
        safe = key.replace("/", "__")
        path = _SNAP_DIR / f"{safe}.json"
        serialized = json.dumps(value, indent=2, default=str)
        if self._update:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(serialized)
            return
        # Missing baseline must FAIL loudly, not silently auto-create — otherwise
        # a deleted/new snapshot would pass and the regression net would be blind.
        if not path.exists():
            raise FileNotFoundError(
                f"No baseline snapshot for {key!r}; record it with "
                f"JAFF_UPDATE_SNAPSHOTS=1"
            )
        expected = path.read_text()
        assert serialized == expected, f"parser output changed for {key}"


@pytest.fixture
def snapshot_store(request):
    # Run once with JAFF_UPDATE_SNAPSHOTS=1 to record the baseline.
    return _SnapshotStore(update=bool(os.environ.get("JAFF_UPDATE_SNAPSHOTS")))
