# ABOUTME: Characterization snapshot of NetworkParser.get_parsed() over all networks
# ABOUTME: Regression net guarding the format-ownership refactor (behavior-preserving)

import glob

import pytest

from jaff.core.parsers.network._engine import NetworkParser

# Every bundled network file the parser can read. `*.rates.jet` is a subset of
# `*.jet`, so dedup to avoid parametrizing the same file twice.
NETWORK_FILES = sorted(
    {
        f
        for pat in ("networks/**/*.jet", "networks/**/*.dat")
        for f in glob.glob(pat, recursive=True)
    }
)


def _fingerprint(path):
    with NetworkParser(path) as p:
        parsed, _ = p.get_parsed()
    return [
        (i, tuple(r["r"]), tuple(r["p"]), r["tmin"], r["tmax"], r["rate"], r["type"])
        for i, r in enumerate(parsed)
    ]


@pytest.mark.parametrize("path", NETWORK_FILES)
def test_parser_output_is_stable(path, snapshot_store):
    fp = _fingerprint(path)
    snapshot_store.check(path, fp)
