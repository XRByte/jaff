# ABOUTME: Shared fixtures for jaff.plotting tests
# ABOUTME: Forces the Agg backend and closes figures after every test

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test opened."""
    yield
    plt.close("all")


@pytest.fixture
def jaff_warnings(monkeypatch):
    """Record JAFF logger warnings (the root conftest disables logging output)."""
    import logging

    messages: list[str] = []
    logger = logging.getLogger("JAFF")
    monkeypatch.setattr(logger, "warning", lambda msg, *a, **k: messages.append(msg))
    return messages
