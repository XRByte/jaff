# ABOUTME: Adapters turning JAFF objects into plotting series dicts
# ABOUTME: rates (rate coefficient vs T) and xsecs (photo cross sections)

"""Adapters: physics objects -> plain series dicts for :mod:`jaff.plotting.spec`."""

from .rates import rate_series
from .xsecs import XSEC_PROCESSES, normalize_processes, xsec_series

__all__ = ["rate_series", "xsec_series", "normalize_processes", "XSEC_PROCESSES"]
