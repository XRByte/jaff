from typing import Any


def to_float_or_none(value: Any) -> float | None:
    """Coerce a band quantity to ``float``, or ``None`` when not representable.

    Band edges and averages may be plain numbers, SymPy numeric objects, or
    ``sympy.oo`` (open upper band, which casts to ``float('inf')``).  A value
    of ``None`` (e.g. the cross section of a custom-rate reaction) or a
    still-symbolic expression maps to ``None`` so it becomes ``NaN`` in a
    :class:`pandas.DataFrame`.
    """
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None
