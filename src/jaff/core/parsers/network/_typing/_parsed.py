from typing import NotRequired, TypedDict

parsedListProps = TypedDict(
    "parsedListProps",
    {
        "r": list[str],
        "p": list[str],
        "tmin": float | None,
        "tmax": float | None,
        "rate": str,
        "type": str,
        "string": str,
        "source_index": NotRequired[int],
    },
)
