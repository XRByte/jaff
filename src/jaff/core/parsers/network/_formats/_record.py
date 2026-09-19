class Record:
    """A detected-but-not-yet-processed reaction line.

    Captured in the engine's detection phase and handed to the owning format's
    ``process`` method. ``metadata`` snapshots the live parser state the format
    needs, so deferred processing never reads stale mutable state.

    Attributes
    ----------
    source_index : int
        File-side reaction number (0-based, stable under dedup).
    line : str
        Raw file line.
    nline : int
        1-based file line number (for error messages).
    format : str
        Owning format name.
    metadata : dict
        Snapshot of the format state this line depends on.
    sub_order : int
        Tie-breaker for a single line expanding into several reactions.
    """

    def __init__(
        self,
        source_index: int,
        line: str,
        nline: int,
        format: str,
        metadata: dict | None = None,
        sub_order: int = 0,
    ):
        self.source_index: int = source_index
        self.line: str = line
        self.nline: int = nline
        self.format: str = format
        self.metadata: dict = metadata if metadata is not None else {}
        self.sub_order: int = sub_order


class ParsedRecord:
    """A fully parsed reaction produced by a format family's ``process``.

    The engine sorts these by ``(source_index, sub_order)`` and converts each to
    the ``parsedListProps`` dict consumed by ``Network``.
    """

    def __init__(
        self,
        r: list[str],
        p: list[str],
        tmin: float | None,
        tmax: float | None,
        rate: str,
        type: str,
        string: str,
        source_index: int,
        sub_order: int = 0,
    ):
        self.r = r
        self.p = p
        self.tmin = tmin
        self.tmax = tmax
        self.rate = rate
        self.type = type
        self.string = string
        self.source_index = source_index
        self.sub_order = sub_order

    def as_parsed_props(self) -> dict:
        """Convert to the engine's ``parsedListProps`` dict (+ source_index)."""
        return {
            "r": self.r,
            "p": self.p,
            "tmin": self.tmin,
            "tmax": self.tmax,
            "rate": self.rate,
            "type": self.type,
            "string": self.string,
            "source_index": self.source_index,
        }


class ParseResult:
    """The output of a :class:`~._parser.Parser`'s ``process``.

    Attributes
    ----------
    reactions : list[ParsedRecord]
        Fully parsed reactions produced from the parser's bucket.
    globals : dict
        Symbolic globals (e.g. ``@var`` / ``VARIABLES`` entries) the parser
        discovered; merged into the engine's global symbol map.
    """

    def __init__(self, reactions: list["ParsedRecord"], globals: dict):
        self.reactions: list["ParsedRecord"] = reactions
        self.globals: dict = globals
