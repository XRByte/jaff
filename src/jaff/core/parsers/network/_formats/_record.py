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
