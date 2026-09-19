from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._context import ParseContext
    from ._record import Record


class NetworkFormat(ABC):
    """Base interface for a single network-file format plugin.

    Subclasses live in their own module under ``_formats`` and register
    themselves so the engine can discover them.  Priority — not file or import
    order — determines the order in which formats are matched against a line.

    Class attributes
    ----------------
    priority : int
        Match order; lower is tried first.
    name : str
        Unique format identifier.
    state_key : str
        Namespace into ``ParseContext.state`` for this format's mutable props.
        Formats that must share live state (e.g. a ``@format`` header and the
        reaction lines it configures) declare the *same* ``state_key``.  The
        empty string means the format keeps no state.
    emits_reactions : bool
        ``True`` for a *reaction* format: on a match the engine ``capture``s the
        line into a per-format bucket for deferred ``process``.  ``False``
        (default) for a *directive* format (e.g. a ``@format`` header or
        ``@var``): the engine calls ``handle`` inline so the state and globals
        those directives set are live before reactions are processed.
    family : str
        Subpackage name grouping this format with its siblings for
        ``FormatFamily``-level processing (e.g. ``"krome"``).
    """

    priority: int
    name: str
    state_key: str = ""
    emits_reactions: bool = False
    family: str = ""

    def default_state(self) -> dict[Any, Any]:
        """Return this format's initial mutable props.

        Merged into ``ParseContext.state[self.state_key]`` once at parser
        construction.  Formats sharing a ``state_key`` have their dicts merged.

        Returns
        -------
        dict
            Initial state for this format (empty by default).
        """
        return {}

    def state(self, ctx: "ParseContext") -> dict:
        """Return this format's live state slice from *ctx*.

        Parameters
        ----------
        ctx : ParseContext
            Shared parse context.

        Returns
        -------
        dict
            The mutable dict at ``ctx.state[self.state_key]``; writes are
            visible to every format sharing the same ``state_key``.
        """
        return ctx.state[self.state_key]

    @abstractmethod
    def _global_re(self, ctx: "ParseContext") -> re.Pattern:
        """Return the compiled regex used to classify a line as this format."""
        pass

    @abstractmethod
    def _local_re(self, ctx: "ParseContext") -> re.Pattern:
        """Return the compiled regex used to extract fields from a line.

        Recomputed per call so it reflects the current ``ctx.state`` (e.g. the
        KROME column counts updated by a ``@format`` header).
        """
        pass

    @abstractmethod
    def handle(self, match: re.Match, ctx: "ParseContext") -> None:
        """Process a matched line, mutating *ctx* (append a reaction, update state)."""
        pass

    def capture(self, match: re.Match, ctx: "ParseContext") -> "Record":
        """Return a Record snapshotting the state this format needs.

        Default snapshots this format's whole state slice. Formats override to
        capture a narrower snapshot.
        """
        from ._record import Record

        meta = dict(self.state(ctx)) if self.state_key else {}
        return Record(
            source_index=-1,  # engine assigns
            line=ctx.line,
            nline=ctx.nline,
            format=self.name,
            metadata=meta,
        )

    def process(self, records: list["Record"], ctx: "ParseContext") -> None:
        """Process this format's captured records into parsed reactions.

        Default: replay each record's snapshot into ctx and call ``handle`` so
        unmigrated formats behave exactly as before.
        """
        for rec in records:
            if self.state_key:
                ctx.state[self.state_key].clear()
                ctx.state[self.state_key].update(rec.metadata)

            ctx.line = rec.line
            ctx.nline = rec.nline
            before = len(ctx.parsed_list)
            m = self._global_re(ctx).match(rec.line)

            if m is None:
                ctx.raise_error(
                    f"{self.name} failed to re-match a captured reaction line"
                )

            self.handle(m, ctx)
            for entry in ctx.parsed_list[before:]:
                entry.setdefault("source_index", rec.source_index)
