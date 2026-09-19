"""KIDA parser: a single fixed-width reaction line-type."""

from .._parser import Parser, register
from .._record import ParsedRecord, ParseResult
from .reaction import KidaReaction


@register
class KidaParser(Parser):
    """Parses the KIDA fixed-width reaction format."""

    name = "kida"
    priority = 80

    def __init__(self):
        self.handlers = [KidaReaction()]

    def process(self, records) -> ParseResult:
        """Parse each KIDA reaction record into a :class:`ParsedRecord`."""
        state = self._initial_state()
        by_name = {h.name: h for h in self.handlers}
        reactions: list[ParsedRecord] = []

        for rec in records:
            handler = by_name[rec.format]
            fields = handler.parse(rec.line, rec.nline, state, self.file)
            reactions.append(
                ParsedRecord(
                    **fields,
                    source_index=rec.source_index,
                    sub_order=rec.sub_order,
                )
            )

        return ParseResult(reactions, {})
