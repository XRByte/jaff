"""KROME parser: ``@format``/``@var`` directives + comma-delimited reactions."""

from .._parser import Parser, register
from .._record import ParsedRecord, ParseResult
from .header import KromeFormatHeader
from .reaction import KromeReaction
from .var import KromeVar


@register
class KromeParser(Parser):
    """Parses the KROME format, tracking column layout and ``@var`` globals."""

    name = "krome"
    priority = 10

    def __init__(self):
        self.handlers = [KromeFormatHeader(), KromeVar(), KromeReaction()]

    def process(self, records) -> ParseResult:
        """Walk the KROME bucket in file order.

        ``@format`` headers update the local column layout, ``@var`` directives
        populate ``globals``, and reaction lines are parsed against the current
        layout into :class:`ParsedRecord` objects.
        """
        state = self._initial_state()
        globals_: dict = {}
        by_name = {h.name: h for h in self.handlers}
        reactions: list[ParsedRecord] = []

        for rec in records:
            handler = by_name[rec.format]
            if handler.is_reaction:
                fields = handler.parse(rec.line, rec.nline, state, self.file)
                reactions.append(
                    ParsedRecord(
                        **fields,
                        source_index=rec.source_index,
                        sub_order=rec.sub_order,
                    )
                )
            else:
                handler.apply(
                    rec.line, rec.nline, state, globals_, self.file, self.logger
                )

        return ParseResult(reactions, globals_)
