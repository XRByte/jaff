"""PRIZMO parser: arrow-notation reactions + ``VARIABLES { }`` directives."""

from .._parser import Parser, register
from .._record import ParsedRecord, ParseResult
from .reaction import PrizmoReaction
from .vars import PrizmoVars


@register
class PrizmoParser(Parser):
    """Parses the PRIZMO arrow-notation format and its variables block."""

    name = "prizmo"
    priority = 30

    def __init__(self):
        self.handlers = [PrizmoVars(), PrizmoReaction()]

    def process(self, records) -> ParseResult:
        """Walk the PRIZMO bucket in file order.

        ``VARIABLES`` directives toggle the local ``parse_vars`` state and
        populate ``globals``; reaction lines are parsed into
        :class:`ParsedRecord` objects.
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
