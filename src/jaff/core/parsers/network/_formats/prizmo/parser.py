"""PRIZMO parser: arrow-notation reactions + ``VARIABLES { }`` directives."""

from sympy import parse_expr

from .._parser import Parser, register
from .._record import ParsedRecord, ParseResult
from .reaction import PrizmoReaction
from .vars import PrizmoVars


@register
class PrizmoParser(Parser):
    """Parses the PRIZMO arrow-notation format and its variables block."""

    name = "prizmo"
    priority = 30

    #: Temperature shorthands common to the KROME/PRIZMO conventions. The
    #: density accessors and ``user_*`` aliases are KROME-specific and live
    #: only on the KROME parser. A file's own ``VARIABLES`` block overrides
    #: these (applied after seeding).
    BASE_GLOBALS = {
        "invt32": "1e0 / t32",
        "invte": "1e0 / te",
        "t32": "tgas/3e2",
        "te": "tgas*8.617343e-5",
        "invtgas": "1e0 / tgas",
        "sqrtgas": "sqrt(tgas)",
    }

    def __init__(self):
        self.handlers = [PrizmoVars(), PrizmoReaction()]

    def process(self, records) -> ParseResult:
        """Walk the PRIZMO bucket in file order.

        ``VARIABLES`` directives toggle the local ``parse_vars`` state and
        populate ``globals``; reaction lines are parsed into
        :class:`ParsedRecord` objects.
        """
        state = self._initial_state()
        globals_: dict = {k: parse_expr(v) for k, v in self.BASE_GLOBALS.items()}
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
