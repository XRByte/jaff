"""KROME parser: ``@format``/``@var`` directives + comma-delimited reactions."""

from sympy import parse_expr

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

    # Canonical JAFF symbol aliases for common Fortran/KROME shorthand
    # variables such as ``t32``, ``te``, ``invtgas``, and ``sqrtgas``, so that
    # they are resolved automatically during rate normalization.  Order
    # matters: compound aliases (invt32, invte) must be listed before the
    # simpler ones they depend on so that resolve_symbolic_dependencies can
    # substitute correctly.
    BASE_GLOBALS = {
        "invt32": "1e0 / t32",
        "invte": "1e0 / te",
        "t32": "tgas/3e2",
        "te": "tgas*8.617343e-5",
        "invtgas": "1e0 / tgas",
        "sqrtgas": "sqrt(tgas)",
        "user_tdust": "tdust",
        "user_av": "av",
        "get_hnuclei(n)": "n_H_nuc",
        "n(idx_h2)": "n_H2",
        "n(idx_h)": "n_H",
        "n_global(idx_h2)": "n_H2",
    }

    def __init__(self):
        self.handlers = [KromeFormatHeader(), KromeVar(), KromeReaction()]

    def process(self, records) -> ParseResult:
        """Walk the KROME bucket in file order.

        ``globals`` is first seeded with the canonical KROME shorthand
        aliases (:attr:`BASE_GLOBALS`), then ``@format`` headers update the
        local column layout, ``@var`` directives populate ``globals``
        (potentially overriding the seeded aliases), and reaction lines are
        parsed against the current layout into :class:`ParsedRecord` objects.
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
