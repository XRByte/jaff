# ABOUTME: Single pass over a template that splits it into text and directive blocks
# ABOUTME: Lark decides where a header ends; the engine collects the body up to $$JAFF END

from pathlib import Path
from typing import List, Optional, Tuple

from ...errors import ParserError
from .blocks import Block, CommandBlock, TextBlock
from .commands import Comment, Define, End, FileSetting
from .options import Options
from .syntax import (
    MARKER,
    Directive,
    DirectiveSyntaxError,
    IncompleteHeaderError,
    SplitOptionError,
    parse_header,
)


def is_directive(line: str) -> bool:
    """Whether *line* starts a ``$$JAFF`` directive."""
    return line.lstrip().startswith(MARKER)


def _split_lines(text: str) -> List[str]:
    """Split on ``\\n`` only, keeping line ends (so ``\\r\\n`` stays attached)."""
    lines = [part + "\n" for part in text.split("\n")]
    lines[-1] = lines[-1][:-1]

    return lines if lines[-1] else lines[:-1]


def _src(line: str) -> str:
    """Source line without its trailing line break, for error messages."""
    return line.rstrip("\r\n")


def _read_header(lines: List[str], start: int, file: Path) -> Tuple[Directive, int]:
    """Parse the directive starting at *start*; return it and the next line index.

    The header continues onto the next line only while the parser reports that it
    ended inside an open option block.
    """
    text = lines[start]
    end = start + 1
    while True:
        try:
            return parse_header(text, start + 1, file), end
        except IncompleteHeaderError:
            if end == len(lines) or is_directive(lines[end]):
                raise ParserError(
                    "Unclosed { in directive header", _src(lines[start]), start + 1, file
                ) from None

            text += lines[end]
            end += 1

        except ParserError as e:
            if end == start + 1 or e.nline == start + 1:
                raise

            hint = f" (in directive opened at line {start + 1}"
            brace = isinstance(e, DirectiveSyntaxError) and not isinstance(
                e, SplitOptionError
            )
            hint += "; missing '}'?)" if brace else ")"

            raise ParserError(e.message + hint, e.line, e.nline, e.fname) from None


def _read_body(
    lines: List[str], opened: int, start: int, file: Path
) -> Tuple[List[str], List[str], int]:
    """Collect body lines from *start* up to ``$$JAFF END``.

    *opened* is the index of the directive's first header line, for errors.
    Returns the body lines, the END line(s) and the index after the END line.
    """
    i = start
    while i < len(lines):
        if is_directive(lines[i]):
            directive, after = _read_header(lines, i, file)
            if not isinstance(directive.command, End):
                raise ParserError(
                    "Nested $$JAFF directive (missing $$JAFF END?)",
                    _src(lines[i]),
                    i + 1,
                    file,
                )

            if directive.options is not None:
                raise ParserError(
                    "$$JAFF END takes no options", _src(lines[i]), i + 1, file
                )

            return lines[start:i], lines[i:after], after

        i += 1

    raise ParserError("Missing $$JAFF END", _src(lines[opened]), opened + 1, file)


def _check_define(
    options: Optional[Options], lines: List[str], start: int, file: Path
) -> None:
    """A DEFINE needs an option block and cannot pull in other sets with USE.

    *start* is the index of the DEFINE's first header line.
    """
    if options is None:
        raise ParserError(
            "DEFINE needs an option block { ... }", _src(lines[start]), start + 1, file
        )

    if options.uses:
        nline = options.use_lines[options.uses[0]]
        raise ParserError(
            "DEFINE cannot contain USE", _src(lines[nline - 1]), nline, file
        )


def _check_setting(
    block: CommandBlock, blocks: List[Block], lines: List[str], file: Path
) -> None:
    """A file setting has no options, comes before other directives, and only once.

    *blocks* are the blocks before *block*.
    """
    keyword = block.command.keyword
    start = block.line - 1

    def error(message: str) -> ParserError:
        return ParserError(message, _src(lines[start]), block.line, file)

    if block.options is not None:
        raise error(f"$$JAFF {keyword} takes no options")

    if isinstance(block.command, Comment) and not block.command.token.strip():
        raise error('$$JAFF COMMENT needs a comment token, e.g. COMMENT "#"')

    earlier = [b.command for b in blocks if isinstance(b, CommandBlock)]
    if any(not isinstance(command, FileSetting) for command in earlier):
        raise error(f"$$JAFF {keyword} must come before the other $$JAFF directives")

    if any(command.keyword == keyword for command in earlier):
        raise error(f"$$JAFF {keyword} given twice")


def split_blocks(text: str, file: Path) -> List[Block]:
    """Split a template into text and directive blocks in one pass.

    Args:
        text: Full template text.
        file: Template path, for error messages.

    Returns:
        Blocks in template order.

    Raises:
        ParserError: On a malformed directive or block structure.
    """
    lines = _split_lines(text)
    blocks: List[Block] = []
    i = 0

    while i < len(lines):
        if not is_directive(lines[i]):
            if blocks and isinstance(blocks[-1], TextBlock):
                blocks[-1].lines.append(lines[i])
            else:
                blocks.append(TextBlock([lines[i]], i + 1))

            i += 1
            continue

        directive, after = _read_header(lines, i, file)
        command, options = directive.command, directive.options
        if isinstance(command, End):
            raise ParserError(
                "$$JAFF END without a directive", _src(lines[i]), i + 1, file
            )

        if isinstance(command, Define):
            _check_define(options, lines, i, file)

        body: List[str] = []
        end_lines: List[str] = []
        end = after
        if command.has_body:
            body, end_lines, end = _read_body(lines, i, after, file)

        header = lines[i:after]
        block = CommandBlock(command, options, header, body, end_lines, i + 1)
        if isinstance(command, FileSetting):
            _check_setting(block, blocks, lines, file)

        blocks.append(block)
        i = end

    return blocks
