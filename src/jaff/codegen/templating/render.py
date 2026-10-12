# ABOUTME: Renders split template blocks: validates every block first, then renders
# ABOUTME: Resolves USE layering, registers DEFINEs and echoes directives as comments

from typing import Dict, List, Optional, Set

from ...errors import ParserError
from .blocks import Block, CommandBlock, TextBlock
from .commands import COMMANDS, Define
from .context import RenderContext
from .errors import TemplateError
from .options import Options
from .resolve import check_groups, compile_replacements, layer


def _all_groups() -> Dict[str, Set[str]]:
    """Every option group any command accepts, with the union of its keys."""
    groups: Dict[str, Set[str]] = {}
    for cls in COMMANDS.values():
        for name, keys in cls.groups.items():
            groups.setdefault(name, set()).update(keys)
    return groups


def _error(
    block: CommandBlock, message: str, nline: Optional[int], ctx: RenderContext
) -> ParserError:
    """ParserError at *nline* (default: the directive's first line)."""
    nline = nline or block.line
    index = nline - block.line
    lines = block.header_lines
    text = lines[index] if 0 <= index < len(lines) else lines[0]
    return ParserError(message, text.rstrip("\r\n"), nline, ctx.file)


def _prepare(block: CommandBlock, ctx: RenderContext) -> Options:
    """Validate one block and return its layered options; register DEFINEs."""
    command = block.command
    try:
        options = layer(block.options, ctx.defines)
        accepted = _all_groups() if isinstance(command, Define) else command.groups
        check_groups(command.keyword, options, accepted)
        compile_replacements(options)
        command.validate(options, ctx)
    except TemplateError as e:
        raise _error(block, str(e), e.nline, ctx) from None

    if isinstance(command, Define):
        if command.name in ctx.defines:
            raise _error(block, f"DEFINE {command.name} given twice", None, ctx)
        ctx.defines[command.name] = options
    return options


def echo(lines: List[str], comment: str) -> str:
    """Directive lines as comments, each keeping its own indentation."""
    out = []
    for line in lines:
        text = line.rstrip("\r\n")
        content = text.lstrip()
        indent = text[: len(text) - len(content)]
        out.append(f"{indent}{comment} {content}".rstrip() + "\n")
    return "".join(out)


def render_blocks(blocks: List[Block], ctx: RenderContext) -> str:
    """Validate every block, then render them all in order.

    Args:
        blocks: Blocks from ``split_blocks``.
        ctx: Render context; its ``defines`` fill up as DEFINEs are met.

    Returns:
        The generated text.

    Raises:
        ParserError: On the first invalid block, before anything is rendered.
    """
    prepared = [
        _prepare(block, ctx) if isinstance(block, CommandBlock) else None
        for block in blocks
    ]
    comment = ctx.cg.lang.comment
    out: List[str] = []
    for block, options in zip(blocks, prepared):
        if isinstance(block, TextBlock):
            out.extend(block.lines)
            continue

        if ctx.echo_directives:
            out.append(echo(block.header_lines, comment))
        out.append(block.command.render(block.body_lines, options, ctx))
        if ctx.echo_directives:
            out.append(echo(block.end_lines, comment))
    return "".join(out)
