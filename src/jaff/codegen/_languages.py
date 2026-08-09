from __future__ import annotations

import copy
import functools
import inspect
import re
from collections.abc import Callable
from typing import Any, ClassVar

import sympy as sp

from ..errors import InvalidLanguageError

# --------------------------------------------------------------------------- #
# Fortran free-form line wrapping                                              #
# --------------------------------------------------------------------------- #

# Significant column count for a free-form Fortran line (max 132 per F90+; we
# wrap conservatively at 120).  Two columns are reserved on a wrapped line for
# the trailing " &" marker.
_FORTRAN_WIDTH = 120
_FORTRAN_CONT_INDENT = "      "
_FORTRAN_CONT_MARK = " &"
_FORTRAN_CONT_LEAD = "&"
_FORTRAN_ATOM = re.compile(
    r"[A-Za-z_]\w*\s*\(\s*\d+(?:\s*,\s*\d+)*\s*\)"  # integer-indexed array ref
    r"|[A-Za-z_]\w*"  # identifier / intrinsic name
    r"|\d+\.?\d*(?:[dDeE][+-]?\d+)?"  # numeric literal (incl. d-exponent)
    r"|\s+"  # run of whitespace
    r"|."  # any single other character
)

_FORTRAN_MAX_EXP = 308
_FORTRAN_CLAMP_LIT = "1.0d300"
# Real literal with an explicit d/e exponent: mantissa, exponent letter, sign,
# digits.  The lookbehind avoids matching inside an identifier or after a dot.
_FORTRAN_REAL_LIT = re.compile(r"(?<![\w.])\d+\.?\d*[dDeE]([+-]?)(\d+)")


def sanitize_fortran_literals(text: str) -> str:
    """Clamp real literals whose exponent overflows real*8 to a finite sentinel.

    A literal such as ``3.04782042452573d+83651`` -- produced when SymPy
    constant-folds an interpolation fit (``10**poly``) at an out-of-domain
    boundary inside an always-false cooling-domain guard -- exceeds the IEEE
    double exponent range (~308) and is unrepresentable, so gfortran rejects it
    with *"Syntax error in expression"*.  Every overflowing positive-exponent
    literal is replaced by ``1.0d300``; because they occur only in numerically
    inert guard conditions the replacement does not change behaviour.  Negative
    (underflowing) exponents are legal and left untouched.

    Parameters
    ----------
    text : str
        Generated Fortran source.

    Returns
    -------
    str
        Source with overflowing real literals clamped.
    """

    def _clamp(m: "re.Match[str]") -> str:
        sign, digits = m.group(1), m.group(2)
        if sign != "-" and int(digits) > _FORTRAN_MAX_EXP:
            return _FORTRAN_CLAMP_LIT
        return m.group(0)

    return _FORTRAN_REAL_LIT.sub(_clamp, text)


_FORTRAN_NOWRAP_LEADERS = (
    "real",
    "integer",
    "logical",
    "complex",
    "double",
    "character",
    "type",
    "implicit",
    "use",
    "module",
    "subroutine",
    "function",
    "end",
    "contains",
    "return",
    "common",
    "parameter",
    "save",
    "dimension",
    "external",
    "intrinsic",
    "data",
    "include",
)


def _fortran_fold(code: str) -> list[str]:
    """Fold free-form ``&`` continuations back into logical statements.

    A free-form continuation is a line whose predecessor ends with ``&``; the
    continued line may optionally begin with its own leading ``&``.  Both
    markers are stripped and the fragments concatenated so each returned entry
    is one whole logical statement (leading indentation of the first physical
    line preserved).
    """
    logical: list[str] = []
    open_cont = False
    for phys in code.split("\n"):
        if open_cont:
            frag = phys.strip()
            if frag.startswith("&"):
                frag = frag[1:].lstrip()
            logical[-1] += frag
        else:
            logical.append(phys.rstrip())
        # A trailing "&" (now on the folded line) opens the next continuation.
        if logical[-1].rstrip().endswith("&"):
            logical[-1] = logical[-1].rstrip()[:-1].rstrip()
            open_cont = True
        else:
            open_cont = False
    return logical


def _fortran_free_wrap(code: str, width: int = _FORTRAN_WIDTH) -> str:
    """Re-flow free-form Fortran so no token is split across a continuation.

    SymPy's ``fcode`` (``source_format='free'``) wraps long expressions by
    breaking wherever the limit falls -- including in the middle of a token
    such as ``nden(1, 1)`` or ``flux(3)``.  A split token defeats regex
    post-processing (and reads badly), so this reconstructs each logical
    statement and re-wraps it, choosing break points only *between* atomic
    tokens.  Wrapped lines end with the free-form ``&`` continuation marker;
    the maximum atom is far shorter than the line width, so every emitted line
    fits within ``width`` columns.

    Parameters
    ----------
    code : str
        Free-form Fortran as produced by :func:`sympy.fcode` (statement lines
        plus ``&``-continued lines).
    width : int, optional
        Significant column count.  Default 132.

    Returns
    -------
    str
        Equivalent code whose continuations never fall inside a token.
    """
    out: list[str] = []
    # Room for the trailing " &" marker on any line that will be continued.
    limit = width - len(_FORTRAN_CONT_MARK)
    for stmt in _fortran_fold(code):
        indent = stmt[: len(stmt) - len(stmt.lstrip())]
        content = stmt.strip()
        if not content:
            out.append("")
            continue
        if len(indent) + len(content) <= width:
            out.append(indent + content)
            continue

        # Continuation lines begin with a leading "&" so a break inside a token
        # (e.g. "**") rejoins exactly, with no inserted indentation blanks.
        cont_prefix = indent + _FORTRAN_CONT_INDENT + _FORTRAN_CONT_LEAD
        tokens = _FORTRAN_ATOM.findall(content)
        line = indent
        prefix = indent
        for tok in tokens:
            # Never start a line with the leftover whitespace from a break: a
            # blank right after a leading "&" would be a significant blank.
            if line == prefix and tok.isspace():
                continue
            if len(line) + len(tok) > limit and line != prefix:
                out.append(line.rstrip() + _FORTRAN_CONT_MARK)
                prefix = cont_prefix
                line = prefix
                if tok.isspace():
                    continue
            line += tok
        if line.strip():
            out.append(line.rstrip())

    return "\n".join(out)


def _fortran_code_gen(expr: Any, **kwargs: Any) -> str:
    """Serialise *expr* to free-form Fortran with split-safe line wrapping."""
    return _fortran_free_wrap(sp.fcode(expr, standard=95, source_format="free", **kwargs))


def wrap_fortran_source(text: str, width: int = _FORTRAN_WIDTH) -> str:
    """Re-wrap every over-long executable statement in a fixed-form source.

    :func:`_fortran_code_gen` only wraps expressions that pass through
    :func:`sympy.fcode`.  Many generated statements -- e.g. the species ODE
    right-hand sides ``dn(i) = -flux(1) + flux(3) + ...`` -- are assembled by
    plain string concatenation and never reach the printer, so they can exceed
    the free-form line limit.  This pass walks the finished source and re-wraps
    each executable statement that is too long, splitting only between atomic
    tokens (array references such as ``flux(3)`` stay intact).

    Declarations, structural keywords and comment lines are left untouched, as
    are statements that already fit within *width*.

    Parameters
    ----------
    text : str
        Complete generated Fortran source.
    width : int, optional
        Significant column count.  Default 72.

    Returns
    -------
    str
        Source with over-long executable statements wrapped.
    """
    # Clamp any real literal whose exponent overflows real*8 before wrapping,
    # so an absurd token (e.g. ``3.04d+83651``) never survives into the output.
    text = sanitize_fortran_literals(text)

    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Gather any continuation lines belonging to this logical statement: in
        # free form, a line is continued when it ends with a trailing "&".
        unit = [line]
        j = i + 1
        while unit[-1].rstrip().endswith("&") and j < len(lines):
            unit.append(lines[j])
            j += 1
        i = j

        stripped = line.lstrip()
        leader = stripped.split("(", 1)[0].split()[0].lower() if stripped else ""
        is_comment = stripped.startswith("!")
        is_decl = "::" in line or leader in _FORTRAN_NOWRAP_LEADERS

        if is_comment or is_decl:
            out.extend(unit)
        elif len(unit) > 1 or len(line) > width:
            out.append(_fortran_free_wrap("\n".join(unit), width))
        else:
            out.append(line)

    return "\n".join(out)


class Language:
    """Base class and factory for target-language code-generation config.

    Each supported language is a subclass that declares its syntax conventions
    as class attributes (brackets, assignment operator, line terminator, SymPy
    printer, index offset, …).  Defining a subclass auto-registers a singleton
    instance plus its aliases, so adding a language requires only a new subclass
    with no central edits.

    Resolve a language by alias with the base factory::

        cxx = Language("c++")  # -> the registered Cxx singleton
        cxx.code_gen(expr)  # sp.cxxcode(expr)
    """

    _register: ClassVar[dict[str, "Language"]] = {}
    LOOKUP: ClassVar[dict[str, str]] = {}

    # Override vocabulary consumed by derive(). "[,]" -> J[i, j]; "[]" -> J[i][j].
    BRACKET_FORMATS: ClassVar[tuple[str, ...]] = ("()", "{}", "[]", "<>")
    MATRIX_FORMATS: ClassVar[dict[str, dict[str, str]]] = {
        "()": {"brac": "()", "sep": ")("},
        "()()": {"brac": "()", "sep": ")("},
        "(,)": {"brac": "()", "sep": ", "},
        "[]": {"brac": "[]", "sep": "]["},
        "[][]": {"brac": "[]", "sep": "]["},
        "[,]": {"brac": "[]", "sep": ", "},
        "{}": {"brac": "{}", "sep": "}{"},
        "{}{}": {"brac": "{}", "sep": "}{"},
        "{,}": {"brac": "{}", "sep": ", "},
        "<>": {"brac": "<>", "sep": "><"},
        "<><>": {"brac": "<>", "sep": "><"},
        "<,>": {"brac": "<>", "sep": ", "},
    }

    # Immutable per-language config set by each subclass.
    name: ClassVar[str]
    aliases: ClassVar[tuple[str, ...]] = ()
    brac: ClassVar[str]
    matrix_sep: ClassVar[str]
    code_gen: ClassVar[Callable[..., str]]
    idx_offset: ClassVar[int]
    comment: ClassVar[str]
    types: ClassVar[dict[str, str]]
    extras: ClassVar[dict[str, Any]]

    # Overridable tokens: class value is the default, derive() shadows them on
    # an ephemeral copy. Plain annotations, not ClassVar (which bans instance
    # assignment). lb/rb/mlb/mrb/sep derived by __init_subclass__ from `brac`.
    assignment_op: str
    line_end: str
    lb: str
    rb: str
    mlb: str
    mrb: str
    sep: str

    def __init_subclass__(cls, **kwargs: dict) -> None:
        """Register the subclass singleton and its aliases on definition."""
        super().__init_subclass__(**kwargs)

        if not getattr(cls, "name", None):
            raise ValueError(f"{cls.__name__} must define a 'name' class attribute")

        for alias in (cls.name, *cls.aliases):
            Language.LOOKUP[alias] = cls.name

        # Matrix brackets default to the 1-D brackets + the language separator.
        cls.lb, cls.rb = cls.brac[0], cls.brac[1]
        cls.mlb, cls.mrb = cls.lb, cls.rb
        cls.sep = cls.matrix_sep

        Language._register[cls.name] = cls()

    def derive(
        self,
        *,
        brac_format: str = "",
        matrix_format: str = "",
        assignment_op: str = "",
        line_end: str = "",
    ) -> "Language":
        """Return a bracket/token-overridden view of this language.

        When no override is supplied, returns ``self`` (the registered
        singleton) unchanged.  Otherwise returns a shallow, *unregistered*
        copy with the overridden attributes shadowing the class defaults, so
        the singleton is never mutated and the registry never grows.

        Parameters
        ----------
        brac_format : str, optional
            1-D bracket style from :attr:`BRACKET_FORMATS` (e.g. ``"()"``).
        matrix_format : str, optional
            2-D bracket/separator format key from :attr:`MATRIX_FORMATS`.
        assignment_op : str, optional
            Assignment operator override.
        line_end : str, optional
            Statement terminator override.

        Returns
        -------
        Language
            ``self`` if nothing was overridden, else an ephemeral copy.

        Raises
        ------
        InvalidLanguageError
            If *brac_format* or *matrix_format* is not a supported format.
        """
        if not (brac_format or matrix_format or assignment_op or line_end):
            return self

        view = copy.copy(self)

        if brac_format:
            if brac_format not in self.BRACKET_FORMATS:
                raise InvalidLanguageError(
                    f"Unsupported bracket format: '{brac_format}'. "
                    f"Supported: {', '.join(self.BRACKET_FORMATS)}"
                )
            view.lb, view.rb = brac_format[0], brac_format[1]

        if matrix_format:
            if matrix_format not in self.MATRIX_FORMATS:
                raise InvalidLanguageError(
                    f"Unsupported matrix format: '{matrix_format}'. "
                    f"Supported: {', '.join(self.MATRIX_FORMATS)}"
                )
            fmt = self.MATRIX_FORMATS[matrix_format]
            view.mlb, view.mrb = fmt["brac"][0], fmt["brac"][1]
            view.sep = fmt["sep"]

        if assignment_op:
            view.assignment_op = assignment_op
        if line_end:
            view.line_end = line_end

        return view

    @classmethod
    def registered(cls) -> tuple["Language", ...]:
        """Return every registered language singleton, one per canonical name."""
        return tuple(Language._register.values())

    @classmethod
    def comments(cls) -> set[str]:
        """Return the set of single-line comment prefixes across all languages."""
        return {lang.comment for lang in Language._register.values()}

    def __new__(cls, lang: str | None = None) -> "Language":
        # Subclasses instantiate normally; only Language(...) acts as a factory.
        if cls is not Language:
            return super().__new__(cls)

        if lang not in cls.LOOKUP:
            supported = ", ".join(sorted(set(cls.LOOKUP.values())))
            raise InvalidLanguageError(
                f"{lang} is not a supported language.\n"
                f"Supported languages are: {supported}"
            )

        return cls._register[cls.LOOKUP[lang]]

    def __repr__(self) -> str:
        return f"<Language {self.name}>"


class Cxx(Language):
    name = "cxx"
    aliases = ("c++", "cpp")
    brac = "[]"
    assignment_op = "="
    line_end = ";"
    matrix_sep = "]["
    code_gen = staticmethod(sp.cxxcode)
    idx_offset = 0
    comment = "//"
    types = {"int": "int ", "float": "float ", "double": "double ", "bool": "bool "}
    extras = {"type_qualifier": "const ", "class_specifier": "static "}


class C(Language):
    name = "c"
    brac = "[]"
    assignment_op = "="
    line_end = ";"
    matrix_sep = "]["
    code_gen = staticmethod(sp.ccode)
    idx_offset = 0
    comment = "//"
    types = {"int": "int ", "float": "float ", "double": "double ", "bool": "_Bool "}
    extras = {"type_qualifier": "const ", "class_specifier": "static "}


class Fortran(Language):
    name = "fortran"
    aliases = ("f90",)
    brac = "()"
    assignment_op = "="
    line_end = ""
    matrix_sep = ", "
    code_gen = staticmethod(_fortran_code_gen)
    idx_offset = 1
    comment = "!"
    types: ClassVar[dict[str, str]] = {}
    extras = {"class_specifier": "save "}


class Python(Language):
    name = "python"
    aliases = ("py",)
    brac = "[]"
    assignment_op = "="
    line_end = ""
    matrix_sep = "]["
    code_gen = staticmethod(sp.pycode)
    idx_offset = 0
    comment = "#"
    types: ClassVar[dict[str, str]] = {}
    extras: ClassVar[dict[str, Any]] = {}


class Rust(Language):
    name = "rust"
    aliases = ("rs",)
    brac = "[]"
    assignment_op = "="
    line_end = ";"
    matrix_sep = "]["
    code_gen = staticmethod(sp.rust_code)
    idx_offset = 0
    comment = "//"
    types = {"int": "i32 ", "float": "f32 ", "double": "f64 ", "bool": "bool "}
    extras = {"type_qualifier": "const ", "class_specifier": ""}


class Julia(Language):
    name = "julia"
    aliases = ("jl",)
    brac = "[]"
    assignment_op = "="
    line_end = ""
    matrix_sep = ", "
    code_gen = staticmethod(sp.julia_code)
    idx_offset = 1
    comment = "#"
    types = {
        "int": "Int64 ",
        "float": "Float32 ",
        "double": "Float64 ",
        "bool": "Bool ",
    }
    extras = {"type_qualifier": "const ", "class_specifier": ""}


class R(Language):
    name = "r"
    brac = "[]"
    assignment_op = "<-"
    line_end = ""
    matrix_sep = ", "
    code_gen = staticmethod(sp.rcode)
    idx_offset = 1
    comment = "#"
    types: ClassVar[dict[str, str]] = {}
    extras: ClassVar[dict[str, Any]] = {}


def scoped_tokens(lang_attr: str = "lang") -> Callable[[Callable], Callable]:
    """Decorator: apply a method's bracket/token overrides to its owner's language.

    Wraps a code-generation method whose owner exposes a :class:`Language` on
    the attribute named *lang_attr*.  Before the call, the override keyword
    arguments the method declares (``brac_format``, ``matrix_format``,
    ``assignment_op``, ``line_end``) are harvested and used to swap the owner's
    language for a :meth:`Language.derive`-d view; the original language is
    restored afterwards.  The wrapped method body reads the tokens straight
    off ``self.<lang_attr>`` (e.g. ``self.lang.lb``) with no per-method
    fallback boilerplate.

    Methods that do not declare a given override keyword are unaffected — the
    harvester falls back to an empty override for absent parameters.

    Parameters
    ----------
    lang_attr : str, optional
        Name of the attribute on the wrapped method's owner that holds the
        :class:`Language` instance.  Default ``"lang"``.

    Returns
    -------
    Callable
        A decorator that wraps a code-generation method.
    """

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            supplied = bound.arguments

            lang: Language = getattr(self, lang_attr)
            original = lang
            setattr(
                self,
                lang_attr,
                lang.derive(
                    brac_format=supplied.get("brac_format", ""),
                    matrix_format=supplied.get("matrix_format", ""),
                    assignment_op=supplied.get("assignment_op", ""),
                    line_end=supplied.get("line_end", ""),
                ),
            )
            try:
                return func(self, *args, **kwargs)
            finally:
                setattr(self, lang_attr, original)

        return wrapper

    return decorator
