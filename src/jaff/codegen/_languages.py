from __future__ import annotations

import copy
import functools
import inspect
from collections.abc import Callable
from typing import Any, ClassVar

import sympy as sp

from ..errors import InvalidLanguageError


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
    code_gen = staticmethod(sp.fcode)
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
