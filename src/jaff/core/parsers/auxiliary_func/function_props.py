# ABOUTME: One parsed ``.jfunc`` auxiliary function: its expression, arguments,
# ABOUTME: and argument doc comments

from typing import Dict, List, Optional

from sympy import Basic, Symbol


class AuxiliaryFunctionProps:
    """A single auxiliary function parsed from a ``.jfunc`` file.

    Attributes
    ----------
    expr : Basic
        Fully expanded SymPy body of the function (``0.0`` until its
        ``@return`` line is parsed).
    args : list[Symbol]
        Declared arguments, in order.
    arg_comments : dict[str, str]
        Argument documentation, keyed by argument name.
    """

    def __init__(
        self,
        expr: Basic,
        args: List[Symbol],
        arg_comments: Optional[Dict[str, str]] = None,
    ):
        """Store the function data; ``arg_comments=None`` starts empty."""
        self.expr: Basic = expr
        self.args: List[Symbol] = args
        self.arg_comments: Dict[str, str] = (
            arg_comments if arg_comments is not None else {}
        )
        self._locals: Dict[str, Basic] = {}
