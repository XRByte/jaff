from typing import Dict, List

from sympy import Expr, Symbol


class AuxiliaryFunctionProps:
    def __init__(
        self,
        expr: Expr,
        args: List[Symbol],
        arg_comments: Dict[str, str],
        locals: Dict[str, Expr] = {},
    ):
        self.expr: Expr = expr
        self.args: List[Symbol] = args
        self.arg_comments: Dict[str, str] = arg_comments
        self.locals: Dict[str, Expr] = locals
