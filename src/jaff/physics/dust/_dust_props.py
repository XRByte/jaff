from dataclasses import dataclass


@dataclass
class DustProps:
    rv: float = 3.1
    u_reduction: str | None = "absorption"
    f_reduction: str | None = "transport"
