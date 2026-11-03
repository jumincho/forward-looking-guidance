"""The FG intervention: verbatim remaining operations, without extra calls."""
from __future__ import annotations

from enum import Enum
from .parsing import Operation


class Stage(str, Enum):
    NONE = "none"
    RETRIEVAL = "retrieval"
    EXTRACTION = "extraction"
    REASONING = "reasoning"
    EXTRACTION_REASONING = "extraction+reasoning"
    ALL = "all"

    def includes(self, module: str) -> bool:
        return self == Stage.ALL or module in self.value.split("+")


def remaining(ops: list[Operation], current: str, solved: set[str]) -> str:
    index = next(i for i, o in enumerate(ops) if o.qid == current)
    return "\n".join(o.verbatim for o in ops[index + 1:] if o.qid not in solved)


def append_guidance(text: str, trajectory: str) -> str:
    if not trajectory:
        return text
    return text + (
        "\n\nRemaining planned operations (forward-looking guidance):\n"
        + trajectory
        + "\n\nResolve the current operation. Preserve grounded details, entity identities, "
        "relations, and qualifiers that these remaining operations will need. "
        "The planned operations are information needs, not evidence or known answers."
    )
