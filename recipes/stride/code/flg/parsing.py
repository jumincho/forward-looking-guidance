"""Parse model output as data, never as executable Python or with gold labels."""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any


class ParseError(ValueError):
    pass


def extracted_facts(text: str) -> str | None:
    """Retain extractor evidence, including structured or plain-text facts.

    Extractor output is passed as data to the reasoner. Formatting differences
    must not silently erase facts; no model-produced text is ever executed.
    """
    cleaned = text.strip()
    if cleaned.lower().strip("` .\n\t") in {"", "none", "null", "n/a"}:
        return None
    try:
        value = structured(cleaned)
    except ParseError:
        return cleaned
    if value is None or value == [] or value == {}:
        return None
    try:
        if isinstance(value, (list, tuple)):
            return "\n".join(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                             for v in value)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        # A safe Python literal can contain sets or non-JSON mapping keys.
        # Retain the original evidence as text instead of aborting the batch.
        return cleaned
    return str(value)


def structured(text: str) -> Any:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    candidates = [text]
    candidates.extend(re.findall(r"```(?:json|python)?\s*(.*?)```", text, flags=re.S))
    for candidate in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(candidate)
            except (ValueError, SyntaxError, TypeError, RecursionError):
                pass
    # Accept a complete JSON response followed by unwanted prose/examples.
    # Adjacent arrays are also a common fact-list format: keep every array
    # until non-JSON commentary begins, rather than discarding later evidence.
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for match in re.finditer(r"[\[{]", candidate):
            try:
                value, end = decoder.raw_decode(candidate[match.start():])
            except (ValueError, RecursionError):
                continue
            if isinstance(value, list):
                tail = candidate[match.start() + end:].lstrip()
                while tail.startswith("["):
                    try:
                        following, end = decoder.raw_decode(tail)
                    except (ValueError, RecursionError):
                        break
                    if not isinstance(following, list):
                        break
                    value.extend(following)
                    tail = tail[end:].lstrip()
            if isinstance(value, (list, dict)):
                return value
    for left, right in [("[", "]"), ("{", "}")]:
        if left in text and right in text:
            candidates.append(text[text.index(left):text.rindex(right) + 1])
    for candidate in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(candidate)
            except (ValueError, SyntaxError, TypeError, RecursionError):
                pass
    raise ParseError("No valid JSON/literal object in response")


def answer_span(text: str | None) -> str | None:
    """Same parser for every method; no reference answer is accepted."""
    if text is None:
        return None
    obj = structured(text)
    if not isinstance(obj, dict) or "answer" not in obj:
        raise ParseError("Missing answer field")
    value = obj["answer"]
    if value is None:
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        raise ParseError("Answer must be a scalar or list, not a mapping")
    value = str(value).strip()
    if value.lower() in {"none", "null", "unknown", "n/a", "not given", "not mentioned", ""}:
        return None
    if value.lower() in {"true", "false"}:
        return "yes" if value.lower() == "true" else "no"
    return value


@dataclass(frozen=True)
class Operation:
    qid: str
    text: str
    verbatim: str


def operations(plan: str) -> list[Operation]:
    """Retain the exact planner lines for guidance, including multi-line text."""
    concrete = re.split(r"Concrete\s+Plan\s*:", plan, flags=re.I)[-1]
    pattern = r"(?m)^\s*(?:\*\*)?(Q\d+)\s*:\s*(?:\*\*)?"
    matches = list(re.finditer(pattern, concrete))
    result = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(concrete)
        raw = concrete[m.start():end].strip().rstrip("`").rstrip()
        body = concrete[m.end():end].strip().rstrip("`").rstrip()
        result.append(Operation(m.group(1), body, raw))
    if not result or len({o.qid for o in result}) != len(result):
        raise ParseError("Missing or duplicate operation IDs in plan")
    return result


def actions(text: str, pending: set[str]) -> list[dict[str, str]]:
    value = structured(text)
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not value:
        raise ParseError("Supervisor must return a nonempty action list")
    result = []
    seen = set()
    for item in value:
        if not isinstance(item, dict):
            raise ParseError("Invalid action object")
        qid, action, query = item.get("qid"), item.get("action"), item.get("query")
        if qid not in pending or qid in seen or action not in {"retrieve", "rewrite", "answer"}:
            raise ParseError("Invalid, solved, or duplicate action ID/type")
        if not isinstance(query, str) or not query.strip():
            raise ParseError("Action query must be nonempty")
        seen.add(qid)
        result.append({"qid": qid, "action": action, "query": query})
    return result
