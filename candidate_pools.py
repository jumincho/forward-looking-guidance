"""Share candidate documents within deterministic groups of questions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def group_candidates(rows: list[dict], group_size: int) -> list[dict]:
    if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 1:
        raise ValueError("group_size must be a positive integer")
    if not rows:
        raise ValueError("Candidate pools are empty")
    by_id = {}
    for row in rows:
        if set(row) != {"id", "document_ids"}:
            raise ValueError("Each candidate pool must contain id and document_ids")
        qid, documents = row["id"], row["document_ids"]
        if not isinstance(qid, str) or not qid or qid in by_id:
            raise ValueError("Question IDs must be nonempty unique strings")
        if (not isinstance(documents, list) or not documents
                or any(not isinstance(d, str) or not d for d in documents)
                or len(documents) != len(set(documents))):
            raise ValueError("Document IDs must be nonempty unique strings")
        by_id[qid] = documents
    ids = sorted(by_id, key=lambda qid: hashlib.sha256(qid.encode()).hexdigest())
    grouped = {}
    for offset in range(0, len(ids), group_size):
        members = ids[offset:offset + group_size]
        documents = sorted({d for qid in members for d in by_id[qid]})
        for qid in members:
            grouped[qid] = documents
    return [{"id": row["id"], "document_ids": list(grouped[row["id"]])} for row in rows]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--group-size", required=True, type=int)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()]
    grouped = group_candidates(rows, args.group_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                         prefix=".candidate-pools-", delete=False) as stream:
            temporary = Path(stream.name)
            for row in grouped:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
