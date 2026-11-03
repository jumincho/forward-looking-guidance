"""Macro answer EM / token F1 / precision / recall. Inference never reads labels."""
from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from .io import digest


def normalize_answer(text: str) -> str:
    text = str(text).lower().strip()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"'|`|,|’|\\|´", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def score(prediction: str | None, gold: str) -> dict[str, float]:
    if prediction is None:
        return dict(em=0.0, f1=0.0, precision=0.0, recall=0.0)
    p, g = normalize_answer(prediction), normalize_answer(gold)
    em = float(p == g)
    if p in {"yes", "no", "noanswer"} or g in {"yes", "no", "noanswer"}:
        return dict(em=em, f1=em, precision=em, recall=em)
    pt, gt = p.split(), g.split()
    if not pt or not gt:
        return dict(em=em, f1=em, precision=em, recall=em)
    n = sum((Counter(pt) & Counter(gt)).values())
    precision, recall = n / len(pt), n / len(gt)
    f1 = 2 * precision * recall / (precision + recall) if n else 0.0
    return dict(em=em, f1=f1, precision=precision, recall=recall)


def evaluate(predictions: list[dict], labels: list[dict], *, allow_partial: bool = False) -> dict:
    pids, gids = [str(x["id"]) for x in predictions], [str(x["id"]) for x in labels]
    if len(set(pids)) != len(pids) or len(set(gids)) != len(gids):
        raise ValueError("Duplicate prediction or label IDs")
    if set(pids) - set(gids):
        raise ValueError("Predictions contain IDs outside this split")
    if not allow_partial and set(pids) != set(gids):
        raise ValueError("Incomplete predictions; use --allow-partial only for a labelled partial report")
    if not predictions:
        raise ValueError("No predictions")
    by_id = {str(x["id"]): x for x in predictions}
    groups = defaultdict(list)
    statuses = Counter()
    for label in labels:
        qid = str(label["id"])
        if qid not in by_id:
            continue
        pred = by_id[qid]
        status = pred.get("status", "ok")
        statuses[status] += 1
        values = score(pred.get("answer"), label["answer"])
        groups["overall"].append(values)
        if label.get("hops"):
            groups[f'{label["hops"]}-hop'].append(values)
    def aggregate(rows):
        return {"n": len(rows), **{k: sum(v[k] for v in rows) / len(rows) for k in rows[0]}}
    return {
        "evaluation": "measured", "complete": set(pids) == set(gids),
        "expected_n": len(labels), "prediction_ids_sha256": digest(sorted(pids)),
        "statuses": dict(statuses), **{k: aggregate(v) for k, v in groups.items()},
    }
