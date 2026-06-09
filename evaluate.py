"""Evaluation: EM / token-F1 / Precision / Recall using normalize_answer, with
optional alias-aware scoring (e.g. MuSiQue answer_aliases)."""
from __future__ import annotations

import sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import normalize_answer, f1_score  # type: ignore
import re as _re


def convert_boolean_answer(predict: str) -> str:
    """Map a verbose yes/no prediction to "yes"/"no"."""
    predict = predict.replace("true", "yes").replace("false", "no")
    predict = _re.sub(r"false|False", "no", predict)
    predict = _re.sub(r"true|True", "yes", predict)
    if "yes" in predict or "Yes," in predict:
        return "yes"
    if "no" in predict or "No," in predict:
        return "no"
    return predict


def _em(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(str(pred)) == normalize_answer(str(gold)) else 0.0


def score(rows: list[dict], use_aliases: bool = False) -> dict:
    n = len(rows)
    em = f1 = p = r = 0.0
    for row in rows:
        pred = str(row.get("pred", "") or "")
        golds = [str(row.get("label", ""))]
        if use_aliases:
            golds += [str(a) for a in (row.get("answer_aliases") or [])]
        golds = [g for g in golds if g.strip()] or [str(row.get("label", ""))]
        # boolean post-processing for yes/no questions
        if str(row.get("label", "")).strip().lower() in ("yes", "no"):
            pred = convert_boolean_answer(pred)
        em += max(_em(pred, g) for g in golds)
        best = max((f1_score(pred, g) for g in golds), key=lambda x: x[0])
        f1 += best[0]; p += best[1]; r += best[2]
    return {"EM": em / n, "F1": f1 / n, "P": p / n, "R": r / n, "n": n}


def score_by_hop(rows: list[dict], use_aliases: bool = False) -> dict:
    out = {}
    for hop in sorted({row.get("hop") for row in rows if row.get("hop") is not None}):
        sub = [row for row in rows if row.get("hop") == hop]
        out[hop] = score(sub, use_aliases)
    out["overall"] = score(rows, use_aliases)
    return out


def fmt(m: dict) -> str:
    return f"EM={m['EM']:.4f} F1={m['F1']:.4f} P={m['P']:.4f} R={m['R']:.4f} n={m['n']}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--aliases", action="store_true")
    ap.add_argument("--by_hop", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    if a.by_hop:
        for k, v in score_by_hop(rows, a.aliases).items():
            print(f"  {k}: {fmt(v)}")
    else:
        print(fmt(score(rows, a.aliases)))
