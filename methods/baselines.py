"""
Simple baselines, batched, sharing the common retrieval/corpus/metric:
  cot        : Chain-of-Thought, no retrieval
  ragcot     : single-step RAG (top-5 on the question) + CoT
  iterretgen : Iter-RetGen — alternate generate/retrieve for T_max rounds
  genground  : GenGround — generate sub-question+answer then ground in docs
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import LLMEngine, Retriever, docs_to_str  # noqa: E402

_SHORT = ("end with a final line exactly: 'Answer: <X>' where X is the answer ONLY — a single "
          "entity / name / place / date / number, NOT a sentence and not a restatement. "
          "If the question is yes/no, X must be exactly 'yes' or 'no'.")
COT_SYS = "Answer the multi-hop question. Think step by step briefly, then " + _SHORT
RAG_SYS = ("Answer the multi-hop question using the documents and your knowledge. Think step "
           "by step briefly, then " + _SHORT)


def _parse_answer(t: str) -> str:
    m = list(re.finditer(r"Answer:\s*(.*)", t))
    if m:
        return m[-1].group(1).strip().rstrip(".").strip()
    return t.strip().split("\n")[-1][:128].strip()


def _cot(engine, questions, docs_list=None, temperature=1.0):
    msgs = []
    for i, q in enumerate(questions):
        if docs_list is not None:
            user = f"Documents:\n{docs_list[i]}\nQuestion: {q}"
            sys_p = RAG_SYS
        else:
            user = f"Question: {q}"
            sys_p = COT_SYS
        msgs.append([{"role": "system", "content": sys_p}, {"role": "user", "content": user}])
    return engine.chat(msgs, temperature=temperature, max_tokens=512)


def run_baseline(method, engine: LLMEngine, retriever, rows, top_k=5, max_iter=5,
                 temperature=1.0):
    questions = [r["question"] for r in rows]
    n = len(rows)

    if method == "cot":
        outs = _cot(engine, questions, None, temperature)
        preds = [_parse_answer(o) for o in outs]

    elif method == "ragcot":
        docs = retriever.batch_retrieve(questions, top_k=top_k)
        dstr = [docs_to_str(d) for d in docs]
        outs = _cot(engine, questions, dstr, temperature)
        preds = [_parse_answer(o) for o in outs]

    elif method == "iterretgen":
        gen = _cot(engine, questions, None, temperature)  # round 1: no docs
        for _ in range(max_iter - 1):
            qq = [f"{questions[i]} {_parse_answer(gen[i])}" for i in range(n)]
            docs = retriever.batch_retrieve(qq, top_k=top_k)
            dstr = [docs_to_str(d) for d in docs]
            gen = _cot(engine, questions, dstr, temperature)
        preds = [_parse_answer(o) for o in gen]

    elif method == "genground":
        accumulated = ["" for _ in range(n)]
        GEN_SYS = ("Decompose progress on the question. Given the question and notes so far, "
                   "produce ONE next simple sub-question and a tentative answer as: "
                   "'Subquestion: <q>\\nTentative: <a>'. If the question can now be answered, "
                   "output 'Subquestion: FINAL\\nTentative: <final answer>'.")
        GROUND_SYS = ("Given documents and a sub-question, give the correct short answer as "
                      "'Answer: <a>' grounded in the documents.")
        for _ in range(max_iter):
            gmsgs = [[{"role": "system", "content": GEN_SYS},
                      {"role": "user", "content": f"Question: {questions[i]}\nNotes:\n{accumulated[i]}"}]
                     for i in range(n)]
            gouts = engine.chat(gmsgs, temperature=temperature, max_tokens=256)
            subqs, finals = [], {}
            for i, go in enumerate(gouts):
                sm = re.search(r"Subquestion:\s*(.*)", go)
                tm = re.search(r"Tentative:\s*(.*)", go)
                sq = sm.group(1).strip() if sm else ""
                tv = tm.group(1).strip() if tm else ""
                if sq.upper().startswith("FINAL"):
                    finals[i] = tv
                    subqs.append(None)
                else:
                    subqs.append((i, sq, tv))
            todo = [s for s in subqs if s]
            if todo:
                docs = retriever.batch_retrieve([s[1] for s in todo], top_k=top_k)
                grmsgs = [[{"role": "system", "content": GROUND_SYS},
                           {"role": "user", "content": f"Documents:\n{docs_to_str(d)}\nSub-question: {s[1]}"}]
                          for s, d in zip(todo, docs)]
                grouts = engine.chat(grmsgs, temperature=temperature, max_tokens=128)
                for s, gr in zip(todo, grouts):
                    i = s[0]
                    accumulated[i] += f"Q: {s[1]} A: {_parse_answer(gr)}\n"
        # final answer
        fmsgs = [[{"role": "system", "content": RAG_SYS},
                  {"role": "user", "content": f"Notes:\n{accumulated[i]}\nQuestion: {questions[i]}"}]
                 for i in range(n)]
        fouts = engine.chat(fmsgs, temperature=temperature, max_tokens=256)
        preds = [_parse_answer(o) for o in fouts]
    else:
        raise ValueError(method)

    return [{
        "id": r["id"], "question": r["question"], "label": r["answer"],
        "answer_aliases": r.get("answer_aliases", []), "hop": r.get("hop"),
        "pred": preds[i] if preds[i] is not None else "",
    } for i, r in enumerate(rows)]
