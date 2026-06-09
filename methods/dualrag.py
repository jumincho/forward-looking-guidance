"""
DualRAG: dual-process loop of Reasoning-augmented Querying (RaQ)
and Progressive Knowledge Aggregation (pKA), implemented in batched form sharing the
Contriever retrieval / corpus / metric, plus a +FG variant.

Per iteration (<= max_iter):
  1. Reason: given question + accumulated knowledge, decide the next sub-query to issue
     (or that the question is answerable -> stop).
  2. Retrieve top-k docs for the sub-query.
  3. Summarize (pKA): extract the relevant knowledge from the docs into the knowledge base.
Then a final answer is composed from the accumulated knowledge.

+FG: a decomposition plan is generated up front; the remaining planned sub-questions are
injected into the knowledge-summarization (evidence) step, so aggregated knowledge is
formed with later reasoning in mind.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import LLMEngine, Retriever, docs_to_str, parse_plan, fg_block, parse_obj  # noqa: E402
from methods.stride import meta_plan  # noqa: E402

REASON_SYS = (
    "You are the reasoning process of a multi-hop QA system. Given the question and the "
    "knowledge gathered so far, decide what to do next.\n"
    "Output a JSON object:\n"
    '{\n  "thought": "<brief reasoning about what is still missing>",\n'
    '  "need_retrieve": <true|false>,\n'
    '  "query": "<a single focused search query for the missing fact; '
    'incorporate known entities. empty if need_retrieve is false>"\n}\n'
    "Set need_retrieve to false only when the knowledge already suffices to answer."
)

KS_SYS = (
    "You extract and summarize only the knowledge relevant to making progress on the "
    "question from the retrieved documents. Output 1-3 short factual sentences grounded in "
    "the documents; if nothing is relevant, output 'None'."
)
KS_SYS_FG = KS_SYS + (
    " You are also given the remaining planned sub-questions that will reuse this knowledge; "
    "retain and fully disambiguate any fact (full names, places, dates) those steps will need."
)

ANSWER_SYS = (
    "Answer the question using the gathered knowledge and your reasoning. "
    "Output a JSON object: {\"analysis\": \"<brief>\", \"answer\": \"<the answer ONLY: a single "
    "entity / name / place / date / number — NOT a sentence, no explanation>\"}. "
    "If the question is yes/no, answer exactly \"yes\" or \"no\"."
)


def _parse_reason(t: str):
    try:
        s = re.findall(r"\{.*\}", t, re.DOTALL)[0]
        d = parse_obj(s)
        return bool(d.get("need_retrieve", True)), str(d.get("query", "") or "")
    except Exception:
        # default: keep retrieving with a generic query
        return True, ""


def _parse_answer(t: str):
    try:
        if "```" in t:
            s = re.findall(r"```json(.*?)```", t, re.DOTALL)[0]
        else:
            s = re.findall(r"\{.*\}", t, re.DOTALL)[0]
        s = re.sub(r"//.*", "", s)
        d = parse_obj(s)
        return str(d["answer"]).strip()
    except Exception:
        m = re.findall(r'answer["\']?\s*:\s*["\']?(.*?)["\']?\s*[}\n]', t, re.DOTALL)
        return m[0].strip() if m else t.strip().split("\n")[-1][:128]


class DState:
    __slots__ = ("idx", "question", "kb", "thoughts", "done", "steps", "plan", "answer")

    def __init__(self, idx, question, plan):
        self.idx = idx
        self.question = question
        self.kb: list[str] = []
        self.thoughts: list[str] = []
        self.done = False
        self.steps = 0
        self.plan = plan
        self.answer = None

    def kb_str(self):
        return "\n".join(f"- {k}" for k in self.kb) if self.kb else "(none yet)"


def run_dualrag(
    engine: LLMEngine,
    retriever: Retriever,
    rows: list[dict],
    top_k: int = 5,
    max_iter: int = 5,
    temperature: float = 1.0,
    fg: bool = False,
) -> list[dict]:
    questions = [r["question"] for r in rows]
    plans = None
    if fg:
        plans = [parse_plan(p) for p in meta_plan(engine, questions)]
    states = [DState(i, questions[i], plans[i] if plans else []) for i in range(len(rows))]

    for _ in range(max_iter):
        active = [st for st in states if not st.done]
        if not active:
            break
        # 1. Reason -> need_retrieve + query
        rmsgs = [[{"role": "system", "content": REASON_SYS},
                  {"role": "user", "content":
                   f"Question: {st.question}\n\nKnowledge so far:\n{st.kb_str()}"}]
                 for st in active]
        routs = engine.chat(rmsgs, temperature=temperature, max_tokens=256)

        retr = []  # (st, query)
        for st, ro in zip(active, routs):
            need, query = _parse_reason(ro)
            st.thoughts.append(ro)
            if not need or not query.strip():
                if st.kb:           # have knowledge -> stop and answer
                    st.done = True
                    continue
                query = st.question  # nothing yet -> retrieve with the question
            retr.append((st, query.strip()))

        # 2 + 3. retrieve + summarize (pKA)
        if retr:
            docs = retriever.batch_retrieve([q for _, q in retr], top_k=top_k)
            kmsgs = []
            for (st, q), d in zip(retr, docs):
                user = f"Question: {st.question}\nSub-query: {q}\n\nDocuments:\n{docs_to_str(d)}"
                if fg and st.plan:
                    rem = st.plan[min(st.steps + 1, len(st.plan)):]
                    user += fg_block(rem)
                kmsgs.append([{"role": "system", "content": KS_SYS_FG if fg else KS_SYS},
                              {"role": "user", "content": user}])
            kouts = engine.chat(kmsgs, temperature=temperature, max_tokens=256)
            for (st, q), ko in zip(retr, kouts):
                ko = ko.strip()
                if ko and ko.lower() != "none":
                    st.kb.append(ko.replace("\n", " "))
                st.steps += 1

    # final answer from accumulated knowledge
    amsgs = []
    for st in states:
        user = f"Question: {st.question}\n\nKnowledge:\n{st.kb_str()}"
        amsgs.append([{"role": "system", "content": ANSWER_SYS},
                      {"role": "user", "content": user}])
    aouts = engine.chat(amsgs, temperature=temperature, max_tokens=512)
    for st, ao in zip(states, aouts):
        st.answer = _parse_answer(ao)

    return [{
        "id": r["id"], "question": r["question"], "label": r["answer"],
        "answer_aliases": r.get("answer_aliases", []), "hop": r.get("hop"),
        "pred": st.answer if st.answer is not None else "", "iters": st.steps,
    } for st, r in zip(states, rows)]
