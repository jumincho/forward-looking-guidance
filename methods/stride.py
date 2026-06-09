"""
STRIDE pipeline (Meta-Planner -> Supervisor -> Extractor + Reasoner + Fallback Reasoner),
implemented in a batched lock-step form for throughput, with Forward-Looking Guidance (FG)
injection points.

FG flags:
  fg_extract  : inject the remaining planned sub-questions G_i into the Extractor input
  fg_reason   : inject G_i into the Reasoner input
  fg_retrieve : inject G_i into the retrieval query — a per-stage ablation control only;
                guiding retrieval is not part of the method (STRIDE+FG = extract + reason)

STRIDE     = all flags False
STRIDE+FG  = fg_extract=True, fg_reason=True   (guidance on extraction + reasoning)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    LLMEngine, Retriever, docs_to_str, parse_plan, remaining_subquestions, fg_block, parse_obj,
)

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
META_PROMPT = (PROMPT_DIR / "meta_plan" / "meta_plan.txt").read_text(encoding="utf-8")
SUP_PROMPT = (PROMPT_DIR / "supervisor" / "default.txt").read_text(encoding="utf-8")
EXT_PROMPT = (PROMPT_DIR / "extractor" / "default.txt").read_text(encoding="utf-8")

# The Reasoner is asked for a minimal answer span (Qwen3-8B otherwise tends to answer in full
# sentences), with explicit yes/no handling for comparison sub-questions.
REA_PROMPT = (
    "Based on the provided facts and your own knowledge, answer the question.\n"
    "First reason briefly, then output a JSON object:\n"
    "```json\n{\n  \"analysis\": \"<one or two short sentences of reasoning>\",\n"
    "  \"answer\": \"<the answer ONLY: a single entity / name / place / date / number — "
    "NOT a sentence, no explanation, do not restate the question>\"\n}\n```\n"
    "If the question is a yes/no question (e.g. a comparison such as 'are both ...', "
    "'is X the same as Y', 'which ... '), answer exactly \"yes\" or \"no\".\n"
    "If the facts are insufficient, set answer to \"unknown\"."
)

# The Fallback Reasoner answers the original question directly when the structured run does not
# resolve it, using accumulated facts + retrieved documents + parametric knowledge.
FALLBACK_SYS = (
    "Answer the original multi-hop question using the documents, the known facts, and your "
    "own knowledge. Reason briefly, then output a JSON object:\n"
    "```json\n{\n  \"analysis\": \"<one or two short sentences>\",\n"
    "  \"answer\": \"<the final answer ONLY: a single entity / name / place / date / number — "
    "NOT a sentence>\"\n}\n```\n"
    "If the question is a yes/no question, answer exactly \"yes\" or \"no\"."
)

FG_EXT_SUFFIX = (
    "\n\n### Forward-Looking Guidance\n"
    "You are also given the *upcoming planned sub-questions* that will reuse the facts you "
    "extract now. In addition to facts that answer the current question, you MUST also keep "
    "every atomic fact (full names, dates, places, identifiers, qualifiers) that those upcoming "
    "sub-questions will need — especially the distinguishing attributes of any entity that a "
    "later step will search for (e.g. keep 'Springfield is in Illinois' so the next hop can "
    "find the right Springfield). Keep every fact fully specified and disambiguated."
)


# ---------------------------------------------------------------------------
def extract_plan_text(meta_out: str) -> str:
    try:
        s = re.findall(r"(Concrete Plan:.*)", meta_out, re.DOTALL)[0].strip()
        return s.replace("Concrete Plan:", "Plan:")
    except IndexError:
        return meta_out.strip()


def meta_plan(engine: LLMEngine, questions: list[str]) -> list[str]:
    msgs = [
        [{"role": "system", "content": META_PROMPT},
         {"role": "user", "content": f"Question: \n{q}"}]
        for q in questions
    ]
    outs = engine.chat(msgs, temperature=1.0, max_tokens=512)
    return [extract_plan_text(o) for o in outs]


def _parse_supervisor(text: str):
    try:
        s = re.findall(r"```json(.*?)```", text, re.DOTALL)[0].strip()
    except IndexError:
        m = re.findall(r"(\[.*\])", text, re.DOTALL)
        if not m:
            return None
        s = m[0]
    s = re.sub(r"//.*", "", s)
    try:
        obj = parse_obj(s)
    except Exception:
        return None
    if not isinstance(obj, list):
        return None
    return obj


def _parse_reasoner(text: str):
    try:
        if len(re.findall(r"```", text, re.DOTALL)) < 2:
            ans = re.findall(r"({.*})", text, re.DOTALL)[0].strip()
        else:
            ans = re.findall(r"```json(.*?)```", text, re.DOTALL)[0].strip()
        ans = re.sub(r"//.*", "", ans)
        d = parse_obj(ans)
        return str(d["answer"]).strip()
    except Exception:
        # fallback: try to grab text after "answer"
        m = re.findall(r'answer["\']?\s*:\s*["\']?(.*?)["\']?\s*[}\n]', text, re.DOTALL)
        if m:
            return m[0].strip()
        return None


NONE_SET = {"none", "n/a", "not mentioned", "not given", "unknown", "", "null", "no answer"}


def _is_none(ans) -> bool:
    if ans is None:
        return True
    if isinstance(ans, str) and ans.strip().lower() in NONE_SET:
        return True
    return False


class QState:
    __slots__ = ("idx", "question", "plan", "solved", "pending", "failure",
                 "facts", "final", "iters", "done", "max_qid")

    def __init__(self, idx, question, plan_text):
        self.idx = idx
        self.question = question
        self.plan = parse_plan(plan_text)
        self.solved: dict[str, str] = {}
        self.pending: list[str] = [qid for qid, _ in self.plan]
        self.failure: dict[str, list] = {}
        self.facts: dict[str, tuple] = {}
        self.final = None
        self.iters = 0
        self.done = False
        self.max_qid = self.plan[-1][0] if self.plan else "Q1"

    def progress_str(self):
        return (f"Solved: {self.solved}\nPending: [" + ", ".join(self.pending) +
                "]\nFailureLog: " + str(self.failure))

    def plan_text(self):
        return "Plan: \n" + "\n".join(f"{qid}: {txt}" for qid, txt in self.plan)

    def sup_input(self):
        return (f"Question: {self.question}\n\n{self.plan_text()}\n\n"
                f"Progress: \n{self.progress_str()}")


def run_stride(
    engine: LLMEngine,
    retriever: Retriever,
    rows: list[dict],
    top_k: int = 3,
    max_iter: int = 5,
    failed_threshold: int = 2,
    fallback_k: int = 5,
    fg_extract: bool = False,
    fg_reason: bool = False,
    fg_retrieve: bool = False,
    temperature: float = 1.0,
    plans: list[str] | None = None,
) -> list[dict]:
    questions = [r["question"] for r in rows]
    if plans is None:
        plans = meta_plan(engine, questions)

    ext_sys = EXT_PROMPT + (FG_EXT_SUFFIX if fg_extract else "")

    states = [QState(i, questions[i], plans[i]) for i in range(len(rows))]
    for st in states:
        if not st.plan:
            st.done = True

    for _ in range(max_iter):
        active = [st for st in states if not st.done]
        if not active:
            break

        # ---- 1. Supervisor (batched) ----
        sup_msgs = [[{"role": "system", "content": SUP_PROMPT},
                     {"role": "user", "content": st.sup_input()}] for st in active]
        sup_outs = engine.chat(sup_msgs, temperature=temperature, max_tokens=512)

        # gather extractor + answer tasks
        ext_tasks = []   # (st, qid, query)
        ans_tasks = []   # (st, qid, query)
        for st, out in zip(active, sup_outs):
            st.iters += 1
            parsed = _parse_supervisor(out)
            if not parsed:
                continue
            for s_ in parsed:
                if not isinstance(s_, dict):
                    continue
                qid = s_.get("qid"); action = s_.get("action"); query = s_.get("query")
                if not qid or not action or query is None:
                    continue
                if qid in st.failure and len(st.failure[qid]) >= failed_threshold:
                    st.done = True
                    break
                if action in ("retrieve", "rewrite"):
                    ext_tasks.append((st, qid, str(query)))
                elif action == "answer":
                    ans_tasks.append((st, qid, str(query)))

        # ---- 2. Retrieval + Extractor (batched) ----
        if ext_tasks:
            queries = []
            for st, qid, query in ext_tasks:
                rq = query
                if fg_retrieve:
                    rem = remaining_subquestions(st.plan, qid)
                    if rem:
                        rq = query + " " + " ".join(t for _, t in rem)
                queries.append(rq)
            retr = retriever.batch_retrieve(queries, top_k=top_k)
            ext_msgs = []
            for (st, qid, query), res in zip(ext_tasks, retr):
                doc_str = docs_to_str(res)
                user = f"Question: \n{query}\n\nDocuments: \n{doc_str}"
                if fg_extract:
                    user += fg_block(remaining_subquestions(st.plan, qid))
                ext_msgs.append([{"role": "system", "content": ext_sys},
                                 {"role": "user", "content": user}])
            ext_outs = engine.chat(ext_msgs, temperature=temperature, max_tokens=512)
            for (st, qid, query), eo in zip(ext_tasks, ext_outs):
                fact = eo.strip().replace("\n", " ")
                if (not fact) or fact == "None" or fact.lower().startswith("none"):
                    st.failure.setdefault(qid, []).append(query)
                    if len(st.failure[qid]) >= failed_threshold:
                        st.done = True
                    continue
                st.facts[qid] = (query, fact)

        # ---- 3. Reasoner (batched) ----
        rea_tasks = []  # (st, qid, query, facts_str)
        for st, qid, query in ext_tasks:
            if qid in st.facts:
                rea_tasks.append((st, qid, query, st.facts[qid][1]))
        for st, qid, query in ans_tasks:
            facts_str = "\n".join(v[1] for v in st.facts.values()).strip()
            rea_tasks.append((st, qid, query, facts_str))

        if rea_tasks:
            rea_msgs = []
            for st, qid, query, facts_str in rea_tasks:
                is_final = (qid == st.max_qid)
                note = (fg_block(remaining_subquestions(st.plan, qid), role="reason")
                        if (fg_reason and not is_final) else "")
                if note:
                    # +FG, non-final: complete answer + disambiguation guidance for the next hop
                    user = (f"Facts: \n{facts_str}\n\n{note}\n\n"
                            f"Now answer ONLY this current question:\nQuestion: \n{query}")
                else:
                    user = f"Facts: \n{facts_str}\n\nQuestion: \n{query}"
                rea_msgs.append([{"role": "system", "content": REA_PROMPT},
                                 {"role": "user", "content": user}])
            rea_outs = engine.chat(rea_msgs, temperature=temperature, max_tokens=512)
            for (st, qid, query, facts_str), ro in zip(rea_tasks, rea_outs):
                ans = _parse_reasoner(ro)
                if _is_none(ans):
                    st.failure.setdefault(qid, []).append(query)
                    if len(st.failure[qid]) >= failed_threshold:
                        st.done = True
                    continue
                st.solved[qid] = ans
                if qid in st.pending:
                    st.pending.remove(qid)
                st.failure.pop(qid, None)
                if qid == st.max_qid or len(st.pending) == 0:
                    st.final = ans
                    st.done = True

    # Fallback Reasoner: for questions the structured run did not resolve, answer the ORIGINAL
    # question directly using retrieved docs + accumulated facts + parametric knowledge. This
    # avoids defaulting to a wrong intermediate sub-answer.
    fb = [st for st in states if st.final is None]
    if fb and retriever is not None:
        fdocs = retriever.batch_retrieve([st.question for st in fb], top_k=fallback_k)
        fmsgs = []
        for st, d in zip(fb, fdocs):
            facts = "\n".join(v[1] for v in st.facts.values()).strip()
            user = (f"Documents:\n{docs_to_str(d)}\n"
                    f"Known facts:\n{facts if facts else '(none)'}\n\n"
                    f"Question: {st.question}")
            fmsgs.append([{"role": "system", "content": FALLBACK_SYS},
                          {"role": "user", "content": user}])
        fouts = engine.chat(fmsgs, temperature=temperature, max_tokens=512)
        for st, o in zip(fb, fouts):
            a = _parse_reasoner(o)
            if not _is_none(a):
                st.final = a

    # final answers: prefer st.final; else the last solved sub-question's answer
    out_rows = []
    for st, r in zip(states, rows):
        pred = st.final
        if pred is None and st.solved:
            # answer of last solved sub-question in plan order
            for qid, _ in reversed(st.plan):
                if qid in st.solved:
                    pred = st.solved[qid]
                    break
        out_rows.append({
            "id": r["id"],
            "question": r["question"],
            "label": r["answer"],
            "answer_aliases": r.get("answer_aliases", []),
            "hop": r.get("hop"),
            "pred": pred if pred is not None else "",
            "iters": st.iters,
            "plan": st.plan_text(),
            "solved": st.solved,
        })
    return out_rows
