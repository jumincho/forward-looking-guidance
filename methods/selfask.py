"""
Self-Ask as an iterative-RAG baseline with Contriever retrieval,
batched lock-step over questions, plus a +FG variant.

Baseline: standard self-ask scaffolding; each follow-up question is answered from
top-k retrieved documents (using the configured dense retriever).

+FG: a decomposition plan of the question is generated up front (same Meta-Planner prompt),
and when producing each intermediate answer from evidence the remaining planned sub-questions
are injected as forward-looking guidance, so intermediate answers are formed disambiguated for
later steps.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import LLMEngine, Retriever, docs_to_str, parse_plan, fg_block  # noqa: E402
from methods.stride import meta_plan  # noqa: E402

FEWSHOT = """Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali

Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952

Question: Who was the maternal grandfather of George Washington?
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball

Question: Are both the directors of Jaws and Casino Royale from the same country?
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate answer: New Zealand.
So the final answer is: No

"""

FOLLOWUP = "Follow up:"
INTERMED = "\nIntermediate answer:"
FINAL = "So the final answer is:"

FINAL_SYS = (
    "Answer the question concisely using the evidence and your own knowledge. "
    "Output only the short answer — a single entity / name / place / date / number; "
    "if the question is yes/no, answer exactly 'yes' or 'no'."
)

ANS_SYS = (
    "You answer a single factual follow-up question. Use the provided documents first, and "
    "your own knowledge to fill gaps. Always commit to a concrete best-guess answer (never say "
    "'not in the documents' or 'I don't know'). Reply with a short, self-contained intermediate "
    "answer (one sentence), no explanation."
)
ANS_SYS_FG = ANS_SYS + (
    " You are also given upcoming planned sub-questions that will reuse this answer; make the "
    "answer fully disambiguated (include full names, places, dates qualifiers) so later steps can use it."
)


def _last_line(t: str) -> str:
    return t.split("\n")[-1] if "\n" in t else t


def _extract_followup(t: str) -> str | None:
    # find the last 'Follow up:' occurrence and take text after it up to newline
    m = list(re.finditer(r"Follow up:\s*(.*)", t))
    if not m:
        return None
    q = m[-1].group(1).strip()
    return q or None


def _extract_final(t: str) -> str | None:
    m = list(re.finditer(r"So the final answer is:\s*(.*)", t))
    if not m:
        return None
    return m[-1].group(1).strip().rstrip(".").strip() or None


class SAState:
    __slots__ = ("idx", "question", "prompt", "final", "done", "steps", "plan")

    def __init__(self, idx, question, plan):
        self.idx = idx
        self.question = question
        self.prompt = FEWSHOT + f"Question: {question}\nAre follow up questions needed here:"
        self.final = None
        self.done = False
        self.steps = 0
        self.plan = plan  # list[(qid,text)] or []


def run_selfask(
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
        plan_texts = meta_plan(engine, questions)
        plans = [parse_plan(p) for p in plan_texts]
    states = [SAState(i, questions[i], plans[i] if plans else []) for i in range(len(rows))]

    # initial decision + first follow-up (or final)
    for _ in range(max_iter + 1):
        active = [st for st in states if not st.done]
        if not active:
            break
        # generate continuation up to the next intermediate-answer slot
        cont = engine.complete(
            [st.prompt for st in active],
            temperature=temperature, max_tokens=256,
            stop=["\nIntermediate answer:", "Intermediate answer:"],
        )
        # parse: follow-up question or final answer
        ans_tasks = []  # (st, followup, text_appended)
        for st, out in zip(active, cont):
            st.prompt += out
            fin = _extract_final(out)
            fu = _extract_followup(out)
            # final answer present and no new follow-up after it
            if fin is not None and (fu is None or out.rfind(FINAL) > out.rfind(FOLLOWUP)):
                st.final = fin
                st.done = True
                continue
            if fu is not None:
                ans_tasks.append((st, fu))
            else:
                # no follow-up, no final -> force final
                st.prompt += "\n" + FINAL
                st.done = False  # will be finalized below
                ans_tasks.append((st, None))

        # answer follow-ups from retrieval
        retr_tasks = [(st, fu) for st, fu in ans_tasks if fu]
        if retr_tasks:
            docs = retriever.batch_retrieve([fu for _, fu in retr_tasks], top_k=top_k)
            msgs = []
            for (st, fu), d in zip(retr_tasks, docs):
                user = f"Documents:\n{docs_to_str(d)}\nQuestion: {fu}"
                if fg and st.plan:
                    rem = st.plan[min(st.steps + 1, len(st.plan)):]
                    user += fg_block(rem)
                msgs.append([{"role": "system", "content": ANS_SYS_FG if fg else ANS_SYS},
                             {"role": "user", "content": user}])
            inter = engine.chat(msgs, temperature=temperature, max_tokens=128)
            for (st, fu), ia in zip(retr_tasks, inter):
                ia = ia.strip().split("\n")[0].strip()
                st.prompt += INTERMED + " " + ia
                st.steps += 1

        # questions that were forced to final (fu is None)
        forced = [st for st, fu in ans_tasks if not fu]
        if forced:
            fin_out = engine.complete(
                [st.prompt for st in forced], temperature=temperature,
                max_tokens=64, stop=["\n"],
            )
            for st, fo in zip(forced, fin_out):
                st.final = (fo.strip().rstrip(".").strip() or None)
                st.done = True

    # any not finished -> answer the original question from the gathered intermediate evidence
    # (a clean CoT over the trace, instead of continuing the rambling self-ask scaffold).
    unfinished = [st for st in states if st.final is None]
    if unfinished:
        msgs = []
        for st in unfinished:
            found = "\n".join(f"- {x.strip()}" for x in re.findall(r"Intermediate answer:\s*(.*)", st.prompt)) or "(none)"
            msgs.append([{"role": "system", "content": FINAL_SYS},
                         {"role": "user", "content": f"Question: {st.question}\n\nWhat we found:\n{found}"}])
        outs = engine.chat(msgs, temperature=temperature, max_tokens=64)
        for st, o in zip(unfinished, outs):
            st.final = o.strip().split("\n")[0].rstrip(".").strip()

    out_rows = []
    for st, r in zip(states, rows):
        out_rows.append({
            "id": r["id"], "question": r["question"], "label": r["answer"],
            "answer_aliases": r.get("answer_aliases", []), "hop": r.get("hop"),
            "pred": st.final if st.final is not None else "", "iters": st.steps,
        })
    return out_rows
