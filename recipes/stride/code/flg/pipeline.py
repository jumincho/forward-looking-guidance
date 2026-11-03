"""STRIDE execution with a stage-local, call-free FG intervention.

Only IDs and questions enter the inference interface. Labels and supporting-fact
annotations live in separate evaluation files. Plans are generated once and shared
between the baseline and all stage ablations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib.resources import files
from typing import Any, Protocol
import re

from .backends import Backend, Request
from .guidance import Stage, append_guidance, remaining
from .io import digest
from .parsing import ParseError, actions, answer_span, extracted_facts, operations, structured


class Retriever(Protocol):
    def search(self, queries: list[str], top_k: int, *, question_ids=None) -> list[list[dict]]: ...


def prompt(name: str) -> str:
    return files("flg").joinpath("prompts", name + ".txt").read_text(encoding="utf-8")


def seed_for(seed: int, sample_id: str, stage: str, iteration: int = 0, qid: str = "") -> int:
    return None


def messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def format_documents(docs: list[dict]) -> str:
    return "\n\n".join(f'Title: {d["title"]}\n{d["text"]}' for d in docs)


@dataclass
class RunConfig:
    method: str = "stride"
    guidance: str = "none"
    top_k: int = 3
    max_iterations: int = 5
    failure_threshold: int = 3
    parse_attempts: int = 3
    seed: int = 42
    trace_prompts: bool = True

    def __post_init__(self):
        Stage(self.guidance)
        if self.max_iterations != 5:
            raise ValueError("Paper runs require max_iterations=5")
        if self.top_k not in {3, 5}:
            raise ValueError("Paper runs require top_k=3 or 5")


@dataclass
class State:
    id: str
    question: str
    plan: str
    solved: dict[str, Any] = field(default_factory=dict)
    facts: dict[str, str] = field(default_factory=dict)
    failures: dict[str, list[str]] = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)
    answer: str | None = None
    status: str = "running"
    iterations: int = 0

    def pending(self) -> set[str]:
        return {o.qid for o in operations(self.plan)} - set(self.solved)


    def ready(self) -> set[str]:
        """Only explicit predecessor references may gate an operation; no gold inputs."""
        return {o.qid for o in operations(self.plan) if o.qid not in self.solved
                and (set(re.findall(r"\bQ\d+\b", o.text)) - {o.qid}) <= set(self.solved)}


class Stride:
    def __init__(self, backend: Backend, retriever: Retriever, config: RunConfig):
        self.backend, self.retriever, self.config = backend, retriever, config
        self.stage = Stage(config.guidance)
        self.prompts = {name: prompt(name) for name in ["planner", "supervisor", "extractor", "reasoner"]}

    def plan(self, questions: list[dict]) -> list[dict]:
        requests = []
        for q in questions:
            if set(q) != {"id", "question"}:
                raise ValueError("Inference accepts only id and question, never labels/contexts")
            requests.append(Request(str(q["id"]), "planner",
                messages(self.prompts["planner"], f'Question: \n{q["question"]}'),
                seed_for(self.config.seed, str(q["id"]), "planner")))
        outputs = self.backend.generate(requests)
        return [{"id": q["id"], "question_sha256": digest(q["question"]),
                 "plan": out.text, "plan_sha256": digest(out.text),
                 "generation": asdict(out), "seed": req.seed,
                 "backend": self.backend.identity, "prompt_sha256": digest(req.messages)}
                for q, req, out in zip(questions, requests, outputs)]

    def _generate(self, tasks: list[tuple[State, str, str, int, str, str]]) -> list[str]:
        """(state, module, qid, iteration, system, user) -> logged raw outputs."""
        requests = [Request(s.id, module, messages(system, user),
                    seed_for(self.config.seed, s.id, module, iteration, qid))
                    for s, module, qid, iteration, system, user in tasks]
        responses = self.backend.generate(requests)
        for (state, module, qid, iteration, _, _), req, out in zip(tasks, requests, responses):
            entry = {"module": module, "qid": qid, "iteration": iteration,
                     "prompt_sha256": digest(req.messages), "seed": req.seed, **asdict(out)}
            if self.config.trace_prompts:
                entry["messages"] = req.messages
            state.trace.append(entry)
        return [r.text for r in responses]

    def _guidance(self, state: State, qid: str, module: str, text: str) -> str:
        if not self.stage.includes(module):
            return text
        trajectory = remaining(operations(state.plan), qid, set(state.solved))
        if module == "retrieval":
            return text + ("\n" + trajectory if trajectory else "")
        return append_guidance(text, trajectory)

    def run(self, questions: list[dict], plans: list[dict]) -> list[dict]:
        if len({str(q["id"]) for q in questions}) != len(questions):
            raise ValueError("Duplicate question IDs")
        by_id = {str(p["id"]): p for p in plans}
        if len(by_id) != len(plans):
            raise ValueError("Duplicate plan IDs")
        states = []
        for q in questions:
            if set(q) != {"id", "question"}:
                raise ValueError("Inference accepts only id and question")
            plan = by_id[str(q["id"])]
            if plan["question_sha256"] != digest(q["question"]) or plan["plan_sha256"] != digest(plan["plan"]):
                raise ValueError("Plan/question integrity mismatch")
            state = State(str(q["id"]), q["question"], plan["plan"])
            try:
                operations(state.plan)
            except ParseError:
                state.status = "plan_parse_error"
            states.append(state)

        for iteration in range(1, self.config.max_iterations + 1):
            active = [s for s in states if s.status == "running"]
            for state in active:
                if not state.ready():
                    state.status = "dependency_deadlock"
            active = [s for s in active if s.status == "running"]
            if not active:
                break
            schedules: dict[str, list[dict]] = {}
            awaiting = list(active)
            for attempt in range(self.config.parse_attempts):
                tasks = []
                for s in awaiting:
                    progress = f"Solved: {s.solved}\nCollected facts: {s.facts}\nPending: {sorted(s.pending())}\nReady: {sorted(s.ready())}\nFailureLog: {s.failures}"
                    user = f"Question: {s.question}\n\n{s.plan}\n\nProgress:\n{progress}"
                    tasks.append((s, "supervisor", "", iteration * 10 + attempt,
                                  self.prompts["supervisor"], user))
                outputs = self._generate(tasks)
                failed = []
                for s, raw in zip(awaiting, outputs):
                    try:
                        proposed = actions(raw, s.pending())
                        ready = s.ready()
                        accepted = [a for a in proposed if a["qid"] in ready]
                        blocked = [a for a in proposed if a["qid"] not in ready]
                        if blocked:
                            s.trace.append({"module": "dependency_guard", "iteration": iteration,
                                            "solved": sorted(s.solved), "ready": sorted(ready),
                                            "blocked": blocked})
                        if not accepted:
                            raise ParseError("No proposed operation has satisfied dependencies")
                        schedules[s.id] = accepted
                    except ParseError:
                        failed.append(s)
                awaiting = failed
                if not awaiting:
                    break
            for s in awaiting:
                s.status = "supervisor_parse_error"

            retrieval_tasks = []
            reasoning_tasks = []
            for s in active:
                if s.status != "running":
                    continue
                s.iterations = iteration
                for action in schedules[s.id]:
                    if len(s.failures.get(action["qid"], [])) >= self.config.failure_threshold:
                        s.status = "unresolved"
                        break
                    if action["action"] in {"retrieve", "rewrite"}:
                        retrieval_tasks.append((s, action))
                    else:
                        reasoning_tasks.append((s, action, "\n".join(s.facts.values())))
            retrieval_tasks = [(s, a) for s, a in retrieval_tasks if s.status == "running"]
            queries = [self._guidance(s, a["qid"], "retrieval", a["query"]) for s, a in retrieval_tasks]
            docs = self.retriever.search(queries, self.config.top_k, question_ids=[s.id for s, _ in retrieval_tasks]) if queries else []
            extraction = []
            for (s, a), query, passages in zip(retrieval_tasks, queries, docs):
                s.trace.append({"module": "retrieval", "qid": a["qid"], "iteration": iteration,
                    "query": query, "documents": passages, "top_k": self.config.top_k})
                operation = next(o for o in operations(s.plan) if o.qid == a['qid'])
                user = f'Question: \n{operation.text}\nResolved references: {s.solved}\nSearch query: {a["query"]}\n\nDocuments: \n{format_documents(passages)}'
                system = self.prompts["extractor"]
                user = self._guidance(s, a["qid"], "extraction", user)
                extraction.append((s, "extractor", a["qid"], iteration, system, user))
            for (s, a), raw in zip(retrieval_tasks, self._generate(extraction)):
                try:
                    fact_text = extracted_facts(raw)
                    if fact_text is None:
                        raise ParseError("No extracted facts")
                    previous = s.facts.get(a["qid"], "")
                    s.facts[a["qid"]] = (previous + "\n" + fact_text).strip() if fact_text not in previous else previous
                    reasoning_tasks.append((s, a, s.facts[a["qid"]]))
                except ParseError:
                    s.failures.setdefault(a["qid"], []).append(a["query"])
            # Answer actions see all facts committed by this retrieval phase.
            reasoning = []
            for s, a, facts in reasoning_tasks:
                if s.status != "running":
                    continue
                if a["action"] == "answer":
                    facts = "\n".join(s.facts.values())
                role = "final" if a["qid"] == operations(s.plan)[-1].qid else "intermediate"
                operation = next(o for o in operations(s.plan) if o.qid == a['qid'])
                question = s.question if role == 'final' else operation.text
                user = f'Operation role: {role}\n\nFacts: \n{facts}\n\nCompleted results: {s.solved}\nOther accumulated facts: {s.facts}\n\nCURRENT QUESTION TO ANSWER: \n{question}'
                system = self.prompts["reasoner"]
                user = self._guidance(s, a["qid"], "reasoning", user)
                reasoning.append((s, "reasoner", a["qid"], iteration, system, user))
            outputs = self._generate(reasoning)
            actions_by_key = {(s.id, a["qid"]): a for s, a, _ in reasoning_tasks}
            for task, raw in zip(reasoning, outputs):
                s, _, qid, _, _, _ = task
                action = actions_by_key[s.id, qid]
                try:
                    answer = answer_span(raw)
                except ParseError:
                    answer = None
                if answer is None:
                    s.failures.setdefault(qid, []).append(action["query"])
                    continue
                s.solved[qid] = answer
                s.failures.pop(qid, None)
                if qid == operations(s.plan)[-1].qid:
                    s.answer, s.status = answer, "ok"
            for s in active:
                if s.status == "running" and any(len(v) >= self.config.failure_threshold for v in s.failures.values()):
                    s.status = "unresolved"

        result = []
        for s in states:
            if s.status == "running":
                s.status = "iteration_limit"
            usage = {}
            for event in s.trace:
                if "input_tokens" not in event:
                    continue
                u = usage.setdefault(event["module"], dict(calls=0, input_tokens=0, output_tokens=0,
                                                           seconds=0.0, cached_calls=0, cost_usd=0.0))
                u["calls"] += 1
                for key in ["input_tokens", "output_tokens", "seconds", "cost_usd"]:
                    u[key] += event[key]
                u["cached_calls"] += int(event["cached"])
            result.append({"id": s.id, "question": s.question, "answer": s.answer,
                "status": s.status, "iterations": s.iterations, "method": self.config.method,
                "guidance": self.config.guidance, "plan": s.plan, "plan_sha256": digest(s.plan),
                "solved": s.solved, "failures": s.failures, "usage": usage, "trace": s.trace})
        return result
