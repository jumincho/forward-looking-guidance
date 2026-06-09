"""
Shared infrastructure: vLLM engine, dense retrieval (Contriever + FAISS IndexFlatIP), answer
normalization / EM / token-F1, plan parsing and the Forward-Looking Guidance (FG) helpers.

Design:
  * One vLLM engine per GPU (tensor_parallel_size=1); the test set is sharded across
    the available GPUs for throughput.
  * Qwen3-8B runs with extended reasoning disabled (enable_thinking=False).
"""
from __future__ import annotations

import os
import re
import sys
import ast
import json
import time
import string
from collections import Counter
from pathlib import Path
from typing import Any


def parse_obj(s: str):
    """Parse a JSON / Python-literal object (dict or list) emitted by a model.

    Tolerates ``//`` line comments; tries strict JSON first, then a Python literal
    (single quotes, True/False/None). Returns the parsed value, or None on failure.
    """
    s = re.sub(r"//.*", "", s)
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(s)
        except Exception:
            pass
    return None


# ---- metrics / normalization ------------------------------------------------------
def remove_articles(text):
    text = text.replace('_', ' ')
    text = text.replace('"', '')
    text = re.sub(r"'|`|,|’|\\|´", '', text)
    return re.sub(r'\b(a|an|the)\b', ' ', text)


def white_space_fix(text):
    return ' '.join(text.split())


def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)


def lower(text):
    return text.lower().strip()


def normalize_answer(s):
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth) -> float:
    if prediction is None or ground_truth is None:
        return 0.0
    return 1.0 if normalize_answer(str(prediction)) == normalize_answer(str(ground_truth)) else 0.0


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def total_f1_score(predictions, ground_truths):
    n = len(predictions)
    tf1 = tp = tr = 0.0
    for i in range(n):
        f1, p, r = f1_score(predictions[i], ground_truths[i])
        tf1 += f1; tp += p; tr += r
    return (tf1 / n if n else 0.0, tp / n if n else 0.0, tr / n if n else 0.0)


def total_exact_match_score(y_true, y_pred):
    s = 0
    for t, p in zip(y_true, y_pred):
        if normalize_answer(t) == normalize_answer(p):
            s += 1
    return s / len(y_true) if y_true else 0.0

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FAISS_DIR = ROOT / "faiss_index"
OUTPUT_DIR = ROOT / "output"
for _d in (DATA_DIR, FAISS_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASETS = ["2wikimultihopqa", "hotpotqa", "musique"]
# top-k retrieved passages for the planning-based executor (per dataset)
STRIDE_TOPK = {"2wikimultihopqa": 3, "hotpotqa": 5, "musique": 3}
# top-k for the non-planning retrieval baselines
ORDINARY_TOPK = 5


# ----------------------------------------------------------------------------------
# Data IO
# ----------------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_path(dataset: str) -> Path:
    return DATA_DIR / f"{dataset}_test1000.jsonl"


def faiss_path(dataset: str) -> Path:
    return FAISS_DIR / dataset / "index"


# ----------------------------------------------------------------------------------
# LLM engine: local models via vLLM (one per GPU), or an OpenAI-hosted model via the API.
# ----------------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", os.environ.get("QWEN3_PATH", "Qwen/Qwen3-8B"))


class LLMEngine:
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.80,  # leave headroom for the Contriever encoder
        max_num_seqs: int = 128,
        seed: int = 0,
    ):
        self.model_path = model_path
        # OpenAI-hosted models (e.g. gpt-4o-mini) use the API; all others run locally via vLLM.
        self.is_openai = model_path.startswith("gpt-") or model_path.startswith("openai/")
        if self.is_openai:
            from openai import OpenAI
            self.client = OpenAI()
            self.openai_model = model_path.split("/", 1)[-1]
            self.is_qwen3 = False
            return
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from vllm import LLM, SamplingParams  # noqa
        from transformers import AutoTokenizer

        self.SamplingParams = SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.is_qwen3 = "Qwen3" in model_path or "qwen3" in model_path.lower()
        self.llm = LLM(
            model=model_path,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            tensor_parallel_size=1,
            seed=seed,
            enforce_eager=False,
            dtype="bfloat16",
            enable_prefix_caching=True,   # shared few-shot prefixes -> big KV savings
            disable_log_stats=True,
        )

    def chat(
        self,
        messages_list: list[list[dict]],
        temperature: float = 1.0,
        max_tokens: int = 512,
        top_p: float = 0.95,
        think: bool = False,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> list[str]:
        """Batched chat generation. Qwen3 thinking is disabled by default."""
        if not messages_list:
            return []
        if self.is_openai:
            return [self._openai_chat(m, temperature, max_tokens, top_p, stop)
                    for m in messages_list]
        sp = self.SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            stop=stop, seed=seed,
        )
        kw = {}
        if self.is_qwen3:
            kw["enable_thinking"] = think
        texts = self.tokenizer.apply_chat_template(
            messages_list, tokenize=False, add_generation_prompt=True, **kw
        )
        outs = self.llm.generate(texts, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def complete(
        self,
        prompts: list[str],
        temperature: float = 1.0,
        max_tokens: int = 512,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> list[str]:
        """Raw text completion (no chat template) — used by SelfAsk few-shot scaffolding."""
        if not prompts:
            return []
        if self.is_openai:
            return [self._openai_chat([{"role": "user", "content": p}],
                                      temperature, max_tokens, top_p, stop) for p in prompts]
        sp = self.SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop,
        )
        outs = self.llm.generate(prompts, sp, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    def _openai_chat(self, messages, temperature, max_tokens, top_p, stop):
        r = self.client.chat.completions.create(
            model=self.openai_model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, top_p=top_p, stop=stop)
        return r.choices[0].message.content or ""


# ----------------------------------------------------------------------------------
# Retrieval: Contriever + FAISS dense retriever
# ----------------------------------------------------------------------------------
def load_encoder(retriever_model_path: str | None = None):
    """Load the Contriever encoder once (shared across datasets in a worker)."""
    from retrieval.encoder import load_contriever_and_tokenizer
    rmp = retriever_model_path or os.environ.get("CONTRIEVER_PATH", "facebook/contriever")
    return load_contriever_and_tokenizer(rmp)


class Retriever:
    def __init__(self, dataset: str, retriever_model_path: str | None = None, encoder=None):
        from retrieval.dense import DenseRetriever

        model, tok = encoder if encoder is not None else load_encoder(retriever_model_path)
        self.r = DenseRetriever(model, tok)
        self.r.load_index(str(faiss_path(dataset)))

    def batch_retrieve(self, queries: list[str], top_k: int = 5, chunk: int = 64):
        # chunk query encoding so the Contriever forward never blows the GPU headroom
        if len(queries) <= chunk:
            return self.r.batch_retrieve(queries, top_k=top_k)
        out = []
        for i in range(0, len(queries), chunk):
            out.extend(self.r.batch_retrieve(queries[i:i + chunk], top_k=top_k))
        return out

    def retrieve(self, query: str, top_k: int = 5):
        return self.r.retrieve(query, top_k=top_k)


def docs_to_str(results: list[dict]) -> str:
    s = ""
    for res in results:
        s += f"Title: {res['title']}\n{res['text']}\n\n"
    return s


# ----------------------------------------------------------------------------------
# Plan parsing + Forward-Looking Guidance helpers
# ----------------------------------------------------------------------------------
def parse_plan(plan_text: str) -> list[tuple[str, str]]:
    """Parse 'Q1: ... \n Q2: ...' into [(qid, text), ...] in order."""
    plan_text = plan_text.replace("Concrete Plan", "Plan")
    items = re.findall(r"(Q\d+)\s*:\s*(.*?)(?=(?:\n\s*Q\d+\s*:)|\Z)", plan_text, re.DOTALL)
    out = []
    seen = set()
    for qid, txt in items:
        qid = qid.strip()
        txt = " ".join(txt.split()).strip()
        if qid not in seen and txt:
            seen.add(qid)
            out.append((qid, txt))
    return out


def remaining_subquestions(plan: list[tuple[str, str]], current_qid: str) -> list[tuple[str, str]]:
    """G_i = (o_{i+1}, ..., o_n): the planned sub-questions AFTER current_qid (by plan order)."""
    idx = None
    for i, (qid, _) in enumerate(plan):
        if qid == current_qid:
            idx = i
            break
    if idx is None:
        return []
    return plan[idx + 1:]


def fg_block(remaining: list[tuple[str, str]], role: str = "extract") -> str:
    """Render the forward-looking guidance block injected into extractor / reasoner inputs.

    role='extract': keep extra facts that later steps will need (verbosity OK — facts).
    role='reason' : keep the answer a short, disambiguated span for the later steps.
    """
    if not remaining:
        return ""
    lines = "\n".join(f"- {qid}: {txt}" for qid, txt in remaining)
    if role == "reason":
        # Guidance placed before the current question, with an explicit "do not answer these".
        return (
            "Forward-looking context (do NOT answer these — they are later planned steps that "
            "will reuse your answer):\n" + lines +
            "\nKeep your answer to the current question a short span, but fully disambiguated "
            "(e.g. 'Springfield, Illinois' not 'Springfield') so those later steps can use it."
        )
    return (
        "\n\nForward-looking guidance — the remaining planned reasoning steps that will reuse "
        "these facts:\n" + lines +
        "\nAlso keep any atomic fact those steps will need, fully specified and disambiguated."
    )


# ----------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------
def score_predictions(rows: list[dict]) -> dict[str, float]:
    """rows: list of {'label': str, 'pred': str}. Returns EM/F1/P/R."""
    y_true = [str(r.get("label", "")) for r in rows]
    y_pred = [str(r.get("pred", "") or "") for r in rows]
    em = total_exact_match_score(y_true, y_pred)
    f1, p, r = total_f1_score(y_pred, y_true)
    return {"EM": em, "F1": f1, "P": p, "R": r, "n": len(rows)}


def fmt_metrics(m: dict[str, float]) -> str:
    return (
        f"EM={m['EM']:.3f} F1={m['F1']:.3f} P={m['P']:.3f} R={m['R']:.3f} (n={int(m['n'])})"
    )
