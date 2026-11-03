"""Local batched inference and a resumable call cache."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .io import canonical, digest


@dataclass
class Request:
    sample_id: str
    stage: str
    messages: list[dict[str, str]]
    seed: int


@dataclass
class Response:
    text: str
    input_tokens: int
    output_tokens: int
    seconds: float
    finish_reason: str = "stop"
    model: str = ""
    fingerprint: str | None = None
    cached: bool = False
    cost_usd: float = 0.0


class Backend(Protocol):
    identity: dict

    def generate(self, requests: list[Request]) -> list[Response]: ...


class CallCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=60, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, response TEXT NOT NULL)")
        self.lock = threading.Lock()

    def get(self, key: str) -> Response | None:
        with self.lock:
            row = self.db.execute("SELECT response FROM calls WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        value = json.loads(row[0])
        value["cached"] = True
        return Response(**value)

    def put(self, key: str, response: Response) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO calls VALUES (?,?)", (key, canonical(asdict(response))))


class CachedBackend:
    def __init__(self, backend: Backend, path: str | Path):
        self.backend, self.cache, self.identity = backend, CallCache(path), backend.identity

    def generate(self, requests: list[Request]) -> list[Response]:
        keys = [digest({"backend": self.identity, "request": asdict(r)}) for r in requests]
        results = [self.cache.get(k) for k in keys]
        missing = [i for i, r in enumerate(results) if r is None]
        if missing:
            new = self.backend.generate([requests[i] for i in missing])
            if len(new) != len(missing):
                raise RuntimeError("Backend returned a different number of responses")
            for i, response in zip(missing, new):
                results[i] = response
                self.cache.put(keys[i], response)
        return results  # type: ignore[return-value]








class VLLMBackend:
    def __init__(self, model: str, *, revision: str | None = None, tensor_parallel: int = 1,
                 memory_fraction: float = .82, max_model_len: int = 8192,
                 temperature: float = 1.0, max_tokens: int = 512, max_num_seqs: int = 64,
                 dtype: str = "auto", enforce_eager: bool = False):
        from .devices import normalize_vllm_mask
        normalize_vllm_mask()
        from transformers import AutoTokenizer
        from vllm import LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = LLM(model=model, revision=revision, tensor_parallel_size=tensor_parallel,
                         gpu_memory_utilization=memory_fraction, max_model_len=max_model_len,
                         max_num_seqs=max_num_seqs, dtype=dtype, enforce_eager=enforce_eager,
                         enable_prefix_caching=True, trust_remote_code=False, disable_custom_all_reduce=True, max_num_batched_tokens=2048)
        self.temperature, self.max_tokens, self.max_model_len = temperature, max_tokens, max_model_len
        self.identity = {"backend": "vllm", "model": model, "revision": revision,
                         "temperature": temperature, "max_tokens": max_tokens,
                         "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0, "enable_thinking": False,
                         "dtype": dtype, "tensor_parallel": tensor_parallel,
                         "max_model_len": max_model_len}

    def generate(self, requests: list[Request]) -> list[Response]:
        from vllm import SamplingParams
        if not requests:
            return []
        prompts = [self.tokenizer.apply_chat_template(r.messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False) for r in requests]
        for prompt in prompts:
            if len(self.tokenizer.encode(prompt)) + self.max_tokens > self.max_model_len:
                raise ValueError("Prompt exceeds model context; no implicit truncation is permitted")
        sampling = [SamplingParams(temperature=self.temperature, top_p=0.95, top_k=20, presence_penalty=0.0,
                    max_tokens=self.max_tokens, seed=r.seed) for r in requests]
        start = time.monotonic()
        outputs = self.model.generate(prompts, sampling, use_tqdm=False)
        elapsed = time.monotonic() - start
        return [Response(o.outputs[0].text, len(o.prompt_token_ids), len(o.outputs[0].token_ids),
                         elapsed / len(outputs), o.outputs[0].finish_reason or "unknown",
                         self.identity["model"]) for o in outputs]


