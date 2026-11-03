"""Unsupervised Contriever mean pooling, L2 normalization, FAISS inner product."""
from __future__ import annotations

import json
from pathlib import Path

from .io import file_digest, read_jsonl, write_json, write_jsonl


class Contriever:
    def __init__(self, model: str = "facebook/contriever", *, device: str = "cuda",
                 revision: str | None = None, batch_size: int = 64, passage_input: str = "body"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device, self.batch_size = torch, device, batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModel.from_pretrained(model, revision=revision).to(device).eval()
        if passage_input not in {"body", "title_body"}:
            raise ValueError("Passage input must be body or title_body")
        self.passage_input = passage_input
        self.identity = {"model": model, "revision": revision, "pooling": "masked_mean",
                         "normalize": True, "max_length": 512, "passage_input": passage_input}

    def encode(self, texts: list[str]):
        import numpy as np
        outputs = []
        for i in range(0, len(texts), self.batch_size):
            tokens = self.tokenizer(texts[i:i+self.batch_size], padding=True, truncation=True,
                                    max_length=512, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens.attention_mask.unsqueeze(-1).float()
                vectors = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors = self.torch.nn.functional.normalize(vectors, p=2, dim=-1)
            outputs.append(vectors.float().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype("float32") if outputs else np.empty((0,768), "float32")


def gpu_index(index, device: int):
    import faiss
    if not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError("FAISS-GPU is required. Install the [gpu] extra or explicitly select --faiss-device cpu")
    resource = faiss.StandardGpuResources()
    resource.setTempMemory(128 * 1024 * 1024)
    return faiss.index_cpu_to_gpu(resource, device, index), resource


def build_index(corpus_path: str | Path, output: str | Path, encoder: Contriever, *, faiss_device: str = "cuda") -> dict:
    import faiss
    import numpy as np
    docs = read_jsonl(corpus_path)
    if not docs or len({d["id"] for d in docs}) != len(docs):
        raise ValueError("Empty corpus or duplicate document IDs")
    if any(set(d) != {"id", "title", "text"} for d in docs):
        raise ValueError("Index input must contain only id, title, text")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    vectors = encoder.encode([(d["title"] + "\n" if encoder.passage_input == "title_body" else "")
                              + d["text"] for d in docs])
    index = faiss.IndexFlatIP(vectors.shape[1])
    resource = None
    if faiss_device == "cuda":
        index, resource = gpu_index(index, 0)
    index.add(np.ascontiguousarray(vectors))
    if faiss_device == "cuda":
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, str(output / "passages.index"))
    write_jsonl(output / "documents.jsonl", docs)
    manifest = {"n": len(docs), "dimension": vectors.shape[1], "encoder": encoder.identity,
                "faiss_device": faiss_device, "index_type": "IndexFlatIP", "faiss_version": faiss.__version__,
                "documents_sha256": file_digest(output / "documents.jsonl"),
                "corpus_sha256": file_digest(corpus_path), "index_sha256": file_digest(output / "passages.index")}
    write_json(output / "manifest.json", manifest)
    return manifest


class DenseRetriever:
    def __init__(self, index_dir: str | Path, encoder: Contriever, *, faiss_device: str = "cuda"):
        import faiss
        index_dir = Path(index_dir)
        self.encoder = encoder
        self.documents = read_jsonl(index_dir / "documents.jsonl")
        manifest = json.loads((index_dir / "manifest.json").read_text())
        if (manifest.get("documents_sha256") is not None and
                file_digest(index_dir / "documents.jsonl") != manifest["documents_sha256"]):
            raise ValueError("Document-to-index mapping hash mismatch")
        if file_digest(index_dir / "passages.index") != manifest["index_sha256"]:
            raise ValueError("Index hash mismatch")
        self.index = faiss.read_index(str(index_dir / "passages.index"))
        if self.index.ntotal != len(self.documents):
            raise ValueError("Document/index cardinality mismatch")
        self.resource = None
        if faiss_device == "cuda":
            self.index, self.resource = gpu_index(self.index, 0)
        self.identity = {**manifest, "runtime_faiss_device": faiss_device}
        self.cache = {}
        self.scope = "global"

    def search(self, queries: list[str], top_k: int, *, question_ids=None) -> list[list[dict]]:
        if not queries:
            return []
        missing = list(dict.fromkeys(q for q in queries if (q, top_k) not in self.cache))
        if missing:
            vectors = self.encoder.encode(missing)
            scores, ids = self.index.search(vectors, min(top_k, self.index.ntotal))
            for query, row_scores, row_ids in zip(missing, scores, ids):
                self.cache[query, top_k] = [{**self.documents[int(i)], "score": float(score)}
                                           for score, i in zip(row_scores, row_ids) if i >= 0]
        return [self.cache[q, top_k] for q in queries]


class CandidatePoolRetriever:
    """Dense retrieval over each question's complete public distractor pool.

    Pools contain all supplied candidates, without supporting-fact flags or answers.
    This scope is explicitly distinct from searching the benchmark-wide corpus.
    """
    def __init__(self, index_dir, encoder, pools_path, *, faiss_device="cuda"):
        import faiss
        import numpy as np
        directory = Path(index_dir)
        self.encoder = encoder
        self.documents = read_jsonl(directory / "documents.jsonl")
        manifest = json.loads((directory / "manifest.json").read_text())
        if file_digest(directory / "passages.index") != manifest["index_sha256"]:
            raise ValueError("Index hash mismatch")
        if (manifest.get("documents_sha256") is not None and
                file_digest(directory / "documents.jsonl") != manifest["documents_sha256"]):
            raise ValueError("Document-to-index mapping hash mismatch")
        self.base = faiss.read_index(str(directory / "passages.index"))
        if self.base.ntotal != len(self.documents):
            raise ValueError("Document/index cardinality mismatch")
        by_id = {d["id"]: i for i, d in enumerate(self.documents)}
        self.pools = {}
        for row in read_jsonl(pools_path):
            if set(row) != {"id", "document_ids"} or str(row["id"]) in self.pools:
                raise ValueError("Candidate pools accept unique IDs and document IDs only")
            ids = row["document_ids"]
            if not ids or len(set(ids)) != len(ids) or any(i not in by_id for i in ids):
                raise ValueError("Candidate pool contains missing or duplicate documents")
            self.pools[str(row["id"])] = np.asarray([by_id[i] for i in ids], dtype="int64")
        self.scope = "question_candidates"
        self.identity = {**manifest, "runtime_faiss_device": faiss_device,
                         "scope": self.scope, "candidate_pools_sha256": file_digest(pools_path)}
        self.faiss_device, self.cache, self.indices = faiss_device, {}, {}
        self.resource = None
        if faiss_device == "cuda":
            if not hasattr(faiss, "StandardGpuResources"):
                raise RuntimeError("FAISS-GPU is required for GPU candidate retrieval")
            self.resource = faiss.StandardGpuResources()
            self.resource.setTempMemory(128 * 1024 * 1024)

    def search(self, queries, top_k, *, question_ids=None):
        import faiss
        if question_ids is None or len(question_ids) != len(queries):
            raise ValueError("Candidate retrieval requires a question ID for every query")
        keys = [(str(qid), query, top_k) for qid, query in zip(question_ids, queries)]
        missing = list(dict.fromkeys(k for k in keys if k not in self.cache))
        if missing:
            vectors = self.encoder.encode([query for _, query, _ in missing])
            for (qid, query, k), vector in zip(missing, vectors):
                if qid not in self.pools:
                    raise ValueError(f"Missing candidate pool for question {qid}")
                mapping = self.pools[qid]
                if qid not in self.indices:
                    index = faiss.IndexFlatIP(self.base.d)
                    index.add(self.base.reconstruct_batch(mapping))
                    if self.resource is not None:
                        index = faiss.index_cpu_to_gpu(self.resource, 0, index)
                    self.indices[qid] = index
                scores, ids = self.indices[qid].search(vector.reshape(1, -1), min(k, len(mapping)))
                self.cache[qid, query, k] = [
                    {**self.documents[int(mapping[i])], "score": float(score)}
                    for score, i in zip(scores[0], ids[0]) if i >= 0]
        return [self.cache[k] for k in keys]
