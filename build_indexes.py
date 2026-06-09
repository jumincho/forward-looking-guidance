"""Build per-dataset Contriever + FAISS indexes from the candidate passages provided with each
test question (deduped). Writes {faiss.index, document.vecstore.npz} per dataset."""
import os, sys, json, time, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from retrieval.encoder import load_contriever_and_tokenizer
from retrieval.dense import DenseRetriever

DATA = Path(__file__).resolve().parent / "data"
IDX = Path(__file__).resolve().parent / "faiss_index"
RETRIEVER = os.environ.get("CONTRIEVER_PATH", "facebook/contriever")


def iter_passages(jsonl_path, dedupe=True):
    seen = set()
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            for ctx in (obj.get("pinned_contexts") or []) + (obj.get("contexts") or []):
                title = str(ctx.get("title", "") or "")
                text = str(ctx.get("paragraph_text", ctx.get("text", "")) or "")
                if not text.strip():
                    continue
                key = hashlib.sha256((title + "\0" + text).encode("utf-8", "replace")).hexdigest()
                if dedupe and key in seen:
                    continue
                seen.add(key)
                yield title, text


def build(dataset, batch_size=256):
    out = IDX / dataset / "index"
    if (out / "faiss.index").exists():
        print(f"[{dataset}] index exists, skip"); return
    model, tok = load_contriever_and_tokenizer(RETRIEVER)
    r = DenseRetriever(model, tok, batch_size=batch_size)
    titles, texts = [], []
    for title, text in iter_passages(DATA / f"{dataset}_test1000.jsonl"):
        titles.append(title); texts.append(text)
        if len(texts) >= batch_size:
            r.add_docs(texts, titles); titles, texts = [], []
    if texts:
        r.add_docs(texts, titles)
    r.save_index(str(out.expanduser().resolve()))
    print(f"[{dataset}] wrote {r.ctr} vectors")


if __name__ == "__main__":
    datasets = sys.argv[1:] or ["2wikimultihopqa", "hotpotqa", "musique"]
    for ds in datasets:
        t0 = time.time(); print(f"[{ds}] building index ...", flush=True)
        build(ds); print(f"[{ds}] done in {time.time()-t0:.0f}s", flush=True)
    print("ALL INDEXES DONE")
