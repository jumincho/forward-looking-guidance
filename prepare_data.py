"""
Build the 1000-question test splits + per-dataset retrieval corpora for
2WikiMultiHopQA, HotpotQA, MuSiQue from the raw HF dumps.

Output rows:
  {id, question, answer, answer_aliases, hop, type,
   contexts: [{title, paragraph_text, is_supporting}, ...]}

The retrieval corpus for each dataset is the deduped union of all `contexts` across that
dataset's test set (the candidate passages provided with each question).

Each dataset samples 1,000 questions with a fixed seed.
"""
from __future__ import annotations

import json
import random
import collections
from pathlib import Path

import pyarrow.parquet as pq

RAW = Path(__file__).resolve().parent / "raw"
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42


def _clean(t: str) -> str:
    return " ".join(str(t).replace("\n", " ").split()).strip()


# ----------------------------------------------------------------------------------
def load_musique() -> list[dict]:
    rows = []
    with open(RAW / "musique_dev.jsonl") as f:
        for line in f:
            o = json.loads(line)
            if not o.get("answerable", True):
                continue
            hop = int(o["id"][0])
            contexts = [
                {
                    "title": _clean(p["title"]),
                    "paragraph_text": _clean(p["paragraph_text"]),
                    "is_supporting": bool(p["is_supporting"]),
                }
                for p in o["paragraphs"]
            ]
            rows.append({
                "id": o["id"],
                "question": _clean(o["question"]),
                "answer": o["answer"],
                "answer_aliases": o.get("answer_aliases", []),
                "hop": hop,
                "type": f"{hop}hop",
                "contexts": contexts,
            })
    return rows


def load_hotpotqa() -> list[dict]:
    t = pq.read_table(RAW / "hotpotqa_dev.parquet").to_pylist()
    rows = []
    for o in t:
        sup_titles = set(o["supporting_facts"]["title"])
        titles = o["context"]["title"]
        sents = o["context"]["sentences"]
        contexts = []
        for ti, se in zip(titles, sents):
            contexts.append({
                "title": _clean(ti),
                "paragraph_text": _clean(" ".join(se)),
                "is_supporting": ti in sup_titles,
            })
        rows.append({
            "id": o["id"],
            "question": _clean(o["question"]),
            "answer": o["answer"],
            "answer_aliases": [],
            "hop": 2,
            "type": o.get("type", ""),
            "contexts": contexts,
        })
    return rows


def load_2wiki() -> list[dict]:
    t = pq.read_table(RAW / "2wiki_dev.parquet").to_pylist()
    rows = []
    for o in t:
        ctx = o["context"]
        sf = o["supporting_facts"]
        if isinstance(ctx, str):
            ctx = json.loads(ctx)
        if isinstance(sf, str):
            sf = json.loads(sf)
        sup_titles = set(x[0] for x in sf)
        contexts = []
        for pair in ctx:
            ti, se = pair[0], pair[1]
            contexts.append({
                "title": _clean(ti),
                "paragraph_text": _clean(" ".join(se)),
                "is_supporting": ti in sup_titles,
            })
        rows.append({
            "id": o["_id"],
            "question": _clean(o["question"]),
            "answer": o["answer"],
            "answer_aliases": [],
            "hop": 2,
            "type": o.get("type", ""),
            "contexts": contexts,
        })
    return rows


# ----------------------------------------------------------------------------------
def sample_1000(rows: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    pool = sorted(rows, key=lambda r: str(r["id"]))
    rng.shuffle(pool)
    return pool[:1000]


def write_split(dataset: str, rows: list[dict]) -> None:
    path = OUT / f"{dataset}_test1000.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # corpus stats
    seen = set()
    sup = 0
    for r in rows:
        for c in r["contexts"]:
            seen.add((c["title"], c["paragraph_text"]))
            sup += int(c["is_supporting"])
    hopc = collections.Counter(r["hop"] for r in rows)
    print(f"[{dataset}] wrote {len(rows)} questions -> {path.name} | "
          f"unique passages={len(seen)} | supporting={sup} | hops={dict(sorted(hopc.items()))}")


def main():
    print("loading raw datasets ...")
    mus = load_musique()
    hot = load_hotpotqa()
    wik = load_2wiki()
    print(f"raw sizes: musique(answerable)={len(mus)} hotpotqa={len(hot)} 2wiki={len(wik)}")

    write_split("musique", sample_1000(mus))
    write_split("hotpotqa", sample_1000(hot))
    write_split("2wikimultihopqa", sample_1000(wik))
    print("DONE prepare_data")


if __name__ == "__main__":
    main()
