"""Deterministic IRCoT/STRIDE-style 500 + 500 split and global passage corpus."""
from __future__ import annotations

import json
import random
import re
import zipfile
from collections import Counter
from pathlib import Path

from .io import digest, file_digest, write_json, write_jsonl

DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")
IRCoT_ARCHIVE = "https://drive.google.com/file/d/1t2BjJtsejSIUZI54PKObMFG6_wMMG3bC/view"


def prepare(archive: str | Path, output: str | Path, *, seed: int = 42,
            exclude_train_n: int = 10000, extra_n: int = 500) -> dict:
    output = Path(output)
    report = {"source": IRCoT_ARCHIVE, "archive_sha256": file_digest(archive), "seed": seed,
              "exclude_train_n": exclude_train_n, "extra_n": extra_n, "datasets": {}}
    with zipfile.ZipFile(archive) as z:
        for dataset in DATASETS:
            prefix = "processed_data/" + dataset + "/"
            def read(name):
                with z.open(prefix + name + ".jsonl") as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)
            base = list(read("test_subsampled"))
            dev = list(read("dev_subsampled"))
            train_ids = [x["question_id"] for x in read("train")]
            excluded_ids = set(random.Random(seed).sample(train_ids, min(exclude_train_n, len(train_ids))))
            blocked = excluded_ids | {x["question_id"] for x in base + dev}
            pool = [qid for qid in train_ids if qid not in blocked]
            if len(base) != 500 or len(pool) < extra_n:
                raise ValueError(f"Unexpected upstream split size for {dataset}")
            selected = random.Random(seed).sample(pool, extra_n)
            chosen = set(selected)
            extras = {x["question_id"]: x for x in read("train") if x["question_id"] in chosen}
            test = base + [extras[qid] for qid in selected]
            if len({x["question_id"] for x in test + dev}) != len(test) + len(dev):
                raise ValueError("Overlapping or duplicate split IDs")
            out = output / dataset
            corpus = {}
            for split, rows in [("test", test), ("dev", dev)]:
                questions, labels, candidate_pools = [], [], []
                for row in rows:
                    qid = row["question_id"]
                    questions.append({"id": qid, "question": row["question_text"]})
                    answers = [str(span) for obj in row["answers_objects"] for span in obj.get("spans", [])]
                    if not answers:
                        raise ValueError(f"No answer span for {qid}")
                    hops = int(m.group(1)) if (m := re.match(r"(\d)hop", qid)) else None
                    labels.append({"id": qid, "answer": answers[0], "aliases": answers[1:], "hops": hops,
                                   "source_split": "ircot_test" if row in base else ("ircot_dev" if split == "dev" else "train_extra")})
                    # Global corpus across all questions. Never index gold answers or annotations.
                    candidate_ids = []
                    for ctx in row.get("pinned_contexts", []) + row["contexts"]:
                        title = ctx["title"]
                        text = ctx["paragraph_text"].replace("\n", " ")
                        key = digest([title, text])
                        corpus.setdefault(key, {"id": key, "title": title, "text": text})
                        candidate_ids.append(key)
                    candidate_pools.append({"id": qid, "document_ids": list(dict.fromkeys(candidate_ids))})
                write_jsonl(out / f"{split}.questions.jsonl", questions)
                write_jsonl(out / f"{split}.labels.jsonl", labels)
                write_jsonl(out / f"{split}.candidate_pools.jsonl", candidate_pools)
            write_jsonl(out / "corpus.jsonl", corpus.values())
            info = {"test_n": len(test), "dev_n": len(dev), "corpus_n": len(corpus),
                    "base_ids": [x["question_id"] for x in base], "extra_ids": selected,
                    "dev_ids": [x["question_id"] for x in dev],
                    "train_exclusion_ids_sha256": digest(sorted(excluded_ids)),
                    "test_hops": dict(Counter(re.match(r"\dhop", x["question_id"])[0]
                                              for x in test if re.match(r"\dhop", x["question_id"]))) }
            info["files"] = {p.name: file_digest(p) for p in sorted(out.glob("*.jsonl"))}
            report["datasets"][dataset] = info
            write_json(out / "manifest.json", info)
    write_json(output / "manifest.json", report)
    return report
