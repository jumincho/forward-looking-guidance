"""
Efficient multi-job worker: load Qwen3-8B ONCE on one GPU, then run many
(method, dataset, fg) jobs for one shard, reusing one Contriever encoder across datasets.

Driven by launch_all.py. Jobs are read from a JSON file:
  [{"method": "stride", "dataset": "musique", "fg": "none", "run_name": "...",
    "top_k": 3, "max_iter": 5}, ...]
"""
from __future__ import annotations

import os, sys, json, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (LLMEngine, Retriever, load_encoder, read_jsonl, write_jsonl,
                    test_path, OUTPUT_DIR, STRIDE_TOPK, ORDINARY_TOPK)
from run import FG_PRESETS


def dispatch(method, engine, retriever, rows, fg, top_k, max_iter, temperature):
    if method == "stride":
        from methods.stride import run_stride
        return run_stride(engine, retriever, rows, top_k=top_k, max_iter=max_iter,
                          temperature=temperature, **FG_PRESETS[fg])
    if method == "dualrag":
        from methods.dualrag import run_dualrag
        return run_dualrag(engine, retriever, rows, top_k=top_k, max_iter=max_iter,
                           temperature=temperature, fg=(fg != "none"))
    if method == "selfask":
        from methods.selfask import run_selfask
        return run_selfask(engine, retriever, rows, top_k=top_k, max_iter=max_iter,
                           temperature=temperature, fg=(fg != "none"))
    if method in ("cot", "ragcot", "iterretgen", "genground"):
        from methods.baselines import run_baseline
        return run_baseline(method, engine, retriever, rows, top_k=top_k,
                            max_iter=max_iter, temperature=temperature)
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_iter", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    jobs = json.loads(Path(args.jobs).read_text())
    # sort by dataset so the retriever index is loaded/reused contiguously
    jobs.sort(key=lambda j: (j["dataset"], j["method"], j["fg"]))

    t0 = time.time()
    engine = LLMEngine(seed=args.seed)
    encoder = load_encoder()
    print(f"[shard {args.shard}] engine+encoder ready in {time.time()-t0:.0f}s", flush=True)

    retr_cache: dict[str, Retriever] = {}
    rows_cache: dict[str, list] = {}

    for job in jobs:
        method, dataset, fg = job["method"], job["dataset"], job["fg"]
        run_name = job.get("run_name") or f"{method}_{dataset}_{fg}"
        out_path = OUTPUT_DIR / run_name / f"shard{args.shard}.jsonl"
        if out_path.exists() and not job.get("overwrite"):
            print(f"[shard {args.shard}] skip existing {run_name}", flush=True)
            continue

        if dataset not in rows_cache:
            r = read_jsonl(test_path(dataset))
            if args.limit > 0:
                r = r[: args.limit]
            rows_cache[dataset] = r
        rows = rows_cache[dataset][args.shard :: args.num_shards]

        need_retr = method in ("stride", "dualrag", "selfask", "ragcot",
                               "iterretgen", "genground")
        retriever = None
        if need_retr:
            if dataset not in retr_cache:
                retr_cache[dataset] = Retriever(dataset, encoder=encoder)
            retriever = retr_cache[dataset]

        top_k = job.get("top_k")
        if not top_k or top_k <= 0:
            top_k = STRIDE_TOPK.get(dataset, 3) if method == "stride" else ORDINARY_TOPK
        max_iter = job.get("max_iter", args.max_iter)
        temperature = job.get("temperature", args.temperature)

        t1 = time.time()
        try:
            out = dispatch(method, engine, retriever, rows, fg, top_k, max_iter, temperature)
        except Exception as e:
            import traceback
            print(f"[shard {args.shard}] !! {run_name} FAILED: {e}", flush=True)
            traceback.print_exc()
            continue
        write_jsonl(out_path, out)
        dt = time.time() - t1
        print(f"[shard {args.shard}] {run_name}: {len(out)} q in {dt:.0f}s "
              f"({dt/max(1,len(out)):.2f}s/q)", flush=True)

    print(f"[shard {args.shard}] ALL JOBS DONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
