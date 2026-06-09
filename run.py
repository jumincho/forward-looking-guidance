"""
Single-shard worker: run one METHOD on one DATASET shard with Qwen3-8B on one GPU.

Usage (driven by launch.py across the available GPUs):
  CUDA_VISIBLE_DEVICES=0 python run.py --method stride --dataset musique \
      --fg none --shard 0 --num_shards N --run_name stride_musique
"""
from __future__ import annotations

import os, sys, json, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    LLMEngine, Retriever, read_jsonl, write_jsonl, test_path, OUTPUT_DIR,
    STRIDE_TOPK, ORDINARY_TOPK,
)

FG_PRESETS = {
    "none":           dict(fg_extract=False, fg_reason=False, fg_retrieve=False),
    "extract":        dict(fg_extract=True,  fg_reason=False, fg_retrieve=False),
    "reason":         dict(fg_extract=False, fg_reason=True,  fg_retrieve=False),
    "retrieve":       dict(fg_extract=False, fg_reason=False, fg_retrieve=True),
    "extract_reason": dict(fg_extract=True,  fg_reason=True,  fg_retrieve=False),  # = +FG
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["stride", "dualrag", "selfask", "cot", "ragcot",
                             "iterretgen", "genground"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fg", default="none", choices=list(FG_PRESETS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--limit", type=int, default=-1, help="cap total questions")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_iter", type=int, default=5)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = read_jsonl(test_path(args.dataset))
    if args.limit > 0:
        rows = rows[: args.limit]
    rows = rows[args.shard :: args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] {args.method}/{args.dataset}/fg={args.fg} "
          f"-> {len(rows)} questions", flush=True)

    top_k = args.top_k if args.top_k > 0 else (
        STRIDE_TOPK.get(args.dataset, 3) if args.method in ("stride",) else ORDINARY_TOPK)

    t0 = time.time()
    engine = LLMEngine(seed=args.seed)
    need_retr = args.method in ("stride", "dualrag", "selfask", "ragcot",
                                "iterretgen", "genground")
    retriever = Retriever(args.dataset) if need_retr else None
    print(f"[shard {args.shard}] engine+retriever ready in {time.time()-t0:.0f}s", flush=True)

    t1 = time.time()
    if args.method == "stride":
        from methods.stride import run_stride
        fg = FG_PRESETS[args.fg]
        out = run_stride(engine, retriever, rows, top_k=top_k, max_iter=args.max_iter,
                         temperature=args.temperature, **fg)
    elif args.method == "dualrag":
        from methods.dualrag import run_dualrag
        out = run_dualrag(engine, retriever, rows, top_k=top_k, max_iter=args.max_iter,
                          temperature=args.temperature, fg=(args.fg != "none"))
    elif args.method == "selfask":
        from methods.selfask import run_selfask
        out = run_selfask(engine, retriever, rows, top_k=top_k, max_iter=args.max_iter,
                          temperature=args.temperature, fg=(args.fg != "none"))
    elif args.method in ("cot", "ragcot", "iterretgen", "genground"):
        from methods.baselines import run_baseline
        out = run_baseline(args.method, engine, retriever, rows, top_k=top_k,
                           max_iter=args.max_iter, temperature=args.temperature)
    else:
        raise ValueError(args.method)

    dt = time.time() - t1
    out_path = OUTPUT_DIR / args.run_name / f"shard{args.shard}.jsonl"
    write_jsonl(out_path, out)
    print(f"[shard {args.shard}] done {len(out)} in {dt:.0f}s "
          f"({dt/max(1,len(out)):.2f}s/q) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
