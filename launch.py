"""
Orchestrator: shard one (method, dataset, fg) run across N GPUs (data parallel),
merge shard outputs, evaluate EM/F1/P/R (and per-hop for musique).

Example:
  python launch.py --method stride --dataset musique --fg none
  python launch.py --method stride --dataset musique --fg extract_reason
"""
from __future__ import annotations

import os, sys, json, time, argparse, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import OUTPUT_DIR, read_jsonl, write_jsonl  # noqa: E402
from evaluate import score, score_by_hop, fmt  # noqa: E402

VENV_PY = os.environ.get("WORKER_PYTHON", "python")
BASE_ENV = {
    **os.environ,
    "TOKENIZERS_PARALLELISM": "false",
    "VLLM_LOGGING_LEVEL": "WARNING",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fg", default="none")
    ap.add_argument("--num_gpus", type=int, default=len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")))
    ap.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_iter", type=int, default=5)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aliases", action="store_true",
                    help="alias-aware scoring (musique answer_aliases)")
    args = ap.parse_args()

    run_name = args.run_name or f"{args.method}_{args.dataset}_{args.fg}"
    gpus = args.gpus.split(",")[: args.num_gpus]
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    t0 = time.time()
    for i, g in enumerate(gpus):
        env = dict(BASE_ENV)
        env["CUDA_VISIBLE_DEVICES"] = g
        cmd = [VENV_PY, str(ROOT / "run.py"),
               "--method", args.method, "--dataset", args.dataset, "--fg", args.fg,
               "--shard", str(i), "--num_shards", str(len(gpus)),
               "--run_name", run_name, "--limit", str(args.limit),
               "--temperature", str(args.temperature), "--max_iter", str(args.max_iter),
               "--top_k", str(args.top_k), "--seed", str(args.seed)]
        log = open(run_dir / f"shard{i}.log", "w")
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((p, log))
        print(f"launched shard {i} on GPU {g} (pid {p.pid})", flush=True)

    rc = 0
    for i, (p, log) in enumerate(procs):
        p.wait()
        log.close()
        if p.returncode != 0:
            rc = p.returncode
            print(f"!! shard {i} exited with code {p.returncode}; see {run_dir}/shard{i}.log",
                  flush=True)

    # merge
    merged = []
    for i in range(len(gpus)):
        sp = run_dir / f"shard{i}.jsonl"
        if sp.exists():
            merged.extend(read_jsonl(sp))
    if not merged:
        print(f"FATAL: no shard outputs for {run_name}. Check logs in {run_dir}", flush=True)
        sys.exit(rc or 1)
    write_jsonl(run_dir / "merged.jsonl", merged)

    # evaluate
    m = score(merged, use_aliases=args.aliases)
    dt = time.time() - t0
    print(f"\n===== {run_name} =====")
    print(f"n={m['n']}  {fmt(m)}  ({dt:.0f}s)")
    result = {"run_name": run_name, "method": args.method, "dataset": args.dataset,
              "fg": args.fg, "metrics": m, "seconds": dt}
    if args.dataset == "musique":
        bh = score_by_hop(merged, use_aliases=args.aliases)
        print("per-hop:")
        for k, v in bh.items():
            print(f"  {k}: {fmt(v)}")
        result["by_hop"] = {str(k): v for k, v in bh.items()}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"saved -> {run_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
