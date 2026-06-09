"""
Efficient matrix launcher: ONE engine load per GPU, all jobs run as one shard each.
Spawns one worker per visible GPU, then merges + evaluates each job.

  python launch_all.py main --aliases
  python launch_all.py ablation
  python launch_all.py all --aliases
  python launch_all.py smoke
"""
from __future__ import annotations

import os, sys, json, time, argparse, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import OUTPUT_DIR, read_jsonl, write_jsonl
from evaluate import score, score_by_hop, fmt

VENV_PY = os.environ.get("WORKER_PYTHON", "python")
DATASETS = ["2wikimultihopqa", "hotpotqa", "musique"]
BASE_ENV = {**os.environ, 
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_LOGGING_LEVEL": "WARNING",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def job(method, dataset, fg):
    return {"method": method, "dataset": dataset, "fg": fg,
            "run_name": f"{method}_{dataset}_{fg}"}


def build_jobs(group):
    jobs = []
    if group in ("main", "all"):
        for ds in DATASETS:
            for m in ["cot", "ragcot", "selfask", "iterretgen", "genground", "dualrag", "stride"]:
                jobs.append(job(m, ds, "none"))
            jobs.append(job("selfask", ds, "extract_reason"))
            jobs.append(job("dualrag", ds, "extract_reason"))
            jobs.append(job("stride", ds, "extract_reason"))
    if group in ("ablation", "all"):
        for fg in ["retrieve", "extract", "reason", "extract_reason"]:
            jobs.append(job("stride", "musique", fg))
    if group == "smoke":
        jobs = [job("stride", "musique", "none"), job("stride", "musique", "extract_reason")]
    # dedup preserving order
    seen, out = set(), []
    for j in jobs:
        k = (j["method"], j["dataset"], j["fg"])
        if k not in seen:
            seen.add(k); out.append(j)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group", choices=["smoke", "main", "ablation", "all", "custom"])
    ap.add_argument("--custom_jobs", default=None,
                    help="path to JSON job list (used when group=custom)")
    ap.add_argument("--num_gpus", type=int, default=len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")))
    ap.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--aliases", action="store_true")
    ap.add_argument("--only", default=None, help="comma filter on methods")
    ap.add_argument("--tag", default="jobs")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_iter", type=int, default=5)
    args = ap.parse_args()

    if args.group == "custom":
        jobs = json.loads(Path(args.custom_jobs).read_text())
    else:
        jobs = build_jobs(args.group)
    if args.only:
        keep = set(args.only.split(","))
        jobs = [j for j in jobs if j["method"] in keep]
    gpus = args.gpus.split(",")[: args.num_gpus]
    jobs_file = OUTPUT_DIR / f"_{args.tag}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs_file.write_text(json.dumps(jobs, indent=2))
    print(f"{len(jobs)} jobs x {len(gpus)} GPUs; jobs file {jobs_file}", flush=True)

    t0 = time.time()
    procs = []
    for i, g in enumerate(gpus):
        env = dict(BASE_ENV); env["CUDA_VISIBLE_DEVICES"] = g
        cmd = [VENV_PY, str(ROOT / "worker.py"), "--jobs", str(jobs_file),
               "--shard", str(i), "--num_shards", str(len(gpus)), "--limit", str(args.limit),
               "--temperature", str(args.temperature), "--max_iter", str(args.max_iter)]
        log = open(OUTPUT_DIR / f"_worker{i}.log", "w")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        print(f"launched worker {i} on GPU {g} (pid {procs[-1][0].pid})", flush=True)

    for i, (p, log) in enumerate(procs):
        p.wait(); log.close()
        print(f"worker {i} exit {p.returncode}", flush=True)

    # merge + evaluate each job
    print(f"\nworkers done in {time.time()-t0:.0f}s. evaluating ...\n", flush=True)
    results = []
    for j in jobs:
        rn = j["run_name"]; rd = OUTPUT_DIR / rn
        merged = []
        for i in range(len(gpus)):
            sp = rd / f"shard{i}.jsonl"
            if sp.exists():
                merged.extend(read_jsonl(sp))
        if not merged:
            print(f"  {rn}: NO OUTPUT"); continue
        write_jsonl(rd / "merged.jsonl", merged)
        m = score(merged, use_aliases=args.aliases)
        res = {"run_name": rn, **{k: j[k] for k in ("method", "dataset", "fg")},
               "metrics": m}
        if j["dataset"] == "musique":
            res["by_hop"] = {str(k): v for k, v in score_by_hop(merged, args.aliases).items()}
        (rd / "metrics.json").write_text(json.dumps(res, indent=2))
        results.append(res)
        print(f"  {rn:<40} {fmt(m)}")
    print(f"\nTOTAL {time.time()-t0:.0f}s. {len(results)} jobs scored.")


if __name__ == "__main__":
    main()
