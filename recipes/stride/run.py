"""Run STRIDE with shared plans and stage guidance."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

from runtime_loader import load_cuda_runtime

if os.environ.get("FLG_CUDA12_LIBRARY"):
    load_cuda_runtime()

from flg import cli
from flg.backends import CachedBackend, VLLMBackend
from flg.io import digest, file_digest, read_jsonl, write_json, write_jsonl
from flg.pipeline import RunConfig, Stride
from planner_adapter import PlannerView
from canonical_answer import format_answer
from supplied_plans import supply_plans

METHODS = ('stride', 'stride_retrieval', 'stride_extraction', 'stride_reasoning', 'stride_fg')
DATASETS = ("2wikimultihopqa", "hotpotqa", "musique")

def methods_for(dataset):
    return METHODS if dataset == "musique" else ("stride", "stride_fg")


def validate_questions(path: Path, shard: int, shards: int) -> list[dict]:
    questions = read_jsonl(path)
    if any(set(q) != {"id", "question"} or not isinstance(q["question"], str) for q in questions):
        raise ValueError("Input must contain only id and question fields")
    if len({str(q["id"]) for q in questions}) != len(questions):
        raise ValueError("Duplicate question IDs")
    assigned = questions[shard::shards]
    if not assigned:
        raise ValueError("Empty shard")
    return assigned


def runtime_check(settings: dict, allow_variation: bool) -> dict:
    cuda12 = load_cuda_runtime()
    import torch
    import vllm
    import transformers
    import numpy
    import faiss

    actual = dict(python=platform.python_version(), torch=torch.__version__,
                  vllm=vllm.__version__, transformers=transformers.__version__,
                  numpy=numpy.__version__, faiss=faiss.__version__)
    expected = settings["measured_runtime"]
    differences = {k: {"expected": v, "actual": actual[k]} for k, v in expected.items()
                   if (not actual[k].startswith(v + ".") if k == "python" else
                       actual[k] != (v.split("+")[0] if k == "vllm" else v))}
    if differences and not allow_variation:
        raise ValueError("Runtime differs from the measured recipe: " + json.dumps(differences))
    if not hasattr(faiss, "StandardGpuResources"):
        raise ValueError("This recipe requires native FAISS-GPU")
    return dict(versions=actual, differences=differences, cuda12_runtime=cuda12,
                variation_explicitly_allowed=allow_variation)


def verify_model_receipt(path: Path, expected: dict) -> dict:
    receipt_path = path / "source_revision.json"
    if not receipt_path.is_file():
        raise ValueError(f"Missing pinned model receipt: {receipt_path}; see the recipe README")
    receipt = json.loads(receipt_path.read_text())
    if any(receipt.get(k) != value for k, value in expected.items()):
        raise ValueError(f"Pinned model revision differs: {path}")
    return dict(receipt=receipt, receipt_sha256=file_digest(receipt_path))


def configure_engine_compatibility(settings: dict) -> None:
    if settings["remove_tp2_scheduler_kwargs"]:
        import vllm
        original_llm = vllm.LLM

        def compatible_llm(**kwargs):
            kwargs.pop("max_num_batched_tokens", None)
            kwargs.pop("disable_custom_all_reduce", None)
            kwargs.update(enable_chunked_prefill=True, max_num_batched_tokens=2048)
            return original_llm(**kwargs)

        vllm.LLM = compatible_llm


def shared_plans(questions: list[dict], path: Path, backend, batch_size: int) -> list[dict]:
    existing = read_jsonl(path) if path.exists() else []
    by_id = {str(q["id"]): q for q in questions}
    known = {str(row["id"]) for row in existing}
    if len(known) != len(existing) or known - by_id.keys():
        raise ValueError("Duplicate or unassigned shared-plan IDs")
    for row in existing:
        if row["question_sha256"] != digest(by_id[str(row["id"])]["question"]):
            raise ValueError("A stored plan belongs to a different question")
    pending = [q for q in questions if str(q["id"]) not in known]
    runner = Stride(backend, None, RunConfig(seed=42))
    for offset in range(0, len(pending), batch_size):
        existing.extend(runner.plan(pending[offset:offset + batch_size]))
        write_jsonl(path, existing)
    return existing


def execution_command(args, settings: dict, dataset: str, seed: int, plans: Path) -> list[str]:
    inputs = Path(args.data_root) / dataset
    return ["run", "--questions", str(inputs / f"{args.split}.questions.jsonl"),
            "--candidate-pools", str(inputs / f"{args.split}.candidate_pools.jsonl"),
            "--index", str(Path(args.index_root) / dataset),
            "--encoder", str(Path(args.encoder_dir).resolve()),
            "--model", str(Path(args.model_dir).resolve()), "--backend", "vllm", "--tensor-parallel", str(settings["tensor_parallel"]),
            "--top-k", "5" if dataset == "hotpotqa" else "3",
            "--methods", ",".join(methods_for(dataset)),
            "--output", str(Path(args.output) / dataset / str(seed)),
            "--batch-size", str(settings["batch_size"]), "--seed", str(seed),
            "--memory-fraction", str(settings["memory_fraction"]),
            "--dtype", settings["dtype"], "--max-model-len", str(settings["max_model_len"]),
            "--enforce-eager", "--plans", str(plans),
            "--shard-index", str(args.shard), "--num-shards", str(args.shards),
            "--retriever-device", settings["retriever_device"],
            "--faiss-device", settings["faiss_device"],
            "--encoder-batch-size", str(settings["encoder_batch_size"])]


def aggregate(output: Path, datasets: list[str], settings: dict, assigned: dict) -> dict:
    counts = {}
    for dataset in datasets:
        counts[dataset] = {}
        questions = {str(q['id']): q['question'] for q in assigned[dataset]}
        for seed in settings['seeds']:
            counts[dataset][str(seed)] = {}
            for method in methods_for(dataset):
                rows = read_jsonl(output / dataset / str(seed) / f'{method}.jsonl')
                if len(rows) != len(questions) or {str(row['id']) for row in rows} != set(questions):
                    raise ValueError('Incomplete or duplicate seeded trajectory')
                formatted = [dict(id=row['id'], answer=format_answer(questions[str(row['id'])], row['answer']), status=row['status']) for row in rows]
                write_jsonl(output / dataset / str(seed) / f'{method}.canonical.jsonl', formatted)
                counts[dataset][str(seed)][method] = len(formatted)
    return counts


def run_isolated_datasets(args, settings: dict, assigned: dict, manifest: dict) -> dict:
    """Give each dataset its own model process and retain the usual output layout."""
    output = Path(args.output).resolve()
    for dataset in args.datasets:
        part = output / ".dataset_processes" / dataset
        child_manifest_path = part / "recipe_manifest.json"
        child_complete_path = part / "recipe_complete.json"
        if not child_complete_path.exists():
            command = [sys.executable, str(ROOT / "run.py"), "--datasets", dataset,
                       "--split", args.split, "--shard", str(args.shard),
                       "--shards", str(args.shards), "--output", str(part)]
            for name in ("data_root", "index_root", "model_dir", "encoder_dir"):
                command += ["--" + name.replace("_", "-"), str(Path(getattr(args, name)).resolve())]
            if args.plans_root:
                command += ["--plans-root", str(Path(args.plans_root).resolve())]
            if args.allow_runtime_variation:
                command.append("--allow-runtime-variation")
            if part.exists():
                command.append("--resume")
            subprocess.run(command, check=True)
        if not child_complete_path.is_file() or not child_manifest_path.is_file():
            raise ValueError("Missing dataset completion receipt: " + dataset)
        child = json.loads(child_manifest_path.read_text())
        complete = json.loads(child_complete_path.read_text())
        if not complete.get("complete") or complete.get("manifest_sha256") != file_digest(child_manifest_path):
            raise ValueError("Dataset completion receipt differs: " + dataset)
        expected = dict(manifest, datasets=[dataset], inputs={dataset: manifest["inputs"][dataset]})
        if child != expected:
            raise ValueError("Dataset process manifest differs: " + dataset)
        source, target = part / dataset, output / dataset
        if source.is_dir() and not target.exists():
            source.rename(target)
        elif source.exists() or not target.is_dir():
            raise ValueError("Ambiguous or missing dataset output: " + dataset)
        counts = aggregate(output, [dataset], settings, {dataset: assigned[dataset]})
        if complete.get("canonical_predictions") != counts:
            raise ValueError("Dataset completion counts differ: " + dataset)
    return aggregate(output, args.datasets, settings, assigned)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "index-root", "model-dir", "encoder-dir", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--plans-root", help="Optional fixed plans: DATASET/plans.jsonl")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-runtime-variation", action="store_true",
                        help="Record a different runtime as a separate, unverified transport")
    args = parser.parse_args(argv)
    if not 0 <= args.shard < args.shards or len(set(args.datasets)) != len(args.datasets):
        parser.error("Invalid or duplicate assignment")
    settings = json.loads((ROOT / "settings.json").read_text())
    for name, expected_hash in settings["postprocessing_sha256"].items():
        if file_digest(ROOT / name) != expected_hash:
            raise ValueError("Postprocessing source differs")
    if cli.source_fingerprint() != settings["source_sha256"]:
        raise ValueError("Frozen scientific source differs")
    for name, value in {"VLLM_USE_V1": "0", "VLLM_ATTENTION_BACKEND": "XFORMERS",
                        "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TOKENIZERS_PARALLELISM": "false",
                        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}.items():
        if name in os.environ and os.environ[name] != value:
            raise ValueError(f"Required environment setting: {name}={value}")
        os.environ[name] = value
    assigned = {}; inputs = {}
    for dataset in args.datasets:
        directory = Path(args.data_root) / dataset
        questions = directory / f"{args.split}.questions.jsonl"
        assigned[dataset] = validate_questions(questions, args.shard, args.shards)
        index = Path(args.index_root) / dataset / "manifest.json"
        index_manifest = json.loads(index.read_text())
        encoder = index_manifest["encoder"]
        if (encoder.get("passage_input") != "title_body" or encoder.get("pooling") != "masked_mean"
                or encoder.get("normalize") is not True or encoder.get("title_repetitions", 1) != 1):
            raise ValueError("Index must use normalized masked mean and one title plus body")
        inputs[dataset] = dict(questions_sha256=file_digest(questions),
            candidates_sha256=file_digest(directory / f"{args.split}.candidate_pools.jsonl"),
            index_manifest_sha256=file_digest(index), assigned_ids=[q["id"] for q in assigned[dataset]])
        if args.plans_root:
            supplied = Path(args.plans_root) / dataset / "plans.jsonl"
            inputs[dataset]["supplied_plans_sha256"] = file_digest(supplied)

    models = {"model": verify_model_receipt(Path(args.model_dir), settings["model"]),
              "encoder": verify_model_receipt(Path(args.encoder_dir), settings["encoder"])}
    runtime = runtime_check(settings, args.allow_runtime_variation)
    configure_engine_compatibility(settings)
    manifest = dict(settings=settings, inputs=inputs, models=models, runtime=runtime,
                    shard=args.shard, shards=args.shards, split=args.split, datasets=args.datasets,
                    recipe_files={p.name: file_digest(p) for p in (ROOT / "run.py", ROOT / "planner_adapter.py",
                                                                 ROOT / "settings.json",
                                                                 ROOT / "runtime_loader.py", ROOT / "canonical_answer.py", ROOT / "supplied_plans.py")})
    output = Path(args.output)
    manifest_path = output / "recipe_manifest.json"
    if output.exists():
        if not args.resume:
            raise ValueError("Output exists; use a fresh path or explicitly --resume")
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume manifest differs")
    else:
        output.mkdir(parents=True)
        write_json(manifest_path, manifest)
    if len(args.datasets) > 1:
        counts = run_isolated_datasets(args, settings, assigned, manifest)
        write_json(output / "recipe_complete.json", dict(complete=True, canonical_predictions=counts,
            manifest_sha256=file_digest(manifest_path), labels_read=False,
            budget="One shared plan plus five execution trajectories per question and method; report the fixed-seed mean"))
        return
    engine = None

    def get_engine():
        nonlocal engine
        if engine is None:
            engine = VLLMBackend(str(Path(args.model_dir).resolve()), memory_fraction=settings["memory_fraction"],
                enforce_eager=True, tensor_parallel=settings["tensor_parallel"], max_num_seqs=settings["batch_size"], dtype=settings["dtype"],
                max_model_len=settings["max_model_len"])
        return engine

    cli.make_backend = lambda options: CachedBackend(get_engine(), options.cache)
    for dataset in args.datasets:
        directory = output / dataset
        directory.mkdir(exist_ok=True)
        plan_path = directory / "shared_plans.jsonl"
        sampling = settings["planning_sampling"][dataset]
        if args.plans_root:
            supply_plans(Path(args.plans_root) / dataset / "plans.jsonl",
                read_jsonl(Path(args.data_root) / dataset / f"{args.split}.questions.jsonl"),
                assigned[dataset], plan_path, sampling)
        else:
            planner = CachedBackend(PlannerView(get_engine(), sampling["top_p"], sampling["top_k"]),
                                    directory / "planning.sqlite")
            shared_plans(assigned[dataset], plan_path, planner, settings["batch_size"])
        for seed in settings["seeds"]:
            cli.main(execution_command(args, settings, dataset, seed, plan_path))
    counts = aggregate(output, args.datasets, settings, assigned)
    write_json(output / "recipe_complete.json", dict(complete=True, canonical_predictions=counts,
        manifest_sha256=file_digest(manifest_path), labels_read=False,
        budget="One shared plan plus five execution trajectories per question and method; report the fixed-seed mean"))


if __name__ == "__main__":
    main()
