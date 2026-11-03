"""Audit every fixed-seed trajectory and score the paired mean across all seeds."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
import numpy as np
from flg.io import digest, file_digest, read_jsonl, write_json
from flg.metrics import evaluate, score
from flg.parsing import answer_span, operations, ParseError
from canonical_answer import format_answer

METHODS = ('stride', 'stride_retrieval', 'stride_extraction', 'stride_reasoning', 'stride_fg')

def methods_for(dataset):
    return METHODS if dataset == "musique" else ("stride", "stride_fg")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_trace(row: dict, plan: dict) -> int:
    require(row["plan_sha256"] == plan["plan_sha256"], "Execution plan differs from the shared plan")
    require(row["plan"] == plan["plan"], "Execution plan text differs")
    try:
        ops = operations(row["plan"])
    except ParseError:
        require(row["status"] == "plan_parse_error", "Unreported plan parse failure")
        return 0
    dependencies = {op.qid: set(re.findall(r"\bQ\d+\b", op.text)) - {op.qid} for op in ops}
    solved = set(); before = set(); iteration = None; checked = 0
    for event in row["trace"]:
        if "output_tokens" in event:
            require(event["output_tokens"] <= 512, "Output token budget exceeded")
        if event["module"] in {"retrieval", "reasoner"}:
            require(event["iteration"] <= 5, "Iteration budget exceeded")
            if event["iteration"] != iteration:
                iteration, before = event["iteration"], set(solved)
            require(dependencies[event["qid"]] <= before, "Unresolved operation dependency")
            checked += 1
        if event["module"] == "reasoner":
            try:
                answer = answer_span(event["text"])
            except ParseError:
                answer = None
            if answer is not None:
                solved.add(event["qid"])
    return checked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="One output directory per shard")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=198042)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args(argv)
    require(100 <= args.bootstrap_repetitions <= 100000, "Bootstrap repetitions must be 100–100000")
    roots = [Path(value).resolve() for value in args.runs]
    require(len(set(roots)) == len(roots), "Duplicate run directory")
    manifests = [json.loads((root / "recipe_manifest.json").read_text()) for root in roots]
    settings = manifests[0]["settings"]; split = manifests[0]["split"]
    require(settings == json.loads((ROOT / "settings.json").read_text()), "Run settings differ from this frozen recipe")
    require(settings["aggregation"] == "canonical_fixed_seed_mean", "Expected fixed-seed mean")
    for name, expected_hash in settings["postprocessing_sha256"].items():
        require(file_digest(ROOT / name) == expected_hash, "Postprocessing source differs")
    for root, manifest in zip(roots, manifests):
        require(all(manifest["recipe_files"].get(name) == value for name, value in settings["postprocessing_sha256"].items()), "Run postprocessing source differs")
        require(manifest["settings"] == settings and manifest["split"] == split, "Mixed scientific recipes or splits")
        require(not manifest["runtime"]["differences"], "A runtime variation requires a separate analysis")
        complete = json.loads((root / "recipe_complete.json").read_text())
        require(complete["complete"] and complete["manifest_sha256"] == file_digest(root / "recipe_manifest.json"),
                "Missing or inconsistent completion receipt")
    datasets = sorted({dataset for manifest in manifests for dataset in manifest["datasets"]})
    results = {}; deltas = {}; usage = {}; audits = {}; input_hashes = {}
    for dataset in datasets:
        directory = Path(args.data_root) / dataset
        questions_path = directory / f"{split}.questions.jsonl"
        candidates_path = directory / f"{split}.candidate_pools.jsonl"
        labels_path = directory / f"{split}.labels.jsonl"
        questions = read_jsonl(questions_path)
        question_ids = [str(row["id"]) for row in questions]
        require(len(set(question_ids)) == len(question_ids), "Duplicate question ID")
        expected = set(question_ids)
        selected = [(root, manifest) for root, manifest in zip(roots, manifests) if dataset in manifest["datasets"]]
        shard_count = selected[0][1]["shards"]
        require(all(manifest["shards"] == shard_count for _, manifest in selected), "Mixed shard counts")
        require(len(selected) == shard_count and {m["shard"] for _, m in selected} == set(range(shard_count)),
                "Missing or duplicate shard")
        compact = {method: [[] for _ in settings["seeds"]] for method in methods_for(dataset)}
        usage[dataset] = {method: Counter() for method in methods_for(dataset)}; checked = 0
        for root, manifest in selected:
            recorded = manifest["inputs"][dataset]
            require(recorded["questions_sha256"] == file_digest(questions_path), "Question input hash differs")
            require(recorded["candidates_sha256"] == file_digest(candidates_path), "Candidate input hash differs")
            assigned = set(question_ids[manifest["shard"]::shard_count])
            require(set(map(str, recorded["assigned_ids"])) == assigned, "Shard assignment differs")
            plan_rows = read_jsonl(root / dataset / "shared_plans.jsonl")
            plans = {str(row["id"]): row for row in plan_rows}
            require(len(plan_rows) == len(assigned) and set(plans) == assigned, "Shared plans have incomplete or duplicate IDs")
            question_text = {str(row["id"]): row["question"] for row in questions}
            for plan in plans.values():
                require(plan["plan_sha256"] == digest(plan["plan"]) and
                        plan["question_sha256"] == digest(question_text[str(plan["id"])]), "Stored plan hash differs")
                require(plan["backend"]["top_p"] == settings["planning_sampling"][dataset]["top_p"] and
                        plan["backend"]["top_k"] == settings["planning_sampling"][dataset]["top_k"],
                        "Planning decoder differs")
            for position, seed in enumerate(settings["seeds"]):
                folder = root / dataset / str(seed)
                run = json.loads((folder / "manifest.json").read_text())
                require(run["source_sha256"] == settings["source_sha256"] and run["seed"] == seed,
                        "Execution source or seed differs")
                require(run["retrieval_scope"] == "question_candidates" and run["temperature"] == 1 and
                        run["max_tokens"] == 512 and run["max_iterations"] == 5, "Execution protocol differs")
                require(all(run["runtime"][key] == settings[key] for key in
                            ("dtype", "batch_size", "memory_fraction", "max_model_len", "retriever_device",
                             "faiss_device", "encoder_batch_size")), "Execution runtime differs")
                require(run["questions_sha256"] == file_digest(questions_path) and run["candidate_pools_sha256"] == file_digest(candidates_path), "Execution input hash differs")
                for method in methods_for(dataset):
                    rows = read_jsonl(folder / f"{method}.jsonl")
                    require(len(rows) == len(assigned) and {str(row["id"]) for row in rows} == assigned,
                            "Incomplete or duplicated trajectory")
                    for row in rows:
                        checked += audit_trace(row, plans[str(row["id"])])
                        for values in row["usage"].values():
                            for key in ("calls", "input_tokens", "output_tokens", "cached_calls", "cost_usd"):
                                usage[dataset][method]["execution_" + key] += values[key]
                    small = [dict(id=row["id"], answer=format_answer(question_text[str(row["id"])], row["answer"]), status=row["status"], trace=row["trace"]) for row in rows]
                    formatted = [{key: row[key] for key in ("id", "answer", "status")} for row in small]
                    require(formatted == read_jsonl(folder / f"{method}.canonical.jsonl"), "Stored canonical predictions differ")
                    compact[method][position].extend(formatted)
            for method in methods_for(dataset):
                for plan in plans.values():
                    usage[dataset][method]["planning_calls"] += 1
                    for key in ("input_tokens", "output_tokens", "cost_usd"):
                        usage[dataset][method]["planning_" + key] += plan["generation"].get(key, 0)
        # All canonical predictions for this dataset were generated and audited before labels.
        labels = read_jsonl(labels_path)
        require(set(str(row['id']) for row in labels) == expected, 'Labels differ from question split')
        results[dataset] = {}; per_question = {}
        for method in methods_for(dataset):
            per_seed = {str(seed): evaluate(rows, labels) for seed, rows in zip(settings['seeds'], compact[method])}
            first = next(iter(per_seed.values()))
            groups = [key for key, value in first.items() if isinstance(value, dict) and 'n' in value and 'em' in value]
            means = {group: {'n': first[group]['n'], **{key: float(np.mean([metric[group][key] for metric in per_seed.values()])) for key in ('em','f1','precision','recall')}} for group in groups}
            results[dataset][method] = dict(per_seed=per_seed, seed_mean=means)
            arrays = []
            for rows in compact[method]:
                by_id = {str(row['id']): row for row in rows}
                arrays.append(np.array([score(by_id[str(label['id'])]['answer'], label['answer'])['em'] for label in sorted(labels, key=lambda row: str(row['id']))]))
            per_question[method] = np.mean(arrays, axis=0)
        deltas[dataset] = {method: per_question[method] - per_question['stride'] for method in methods_for(dataset) if method != 'stride'}
        audits[dataset] = dict(checked_actions=checked, explicit_dependency_violations=0,
                               questions=len(expected), raw_predictions=len(expected) * len(methods_for(dataset)) * len(settings["seeds"]))
        input_hashes[dataset] = dict(questions=file_digest(questions_path), candidates=file_digest(candidates_path),
                                    labels=file_digest(labels_path))
    rng = np.random.default_rng(args.bootstrap_seed); draws = {}; effects = {}
    for dataset, stages in deltas.items():
        effects[dataset] = {}; draws[dataset] = {}
        baseline = results[dataset]["stride"]["seed_mean"]["overall"]["em"]
        for method, values in stages.items():
            sampled = np.concatenate([values[rng.integers(0, len(values), size=(min(1000, args.bootstrap_repetitions - offset), len(values)))].mean(axis=1)
                for offset in range(0, args.bootstrap_repetitions, 1000)])
            draws[dataset][method] = sampled
            effects[dataset][method] = dict(delta_em=float(values.mean()),
                relative_gain=float(values.mean()/baseline) if baseline else None,
                paired_ci95=np.quantile(sampled,[.025,.975]).tolist())
    report = dict(kind="measured", complete=True, settings=settings, results=results, effects=effects,
        balanced_macro_delta_ci95={method: np.quantile(np.stack([values[method] for values in draws.values()]).mean(axis=0), [.025, .975]).tolist() for method in METHODS if method != 'stride' and all(method in values for values in draws.values())},
        bootstrap=dict(seed=args.bootstrap_seed, repetitions=args.bootstrap_repetitions, unit="question"),
        audit=audits, logical_usage=usage, input_hashes=input_hashes,
        recipe_manifests={str(root): file_digest(root / "recipe_manifest.json") for root in roots},
        labels_read_only_by_scorer=True, aggregation="canonical_fixed_seed_mean", no_answer_voting=True, no_seed_selection=True, additional_model_calls_for_postprocessing=0, gpu_monetary_cost=None, scorer_sha256=file_digest(Path(__file__)))
    write_json(args.output, report)
    print(json.dumps(dict(complete=True, effects=effects), indent=2))


if __name__ == "__main__":
    main()
