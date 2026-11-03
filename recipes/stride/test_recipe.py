"""CPU-only checks for input isolation and interruption-safe shared planning."""
import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import run
from flg.backends import Response
from flg.io import read_jsonl, write_jsonl


class RecordedPlanner:
    identity = {"backend": "unit-test", "top_p": .95, "top_k": 20}

    def __init__(self):
        self.ids = []

    def generate(self, requests):
        self.ids.extend(r.sample_id for r in requests)
        return [Response("Q1: [Retrieve] Which city is given?\nQ2: [Answer] Return Q1.", 10, 10, 0)
                for _ in requests]


class RecipeInvariants(unittest.TestCase):
    def test_legacy_engine_adapter_preserves_context_and_removes_scheduler_override(self):
        recorded = []
        module = SimpleNamespace(LLM=lambda **kwargs: recorded.append(kwargs))
        with patch.dict("sys.modules", {"vllm": module}):
            run.configure_engine_compatibility({"remove_tp2_scheduler_kwargs": True})
            module.LLM(model="pinned-model", max_model_len=8192, max_num_batched_tokens=2048,
                       disable_custom_all_reduce=True, dtype="float16", max_num_seqs=32)
        self.assertEqual(recorded, [{"model": "pinned-model", "max_model_len": 8192,
                                    "dtype": "float16", "max_num_seqs": 32,
                                    "enable_chunked_prefill": True, "max_num_batched_tokens": 2048}])

    def test_question_input_rejects_reference_fields_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            write_jsonl(path, [{"id": "a", "question": "Where?", "answer": "Hidden"}])
            with self.assertRaises(ValueError):
                run.validate_questions(path, 0, 1)
            write_jsonl(path, [{"id": "a", "question": "Where?"}] * 2)
            with self.assertRaises(ValueError):
                run.validate_questions(path, 0, 1)

    def test_resume_retains_plans_and_rejects_changed_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plans.jsonl"
            questions = [{"id": str(i), "question": f"Synthetic question {i}?"} for i in range(3)]
            backend = RecordedPlanner()
            first = run.shared_plans(questions[:1], path, backend, 2)[0]
            completed = run.shared_plans(questions, path, backend, 2)
            self.assertEqual(backend.ids, ["0", "1", "2"])
            self.assertEqual(first, completed[0])
            run.shared_plans(questions, path, backend, 2)
            self.assertEqual(backend.ids, ["0", "1", "2"])
            changed = [dict(questions[0], question="Different question?")] + questions[1:]
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                run.shared_plans(changed, path, backend, 2)
            self.assertEqual(before, path.read_bytes())

    def test_incomplete_seed_cannot_produce_vote(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assigned = {"musique": [{"id": "a", "question": "One?"}, {"id": "b", "question": "Two?"}]}
            for seed in (42, 43):
                for method in run.METHODS:
                    rows = [{"id": "a", "answer": "Northhaven", "status": "ok", "trace": []}]
                    if seed == 42:
                        rows.append({"id": "b", "answer": None, "status": "unresolved", "trace": []})
                    write_jsonl(root / "musique" / str(seed) / (method + ".jsonl"), rows)
            with self.assertRaises(ValueError):
                run.aggregate(root, ["musique"], {"seeds": [42, 43]}, assigned)
            self.assertFalse((root / "musique/stride.voted.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
