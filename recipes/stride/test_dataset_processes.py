"""Dataset process isolation, interruption recovery, and output integrity."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import run
from flg.io import file_digest, read_jsonl, write_json, write_jsonl


class DatasetProcesses(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.output = Path(self.work.name) / "output"
        self.output.mkdir()
        self.datasets = ["2wikimultihopqa", "hotpotqa", "musique"]
        self.settings = {"seeds": [0, 1]}
        self.assigned = {name: [{"id": name + "-a", "question": "Which city is named?"}]
                         for name in self.datasets}
        self.manifest = dict(settings=self.settings, inputs={name: {"hash": name} for name in self.datasets},
                             datasets=self.datasets, shard=0, shards=1, runtime={"version": "test"})
        self.args = SimpleNamespace(output=str(self.output), datasets=self.datasets, split="test",
            shard=0, shards=1, plans_root="plans", allow_runtime_variation=False,
            data_root="data", index_root="indexes", model_dir="model", encoder_dir="encoder")
        self.children = []

    def child(self, command, check):
        self.assertTrue(check)
        dataset = command[command.index("--datasets") + 1]
        output = Path(command[command.index("--output") + 1])
        self.children.append(dataset)
        manifest = dict(self.manifest, datasets=[dataset], inputs={dataset: self.manifest["inputs"][dataset]})
        write_json(output / "recipe_manifest.json", manifest)
        for seed in self.settings["seeds"]:
            for method in run.methods_for(dataset):
                write_jsonl(output / dataset / str(seed) / (method + ".jsonl"),
                            [dict(id=dataset + "-a", answer="Northhaven", status="ok", trace=[])])
        counts = run.aggregate(output, [dataset], self.settings, {dataset: self.assigned[dataset]})
        write_json(output / "recipe_complete.json", dict(complete=True, canonical_predictions=counts,
            manifest_sha256=file_digest(output / "recipe_manifest.json")))

    def invoke(self, side_effect=None):
        with patch.object(run.subprocess, "run", side_effect=side_effect or self.child):
            return run.run_isolated_datasets(self.args, self.settings, self.assigned, self.manifest)

    def test_separate_processes_keep_flat_dataset_outputs_and_resume_without_generation(self):
        counts = self.invoke()
        self.assertEqual(self.children, self.datasets)
        self.assertEqual(set(counts), set(self.datasets))
        self.assertEqual(read_jsonl(self.output / "musique/0/stride_fg.jsonl")[0]["answer"], "Northhaven")
        before = {str(p): file_digest(p) for p in self.output.rglob("*.jsonl")}
        self.invoke(side_effect=AssertionError("Completed dataset must not be regenerated"))
        self.assertEqual(before, {str(p): file_digest(p) for p in self.output.rglob("*.jsonl")})

    def test_interrupted_dispatch_keeps_completed_data_and_resumes_remaining_datasets(self):
        def interrupted(command, check):
            dataset = command[command.index("--datasets") + 1]
            if dataset == "hotpotqa":
                raise RuntimeError("Interrupted child")
            self.child(command, check)
        with self.assertRaisesRegex(RuntimeError, "Interrupted"):
            self.invoke(interrupted)
        original = file_digest(self.output / "2wikimultihopqa/0/stride.jsonl")
        self.children.clear()
        self.invoke()
        self.assertEqual(self.children, ["hotpotqa", "musique"])
        self.assertEqual(original, file_digest(self.output / "2wikimultihopqa/0/stride.jsonl"))

    def test_changed_child_manifest_is_rejected_before_consuming_outputs(self):
        def changed(command, check):
            self.child(command, check)
            output = Path(command[command.index("--output") + 1])
            value = copy.deepcopy(self.manifest)
            value["settings"]["seeds"] = [9]
            write_json(output / "recipe_manifest.json", value)
            complete = json.loads((output / "recipe_complete.json").read_text())
            complete["manifest_sha256"] = file_digest(output / "recipe_manifest.json")
            write_json(output / "recipe_complete.json", complete)
        with self.assertRaisesRegex(ValueError, "process manifest"):
            self.invoke(changed)

    def test_missing_completed_output_is_not_silently_regenerated(self):
        self.invoke()
        (self.output / "musique/0/stride_fg.jsonl").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.invoke(side_effect=AssertionError("Do not regenerate after a completion receipt"))


if __name__ == "__main__":
    unittest.main()
