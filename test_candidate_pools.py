import copy
import tempfile
import unittest
from pathlib import Path

from candidate_pools import group_candidates, main


class CandidatePools(unittest.TestCase):
    def test_grouping_is_order_independent_and_retains_the_document_universe(self):
        rows = [{"id": f"q{i}", "document_ids": [f"d{i}", "shared"]} for i in range(5)]
        before = copy.deepcopy(rows)
        grouped = group_candidates(rows, 2)
        self.assertEqual(rows, before)
        self.assertEqual({r["id"]: r["document_ids"] for r in grouped},
                         {r["id"]: r["document_ids"] for r in group_candidates(rows[::-1], 2)})
        self.assertEqual({d for r in rows for d in r["document_ids"]},
                         {d for r in grouped for d in r["document_ids"]})
        original = {r["id"]: set(r["document_ids"]) for r in rows}
        for row in grouped:
            self.assertLessEqual(original[row["id"]], set(row["document_ids"]))
        self.assertEqual(sum(len(r["document_ids"]) == 2 for r in grouped), 1)

    def test_invalid_input_cannot_replace_an_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "input.jsonl", Path(directory) / "output.jsonl"
            source.write_text('{"id":"a","document_ids":["d","d"]}\n')
            target.write_text("preserved\n")
            with self.assertRaises(ValueError):
                main(["--input", str(source), "--output", str(target), "--group-size", "2"])
            self.assertEqual(target.read_text(), "preserved\n")
        with self.assertRaises(ValueError):
            group_candidates([{"id": "a", "document_ids": ["d"]}] * 2, 2)
        with self.assertRaises(ValueError):
            group_candidates([{"id": "a", "document_ids": ["d"], "answer": "unused"}], 2)


if __name__ == "__main__":
    unittest.main()
