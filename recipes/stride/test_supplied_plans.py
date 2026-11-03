import copy
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
from flg.io import digest, read_jsonl, write_jsonl
from supplied_plans import supply_plans
from flg.pipeline import seed_for


class SuppliedPlanIntegrity(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.questions = [{'id': str(i), 'question': f'Invented question {i}?'} for i in range(5)]
        self.rows = [dict(id=q['id'], question_sha256=digest(q['question']),
            plan='o1 = Retrieve the named attribute', plan_sha256=digest('o1 = Retrieve the named attribute'),
            seed=seed_for(42,q['id'],'planner'), backend=dict(top_p=.95, top_k=20, temperature=1, max_tokens=512, enable_thinking=False))
            for q in self.questions]
        self.source = self.root / 'plans.jsonl'
        write_jsonl(self.source, self.rows)
        self.sampling = dict(top_p=.95, top_k=20)

    def call(self, rows=None, questions=None, output='out.jsonl'):
        if rows is not None:
            write_jsonl(self.source, rows)
        return supply_plans(self.source, questions or self.questions,
                            self.questions[1::2], self.root / output, self.sampling)

    def test_uneven_shard_preserves_exact_records(self):
        selected = self.call()
        self.assertEqual(selected, [self.rows[1], self.rows[3]])
        self.assertEqual(read_jsonl(self.root / 'out.jsonl'), selected)
        self.assertEqual(self.call(), selected)

    def test_duplicate_or_missing_question_plan_is_rejected(self):
        for rows in [self.rows[:-1], self.rows + [self.rows[0]]]:
            with self.assertRaises(ValueError):
                self.call(rows)

    def test_plan_and_question_tampering_is_rejected(self):
        for key in ['plan_sha256', 'question_sha256']:
            rows = copy.deepcopy(self.rows)
            rows[0][key] = 'altered'
            with self.assertRaises(ValueError):
                self.call(rows)

    def test_seed_decoder_and_thinking_drift_is_rejected(self):
        for key, value in [('seed', 43), ('top_p', .8), ('top_k', 50), ('temperature', 0),
                           ('max_tokens', 1024), ('enable_thinking', True)]:
            rows = copy.deepcopy(self.rows)
            target = rows[0] if key == 'seed' else rows[0]['backend']
            target[key] = value
            with self.assertRaises(ValueError):
                self.call(rows)

    def test_existing_checkpoint_must_match_selected_shard(self):
        self.call()
        write_jsonl(self.root / 'out.jsonl', [self.rows[0]])
        with self.assertRaises(ValueError):
            self.call()


if __name__ == '__main__':
    unittest.main()
