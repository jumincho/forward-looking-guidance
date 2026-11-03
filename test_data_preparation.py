import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from data import prepare_inputs
from flg.data import DATASETS, prepare
from flg.io import file_digest


class PreparationTests(unittest.TestCase):
    def test_grouping_preserves_questions_labels_corpus_and_manifest_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'inputs.zip'
            with zipfile.ZipFile(archive, 'w') as zipped:
                for dataset in DATASETS:
                    for split, count in [('test_subsampled', 500), ('dev_subsampled', 1), ('train', 3)]:
                        rows = [dict(question_id=f'{dataset}-{split}-{i}',
                                     question_text=f'Where is museum {i}?',
                                     answers_objects=[dict(spans=[f'City {i}'])],
                                     contexts=[dict(title=f'Museum {split} {i}',
                                                    paragraph_text=f'The museum is in City {i}.')])
                                for i in range(count)]
                        zipped.writestr(f'processed_data/{dataset}/{split}.jsonl',
                                        ''.join(json.dumps(row) + '\n' for row in rows))
            plain, grouped = root / 'plain', root / 'grouped'
            prepare(archive, plain, seed=7, exclude_train_n=0, extra_n=2)
            report = prepare_inputs(archive, grouped, seed=7, exclude_train_n=0,
                                    extra_n=2, candidate_group_size=32)
            for dataset in DATASETS:
                for name in ('test.questions.jsonl', 'dev.questions.jsonl',
                             'test.labels.jsonl', 'dev.labels.jsonl', 'corpus.jsonl'):
                    self.assertEqual((plain/dataset/name).read_bytes(),
                                     (grouped/dataset/name).read_bytes())
                for split in ('test', 'dev'):
                    name = f'{split}.candidate_pools.jsonl'
                    original = {r['id']: set(r['document_ids']) for r in
                                map(json.loads, (plain/dataset/name).read_text().splitlines())}
                    expanded = {r['id']: set(r['document_ids']) for r in
                                map(json.loads, (grouped/dataset/name).read_text().splitlines())}
                    self.assertEqual(set(original), set(expanded))
                    self.assertTrue(all(original[qid] <= expanded[qid] for qid in original))
                    self.assertEqual(set.union(*original.values()), set.union(*expanded.values()))
                info = json.loads((grouped/dataset/'manifest.json').read_text())
                self.assertEqual(info, report['datasets'][dataset])
                self.assertTrue(all(file_digest(grouped/dataset/name) == digest
                                    for name, digest in info['files'].items()))

    def test_invalid_group_size_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'output'
            with self.assertRaises(ValueError):
                prepare_inputs('unused.zip', target, candidate_group_size=0)
            self.assertFalse(target.exists())
