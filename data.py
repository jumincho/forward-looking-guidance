"""Prepare question splits, candidate pools, or Contriever/FAISS indexes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent / 'recipes/stride'
sys.path.insert(0, str(ROOT))
if len(sys.argv) > 1 and sys.argv[1] == 'index' and '--help' not in sys.argv:
    from runtime_loader import load_cuda_runtime
    load_cuda_runtime()
sys.path.insert(0, str(ROOT / 'code'))
from flg.cli import main as core_main


def prepare_inputs(archive, output, *, seed=42, exclude_train_n=10000,
                   extra_n=500, candidate_group_size=32):
    if candidate_group_size < 1:
        raise ValueError('Candidate group size must be positive')
    from candidate_pools import main as group_main
    from flg.data import prepare
    from flg.io import file_digest, write_json
    output = Path(output)
    report = prepare(archive, output, seed=seed,
                     exclude_train_n=exclude_train_n, extra_n=extra_n)
    for dataset, info in report['datasets'].items():
        for split in ('test', 'dev'):
            pool = output / dataset / f'{split}.candidate_pools.jsonl'
            group_main(['--input', str(pool), '--output', str(pool),
                        '--group-size', str(candidate_group_size)])
            info['files'][pool.name] = file_digest(pool)
        info['candidate_group_size'] = candidate_group_size
        info['candidate_group_order'] = 'sha256_id'
        write_json(output / dataset / 'manifest.json', info)
    report['candidate_group_size'] = candidate_group_size
    write_json(output / 'manifest.json', report)
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'prepare':
        parser = argparse.ArgumentParser(prog='data.py prepare', description=__doc__)
        parser.add_argument('--archive', required=True)
        parser.add_argument('--output', required=True)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--exclude-train-n', type=int, default=10000)
        parser.add_argument('--extra-n', type=int, default=500)
        settings = json.loads((ROOT / 'settings.json').read_text())
        parser.add_argument('--candidate-group-size', type=int,
                            default=settings.get('candidate_group_size', 1))
        args = parser.parse_args(argv[1:])
        report = prepare_inputs(**vars(args))
        print(json.dumps({dataset: {key: info[key] for key in
                         ('test_n', 'dev_n', 'corpus_n')}
                         for dataset, info in report['datasets'].items()}, sort_keys=True))
        return
    if argv and argv[0] not in ('index', '--help', '-h'):
        raise SystemExit('Use prepare or index; execution uses reproduce.py.')
    core_main(argv)

if __name__ == '__main__':
    main()
