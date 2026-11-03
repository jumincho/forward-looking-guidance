"""Run the fixed Qwen3-8B experiment recipes."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    registry = json.loads((ROOT / 'recipes.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'score', 'install-runtime', 'fetch-models'])
    parser.add_argument('recipe', choices=sorted(registry))
    args = parser.parse_args(sys.argv[1:3])
    extra = sys.argv[3:]
    entry = {'run': 'run.py', 'score': 'score.py',
             'install-runtime': 'install_runtime.py', 'fetch-models': 'fetch_models.py'}[args.action]
    directory = ROOT / registry[args.recipe]['path']
    if args.action in ['install-runtime', 'fetch-models'] and not (directory / entry).is_file():
        directory = ROOT / registry['stride']['path']
    return subprocess.call([sys.executable, str(directory / entry), *extra])

if __name__ == '__main__':
    raise SystemExit(main())
