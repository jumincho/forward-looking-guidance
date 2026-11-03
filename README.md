# Forward-Looking Guidance for Iterative RAG in Multi-Hop Question Answering

STRIDE and forward-looking guidance with Qwen3-8B and Contriever.

## Installation

Use Python 3.12 on Linux with NVIDIA GPUs.

```bash
python reproduce.py install-runtime stride --wheelhouse wheelhouse --target runtime
export PYTHONPATH="$PWD/runtime"
export FLG_CUDA12_LIBRARY="$PWD/runtime/cuda12_overlay/lib/libcudart.so.12"
python reproduce.py fetch-models stride --output-root models
```

## Data

Download the [IRCoT processed-data archive](https://drive.google.com/file/d/1t2BjJtsejSIUZI54PKObMFG6_wMMG3bC/view).

```bash
python data.py prepare --archive processed_data.zip --output data
for dataset in 2wikimultihopqa hotpotqa musique; do
  python data.py index --corpus data/$dataset/corpus.jsonl --output artifacts/indexes/$dataset --encoder models/facebook--contriever
done
```

## Usage

Run each dataset in a separate process. Launch shards 0 through 19 with distinct output directories and visible GPUs.

```bash
shard=0
for dataset in 2wikimultihopqa hotpotqa musique; do
  python reproduce.py run stride --datasets "$dataset" --data-root data --index-root artifacts/indexes --model-dir models/Qwen--Qwen3-8B --encoder-dir models/facebook--contriever --split test --shard "$shard" --shards 20 --output "runs/$dataset/shard$shard"
done
```

The MuSiQue command includes retrieval, extraction, reasoning, and joint guidance configurations. Plans are shared across configurations. Pass `--plans-root` to use existing plans, or `--resume` to continue an output directory.

```bash
python reproduce.py score stride --runs runs/*/shard* --data-root data --output scores.json
```
