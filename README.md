# Forward-Looking Guidance for Iterative RAG

Code for **Forward-Looking Guidance (FG)** in planning-based iterative retrieval-augmented
multi-hop question answering. A planner produces a reasoning plan `(o_1 … o_n)`; when the
executor resolves operation `o_i`, the *remaining* planned sub-questions `G_i = (o_{i+1} … o_n)`
are injected into the modules that form intermediate outputs (the Extractor and the Reasoner),
so those outputs are produced with later steps in mind. FG is implemented as a plug-in over
STRIDE, DualRAG, and Self-Ask.

Backbone: a local model via vLLM (default **Qwen3-8B**, extended reasoning disabled) or an
OpenAI model (e.g. `gpt-4o-mini`) via the API — selected with `MODEL_PATH`. Dense retrieval with
an unsupervised **Contriever + FAISS `IndexFlatIP`**, EM / token-F1 evaluation.

## Installation

```bash
pip install -r requirements.txt
```

Install a CUDA-matched PyTorch first, e.g. for CUDA 12.6:

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
```

## Layout

| Component | Module |
|-----------|--------|
| STRIDE (Meta-Planner → Supervisor → Extractor + Reasoner + Fallback Reasoner) | `methods/stride.py` |
| DualRAG (reasoning-augmented querying + progressive knowledge aggregation) | `methods/dualrag.py` |
| Self-Ask (iterative follow-up questions) | `methods/selfask.py` |
| CoT / RAG w/ CoT / Iter-RetGen / GenGround | `methods/baselines.py` |
| vLLM engine, retrieval, metrics, plan + FG helpers | `common.py` |
| Contriever encoder + dense FAISS retriever | `retrieval/` |
| Prompts (Meta-Planner / Supervisor / Extractor / Reasoner / Fallback) | `prompts/` |
| Build the test sets + per-dataset corpora | `prepare_data.py` |
| Build the Contriever + FAISS index | `build_indexes.py` |
| Shard the matrix over GPUs, merge, evaluate | `launch_all.py`, `worker.py`, `run.py` |
| Metrics | `evaluate.py` |

`+FG` for STRIDE injects `G_i` into the Extractor and Reasoner; for DualRAG into the
knowledge-summarization step; for Self-Ask into the intermediate-answer step.

## Usage

`prepare_data.py` builds the test sets and per-dataset retrieval corpora for 2WikiMultiHopQA,
HotpotQA, and MuSiQue from the source dataset dumps placed under `raw/`. Each row is
`{id, question, answer, answer_aliases, hop, type, contexts:[…]}`.

```bash
python prepare_data.py                         # test sets + corpora
python build_indexes.py                        # Contriever + FAISS index per dataset
python launch_all.py all --aliases --tag run   # method × dataset × {baseline, +FG} matrix
```

`launch_all.py` shards the test sets across the visible GPUs (one engine per GPU). Settings live
in `common.py`; `CONTRIEVER_PATH` selects the retriever and `MODEL_PATH` the backbone (a local
vLLM model or an OpenAI model id such as `gpt-4o-mini`).
