# Retrieval Agent — AutoModelAdvisor

Code for the retrieval component of AutoModelAdvisor: given a **cold-start
dataset** (a dataset not in the ArtifactBench graph, so it has no edges and no
graph method can score it directly), recommend candidate models. Two variants
are provided; both share the same first steps — GPT-4o card summary, then a
voyage-3 embedding of the summary — and differ in how they use the graph.

| | Method 1 — cosine retrieval (GNN + LLM) | Method 2 — GNN directly |
|---|---|---|
| Slides | 10–11 of the project presentation | 12 |
| Idea | Induce a neighborhood: retrieve the top-K most similar dataset nodes already in the benchmark; the models evaluated on them feed graph-conditioned LLM prompting | Insert the dataset as a new node in the graph and let ArtifactLinker's trained joint GNN score every candidate model for it |
| Score | cosine similarity between summary embeddings | `combined_score = link_probability × predicted_attribute_value` (link × attribute, inductive use of the checkpoint) |
| Needs | `OPENAI_API_KEY`, `VOYAGE_API_KEY`, the ArtifactBench full split (bundled in this repo) | An `agents_project` environment with the ArtifactLinker repo, the ArtifactBench splits, and the trained checkpoint `trans_joint_gatv2_model_emb.pth` (see `method2_gnn_inference/README_INSTRUCTIONS.txt`) |

## Method 1 — `method1_cosine_retrieval/`

Pipeline: **your dataset → LLM summary → Voyage embedding → cosine retrieval
→ (LLM recommends)**. The summarization prompt, truncation limits, and
embedding settings are verbatim from the pipeline that built the benchmark
node embeddings (`scripts/step5_summarize_and_normalize.py`): GPT-4o,
temperature 0, strict-JSON 150–250-word summary with benchmark scores
excluded, card truncated to 12,000 chars; `voyage-3`,
`input_type="document"`, summary truncated to 8,000 chars, L2-normalized.

```bash
export OPENAI_API_KEY=... VOYAGE_API_KEY=...
python method1_cosine_retrieval/retrieve_similar_datasets.py openai/gsm8k --top-k 25
# local card instead of the Hub:
python method1_cosine_retrieval/retrieve_similar_datasets.py my-data --card README.md
# save summary + embedding + full similarity list:
python method1_cosine_retrieval/retrieve_similar_datasets.py openai/gsm8k --save-dir out/
```

By default it reads the benchmark from
`../../artifact_bench/artifact_bench_data/full/`; override with `--data-dir`.

## Method 2 — `method2_gnn_inference/`

ArtifactLinker inference, as in the paper. The main entry point is
`rank_hf_dataset_combined.py`: it resolves the dataset to a graph node (or,
with `--augment-missing`, appends it as a new isolated node to an augmented
copy of the split) and ranks all candidate models by the joint GNN's
link × attribute score.

```bash
export AGENTS_PROJECT_ROOT=/path/to/agents_project   # or place scripts in its root
# dataset already in ArtifactBench:
python rank_hf_dataset_combined.py https://huggingface.co/datasets/tau/commonsense_qa --top-k 25
# cold-start dataset (slide 12): summary + embedding, new node, GNN scores all models
export VOYAGE_API_KEY=... OPENAI_API_KEY=...
python rank_hf_dataset_combined.py https://huggingface.co/datasets/xai-org/RealworldQA \
    --augment-missing --summarize-card --top-k 25
```

`--summarize-card` applies the same GPT-4o summary step as Method 1 before
embedding (the benchmark node embeddings were computed on such summaries, so
this keeps the new node's embedding in-distribution). Omit it to reproduce the
original partner-bundle behavior of embedding the raw card text.

Helper scripts (`score_dataset_links.py`, `score_dataset_attributes.py`,
`score_model_dataset_pair.py`, `score_top_link_attributes.py`), environment
setup, data download, and checkpoint training instructions are documented in
`method2_gnn_inference/README_INSTRUCTIONS.txt`.

## Evaluation rounds - initial_run / revision_run

Recommendations for the `candidate_datasets_100.txt` pool are produced in two
rounds:

1. **`initial_run`** (`run_initial_run.py`) - the full pipeline (Method 1 +
   Method 2, conditions A/B/C) over all candidate datasets, constrained to
   models with at most 14B parameters that appear in the ArtifactBench graph.
   Output: `data/advisor_runs_initial_run_max14b/`.
2. **External evaluation** - a colleague runs the initial run's recommended
   models (condition A) on each dataset and reports per-model metrics +
   error/weakness analysis. `build_eval_template.py` generates the CSV they
   fill in (`dataset,model_name,metric_name,metric_value,error_notes`).
3. **`revision_run`** (`run_revision.py`) - feeds the original evidence, the
   round-1 recommendation, and the round-2 evaluation feedback back into the
   LLM so it can confirm, drop, or replace models based on what actually
   happened, not just re-predict from the graph again. Output:
   `data/advisor_runs_revision_run_max14b/`.

```bash
.venv-m2/bin/python retrieval_agent/run_initial_run.py
python retrieval_agent/build_eval_template.py \
    --run-root data/advisor_runs_initial_run_max14b \
    --out data/advisor_runs_initial_run_max14b/evaluation_results_template.csv
# ... colleague fills in the template and runs evaluation ...
.venv-m2/bin/python retrieval_agent/run_revision.py \
    --initial-run-root data/advisor_runs_initial_run_max14b \
    --eval-results data/advisor_runs_initial_run_max14b/evaluation_results.csv \
    --out-root data/advisor_runs_revision_run_max14b
```
