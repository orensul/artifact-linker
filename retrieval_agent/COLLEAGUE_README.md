# AutoModelAdvisor retrieval agent - guide for the evaluation loop

You're implementing the *evaluation* half of the loop: actually running the
models we recommend on each dataset, measuring how well they do, and telling
us where they fail. This doc covers what's already done, what we need back
from you, and how to call the retrieval agent yourself for anything beyond
that.

## What's already done

We ran the recommendation pipeline once already (`initial_run`, round 1)
over a 94-dataset candidate pool, constrained to models with at most 14B
parameters that appear in the ArtifactBench graph:

- **Candidate pool**: `retrieval_agent/candidate_datasets_100.txt` (94
  datasets, spans 13 domains and 4 task-format families - classification,
  NER, free-text generation, regression; see the file's header comment for
  the exact rationale and known caveats).
- **Round-1 recommendations**: `data/advisor_runs_initial_run_max14b/` - one
  subdirectory per dataset, each with `final_recommendations.csv` (top-5
  models under the merged evidence condition) plus the raw evidence that
  produced them (`merged_prompt.md`, `graph_context.md`, `gnn_context.md`).
  90 of the 94 datasets succeeded end-to-end; the other 4
  (`cais/mmlu`, `ybisk/piqa`, `facebook/anli`, `ought/raft`) are legitimate
  datasets that hit a graph-resolver edge case and couldn't get a Method-2
  recommendation this round - left in the candidate list since they're
  valid, just not currently runnable as-is. (4 *other* datasets were
  removed from the candidate list entirely after going dead on HuggingFace
  since it was built - not counted in the 94.) See the candidate file's
  header for the full detail.
- **Model recommendations for all 4 conditions (A/B/C/D) on the 90 datasets**:
  **`data/advisor_runs_initial_run_max14b/evaluation_conditions_intersection90.csv`**
  - the merged (A), method1-only (B), method2-only (C), and no-evidence
  Claude-baseline (D) recommendations, side by side, for every dataset that
  succeeded in all four.
- **What you need to fill in**:
  `data/advisor_runs_initial_run_max14b/evaluation_results_template.csv` -
  450 rows (90 datasets x top-5 models), `dataset` and `model_name`
  pre-filled, three blank columns for you:

  | column | meaning |
  |---|---|
  | `metric_name` | whatever you measured - accuracy, F1, ROUGE-L, exact-match, etc. Task type varies per dataset (see below) |
  | `metric_value` | the number. Leave blank if you only have qualitative notes for that model |
  | `error_notes` | free text - failure modes, weaknesses worth citing later (e.g. "hallucinates on >2k token inputs", "confuses stance labels for/against", "refuses to answer clinical questions") |

  Fill in as many rows as you can; you don't need every model on every
  dataset - partial coverage is fine, `revise()` (below) only needs whatever
  you actually measured for a given dataset.

### Task format varies per dataset - matters for how you score

The candidate pool is not uniformly one task type. Roughly:

- **Classification / multiple-choice** (~40 datasets) - score with
  exact-match / accuracy. E.g. `cais/mmlu`, `fever/fever`,
  `google-research-datasets/paws`.
- **Span-level NER / keyphrase extraction** (~10 datasets) - score with
  span-F1. E.g. `bigbio/chemprot`, `midas/inspec`.
- **Free-text generation** (QA, summarization, dialogue; ~30 datasets) -
  score with ROUGE/BLEU, exact-numeric-match (math datasets), or an
  LLM-judge - exact-match on the whole string won't work. E.g.
  `deepset/covid_qa_deepset`, `allenai/mslr2022`, `knkarthick/samsum`.
- **Regression / continuous score** (a handful) - e.g.
  `james-burton/wine_reviews`, `TheFinAI/fiqa-sentiment-classification`.

If you're unsure what a specific dataset's task/label format is, check its
HF card (linked from `candidate_datasets_100.csv`'s `huggingface_url`
column) or ask - don't guess, since it changes what `metric_name` means.

## The loop: recommend -> evaluate -> revise -> repeat

Once you have results for a dataset, feed them back in and the retrieval
agent will revise its recommendation - confirming models that did well,
dropping ones that didn't (citing your specific error notes), and promoting
alternatives from the original evidence pool that it thinks might avoid the
same failure mode.

```python
import sys
sys.path.insert(0, "/path/to/artifact-linker")   # repo root on sys.path

from retrieval_agent import recommend, revise

# Already have round-1 recs for the 90 datasets under
# data/advisor_runs_initial_run_max14b/ - read final_recommendations.csv
# directly, or just call recommend() again (it's cached, so this is instant
# and makes no new API calls):
recs = recommend("deepset/covid_qa_deepset", max_params_b=14,
                  out_root="data/advisor_runs_initial_run_max14b")
for r in recs:
    print(r.rank, r.model_name, r.recommendation_score, r.source_method)

# You run/evaluate each r.model_name on the dataset however you do that.
# Then package what you found - same shape as evaluation_results_template.csv:
eval_results = [
    {"model_name": "mradermacher/Llama3-Aloe-8B-Alpha-GGUF",
     "metric_name": "accuracy", "metric_value": 0.81,
     "error_notes": "strong on factual QA, occasionally hallucinates dosage numbers"},
    {"model_name": "google/medgemma-4b-it",
     "metric_name": "accuracy", "metric_value": 0.55,
     "error_notes": "frequently refuses to answer, over-triggers safety guardrails"},
    # one dict per (model, metric) observation you have - doesn't need to be
    # every model, and metric_value/error_notes can be omitted if unknown
]

revised = revise("deepset/covid_qa_deepset", eval_results,
                  initial_run_root="data/advisor_runs_initial_run_max14b")
for r in revised:
    print(r.rank, r.model_name, r.change_from_initial, r.reasoning)
    # r.change_from_initial is "confirmed" | "revised" | "new"
```

That's the whole loop. Call `revise()` again with a fresh `eval_results` list
whenever you have new information for a dataset - each call is independent,
you don't need to accumulate history yourself (the agent re-reads round 1's
evidence + recommendation from disk each time).

For a dataset outside the original 94, or to bulk-run many at once:

```python
from retrieval_agent import recommend_batch

all_recs = recommend_batch(["your-org/new-dataset-1", "your-org/new-dataset-2"])
```

This paces calls automatically (20s between datasets) to avoid the
Voyage API's free-tier rate limit (3 requests/minute) - don't loop over
`recommend()` yourself without that pacing or you'll hit `RateLimitError`.

## Batch CLI alternative

If you'd rather work file-to-file instead of writing Python: fill in
`evaluation_results_template.csv` directly, save it (anywhere, e.g. as
`evaluation_results.csv` next to the template), then run:

```bash
.venv-m2/bin/python retrieval_agent/run_revision.py \
    --initial-run-root data/advisor_runs_initial_run_max14b \
    --eval-results data/advisor_runs_initial_run_max14b/evaluation_results.csv \
    --out-root data/advisor_runs_revision_run_max14b
```

This does the same thing as calling `revise()` for every dataset in the
template, and writes `final_recommendations_revision.csv` per dataset plus
an aggregated `all_final_recommendations_revision.csv` under `--out-root`.

## Environment setup

```bash
cd /path/to/artifact-linker
# .venv-m2 already has the right versions (torch 2.13 + PyG 2.8 + openai + voyageai)
source .venv-m2/bin/activate   # or prefix commands with .venv-m2/bin/python
export AGENTS_PROJECT_ROOT="$(pwd)/data/agents_project"
```

You'll also need `OPENAI_API_KEY` and `VOYAGE_API_KEY` - either exported in
your shell, or in a `.env` file at the repo root (auto-loaded, not
committed to git). Ask for these if you don't have them.

## Known gotchas

- **Voyage rate limits**: the shared key is free-tier (3 RPM). `recommend()`
  / `recommend_batch()` retry automatically (3 attempts, 45s apart) but if
  you write your own loop, add pacing - don't fire requests back-to-back.
- **A dataset can permanently fail**: either because it's been removed/gated
  on HuggingFace since we built the candidate list, or because
  ArtifactLinker's graph resolver flags it as ambiguous against an unrelated
  but similarly-named graph node (a safety check against accidental
  contamination, not a bug - see `candidate_datasets_100.txt`'s header for
  the current list of affected datasets). Both raise `RecommendationError`
  from the Python API; check the exception message for which case it is.
- **`revise()` needs round-1 output first**: it reads
  `final_recommendations.csv` + `merged_prompt.md` from `initial_run_root`,
  so `recommend()` (or `run_initial_run.py`) must have already succeeded for
  that dataset there.

## Where things are

| What | Path |
|---|---|
| Candidate dataset list | `retrieval_agent/candidate_datasets_100.txt` / `.csv` |
| Round-1 (initial_run) results | `data/advisor_runs_initial_run_max14b/` |
| Master results file (A/B/C/D, 90-dataset intersection) | **`data/advisor_runs_initial_run_max14b/evaluation_conditions_intersection90.csv`** |
| Template for you to fill in | `data/advisor_runs_initial_run_max14b/evaluation_results_template.csv` |
| Library API (`recommend`/`revise`) | `retrieval_agent/api.py` |
| Full technical docs (Method 1/2 internals, batch CLI details) | `retrieval_agent/README.md` |
