#!/usr/bin/env python3
"""
Public library API for the AutoModelAdvisor retrieval agent.

Wraps the underlying pipeline (Method 1 cosine retrieval + Method 2 GNN
inference + merge + LLM recommendation, plus the round-2 revision flow) as
plain Python functions, so a caller can drive their own evaluation loop:

    1. get recommendations for a dataset          -> recommend()
    2. run/evaluate the recommended models themselves (own code)
    3. feed those results back in for a revision   -> revise()
    4. repeat for additional datasets              -> recommend_batch()

No CLI subprocess calls, no hand-parsed CSVs - just Python data in, Python
data out. See "Library usage" in README.md for a worked example.

Environment requirements (unchanged from the CLI scripts):
  - OPENAI_API_KEY, VOYAGE_API_KEY in the environment or a .env file at the
    repo root (auto-loaded).
  - AGENTS_PROJECT_ROOT pointing at the Method 2 GNN checkpoint + augmented
    splits (see method2_gnn_inference/README_INSTRUCTIONS.txt).
  - The ArtifactBench full split under
    artifact_bench/artifact_bench_data/full/.
  - Run under the .venv-m2 interpreter (torch + torch_geometric + openai +
    voyageai); the repo's default venv/ is Streamlit-only.
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pipeline import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    load_dotenv,
    parse_repo_id,
    run_dataset,
    safe_name,
)
from run_revision import run_revision as _run_revision_impl  # noqa: E402

load_dotenv()

DEFAULT_MAX_PARAMS_B = 14.0
DEFAULT_LLM_MODEL = "gpt-5.5"
DEFAULT_N = 5

# Voyage's free-tier key is rate-limited to 3 RPM; pacing/retry defaults
# match run_initial_run.py, which validated these values against real usage.
DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY_S = 45
DEFAULT_INTER_DATASET_DELAY_S = 20


@dataclass
class Recommendation:
    rank: int
    model_name: str
    recommendation_score: float
    source_method: str
    reasoning: str
    query_dataset: str
    # Only set on revise() output: "confirmed" | "revised" | "new".
    change_from_initial: Optional[str] = None


class RecommendationError(RuntimeError):
    """Raised when a dataset permanently fails to produce recommendations."""


def _read_recommendations_csv(path: Path) -> list[Recommendation]:
    if not path.exists():
        raise RecommendationError(f"expected output not found: {path}")
    recs = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            recs.append(Recommendation(
                rank=int(row["rank"]),
                model_name=row["model_name"],
                recommendation_score=float(row["recommendation_score"]),
                source_method=row["source_method"],
                reasoning=row.get("reasoning", ""),
                query_dataset=row["query_dataset"],
                change_from_initial=row.get("change_from_initial") or None,
            ))
    return recs


def recommend(
    dataset: str,
    *,
    max_params_b: float = DEFAULT_MAX_PARAMS_B,
    n: int = DEFAULT_N,
    llm_model: str = DEFAULT_LLM_MODEL,
    out_root: "str | Path" = DEFAULT_OUT_ROOT,
    retries: int = DEFAULT_RETRIES,
    retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
) -> list[Recommendation]:
    """Run the full Method1+Method2+merge+LLM pipeline for one dataset.

    Returns the top-`n` recommendations, constrained to models with at most
    `max_params_b` parameters that appear in the ArtifactBench graph.

    Idempotent: stages already computed under `out_root` for this dataset
    are reused (matches the CLI pipeline's file-based caching) - safe to
    call repeatedly, and cheap to re-call after a partial failure. Retries
    transient failures (e.g. Voyage RateLimitError) up to `retries` times
    with `retry_delay_s` between attempts before raising.

    Raises RecommendationError if the dataset can't be resolved/processed
    (e.g. removed from HuggingFace, or an ambiguous match in the graph -
    see README.md's "known dataset caveats").
    """
    out_root = Path(out_root)
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            result = run_dataset(
                dataset, out_root=out_root, max_params_b=max_params_b,
                llm_model=llm_model, num_recommendations=n,
            )
            if "error" in result:
                raise RuntimeError(result["error"])
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(retry_delay_s)
    if last_exc is not None:
        raise RecommendationError(
            f"recommend({dataset!r}) failed after {retries} attempts: {last_exc}"
        ) from last_exc

    out_csv = out_root / safe_name(parse_repo_id(dataset)) / "final_recommendations.csv"
    return _read_recommendations_csv(out_csv)


def recommend_batch(
    datasets: Iterable[str],
    *,
    inter_dataset_delay_s: float = DEFAULT_INTER_DATASET_DELAY_S,
    **kwargs,
) -> dict[str, list[Recommendation]]:
    """recommend() over many datasets, paced to avoid the Voyage free-tier
    rate limit (same pacing run_initial_run.py used for the 94-dataset run).

    A dataset that fails permanently (after retries) is omitted from the
    result rather than aborting the batch - check the returned dict's keys
    against your input list to see what's missing, and why (a message is
    printed to stderr for each skipped dataset).
    """
    out: dict[str, list[Recommendation]] = {}
    datasets = list(datasets)
    for i, ds in enumerate(datasets):
        try:
            out[parse_repo_id(ds)] = recommend(ds, **kwargs)
        except RecommendationError as exc:
            print(f"recommend_batch: skipping {ds!r}: {exc}", file=sys.stderr)
        if i < len(datasets) - 1:
            time.sleep(inter_dataset_delay_s)
    return out


def revise(
    dataset: str,
    eval_results: list[dict],
    *,
    initial_run_root: "str | Path",
    out_root: "str | Path | None" = None,
    max_params_b: float = DEFAULT_MAX_PARAMS_B,
    n: int = DEFAULT_N,
    llm_model: str = DEFAULT_LLM_MODEL,
) -> list[Recommendation]:
    """Revise a dataset's recommendation using real evaluation feedback.

    `eval_results` is a list of dicts, one per (model, metric) observation -
    the same shape as a row of the evaluation_results_template.csv that
    build_eval_template.py generates:
        {"model_name": ..., "metric_name": ..., "metric_value": ...,
         "error_notes": ...}
    `metric_name`/`metric_value`/`error_notes` may be omitted or blank if
    only some of them are known.

    `initial_run_root` must contain this dataset's round-1 output
    (final_recommendations.csv + merged_prompt.md) - i.e. recommend() (or
    run_initial_run.py) must have already succeeded for this dataset there.
    `out_root` defaults to `initial_run_root` (writes alongside round 1);
    pass a different path to keep rounds in separate directories.

    Returns the revised top-`n` recommendations. Each Recommendation's
    `change_from_initial` field says whether the LLM kept ("confirmed" or
    "revised") or newly promoted ("new") that model.
    """
    initial_run_root = Path(initial_run_root)
    out_root = Path(out_root) if out_root is not None else initial_run_root

    name = parse_repo_id(dataset)
    src_d = initial_run_root / safe_name(name)
    initial_csv, merged_md = src_d / "final_recommendations.csv", src_d / "merged_prompt.md"
    if not initial_csv.exists() or not merged_md.exists():
        raise RecommendationError(
            f"no round-1 output for {dataset!r} under {initial_run_root} "
            "(expected final_recommendations.csv + merged_prompt.md) - call recommend() first."
        )

    with initial_csv.open(encoding="utf-8") as f:
        initial_rows = list(csv.reader(f))
    query_name = initial_rows[1][0] if len(initial_rows) > 1 else name
    initial_csv_text = "\n".join(",".join(r) for r in initial_rows)
    original_evidence = merged_md.read_text(encoding="utf-8").strip()

    lines = []
    for r in eval_results:
        metric_name = str(r.get("metric_name", "")).strip()
        metric_value = str(r.get("metric_value", "")).strip()
        metric = f"{metric_name}={metric_value}" if metric_name or metric_value else "(no metric reported)"
        notes = str(r.get("error_notes", "")).strip() or "(no notes)"
        lines.append(f"- {r.get('model_name', '')}: {metric} | {notes}")
    eval_text = "\n".join(lines)

    out_d = out_root / safe_name(name)
    status = _run_revision_impl(query_name, initial_csv_text, original_evidence, eval_text,
                                out_d, llm_model, n, max_params_b)
    out_csv = out_d / "final_recommendations_revision.csv"
    if not out_csv.exists():
        raise RecommendationError(f"revise({dataset!r}) did not produce output: {status}")
    return _read_recommendations_csv(out_csv)
