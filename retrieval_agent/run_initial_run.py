#!/usr/bin/env python3
"""
Initial run - round 1 of the AutoModelAdvisor evaluation loop.

Round structure:
  1. initial_run   (this script)      -> data/advisor_runs_initial_run_max14b/
                                          Runs the full pipeline (Method 1 +
                                          Method 2 + merge + LLM recommendation)
                                          over the 98-dataset candidate pool,
                                          conditions A (merged), B (method1-only),
                                          C (method2-only), D (ask-Claude-Code
                                          baseline, no ArtifactLinker evidence),
                                          all constrained to <=14B-parameter
                                          models that appear in the
                                          ArtifactBench graph.
  2. external evaluation               -> a colleague runs the recommended
                                          models (condition A) on each dataset
                                          and reports per-model metrics + error
                                          analysis. Use build_eval_template.py
                                          to generate the CSV they fill in.
  3. revision_run  (run_revision.py)   -> feeds the evaluation results and
                                          error analysis back into the LLM so
                                          it can confirm/drop/replace models.

Usage:
    .venv-m2/bin/python retrieval_agent/run_initial_run.py
"""

import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from pipeline import run_dataset, aggregate_final_csvs, parse_repo_id, safe_name  # noqa: E402
import run_conditions  # noqa: E402
import run_baseline  # noqa: E402

DATASETS_FILE = HERE / "candidate_datasets_100.txt"
OUT_ROOT = REPO / "data" / "advisor_runs_initial_run_max14b"
MAX_PARAMS_B = 14.0
LLM_MODEL = "gpt-5.5"
BASELINE_MODEL = "claude-opus-4-7"
N_RECOMMENDATIONS = 5
CONSTRAINT_TEXT = f"Only recommend models with at most {MAX_PARAMS_B:g}B parameters."

# Voyage's free-tier key is rate-limited to 3 RPM; Method 1 and Method 2
# (--summarize-card) each make an embedding call per dataset, so back-to-back
# datasets reliably hit RateLimitError without pacing + retry.
INTER_DATASET_DELAY_S = 20
MAX_RETRIES = 3
RETRY_DELAY_S = 45


def load_datasets(path: Path) -> list:
    datasets = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        datasets.append(line)
    return datasets


def run_dataset_with_retry(ds, **kwargs) -> dict:
    """run_dataset(), retrying transient failures (e.g. Voyage RateLimitError).

    Stages are file-cached (pipeline.stage()), so a retry resumes from
    whichever stage actually failed instead of redoing completed work.
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return run_dataset(ds, **kwargs)
        except Exception as exc:
            last_exc = exc
            print(f"  attempt {attempt}/{MAX_RETRIES} failed for {ds}: {str(exc)[-300:]}")
            if attempt < MAX_RETRIES:
                print(f"  retrying in {RETRY_DELAY_S}s...")
                time.sleep(RETRY_DELAY_S)
    print(f"ERROR on {ds} after {MAX_RETRIES} attempts: {last_exc}")
    return {"dataset": parse_repo_id(ds), "error": str(last_exc)}


def run_condition_safe(d, cond, *args, **kwargs) -> str:
    """run_conditions.run_condition(), isolating one dataset's failure from
    the rest of the loop (an unhandled OpenAI/network error would otherwise
    kill the whole multi-hour run)."""
    try:
        return run_conditions.run_condition(d, cond, *args, **kwargs)
    except Exception as exc:
        return f"ERROR: {str(exc)[-300:]}"


def run_baseline_safe(*args, **kwargs) -> str:
    try:
        return run_baseline.run_baseline(*args, **kwargs)
    except Exception as exc:
        return f"ERROR: {str(exc)[-300:]}"


def main():
    datasets = load_datasets(DATASETS_FILE)
    print(f"{len(datasets)} datasets loaded from {DATASETS_FILE}")
    print(f"Output root: {OUT_ROOT}")
    print(f"Constraint: <= {MAX_PARAMS_B:g}B params, models must appear in the ArtifactBench graph")

    # Condition A: merged Method1+Method2 recommendation (also builds the
    # graph_context.md / gnn_context.md that conditions B and C reuse).
    for i, ds in enumerate(datasets):
        print(f"=== [{i + 1}/{len(datasets)}] {parse_repo_id(ds)} ===")
        run_dataset_with_retry(
            ds, out_root=OUT_ROOT,
            max_params_b=MAX_PARAMS_B, llm_model=LLM_MODEL,
            num_recommendations=N_RECOMMENDATIONS,
        )
        if i < len(datasets) - 1:
            time.sleep(INTER_DATASET_DELAY_S)
    aggregate_final_csvs(OUT_ROOT)

    # Conditions B (method1-only) and C (method2-only), single-source ablations.
    dirs = [d for d in sorted(OUT_ROOT.glob("*/")) if (d / "merged_prompt.md").exists()]
    print(f"\n{len(dirs)} dataset runs ready for conditions B/C")
    for d in dirs:
        for cond in ("m1", "m2"):
            status = run_condition_safe(
                d, cond, LLM_MODEL, N_RECOMMENDATIONS, constraint=CONSTRAINT_TEXT,
            )
            print(f"  {d.name:45s} {cond}: {status}")

    # Condition D: ask-Claude-Code baseline, no ArtifactLinker evidence at all,
    # same <=14B allowed-models candidate pool passed in the prompt.
    candidates_file = OUT_ROOT / f"allowed_models_max{MAX_PARAMS_B:g}b.txt"
    rows = run_baseline.build_candidates(MAX_PARAMS_B, candidates_file)
    candidates_text = candidates_file.read_text(encoding="utf-8")
    print(f"\n{len(rows)} candidate models <= {MAX_PARAMS_B:g}B -> {candidates_file}")
    with tempfile.TemporaryDirectory(prefix="baseline_empty_") as workdir:
        for ds in datasets:
            name = parse_repo_id(ds)
            status = run_baseline_safe(
                name, candidates_text, len(rows), MAX_PARAMS_B, N_RECOMMENDATIONS,
                BASELINE_MODEL, OUT_ROOT / safe_name(name), Path(workdir),
            )
            print(f"  {name:45s} D: {status}")

    agg = run_conditions.aggregate_conditions(OUT_ROOT)
    print(f"\nMaster conditions file (A + B + C + D): {agg}")


if __name__ == "__main__":
    main()
