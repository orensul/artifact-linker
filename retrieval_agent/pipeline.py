#!/usr/bin/env python3
"""
AutoModelAdvisor - end-to-end pipeline over a list of HuggingFace datasets.

For each dataset it runs the full chain (same components as the Colab
notebook), saving every artifact under <out_root>/<dataset>/:

  1. Method 1  retrieve_similar_datasets.py   -> summary.txt, similarities.json, method1.csv
  2.           build_graph_context.py         -> graph_context.md/.json
  3. Method 2  rank_hf_dataset_combined.py    -> method2_predictions.json, method2.csv
               (--augment-missing --summarize-card: cold-start datasets are
               appended to an augmented split copy, summarized like Method 1)
  4.           build_gnn_context.py           -> gnn_context.md/.json
  5.           merge_contexts.py              -> merged_prompt.md
  6.           run_recommendation.py          -> final_recommendation.md, final_recommendations.csv

Stages whose outputs already exist are skipped (delete a file to recompute).
Finished datasets are aggregated into <out_root>/all_final_recommendations.csv.

Library use:
    from pipeline import run_pipeline
    results = run_pipeline(["tau/commonsense_qa", "xai-org/RealworldQA"])

CLI use:
    .venv-m2/bin/python retrieval_agent/pipeline.py tau/commonsense_qa xai-org/RealworldQA

Requires OPENAI_API_KEY + VOYAGE_API_KEY (steps 1, 3 on cold-start, and 6),
and the local Method 2 setup (checkpoint + transductive split under data/).
"""

import csv
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PY = sys.executable

FULL_DATA = REPO / "artifact_bench" / "artifact_bench_data" / "full"
AGENTS_ROOT = REPO / "data" / "agents_project"
DEFAULT_OUT_ROOT = REPO / "data" / "advisor_runs"


def load_dotenv(path: Path = REPO / ".env"):
    """Load KEY=VALUE lines from .env into os.environ (no overriding)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


load_dotenv()

STAGES = ["method1", "graph_context", "method2", "gnn_context", "merge", "recommend"]


def parse_repo_id(value: str) -> str:
    value = value.strip().rstrip("/")
    m = re.search(r"huggingface\.co/datasets/([^/]+/[^/?#]+)", value)
    return m.group(1) if m else value


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _run(cmd, log, extra_env=None):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env)
    out = (proc.stdout or "") + (proc.stderr or "")
    log(out.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"step failed (exit {proc.returncode}): {' '.join(str(c) for c in cmd)}\n{out[-3000:]}")
    return out


def run_dataset(
    dataset: str,
    out_root: Path = DEFAULT_OUT_ROOT,
    top_k_similar: int = 25,
    context_top_k: int = 10,
    gnn_context_top_k: int = 10,
    top_k_models: int = 25,
    llm_model: str = "gpt-5.5",
    num_recommendations: int = 5,
    skip_final_llm: bool = False,
    max_params_b: float = None,
    log=print,
    on_stage=None,
) -> dict:
    """Run the full pipeline for one dataset. Returns paths + status per stage."""
    name = parse_repo_id(dataset)
    d = Path(out_root) / safe_name(name)
    d.mkdir(parents=True, exist_ok=True)
    result = {"dataset": name, "dir": str(d), "stages": {}}
    size_filter = ["--max-params-b", max_params_b] if max_params_b is not None else []
    if max_params_b is not None:
        # Large models dominate the raw GNN ranking; rank deeper so enough
        # small models survive the size filter.
        top_k_models = max(top_k_models, 500)

    def stage(key, outputs, fn):
        if on_stage:
            on_stage(key)
        if all(Path(p).exists() for p in outputs):
            result["stages"][key] = "cached"
            log(f"[{name}] {key}: cached, skipping")
            return
        fn()
        result["stages"][key] = "done"

    # 1. Method 1 retrieval (GPT-4o summary + voyage embedding + cosine)
    stage("method1", [d / "similarities.json", d / "summary.txt"], lambda: _run([
        PY, HERE / "method1_cosine_retrieval" / "retrieve_similar_datasets.py", name,
        "--top-k", top_k_similar, "--data-dir", FULL_DATA,
        "--save-dir", d, "--csv", d / "method1.csv",
    ], log))

    # 2. Method 1 graph context
    stage("graph_context", [d / "graph_context.md"], lambda: _run([
        PY, HERE / "method1_cosine_retrieval" / "build_graph_context.py",
        "--similarities", d / "similarities.json", "--query-name", name,
        "--query-summary-file", d / "summary.txt", "--data-dir", FULL_DATA,
        "--top-k", context_top_k, "--max-models", 15,
        "--out-text", d / "graph_context.md", "--out-json", d / "graph_context.json",
    ] + size_filter, log))

    # 3. Method 2 GNN inference (augments cold-start datasets automatically)
    pred = d / "method2_predictions.json"
    stage("method2", [pred], lambda: _run([
        PY, HERE / "method2_gnn_inference" / "rank_hf_dataset_combined.py", name,
        "--top-k", top_k_models, "--augment-missing", "--summarize-card",
        "--output", pred, "--csv", d / "method2.csv",
    ], log, extra_env={"AGENTS_PROJECT_ROOT": str(AGENTS_ROOT)}))

    # 4. Method 2 context
    stage("gnn_context", [d / "gnn_context.md"], lambda: _run([
        PY, HERE / "method2_gnn_inference" / "build_gnn_context.py",
        "--predictions", pred, "--data-dir", FULL_DATA,
        "--top-k", gnn_context_top_k, "--max-evidence", 5,
        "--out-text", d / "gnn_context.md", "--out-json", d / "gnn_context.json",
    ] + size_filter, log))

    # 5. Merge into one prompt
    stage("merge", [d / "merged_prompt.md"], lambda: _run([
        PY, HERE / "merge_contexts.py",
        "--method1-md", d / "graph_context.md", "--method1-json", d / "graph_context.json",
        "--method2-md", d / "gnn_context.md", "--method2-json", d / "gnn_context.json",
        "--num-recommendations", num_recommendations, "--out", d / "merged_prompt.md",
    ] + (["--constraint", f"Only recommend models with at most {max_params_b:g}B parameters."]
         if max_params_b is not None else []), log))

    # 6. Final recommendation LLM
    if not skip_final_llm:
        stage("recommend", [d / "final_recommendations.csv"], lambda: _run([
            PY, HERE / "run_recommendation.py",
            "--prompt", d / "merged_prompt.md", "--query-name", name,
            "--model", llm_model,
            "--out-md", d / "final_recommendation.md",
            "--out-csv", d / "final_recommendations.csv",
        ], log))
    else:
        result["stages"]["recommend"] = "skipped"

    return result


def aggregate_final_csvs(out_root: Path = DEFAULT_OUT_ROOT) -> Path:
    """Concatenate every per-dataset final_recommendations.csv into one file."""
    out_root = Path(out_root)
    rows, header = [], None
    for f in sorted(out_root.glob("*/final_recommendations.csv")):
        with f.open(encoding="utf-8") as fh:
            r = list(csv.reader(fh))
        if not r:
            continue
        if header is None:
            header = r[0]
        rows.extend(r[1:])
    agg = out_root / "all_final_recommendations.csv"
    if header:
        with agg.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
    return agg


def run_pipeline(datasets, out_root: Path = DEFAULT_OUT_ROOT, log=print, **kwargs) -> list:
    """Run the full pipeline for a list of HF datasets (urls or repo ids)."""
    results = []
    for ds in datasets:
        log(f"=== {parse_repo_id(ds)} ===")
        try:
            results.append(run_dataset(ds, out_root=out_root, log=log, **kwargs))
        except Exception as exc:
            results.append({"dataset": parse_repo_id(ds), "error": str(exc)})
            log(f"ERROR on {ds}: {exc}")
    aggregate_final_csvs(out_root)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the AutoModelAdvisor pipeline over HF datasets.")
    parser.add_argument("datasets", nargs="+", help="HF dataset urls or owner/name ids")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--num-recommendations", type=int, default=5)
    parser.add_argument("--skip-final-llm", action="store_true",
                        help="Stop after merged_prompt.md (no recommendation LLM call)")
    parser.add_argument("--max-params-b", type=float, default=None,
                        help="Constrain evidence and recommendations to models <= this many B params")
    args = parser.parse_args()

    res = run_pipeline(args.datasets, out_root=Path(args.out_root),
                       llm_model=args.model, num_recommendations=args.num_recommendations,
                       skip_final_llm=args.skip_final_llm, max_params_b=args.max_params_b)
    print(json.dumps(res, indent=2))
