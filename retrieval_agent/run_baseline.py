#!/usr/bin/env python3
"""
Condition D - baseline: ask Claude Code (no ArtifactLinker evidence).

For each dataset, runs `claude -p --model <model>` from an EMPTY working
directory (no repo context, no graph evidence, told not to use tools) with a
standard prompt: the HF dataset id + the full list of allowed candidate models
(ArtifactBench models passing the size constraint), and asks for the 5 best
models. Output matches the other conditions' CSV schema with
source_method = "claude_baseline".

Usage:
    .venv-m2/bin/python retrieval_agent/run_baseline.py \
        --out-root data/advisor_runs_max3b --max-params-b 3 \
        --datasets-file retrieval_agent/eval_datasets.txt \
        [--model claude-opus-4-7]

Existing final_recommendations_baseline.csv files are skipped.
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from model_size import model_size_b  # noqa: E402
from pipeline import parse_repo_id, safe_name  # noqa: E402
from run_recommendation import extract_csv  # noqa: E402

BASELINE_TEMPLATE = """\
You are recommending machine learning models for a HuggingFace dataset.

Dataset: https://huggingface.co/datasets/{dataset}

Below is the complete list of candidate models you may choose from (name,
approximate parameter count in billions, HF downloads). HARD CONSTRAINTS:
recommend ONLY models from this list ({n_candidates} candidates, all at most
{max_b:g}B parameters); do not use any tools or external lookups - answer
directly from your own knowledge of the dataset and these models.

Task: choose the {n} most promising models for achieving strong performance on
this dataset, ranked from strongest to weakest.

First, briefly explain your overall reasoning (a short paragraph).

Then output the final recommendations as a CSV code block with EXACTLY this
header and one row per recommended model:

```csv
rank,model_name,recommendation_score,source_method,reasoning
```

Column rules:
- rank: 1 = strongest recommendation.
- model_name: the exact model name as it appears in the candidate list.
- recommendation_score: your overall score for the recommendation in [0, 1]
  (1.0 = near-certain excellent fit), consistent with the ranking order.
- source_method: always exactly "claude_baseline".
- reasoning: one or two sentences justifying the recommendation. Enclose the
  whole field in double quotes.

======================================================================
# CANDIDATE MODELS (name\tparams_B\tdownloads)
======================================================================

{candidates}
"""


def build_candidates(max_b: float, out_file: Path) -> list:
    meta = json.load(open(REPO / "artifact_bench" / "artifact_bench_data" / "full" / "node_metadata.json"))
    rows = []
    for v in meta.values():
        if v.get("type") != "model":
            continue
        size = model_size_b(v["name"])
        if size is not None and size <= max_b:
            rows.append((v["name"], size, v.get("downloads", 0)))
    rows.sort()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for name, size, dl in rows:
            f.write(f"{name}\t{size:g}\t{dl}\n")
    return rows


def run_baseline(dataset: str, candidates_text: str, n_candidates: int,
                 max_b: float, n: int, model: str, out_dir: Path, workdir: Path) -> str:
    out_csv = out_dir / "final_recommendations_baseline.csv"
    out_md = out_dir / "final_recommendation_baseline.md"
    if out_csv.exists():
        return "cached"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = BASELINE_TEMPLATE.format(
        dataset=dataset, n=n, max_b=max_b,
        n_candidates=n_candidates, candidates=candidates_text,
    )
    proc = subprocess.run(
        ["claude", "-p", "--model", model],
        input=prompt, capture_output=True, text=True, cwd=str(workdir), timeout=900,
    )
    response = proc.stdout or ""
    if proc.returncode != 0 or not response.strip():
        return f"claude CLI failed (exit {proc.returncode}): {proc.stderr[-500:]}"

    out_md.write_text(f"<!-- model: {model} | condition: baseline | query: {dataset} -->\n\n{response}\n",
                      encoding="utf-8")
    rows = extract_csv(response)
    if rows is None:
        return "NO CSV IN RESPONSE (see md)"
    header, data = rows[0], rows[1:]

    allowed = {line.split("\t")[0] for line in candidates_text.splitlines() if line.strip()}
    name_idx = header.index("model_name")
    violations = [r[name_idx] for r in data if r[name_idx] not in allowed]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_dataset"] + header)
        for r in data:
            w.writerow([dataset] + r)
    status = f"done ({len(data)} recs)"
    if violations:
        status += f" | WARNING off-list picks: {violations}"
    return status


def main():
    parser = argparse.ArgumentParser(description="Run the ask-Claude-Code baseline (condition D).")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--datasets-file", required=True, help="File with one HF dataset id per line")
    parser.add_argument("--max-params-b", type=float, default=3.0)
    parser.add_argument("--model", default="claude-opus-4-7")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    candidates_file = out_root / f"allowed_models_max{args.max_params_b:g}b.txt"
    rows = build_candidates(args.max_params_b, candidates_file)
    candidates_text = candidates_file.read_text(encoding="utf-8")
    print(f"{len(rows)} candidate models <= {args.max_params_b:g}B -> {candidates_file}")

    datasets = [parse_repo_id(l) for l in Path(args.datasets_file).read_text().splitlines() if l.strip()]
    with tempfile.TemporaryDirectory(prefix="baseline_empty_") as workdir:
        for ds in datasets:
            status = run_baseline(ds, candidates_text, len(rows), args.max_params_b,
                                  args.n, args.model, out_root / safe_name(ds), Path(workdir))
            print(f"  {ds:45s} {status}", flush=True)


if __name__ == "__main__":
    main()
