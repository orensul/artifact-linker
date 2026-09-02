#!/usr/bin/env python3
"""
Evaluation conditions B and C: single-source recommendation runs.

For every dataset run directory under the output root (produced by pipeline.py):
  B  method1-only: graph_context.md   -> final_recommendation_m1.md / final_recommendations_m1.csv
  C  method2-only: gnn_context.md     -> final_recommendation_m2.md / final_recommendations_m2.csv

Then aggregates conditions A (merged, from pipeline.py), B, and C into
<out_root>/evaluation_conditions.csv with a `condition` column - the master
file for the model-running evaluation.

Usage:
    .venv-m2/bin/python retrieval_agent/run_conditions.py [--model gpt-5.5] [--n 5]

Existing outputs are skipped (delete a file to regenerate).
"""

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline import DEFAULT_OUT_ROOT, load_dotenv  # noqa: E402
from run_recommendation import call_llm, extract_csv  # noqa: E402

SINGLE_SOURCE_TEMPLATE = """\
You are selecting the most promising models for a new dataset. You are given
evidence from ArtifactBench, a graph of real model evaluations:

{source_description}

Prefer models with consistent results across several similar datasets, and
weigh measured evidence over predictions.
{constraint}
Task: prepare a final list of the {n} most promising models for the query
dataset, ranked from strongest to weakest.

First, briefly explain your overall reasoning (a short paragraph).

Then output the final recommendations as a CSV code block with EXACTLY this
header and one row per recommended model:

```csv
rank,model_name,recommendation_score,source_method,reasoning
```

Column rules:
- rank: 1 = strongest recommendation.
- model_name: the exact model name as it appears in the evidence.
- recommendation_score: your overall score for the recommendation in [0, 1]
  (1.0 = near-certain excellent fit), consistent with the ranking order.
- source_method: always exactly "{source_method}".
- reasoning: one or two sentences justifying the recommendation, citing the
  specific dataset names and metric values from the evidence. Enclose the
  whole field in double quotes.

======================================================================
# EVIDENCE
======================================================================

{context}
"""

CONDITIONS = {
    "m1": {
        "context_file": "graph_context.md",
        "source_method": "method1",
        "source_description": (
            "- Retrieved neighborhood: datasets most similar to the query dataset, "
            "with REAL MEASURED results of models on them. Grounded but indirect - "
            "the results are on similar datasets, not the query itself."
        ),
    },
    "m2": {
        "context_file": "gnn_context.md",
        "source_method": "method2",
        "source_description": (
            "- Graph neural network ranking: a GNN trained on the evaluation graph "
            "scored every candidate model for the query dataset. Its scores are "
            "PREDICTIONS, not measurements, but each recommended model also lists "
            "its real measured track record on other datasets."
        ),
    },
}


def run_condition(d: Path, cond: str, model: str, n: int, constraint: str = "") -> str:
    spec = CONDITIONS[cond]
    out_csv = d / f"final_recommendations_{cond}.csv"
    out_md = d / f"final_recommendation_{cond}.md"
    if out_csv.exists():
        return "cached"

    ctx_file = d / spec["context_file"]
    if not ctx_file.exists():
        return f"missing {spec['context_file']}"
    query_name = d.name.replace("_", "/", 1)
    a_csv = d / "final_recommendations.csv"
    if a_csv.exists():
        with a_csv.open(encoding="utf-8") as fh:
            first = list(csv.DictReader(fh))
        if first:
            query_name = first[0]["query_dataset"]

    constraint_text = ""
    if constraint:
        constraint_text = f"\nHARD CONSTRAINT: {constraint} Only recommend models that appear in the evidence below.\n"
    prompt = SINGLE_SOURCE_TEMPLATE.format(
        source_description=spec["source_description"],
        source_method=spec["source_method"],
        n=n,
        constraint=constraint_text,
        context=ctx_file.read_text(encoding="utf-8").strip(),
    )
    response = call_llm(prompt, model, None)
    out_md.write_text(f"<!-- model: {model} | condition: {cond} | query: {query_name} -->\n\n{response}\n",
                      encoding="utf-8")
    rows = extract_csv(response)
    if rows is None:
        return "NO CSV IN RESPONSE (see md)"
    header, data = rows[0], rows[1:]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_dataset"] + header)
        for r in data:
            w.writerow([query_name] + r)
    return f"done ({len(data)} recs)"


def aggregate_conditions(out_root: Path) -> Path:
    """Master CSV over conditions A (merged), B (m1), C (m2)."""
    spec = [("A_merged", "final_recommendations.csv"),
            ("B_method1_only", "final_recommendations_m1.csv"),
            ("C_method2_only", "final_recommendations_m2.csv"),
            ("D_claude_baseline", "final_recommendations_baseline.csv")]
    rows, header = [], None
    for d in sorted(out_root.glob("*/")):
        for cond, fname in spec:
            f = d / fname
            if not f.exists():
                continue
            with f.open(encoding="utf-8") as fh:
                r = list(csv.reader(fh))
            if not r:
                continue
            if header is None:
                header = ["condition"] + r[0]
            rows += [[cond] + row for row in r[1:]]
    agg = out_root / "evaluation_conditions.csv"
    if header:
        with agg.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
    return agg


def main():
    parser = argparse.ArgumentParser(description="Run single-source recommendation conditions B and C.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--constraint", default="",
                        help="Hard constraint text inserted into the instruction")
    args = parser.parse_args()

    load_dotenv()
    out_root = Path(args.out_root)
    dirs = [d for d in sorted(out_root.glob("*/")) if (d / "merged_prompt.md").exists()]
    print(f"{len(dirs)} dataset runs found under {out_root}")
    for d in dirs:
        for cond in ("m1", "m2"):
            status = run_condition(d, cond, args.model, args.n, constraint=args.constraint)
            print(f"  {d.name:45s} {cond}: {status}")
    agg = aggregate_conditions(out_root)
    print(f"Master conditions file: {agg}")


if __name__ == "__main__":
    main()
