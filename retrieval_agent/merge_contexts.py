#!/usr/bin/env python3
"""
Retrieval agent - merge Method 1 and Method 2 evidence into one LLM prompt.

Inputs are the artifacts the two context builders already produce:
  Method 1: graph_context.md / graph_context.json   (build_graph_context.py)
  Method 2: gnn_context.md / gnn_context.json       (build_gnn_context.py)

The merged prompt frames the two sources for the LLM:
  A. cosine-retrieved neighborhood -> real measured results on similar datasets
  B. GNN ranking -> learned predictions, each with the model's real track record
plus a computed AGREEMENT section: models recommended by the GNN that also
appear with measured results in the retrieved neighborhood.

Usage:
    python merge_contexts.py \
        --method1-md graph_context.md --method1-json graph_context.json \
        --method2-md gnn_context.md   --method2-json gnn_context.json \
        --num-recommendations 5 \
        --out merged_prompt.md

Pure code, no LLM calls: the output is the prompt you paste into the LLM.
"""

import argparse
import json


INSTRUCTION_TEMPLATE = """\
You are selecting the most promising models for a new dataset. You are given two
independent sources of evidence from ArtifactBench, a graph of real model
evaluations:

- SOURCE A (retrieved neighborhood): datasets most similar to the query dataset,
  with REAL MEASURED results of models on them. Grounded but indirect - the
  results are on similar datasets, not the query itself.
- SOURCE B (graph neural network): a GNN trained on the evaluation graph scored
  every candidate model for the query dataset. Its scores are PREDICTIONS, not
  measurements, but each recommended model also lists its real measured track
  record on other datasets.
- AGREEMENT: models that appear in both sources - typically the strongest candidates.

Weigh measured evidence over predictions when they conflict. Prefer models with
consistent results across several similar datasets.
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
- source_method: exactly one of "method1" (supported only by SOURCE A, the
  retrieved neighborhood), "method2" (supported only by SOURCE B, the GNN
  ranking), or "both" (appears in both sources / the AGREEMENT section).
- reasoning: one or two sentences justifying the recommendation, citing the
  specific dataset names and metric values from the evidence. Enclose the
  whole field in double quotes.
"""


def method1_model_names(ctx):
    names = set()
    for nb in ctx.get("neighbor_datasets", []):
        for m in nb.get("models", []):
            names.add(m["name"])
    for m in ctx.get("models_on_multiple_neighbors", []):
        names.add(m["name"])
    return names


def agreement_section(m1_ctx, m2_ctx):
    if not (m1_ctx and m2_ctx):
        return None
    m1_names = method1_model_names(m1_ctx)
    hits = []
    for rec in m2_ctx.get("recommendations", []):
        if rec["name"] in m1_names:
            hits.append(rec)
    if not hits:
        return ("## AGREEMENT between the two sources\n\n"
                "No model recommended by the GNN also appears in the retrieved "
                "neighborhood's measured results. Treat the two sources as "
                "independent lines of evidence.\n")
    lines = [
        "## AGREEMENT between the two sources",
        "",
        "These models are BOTH ranked highly by the GNN AND have real measured "
        "results on datasets similar to the query:",
        "",
    ]
    for rec in hits:
        lines.append(
            f"- {rec['name']} (GNN rank {rec['rank']}, combined score "
            f"{rec['combined_score']:.4f}; also measured on similar datasets in Source A)"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Merge Method 1 + Method 2 contexts into one LLM prompt.")
    parser.add_argument("--method1-md", required=True, help="graph_context.md from build_graph_context.py")
    parser.add_argument("--method2-md", required=True, help="gnn_context.md from build_gnn_context.py")
    parser.add_argument("--method1-json", help="graph_context.json (enables the agreement section)")
    parser.add_argument("--method2-json", help="gnn_context.json (enables the agreement section)")
    parser.add_argument("--num-recommendations", type=int, default=5)
    parser.add_argument("--constraint", default="",
                        help="Hard constraint text inserted into the instruction, e.g. "
                             "'Only recommend models with at most 3B parameters.'")
    parser.add_argument("--out", default="merged_prompt.md", help="Output prompt path")
    args = parser.parse_args()

    with open(args.method1_md, encoding="utf-8") as f:
        m1_md = f.read().strip()
    with open(args.method2_md, encoding="utf-8") as f:
        m2_md = f.read().strip()

    m1_ctx = m2_ctx = None
    if args.method1_json and args.method2_json:
        with open(args.method1_json, encoding="utf-8") as f:
            m1_ctx = json.load(f)
        with open(args.method2_json, encoding="utf-8") as f:
            m2_ctx = json.load(f)

    constraint = ""
    if args.constraint:
        constraint = f"\nHARD CONSTRAINT: {args.constraint} Only recommend models that appear in the evidence below.\n"
    parts = [
        INSTRUCTION_TEMPLATE.format(n=args.num_recommendations, constraint=constraint),
        "",
        "=" * 70,
        "# SOURCE A - retrieved neighborhood (real measured results)",
        "=" * 70,
        "",
        m1_md,
        "",
        "=" * 70,
        "# SOURCE B - GNN ranking (predictions + each model's measured track record)",
        "=" * 70,
        "",
        m2_md,
    ]
    agree = agreement_section(m1_ctx, m2_ctx)
    if agree:
        parts += ["", "=" * 70, agree]

    prompt = "\n".join(parts) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(prompt)

    print(f"Merged prompt: ~{len(prompt.split())} words")
    if agree and "No model" not in agree:
        n_agree = agree.count("\n- ")
        print(f"Agreement section: {n_agree} model(s) supported by both sources")
    elif agree:
        print("Agreement section: no overlap between sources")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
