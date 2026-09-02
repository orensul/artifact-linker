#!/usr/bin/env python3
"""
Retrieval agent - Method 1, step 5 input: graph-conditioned prompt context.

Takes the similar-dataset ranking produced by retrieve_similar_datasets.py and
extracts, for the top-K neighbor datasets, their ArtifactBench neighborhood:

  content   each neighbor's card summary, plus (optionally) summaries of the
            strongest candidate models
  topology  the eval edges (model -> dataset, with measured metrics), which
            models recur across several neighbors, and base-model lineage
            edges among the involved models

The result is written as structured JSON and as a markdown context block ready
to inject into an LLM prompt ("given this evidence, recommend models for the
query dataset").

Usage:
    python build_graph_context.py \
        --similarities out/similarities.json \
        --query-name tau/commonsense_qa \
        --query-summary-file out/summary.txt \
        --data-dir path/to/artifact_bench_full \
        --top-k 10 --max-models 15 \
        --out-text out/graph_context.md --out-json out/graph_context.json

Needs no API keys and no GPU: it only reads the benchmark files.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(
    HERE, "..", "..", "artifact_bench", "artifact_bench_data", "full"
)
sys.path.insert(0, os.path.dirname(HERE))
from model_size import passes_max_params  # noqa: E402


def load_benchmark(data_dir):
    with open(os.path.join(data_dir, "node_metadata.json")) as f:
        meta = {int(k): v for k, v in json.load(f).items()}

    # dataset node id -> [(model node id, {metric: value})]
    evals = defaultdict(list)
    with open(os.path.join(data_dir, "edge_metadata.json")) as f:
        for key, edge in json.load(f).items():
            if edge.get("edge_type") != "eval":
                continue
            a, b = (int(x) for x in key.split(","))
            mid, did = (a, b) if meta.get(a, {}).get("type") == "model" else (b, a)
            evals[did].append((mid, edge.get("metrics", {})))

    # model node id -> [base model node id]
    base_of = defaultdict(list)
    base_path = os.path.join(data_dir, "edge_metadata_base_model.json")
    if os.path.exists(base_path):
        with open(base_path) as f:
            for key in json.load(f):
                src, dst = (int(x) for x in key.split(","))
                base_of[src].append(dst)
    return meta, evals, base_of


def primary_metric_value(metrics):
    """Single number used only for sorting models within a dataset."""
    for name in ("accuracy", "f1", "bleu", "chrf", "rouge-l", "rouge-2", "top-k_accuracy"):
        if name in metrics:
            return metrics[name]
    vals = [v for v in metrics.values() if isinstance(v, (int, float))]
    return max(vals) if vals else 0.0


def fmt_metrics(metrics):
    return ", ".join(f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}"
                     for k, v in metrics.items()) or "no recorded metrics"


def build_context(similarities, meta, evals, base_of, query_name, query_summary,
                  top_k, max_models, candidate_summaries, keep_self, max_params_b=None):
    norm = lambda s: "".join(c for c in s.lower() if c.isalnum())
    neighbors = []
    for row in similarities:
        if len(neighbors) >= top_k:
            break
        nid = int(row["node_id"])
        name = row["name"]
        if not keep_self and (norm(name) == norm(query_name)
                              or norm(name.split("/")[-1]) == norm(query_name.split("/")[-1])):
            continue  # skip the query dataset itself in sanity runs on in-graph datasets
        models = sorted(evals.get(nid, []), key=lambda mv: primary_metric_value(mv[1]), reverse=True)
        if max_params_b is not None:
            models = [(mid, mm) for mid, mm in models
                      if passes_max_params(meta.get(mid, {}).get("name", ""), max_params_b)]
        neighbors.append({
            "node_id": nid,
            "name": name,
            "cosine_similarity": row["cosine_similarity"],
            "summary": meta.get(nid, {}).get("info", ""),
            "num_evaluated_models": len(models),
            "models": [
                {
                    "node_id": mid,
                    "name": meta.get(mid, {}).get("name", str(mid)),
                    "downloads": meta.get(mid, {}).get("downloads", 0),
                    "metrics": metrics,
                }
                for mid, metrics in models[:max_models]
            ],
        })

    # Topology across neighbors: models evaluated on several of the top-K datasets.
    seen_on = defaultdict(list)
    for nb in neighbors:
        for m in nb["models"]:
            seen_on[m["node_id"]].append((nb["name"], m["metrics"]))
    recurring = sorted(
        ((mid, hits) for mid, hits in seen_on.items() if len(hits) >= 2),
        key=lambda x: (-len(x[1]), -sum(primary_metric_value(h[1]) for h in x[1]) / len(x[1])),
    )
    cross = [
        {
            "node_id": mid,
            "name": meta.get(mid, {}).get("name", str(mid)),
            "num_neighbor_datasets": len(hits),
            "results": [{"dataset": d, "metrics": mm} for d, mm in hits],
            "summary": (meta.get(mid, {}).get("info", "") if rank < candidate_summaries else ""),
        }
        for rank, (mid, hits) in enumerate(recurring)
    ]

    # Lineage edges among all involved models.
    involved = set(seen_on)
    lineage = []
    for mid in sorted(involved):
        for base in base_of.get(mid, []):
            if base in involved:
                lineage.append({
                    "model": meta.get(mid, {}).get("name", str(mid)),
                    "base_model": meta.get(base, {}).get("name", str(base)),
                })

    return {
        "query_dataset": query_name,
        "query_summary": query_summary,
        "max_params_b": max_params_b,
        "neighbor_datasets": neighbors,
        "models_on_multiple_neighbors": cross,
        "base_model_lineage": lineage,
    }


def render_markdown(ctx):
    lines = [
        f"# Graph evidence for query dataset: {ctx['query_dataset']}",
        "",
        "The query dataset is new and has no evaluation history. Below is the "
        "neighborhood of the most semantically similar benchmark datasets in the "
        "ArtifactBench graph: what each contains, which models were actually "
        "evaluated on them, and the measured results.",
        "",
        "## Query dataset summary",
        ctx["query_summary"] or "(no summary provided)",
        "",
        "## Similar benchmark datasets and their evaluated models",
    ]
    if ctx.get("max_params_b") is not None:
        lines.insert(4, f"CONSTRAINT: only models with at most {ctx['max_params_b']:g}B "
                        "parameters are listed (larger models were removed).")
        lines.insert(5, "")
    for i, nb in enumerate(ctx["neighbor_datasets"], 1):
        lines += [
            "",
            f"### {i}. {nb['name']}  (cosine similarity {nb['cosine_similarity']:.4f})",
            f"{nb['summary']}",
            "",
            f"Measured results ({min(len(nb['models']), nb['num_evaluated_models'])} of "
            f"{nb['num_evaluated_models']} evaluated models, best first):",
        ]
        for m in nb["models"]:
            lines.append(f"- {m['name']}: {fmt_metrics(m['metrics'])} "
                         f"(downloads: {m['downloads']:,})")

    if ctx["models_on_multiple_neighbors"]:
        lines += [
            "",
            "## Models with measured results on several of these similar datasets",
            "(consistent performance across the neighborhood is the strongest evidence)",
        ]
        for m in ctx["models_on_multiple_neighbors"]:
            per_ds = "; ".join(f"{r['dataset']}: {fmt_metrics(r['metrics'])}" for r in m["results"])
            lines.append(f"- {m['name']} — on {m['num_neighbor_datasets']} datasets — {per_ds}")
            if m["summary"]:
                lines.append(f"  - About this model: {m['summary']}")

    if ctx["base_model_lineage"]:
        lines += ["", "## Model lineage among the models above"]
        for e in ctx["base_model_lineage"]:
            lines.append(f"- {e['model']} is derived from {e['base_model']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Build LLM prompt context from the graph neighborhood of retrieved similar datasets."
    )
    parser.add_argument("--similarities", required=True,
                        help="similarities.json written by retrieve_similar_datasets.py --save-dir")
    parser.add_argument("--query-name", required=True, help="Name of the query dataset (for the prompt header)")
    parser.add_argument("--query-summary-file", help="summary.txt written by retrieve_similar_datasets.py")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="ArtifactBench full-split dir (node_metadata.json, edge_metadata*.json)")
    parser.add_argument("--top-k", type=int, default=10, help="Neighbor datasets to include")
    parser.add_argument("--max-models", type=int, default=15, help="Models listed per neighbor dataset")
    parser.add_argument("--candidate-summaries", type=int, default=8,
                        help="Include full model summaries for the top N recurring candidate models")
    parser.add_argument("--keep-self", action="store_true",
                        help="Do not drop a neighbor whose name matches the query dataset")
    parser.add_argument("--max-params-b", type=float, default=None,
                        help="Only include models with <= this many billions of parameters "
                             "(models with undeterminable size are excluded)")
    parser.add_argument("--out-text", default="graph_context.md", help="Markdown context output path")
    parser.add_argument("--out-json", default="graph_context.json", help="Structured JSON output path")
    args = parser.parse_args()

    with open(args.similarities) as f:
        similarities = json.load(f)
    query_summary = ""
    if args.query_summary_file:
        with open(args.query_summary_file, encoding="utf-8") as f:
            query_summary = f.read().strip()

    meta, evals, base_of = load_benchmark(args.data_dir)
    ctx = build_context(similarities, meta, evals, base_of,
                        args.query_name, query_summary,
                        args.top_k, args.max_models, args.candidate_summaries, args.keep_self,
                        max_params_b=args.max_params_b)

    for path in (args.out_text, args.out_json):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
    text = render_markdown(ctx)
    with open(args.out_text, "w", encoding="utf-8") as f:
        f.write(text)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(ctx, f, indent=2, ensure_ascii=False)

    n_models = sum(len(nb["models"]) for nb in ctx["neighbor_datasets"])
    print(f"Neighbors: {len(ctx['neighbor_datasets'])} datasets, {n_models} model results, "
          f"{len(ctx['models_on_multiple_neighbors'])} recurring models, "
          f"{len(ctx['base_model_lineage'])} lineage edges")
    print(f"Context: ~{len(text.split())} words")
    print(f"Saved: {args.out_text}")
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
