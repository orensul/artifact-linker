#!/usr/bin/env python3
"""
Retrieval agent - Method 2 companion: GNN recommendations as LLM prompt context.

Takes the ranked-predictions JSON written by rank_hf_dataset_combined.py and
renders it as a prompt-ready markdown block, enriching each recommended model
with graph content:

  scores    the GNN outputs (combined = link_probability x predicted_attribute_value)
  content   the model's card summary and download count
  evidence  the model's measured track record: datasets it was actually
            evaluated on in ArtifactBench, with real metric values
  topology  base-model lineage among the recommended models

Usage:
    python build_gnn_context.py \
        --predictions data/custom_combined_predictions/dataset_10194_..._combined.json \
        --data-dir path/to/artifact_bench_full \
        --top-k 15 --max-evidence 5 \
        --out-text gnn_context.md --out-json gnn_context.json

Needs no API keys, no GPU, and no checkpoint: it only reads the prediction
JSON and the benchmark files.
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

    # model node id -> [(dataset node id, {metric: value})]
    track = defaultdict(list)
    with open(os.path.join(data_dir, "edge_metadata.json")) as f:
        for key, edge in json.load(f).items():
            if edge.get("edge_type") != "eval":
                continue
            a, b = (int(x) for x in key.split(","))
            mid, did = (a, b) if meta.get(a, {}).get("type") == "model" else (b, a)
            track[mid].append((did, edge.get("metrics", {})))

    base_of = defaultdict(list)
    base_path = os.path.join(data_dir, "edge_metadata_base_model.json")
    if os.path.exists(base_path):
        with open(base_path) as f:
            for key in json.load(f):
                src, dst = (int(x) for x in key.split(","))
                base_of[src].append(dst)
    return meta, track, base_of


def fmt_metrics(metrics):
    return ", ".join(f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}"
                     for k, v in metrics.items()) or "no recorded metrics"


def build_context(payload, meta, track, base_of, top_k, max_evidence, max_params_b=None):
    dataset_id = payload.get("dataset_id")
    query_summary = meta.get(int(dataset_id), {}).get("info", "") if dataset_id is not None else ""

    results = payload.get("results", [])
    if max_params_b is not None:
        results = [r for r in results
                   if passes_max_params(r.get("model_name") or "", max_params_b)]

    recs = []
    for row in results[:top_k]:
        mid = int(row["model_id"])
        # Exclude results on the query dataset itself: the prompt must not
        # contain the answer it is asking the LLM to predict.
        evidence = sorted((dm for dm in track.get(mid, [])
                           if dataset_id is None or dm[0] != int(dataset_id)),
                          key=lambda dm: max([v for v in dm[1].values()
                                              if isinstance(v, (int, float))] or [0]),
                          reverse=True)
        recs.append({
            "rank": row.get("rank"),
            "node_id": mid,
            "name": row.get("model_name") or meta.get(mid, {}).get("name", str(mid)),
            "combined_score": row.get("combined_score"),
            "link_probability": row.get("link_probability"),
            "predicted_attribute_value": row.get("predicted_attribute_value"),
            "downloads": meta.get(mid, {}).get("downloads", 0),
            "summary": meta.get(mid, {}).get("info", ""),
            "num_known_evaluations": len(evidence),
            "measured_track_record": [
                {"dataset": meta.get(did, {}).get("name", str(did)), "metrics": mm}
                for did, mm in evidence[:max_evidence]
            ],
        })

    rec_ids = {r["node_id"] for r in recs}
    lineage = []
    for r in recs:
        for base in base_of.get(r["node_id"], []):
            if base in rec_ids:
                lineage.append({
                    "model": r["name"],
                    "base_model": meta.get(base, {}).get("name", str(base)),
                })

    return {
        "query_dataset": payload.get("dataset_name") or payload.get("input_dataset"),
        "query_summary": query_summary,
        "max_params_b": max_params_b,
        "resolution": payload.get("resolution"),
        "candidate_count": payload.get("candidate_count"),
        "score_formula": payload.get("combined_score_formula",
                                     "link_probability * predicted_attribute_value"),
        "recommendations": recs,
        "base_model_lineage": lineage,
    }


def render_markdown(ctx):
    lines = [
        f"# GNN model recommendations for dataset: {ctx['query_dataset']}",
        "",
        f"A graph neural network trained on the ArtifactBench evaluation graph scored "
        f"{ctx['candidate_count']:,} candidate models for this dataset. Each model's score is "
        f"`{ctx['score_formula']}`, where link_probability estimates how plausible it is that "
        "this model would be evaluated on this dataset, and predicted_attribute_value is the "
        "predicted normalized performance (NOT a measured result). The measured track record "
        "listed under each model IS real: those are actual recorded results on other datasets.",
        "",
        "## Query dataset summary",
        ctx["query_summary"] or "(not in the benchmark; no stored summary)",
        "",
        "## Recommended models (ranked by GNN combined score)",
    ]
    if ctx.get("max_params_b") is not None:
        lines.insert(4, f"CONSTRAINT: only models with at most {ctx['max_params_b']:g}B "
                        "parameters are listed (larger models were removed; the rank numbers "
                        "below are the models' positions in the unfiltered GNN ranking).")
        lines.insert(5, "")
    for r in ctx["recommendations"]:
        lines += [
            "",
            f"### {r['rank']}. {r['name']}",
            f"- GNN scores: combined {r['combined_score']:.4f} = "
            f"link {r['link_probability']:.4f} x predicted-performance {r['predicted_attribute_value']:.4f}",
            f"- Downloads: {r['downloads']:,}",
        ]
        if r["summary"]:
            lines.append(f"- About this model: {r['summary']}")
        if r["measured_track_record"]:
            lines.append(f"- Measured results on other datasets "
                         f"({min(len(r['measured_track_record']), r['num_known_evaluations'])} of "
                         f"{r['num_known_evaluations']} known evaluations, best first):")
            for ev in r["measured_track_record"]:
                lines.append(f"  - {ev['dataset']}: {fmt_metrics(ev['metrics'])}")
        else:
            lines.append("- No measured evaluations recorded in the benchmark "
                         "(this recommendation rests on the GNN prediction alone).")

    if ctx["base_model_lineage"]:
        lines += ["", "## Model lineage among the recommendations"]
        for e in ctx["base_model_lineage"]:
            lines.append(f"- {e['model']} is derived from {e['base_model']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Render rank_hf_dataset_combined.py output as LLM prompt context."
    )
    parser.add_argument("--predictions", required=True,
                        help="JSON written by rank_hf_dataset_combined.py")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="ArtifactBench full-split dir (node_metadata.json, edge_metadata*.json)")
    parser.add_argument("--top-k", type=int, default=15, help="Recommendations to include")
    parser.add_argument("--max-evidence", type=int, default=5,
                        help="Measured results listed per model")
    parser.add_argument("--max-params-b", type=float, default=None,
                        help="Only include models with <= this many billions of parameters "
                             "(models with undeterminable size are excluded)")
    parser.add_argument("--out-text", default="gnn_context.md", help="Markdown context output path")
    parser.add_argument("--out-json", default="gnn_context.json", help="Structured JSON output path")
    args = parser.parse_args()

    with open(args.predictions) as f:
        payload = json.load(f)

    meta, track, base_of = load_benchmark(args.data_dir)
    ctx = build_context(payload, meta, track, base_of, args.top_k, args.max_evidence,
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

    with_evidence = sum(1 for r in ctx["recommendations"] if r["measured_track_record"])
    print(f"Recommendations: {len(ctx['recommendations'])} models "
          f"({with_evidence} with measured track records), "
          f"{len(ctx['base_model_lineage'])} lineage edges")
    print(f"Context: ~{len(text.split())} words")
    print(f"Saved: {args.out_text}")
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
