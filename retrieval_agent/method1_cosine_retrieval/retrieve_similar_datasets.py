#!/usr/bin/env python3
"""
Retrieval agent - Method 1 (summary + embedding + cosine retrieval).

Give a cold-start dataset a neighborhood: a brand-new dataset is a node with no
edges, so no graph method can score it. This script induces a neighborhood by
ranking every dataset node already in ArtifactBench by semantic similarity.

Pipeline (identical to how the benchmark node embeddings were built):
  1. CARD     take the dataset's HuggingFace card (fetched live from the Hub,
              or a local README.md file)
  2. SUMMARY  summarize it with GPT-4o using the exact benchmark prompt
              (strict-JSON "info" key, 150-250 words, no benchmark scores,
              card truncated to 12,000 chars, temperature 0)
  3. EMBED    embed the summary with voyage-3 (input_type="document",
              summary truncated to 8,000 chars), L2-normalized, 1024-dim
  4. RETRIEVE cosine similarity against every `dataset` node row of
              node_embeddings_voyage.npy -> top-K similar datasets

The top-K neighbors (and the models evaluated on them) are then handed to the
downstream recommendation step (graph-conditioned LLM prompting).

Usage:
    python retrieve_similar_datasets.py openai/gsm8k
    python retrieve_similar_datasets.py my-dataset --card path/to/README.md
    python retrieve_similar_datasets.py openai/gsm8k --top-k 25 --save-dir out/

Requires OPENAI_API_KEY and VOYAGE_API_KEY.
"""

import argparse
import json
import os
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# ArtifactBench "full" split: node_metadata.json + node_embeddings_voyage.npy
DEFAULT_DATA_DIR = os.path.join(
    HERE, "..", "..", "artifact_bench", "artifact_bench_data", "full"
)

CARD_TRUNCATE = 12000    # chars of raw card sent to the LLM
SUMMARY_TRUNCATE = 8000  # chars of summary sent to Voyage

# ── Prompts (verbatim from the benchmark-building pipeline,
#    scripts/step5_summarize_and_normalize.py) ──────────────────────────────

SUMMARIZE_SYSTEM = (
    "You are a strict JSON generator. Return ONLY a valid JSON object with the key:\n"
    '- "info": string (150-250 word summary)\n'
    "Use double quotes for keys/strings. No extra text."
)

DATASET_INSTRUCTION = (
    "Summarize the following dataset README. Focus on what the dataset contains, its format, "
    "size, domain, and intended use cases. Keep 150-250 words.\n"
    "CRITICAL: Do NOT include any evaluation results, benchmark scores, metric values, "
    "or performance numbers. These must be completely excluded to prevent information leakage.\n"
    "Return only the JSON object."
)


def _parse_json_from_llm(raw: str) -> dict:
    """Parse JSON from LLM response, with fallback (verbatim from the pipeline)."""
    cleaned = (raw or "").strip().replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        try:
            obj = json.loads(cleaned[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {"info": cleaned[:500]}


def fetch_hf_card(name: str) -> str:
    url = f"https://huggingface.co/datasets/{name}/raw/main/README.md"
    req = urllib.request.Request(url, headers={"User-Agent": "artifact-linker"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def summarize_card(card: str, llm_model: str = "gpt-4o") -> str:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required for card summarization.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": SUMMARIZE_SYSTEM},
            {"role": "user", "content": f"{DATASET_INSTRUCTION}\n\n{card[:CARD_TRUNCATE]}"},
        ],
        temperature=0,
        max_tokens=1024,
    )
    parsed = _parse_json_from_llm(resp.choices[0].message.content or "")
    return parsed.get("info", "")


def embed_summary(summary: str, model_name: str = "voyage-3") -> np.ndarray:
    import voyageai

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise EnvironmentError("VOYAGE_API_KEY is required.")
    client = voyageai.Client(api_key=api_key)
    result = client.embed([summary[:SUMMARY_TRUNCATE]], model=model_name, input_type="document")
    vec = np.array(result.embeddings[0], dtype=np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


def rank_dataset_nodes(query_vec: np.ndarray, data_dir: str):
    """Return [(node_id, name, cosine_sim)] over all dataset nodes, best first."""
    with open(os.path.join(data_dir, "node_metadata.json")) as f:
        meta = json.load(f)
    emb = np.load(os.path.join(data_dir, "node_embeddings_voyage.npy")).astype(np.float32)

    dataset_ids = [int(k) for k, v in meta.items() if v.get("type") == "dataset"]
    sims = emb[dataset_ids] @ query_vec  # cosine similarity (both sides unit-norm)
    order = np.argsort(-sims)
    return [(dataset_ids[i], meta[str(dataset_ids[i])]["name"], float(sims[i])) for i in order]


def main():
    parser = argparse.ArgumentParser(
        description="Method 1: rank ArtifactBench dataset nodes by cosine similarity to a new dataset."
    )
    parser.add_argument("dataset", help="HuggingFace dataset id (owner/name), or a label when --card is given")
    parser.add_argument("--card", help="Path to a local card README.md (skips the Hub fetch)")
    parser.add_argument("--top-k", type=int, default=10, help="Results to print (-1 for all)")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="ArtifactBench full-split dir with node_metadata.json and node_embeddings_voyage.npy")
    parser.add_argument("--save-dir", help="Directory to save summary.txt, summary_voyage.npy, and similarities.json")
    parser.add_argument("--csv", nargs="?", const="", default=None, metavar="PATH",
                        help="Also write the ranked datasets as CSV; optional path, defaults to similarities.csv in --save-dir or cwd")
    args = parser.parse_args()

    if args.card:
        with open(args.card, encoding="utf-8", errors="ignore") as f:
            card = f.read()
        print(f"Loaded card from {args.card} ({len(card):,} chars)")
    else:
        print(f"Fetching card for {args.dataset} from the HuggingFace Hub...")
        card = fetch_hf_card(args.dataset)
        print(f"Fetched {len(card):,} chars")

    print("Summarizing with gpt-4o...")
    summary = summarize_card(card)
    print(f"\n--- Summary ({len(summary.split())} words) ---\n{summary}\n")

    print("Embedding with voyage-3...")
    query_vec = embed_summary(summary)

    ranked = rank_dataset_nodes(query_vec, args.data_dir)
    top = ranked if args.top_k == -1 else ranked[: args.top_k]
    print(f"\nTop {len(top)} of {len(ranked)} benchmark dataset nodes by cosine similarity:")
    print(f"{'rank':>4}  {'node':>6}  {'cosine':>7}  name")
    for rank, (nid, name, sim) in enumerate(top, 1):
        print(f"{rank:>4}  {nid:>6}  {sim:>7.4f}  {name}")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(summary)
        np.save(os.path.join(args.save_dir, "summary_voyage.npy"), query_vec)
        with open(os.path.join(args.save_dir, "similarities.json"), "w", encoding="utf-8") as f:
            json.dump(
                [{"node_id": nid, "name": name, "cosine_similarity": sim} for nid, name, sim in ranked],
                f, indent=2,
            )
        print(f"\nSaved summary, embedding, and full similarity list to {args.save_dir}/")

    if args.csv is not None:
        import csv

        csv_path = args.csv or os.path.join(args.save_dir or ".", "similarities.csv")
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["method", "rank", "query_dataset", "dataset_node_id", "dataset_name", "cosine_similarity"])
            for rank, (nid, name, sim) in enumerate(ranked, 1):
                writer.writerow(["method1_cosine_retrieval", rank, args.dataset, nid, name, sim])
        print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
