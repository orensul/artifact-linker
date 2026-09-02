#!/usr/bin/env python3
"""Rank model candidates for a Hugging Face dataset with ArtifactLinker's joint GNN.

Given a Hugging Face dataset URL/repo id, dataset node id, or dataset name, this
script resolves the corresponding dataset node in the downloaded ArtifactBench
split, scores candidate model-dataset pairs with the trained joint GNN, and
ranks by the ArtifactLinker-style combined score:

    combined_score = link_probability * predicted_attribute_value

If the dataset is missing, pass --augment-missing to create an augmented copy of
the split, append the dataset as an isolated node, embed its HF README with the
same Voyage model used by ArtifactLinker (voyage-3), then score it. The official
split is never modified. Add --summarize-card to first summarize the README with
GPT-4o using the same prompt that built the benchmark node summaries (recommended:
the existing node embeddings were computed on such summaries, not raw cards).

Notes:
  * Existing-node scoring matches the trained transductive graph nodes.
  * Missing-node scoring is an ad hoc inductive use of the trained checkpoint:
    the new dataset has a text embedding but no graph neighbors.
  * predicted_attribute_value is the model's normalized attribute-head output,
    not a measured benchmark result.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import torch

ROOT = Path(os.environ.get('AGENTS_PROJECT_ROOT', Path(__file__).resolve().parent))
REPO = ROOT / 'external' / 'artifact-linker'
sys.path.append(str(REPO))

from artifact_graph.runners.joint_gnn_runner import load_joint_model  # noqa: E402
from artifact_graph.runners.runner_utils import load_node_embeddings  # noqa: E402


class DatasetNotFound(Exception):
    def __init__(self, query: str, parsed: str, nearby: list[tuple[int, dict[str, Any]]]):
        self.query = query
        self.parsed = parsed
        self.nearby = nearby
        super().__init__(f'No ArtifactBench dataset node matched {query!r} parsed as {parsed!r}.')


class AmbiguousDataset(Exception):
    def __init__(self, query: str, parsed: str, candidates: list[tuple[int, dict[str, Any]]]):
        self.query = query
        self.parsed = parsed
        self.candidates = candidates
        super().__init__(f'Ambiguous dataset query {query!r} parsed as {parsed!r}.')


def parse_dataset_query(value: str) -> str:
    """Accept a HF dataset URL, repo id, node id, or plain name."""
    value = value.strip()
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        parts = [p for p in parsed.path.split('/') if p]
        if parts[:1] == ['datasets'] and len(parts) >= 3:
            return '/'.join(parts[1:3])
        if len(parts) >= 2:
            return '/'.join(parts[:2])
    return value.rstrip('/')


def normalize_name(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', value.lower())


def safe_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('_') or 'dataset'


def load_metadata(split_dir: Path) -> dict[int, dict[str, Any]]:
    with (split_dir / 'train_split' / 'node_metadata.json').open(encoding='utf-8') as f:
        return {int(k): v for k, v in json.load(f).items()}


def write_metadata(path: Path, metadata: dict[int, dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in sorted(metadata.items())}, f)


def resolve_dataset(query: str, metadata: dict[int, dict[str, Any]]) -> tuple[int, dict[str, Any], str]:
    datasets = [(nid, meta) for nid, meta in metadata.items() if meta.get('type') == 'dataset']
    parsed = parse_dataset_query(query)

    if parsed.isdigit():
        nid = int(parsed)
        meta = metadata.get(nid)
        if meta and meta.get('type') == 'dataset':
            return nid, meta, 'node_id'
        raise DatasetNotFound(query, parsed, [])

    q_lower = parsed.lower()
    exact = [(nid, meta) for nid, meta in datasets if str(meta.get('name', '')).lower() == q_lower]
    if len(exact) == 1:
        return exact[0][0], exact[0][1], 'exact_name'

    contains = [(nid, meta) for nid, meta in datasets if q_lower in str(meta.get('name', '')).lower()]
    if len(contains) == 1:
        return contains[0][0], contains[0][1], 'substring'

    suffix = parsed.split('/')[-1]
    norm_query = normalize_name(suffix)
    normalized = [
        (nid, meta) for nid, meta in datasets
        if normalize_name(str(meta.get('name', '')).split('/')[-1]) == norm_query
    ]
    if len(normalized) == 1:
        return normalized[0][0], normalized[0][1], 'normalized_suffix'

    partial_normalized = [
        (nid, meta) for nid, meta in datasets
        if norm_query and norm_query in normalize_name(str(meta.get('name', '')))
    ]
    if len(partial_normalized) == 1:
        return partial_normalized[0][0], partial_normalized[0][1], 'normalized_partial'

    candidates = contains or normalized or partial_normalized
    if candidates:
        raise AmbiguousDataset(query, parsed, candidates)

    nearby = [
        (nid, meta) for nid, meta in datasets
        if normalize_name(suffix)[:6] and normalize_name(suffix)[:6] in normalize_name(str(meta.get('name', '')))
    ][:25]
    raise DatasetNotFound(query, parsed, nearby)


def print_resolution_error(exc: Exception) -> None:
    if isinstance(exc, AmbiguousDataset):
        print(str(exc), file=sys.stderr)
        for nid, meta in exc.candidates[:40]:
            print(f'  {nid}\t{meta.get("name")}', file=sys.stderr)
        if len(exc.candidates) > 40:
            print(f'  ... {len(exc.candidates) - 40} more', file=sys.stderr)
        print('Please rerun with an exact ArtifactBench dataset name or numeric node id.', file=sys.stderr)
        return
    if isinstance(exc, DatasetNotFound):
        print(str(exc), file=sys.stderr)
        if exc.nearby:
            print('Nearby-looking dataset nodes:', file=sys.stderr)
            for nid, meta in exc.nearby:
                print(f'  {nid}\t{meta.get("name")}', file=sys.stderr)
        print('Use --augment-missing to append this HF dataset as a new isolated node in an augmented split copy.', file=sys.stderr)


def fetch_dataset_text(repo_id: str, limit: int = 8000) -> str:
    """Fetch dataset-card text from Hugging Face; fall back to repo id if absent."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit('huggingface_hub is required to fetch missing dataset cards.') from exc

    try:
        readme_path = hf_hub_download(repo_id=repo_id, repo_type='dataset', filename='README.md')
        text = Path(readme_path).read_text(encoding='utf-8', errors='replace')
        if text.strip():
            return text[:limit]
    except Exception as exc:
        print(f'Warning: could not fetch README.md for {repo_id}: {exc}', file=sys.stderr)
    return f'{repo_id}: Hugging Face dataset.'


# Prompts verbatim from the ArtifactBench-building pipeline
# (scripts/step5_summarize_and_normalize.py), same as retrieval Method 1.
SUMMARIZE_SYSTEM = (
    'You are a strict JSON generator. Return ONLY a valid JSON object with the key:\n'
    '- "info": string (150-250 word summary)\n'
    'Use double quotes for keys/strings. No extra text.'
)

DATASET_INSTRUCTION = (
    'Summarize the following dataset README. Focus on what the dataset contains, its format, '
    'size, domain, and intended use cases. Keep 150-250 words.\n'
    'CRITICAL: Do NOT include any evaluation results, benchmark scores, metric values, '
    'or performance numbers. These must be completely excluded to prevent information leakage.\n'
    'Return only the JSON object.'
)


def summarize_dataset_card(card: str, llm_model: str = 'gpt-4o') -> str:
    """GPT-4o summary of the dataset card, matching the benchmark node summaries."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit('openai is required for --summarize-card. Install it in the venv first.') from exc

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise SystemExit('OPENAI_API_KEY is required for --summarize-card.')
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=llm_model,
        messages=[
            {'role': 'system', 'content': SUMMARIZE_SYSTEM},
            {'role': 'user', 'content': f'{DATASET_INSTRUCTION}\n\n{card[:12000]}'},
        ],
        temperature=0,
        max_tokens=1024,
    )
    raw = (resp.choices[0].message.content or '').strip().replace('```json', '').replace('```', '').strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and obj.get('info'):
            return str(obj['info'])
    except Exception:
        pass
    start, end = raw.find('{'), raw.rfind('}')
    if start != -1 and end != -1:
        try:
            obj = json.loads(raw[start:end + 1])
            if isinstance(obj, dict) and obj.get('info'):
                return str(obj['info'])
        except Exception:
            pass
    return raw[:500]


def voyage_embed_one(text: str) -> np.ndarray:
    api_key = os.environ.get('VOYAGE_API_KEY')
    if not api_key:
        raise SystemExit('VOYAGE_API_KEY is required for --augment-missing.')
    try:
        import voyageai
    except ImportError as exc:
        raise SystemExit('voyageai is required for --augment-missing. Install it in the venv first, e.g. pip install voyageai.') from exc

    client = voyageai.Client(api_key=api_key)
    result = client.embed([text], model='voyage-3', input_type='document')
    return np.array(result.embeddings, dtype=np.float32)


def random_embed_one(dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    emb = rng.normal(0, 1, (1, dim)).astype(np.float32)
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)


def copy_split_if_needed(src: Path, dst: Path, force: bool) -> None:
    if dst.exists():
        if not force:
            return
        shutil.rmtree(dst)
    shutil.copytree(src.resolve(), dst)


def augment_split_with_dataset(
    source_split: Path,
    dataset_query: str,
    output_split: Path | None,
    force_rebuild: bool,
    summarize_card: bool = False,
) -> tuple[Path, int, dict[str, Any], str]:
    repo_id = parse_dataset_query(dataset_query)
    if '/' not in repo_id:
        raise SystemExit(f'--augment-missing requires a Hugging Face dataset repo id or URL, got {dataset_query!r}')

    augmented_dir = output_split or (
        source_split.parent / 'augmented_splits' / f'{safe_name(repo_id)}_transductive'
    )
    copy_split_if_needed(source_split, augmented_dir, force_rebuild)

    metadata = load_metadata(augmented_dir)
    try:
        dataset_id, dataset_meta, resolution = resolve_dataset(repo_id, metadata)
        return augmented_dir, dataset_id, dataset_meta, f'augmented_existing_{resolution}'
    except DatasetNotFound:
        pass

    voyage_path = augmented_dir / 'node_embeddings_voyage.npy'
    voyage = np.load(voyage_path)
    if hasattr(voyage.dtype, 'names') and voyage.dtype.names and 'embedding' in voyage.dtype.names:
        raise SystemExit('Structured node_embeddings_voyage.npy is not supported by this augmentation helper.')
    if voyage.ndim != 2:
        raise SystemExit(f'Unexpected voyage embedding shape: {voyage.shape}')

    max_id = max(metadata)
    new_id = int(voyage.shape[0])
    if new_id != max_id + 1:
        raise SystemExit(f'Cannot safely append: max metadata id={max_id}, embedding rows={voyage.shape[0]}')

    if summarize_card:
        card = fetch_dataset_text(repo_id, limit=12000)
        text = summarize_dataset_card(card)
        embed_text = text[:8000]
    else:
        text = fetch_dataset_text(repo_id)
        embed_text = text
    new_voyage = voyage_embed_one(embed_text)
    if new_voyage.shape[1] != voyage.shape[1]:
        raise SystemExit(f'Voyage dim mismatch: existing={voyage.shape[1]}, new={new_voyage.shape[1]}')

    entry = {
        'type': 'dataset',
        'name': repo_id,
        'downloads': 0,
        'info': text,
        'source': 'added_by_rank_hf_dataset_combined',
    }
    for split_name in ('train_split', 'test_split'):
        meta_path = augmented_dir / split_name / 'node_metadata.json'
        split_meta = load_metadata(augmented_dir) if split_name == 'train_split' else {
            int(k): v for k, v in json.load(meta_path.open(encoding='utf-8')).items()
        }
        split_meta[new_id] = entry
        write_metadata(meta_path, split_meta)

    np.save(voyage_path, np.vstack([voyage, new_voyage]))

    random_path = augmented_dir / 'node_embeddings_random.npy'
    if random_path.exists():
        random_emb = np.load(random_path)
        if random_emb.ndim == 2:
            np.save(random_path, np.vstack([random_emb, random_embed_one(random_emb.shape[1], 42 + new_id)]))

    return augmented_dir, new_id, entry, 'augmented_new_isolated_node'


def observed_model_dataset_edges(split_dir: Path, metadata: dict[int, dict[str, Any]]) -> set[tuple[int, int]]:
    observed: set[tuple[int, int]] = set()
    for split_name in ('train_split', 'test_split'):
        path = split_dir / split_name / 'pos_edges.npz'
        if not path.exists():
            continue
        arr = np.load(path)['edges']
        for i in range(arr.shape[1]):
            u, v = int(arr[0, i]), int(arr[1, i])
            um, vm = metadata.get(u, {}), metadata.get(v, {})
            if um.get('type') == 'model' and vm.get('type') == 'dataset':
                observed.add((u, v))
            elif um.get('type') == 'dataset' and vm.get('type') == 'model':
                observed.add((v, u))
    return observed


def load_edge_metrics(split_dir: Path, model_id: int, dataset_id: int) -> dict[str, Any] | None:
    edge_keys = [f'{model_id},{dataset_id}', f'{dataset_id},{model_id}']
    for split_name in ('train_split', 'test_split'):
        path = split_dir / split_name / 'edge_metadata_normalized.json'
        if not path.exists():
            continue
        with path.open(encoding='utf-8') as f:
            edge_meta = json.load(f)
        for key in edge_keys:
            if key in edge_meta:
                return {'split': split_name, 'edge_key': key, 'metrics': edge_meta[key]}
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description='Rank models for one HF dataset using link * attribute score.')
    parser.add_argument('dataset', help='HF dataset URL/repo id, ArtifactBench dataset name, or dataset node id')
    parser.add_argument('--top-k', type=int, default=25)
    parser.add_argument('--include-observed', action='store_true', help='Do not filter known train/test positive edges')
    parser.add_argument('--augment-missing', action='store_true', help='If dataset is missing, add it as an isolated node in an augmented split copy')
    parser.add_argument('--summarize-card', action='store_true', help='With --augment-missing: summarize the HF card with GPT-4o (same prompt as the benchmark node summaries) before Voyage embedding, instead of embedding the raw card text')
    parser.add_argument('--augmented-split-dir', default='', help='Where to create/reuse the augmented split; default is data/augmented_splits/<repo>_transductive')
    parser.add_argument('--force-rebuild-augmented', action='store_true', help='Delete and rebuild the augmented split directory before appending')
    parser.add_argument('--split-dir', default=str(REPO / 'data' / 'artifact_graph_splits_v3_0314_transductive'))
    parser.add_argument('--model-path', default=str(REPO / 'data' / 'joint_sweep_gatv2_trans' / 'trans_joint_gatv2_model_emb.pth'))
    parser.add_argument('--output', default='', help='Output JSON path; default writes under data/custom_combined_predictions')
    parser.add_argument('--csv', nargs='?', const='', default=None, metavar='PATH',
                        help='Also write the ranked results as CSV; optional path, defaults to the JSON path with .csv extension')
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    metadata = load_metadata(split_dir)
    augmented_from: str | None = None
    try:
        dataset_id, dataset_meta, resolution = resolve_dataset(args.dataset, metadata)
    except (DatasetNotFound, AmbiguousDataset) as exc:
        if not args.augment_missing or isinstance(exc, AmbiguousDataset):
            print_resolution_error(exc)
            raise SystemExit(2) from exc
        augmented_from = str(split_dir)
        split_dir, dataset_id, dataset_meta, resolution = augment_split_with_dataset(
            split_dir,
            args.dataset,
            Path(args.augmented_split_dir) if args.augmented_split_dir else None,
            args.force_rebuild_augmented,
            summarize_card=args.summarize_card,
        )
        metadata = load_metadata(split_dir)

    dataset_name = dataset_meta.get('name')
    model_ids = sorted(nid for nid, meta in metadata.items() if meta.get('type') == 'model')
    observed = observed_model_dataset_edges(split_dir, metadata)
    candidates = [mid for mid in model_ids if args.include_observed or (mid, dataset_id) not in observed]
    if not candidates:
        raise SystemExit('No candidate model links remain after filtering observed positives.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = load_node_embeddings(split_dir, 'embedding').to(device)
    edge_index = torch.from_numpy(np.load(split_dir / 'test_split' / 'edges.npz')['edges']).long().to(device)
    model = load_joint_model(args.model_path, device)

    scored: list[dict[str, Any]] = []
    batch_size = 200_000
    with torch.no_grad():
        z = model.encode(x, edge_index)
        for start in range(0, len(candidates), batch_size):
            mids = candidates[start:start + batch_size]
            pairs = torch.tensor([[mid, dataset_id] for mid in mids], dtype=torch.long, device=device).t()
            link = torch.sigmoid(model.decode_link(z, pairs)).detach().cpu().numpy().reshape(-1)
            attr_logits = model.decode_attr(z, pairs).squeeze(-1)
            attr = torch.sigmoid(torch.clamp(attr_logits, -10, 10)).detach().cpu().numpy().reshape(-1)
            for mid, link_score, attr_score in zip(mids, link, attr):
                link_f = float(link_score)
                attr_f = float(attr_score)
                scored.append({
                    'model_id': int(mid),
                    'model_name': metadata.get(int(mid), {}).get('name'),
                    'dataset_id': dataset_id,
                    'dataset_name': dataset_name,
                    'link_probability': link_f,
                    'predicted_attribute_value': attr_f,
                    'combined_score': link_f * attr_f,
                    'observed_positive': (int(mid), dataset_id) in observed,
                })

    scored.sort(key=lambda row: row['combined_score'], reverse=True)
    rows = []
    for rank, row in enumerate(scored[:args.top_k], start=1):
        row = dict(row)
        row['rank'] = rank
        row['observed_edge_metadata'] = load_edge_metrics(split_dir, row['model_id'], dataset_id)
        rows.append(row)

    out = Path(args.output) if args.output else (
        REPO / 'data' / 'custom_combined_predictions' / f'dataset_{dataset_id}_{safe_name(str(dataset_name or dataset_id))}_top{args.top_k}_combined.json'
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'input_dataset': args.dataset,
        'parsed_dataset': parse_dataset_query(args.dataset),
        'resolution': resolution,
        'dataset_id': dataset_id,
        'dataset_name': dataset_name,
        'checkpoint': str(Path(args.model_path)),
        'split_dir': str(split_dir),
        'augmented_from_split_dir': augmented_from,
        'summarize_card': args.summarize_card,
        'candidate_count': len(candidates),
        'filtered_observed_positives': not args.include_observed,
        'top_k': args.top_k,
        'ranking_key': 'combined_score',
        'combined_score_formula': 'link_probability * predicted_attribute_value',
        'attribute_value_note': 'predicted_attribute_value is ArtifactLinker normalized attribute-head output, not a verified measured score.',
        'missing_dataset_note': 'If resolution is augmented_new_isolated_node, this is an ad hoc inductive use: the new dataset has a Voyage text embedding but no graph neighbors.',
        'results': rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    csv_path = None
    if args.csv is not None:
        import csv

        csv_path = Path(args.csv) if args.csv else out.with_suffix('.csv')
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = ['method', 'rank', 'query_dataset', 'model_id', 'model_name',
                  'combined_score', 'link_probability', 'predicted_attribute_value', 'observed_positive']
        with csv_path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, 'method': 'method2_gnn_direct', 'query_dataset': dataset_name})

    print(f'Input dataset: {args.dataset}')
    print(f'Resolved:      {dataset_id}\t{dataset_name} [{resolution}]')
    print(f'Split dir:     {split_dir}')
    print(f'Candidates:    {len(candidates)}')
    print(f'Saved:         {out}')
    if csv_path is not None:
        print(f'Saved CSV:     {csv_path}')
    for row in rows[:min(10, len(rows))]:
        print(
            f"{row['rank']:>3}  combined={row['combined_score']:.6f}  "
            f"link={row['link_probability']:.6f}  "
            f"attr={row['predicted_attribute_value']:.6f}  "
            f"{row['model_id']}  {row['model_name']}"
        )


if __name__ == '__main__':
    main()
