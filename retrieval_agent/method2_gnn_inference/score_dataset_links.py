#!/usr/bin/env python3
"""Score candidate model links for one existing ArtifactBench dataset node.

This is a project-local utility. It does not modify ArtifactLinker's official code;
it loads the official trained joint checkpoint and uses its link decoder.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get('AGENTS_PROJECT_ROOT', Path(__file__).resolve().parent))
REPO = ROOT / 'external' / 'artifact-linker'
sys.path.append(str(REPO))

from artifact_graph.runners.joint_gnn_runner import load_joint_model  # noqa: E402
from artifact_graph.runners.runner_utils import load_node_embeddings  # noqa: E402


def load_metadata(split_dir: Path) -> dict[int, dict]:
    with (split_dir / 'train_split' / 'node_metadata.json').open() as f:
        return {int(k): v for k, v in json.load(f).items()}


def find_dataset(query: str, metadata: dict[int, dict]) -> tuple[int, dict]:
    datasets = [(nid, meta) for nid, meta in metadata.items() if meta.get('type') == 'dataset']
    if query.isdigit():
        nid = int(query)
        meta = metadata.get(nid)
        if meta and meta.get('type') == 'dataset':
            return nid, meta
        raise SystemExit(f'No dataset node with id {nid}')

    q = query.lower()
    exact = [(nid, meta) for nid, meta in datasets if str(meta.get('name', '')).lower() == q]
    if len(exact) == 1:
        return exact[0]
    matches = [(nid, meta) for nid, meta in datasets if q in str(meta.get('name', '')).lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f'No dataset name matching {query!r}')

    print(f'Ambiguous dataset query {query!r}; matches:', file=sys.stderr)
    for nid, meta in matches[:25]:
        print(f'  {nid}\t{meta.get("name")}', file=sys.stderr)
    if len(matches) > 25:
        print(f'  ... {len(matches) - 25} more', file=sys.stderr)
    raise SystemExit('Please rerun with an exact name or numeric dataset_id.')


def observed_model_dataset_edges(split_dir: Path, metadata: dict[int, dict]) -> set[tuple[int, int]]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description='Rank model links for one ArtifactBench dataset.')
    parser.add_argument('dataset', help='Dataset node id, exact name, or unambiguous name substring')
    parser.add_argument('--top-k', type=int, default=25)
    parser.add_argument('--include-observed', action='store_true', help='Do not filter known train/test positives')
    parser.add_argument('--split-dir', default=str(REPO / 'data' / 'artifact_graph_splits_v3_0314_transductive'))
    parser.add_argument('--model-path', default=str(REPO / 'data' / 'joint_sweep_gatv2_trans' / 'trans_joint_gatv2_model_emb.pth'))
    parser.add_argument('--output', default='', help='Output JSON path; default writes under data/custom_link_predictions')
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    metadata = load_metadata(split_dir)
    dataset_id, dataset_meta = find_dataset(args.dataset, metadata)
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

    scores: list[float] = []
    batch_size = 200_000
    with torch.no_grad():
        z = model.encode(x, edge_index)
        for start in range(0, len(candidates), batch_size):
            mids = candidates[start:start + batch_size]
            pairs = torch.tensor([[mid, dataset_id] for mid in mids], dtype=torch.long, device=device).t()
            logits = model.decode_link(z, pairs)
            scores.extend(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1).tolist())

    ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
    rows = []
    for rank, (mid, score) in enumerate(ranked[:args.top_k], start=1):
        rows.append({
            'rank': rank,
            'model_id': mid,
            'model_name': metadata.get(mid, {}).get('name'),
            'dataset_id': dataset_id,
            'dataset_name': dataset_name,
            'link_probability': float(score),
            'observed_positive': (mid, dataset_id) in observed,
        })

    out = Path(args.output) if args.output else REPO / 'data' / 'custom_link_predictions' / f'dataset_{dataset_id}_top{args.top_k}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'dataset_id': dataset_id,
        'dataset_name': dataset_name,
        'checkpoint': str(Path(args.model_path)),
        'split_dir': str(split_dir),
        'candidate_count': len(candidates),
        'filtered_observed_positives': not args.include_observed,
        'top_k': args.top_k,
        'results': rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print(f'Dataset: {dataset_id}\t{dataset_name}')
    print(f'Candidates scored: {len(candidates)}')
    print(f'Saved: {out}')
    for row in rows[: min(10, len(rows))]:
        print(f"{row['rank']:>3}  {row['link_probability']:.6f}  {row['model_id']}  {row['model_name']}")


if __name__ == '__main__':
    main()
