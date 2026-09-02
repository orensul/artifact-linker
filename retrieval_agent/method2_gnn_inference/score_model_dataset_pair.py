#!/usr/bin/env python3
"""Score one model-dataset pair with a trained ArtifactLinker joint GNN.

This project-local utility does not modify ArtifactLinker's official code. It
loads the trained joint checkpoint and reports both heads:
  * link_probability: probability that the model-dataset edge should exist
  * predicted_attribute_value: sigmoid decode_attr output in the paper's
    normalized metric-value space, usually accuracy-like values in [0, 1]

It can only score nodes already present in the split's node_metadata and
node_embeddings array.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(os.environ.get('AGENTS_PROJECT_ROOT', Path(__file__).resolve().parent))
REPO = ROOT / 'external' / 'artifact-linker'
sys.path.append(str(REPO))

from artifact_graph.runners.joint_gnn_runner import load_joint_model  # noqa: E402
from artifact_graph.runners.runner_utils import load_node_embeddings  # noqa: E402


def load_metadata(split_dir: Path) -> dict[int, dict[str, Any]]:
    with (split_dir / 'train_split' / 'node_metadata.json').open(encoding='utf-8') as f:
        return {int(k): v for k, v in json.load(f).items()}


def find_node(query: str, node_type: str, metadata: dict[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    nodes = [(nid, meta) for nid, meta in metadata.items() if meta.get('type') == node_type]
    if query.isdigit():
        nid = int(query)
        meta = metadata.get(nid)
        if meta and meta.get('type') == node_type:
            return nid, meta
        raise SystemExit(f'No {node_type} node with id {nid}')

    q = query.lower()
    exact = [(nid, meta) for nid, meta in nodes if str(meta.get('name', '')).lower() == q]
    if len(exact) == 1:
        return exact[0]

    matches = [(nid, meta) for nid, meta in nodes if q in str(meta.get('name', '')).lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f'No {node_type} name matching {query!r}')

    print(f'Ambiguous {node_type} query {query!r}; matches:', file=sys.stderr)
    for nid, meta in matches[:25]:
        print(f'  {nid}\t{meta.get("name")}', file=sys.stderr)
    if len(matches) > 25:
        print(f'  ... {len(matches) - 25} more', file=sys.stderr)
    raise SystemExit(f'Please rerun with an exact {node_type} name or numeric id.')


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
    parser = argparse.ArgumentParser(description='Score one model-dataset pair with link and attribute heads.')
    parser.add_argument('--model', required=True, help='Model node id, exact name, or unambiguous name substring')
    parser.add_argument('--dataset', required=True, help='Dataset node id, exact name, or unambiguous name substring')
    parser.add_argument('--split-dir', default=str(REPO / 'data' / 'artifact_graph_splits_v3_0314_transductive'))
    parser.add_argument('--model-path', default=str(REPO / 'data' / 'joint_sweep_gatv2_trans' / 'trans_joint_gatv2_model_emb.pth'))
    parser.add_argument('--output', default='', help='Output JSON path; default writes under data/custom_attribute_predictions')
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    metadata = load_metadata(split_dir)
    model_id, model_meta = find_node(args.model, 'model', metadata)
    dataset_id, dataset_meta = find_node(args.dataset, 'dataset', metadata)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = load_node_embeddings(split_dir, 'embedding').to(device)
    edge_index = torch.from_numpy(np.load(split_dir / 'test_split' / 'edges.npz')['edges']).long().to(device)
    model = load_joint_model(args.model_path, device)
    pair = torch.tensor([[model_id], [dataset_id]], dtype=torch.long, device=device)

    with torch.no_grad():
        z = model.encode(x, edge_index)
        link_probability = torch.sigmoid(model.decode_link(z, pair)).item()
        attr_logit = model.decode_attr(z, pair).squeeze()
        predicted_attribute_value = torch.sigmoid(torch.clamp(attr_logit, -10, 10)).item()

    observed = load_edge_metrics(split_dir, model_id, dataset_id)
    payload = {
        'model_id': model_id,
        'model_name': model_meta.get('name'),
        'dataset_id': dataset_id,
        'dataset_name': dataset_meta.get('name'),
        'checkpoint': str(Path(args.model_path)),
        'split_dir': str(split_dir),
        'link_probability': float(link_probability),
        'predicted_attribute_value': float(predicted_attribute_value),
        'attribute_value_note': 'decode_attr sigmoid output in ArtifactLinker normalized metric space; not independently verified accuracy.',
        'observed_edge': observed is not None,
        'observed_edge_metadata': observed,
    }

    out = Path(args.output) if args.output else (
        REPO / 'data' / 'custom_attribute_predictions' / f'model_{model_id}_dataset_{dataset_id}.json'
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print(f"Model:   {model_id}\t{payload['model_name']}")
    print(f"Dataset: {dataset_id}\t{payload['dataset_name']}")
    print(f"link_probability:          {link_probability:.6f}")
    print(f"predicted_attribute_value: {predicted_attribute_value:.6f}")
    print(f"observed_edge:             {payload['observed_edge']}")
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
