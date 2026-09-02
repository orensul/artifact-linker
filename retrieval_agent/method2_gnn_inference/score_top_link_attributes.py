#!/usr/bin/env python3
"""Predict attribute values for the top links from score_dataset_links.py output.

This is a project-local utility. It reads a custom link-prediction JSON file,
loads the trained ArtifactLinker joint checkpoint, and uses decode_attr for the
top N model-dataset pairs. It does one graph encoding pass for all requested
pairs.
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
    parser = argparse.ArgumentParser(description='Predict attributes for top ranked custom link predictions.')
    parser.add_argument('--links-json', default=str(REPO / 'data' / 'custom_link_predictions' / 'dataset_10194_top25.json'))
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--split-dir', default=str(REPO / 'data' / 'artifact_graph_splits_v3_0314_transductive'))
    parser.add_argument('--model-path', default=str(REPO / 'data' / 'joint_sweep_gatv2_trans' / 'trans_joint_gatv2_model_emb.pth'))
    parser.add_argument('--output', default='', help='Output JSON path; default is next to links-json')
    args = parser.parse_args()

    links_path = Path(args.links_json)
    payload = json.loads(links_path.read_text(encoding='utf-8'))
    rows = payload.get('results', [])[:args.top_n]
    if not rows:
        raise SystemExit(f'No results found in {links_path}')

    split_dir = Path(args.split_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = load_node_embeddings(split_dir, 'embedding').to(device)
    edge_index = torch.from_numpy(np.load(split_dir / 'test_split' / 'edges.npz')['edges']).long().to(device)
    model = load_joint_model(args.model_path, device)

    pairs = torch.tensor(
        [[int(row['model_id']), int(row['dataset_id'])] for row in rows],
        dtype=torch.long,
        device=device,
    ).t()

    with torch.no_grad():
        z = model.encode(x, edge_index)
        attr_logits = model.decode_attr(z, pairs).squeeze(-1)
        attr_values = torch.sigmoid(torch.clamp(attr_logits, -10, 10)).detach().cpu().numpy().reshape(-1)

    out_rows = []
    for row, attr_value in zip(rows, attr_values):
        model_id = int(row['model_id'])
        dataset_id = int(row['dataset_id'])
        out_row = dict(row)
        out_row['predicted_attribute_value'] = float(attr_value)
        out_row['attribute_value_note'] = 'decode_attr sigmoid output in ArtifactLinker normalized metric space; not independently verified accuracy.'
        observed = load_edge_metrics(split_dir, model_id, dataset_id)
        out_row['observed_edge'] = observed is not None
        out_row['observed_edge_metadata'] = observed
        out_rows.append(out_row)

    out_payload = {
        'source_links_json': str(links_path),
        'checkpoint': str(Path(args.model_path)),
        'split_dir': str(split_dir),
        'top_n': args.top_n,
        'dataset_id': payload.get('dataset_id'),
        'dataset_name': payload.get('dataset_name'),
        'attribute_value_note': 'Predictions are normalized ArtifactLinker attribute-head outputs, not verified measured scores.',
        'results': out_rows,
    }

    out_path = Path(args.output) if args.output else links_path.with_name(
        links_path.stem + f'_top{args.top_n}_attributes.json'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2), encoding='utf-8')

    print(f'Saved: {out_path}')
    for row in out_rows:
        print(
            f"{row['rank']:>3}  link={row['link_probability']:.6f}  "
            f"attr={row['predicted_attribute_value']:.6f}  "
            f"{row['model_id']}  {row['model_name']}"
        )


if __name__ == '__main__':
    main()
