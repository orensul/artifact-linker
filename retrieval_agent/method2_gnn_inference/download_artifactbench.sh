#!/usr/bin/env bash
set -euo pipefail

ROOT="${AGENTS_PROJECT_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
REPO="$ROOT/external/artifact-linker"
PY="$ROOT/.venv-artifact-linker/bin/python"

cd "$REPO"
mkdir -p data

"$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('lwaekfjlk/artifact-bench', repo_type='dataset', local_dir='data/hf_graph')"

for subdir in transductive inductive full; do
    if [[ ! -d "data/hf_graph/$subdir" ]]; then
        echo "Missing expected dataset directory: data/hf_graph/$subdir" >&2
        exit 1
    fi
done

ln -sfn hf_graph/transductive data/artifact_graph_splits_v3_0314_transductive
ln -sfn hf_graph/inductive data/artifact_graph_splits_v3_0314_inductive
ln -sfn hf_graph/full data/artifact_graph_data_v3_0314

echo "ArtifactBench downloaded and symlinks created under $REPO/data"
