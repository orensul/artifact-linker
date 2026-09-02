ArtifactLinker Helper Scripts - Setup and Usage
==============================================

This bundle contains helper scripts for ranking Hugging Face models for a dataset
using a trained ArtifactLinker joint GNN checkpoint.

Included files
--------------
- score_dataset_links.py
  Ranks models for a dataset by link_probability only.

- score_dataset_attributes.py
  Ranks models for a dataset by predicted_attribute_value only.

- score_model_dataset_pair.py
  Scores one model-dataset pair with both link_probability and predicted_attribute_value.

- score_top_link_attributes.py
  Reads a link-ranking JSON and adds attribute predictions for its top N pairs.

- rank_hf_dataset_combined.py
  Streamlined pipeline. Ranks models for a Hugging Face dataset by the paper-like score:
      combined_score = link_probability * predicted_attribute_value
  It can also add a missing HF dataset as a new isolated graph node if --augment-missing
  is used and VOYAGE_API_KEY is set.

- download_artifactbench.sh
  Convenience script from this project for downloading/symlinking ArtifactBench.

- artifactlinker_requirements.txt
  Copy of the official ArtifactLinker Python requirements.txt.

Portability note (this copy)
----------------------------
Unlike the original partner bundle, these scripts do not hardcode a machine
path. They resolve the agents_project root as:

  1. the AGENTS_PROJECT_ROOT environment variable, if set
  2. otherwise the directory containing the script itself

So either place the scripts in the root of the agents_project directory (the
layout below), or export AGENTS_PROJECT_ROOT to point at it.

rank_hf_dataset_combined.py additionally accepts --summarize-card (with
--augment-missing): summarize the HF card with GPT-4o using the same prompt
that built the benchmark node summaries, then embed the summary — instead of
embedding the raw card text. This matches how the existing node embeddings
were computed and requires OPENAI_API_KEY.

Expected directory layout
-------------------------
Put these scripts in the root of an agents_project-style directory containing:

  agents_project/
  |-- score_dataset_links.py
  |-- score_dataset_attributes.py
  |-- score_model_dataset_pair.py
  |-- score_top_link_attributes.py
  |-- rank_hf_dataset_combined.py
  |-- download_artifactbench.sh
  |-- .venv-artifact-linker/
  `-- external/artifact-linker/
      |-- artifact_graph/
      |-- scripts/
      |-- requirements.txt
      `-- data/
          |-- artifact_graph_splits_v3_0314_transductive/
          |-- artifact_graph_splits_v3_0314_inductive/
          |-- artifact_graph_data_v3_0314/
          `-- joint_sweep_gatv2_trans/
              `-- trans_joint_gatv2_model_emb.pth

The trained checkpoint is required:

  external/artifact-linker/data/joint_sweep_gatv2_trans/trans_joint_gatv2_model_emb.pth

Environment setup
-----------------
From agents_project:

  git clone https://github.com/allenai/artifact-linker.git external/artifact-linker

  python3 -m venv .venv-artifact-linker
  .venv-artifact-linker/bin/python -m pip install -r external/artifact-linker/requirements.txt
  .venv-artifact-linker/bin/python -m pip install -e external/artifact-linker

If adding missing Hugging Face datasets as new graph nodes, also install Voyage:

  .venv-artifact-linker/bin/python -m pip install voyageai
  export VOYAGE_API_KEY="YOUR_VOYAGE_KEY"

Download ArtifactBench
----------------------
From agents_project, after the official repo exists:

  ./download_artifactbench.sh

Or manually:

  cd external/artifact-linker
  python -c "from huggingface_hub import snapshot_download; snapshot_download('lwaekfjlk/artifact-bench', repo_type='dataset', local_dir='data/hf_graph')"
  ln -sfn hf_graph/transductive data/artifact_graph_splits_v3_0314_transductive
  ln -sfn hf_graph/inductive data/artifact_graph_splits_v3_0314_inductive
  ln -sfn hf_graph/full data/artifact_graph_data_v3_0314

Train or copy checkpoint
------------------------
These helper scripts assume this checkpoint exists:

  external/artifact-linker/data/joint_sweep_gatv2_trans/trans_joint_gatv2_model_emb.pth

If it is not already copied from the original machine, train it with the official
ArtifactLinker script, for example:

  cd external/artifact-linker
  ../../.venv-artifact-linker/bin/python scripts/train_joint_gnn.py \
    --split-dir data/artifact_graph_splits_v3_0314_transductive \
    --output-dir data/joint_sweep_gatv2_trans \
    --backbone gatv2 \
    --model-path data/joint_sweep_gatv2_trans/trans_joint_gatv2_model_emb.pth \
    --embedding-mode embedding \
    --num-layers 3 \
    --hidden 128 \
    --heads 8 \
    --epochs 1500 \
    --lr 0.002 \
    --attr-weight 5.0 \
    --neg-ratio 2 \
    --seed 42

Non-combined link ranking
-------------------------
Ranks models by link_probability only:

  cd agents_project
  .venv-artifact-linker/bin/python score_dataset_links.py 10194 --top-k 25

Or by dataset name:

  .venv-artifact-linker/bin/python score_dataset_links.py "HuggingFaceFW/CommonsenseQA" --top-k 25

Output:

  external/artifact-linker/data/custom_link_predictions/dataset_<ID>_top<K>.json

Main field:

  link_probability

Combined ranking, paper-like score
----------------------------------
Ranks by:

  combined_score = link_probability * predicted_attribute_value

Example with an existing/alias-resolved HF dataset:

  .venv-artifact-linker/bin/python rank_hf_dataset_combined.py \
    https://huggingface.co/datasets/tau/commonsense_qa \
    --top-k 25

Or with a known ArtifactBench dataset node id:

  .venv-artifact-linker/bin/python rank_hf_dataset_combined.py 10194 --top-k 25

Output:

  external/artifact-linker/data/custom_combined_predictions/

Main fields:

  combined_score
  link_probability
  predicted_attribute_value

Attribute-only ranking
----------------------
Ranks by predicted_attribute_value only:

  .venv-artifact-linker/bin/python score_dataset_attributes.py 10194 --top-k 10 --include-link-probability

Output:

  external/artifact-linker/data/custom_attribute_predictions/

One model-dataset pair
----------------------
Scores a single pair:

  .venv-artifact-linker/bin/python score_model_dataset_pair.py \
    --model "amd/Instella-3B" \
    --dataset 10194

Output:

  external/artifact-linker/data/custom_attribute_predictions/model_<MODEL_ID>_dataset_<DATASET_ID>.json

Missing Hugging Face dataset nodes
----------------------------------
If a dataset is not already in ArtifactBench, use --augment-missing:

  export VOYAGE_API_KEY="YOUR_VOYAGE_KEY"

  .venv-artifact-linker/bin/python rank_hf_dataset_combined.py \
    https://huggingface.co/datasets/xai-org/RealworldQA \
    --augment-missing \
    --top-k 25

This creates an augmented copy of the split, appends the dataset as an isolated
node, embeds the dataset README with Voyage voyage-3, and ranks candidates.
The official ArtifactBench split is not modified.

Interpretation notes
--------------------
- link_probability means the model thinks an evaluation link is plausible.
- predicted_attribute_value is the ArtifactLinker attribute head output in a
  normalized metric space. It is not a verified measured benchmark score.
- combined_score follows the ArtifactLinker two-stage ranking idea:
      link_probability * predicted_attribute_value
- For --augment-missing, the new dataset has text features but no graph neighbors,
  so this is an exploratory inductive-style use of a transductive checkpoint.
