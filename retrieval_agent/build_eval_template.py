#!/usr/bin/env python3
"""
Generate the evaluation-results template handed to whoever runs round-2
evaluation (see run_revision.py for how it's consumed).

Scans an initial_run output root for each dataset's final_recommendations.csv
(condition A, merged Method1+Method2) and emits one row per recommended
model, with dataset + model_name pre-filled and metric_name / metric_value /
error_notes left blank to fill in:

    dataset,model_name,metric_name,metric_value,error_notes

Usage:
    python retrieval_agent/build_eval_template.py \\
        --run-root data/advisor_runs_initial_run_max14b \\
        --out data/advisor_runs_initial_run_max14b/evaluation_results_template.csv
"""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build the round-2 evaluation-results template CSV.")
    parser.add_argument("--run-root", required=True, help="initial_run output root (run_initial_run.py)")
    parser.add_argument("--out", required=True, help="Template CSV output path")
    args = parser.parse_args()

    run_root, out_path = Path(args.run_root), Path(args.out)
    rows = []
    for f in sorted(run_root.glob("*/final_recommendations.csv")):
        with f.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append((row["query_dataset"], row["model_name"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model_name", "metric_name", "metric_value", "error_notes"])
        w.writerows([[ds, model, "", "", ""] for ds, model in rows])

    print(f"{len(rows)} (dataset, model) rows -> {out_path}")
    print("Fill in metric_name/metric_value/error_notes, then pass this file to run_revision.py --eval-results")


if __name__ == "__main__":
    main()
