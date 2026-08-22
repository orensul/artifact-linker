#!/usr/bin/env python3
"""
Revision run - round 2 of the AutoModelAdvisor evaluation loop.

Round structure (see also run_initial_run.py):
  1. initial_run   (run_initial_run.py) -> produces final_recommendations.csv
                                            + merged_prompt.md per dataset.
  2. external evaluation                 -> a colleague runs the initial run's
                                            recommended models on each dataset
                                            and reports per-model metrics +
                                            error/weakness analysis.
  3. revision_run  (this script)         -> feeds the ORIGINAL evidence, the
                                            ROUND-1 recommendation, and the
                                            ROUND-2 evaluation feedback back
                                            into the LLM, so it can confirm,
                                            drop, or replace models based on
                                            what actually happened - not just
                                            re-predict from the graph again.

Expects an evaluation-results CSV (one row per dataset+model that was
evaluated) with columns:
    dataset,model_name,metric_name,metric_value,error_notes
- dataset: exact HF id, matching the `query_dataset` column already produced
  by the initial run (e.g. "openlifescienceai/medmcqa").
- model_name: exact model name, matching final_recommendations.csv.
- metric_name / metric_value: whatever the colleague measured (accuracy, F1,
  exact-match, ...); metric_value may be left blank if only qualitative notes
  are available.
- error_notes: free text - failure modes, weaknesses, anything worth citing
  in the revised reasoning (e.g. "hallucinates on >2k token inputs",
  "confuses stance labels for/against").

Use build_eval_template.py to generate a starter CSV (dataset + model_name
pre-filled from the initial run's condition-A recommendations, other columns
blank) to hand to whoever runs the evaluation.

ASSUMPTIONS (flag if wrong): this format, and evaluating only the condition-A
(merged) recommendations rather than B/C separately, are both my proposal -
not yet confirmed against what the colleague will actually produce.

Usage:
    .venv-m2/bin/python retrieval_agent/run_revision.py \\
        --initial-run-root data/advisor_runs_initial_run_max14b \\
        --eval-results data/advisor_runs_initial_run_max14b/evaluation_results.csv \\
        --out-root data/advisor_runs_revision_run_max14b
"""

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pipeline import load_dotenv  # noqa: E402
from run_recommendation import call_llm, extract_csv  # noqa: E402

REVISION_TEMPLATE = """\
You are REVISING model recommendations for a dataset after a round of real
evaluation. In round 1 you recommended models for this dataset using
ArtifactBench evidence (a graph of real model evaluations on OTHER
datasets - predictions, not measurements, for this specific dataset). Those
recommended models were then actually run on THIS dataset, and below is what
happened.

Dataset: https://huggingface.co/datasets/{dataset}

======================================================================
# ROUND 1 - INITIAL RECOMMENDATION
======================================================================

{initial_recommendation}

======================================================================
# ROUND 2 - EVALUATION RESULTS AND ERROR ANALYSIS (real measurements)
======================================================================

{eval_feedback}

======================================================================
# ORIGINAL EVIDENCE (ArtifactBench graph, same as round 1)
======================================================================

{original_evidence}

Task: using the real evaluation results and error analysis above, REVISE the
recommendation:
- Confirm models that performed well.
- Drop or deprioritize models that underperformed or showed the reported
  weaknesses/errors - explain specifically which error pattern disqualifies
  them.
- You may promote a different model from the ORIGINAL EVIDENCE that was not
  recommended in round 1, if it plausibly avoids the reported errors (e.g. a
  larger-context or differently-trained model from the same evidence pool).
- Weigh round-2 real measurements over round-1 predictions when they conflict.
- Ground every decision explicitly in either the round-1 evidence or the
  round-2 evaluation feedback - do not speculate beyond what is given.
{constraint}
First, briefly explain what changed vs. the initial recommendation and why.

Then output the final recommendations as a CSV code block with EXACTLY this
header and one row per recommended model:

```csv
rank,model_name,recommendation_score,source_method,reasoning,change_from_initial
```

Column rules:
- rank: 1 = strongest recommendation.
- model_name: the exact model name as it appears in the evidence.
- recommendation_score: your overall score for the recommendation in [0, 1]
  (1.0 = near-certain excellent fit), consistent with the ranking order.
- source_method: exactly one of "method1", "method2", or "both" (per the
  ORIGINAL EVIDENCE section, same convention as round 1).
- reasoning: one or two sentences justifying the recommendation, citing
  specific evidence and/or round-2 evaluation results/errors. Enclose the
  whole field in double quotes.
- change_from_initial: exactly one of "confirmed" (recommended in round 1,
  still recommended, same reasoning), "revised" (recommended in round 1,
  still recommended, but reasoning/score materially updated by round-2
  feedback), or "new" (not recommended in round 1, promoted now).
"""


def load_eval_feedback(path: Path) -> dict:
    """dataset -> formatted feedback text, from the evaluation-results CSV."""
    by_dataset = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = (row.get("dataset") or "").strip()
            if not ds:
                continue
            by_dataset[ds].append(row)
    feedback = {}
    for ds, rows in by_dataset.items():
        lines = []
        for r in rows:
            name = (r.get("metric_name") or "").strip()
            value = (r.get("metric_value") or "").strip()
            metric = f"{name}={value}" if name or value else "(no metric reported)"
            notes = (r.get("error_notes") or "").strip() or "(no notes)"
            lines.append(f"- {(r.get('model_name') or '').strip()}: {metric} | {notes}")
        feedback[ds] = "\n".join(lines)
    return feedback


def run_revision(query_name: str, initial_csv_text: str, original_evidence: str,
                 eval_text: str, out_dir: Path, model: str, n: int, max_params_b: float) -> str:
    out_csv = out_dir / "final_recommendations_revision.csv"
    out_md = out_dir / "final_recommendation_revision.md"
    if out_csv.exists():
        return "cached"

    constraint_text = ""
    if max_params_b is not None:
        constraint_text = (
            f"\nHARD CONSTRAINT: Only recommend models with at most {max_params_b:g}B "
            "parameters, and only choose models that appear in the ORIGINAL EVIDENCE "
            "above (i.e. in the ArtifactBench graph).\n"
        )

    prompt = REVISION_TEMPLATE.format(
        dataset=query_name,
        initial_recommendation=initial_csv_text,
        eval_feedback=eval_text or "(no evaluation results reported for this dataset)",
        original_evidence=original_evidence,
        constraint=constraint_text,
    )
    response = call_llm(prompt, model, None)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md.write_text(f"<!-- model: {model} | condition: revision | query: {query_name} -->\n\n{response}\n",
                      encoding="utf-8")
    rows = extract_csv(response)
    if rows is None:
        return "NO CSV IN RESPONSE (see md)"
    header, data = rows[0], rows[1:]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_dataset"] + header)
        for r in data:
            w.writerow([query_name] + r)
    return f"done ({len(data)} recs)"


def aggregate_revisions(out_root: Path) -> Path:
    rows, header = [], None
    for f in sorted(out_root.glob("*/final_recommendations_revision.csv")):
        with f.open(encoding="utf-8") as fh:
            r = list(csv.reader(fh))
        if not r:
            continue
        if header is None:
            header = r[0]
        rows.extend(r[1:])
    agg = out_root / "all_final_recommendations_revision.csv"
    if header:
        with agg.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)
    return agg


def main():
    parser = argparse.ArgumentParser(description="Run the revision round (round 2) over an initial run's outputs.")
    parser.add_argument("--initial-run-root", required=True, help="Output root of run_initial_run.py")
    parser.add_argument("--eval-results", required=True,
                        help="CSV: dataset,model_name,metric_name,metric_value,error_notes")
    parser.add_argument("--out-root", required=True, help="Where to write the revision run's outputs")
    parser.add_argument("--max-params-b", type=float, default=14.0)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    load_dotenv()
    src_root, out_root = Path(args.initial_run_root), Path(args.out_root)
    eval_feedback = load_eval_feedback(Path(args.eval_results))

    dirs = [d for d in sorted(src_root.glob("*/")) if (d / "final_recommendations.csv").exists()]
    print(f"{len(dirs)} dataset runs found under {src_root}")

    for src_d in dirs:
        with (src_d / "final_recommendations.csv").open(encoding="utf-8") as f:
            initial_rows = list(csv.reader(f))
        query_name = initial_rows[1][0] if len(initial_rows) > 1 else src_d.name.replace("_", "/", 1)
        initial_csv_text = "\n".join(",".join(r) for r in initial_rows)
        original_evidence = (src_d / "merged_prompt.md").read_text(encoding="utf-8").strip()
        eval_text = eval_feedback.get(query_name, "")

        out_d = out_root / src_d.name
        status = run_revision(query_name, initial_csv_text, original_evidence, eval_text,
                              out_d, args.model, args.n, args.max_params_b)
        if not eval_text:
            status += " | WARNING: no evaluation feedback found for this dataset"
        print(f"  {src_d.name:45s} {status}")

    agg = aggregate_revisions(out_root)
    print(f"\nMaster revision file: {agg}")


if __name__ == "__main__":
    main()
