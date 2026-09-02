#!/usr/bin/env python3
"""
Retrieval agent - final step: run the merged prompt through an LLM.

Sends merged_prompt.md (from merge_contexts.py) to an OpenAI chat model and
saves:
  - the full response as markdown (the model's reasoning + CSV block)
  - the extracted, validated recommendations CSV, with a query_dataset column
    prepended so files from different runs can be concatenated

Usage:
    export OPENAI_API_KEY=...
    python run_recommendation.py \
        --prompt merged_prompt.md \
        --query-name tau/commonsense_qa \
        --model gpt-5.5 \
        --out-md final_recommendation.md \
        --out-csv final_recommendations.csv
"""

import argparse
import csv
import io
import os
import re
import sys


def call_llm(prompt: str, model: str, temperature) -> str:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required.")
    client = OpenAI(api_key=api_key)

    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:
        # Reasoning-class models reject explicit temperature; retry without it.
        if temperature is not None and "temperature" in str(exc):
            print("Note: model rejected the temperature setting; retrying with default.",
                  file=sys.stderr)
            kwargs.pop("temperature")
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    return resp.choices[0].message.content or ""


def extract_csv(response: str):
    """Return validated rows from the ```csv ...``` block, or None."""
    match = re.search(r"```csv\s*(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if not match:
        # Fallback: any fenced block whose first line looks like the expected header
        for block in re.findall(r"```[a-z]*\s*(.*?)```", response, re.DOTALL):
            if block.strip().lower().startswith("rank,"):
                match_text = block
                break
        else:
            return None
    else:
        match_text = match.group(1)

    rows = list(csv.reader(io.StringIO(match_text.strip())))
    rows = [r for r in rows if any(cell.strip() for cell in r)]
    if len(rows) < 2:
        return None
    header = [h.strip() for h in rows[0]]
    if header[0].lower() != "rank":
        return None
    width = len(header)
    return [header] + [r for r in rows[1:] if len(r) == width]


def main():
    parser = argparse.ArgumentParser(description="Run the merged evidence prompt through an LLM.")
    parser.add_argument("--prompt", required=True, help="merged_prompt.md from merge_contexts.py")
    parser.add_argument("--query-name", required=True, help="Query dataset name (added as a CSV column)")
    parser.add_argument("--model", default="gpt-5.5", help="OpenAI model name (default: gpt-5.5)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature; omit to use the model's default "
                             "(reasoning models require the default)")
    parser.add_argument("--out-md", default="final_recommendation.md", help="Full LLM response output path")
    parser.add_argument("--out-csv", default="final_recommendations.csv", help="Extracted CSV output path")
    args = parser.parse_args()

    with open(args.prompt, encoding="utf-8") as f:
        prompt = f.read()

    print(f"Calling {args.model} (~{len(prompt.split())} prompt words)...")
    response = call_llm(prompt, args.model, args.temperature)

    for path in (args.out_md, args.out_csv):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(f"<!-- model: {args.model} | query: {args.query_name} -->\n\n{response}\n")
    print(f"Saved full response: {args.out_md}")

    rows = extract_csv(response)
    if rows is None:
        print("WARNING: no valid CSV block found in the response; "
              "inspect the markdown output and re-run if needed.", file=sys.stderr)
        sys.exit(1)

    header, data = rows[0], rows[1:]
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_dataset"] + header)
        for r in data:
            writer.writerow([args.query_name] + r)
    print(f"Saved recommendations CSV ({len(data)} rows): {args.out_csv}")
    for r in data:
        print("  " + " | ".join(r[:4]))


if __name__ == "__main__":
    main()
