
import os
import time
import argparse
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

CLUSTERING_SYSTEM_PROMPT = """
You are helping code research data for a study on LLM-generated thesis-angle homogeneity.

You will receive thesis angles generated for ONE AP Language prompt.

Your job:
- Group ideas into BROAD argumentative buckets, not overly specific subcategories.
- The goal is to identify whether outputs converge around the same general thesis angle.
- Prefer fewer broader clusters over many narrow clusters.
- If two ideas would lead to basically the same AP Lang essay thesis, place them in the same cluster.
- Treat minor wording differences and synonyms as the same idea.
- Keep ideas separate only if they would lead to meaningfully different essays.

Return TSV only. Do not use markdown. Do not use code fences. Do not include explanations.

Each output row must have exactly 3 tab-separated columns:
idea_key    canonical_idea    cluster_id

Rules:
- Include one row for every idea_key provided.
- Copy idea_key exactly.
- canonical_idea should be a broad standardized thesis angle.
- cluster_id should be a short lowercase snake_case label.
- Do not include a header row.
"""

def parse_tsv_output(text):
    rows = []
    bad_lines = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")

        if len(parts) != 3:
            bad_lines.append(line)
            continue

        idea_key, canonical_idea, cluster_id = parts

        rows.append({
            "idea_key": idea_key.strip(),
            "canonical_idea": canonical_idea.strip(),
            "cluster_id": cluster_id.strip()
        })

    return rows, bad_lines

def call_cluster_model(model, prompt_text, idea_records, max_retries=5, max_output_tokens=8000):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    idea_list = "\n".join([
        f"{rec['idea_key']}: {rec['raw_idea']}"
        for rec in idea_records
    ])

    user_prompt = f"""
AP Language prompt:
{prompt_text}

Cluster the following thesis-angle ideas for this prompt.

Important:
- Return one TSV row for each idea_key.
- Do not copy the raw idea back.
- Use tabs between the three columns.

Ideas:
{idea_list}
"""

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"    API attempt {attempt}/{max_retries}...")

            response = client.responses.create(
                model=model,
                instructions=CLUSTERING_SYSTEM_PROMPT,
                input=user_prompt,
                max_output_tokens=max_output_tokens,
            )

            mappings, bad_lines = parse_tsv_output(response.output_text)

            if not mappings:
                raise ValueError(f"No valid TSV rows parsed. Raw output start: {response.output_text[:1000]}")

            return {
                "mappings": mappings,
                "bad_lines": bad_lines,
                "raw_output": response.output_text
            }

        except Exception as e:
            last_error = e
            err = str(e).lower()

            if "model_not_found" in err or "does not exist" in err:
                raise RuntimeError(f"Non-retryable model error: {repr(e)}")

            wait_seconds = min(60, 2 ** attempt)
            print(f"    Attempt {attempt} failed: {repr(e)}")
            print(f"    Backing off for {wait_seconds} seconds...")
            time.sleep(wait_seconds)

    raise RuntimeError(f"Clustering failed after {max_retries} attempts: {repr(last_error)}")

def build_cluster_preview(mappings):
    if not mappings:
        return "    No mappings returned."

    temp_df = pd.DataFrame(mappings)

    lines = ["    Cluster preview:"]

    for cluster_id, group in temp_df.groupby("cluster_id"):
        canonical = group["canonical_idea"].iloc[0]
        count = len(group)
        lines.append(f"      - {cluster_id}: {count} idea(s) | {canonical}")

    return "\n".join(lines)

def cluster_one_prompt(idx, total_prompts, prompt_id, group, prompt_lookup, args):
    prompt_id = str(prompt_id)
    prompt_text = prompt_lookup.get(prompt_id, "")

    unique_df = (
        group[["raw_idea"]]
        .dropna()
        .astype(str)
        .apply(lambda col: col.str.strip())
        .drop_duplicates()
        .reset_index(drop=True)
    )

    idea_records = []
    for i, row in unique_df.iterrows():
        idea_records.append({
            "idea_key": f"{prompt_id}_idea_{i+1}",
            "raw_idea": row["raw_idea"]
        })

    print("=" * 80)
    print(f"Starting prompt {idx}/{total_prompts}: {prompt_id}")
    print(f"Rows for prompt: {len(group)}")
    print(f"Unique ideas to cluster: {len(idea_records)}")

    if len(idea_records) == 0:
        return {
            "ok": True,
            "prompt_id": prompt_id,
            "mappings": [],
            "preview": "No ideas to cluster.",
            "bad_lines": []
        }

    result = call_cluster_model(
        model=args.model,
        prompt_text=prompt_text,
        idea_records=idea_records,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
    )

    mappings = result.get("mappings", [])
    bad_lines = result.get("bad_lines", [])

    idea_lookup = {
        rec["idea_key"]: rec["raw_idea"]
        for rec in idea_records
    }

    cleaned_mappings = []

    for m in mappings:
        idea_key = str(m.get("idea_key", "")).strip()

        if idea_key not in idea_lookup:
            continue

        cleaned_mappings.append({
            "prompt_id": prompt_id,
            "idea_key": idea_key,
            "raw_idea": idea_lookup[idea_key],
            "canonical_idea": str(m.get("canonical_idea", "")).strip(),
            "cluster_id": str(m.get("cluster_id", "")).strip()
        })

    returned_keys = {m["idea_key"] for m in cleaned_mappings}
    expected_keys = {rec["idea_key"] for rec in idea_records}
    missing_keys = sorted(expected_keys - returned_keys)

    preview = build_cluster_preview(cleaned_mappings)

    if bad_lines:
        preview += f"\n    Warning: {len(bad_lines)} bad TSV lines were skipped."
        preview += "\n    First few bad lines:"
        for line in bad_lines[:5]:
            preview += f"\n      - {line[:200]}"

    if missing_keys:
        preview += f"\n    Warning: {len(missing_keys)} idea keys were not mapped."
        preview += "\n    First few missing keys:"
        for key in missing_keys[:5]:
            preview += f"\n      - {key}: {idea_lookup[key][:160]}"

    return {
        "ok": True,
        "prompt_id": prompt_id,
        "mappings": cleaned_mappings,
        "preview": preview,
        "unique_ideas": len(idea_records),
        "returned_mappings": len(cleaned_mappings),
        "bad_lines": bad_lines
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", default="llm_outputs.csv")
    parser.add_argument("--prompts", default="prompts.csv")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--map-out", default="idea_cluster_map.csv")
    parser.add_argument("--coded-out", default="coded_ideas.csv")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=8000)
    parser.add_argument("--max-workers", type=int, default=10)
    args = parser.parse_args()

    print("Loading files...")
    outputs = pd.read_csv(args.outputs)
    prompts = pd.read_csv(args.prompts)

    required_outputs = {"output_id", "prompt_id", "provider", "model", "run_id", "idea_rank", "raw_idea"}
    missing_outputs = required_outputs - set(outputs.columns)
    if missing_outputs:
        raise ValueError(f"Missing required columns in outputs file: {missing_outputs}")

    required_prompts = {"prompt_id", "prompt_text"}
    missing_prompts = required_prompts - set(prompts.columns)
    if missing_prompts:
        raise ValueError(f"Missing required columns in prompts file: {missing_prompts}")

    prompt_lookup = dict(zip(prompts["prompt_id"].astype(str), prompts["prompt_text"].astype(str)))

    prompt_groups = list(outputs.groupby("prompt_id"))
    total_prompts = len(prompt_groups)

    print(f"Found {len(outputs)} total output rows.")
    print(f"Found {total_prompts} prompts to cluster.")
    print(f"Using clustering model: {args.model}")
    print(f"Max workers: {args.max_workers}")
    print()

    all_mappings = []
    failures = []
    all_bad_lines = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_prompt = {}

        for idx, (prompt_id, group) in enumerate(prompt_groups, start=1):
            future = executor.submit(
                cluster_one_prompt,
                idx,
                total_prompts,
                prompt_id,
                group.copy(),
                prompt_lookup,
                args,
            )
            future_to_prompt[future] = str(prompt_id)

        for completed_idx, future in enumerate(as_completed(future_to_prompt), start=1):
            prompt_id = future_to_prompt[future]

            try:
                result = future.result()

                mappings = result.get("mappings", [])
                all_mappings.extend(mappings)

                for line in result.get("bad_lines", []):
                    all_bad_lines.append({
                        "prompt_id": prompt_id,
                        "bad_line": line
                    })

                print("=" * 80)
                print(f"Finished {completed_idx}/{total_prompts}: {prompt_id}")
                print(f"Returned mappings: {len(mappings)}")
                print(result.get("preview", ""))

            except Exception as e:
                failures.append({
                    "prompt_id": prompt_id,
                    "error": repr(e)
                })
                print("=" * 80)
                print(f"FAILED {completed_idx}/{total_prompts}: {prompt_id}")
                print(repr(e))

    if failures:
        pd.DataFrame(failures).to_csv("cluster_failures.csv", index=False)
        print("Saved failures to cluster_failures.csv")

    if all_bad_lines:
        pd.DataFrame(all_bad_lines).to_csv("bad_cluster_lines.csv", index=False)
        print("Saved bad TSV lines to bad_cluster_lines.csv")

    print("=" * 80)
    print("Building final coded file...")

    map_df = pd.DataFrame(all_mappings)

    if map_df.empty:
        raise ValueError("No mappings were created. Check cluster_failures.csv.")

    required_map = {"prompt_id", "raw_idea", "canonical_idea", "cluster_id"}
    if not required_map.issubset(set(map_df.columns)):
        raise ValueError(f"Cluster output missing required columns. Got: {map_df.columns}")

    for col in ["prompt_id", "raw_idea", "canonical_idea", "cluster_id"]:
        map_df[col] = map_df[col].astype(str).str.strip()

    map_df.to_csv(args.map_out, index=False)

    outputs["prompt_id"] = outputs["prompt_id"].astype(str).str.strip()
    outputs["raw_idea_clean"] = outputs["raw_idea"].astype(str).str.strip()

    coded = outputs.merge(
        map_df,
        left_on=["prompt_id", "raw_idea_clean"],
        right_on=["prompt_id", "raw_idea"],
        how="left",
        suffixes=("", "_map")
    )

    coded.drop(columns=["raw_idea_clean", "raw_idea_map"], inplace=True, errors="ignore")

    missing = coded["canonical_idea"].isna().sum()
    if missing:
        print(f"Warning: {missing} output rows did not receive a canonical idea.")
        missing_rows = coded[coded["canonical_idea"].isna()][["prompt_id", "model", "run_id", "idea_rank", "raw_idea"]]
        missing_rows.to_csv("missing_cluster_mappings.csv", index=False)
        print("Saved missing rows to missing_cluster_mappings.csv")

    coded.to_csv(args.coded_out, index=False)

    print("Done.")
    print(f"Saved mapping to {args.map_out}")
    print(f"Saved coded data to {args.coded_out}")

if __name__ == "__main__":
    main()
