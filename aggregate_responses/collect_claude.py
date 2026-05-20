
import os
import re
import time
import argparse
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

SYSTEM_PROMPT = """
You are participating in a research study about whether LLMs generate homogeneous thesis angles for AP Language-style writing prompts.

Your task is NOT to write an essay.

For the given prompt, generate exactly 5 distinct thesis angles or argumentative approaches that a high school AP Language student could take.

Rules:
- Each angle must be one concise sentence.
- Do not write body paragraphs.
- Do not include evidence paragraphs.
- Do not repeat the same basic idea.
- Focus on idea-level argument angles, not polished wording.
- Return only a numbered list from 1 to 5.
"""

def parse_numbered_list(text):
    ideas = []
    for line in text.splitlines():
        line = line.strip()
        match = re.match(r"^\s*(\d+)[\).\s-]+(.+)$", line)
        if match:
            ideas.append(match.group(2).strip())
    return ideas[:5]

def extract_text(message):
    parts = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts)

def call_claude(model, prompt_id, prompt_text, run_id, temperature, max_retries=5):
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    user_prompt = f"AP Language prompt:\n\n{prompt_text}"
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=5000,
                temperature=temperature,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
            )

            raw_text = extract_text(message)
            ideas = parse_numbered_list(raw_text)

            rows = []
            for idea_rank, idea in enumerate(ideas, start=1):
                output_id = f"{prompt_id}__anthropic__{model}__run{run_id}__idea{idea_rank}"
                rows.append({
                    "output_id": output_id,
                    "prompt_id": prompt_id,
                    "provider": "anthropic",
                    "model": model,
                    "run_id": run_id,
                    "idea_rank": idea_rank,
                    "raw_idea": idea
                })

            return {
                "ok": True,
                "prompt_id": prompt_id,
                "model": model,
                "run_id": run_id,
                "rows": rows,
                "raw_text": raw_text,
                "parsed_count": len(ideas)
            }

        except Exception as e:
            last_error = e
            wait = min(60, 2 ** attempt)
            print(f"[Claude retry] {prompt_id} | {model} | run {run_id} | attempt {attempt} failed: {repr(e)}")
            print(f"Sleeping {wait}s...")
            time.sleep(wait)

    return {
        "ok": False,
        "prompt_id": prompt_id,
        "model": model,
        "run_id": run_id,
        "error": repr(last_error),
        "rows": []
    }

def append_rows(rows, out_file):
    df_new = pd.DataFrame(rows)
    if Path(out_file).exists():
        df_old = pd.read_csv(out_file)
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["output_id"], keep="last")
    else:
        df = df_new

    df.to_csv(out_file, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prompts", default="prompts.csv")
    parser.add_argument("--out", default="llm_outputs.csv")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    prompts = pd.read_csv(args.prompts)

    tasks = []
    for _, row in prompts.iterrows():
        prompt_id = str(row["prompt_id"]).strip()
        prompt_text = str(row["prompt_text"]).strip()

        if not prompt_text or prompt_text.lower() == "nan":
            print(f"Skipping empty prompt: {prompt_id}")
            continue

        for run_id in range(1, args.runs + 1):
            tasks.append((args.model, prompt_id, prompt_text, run_id, args.temperature, args.max_retries))

    print(f"Starting Claude collection")
    print(f"Model: {args.model}")
    print(f"Tasks: {len(tasks)}")
    print(f"Max workers: {args.max_workers}")

    all_rows = []
    failures = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(call_claude, *task): task
            for task in tasks
        }

        for i, future in enumerate(as_completed(future_to_task), start=1):
            result = future.result()

            if result["ok"]:
                all_rows.extend(result["rows"])
                status = "OK" if result["parsed_count"] == 5 else f"WARNING parsed {result['parsed_count']}/5"
                print(f"[{i}/{len(tasks)}] {status}: {result['prompt_id']} | {result['model']} | run {result['run_id']}")
                if result["parsed_count"] != 5:
                    print(result["raw_text"])
            else:
                failures.append(result)
                print(f"[{i}/{len(tasks)}] FAILED: {result['prompt_id']} | {result['model']} | run {result['run_id']} | {result['error']}")

    append_rows(all_rows, args.out)

    if failures:
        pd.DataFrame(failures).to_csv(f"failures_claude_{args.model}.csv", index=False)
        print(f"Saved failures to failures_claude_{args.model}.csv")

    print(f"Appended {len(all_rows)} rows to {args.out}")

if __name__ == "__main__":
    main()
