#!/usr/bin/env python3
import json
import os
import argparse
from tqdm import tqdm
from openai import OpenAI


def normalize_text(x):
    """Normalize any text or list of texts to a clean string."""
    if isinstance(x, list):
        return ", ".join([normalize_text(i) for i in x])
    return str(x).strip()


def gpt_check_correctness(client, prediction_raw, ground_truth_raw, model="gpt-4o"):
    """Ask GPT whether the prediction matches the ground truth semantically."""
    prediction = normalize_text(prediction_raw)
    ground_truth = normalize_text(ground_truth_raw)

    prompt = f"""You are an expert in evaluating question answering results.

[Ground-truth Answer]
{ground_truth}

[Model Prediction]
{prediction}

Is the prediction semantically equivalent to the ground-truth answer? 
Please respond with only one word: "CORRECT" or "INCORRECT".
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        return f"ERROR: {e}"


def main(args):
    # Load API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY or pass --api_key")

    client = OpenAI(api_key=api_key)

    input_file = args.input_file
    output_file = args.output_file
    model = args.model
    total_length = args.total_length

    results = []
    correct_count = 0
    total_count = 0

    print(f"🔍 Evaluating with model: {model}")
    print(f"📂 Input:  {input_file}")
    print(f"💾 Output: {output_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_length or None):
            item = json.loads(line)
            pred_raw = item.get("prediction", "")
            gt_raw = item.get("ground_truth", "")

            if not gt_raw:
                continue

            label = gpt_check_correctness(client, pred_raw, gt_raw, model=model)
            item["gpt_eval"] = label
            results.append(item)

            total_count += 1
            if label == "CORRECT":
                correct_count += 1

    # Write results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"GPT evaluation complete.")
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-based QA evaluation script")

    parser.add_argument("--input_file", type=str, required=True,
                        help="Input JSONL file with predictions and ground truths.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL file for results.")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="OpenAI model name (default: gpt-4o).")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (optional if OPENAI_API_KEY is set).")
    parser.add_argument("--total_length", type=int, default=None,
                        help="Total samples for tqdm progress bar (optional).")

    args = parser.parse_args()
    main(args)