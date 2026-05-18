#!/usr/bin/env python3
"""
Amazon Bedrock Parameter Sweep Runner

Run parameter sweeps against Amazon Bedrock models (Claude, Llama, Mistral, etc).

Usage:
    python3 bedrock_sweep.py --preset summarization --trials 3
    python3 bedrock_sweep.py --preset rag_qa --temps 0 0.2 0.5 --trials 5
"""

import argparse
import itertools
import json
import os
import time
from datetime import datetime

import boto3
from botocore.exceptions import BotoCoreError, ClientError
try:
    from scripts.utils_env import env
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils_env import env

# Presets matching other cloud providers
PRESETS = {
    "summarization": {
        "system": "You are a concise enterprise summarizer. Be faithful to the input.",
        "user": "Summarize the following text in 5 bullet points. Text: {text}",
        "default_params": {
            "temperature": 0.8,
            "topP": 1.0,
            "maxTokens": 256,
        },
        "sample_text": "Procurement analytics has shifted from reactive reporting to proactive forecasting, enabling teams to reduce maverick spend and improve supplier performance.",
    },
    "rag_qa": {
        "system": "Answer using ONLY the provided context. If the answer is not in the context, say you don't know.",
        "user": "Question: {question}\n\nContext:\n---\n{context}\n---",
        "default_params": {
            "temperature": 0.1,
            "topP": 1.0,
            "maxTokens": 256,
        },
        "sample_question": "What is the key benefit described?",
        "sample_context": "Procurement analytics has shifted from reactive reporting to proactive forecasting, enabling teams to reduce maverick spend and improve supplier performance.",
    },
    "classification": {
        "system": "Classify the user's request into exactly one intent from the allowed list.",
        "user": "Text: {text}\nReturn JSON with keys intent and rationale.",
        "default_params": {
            "temperature": 0.0,
            "topP": 1.0,
            "maxTokens": 64,
        },
        "sample_text": "Please reset my laptop password and help me enroll in device management.",
    },
    "code_generation": {
        "system": "You write correct, minimal Python. Follow the requirements exactly.",
        "user": "Write a Python function that {spec}. Return ONLY code.",
        "default_params": {
            "temperature": 0.2,
            "topP": 1.0,
            "maxTokens": 512,
        },
        "sample_spec": "parses a CSV string and returns a list of dicts",
    },
}


def build_messages(preset_name: str):
    """Build messages for Bedrock API."""
    p = PRESETS[preset_name]

    if preset_name == "summarization":
        user_msg = p["user"].format(text=p["sample_text"])
    elif preset_name == "rag_qa":
        user_msg = p["user"].format(question=p["sample_question"], context=p["sample_context"])
    elif preset_name == "classification":
        user_msg = p["user"].format(text=p["sample_text"])
    elif preset_name == "code_generation":
        user_msg = p["user"].format(spec=p["sample_spec"])
    else:
        raise ValueError("Unknown preset")

    return [
        {"role": "user", "content": user_msg},
    ]


def call_bedrock(region: str, model_id: str, messages: list, params: dict, timeout=60):
    """Call Bedrock Converse API (works with Claude, Llama, Mistral, etc)."""
    client = boto3.client("bedrock-runtime", region_name=region)

    # Build inference params (compatible with multiple models)
    inference_params = {
        "temperature": params.get("temperature", 0.8),
        "maxTokens": params.get("maxTokens", 256),
    }

    # Add topP if specified
    if params.get("topP", 1.0) < 1.0:
        inference_params["topP"] = params["topP"]

    try:
        response = client.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig=inference_params,
        )

        # Extract text from response
        text = ""
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                text += block["text"]

        return text
    except ClientError as e:
        return f"Bedrock Error: {e.response['Error']['Code']} - {e.response['Error']['Message']}"
    except BotoCoreError as e:
        return f"Boto Error: {str(e)}"


def iter_grid(base_params, temps, top_ps, max_tokens):
    """Generate all parameter combinations."""
    grid = {
        "temperature": temps if temps else [base_params.get("temperature", 0.8)],
        "topP": top_ps if top_ps else [base_params.get("topP", 1.0)],
        "maxTokens": max_tokens if max_tokens else [base_params.get("maxTokens", 256)],
    }

    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        p = dict(zip(keys, values))
        yield p


def main():
    ap = argparse.ArgumentParser(description="Run parameter sweeps against Amazon Bedrock.")
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()), required=True)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--temps", type=float, nargs="*")
    ap.add_argument("--top-ps", type=float, nargs="*")
    ap.add_argument("--max-tokens", type=int, nargs="*")
    ap.add_argument("--region", default=env("AWS_REGION", "us-east-1"))
    ap.add_argument("--model", default=env("BEDROCK_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0"))

    args = ap.parse_args()

    preset = PRESETS[args.preset]
    messages = build_messages(args.preset)

    run_dir = os.path.join(os.path.dirname(__file__), "..", "..", "runs")
    os.makedirs(run_dir, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(run_dir, f"{stamp}_{args.preset}_bedrock.jsonl")

    configs = list(iter_grid(
        preset["default_params"],
        args.temps, args.top_ps, args.max_tokens
    ))

    print(f"Running {len(configs) * args.trials} calls ({len(configs)} configs × {args.trials} trials)")
    print(f"Model: {args.model}")
    print(f"Region: {args.region}")
    print()

    with open(out_path, "w", encoding="utf-8") as f:
        for i, cfg in enumerate(configs, 1):
            for t in range(args.trials):
                print(f"  [{i}/{len(configs)}] Trial {t+1}/{args.trials}: {cfg}", end="", flush=True)

                try:
                    text = call_bedrock(args.region, args.model, messages, cfg)
                    status = "ok"
                except Exception as e:
                    text = f"Error: {str(e)}"
                    status = "error"

                record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "params": cfg,
                    "text": text,
                    "status": status,
                }

                f.write(json.dumps(record) + "\n")
                f.flush()

                print(f" → {status}\n", end="", flush=True)
                time.sleep(0.5)  # Rate limiting

    print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()

