#!/usr/bin/env python3
"""
Google Vertex AI Parameter Sweep Runner

Run parameter sweeps against Google Vertex AI models (Gemini, etc).

Usage:
    python3 vertex_sweep.py --preset summarization --trials 3
    python3 vertex_sweep.py --preset rag_qa --temps 0 0.2 0.5 --trials 5
"""

import argparse
import itertools
import json
import os
import time
from datetime import datetime

from google.cloud import aiplatform
from google.api_core import exceptions
try:
    from scripts.utils_env import env
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils_env import env

# Presets matching the same use cases as other cloud providers
PRESETS = {
    "summarization": {
        "system": "You are a concise enterprise summarizer. Be faithful to the input.",
        "user": "Summarize the following text in 5 bullet points. Text: {text}",
        "default_params": {
            "temperature": 0.2,
            "top_p": 1.0,
            "top_k": 0,
            "max_output_tokens": 256,
        },
        "sample_text": "Procurement analytics has shifted from reactive reporting to proactive forecasting, enabling teams to reduce maverick spend and improve supplier performance.",
    },
    "rag_qa": {
        "system": "Answer using ONLY the provided context. If the answer is not in the context, say you don't know.",
        "user": "Question: {question}\n\nContext:\n---\n{context}\n---",
        "default_params": {
            "temperature": 0.1,
            "top_p": 1.0,
            "top_k": 0,
            "max_output_tokens": 256,
        },
        "sample_question": "What is the key benefit described?",
        "sample_context": "Procurement analytics has shifted from reactive reporting to proactive forecasting, enabling teams to reduce maverick spend and improve supplier performance.",
    },
    "classification": {
        "system": "Classify the user's request into exactly one intent from the allowed list.",
        "user": "Text: {text}\nReturn JSON with keys intent and rationale.",
        "default_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_output_tokens": 64,
        },
        "sample_text": "Please reset my laptop password and help me enroll in device management.",
    },
    "code_generation": {
        "system": "You write correct, minimal Python. Follow the requirements exactly.",
        "user": "Write a Python function that {spec}. Return ONLY code.",
        "default_params": {
            "temperature": 0.2,
            "top_p": 1.0,
            "top_k": 0,
            "max_output_tokens": 512,
        },
        "sample_spec": "parses a CSV string and returns a list of dicts",
    },
}


def build_prompt(preset_name: str):
    """Build the full prompt for a given preset."""
    p = PRESETS[preset_name]
    system = p["system"]

    if preset_name == "summarization":
        user = p["user"].format(text=p["sample_text"])
    elif preset_name == "rag_qa":
        user = p["user"].format(question=p["sample_question"], context=p["sample_context"])
    elif preset_name == "classification":
        user = p["user"].format(text=p["sample_text"])
    elif preset_name == "code_generation":
        user = p["user"].format(spec=p["sample_spec"])
    else:
        raise ValueError("Unknown preset")

    return system, user


def call_vertex_ai(project_id: str, region: str, model_name: str, system: str, user: str, params: dict, timeout=60):
    """Call Vertex AI Generative AI API."""
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)

    # Create GenerativeModel
    model = aiplatform.GenerativeModel(model_name)

    # Build generation config
    generation_config = {
        "temperature": params.get("temperature", 1.0),
        "top_p": params.get("top_p", 1.0),
        "max_output_tokens": params.get("max_output_tokens", 256),
    }

    # Add top_k if specified
    if params.get("top_k", 0) > 0:
        generation_config["top_k"] = params["top_k"]

    # Build messages (Vertex AI accepts role + content)
    messages = [
        {"role": "user", "parts": [{"text": f"{system}\n\n{user}"}]},
    ]

    try:
        response = model.generate_content(
            messages,
            generation_config=generation_config,
            stream=False,
        )
        text = response.text if response.text else ""
        return text
    except exceptions.InvalidArgument as e:
        return f"Error: {str(e)}"
    except exceptions.GoogleAPIError as e:
        return f"API Error: {str(e)}"


def iter_grid(base_params, temps, top_ps, top_ks, max_tokens):
    """Generate all parameter combinations."""
    grid = {
        "temperature": temps if temps else [base_params.get("temperature", 0.2)],
        "top_p": top_ps if top_ps else [base_params.get("top_p", 1.0)],
        "top_k": top_ks if top_ks else [base_params.get("top_k", 0)],
        "max_output_tokens": max_tokens if max_tokens else [base_params.get("max_output_tokens", 256)],
    }

    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        p = dict(zip(keys, values))
        yield p


def main():
    ap = argparse.ArgumentParser(description="Run parameter sweeps against Google Vertex AI.")
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()), required=True)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--temps", type=float, nargs="*")
    ap.add_argument("--top-ps", type=float, nargs="*")
    ap.add_argument("--top-ks", type=int, nargs="*")
    ap.add_argument("--max-tokens", type=int, nargs="*")
    ap.add_argument("--project", default=env("GOOGLE_PROJECT_ID", required=False))
    ap.add_argument("--region", default=env("GOOGLE_REGION", "us-central1"))
    ap.add_argument("--model", default=env("VERTEX_MODEL", "gemini-1.5-pro"))

    args = ap.parse_args()

    if not args.project:
        print("Error: Set GOOGLE_PROJECT_ID env var or use --project")
        return

    preset = PRESETS[args.preset]
    system, user = build_prompt(args.preset)

    run_dir = os.path.join(os.path.dirname(__file__), "..", "..", "runs")
    os.makedirs(run_dir, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(run_dir, f"{stamp}_{args.preset}_vertex.jsonl")

    configs = list(iter_grid(
        preset["default_params"],
        args.temps, args.top_ps, args.top_ks, args.max_tokens
    ))

    print(f"Running {len(configs) * args.trials} calls ({len(configs)} configs × {args.trials} trials)")
    print(f"Model: {args.model}")
    print(f"Project: {args.project}, Region: {args.region}")
    print()

    with open(out_path, "w", encoding="utf-8") as f:
        for i, cfg in enumerate(configs, 1):
            for t in range(args.trials):
                print(f"  [{i}/{len(configs)}] Trial {t+1}/{args.trials}: {cfg}", end="", flush=True)

                try:
                    text = call_vertex_ai(
                        args.project, args.region, args.model, system, user, cfg
                    )
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

