import argparse
import itertools
import json
import os
import time
from datetime import datetime

import requests

try:
    from scripts.utils_env import env
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils_env import env

PRESETS = {
    "summarization": {
        "system": "You are a concise enterprise summarizer. Be faithful to the input.",
        "user": "Summarize the following text in 5 bullet points. Text: {text}",
        "default_params": {
            "temperature": 0.2,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "max_tokens": 256,
        },
        "sample_text": "Procurement analytics has shifted from reactive reporting to proactive forecasting, enabling teams to reduce maverick spend and improve supplier performance.",
    },
    "rag_qa": {
        "system": "Answer using ONLY the provided context. If the answer is not in the context, say you don't know.",
        "user": "Question: {question}\n\nContext:\n---\n{context}\n---",
        "default_params": {
            "temperature": 0.1,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "max_tokens": 256,
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
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "max_tokens": 64,
        },
        "sample_text": "Please reset my laptop password and help me enroll in device management.",
    },
    "code_generation": {
        "system": "You write correct, minimal Python. Follow the requirements exactly.",
        "user": "Write a Python function that {spec}. Return ONLY code.",
        "default_params": {
            "temperature": 0.2,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "max_tokens": 512,
            "stop": ["```"],
        },
        "sample_spec": "parses a CSV string and returns a list of dicts",
    },
}


def build_messages(preset_name: str):
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

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_azure_chat(endpoint, deployment, api_version, api_key, messages, params, timeout=60):
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    body = {
        "messages": messages,
        **params,
    }
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def iter_grid(base_params, temps, top_ps, pres, freq, max_tokens):
    grid = {
        "temperature": temps if temps else [base_params.get("temperature", 0.2)],
        "top_p": top_ps if top_ps else [base_params.get("top_p", 1.0)],
        "presence_penalty": pres if pres else [base_params.get("presence_penalty", 0.0)],
        "frequency_penalty": freq if freq else [base_params.get("frequency_penalty", 0.0)],
        "max_tokens": max_tokens if max_tokens else [base_params.get("max_tokens", 256)],
    }

    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        p = dict(zip(keys, values))
        # carry over any extra preset params like stop
        for k, v in base_params.items():
            if k not in p:
                p[k] = v
        yield p


def main():
    ap = argparse.ArgumentParser(description="Run parameter sweeps against Azure OpenAI Chat Completions (REST).")
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()), required=True)
    ap.add_argument("--trials", type=int, default=3)

    ap.add_argument("--temps", type=float, nargs="*")
    ap.add_argument("--top-ps", type=float, nargs="*")
    ap.add_argument("--presence", type=float, nargs="*")
    ap.add_argument("--frequency", type=float, nargs="*")
    ap.add_argument("--max-tokens", type=int, nargs="*")

    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    endpoint = env("AZURE_OPENAI_ENDPOINT", required=True)
    api_key = env("AZURE_OPENAI_API_KEY", required=True)
    deployment = env("AZURE_OPENAI_DEPLOYMENT", required=True)
    api_version = env("AZURE_OPENAI_API_VERSION", "2024-10-21")

    preset = PRESETS[args.preset]
    messages = build_messages(args.preset)

    run_dir = os.path.join(os.path.dirname(__file__), "..", "runs")
    os.makedirs(run_dir, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(run_dir, f"{stamp}_{args.preset}.jsonl")

    configs = list(iter_grid(
        preset["default_params"],
        args.temps, args.top_ps, args.presence, args.frequency, args.max_tokens
    ))

    with open(out_path, "w", encoding="utf-8") as f:
        for cfg in configs:
            for t in range(args.trials):
                params = dict(cfg)
                params["seed"] = args.seed

                resp = call_azure_chat(endpoint, deployment, api_version, api_key, messages, params)

                # Azure chat completions typically returns choices[0].message.content
                text = resp.get("choices", [{}])[0].get("message", {}).get("content")

                record = {
                    "time": time.time(),
                    "preset": args.preset,
                    "trial": t,
                    "params": params,
                    "messages": messages,
                    "output": text,
                    "raw": resp,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"{args.preset} cfg={cfg} trial={t} => {str(text)[:80]}")

    # Update latest pointer
    latest = os.path.join(run_dir, "latest.jsonl")
    try:
        if os.path.exists(latest):
            os.remove(latest)
        # copy, not symlink (portable)
        import shutil
        shutil.copyfile(out_path, latest)
    except Exception:
        pass

    print(f"\nWrote: {out_path}\nLatest: {latest}")

if __name__ == "__main__":
    main()
