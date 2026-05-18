import argparse
import json
import math
import re
from collections import Counter, defaultdict

import pandas as pd


def tokenize_words(text: str):
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9_']+", text.lower())


def distinct_n(words, n=2):
    if len(words) < n:
        return 0.0
    grams = list(zip(*[words[i:] for i in range(n)]))
    return len(set(grams)) / max(1, len(grams))


def repetition_score(words, n=2):
    if len(words) < n:
        return 0.0
    grams = list(zip(*[words[i:] for i in range(n)]))
    c = Counter(grams)
    repeats = sum(v - 1 for v in c.values() if v > 1)
    return repeats / max(1, len(grams))


def is_json(text: str):
    if not text:
        return False
    text = text.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return False
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description="Analyze JSONL run outputs.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out = rec.get("output") or ""
            words = tokenize_words(out)
            rows.append({
                "preset": rec.get("preset"),
                "trial": rec.get("trial"),
                "temperature": rec.get("params", {}).get("temperature"),
                "top_p": rec.get("params", {}).get("top_p"),
                "presence_penalty": rec.get("params", {}).get("presence_penalty"),
                "frequency_penalty": rec.get("params", {}).get("frequency_penalty"),
                "max_tokens": rec.get("params", {}).get("max_tokens"),
                "chars": len(out),
                "words": len(words),
                "distinct_2": distinct_n(words, 2),
                "repeat_2": repetition_score(words, 2),
                "looks_like_json": is_json(out),
            })

    df = pd.DataFrame(rows)

    # Aggregate
    group_cols = ["preset", "temperature", "top_p", "presence_penalty", "frequency_penalty", "max_tokens"]
    agg = df.groupby(group_cols).agg({
        "chars": "mean",
        "words": "mean",
        "distinct_2": "mean",
        "repeat_2": "mean",
        "looks_like_json": "mean",
    }).reset_index()

    print("=== Summary (mean across trials) ===")
    with pd.option_context('display.max_rows', 200, 'display.max_columns', 200):
        print(agg.sort_values(["preset", "temperature", "top_p"]))

    if args.out:
        agg.to_csv(args.out, index=False)
        print(f"Wrote CSV: {args.out}")


if __name__ == "__main__":
    main()
