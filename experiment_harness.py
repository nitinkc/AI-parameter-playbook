#!/usr/bin/env python3
"""Local experiment harness for parameter sweeps without API calls.

This script reuses the simulator math in `notebooks/sampling_simulator.py`
and runs configurable sweep presets, writing CSV/JSON artifacts for analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from docs.notebooks.sampling_simulator import run_experiment


DEFAULT_TOKENS = [
    "approve",
    "reject",
    "review",
    "escalate",
    "delay",
    "audit",
    "optimize",
    "notify",
    "assign",
    "close",
]

DEFAULT_LOGITS = np.array([2.2, 1.8, 1.4, 0.9, 0.2, 0.1, -0.3, -0.6, -0.8, -1.0], dtype=float)


@dataclass
class RunSpec:
    name: str
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float | None = None
    typical_p: float = 1.0
    repetition_penalty: float = 1.0
    sequence_len: int = 0


def parse_float_list(raw: str) -> list[float]:
    return [float(v.strip()) for v in raw.split(",") if v.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(v.strip()) for v in raw.split(",") if v.strip()]


def load_scenario(path: Path | None) -> tuple[list[str], np.ndarray]:
    if path is None:
        return DEFAULT_TOKENS, DEFAULT_LOGITS.copy()

    payload = json.loads(path.read_text(encoding="utf-8"))
    tokens = payload.get("tokens")
    logits = payload.get("logits")
    if not isinstance(tokens, list) or not isinstance(logits, list):
        raise ValueError("Scenario file must contain JSON arrays for 'tokens' and 'logits'.")
    if len(tokens) == 0 or len(tokens) != len(logits):
        raise ValueError("Scenario 'tokens' and 'logits' must have the same non-zero length.")
    return [str(t) for t in tokens], np.array(logits, dtype=float)


def preset_runs(args: argparse.Namespace) -> list[RunSpec]:
    if args.preset == "temperature_sweep":
        return [
            RunSpec(name=f"T={t:g}", temperature=t)
            for t in parse_float_list(args.temperatures)
        ]

    if args.preset == "top_p_sweep":
        return [
            RunSpec(name=f"top_p={p:g}", top_p=p)
            for p in parse_float_list(args.top_ps)
        ]

    if args.preset == "top_k_sweep":
        return [
            RunSpec(name=f"top_k={k}", top_k=k)
            for k in parse_int_list(args.top_ks)
        ]

    if args.preset == "repetition_sweep":
        return [
            RunSpec(name=f"rep_penalty={r:g}", temperature=0.8, repetition_penalty=r, sequence_len=8)
            for r in parse_float_list(args.repetition_penalties)
        ]

    # combined
    return [
        RunSpec(name="Safety", temperature=0.3, top_k=0, top_p=1.0),
        RunSpec(name="Balanced", temperature=0.7, top_k=0, top_p=0.95),
        RunSpec(name="Creative", temperature=1.2, top_k=0, top_p=0.9),
        RunSpec(name="Diverse+Safe", temperature=0.9, top_k=10, top_p=0.95),
        RunSpec(name="Broad", temperature=0.8, top_k=40, top_p=1.0),
    ]


def execute_runs(
    runs: list[RunSpec],
    tokens: list[str],
    logits: np.ndarray,
    n_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for idx, run in enumerate(runs):
        out = run_experiment(
            token_names=tokens,
            base_logits=logits,
            n_samples=n_samples,
            seed=seed + idx,
            temperature=run.temperature,
            top_k=run.top_k,
            top_p=run.top_p,
            min_p=run.min_p,
            typical_p=run.typical_p,
            repetition_penalty=run.repetition_penalty,
            sequence_len=run.sequence_len,
        )

        probs = out["probs_emp"]
        top_token, top_prob = out["top_tokens"][0]
        viable_tokens = int(np.sum(probs > 0.0))

        rows.append(
            {
                "name": run.name,
                "temperature": run.temperature,
                "top_k": run.top_k,
                "top_p": run.top_p,
                "min_p": run.min_p,
                "typical_p": run.typical_p,
                "repetition_penalty": run.repetition_penalty,
                "sequence_len": run.sequence_len,
                "entropy": round(float(out["entropy"]), 6),
                "top_token": top_token,
                "top_token_prob": round(float(top_prob), 6),
                "viable_tokens": viable_tokens,
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\nRun Summary")
    print("-" * 78)
    print(f"{'name':20s} {'entropy':>9s} {'top_token':15s} {'top_prob':>10s} {'viable':>8s}")
    print("-" * 78)
    for row in rows:
        print(
            f"{row['name'][:20]:20s} "
            f"{row['entropy']:9.3f} "
            f"{str(row['top_token'])[:15]:15s} "
            f"{row['top_token_prob']:10.3f} "
            f"{row['viable_tokens']:8d}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local parameter sweep notebooks.")
    parser.add_argument(
        "--preset",
        choices=["combined", "temperature_sweep", "top_p_sweep", "top_k_sweep", "repetition_sweep"],
        default="combined",
        help="Sweep preset to execute.",
    )
    parser.add_argument("--samples", type=int, default=10000, help="Samples per run.")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument("--output-dir", default="runs/local", help="Output directory for artifacts.")
    parser.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="both",
        help="Artifact format to write.",
    )

    parser.add_argument("--temperatures", default="0.1,0.2,0.5,1.0,1.5,2.0")
    parser.add_argument("--top-ps", default="0.2,0.5,0.8,0.95,1.0")
    parser.add_argument("--top-ks", default="1,3,5,10,0")
    parser.add_argument("--repetition-penalties", default="1.0,1.1,1.3,1.5,2.0")

    parser.add_argument(
        "--scenario-file",
        default=None,
        help="Optional JSON file with {'tokens': [...], 'logits': [...]}.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    scenario_file = Path(args.scenario_file) if args.scenario_file else None
    tokens, logits = load_scenario(scenario_file)

    runs = preset_runs(args)
    rows = execute_runs(runs, tokens, logits, n_samples=args.samples, seed=args.seed)
    print_summary(rows)

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "created_at_utc": timestamp,
            "preset": args.preset,
            "samples": args.samples,
            "seed": args.seed,
            "scenario_file": str(scenario_file) if scenario_file else None,
        },
        "results": rows,
    }

    csv_path = out_dir / f"harness_{args.preset}_{timestamp}.csv"
    json_path = out_dir / f"harness_{args.preset}_{timestamp}.json"

    if args.format in {"csv", "both"}:
        write_csv(csv_path, rows)
        print(f"\nWrote CSV: {csv_path}")

    if args.format in {"json", "both"}:
        write_json(json_path, payload)
        print(f"Wrote JSON: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

