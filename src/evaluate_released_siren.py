#!/usr/bin/env python3
"""Evaluate released CSSLab/SIREN artifacts with llm-siren runtime.

Use this for English-model baselines, e.g.
  UofTCSSLab/SIREN-Qwen3-0.6B
  UofTCSSLab/SIREN-Llama-3.2-1B

The official runtime supports prompt-level and prompt+response moderation via
SirenGuard.score(prompt=..., response=...).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import torch
from tqdm import tqdm

from common import append_jsonl, prediction_key, read_jsonl


def load_existing_keys(path: Path) -> set[Tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for row in read_jsonl(path):
        if {"id", "evaluator", "lang"}.issubset(row):
            keys.add(prediction_key(row))
    return keys


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate released SIREN artifacts")
    p.add_argument("--data", required=True)
    p.add_argument("--artifact", default="UofTCSSLab/SIREN-Qwen3-0.6B")
    p.add_argument("--evaluator-name", default=None)
    p.add_argument("--output", default="results/raw_predictions.jsonl")
    p.add_argument("--langs", nargs="+", default=["en", "ko"], choices=["en", "ko"])
    p.add_argument("--split", default="test", choices=["all", "train", "val", "test"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--threshold", type=float, default=None, help="Default: artifact threshold")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    try:
        from siren_guard import SirenGuard
    except ImportError as exc:
        raise SystemExit("Install released runtime with `pip install llm-siren` before using this script.") from exc

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    guard = SirenGuard.from_pretrained(args.artifact, device=args.device, dtype=dtype)
    evaluator = args.evaluator_name or args.artifact.split("/")[-1]

    rows = read_jsonl(args.data)
    out_path = Path(args.output)
    if args.overwrite and out_path.exists():
        out_path.unlink()
    existing = set() if args.overwrite else load_existing_keys(out_path)
    out_rows = []
    now = datetime.now(timezone.utc).isoformat()
    for row in tqdm(rows, desc=f"Score {evaluator}"):
        if args.split != "all" and row.get("split") != args.split:
            continue
        for lang in args.langs:
            prompt = row.get(f"prompt_{lang}")
            response = row.get(f"response_{lang}")
            if not prompt or not response:
                continue
            result = guard.score(prompt=prompt, response=response, threshold=args.threshold) if args.threshold is not None else guard.score(prompt=prompt, response=response)
            pred = "unsafe" if bool(result.is_harmful) else "safe"
            out = {
                "id": str(row["id"]),
                "evaluator": evaluator,
                "lang": lang,
                "pred": pred,
                "unsafe_prob": float(result.score),
                "raw_output": str(result),
                "model_name": evaluator,
                "backbone_name_or_path": args.artifact,
                "checkpoint": args.artifact,
                "split": row.get("split", ""),
                "created_at": now,
            }
            key = prediction_key(out)
            if key not in existing:
                out_rows.append(out)
                existing.add(key)
    append_jsonl(out_rows, out_path)
    print(f"Wrote {len(out_rows)} predictions to {out_path}")


if __name__ == "__main__":
    main()
    if os.environ.get("KSIREN_FORCE_EXIT") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
