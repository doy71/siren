#!/usr/bin/env python3
"""Convert K-SafeEval-R paired EN/KO JSONL into SIREN-style JSONL.

Input row fields:
  id, prompt_en, response_en, prompt_ko, response_ko, gold_label, split, ...
Output row fields:
  id, pair_id, split, lang, text, label, prompt, response, response_type, category, perturbation

label follows CSSLab/SIREN convention: 0=safe, 1=harmful/unsafe.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from common import label_to_int, read_jsonl, split_group_id, validate_dataset, write_jsonl


def format_pair(prompt: str, response: str, mode: str) -> str:
    prompt = prompt.strip()
    response = response.strip()
    if mode == "prompt_only":
        return prompt
    if mode == "response_only":
        return response
    if mode == "siren_pair":
        # Simple text format used before tokenization.  The extractor uses raw text,
        # not generation, so this avoids model-specific chat-template differences
        # when comparing Qwen/Llama/EXAONE backbones.
        return f"User: {prompt}\nAssistant: {response}"
    raise ValueError(f"Unknown mode: {mode}")


def convert_rows(rows: List[Dict[str, Any]], langs: list[str], mode: str, split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if split != "all" and row.get("split") != split:
            continue
        for lang in langs:
            prompt_key = f"prompt_{lang}"
            response_key = f"response_{lang}"
            if not row.get(prompt_key) or not row.get(response_key):
                continue
            out.append(
                {
                    "id": str(row["id"]),
                    "pair_id": str(row.get("pair_id") or split_group_id(row)),
                    "source": row.get("source", "ksafeevalr"),
                    "split": row.get("split", ""),
                    "lang": lang,
                    "text": format_pair(row[prompt_key], row[response_key], mode),
                    "label": label_to_int(row["gold_label"]),
                    "gold_label": row["gold_label"],
                    "prompt": row[prompt_key],
                    "response": row[response_key],
                    "response_type": row.get("response_type", ""),
                    "category": row.get("category", ""),
                    "perturbation": row.get("perturbation", "none"),
                }
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare SIREN-compatible JSONL from K-SafeEval-R")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--langs", nargs="+", default=["en", "ko"], choices=["en", "ko"])
    p.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    p.add_argument("--mode", default="siren_pair", choices=["siren_pair", "prompt_only", "response_only"])
    p.add_argument("--skip-validation", action="store_true")
    args = p.parse_args()

    rows = read_jsonl(args.input)
    if not args.skip_validation:
        ok, errors = validate_dataset(rows)
        if not ok:
            raise SystemExit("Dataset validation failed:\n" + "\n".join(errors[:50]))
    out = convert_rows(rows, args.langs, args.mode, args.split)
    write_jsonl(out, args.output)
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
    if os.environ.get("KSIREN_FORCE_EXIT") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
