#!/usr/bin/env python3
"""Convert a paired EN/KO CSV into the project JSONL schema.

Required CSV columns:
  category,prompt_en,prompt_ko,response_type,response_en,response_ko
Optional columns:
  id,pair_id,source,gold_label,perturbation,split

If gold_label is omitted, it is inferred from response_type:
  safe_answer/refusal -> safe
  partial_compliance/unsafe -> unsafe
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from common import SAFE_RESPONSE_TYPES, UNSAFE_RESPONSE_TYPES, validate_dataset, write_jsonl


def infer_gold(response_type: str) -> str:
    if response_type in SAFE_RESPONSE_TYPES:
        return "safe"
    if response_type in UNSAFE_RESPONSE_TYPES:
        return "unsafe"
    raise ValueError(f"Cannot infer gold_label for response_type={response_type!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert paired EN/KO CSV to K-SafeEval/Ko-SIREN JSONL")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--id-prefix", default="ksiren")
    p.add_argument("--source", default="Manual")
    args = p.parse_args()

    rows: List[Dict[str, Any]] = []
    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, r in enumerate(reader, start=1):
            response_type = (r.get("response_type") or "").strip()
            row = {
                "id": (r.get("id") or f"{args.id_prefix}_{idx:06d}").strip(),
                "pair_id": (r.get("pair_id") or r.get("id") or f"{args.id_prefix}_{idx:06d}").strip(),
                "source": (r.get("source") or args.source).strip(),
                "category": (r.get("category") or "other").strip(),
                "prompt_en": (r.get("prompt_en") or "").strip(),
                "prompt_ko": (r.get("prompt_ko") or "").strip(),
                "response_type": response_type,
                "response_en": (r.get("response_en") or "").strip(),
                "response_ko": (r.get("response_ko") or "").strip(),
                "perturbation": (r.get("perturbation") or "none").strip(),
                "gold_label": (r.get("gold_label") or infer_gold(response_type)).strip(),
            }
            if r.get("split"):
                row["split"] = r["split"].strip()
            rows.append(row)

    ok, errors = validate_dataset(rows)
    if not ok:
        for e in errors[:200]:
            print("-", e)
        raise SystemExit(1)
    write_jsonl(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
    if os.environ.get("KSIREN_FORCE_EXIT") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
