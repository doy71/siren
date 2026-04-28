#!/usr/bin/env python3
"""Create leakage-free train/val/test splits for paired EN/KO rows.

The split is assigned at pair/base-id level so EN/KO versions and perturbations
of the same semantic item never cross split boundaries.
"""
from __future__ import annotations

import os
import sys

import argparse
from collections import defaultdict
from typing import Dict, List

from common import read_jsonl, split_group_id, stable_hash_float, validate_dataset, write_jsonl


def assign_split(group: str, train: float, val: float, salt: str) -> str:
    x = stable_hash_float(f"{salt}:{group}")
    if x < train:
        return "train"
    if x < train + val:
        return "val"
    return "test"


def main() -> None:
    p = argparse.ArgumentParser(description="Assign deterministic split field to paired dataset")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--salt", default="ksiren-v1")
    p.add_argument("--overwrite-existing", action="store_true")
    args = p.parse_args()

    if not (0 < args.train_ratio < 1) or not (0 <= args.val_ratio < 1):
        raise SystemExit("Ratios must be in [0,1], and train must be > 0")
    if args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("train_ratio + val_ratio must be < 1")

    rows = read_jsonl(args.input)
    ok, errors = validate_dataset(rows)
    if not ok:
        for e in errors[:200]:
            print("-", e)
        raise SystemExit(1)

    group_to_split: Dict[str, str] = {}
    for row in rows:
        g = split_group_id(row)
        group_to_split.setdefault(g, assign_split(g, args.train_ratio, args.val_ratio, args.salt))

    for row in rows:
        if args.overwrite_existing or not row.get("split"):
            row["split"] = group_to_split[split_group_id(row)]

    write_jsonl(rows, args.output)

    counts = defaultdict(int)
    groups = defaultdict(set)
    for row in rows:
        counts[row["split"]] += 1
        groups[row["split"]].add(split_group_id(row))
    print(f"Wrote {len(rows)} rows to {args.output}")
    for split in ["train", "val", "test"]:
        print(f"{split}: rows={counts[split]} groups={len(groups[split])}")


if __name__ == "__main__":
    main()
    if os.environ.get("KSIREN_FORCE_EXIT") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
