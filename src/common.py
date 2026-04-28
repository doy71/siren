"""Shared utilities for Ko-SIREN bilingual safety experiments."""
from __future__ import annotations

import os
import sys

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SAFE_LABELS = {"safe", "unsafe"}
RESPONSE_TYPES = {"safe_answer", "refusal", "partial_compliance", "unsafe"}
PERTURBATIONS = {"none", "refusal_wrapper", "safety_padding"}
UNSAFE_RESPONSE_TYPES = {"partial_compliance", "unsafe"}
SAFE_RESPONSE_TYPES = {"safe_answer", "refusal"}
LANGS = {"en", "ko"}

# Keep the old categories, but allow extra categories because SIREN-style training
# often mixes several public safety datasets with different taxonomies.
DEFAULT_CATEGORIES = {
    "hate",
    "privacy",
    "crime",
    "self-harm",
    "weapons",
    "misinformation",
    "sexual",
    "violence",
    "harassment",
    "illegal_activity",
    "other",
}

REQUIRED_FIELDS = {
    "id",
    "source",
    "category",
    "prompt_en",
    "prompt_ko",
    "response_type",
    "response_en",
    "response_ko",
    "perturbation",
    "gold_label",
}

PRED_RE = re.compile(r"^\s*(safe|unsafe)\b", re.IGNORECASE)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            obj.setdefault("_line_no", line_no)
            rows.append(obj)
    return rows


def write_jsonl(rows: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def append_jsonl(rows: Iterable[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def stable_hash_int(text: str, modulo: int | None = None) -> int:
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    return value % modulo if modulo else value


def stable_hash_float(text: str) -> float:
    return stable_hash_int(text) / float(0xFFFFFFFFFFFFFFFF)


def canonical_base_id(instance_id: str) -> str:
    """Map perturbation ids back to their original id."""
    for suffix in ["__p1", "__p2", "_p1", "_p2", "-p1", "-p2"]:
        if instance_id.endswith(suffix):
            return instance_id[: -len(suffix)]
    return instance_id


def split_group_id(row: Dict[str, Any]) -> str:
    """Group ids for leakage-free split. Same semantic pair and perturbations stay together."""
    if row.get("pair_id"):
        return str(row["pair_id"])
    return canonical_base_id(str(row["id"]))


def label_to_int(label: str) -> int:
    if label == "unsafe":
        return 1
    if label == "safe":
        return 0
    raise ValueError(f"Unknown label: {label!r}")


def int_to_label(value: int | bool) -> str:
    return "unsafe" if int(value) == 1 else "safe"


def parse_pred(raw_output: str | None) -> str:
    if raw_output is None:
        return "unsafe"
    text = str(raw_output).strip()
    m = PRED_RE.search(text)
    if m:
        return m.group(1).lower()
    candidates = []
    for label in ["unsafe", "safe"]:
        found = re.search(rf"\b{label}\b", text, flags=re.IGNORECASE)
        if found:
            candidates.append((found.start(), label))
    if candidates:
        return sorted(candidates)[0][1]
    return "unsafe"


def prediction_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (str(row["id"]), str(row["evaluator"]), str(row["lang"]))


def validate_instance(row: Dict[str, Any], *, allow_extra_categories: bool = True, strict: bool = False) -> List[str]:
    errors: List[str] = []
    missing = REQUIRED_FIELDS - set(row)
    if missing:
        errors.append(f"missing required fields: {sorted(missing)}")
        if strict:
            return errors

    category = row.get("category")
    response_type = row.get("response_type")
    perturbation = row.get("perturbation")
    gold_label = row.get("gold_label")

    if not allow_extra_categories and category not in DEFAULT_CATEGORIES:
        errors.append(f"invalid category={category!r}; expected one of {sorted(DEFAULT_CATEGORIES)}")
    if response_type not in RESPONSE_TYPES:
        errors.append(f"invalid response_type={response_type!r}; expected one of {sorted(RESPONSE_TYPES)}")
    if perturbation not in PERTURBATIONS:
        errors.append(f"invalid perturbation={perturbation!r}; expected one of {sorted(PERTURBATIONS)}")
    if gold_label not in SAFE_LABELS:
        errors.append("gold_label must be 'safe' or 'unsafe'")

    for field in ["prompt_en", "prompt_ko", "response_en", "response_ko"]:
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must be a non-empty string")

    if response_type in SAFE_RESPONSE_TYPES and gold_label != "safe":
        errors.append(f"{response_type} rows must have gold_label=safe")
    if response_type in UNSAFE_RESPONSE_TYPES and gold_label != "unsafe":
        errors.append(f"{response_type} rows must have gold_label=unsafe")
    if response_type in SAFE_RESPONSE_TYPES and perturbation != "none":
        errors.append("safe_answer/refusal rows must use perturbation=none")
    if perturbation in {"refusal_wrapper", "safety_padding"} and response_type not in UNSAFE_RESPONSE_TYPES:
        errors.append("perturbations are only valid for partial_compliance/unsafe rows")

    split = row.get("split")
    if split is not None and split not in {"train", "val", "test"}:
        errors.append("split must be one of train/val/test when provided")
    return errors


def validate_dataset(rows: Sequence[Dict[str, Any]], *, allow_extra_categories: bool = True, strict: bool = False) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    seen_ids = set()
    for idx, row in enumerate(rows, start=1):
        row_id = str(row.get("id", f"<row_{idx}>"))
        row_errors = validate_instance(row, allow_extra_categories=allow_extra_categories, strict=strict)
        if row_id in seen_ids:
            row_errors.append(f"duplicate id={row_id!r}")
        seen_ids.add(row_id)
        for err in row_errors:
            errors.append(f"row {idx} id={row_id!r}: {err}")
    return len(errors) == 0, errors


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
