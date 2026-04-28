#!/usr/bin/env python3
"""
Step 1: BeaverTails → K-SafeEval-R JSONL (EN only, KO fields 비어있음)

PKU-Alignment/BeaverTails에서 500개를 stratified 샘플링합니다:
  - 250 safe (is_safe=True)
  - 250 unsafe (is_safe=False)

출력: data/beavertails_raw_500.jsonl
  - prompt_ko / response_ko 필드는 비어 있음 → 02_translate_ko.py에서 채움
  - split="test" (SIREN eval 스크립트 필터 조건에 맞춤)

Usage:
    python experiments/01_sample_beavertails.py
    python experiments/01_sample_beavertails.py --seed 42 --n 500 --output data/beavertails_raw_500.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# BeaverTails category dict → 단일 문자열 카테고리
# ---------------------------------------------------------------------------
CATEGORY_MAP = {
    "animal_abuse": "violence",
    "child_abuse": "sexual",
    "controversial_topics,politics": "other",
    "discrimination,stereotype,injustice": "hate",
    "drug_abuse,weapons,banned_substance": "weapons",
    "financial_crime,property_crime,theft": "crime",
    "hate_speech,offensive_language": "hate",
    "misinformation_regarding_ethics,laws_and_safety": "misinformation",
    "non_violent_unethical_behavior": "other",
    "privacy_violation": "privacy",
    "self_harm": "self-harm",
    "sexually_explicit,adult_content": "sexual",
    "terrorism,organized_crime": "crime",
    "violence,aiding_and_abetting,incitement": "violence",
}


def extract_primary_category(category_field: Any) -> str:
    """BeaverTails category 필드(dict 또는 str)에서 K-SafeEval-R 카테고리 추출."""
    if isinstance(category_field, dict):
        for key, active in category_field.items():
            if active:
                return CATEGORY_MAP.get(key, "other")
        return "other"
    if isinstance(category_field, str):
        return CATEGORY_MAP.get(category_field, "other")
    return "other"


def convert_row(row: dict[str, Any], idx: int) -> dict[str, Any]:
    """BeaverTails 레코드 → K-SafeEval-R JSONL 레코드 (EN only)."""
    is_safe: bool = bool(row.get("is_safe", True))
    gold_label = "safe" if is_safe else "unsafe"
    # BeaverTails는 refusal / partial_compliance 구분이 없으므로
    # safe → safe_answer, unsafe → unsafe 로 단순 매핑
    response_type = "safe_answer" if is_safe else "unsafe"

    return {
        "id": f"bt_{idx:06d}",
        "source": "BeaverTails",
        "category": extract_primary_category(row.get("category")),
        "prompt_en": str(row.get("prompt", "")).strip(),
        "prompt_ko": "",          # Step 02에서 채워짐
        "response_type": response_type,
        "response_en": str(row.get("response", "")).strip(),
        "response_ko": "",        # Step 02에서 채워짐
        "perturbation": "none",
        "gold_label": gold_label,
        "split": "test",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample BeaverTails → K-SafeEval-R JSONL")
    parser.add_argument("--n", type=int, default=500, help="총 샘플 수 (짝수여야 함)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/beavertails_raw_500.jsonl")
    parser.add_argument("--hf-split", default="30k_test",
                        help="BeaverTails HuggingFace split (30k_train or 30k_test)")
    args = parser.parse_args()

    if args.n % 2 != 0:
        raise SystemExit("--n은 짝수여야 합니다 (safe : unsafe = 1 : 1 stratified)")

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("datasets 패키지가 없습니다: pip install datasets")

    print(f"BeaverTails 로드 중 (split={args.hf_split})...")
    ds = load_dataset("PKU-Alignment/BeaverTails", split=args.hf_split)
    print(f"  전체 레코드 수: {len(ds)}")

    # safe / unsafe 분리
    safe_rows = [r for r in ds if r.get("is_safe") is True]
    unsafe_rows = [r for r in ds if r.get("is_safe") is False]
    print(f"  safe: {len(safe_rows)}, unsafe: {len(unsafe_rows)}")

    half = args.n // 2
    if len(safe_rows) < half or len(unsafe_rows) < half:
        raise SystemExit(
            f"샘플 수 부족: safe={len(safe_rows)}, unsafe={len(unsafe_rows)}, 필요={half}"
        )

    rng = random.Random(args.seed)
    sampled_safe = rng.sample(safe_rows, half)
    sampled_unsafe = rng.sample(unsafe_rows, half)

    all_rows = sampled_safe + sampled_unsafe
    rng.shuffle(all_rows)   # 순서 섞기

    # 변환
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    converted = [convert_row(r, idx + 1) for idx, r in enumerate(all_rows)]

    # 기본 검증 (EN 필드 누락 체크)
    bad = [r for r in converted if not r["prompt_en"] or not r["response_en"]]
    if bad:
        print(f"  ⚠️  prompt_en 또는 response_en이 비어있는 레코드 {len(bad)}개 제거")
        converted = [r for r in converted if r["prompt_en"] and r["response_en"]]

    with out_path.open("w", encoding="utf-8") as f:
        for row in converted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 요약
    safe_count = sum(1 for r in converted if r["gold_label"] == "safe")
    unsafe_count = sum(1 for r in converted if r["gold_label"] == "unsafe")
    print(f"\n✅  {len(converted)}개 저장 → {out_path}")
    print(f"   safe={safe_count}, unsafe={unsafe_count}")
    print(f"\n⚠️  prompt_ko / response_ko 필드는 비어 있습니다.")
    print(f"   다음 단계: python experiments/02_translate_ko.py --input {out_path}")


if __name__ == "__main__":
    main()
