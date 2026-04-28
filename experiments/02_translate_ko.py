#!/usr/bin/env python3
"""
Step 2: Llama-3.1-8B-Instruct로 EN → KO 번역 (prompt + response)

입력:  data/beavertails_raw_500.jsonl  (prompt_ko / response_ko 비어있음)
출력:  data/beavertails_500.jsonl      (번역 완료, K-SafeEval-R 호환 JSONL)

특징:
  - 배치 처리 (기본 batch_size=4)
  - 배치 단위 체크포인트 저장 → 중단 후 재시작 가능
  - 유해 콘텐츠 번역 거부 감지 → 재시도 + fallback 기록
  - 번역 완료 후 간단한 품질 통계 출력

Usage:
    python experiments/02_translate_ko.py
    python experiments/02_translate_ko.py \\
        --input data/beavertails_raw_500.jsonl \\
        --output data/beavertails_500.jsonl \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --batch-size 4 --max-new-tokens 1024
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# 번역 프롬프트
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a professional Korean translator working on academic safety research. "
    "Your task is to translate English text to Korean accurately, preserving the "
    "original meaning, tone, and style—including any sensitive or explicit content, "
    "which must be translated faithfully for research purposes. "
    "Output ONLY the Korean translation. Do not add any explanation, warning, "
    "disclaimer, or commentary. Do not refuse."
)

USER_TEMPLATE = "Translate the following English text to Korean:\n\n{text}"

# 거부 감지 패턴
REFUSAL_PATTERNS = [
    r"I('m| am) (sorry|unable|not able)",
    r"I (cannot|can't|won't) (translate|help|provide)",
    r"^(죄송합니다|번역할 수 없습니다|도움을 드릴 수 없습니다)",
    r"(ethical|appropriate|safe) (reasons?|concerns?|guidelines?)",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)

RETRY_SYSTEM = (
    "You are a neutral translation engine for academic corpus annotation. "
    "Translate the given English text to Korean verbatim. "
    "Do not interpret, judge, or modify the content. "
    "Output Korean only."
)


# ---------------------------------------------------------------------------
# 모델 로드
# ---------------------------------------------------------------------------

def load_model(model_id: str, dtype_str: str, device_map: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}.get(dtype_str, torch.bfloat16)

    print(f"모델 로드: {model_id} (dtype={dtype_str})")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 생성 시 left-padding

    load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if device_map:
        load_kwargs["device_map"] = device_map
    else:
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    print(f"  완료. 파라미터 수: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    return model, tokenizer


# ---------------------------------------------------------------------------
# 번역 (단일 텍스트, 재시도 포함)
# ---------------------------------------------------------------------------

def translate_one(
    text: str,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[str, bool]:
    """(번역문, 거부여부) 반환."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, input_ids.shape[-1]:]
    translation = tokenizer.decode(generated, skip_special_tokens=True).strip()

    refused = bool(REFUSAL_RE.search(translation[:200]))
    return translation, refused


def translate_with_retry(
    text: str, model, tokenizer, device: str, max_new_tokens: int
) -> tuple[str, str]:
    """(번역문, 상태) 반환. 상태: 'ok' | 'retry_ok' | 'refused'"""
    translation, refused = translate_one(text, model, tokenizer, device, max_new_tokens)
    if not refused:
        return translation, "ok"

    # 재시도: 더 중립적인 system prompt
    translation, refused = translate_one(
        text, model, tokenizer, device, max_new_tokens, system_prompt=RETRY_SYSTEM
    )
    if not refused:
        return translation, "retry_ok"

    return f"[TRANSLATION_REFUSED] {translation[:200]}", "refused"


# ---------------------------------------------------------------------------
# 배치 번역
# ---------------------------------------------------------------------------

def translate_batch(
    texts: list[str],
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
) -> list[tuple[str, str]]:
    """리스트로 받아 각 텍스트를 개별 번역 (Llama instruct는 배치 생성보다 개별이 안정적)."""
    results = []
    for text in texts:
        result = translate_with_retry(text, model, tokenizer, device, max_new_tokens)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# 체크포인트 관리
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> dict[str, dict[str, str]]:
    """id → {"prompt_ko": ..., "response_ko": ..., "prompt_status": ..., "response_status": ...}"""
    if not path.exists():
        return {}
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out[obj["id"]] = obj
    return out


def save_checkpoint_entry(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="BeaverTails EN→KO 번역 (Llama 3.1-8B-Instruct)")
    parser.add_argument("--input", default="data/beavertails_raw_500.jsonl")
    parser.add_argument("--output", default="data/beavertails_500.jsonl")
    parser.add_argument("--checkpoint", default="data/.translate_checkpoint.jsonl",
                        help="배치별 진행 상황 저장 파일 (재시작 지원)")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default=None,
                        help="None이면 'auto' 사용. multi-GPU 환경에서 유용")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="GPU 메모리에 따라 조정. A100 40GB → 4~8, RTX 3090 → 2~4")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--save-every", type=int, default=10,
                        help="N 배치마다 체크포인트 저장")
    args = parser.parse_args()

    # 입력 로드
    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"입력 파일 없음: {in_path}\n먼저 01_sample_beavertails.py를 실행하세요.")
    rows: list[dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"입력: {len(rows)}개 레코드 ({in_path})")

    # 체크포인트 (기존 번역 결과 로드)
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(ckpt_path)
    already_done = set(checkpoint.keys())
    todo = [r for r in rows if r["id"] not in already_done]
    print(f"번역 완료: {len(already_done)}개 / 남은 작업: {len(todo)}개")

    if not todo:
        print("모든 번역 완료. 최종 파일 생성으로 진행합니다.")
    else:
        # 모델 로드
        model, tokenizer = load_model(args.model, args.dtype, args.device_map)
        device = args.device if not args.device_map else "cuda"

        # 번역 진행
        stats = {"ok": 0, "retry_ok": 0, "refused": 0}
        batch_count = 0
        t0 = time.time()

        for i in range(0, len(todo), args.batch_size):
            batch = todo[i : i + args.batch_size]
            batch_num = i // args.batch_size + 1
            total_batches = (len(todo) + args.batch_size - 1) // args.batch_size
            print(f"\r배치 {batch_num}/{total_batches} "
                  f"({i + len(batch)}/{len(todo)}) "
                  f"[refused={stats['refused']}]", end="", flush=True)

            prompts = [r["prompt_en"] for r in batch]
            responses = [r["response_en"] for r in batch]

            prompt_results = translate_batch(prompts, model, tokenizer, device, args.max_new_tokens)
            response_results = translate_batch(responses, model, tokenizer, device, args.max_new_tokens)

            for row, (p_ko, p_st), (r_ko, r_st) in zip(batch, prompt_results, response_results):
                entry = {
                    "id": row["id"],
                    "prompt_ko": p_ko,
                    "response_ko": r_ko,
                    "prompt_status": p_st,
                    "response_status": r_st,
                }
                checkpoint[row["id"]] = entry
                save_checkpoint_entry(ckpt_path, entry)
                stats[p_st if p_st == "refused" else "ok"] += (1 if p_st != "refused" else 0)
                if r_st == "refused":
                    stats["refused"] += 1

            batch_count += 1

        elapsed = time.time() - t0
        print(f"\n\n번역 완료: {elapsed:.0f}초")
        print(f"  정상: {stats['ok']}, 재시도 성공: {stats['retry_ok']}, 거부: {stats['refused']}")

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 최종 JSONL 생성 (번역 결과 병합)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    refused_ids = []
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            row_id = row["id"]
            ckpt = checkpoint.get(row_id, {})
            row["prompt_ko"] = ckpt.get("prompt_ko", "")
            row["response_ko"] = ckpt.get("response_ko", "")
            if ckpt.get("prompt_status") == "refused" or ckpt.get("response_status") == "refused":
                refused_ids.append(row_id)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✅  최종 파일 저장: {out_path} ({len(rows)}개)")
    if refused_ids:
        print(f"⚠️  번역 거부 레코드 {len(refused_ids)}개 (SIREN 평가 시 제외 권장):")
        for rid in refused_ids[:10]:
            print(f"   {rid}")
        if len(refused_ids) > 10:
            print(f"   ... 외 {len(refused_ids) - 10}개")
    print(f"\n다음 단계: bash experiments/03_eval_siren.sh")


if __name__ == "__main__":
    main()
