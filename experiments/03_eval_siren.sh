#!/usr/bin/env bash
# =============================================================================
# Step 3: SIREN 공식 모델로 BeaverTails EN/KO 평가
#
# 실행하는 조합:
#   모델 A: UofTCSSLab/SIREN-Qwen3-4B    (langs: en, ko)
#   모델 B: UofTCSSLab/SIREN-Llama-3.1-8B (langs: en, ko)
#
# 출력: results/raw_predictions.jsonl (4 evaluator × lang 조합)
#
# 사전 조건:
#   pip install llm-siren
#   python experiments/02_translate_ko.py 완료
#
# Usage:
#   bash experiments/03_eval_siren.sh
#   bash experiments/03_eval_siren.sh --device cpu  # CPU fallback
# =============================================================================
set -euo pipefail

# --- 설정 -------------------------------------------------------------------
DATA="data/beavertails_500.jsonl"
OUTPUT="results/raw_predictions.jsonl"
DEVICE="cuda"
DTYPE="bfloat16"
SPLIT="test"

# 인자 파싱 (간단 버전)
while [[ $# -gt 0 ]]; do
    case $1 in
        --device) DEVICE="$2"; shift 2;;
        --dtype)  DTYPE="$2";  shift 2;;
        --data)   DATA="$2";   shift 2;;
        --output) OUTPUT="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# --- 전처리 체크 -------------------------------------------------------------
if [[ ! -f "$DATA" ]]; then
    echo "❌ 데이터 파일 없음: $DATA"
    echo "   먼저 실행하세요: python experiments/02_translate_ko.py"
    exit 1
fi

# 번역 거부 레코드가 있는 경우 KO 필드가 비어있을 수 있음 → 경고만 출력
KO_EMPTY=$(python3 -c "
import json
empty = sum(1 for l in open('$DATA') if not json.loads(l).get('response_ko','').strip())
print(empty)
")
if [[ "$KO_EMPTY" -gt 0 ]]; then
    echo "⚠️  response_ko 비어있는 레코드: ${KO_EMPTY}개"
    echo "   해당 레코드는 ko 평가에서 자동 스킵됩니다 (SirenGuard 동작)."
fi

mkdir -p results

echo "================================================================"
echo "SIREN 이중언어 평가 시작"
echo "  데이터:  $DATA"
echo "  출력:    $OUTPUT"
echo "  디바이스: $DEVICE / $DTYPE"
echo "================================================================"

# =============================================================================
# 모델 A: SIREN-Qwen3-4B
# =============================================================================
echo ""
echo "[1/2] UofTCSSLab/SIREN-Qwen3-4B 평가 (EN + KO)..."
python src/evaluate_released_siren.py \
    --data      "$DATA" \
    --artifact  "UofTCSSLab/SIREN-Qwen3-4B" \
    --evaluator-name "siren_qwen3_4b" \
    --output    "$OUTPUT" \
    --split     "$SPLIT" \
    --langs     en ko \
    --device    "$DEVICE" \
    --dtype     "$DTYPE"

echo "  ✅ Qwen3-4B 완료"

# =============================================================================
# 모델 B: SIREN-Llama-3.1-8B
# =============================================================================
echo ""
echo "[2/2] UofTCSSLab/SIREN-Llama-3.1-8B 평가 (EN + KO)..."
python src/evaluate_released_siren.py \
    --data      "$DATA" \
    --artifact  "UofTCSSLab/SIREN-Llama-3.1-8B" \
    --evaluator-name "siren_llama3_1_8b" \
    --output    "$OUTPUT" \
    --split     "$SPLIT" \
    --langs     en ko \
    --device    "$DEVICE" \
    --dtype     "$DTYPE"

echo "  ✅ Llama-3.1-8B 완료"

# =============================================================================
# 분석
# =============================================================================
echo ""
echo "================================================================"
echo "분석 실행 중..."
echo "================================================================"

python src/analyze.py \
    --data        "$DATA" \
    --predictions "$OUTPUT" \
    --out-dir     "results/analysis" \
    --split       "$SPLIT" \
    --save-joined

echo ""
echo "================================================================"
echo "✅ 실험 완료"
echo ""
echo "  예측 결과:  $OUTPUT"
echo "  분석 결과:  results/analysis/"
echo ""
echo "  주요 테이블:"
echo "    results/analysis/table1_overall_metrics.csv      ← F1/AUROC 종합"
echo "    results/analysis/table4_language_consistency.csv ← EN↔KO 레이블 flip"
echo "    results/analysis/figures/                        ← 시각화"
echo "================================================================"
