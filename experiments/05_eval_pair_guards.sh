#!/usr/bin/env bash
# =============================================================================
# Step 5: Prompt-response pair guard 모델 평가 추가 실행
#
# 전제:
#   bash experiments/run_experiment.sh 실행 완료
#   data/beavertails_500.jsonl 존재
#
# 추가 모델:
#   1) kakaocorp/kanana-safeguard-8b
#   2) SamsungSDS-Research/SGuard-ContentFilter-2B-v1
#   3) iknow-lab/llama-3.2-3B-wildguard-ko-2410
#
# 출력:
#   results/raw_predictions.jsonl 에 append
#   results/analysis/ 재생성
#
# Usage:
#   bash experiments/05_eval_pair_guards.sh
#   bash experiments/05_eval_pair_guards.sh --load-in-4bit
#   bash experiments/05_eval_pair_guards.sh --langs ko
#   bash experiments/05_eval_pair_guards.sh --overwrite-evaluator
# =============================================================================
set -euo pipefail

DATA="data/beavertails_500.jsonl"
OUTPUT="results/raw_predictions.jsonl"
ANALYSIS_DIR="results/analysis"
SPLIT="test"
LANGS=(en ko)
TORCH_DTYPE="bfloat16"
DEVICE_MAP="auto"
LOAD_IN_4BIT=false
LOAD_IN_8BIT=false
OVERWRITE_EVALUATOR=false
LIMIT=""
TRUST_REMOTE_CODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data) DATA="$2"; shift 2;;
        --output) OUTPUT="$2"; shift 2;;
        --analysis-dir) ANALYSIS_DIR="$2"; shift 2;;
        --split) SPLIT="$2"; shift 2;;
        --langs)
            shift
            LANGS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                LANGS+=("$1")
                shift
            done
            ;;
        --torch-dtype) TORCH_DTYPE="$2"; shift 2;;
        --device-map) DEVICE_MAP="$2"; shift 2;;
        --load-in-4bit) LOAD_IN_4BIT=true; shift;;
        --load-in-8bit) LOAD_IN_8BIT=true; shift;;
        --overwrite-evaluator) OVERWRITE_EVALUATOR=true; shift;;
        --trust-remote-code) TRUST_REMOTE_CODE=true; shift;;
        --limit) LIMIT="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [[ ! -f "$DATA" ]]; then
    echo "❌ 데이터 파일 없음: $DATA"
    echo "   먼저 실행: bash experiments/run_experiment.sh"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")" "$ANALYSIS_DIR"

COMMON_ARGS=(
    --data "$DATA"
    --output "$OUTPUT"
    --langs "${LANGS[@]}"
    --split "$SPLIT"
    --torch-dtype "$TORCH_DTYPE"
    --device-map "$DEVICE_MAP"
)

if [[ "$LOAD_IN_4BIT" == "true" ]]; then
    COMMON_ARGS+=(--load-in-4bit)
fi
if [[ "$LOAD_IN_8BIT" == "true" ]]; then
    COMMON_ARGS+=(--load-in-8bit)
fi
if [[ "$OVERWRITE_EVALUATOR" == "true" ]]; then
    COMMON_ARGS+=(--overwrite-evaluator)
fi
if [[ "$TRUST_REMOTE_CODE" == "true" ]]; then
    COMMON_ARGS+=(--trust-remote-code)
fi
if [[ -n "$LIMIT" ]]; then
    COMMON_ARGS+=(--limit "$LIMIT")
fi

print_header() {
    echo ""
    echo "================================================================"
    echo "$1"
    echo "================================================================"
}

print_header "Prompt-response guard 평가 시작"
echo "  데이터:      $DATA"
echo "  출력:        $OUTPUT"
echo "  분석 폴더:   $ANALYSIS_DIR"
echo "  언어:        ${LANGS[*]}"
echo "  dtype/map:   $TORCH_DTYPE / $DEVICE_MAP"
echo "  4bit/8bit:   $LOAD_IN_4BIT / $LOAD_IN_8BIT"

# =============================================================================
# 1. Kanana Safeguard 8B
# =============================================================================
print_header "[1/3] kakaocorp/kanana-safeguard-8b"
python src/evaluate_pair_guards.py \
    "${COMMON_ARGS[@]}" \
    --model-type kanana \
    --model-name-or-path kakaocorp/kanana-safeguard-8b \
    --evaluator-name kanana_safeguard_8b

# =============================================================================
# 2. Samsung SDS SGuard ContentFilter 2B
# =============================================================================
print_header "[2/3] SamsungSDS-Research/SGuard-ContentFilter-2B-v1"
python src/evaluate_pair_guards.py \
    "${COMMON_ARGS[@]}" \
    --model-type sguard \
    --model-name-or-path SamsungSDS-Research/SGuard-ContentFilter-2B-v1 \
    --evaluator-name sguard_content_filter_2b \
    --sguard-thresholds 0.5,0.5,0.5,0.5,0.5

# =============================================================================
# 3. WildGuard-ko 3B
# =============================================================================
print_header "[3/3] iknow-lab/llama-3.2-3B-wildguard-ko-2410"
python src/evaluate_pair_guards.py \
    "${COMMON_ARGS[@]}" \
    --model-type wildguard \
    --model-name-or-path iknow-lab/llama-3.2-3B-wildguard-ko-2410 \
    --evaluator-name wildguard_ko_3b \
    --wildguard-decision response

# =============================================================================
# 분석 재실행
# =============================================================================
print_header "분석 재실행"
python src/analyze.py \
    --data "$DATA" \
    --predictions "$OUTPUT" \
    --out-dir "$ANALYSIS_DIR" \
    --split "$SPLIT" \
    --save-joined

python experiments/04_summarize.py --analysis-dir "$ANALYSIS_DIR"

print_header "✅ 추가 guard 평가 완료"
echo "  예측 결과: $OUTPUT"
echo "  분석 결과: $ANALYSIS_DIR"
echo "  전체 비교: $ANALYSIS_DIR/table1_overall_metrics.csv"
echo "  언어 일관성: $ANALYSIS_DIR/table4_language_consistency.csv"
