#!/usr/bin/env bash
# =============================================================================
# 전체 실험 원스텝 실행
#
# Usage:
#   bash experiments/run_experiment.sh
#   bash experiments/run_experiment.sh --skip-translate  # 번역 완료 후 재실행 시
# =============================================================================
set -euo pipefail

SKIP_SAMPLE=false
SKIP_TRANSLATE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-sample)    SKIP_SAMPLE=true;    shift;;
        --skip-translate) SKIP_TRANSLATE=true; shift;;
        *) echo "Unknown: $1"; exit 1;;
    esac
done

echo "========================================================"
echo "Ko-SIREN BeaverTails Bilingual Experiment"
echo "========================================================"

if [[ "$SKIP_SAMPLE" == "false" ]]; then
    echo ""
    echo "[Step 1/4] BeaverTails 샘플링..."
    python experiments/01_sample_beavertails.py
fi

if [[ "$SKIP_TRANSLATE" == "false" ]]; then
    echo ""
    echo "[Step 2/4] 한국어 번역 (Llama-3.1-8B-Instruct)..."
    echo "  ⏱  약 500샘플 × 2필드 = 1,000회 번역. A100 기준 약 30~60분 예상."
    python experiments/02_translate_ko.py
fi

echo ""
echo "[Step 3/4] SIREN 평가 (Qwen3-4B + Llama-3.1-8B)..."
bash experiments/03_eval_siren.sh

echo ""
echo "[Step 4/4] 결과 요약..."
python experiments/04_summarize.py

echo ""
echo "========================================================"
echo "실험 완료. 주요 결과:"
echo "  results/analysis/experiment_summary.md"
echo "  results/analysis/table1_overall_metrics.csv"
echo "  results/analysis/table4_language_consistency.csv"
echo "  results/analysis/figures/*.png"
echo "========================================================"
