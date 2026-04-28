#!/usr/bin/env bash
set -euo pipefail
DATA=${1:-data/ksafeevalr.jsonl}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 0) Build SIREN-format train/val/test files from the paired EN/KO benchmark.
python src/prepare_siren_dataset.py --input "$DATA" --output data/siren_train_bi.jsonl --split train --langs en ko
python src/prepare_siren_dataset.py --input "$DATA" --output data/siren_val_bi.jsonl   --split val   --langs en ko
python src/prepare_siren_dataset.py --input "$DATA" --output data/siren_test_bi.jsonl  --split test  --langs en ko

# 1) Train official-format SIREN on Qwen backbone.
python src/train_siren_ko.py \
  --train-jsonl data/siren_train_bi.jsonl \
  --val-jsonl data/siren_val_bi.jsonl \
  --test-jsonl data/siren_test_bi.jsonl \
  --output-dir results/ksiren_qwen3_0_6b_bi \
  --model qwen3-0.6b \
  --pooling-types residual_mean mlp_mean \
  --final-pooling-type residual_mean \
  --batch-size 16 \
  --max-length 512 \
  --device cuda

python src/evaluate_siren_ko.py \
  --data "$DATA" \
  --checkpoint results/ksiren_qwen3_0_6b_bi/best_model.pkl \
  --evaluator-name ksiren_qwen3_0_6b_bi \
  --output results/raw_predictions.jsonl \
  --split test \
  --langs en ko \
  --model qwen3-0.6b

# 2) Train official-format SIREN on Korean/bilingual EXAONE backbone.
python src/train_siren_ko.py \
  --train-jsonl data/siren_train_bi.jsonl \
  --val-jsonl data/siren_val_bi.jsonl \
  --test-jsonl data/siren_test_bi.jsonl \
  --output-dir results/ksiren_exaone3_5_2_4b_bi \
  --model exaone3.5-2.4b \
  --pooling-types residual_mean mlp_mean \
  --final-pooling-type residual_mean \
  --batch-size 8 \
  --max-length 512 \
  --device cuda

python src/evaluate_siren_ko.py \
  --data "$DATA" \
  --checkpoint results/ksiren_exaone3_5_2_4b_bi/best_model.pkl \
  --evaluator-name ksiren_exaone3_5_2_4b_bi \
  --output results/raw_predictions.jsonl \
  --split test \
  --langs en ko \
  --model exaone3.5-2.4b

# 3) Evaluate released English SIREN artifacts as baselines.
python src/evaluate_released_siren.py \
  --data "$DATA" \
  --artifact UofTCSSLab/SIREN-Qwen3-0.6B \
  --evaluator-name released_siren_qwen3_0_6b \
  --output results/raw_predictions.jsonl \
  --split test \
  --langs en ko

python src/evaluate_released_siren.py \
  --data "$DATA" \
  --artifact UofTCSSLab/SIREN-Llama-3.2-1B \
  --evaluator-name released_siren_llama3_2_1b \
  --output results/raw_predictions.jsonl \
  --split test \
  --langs en ko

python src/analyze.py --data "$DATA" --predictions results/raw_predictions.jsonl --out-dir results/analysis --split test --save-joined
