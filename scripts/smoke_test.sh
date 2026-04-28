#!/usr/bin/env bash
set -euo pipefail
export KSIREN_FORCE_EXIT=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python src/build_dataset.py \
  --input data/ksafeevalr_sample.jsonl \
  --output data/ksafeevalr_sample_augmented.jsonl \
  --augment-perturbations \
  --summary

python src/make_splits.py \
  --input data/ksafeevalr_sample_augmented.jsonl \
  --output data/ksafeevalr_sample_split.jsonl \
  --overwrite-existing

python src/prepare_siren_dataset.py --input data/ksafeevalr_sample_split.jsonl --output data/siren_train.jsonl --split train --langs en ko
python src/prepare_siren_dataset.py --input data/ksafeevalr_sample_split.jsonl --output data/siren_val.jsonl --split val --langs en ko
python src/prepare_siren_dataset.py --input data/ksafeevalr_sample_split.jsonl --output data/siren_test.jsonl --split test --langs en ko

python src/train_siren_ko.py \
  --train-jsonl data/siren_train.jsonl \
  --val-jsonl data/siren_val.jsonl \
  --test-jsonl data/siren_test.jsonl \
  --output-dir results/smoke_mock_siren \
  --model qwen3-0.6b \
  --pooling-types residual_mean \
  --final-pooling-type residual_mean \
  --mock \
  --epochs 5 \
  --patience 2 \
  --mlp-dims 32

python -m py_compile src/*.py src/siren_ext/*.py

echo "Smoke test passed."
