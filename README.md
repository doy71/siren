# Ko-SIREN Bilingual Safety Experiment

이 프로젝트는 `CSSLab/SIREN`의 실제 코드 구조를 기준으로, 영어 backbone(Qwen/Llama)과 한국어/한영 backbone(EXAONE 등)에 SIREN을 적용해 비교하는 실험 코드입니다.

핵심 변경점은 이전 버전의 독자적인 `SIREN-style` 구현 대신, 공식 SIREN의 다음 구조와 저장 포맷을 따르도록 수정한 것입니다.

- `residual_mean`, `mlp_mean` representation 추출
- layer-wise L1 linear probe 학습
- cumulative importance threshold 기반 salient neuron 선택
- validation F1 기반 layer weighting
- aggregated selected-neuron feature 위에 `AdaptiveMLPClassifier` 학습
- 공식 평가 코드와 유사한 `best_model.pkl` 저장 포맷

공식 repo 기준으로 SIREN은 frozen LLM 내부 표현을 사용하며, layer-wise linear probing과 layer aggregation 후 작은 classifier만 학습합니다. 이 프로젝트의 `src/train_siren_ko.py`는 그 흐름을 K-SafeEval-R 형태의 bilingual 데이터셋과 EXAONE 같은 모델에 맞게 일반화한 버전입니다.

## 1. 설치

```bash
conda create -n ksiren python=3.11 -y
conda activate ksiren
pip install -r requirements.txt
```

공식 SIREN release artifact만 평가하려면 `llm-siren`이 필요합니다.

```bash
pip install llm-siren
```

EXAONE처럼 `trust_remote_code=True`가 필요한 모델은 기본적으로 허용되어 있습니다. 보안상 끄고 싶으면 `--no-trust-remote-code`를 사용하세요.

## 2. 데이터 포맷

원본 paired dataset은 다음 필드를 갖는 JSONL입니다.

```json
{
  "id": "ex001",
  "source": "manual",
  "category": "weapons",
  "prompt_en": "...",
  "prompt_ko": "...",
  "response_type": "unsafe",
  "response_en": "...",
  "response_ko": "...",
  "perturbation": "none",
  "gold_label": "unsafe",
  "split": "train"
}
```

`gold_label`은 SIREN 학습에서 `safe=0`, `unsafe=1`로 변환됩니다. `partial_compliance`와 `unsafe`는 기본적으로 unsafe로 두는 것이 이 프로젝트의 설계입니다.

## 3. 데이터 생성/분할

CSV에서 시작한다면:

```bash
python src/make_dataset_from_csv.py \
  --input data/paired_dataset_template.csv \
  --output data/ksafeevalr_base.jsonl
```

perturbation과 split 생성:

```bash
python src/build_dataset.py \
  --input data/ksafeevalr_base.jsonl \
  --output data/ksafeevalr_augmented.jsonl \
  --augment-perturbations \
  --summary

python src/make_splits.py \
  --input data/ksafeevalr_augmented.jsonl \
  --output data/ksafeevalr.jsonl \
  --overwrite-existing
```

SIREN 학습용 JSONL 생성:

```bash
python src/prepare_siren_dataset.py --input data/ksafeevalr.jsonl --output data/siren_train_bi.jsonl --split train --langs en ko
python src/prepare_siren_dataset.py --input data/ksafeevalr.jsonl --output data/siren_val_bi.jsonl   --split val   --langs en ko
python src/prepare_siren_dataset.py --input data/ksafeevalr.jsonl --output data/siren_test_bi.jsonl  --split test  --langs en ko
```

## 4. Ko-SIREN 학습

Qwen 기반 bilingual SIREN:

```bash
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
```

EXAONE 기반 Korean/bilingual SIREN:

```bash
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
```

출력:

```text
results/<run_name>/best_model.pkl
results/<run_name>/probes.pkl
results/<run_name>/metrics.json
results/<run_name>/selected_neurons.json
```

`best_model.pkl`은 공식 SIREN evaluation 코드가 기대하는 핵심 키와 호환되도록 저장됩니다.

```python
{
  "pooling_type": "residual_mean",
  "selected_neurons_dict": {...},
  "layer_weights": {...},
  "selected_layers": [...],
  "final_mlp": AdaptiveMLPClassifier(...),
  "best_params": {...}
}
```

## 5. 평가

직접 학습한 Ko-SIREN 평가:

```bash
python src/evaluate_siren_ko.py \
  --data data/ksafeevalr.jsonl \
  --checkpoint results/ksiren_exaone3_5_2_4b_bi/best_model.pkl \
  --evaluator-name ksiren_exaone3_5_2_4b_bi \
  --output results/raw_predictions.jsonl \
  --split test \
  --langs en ko \
  --model exaone3.5-2.4b
```

공식 released SIREN artifact 평가:

```bash
python src/evaluate_released_siren.py \
  --data data/ksafeevalr.jsonl \
  --artifact UofTCSSLab/SIREN-Qwen3-0.6B \
  --evaluator-name released_siren_qwen3_0_6b \
  --output results/raw_predictions.jsonl \
  --split test \
  --langs en ko
```

## 6. 분석

```bash
python src/analyze.py \
  --data data/ksafeevalr.jsonl \
  --predictions results/raw_predictions.jsonl \
  --out-dir results/analysis \
  --split test \
  --save-joined
```

생성되는 주요 결과:

- `table1_overall_metrics.csv`
- `table2_response_type_confusion.csv`
- `table3_perturbation_recall_drop.csv`
- `table4_language_consistency.csv`
- `table5_response_type_scores.csv`
- `figures/fig1_language_consistency.png`
- `figures/fig2_score_gap.png`
- `figures/fig3_response_type_scores.png`

## 7. 전체 실행 템플릿

```bash
bash scripts/run_qwen_exaone_official_template.sh data/ksafeevalr.jsonl
```

## 8. 빠른 점검

GPU 없이 코드 흐름만 점검하려면:

```bash
bash scripts/smoke_test.sh
```

이 smoke test는 실제 HF 모델을 로드하지 않고 `--mock` representation을 사용합니다.

## 9. 실험 설계 권장안

최소 실험:

```text
1. released_siren_qwen3_0_6b: 공식 영어/다국어 SIREN artifact baseline
2. released_siren_llama3_2_1b: 공식 Llama SIREN artifact baseline
3. ksiren_qwen3_0_6b_bi: 같은 데이터로 재학습한 bilingual SIREN
4. ksiren_exaone3_5_2_4b_bi: 한국어/한영 backbone 기반 Ko-SIREN
```

논문/발표에서의 질문:

```text
RQ1. EXAONE 기반 Ko-SIREN은 released English SIREN보다 KO test에서 높은 F1/AUROC를 보이는가?
RQ2. Ko-SIREN은 EN-KO label flip rate와 score gap을 줄이는가?
RQ3. selected layer/neuron 분포가 Qwen/Llama/EXAONE 간에 다르게 나타나는가?
RQ4. partial compliance와 refusal response에서 generative guard보다 안정적인 score separation을 보이는가?
```

## 10. 주의

- 공식 CSSLab/SIREN repo는 `Qwen3RepresentationExtractor` 이름을 쓰지만, 실제 구현은 Llama alias도 같은 extractor를 사용합니다. 이 프로젝트는 그 인터페이스를 유지하면서 EXAONE까지 처리하기 위해 `GenericRepresentationExtractor`를 추가했습니다.
- `mlp_mean` hook은 모델 아키텍처마다 MLP 모듈 이름이 다를 수 있습니다. EXAONE에서 `mlp_mean`이 실패하면 우선 `--pooling-types residual_mean --final-pooling-type residual_mean`으로 실험을 시작하세요.
- 큰 데이터셋에서는 representation extraction이 가장 오래 걸립니다. 먼저 Qwen3-0.6B와 EXAONE 2.4B로 pipeline을 고정한 뒤 4B/7.8B 이상으로 확장하는 편이 안전합니다.

## 11. Layer/neuron 분석

학습된 checkpoint 간 selected layer/neuron 분포를 비교하려면:

```bash
python src/inspect_siren_model.py \
  --checkpoints results/ksiren_qwen3_0_6b_bi results/ksiren_exaone3_5_2_4b_bi \
  --output results/analysis/selected_layer_summary.csv
```
