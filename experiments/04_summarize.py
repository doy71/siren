#!/usr/bin/env python3
"""
Step 4: 실험 결과 요약 출력

analyze.py가 생성한 CSV들을 읽어 논문/발표용 비교 테이블을 터미널에 출력하고
results/analysis/experiment_summary.md 로 저장합니다.

핵심 질문:
  Q1. 두 SIREN 모델의 EN F1 vs KO F1 차이는?
  Q2. EN→KO 레이블 flip rate (language consistency)는?
  Q3. 두 모델 중 KO 성능 하락이 더 적은 모델은?

Usage:
    python experiments/04_summarize.py
    python experiments/04_summarize.py --analysis-dir results/analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    raise SystemExit("pandas가 필요합니다: pip install pandas")


EVALUATOR_DISPLAY = {
    "siren_qwen3_4b":    "SIREN-Qwen3-4B",
    "siren_llama3_1_8b": "SIREN-Llama-3.1-8B",
}


def fmt(val) -> str:
    if pd.isna(val):
        return "n/a"
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return str(val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="results/analysis")
    args = parser.parse_args()

    adir = Path(args.analysis_dir)
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("# Ko-SIREN Experiment: BeaverTails EN vs KO")
    emit(f"분석 디렉토리: {adir}")
    emit()

    # ------------------------------------------------------------------
    # Table 1: Overall metrics (EN / KO 별 F1, AUROC)
    # ------------------------------------------------------------------
    t1_path = adir / "table1_overall_metrics.csv"
    if t1_path.exists():
        t1 = pd.read_csv(t1_path)
        emit("## Table 1. Overall Metrics (EN / KO)")
        emit()
        emit(f"{'Evaluator':<25} {'Lang':<6} {'N':>5} {'Acc':>8} {'F1':>8} {'Recall':>8} {'AUROC':>8}")
        emit("-" * 75)
        for ev in ["siren_qwen3_4b", "siren_llama3_1_8b"]:
            for lang in ["en", "ko"]:
                sub = t1[(t1["evaluator"] == ev) & (t1["lang"] == lang)]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                display = EVALUATOR_DISPLAY.get(ev, ev)
                emit(
                    f"{display:<25} {lang:<6} {int(r['n']):>5} "
                    f"{fmt(r.get('accuracy')):>8} "
                    f"{fmt(r.get('f1_unsafe')):>8} "
                    f"{fmt(r.get('recall_unsafe')):>8} "
                    f"{fmt(r.get('auroc')):>8}"
                )
            emit()  # blank line between models
        emit()

        # EN vs KO F1 drop
        emit("### EN → KO F1 하락")
        emit(f"{'Evaluator':<25} {'F1(EN)':>10} {'F1(KO)':>10} {'Drop':>10}")
        emit("-" * 55)
        for ev in ["siren_qwen3_4b", "siren_llama3_1_8b"]:
            en_row = t1[(t1["evaluator"] == ev) & (t1["lang"] == "en")]
            ko_row = t1[(t1["evaluator"] == ev) & (t1["lang"] == "ko")]
            if en_row.empty or ko_row.empty:
                continue
            f1_en = float(en_row.iloc[0].get("f1_unsafe", float("nan")))
            f1_ko = float(ko_row.iloc[0].get("f1_unsafe", float("nan")))
            drop = f1_en - f1_ko
            display = EVALUATOR_DISPLAY.get(ev, ev)
            emit(f"{display:<25} {f1_en:>10.4f} {f1_ko:>10.4f} {drop:>+10.4f}")
        emit()
    else:
        emit(f"⚠️  {t1_path} 없음. analyze.py를 먼저 실행하세요.")
        emit()

    # ------------------------------------------------------------------
    # Table 4: Language consistency (EN↔KO flip rate)
    # ------------------------------------------------------------------
    t4_path = adir / "table4_language_consistency.csv"
    if t4_path.exists():
        t4 = pd.read_csv(t4_path)
        emit("## Table 4. Language Consistency (EN↔KO Label Flip)")
        emit()
        # 컬럼 명은 analyze.py 구현에 따라 다를 수 있으므로 방어적으로 처리
        emit(t4.to_string(index=False))
        emit()

        # flip rate 추출 시도
        flip_col = next((c for c in t4.columns if "flip" in c.lower() or "inconsist" in c.lower()), None)
        if flip_col and "evaluator" in t4.columns:
            emit("### Flip Rate 비교")
            for ev in ["siren_qwen3_4b", "siren_llama3_1_8b"]:
                sub = t4[t4["evaluator"] == ev]
                if sub.empty:
                    continue
                display = EVALUATOR_DISPLAY.get(ev, ev)
                rate = sub.iloc[0][flip_col]
                emit(f"  {display}: {fmt(rate)}")
            emit()
    else:
        emit(f"⚠️  {t4_path} 없음.")
        emit()

    # ------------------------------------------------------------------
    # 그림 목록
    # ------------------------------------------------------------------
    fig_dir = adir / "figures"
    if fig_dir.exists():
        figs = sorted(fig_dir.glob("*.png"))
        if figs:
            emit("## 생성된 시각화")
            for f in figs:
                emit(f"  - {f}")
            emit()

    # ------------------------------------------------------------------
    # 해석 가이드
    # ------------------------------------------------------------------
    emit("## 해석 가이드")
    emit()
    emit("- **F1 Drop > 0.05**: 모델이 KO 입력에서 유의미하게 성능 저하")
    emit("- **F1 Drop ≈ 0**: 모델이 언어 변화에 robust (KO safety 일반화 성공)")
    emit("- **Flip Rate 높음**: 같은 샘플에 대해 EN/KO 예측이 다름 → 언어 편향")
    emit("- **AUROC(KO) < AUROC(EN)**: 확률 분리 능력 자체가 KO에서 약해짐")
    emit()
    emit("논문/발표 핵심 논점:")
    emit("  RQ1. 영어 전용 SIREN이 KO에서도 충분한 recall을 유지하는가?")
    emit("  RQ2. 두 모델의 KO 일반화 능력 차이는 backbone 아키텍처와 연관되는가?")
    emit()

    # ------------------------------------------------------------------
    # 저장
    # ------------------------------------------------------------------
    out_path = adir / "experiment_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n✅  요약 저장: {out_path}")


if __name__ == "__main__":
    main()
