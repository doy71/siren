#!/usr/bin/env python3
"""Step 4/5: 분석 결과 요약 출력.

analyze.py가 생성한 CSV들을 읽어 전체 evaluator 비교를 터미널에 출력하고
results/analysis/experiment_summary.md 로 저장합니다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    raise SystemExit("pandas가 필요합니다: pip install pandas")


EVALUATOR_DISPLAY = {
    "siren_qwen3_4b": "SIREN-Qwen3-4B",
    "siren_llama3_1_8b": "SIREN-Llama-3.1-8B",
    "kanana_safeguard_8b": "Kanana Safeguard 8B",
    "sguard_content_filter_2b": "SGuard ContentFilter 2B",
    "wildguard_ko_3b": "WildGuard-ko 3B",
    "llama_guard": "Llama Guard 3",
    "kosafeguard": "KoSafeGuard",
}


def fmt(val) -> str:
    if pd.isna(val):
        return "n/a"
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return str(val)


def display(ev: str) -> str:
    return EVALUATOR_DISPLAY.get(ev, ev)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="results/analysis")
    args = parser.parse_args()

    adir = Path(args.analysis_dir)
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("# Prompt-Response Safety Guard Experiment: BeaverTails EN vs KO")
    emit(f"분석 디렉토리: {adir}")
    emit()

    t1_path = adir / "table1_overall_metrics.csv"
    if t1_path.exists():
        t1 = pd.read_csv(t1_path)
        t1["_name"] = t1["evaluator"].map(display)
        evaluators = sorted(t1["evaluator"].dropna().unique().tolist(), key=lambda x: display(x))

        emit("## Table 1. Overall Metrics (EN / KO)")
        emit()
        emit(f"{'Evaluator':<30} {'Lang':<6} {'N':>5} {'Acc':>8} {'F1':>8} {'Recall':>8} {'AUROC':>8}")
        emit("-" * 82)
        for ev in evaluators:
            for lang in ["en", "ko"]:
                sub = t1[(t1["evaluator"] == ev) & (t1["lang"] == lang)]
                if sub.empty:
                    continue
                r = sub.iloc[0]
                emit(
                    f"{display(ev):<30} {lang:<6} {int(r['n']):>5} "
                    f"{fmt(r.get('accuracy')):>8} "
                    f"{fmt(r.get('f1_unsafe')):>8} "
                    f"{fmt(r.get('recall_unsafe')):>8} "
                    f"{fmt(r.get('auroc')):>8}"
                )
            emit()
        emit()

        emit("### EN → KO F1 변화")
        emit(f"{'Evaluator':<30} {'F1(EN)':>10} {'F1(KO)':>10} {'EN-KO':>10}")
        emit("-" * 65)
        for ev in evaluators:
            en_row = t1[(t1["evaluator"] == ev) & (t1["lang"] == "en")]
            ko_row = t1[(t1["evaluator"] == ev) & (t1["lang"] == "ko")]
            if en_row.empty or ko_row.empty:
                continue
            f1_en = float(en_row.iloc[0].get("f1_unsafe", float("nan")))
            f1_ko = float(ko_row.iloc[0].get("f1_unsafe", float("nan")))
            emit(f"{display(ev):<30} {fmt(f1_en):>10} {fmt(f1_ko):>10} {f1_en - f1_ko:>+10.4f}")
        emit()
    else:
        emit(f"⚠️  {t1_path} 없음. analyze.py를 먼저 실행하세요.")
        emit()

    t4_path = adir / "table4_language_consistency.csv"
    if t4_path.exists():
        t4 = pd.read_csv(t4_path)
        if not t4.empty:
            t4["evaluator_name"] = t4["evaluator"].map(display)
        emit("## Table 4. Language Consistency (EN↔KO Label Flip)")
        emit()
        cols = [c for c in ["evaluator_name", "n_pairs", "consistency", "label_flip_rate", "mean_abs_score_gap"] if c in t4.columns]
        emit(t4[cols].to_string(index=False) if cols else t4.to_string(index=False))
        emit()
    else:
        emit(f"⚠️  {t4_path} 없음.")
        emit()

    t6_path = adir / "table6_pairwise_disagreement.csv"
    if t6_path.exists():
        t6 = pd.read_csv(t6_path)
        emit("## Table 6. Pairwise Disagreement")
        emit()
        if not t6.empty:
            shown = t6.copy()
            shown["evaluator_a_name"] = shown["evaluator_a"].map(display)
            shown["evaluator_b_name"] = shown["evaluator_b"].map(display)
            cols = ["lang", "evaluator_a_name", "evaluator_b_name", "n_pairs", "disagreement_rate", "cohen_kappa"]
            emit(shown[cols].to_string(index=False))
        else:
            emit("<empty>")
        emit()

    fig_dir = adir / "figures"
    if fig_dir.exists():
        figs = sorted(fig_dir.glob("*.png"))
        if figs:
            emit("## 생성된 시각화")
            for f in figs:
                emit(f"  - {f}")
            emit()

    emit("## 해석 가이드")
    emit()
    emit("- **F1 EN-KO > 0.05**: 영어 대비 한국어에서 unsafe 탐지 성능이 눈에 띄게 낮아진 경우")
    emit("- **Flip Rate 높음**: 같은 prompt-response 쌍을 번역했을 때 EN/KO 예측이 자주 바뀜")
    emit("- **WildGuard-ko의 unsafe_prob**: 모델 카드 예시는 확률을 제공하지 않으므로 binary score로 저장됨")
    emit("- **SGuard unsafe_prob**: 5개 카테고리 중 최대 unsafe 확률로 저장됨")
    emit()

    out_path = adir / "experiment_summary.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n✅  요약 저장: {out_path}")


if __name__ == "__main__":
    main()
