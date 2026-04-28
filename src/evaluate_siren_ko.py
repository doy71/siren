#!/usr/bin/env python3
"""Evaluate an official-format Ko-SIREN best_model.pkl on K-SafeEval-R rows.

Output is the same raw_predictions.jsonl schema used by src/analyze.py.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from common import append_jsonl, prediction_key, read_jsonl
from prepare_siren_dataset import convert_rows
from siren_ext.config import resolve_model_config
from siren_ext.model_hooks import GenericRepresentationExtractor
from siren_ext.aggregation import aggregate_features


def predict_prob(model, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            logits = model(torch.as_tensor(X[i : i + batch_size], dtype=torch.float32))
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
    return np.asarray(probs, dtype=np.float32)


def load_existing_keys(path: Path) -> set[Tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for row in read_jsonl(path):
        if {"id", "evaluator", "lang"}.issubset(row):
            keys.add(prediction_key(row))
    return keys


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate Ko-SIREN best_model.pkl")
    p.add_argument("--data", required=True, help="Original K-SafeEval-R paired JSONL")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pkl or its directory")
    p.add_argument("--output", default="results/raw_predictions.jsonl")
    p.add_argument("--evaluator-name", required=True)
    p.add_argument("--langs", nargs="+", default=["en", "ko"], choices=["en", "ko"])
    p.add_argument("--split", default="test", choices=["all", "train", "val", "test"])
    p.add_argument("--mode", default="siren_pair", choices=["siren_pair", "prompt_only", "response_only"])
    p.add_argument("--model", default=None, help="Registry key; defaults to checkpoint['model']")
    p.add_argument("--model-name-or-path", default=None, help="Override checkpoint model path")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", default=None)
    p.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--threshold", type=float, default=None, help="Default: checkpoint['threshold'] or 0.5")
    p.add_argument("--no-trust-remote-code", action="store_true")
    p.add_argument("--no-fast-tokenizer", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "best_model.pkl"
    with ckpt_path.open("rb") as f:
        siren_model = pickle.load(f)
    pooling_type = siren_model["pooling_type"]
    selected_neurons_dict = siren_model["selected_neurons_dict"]
    layer_weights = siren_model["layer_weights"]
    selected_layers = siren_model["selected_layers"]
    final_mlp = siren_model["final_mlp"].cpu().eval()
    threshold = float(args.threshold if args.threshold is not None else siren_model.get("threshold", 0.5))

    original_rows = read_jsonl(args.data)
    rows = convert_rows(original_rows, args.langs, args.mode, args.split)
    if not rows:
        raise SystemExit("No rows selected for evaluation")

    model_key = args.model or siren_model.get("model")
    model_path = args.model_name_or_path or siren_model.get("model_name_or_path")
    cfg = resolve_model_config(model_key, model_path)
    extractor = GenericRepresentationExtractor(
        cfg["model_path"],
        device=args.device,
        batch_size=args.batch_size,
        rep_types=[pooling_type],
        torch_dtype=args.torch_dtype,
        trust_remote_code=not args.no_trust_remote_code,
        max_length=args.max_length,
        device_map=args.device_map,
        no_fast_tokenizer=args.no_fast_tokenizer,
    )
    extractor.register_hooks()
    representations = []
    try:
        texts = [r["text"] for r in rows]
        for i in tqdm(range(0, len(texts), args.batch_size), desc="Extract eval representations"):
            representations.extend(extractor.extract_batch(texts[i : i + args.batch_size]))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        extractor.close()

    X, _ = aggregate_features(representations, pooling_type, selected_neurons_dict, layer_weights, selected_layers)
    probs = predict_prob(final_mlp, X)

    out_path = Path(args.output)
    if args.overwrite and out_path.exists():
        out_path.unlink()
    existing = set() if args.overwrite else load_existing_keys(out_path)
    now = datetime.now(timezone.utc).isoformat()
    out_rows = []
    for row, p_unsafe in zip(rows, probs.tolist()):
        pred = "unsafe" if p_unsafe >= threshold else "safe"
        out = {
            "id": row["id"],
            "evaluator": args.evaluator_name,
            "lang": row["lang"],
            "pred": pred,
            "unsafe_prob": float(p_unsafe),
            "raw_output": f"unsafe_prob={p_unsafe:.6f}; threshold={threshold:.6f}",
            "model_name": args.evaluator_name,
            "backbone_name_or_path": cfg["model_path"],
            "checkpoint": str(ckpt_path),
            "split": row.get("split", ""),
            "created_at": now,
        }
        key = prediction_key(out)
        if key not in existing:
            out_rows.append(out)
            existing.add(key)
    append_jsonl(out_rows, out_path)
    print(f"Wrote {len(out_rows)} predictions to {out_path}")


if __name__ == "__main__":
    main()
    if os.environ.get("KSIREN_FORCE_EXIT") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
