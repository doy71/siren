#!/usr/bin/env python3
"""Train an official-format SIREN model on Korean/bilingual safety data.

This follows CSSLab/SIREN's train_general_siren.py structure:
  1. extract residual_mean/mlp_mean representations;
  2. train layer-wise L1 linear probes;
  3. select salient neurons by cumulative importance threshold;
  4. weight layers by validation F1;
  5. train an AdaptiveMLPClassifier over aggregated features;
  6. save best_model.pkl with official-compatible keys.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from tqdm import tqdm

from common import read_jsonl, stable_hash_int
from siren_ext.config import resolve_model_config
from siren_ext.mlp import AdaptiveMLPClassifier
from siren_ext.model_hooks import GenericRepresentationExtractor
from siren_ext.probe_trainer import train_and_evaluate_probe, extract_layer_features, compute_per_dataset_f1
from siren_ext.aggregation import aggregate_features


def load_siren_jsonl(path: str | Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    texts = [str(r["text"]) for r in rows]
    labels = np.asarray([int(r["label"]) for r in rows], dtype=np.int64)
    dataset_ids = np.asarray([stable_hash_int(str(r.get("source", "ksafeevalr")), 10_000) for r in rows], dtype=np.int64)
    meta = rows
    return {"texts": texts, "labels": labels, "dataset_ids": dataset_ids, "meta": meta}


def make_mock_representations(rows: list[dict[str, Any]], rep_types: list[str], *, num_layers: int, hidden_size: int) -> list[dict[int, dict[str, np.ndarray]]]:
    reps = []
    for row in rows:
        label = int(row["label"])
        seed = stable_hash_int(f"{row.get('id')}:{row.get('lang')}:{row.get('text')}", 2**32)
        rng = np.random.default_rng(seed)
        sample: dict[int, dict[str, np.ndarray]] = {}
        for layer in range(num_layers):
            layer_rep: dict[str, np.ndarray] = {}
            signal = (layer + 1) / num_layers
            for rep_type in rep_types:
                vec = rng.normal(0, 0.2, size=hidden_size).astype(np.float32)
                vec[: min(16, hidden_size)] += (1 if label else -1) * signal
                layer_rep[rep_type] = vec
            sample[layer] = layer_rep
        reps.append(sample)
    return reps


def extract_all_representations(args: argparse.Namespace, split_data: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if args.mock:
        out = {}
        for split_name, data in split_data.items():
            out[split_name] = {
                "representations": make_mock_representations(
                    data["meta"], args.pooling_types, num_layers=args.mock_num_layers, hidden_size=args.mock_hidden_size
                ),
                "labels": data["labels"],
                "dataset_ids": data["dataset_ids"],
                "num_layers": args.mock_num_layers,
            }
        return out

    cfg = resolve_model_config(args.model, args.model_name_or_path)
    extractor = GenericRepresentationExtractor(
        cfg["model_path"],
        device=args.device,
        batch_size=args.batch_size,
        rep_types=args.pooling_types,
        torch_dtype=args.torch_dtype,
        trust_remote_code=not args.no_trust_remote_code,
        max_length=args.max_length,
        device_map=args.device_map,
        no_fast_tokenizer=args.no_fast_tokenizer,
    )
    extractor.register_hooks()
    out = {}
    try:
        for split_name, data in split_data.items():
            print(f"\nProcessing {split_name} split: {len(data['texts'])} samples")
            representations = []
            for i in tqdm(range(0, len(data["texts"]), args.batch_size), desc=f"Extracting {split_name}"):
                batch_texts = data["texts"][i : i + args.batch_size]
                with torch.no_grad():
                    representations.extend(extractor.extract_batch(batch_texts))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            out[split_name] = {
                "representations": representations,
                "labels": data["labels"],
                "dataset_ids": data["dataset_ids"],
                "num_layers": extractor.num_layers,
            }
    finally:
        extractor.close()
    return out


def train_probes(train_reps, train_labels, val_reps, val_labels, val_dataset_ids, num_layers, c_values, pooling_types, device):
    best_probes = {}
    for pooling_type in pooling_types:
        print(f"\nTraining {pooling_type} probes...")
        for layer_idx in range(num_layers):
            try:
                probe, train_f1, val_f1, best_C = train_and_evaluate_probe(
                    train_reps,
                    train_labels,
                    val_reps,
                    val_labels,
                    val_dataset_ids,
                    layer_idx,
                    pooling_type,
                    c_values,
                    device,
                    metric="f1_macro",
                )
            except KeyError as exc:
                print(f"  Skipping layer {layer_idx} for {pooling_type}: {exc}")
                continue
            key = f"layer{layer_idx}_{pooling_type}"
            best_probes[key] = {
                "layer": layer_idx,
                "pooling_type": pooling_type,
                "best_C": best_C,
                "train_f1": float(train_f1),
                "val_f1": float(val_f1),
                "probe": probe,
            }
            print(f"  Layer {layer_idx:2d}: Train_F1={train_f1:.4f} Val_F1={val_f1:.4f} C={best_C}")
    return best_probes


def select_salient_neurons(probe, threshold: float) -> list[int]:
    weights = probe.get_feature_importance()
    total_importance = float(np.sum(weights))
    sorted_indices = np.argsort(weights)[::-1]
    if total_importance <= 0:
        return [int(i) for i in sorted_indices[: min(64, len(sorted_indices))]]
    selected_indices: list[int] = []
    cumulative_importance = 0.0
    for idx in sorted_indices:
        selected_indices.append(int(idx))
        cumulative_importance += float(weights[idx])
        if cumulative_importance >= threshold * total_importance:
            break
    return selected_indices


def get_layer_weights(best_probes, pooling_type: str, num_layers: int) -> dict[int, float]:
    layer_scores = {}
    for layer_idx in range(num_layers):
        key = f"layer{layer_idx}_{pooling_type}"
        if key in best_probes:
            layer_scores[layer_idx] = float(best_probes[key]["val_f1"])
    if not layer_scores:
        return {}
    max_score = max(layer_scores.values())
    min_score = min(layer_scores.values())
    score_range = max_score - min_score if max_score > min_score else 1.0
    return {layer_idx: max(0.1, (score - min_score) / score_range) for layer_idx, score in layer_scores.items()}


def train_final_model(X_train, y_train, X_val, y_val, val_dataset_ids, args) -> tuple[AdaptiveMLPClassifier, dict[str, Any]]:
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    input_dim = int(X_train.shape[1])
    layer_dims = [int(x) for x in args.mlp_dims]
    dropout_rates = [float(args.dropout)] * len(layer_dims)
    model = AdaptiveMLPClassifier(input_dim, layer_dims, dropout_rates).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_val_f1 = -1.0
    best_state = None
    patience_counter = 0

    X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32)

    for epoch in tqdm(range(args.epochs), desc="Training final MLP"):
        model.train()
        indices = torch.randperm(len(X_train_t))
        for i in range(0, len(X_train_t), args.mlp_batch_size):
            idx = indices[i : i + args.mlp_batch_size]
            batch_X = X_train_t[idx].to(device)
            batch_y = y_train_t[idx].to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_val_t), args.mlp_batch_size):
                logits = model(X_val_t[i : i + args.mlp_batch_size].to(device))
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
        val_f1 = compute_per_dataset_f1(y_val, np.asarray(preds), val_dataset_ids)
        if val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu().eval()
    return model, {"input_dim": input_dim, "layer_dims": layer_dims, "dropout_rates": dropout_rates, "val_f1": best_val_f1}


def predict_with_mlp(model: AdaptiveMLPClassifier, X: np.ndarray, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    probs = []
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            logits = model(torch.as_tensor(X[i : i + batch_size], dtype=torch.float32))
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p.tolist())
            preds.extend((p >= 0.5).astype(int).tolist())
    return np.asarray(preds, dtype=np.int64), np.asarray(probs, dtype=np.float32)


def metric_dict(y_true: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "precision_unsafe": float(precision_score(y_true, pred, zero_division=0)),
        "recall_unsafe": float(recall_score(y_true, pred, zero_division=0)),
    }
    if len(set(y_true.tolist())) == 2:
        out["auroc"] = float(roc_auc_score(y_true, prob))
        out["auprc"] = float(average_precision_score(y_true, prob))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train Ko-SIREN in official SIREN-compatible format")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--test-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="qwen3-0.6b", help="Registry key; see siren_ext/config.py")
    p.add_argument("--model-name-or-path", default=None, help="Override registry model path")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--pooling-types", nargs="+", default=["residual_mean"], help="e.g., residual_mean mlp_mean")
    p.add_argument("--final-pooling-type", default="residual_mean")
    p.add_argument("--c-values", type=float, nargs="+", default=[50.0, 100.0, 200.0])
    p.add_argument("--threshold", type=float, default=0.8, help="cumulative importance threshold for salient neurons")
    p.add_argument("--mlp-dims", type=int, nargs="+", default=[256, 128])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--mlp-batch-size", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", default=None)
    p.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--no-trust-remote-code", action="store_true")
    p.add_argument("--no-fast-tokenizer", action="store_true")
    p.add_argument("--mock", action="store_true", help="CPU-only deterministic smoke test")
    p.add_argument("--mock-num-layers", type=int, default=4)
    p.add_argument("--mock-hidden-size", type=int, default=64)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_data = {
        "train": load_siren_jsonl(args.train_jsonl),
        "validation": load_siren_jsonl(args.val_jsonl),
        "test": load_siren_jsonl(args.test_jsonl),
    }
    print(f"Model: {args.model} | pooling: {args.pooling_types} | final={args.final_pooling_type}")
    print("[1/3] Extracting representations...")
    all_reps = extract_all_representations(args, split_data)

    print("\n[2/3] Training sparse layer probes...")
    best_probes = train_probes(
        all_reps["train"]["representations"],
        all_reps["train"]["labels"],
        all_reps["validation"]["representations"],
        all_reps["validation"]["labels"],
        all_reps["validation"]["dataset_ids"],
        all_reps["train"]["num_layers"],
        args.c_values,
        args.pooling_types,
        args.device,
    )
    with (out_dir / "probes.pkl").open("wb") as f:
        pickle.dump({"best_probes": best_probes, "model": args.model, "dataset": "ksafeevalr"}, f)

    print("\n[3/3] Selecting neurons, aggregating layers, training final MLP...")
    pooling_type = args.final_pooling_type
    layer_weights = get_layer_weights(best_probes, pooling_type, all_reps["train"]["num_layers"])
    if not layer_weights:
        raise SystemExit(f"No layer weights for pooling_type={pooling_type}; available keys={list(best_probes)[:5]}")
    selected_neurons_dict: dict[str, list[int]] = {}
    for layer_idx in layer_weights.keys():
        key = f"layer{layer_idx}_{pooling_type}"
        if key in best_probes:
            selected_neurons_dict[key] = select_salient_neurons(best_probes[key]["probe"], args.threshold)

    X_train, selected_layers = aggregate_features(all_reps["train"]["representations"], pooling_type, selected_neurons_dict, layer_weights)
    X_val, _ = aggregate_features(all_reps["validation"]["representations"], pooling_type, selected_neurons_dict, layer_weights, selected_layers)
    X_test, _ = aggregate_features(all_reps["test"]["representations"], pooling_type, selected_neurons_dict, layer_weights, selected_layers)
    print(f"Aggregated dims: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    final_mlp, mlp_info = train_final_model(
        X_train,
        all_reps["train"]["labels"],
        X_val,
        all_reps["validation"]["labels"],
        all_reps["validation"]["dataset_ids"],
        args,
    )
    test_pred, test_prob = predict_with_mlp(final_mlp, X_test)
    val_pred, val_prob = predict_with_mlp(final_mlp, X_val)
    train_pred, train_prob = predict_with_mlp(final_mlp, X_train)
    metrics = {
        "train": metric_dict(all_reps["train"]["labels"], train_pred, train_prob),
        "validation": metric_dict(all_reps["validation"]["labels"], val_pred, val_prob),
        "test": metric_dict(all_reps["test"]["labels"], test_pred, test_prob),
        "mlp_info": mlp_info,
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # Official-compatible keys: evaluate_general_siren.py expects these names.
    model_obj = {
        "pooling_type": pooling_type,
        "selected_neurons_dict": selected_neurons_dict,
        "layer_weights": {str(k): float(v) for k, v in layer_weights.items()},
        "selected_layers": [int(x) for x in selected_layers],
        "final_mlp": final_mlp,
        "best_params": {
            "n_layers": len(args.mlp_dims),
            **{f"hidden_dim_layer{i}": int(v) for i, v in enumerate(args.mlp_dims)},
            **{f"dropout_layer{i}": float(args.dropout) for i in range(len(args.mlp_dims))},
            "lr": float(args.lr),
        },
        "model": args.model,
        "model_name_or_path": args.model_name_or_path or resolve_model_config(args.model)["model_path"],
        "dataset": "ksafeevalr",
        "threshold": 0.5,
        "salience_threshold": args.threshold,
        "pooling_types_trained": args.pooling_types,
        "num_layers": all_reps["train"]["num_layers"],
        "metrics": metrics,
    }
    with (out_dir / "best_model.pkl").open("wb") as f:
        pickle.dump(model_obj, f)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (out_dir / "selected_neurons.json").open("w", encoding="utf-8") as f:
        json.dump({k: list(map(int, v)) for k, v in selected_neurons_dict.items()}, f, ensure_ascii=False, indent=2)
    print(f"Saved official-compatible model to {out_dir / 'best_model.pkl'}")


if __name__ == "__main__":
    main()
    if os.environ.get("KSIREN_FORCE_EXIT") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
