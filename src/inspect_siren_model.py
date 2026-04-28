#!/usr/bin/env python3
"""Inspect selected layers/neuron counts from official-format SIREN checkpoints."""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--output", default="results/selected_layer_summary.csv")
    args = p.parse_args()
    rows = []
    for ckpt in args.checkpoints:
        path = Path(ckpt)
        if path.is_dir():
            path = path / "best_model.pkl"
        with path.open("rb") as f:
            obj = pickle.load(f)
        pooling_type = obj.get("pooling_type", "")
        layer_weights = obj.get("layer_weights", {})
        selected = obj.get("selected_neurons_dict", {})
        for key, neuron_ids in selected.items():
            # key format: layer{idx}_{pooling_type}
            layer_str = key.split("_")[0].replace("layer", "")
            rows.append(
                {
                    "checkpoint": str(path),
                    "model": obj.get("model", ""),
                    "model_name_or_path": obj.get("model_name_or_path", ""),
                    "pooling_type": pooling_type,
                    "layer": int(layer_str) if layer_str.isdigit() else layer_str,
                    "n_selected_neurons": len(neuron_ids),
                    "layer_weight": layer_weights.get(str(layer_str), layer_weights.get(int(layer_str), "")) if layer_str.isdigit() else "",
                    "salience_threshold": obj.get("salience_threshold", ""),
                    "test_f1_macro": obj.get("metrics", {}).get("test", {}).get("f1_macro", ""),
                    "test_auroc": obj.get("metrics", {}).get("test", {}).get("auroc", ""),
                }
            )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
    if os.environ.get("KSIREN_FORCE_EXIT") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
