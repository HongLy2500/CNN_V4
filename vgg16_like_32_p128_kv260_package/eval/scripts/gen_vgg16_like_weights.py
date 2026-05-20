#!/usr/bin/env python3
"""
gen_vgg16_like_weights.py
Generate deterministic INT8 weights for the fixed VGG16-like 96x96 P256 benchmark.

This benchmark is not canonical pretrained VGG16, so TorchVision weights do not
match the scaled channel counts. The generated deterministic weights are used by
both the PyTorch golden model and FPGA DDR preload.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_int(x, default=None) -> int:
    if x is None or str(x).strip() == "":
        if default is None:
            raise ValueError("missing integer")
        return int(default)
    s = str(x).strip().replace("_", "")
    return int(s, 16) if s.lower().startswith("0x") else int(float(s))


def read_layers_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def make_weight(layer_id: int, f_out: int, c_in: int, k: int, pattern: str, seed: int) -> np.ndarray:
    if pattern == "deterministic_small":
        w = np.zeros((f_out, c_in, k, k), dtype=np.int8)
        for f in range(f_out):
            for c in range(c_in):
                for ky in range(k):
                    for kx in range(k):
                        # Small signed weights to reduce overflow risk while still
                        # exercising negative/zero/positive paths.
                        val = ((3 * f + 5 * c + 7 * ky + 11 * kx + 13 * layer_id) % 5) - 2
                        w[f, c, ky, kx] = np.int8(val)
        return w

    if pattern == "random_small":
        rng = np.random.default_rng(seed + layer_id)
        return rng.integers(-2, 3, size=(f_out, c_in, k, k), dtype=np.int8)

    raise ValueError(f"Unsupported pattern: {pattern}")


def write_i8_byte_hex(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = arr.reshape(-1)
    with path.open("w", encoding="utf-8") as f:
        for v in flat:
            f.write(f"{int(v) & 0xFF:02X}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pattern", choices=["deterministic_small", "random_small"], default="deterministic_small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    layers_csv = Path(args.layers_csv)
    out_dir = Path(args.out_dir)
    layers_dir = out_dir / "layers"
    meta_path = out_dir / "metadata.json"

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"{out_dir} already has files. Use --overwrite to replace/add deterministically.")

    rows = read_layers_csv(layers_csv)
    layers_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "tool": "gen_vgg16_like_weights.py",
        "source": "deterministic INT8 generated weights",
        "layers_csv": str(layers_csv),
        "pattern": args.pattern,
        "seed": args.seed,
        "tensor_layout": "F,C,K,K",
        "dtype": "int8",
        "layers": []
    }

    for i, row in enumerate(rows):
        layer_id = parse_int(row.get("layer_id"), i)
        name = row.get("layer_name") or f"L{layer_id}"
        f_out = parse_int(row.get("f_out"))
        c_in = parse_int(row.get("c_in"))
        k = parse_int(row.get("k"), 3)
        w = make_weight(layer_id, f_out, c_in, k, args.pattern, args.seed)

        npy_path = layers_dir / f"layer{layer_id:02d}_{name}_weight_i8.npy"
        hex_path = layers_dir / f"layer{layer_id:02d}_{name}_weight_i8_bytes.hex"
        np.save(npy_path, w)
        write_i8_byte_hex(hex_path, w)

        metadata["layers"].append({
            "layer_id": layer_id,
            "layer_name": name,
            "shape_f_c_k_k": [int(x) for x in w.shape],
            "weight_i8_npy": str(npy_path),
            "weight_i8_bytes_hex": str(hex_path),
            "min": int(w.min()),
            "max": int(w.max()),
            "num_weights": int(w.size)
        })
        print(f"[OK] L{layer_id:02d} {name}: shape={w.shape} -> {npy_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Weight metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
