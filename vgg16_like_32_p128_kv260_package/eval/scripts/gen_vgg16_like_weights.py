#!/usr/bin/env python3
"""
gen_vgg16_like_weights.py
Generate deterministic INT8 weights for CNN_V4 VGG16-like benchmarks.

The generated weights are intended for RTL/FPGA functional verification rather
than model accuracy.  Several deterministic patterns are provided:

  - center_tap_identity:
      one center tap of +1 per output filter; safest for layout smoke tests.

  - single_tap_varied / two_tap_varied:
      sparse, non-saturating-ish patterns with varied channel/tap selection.

  - balanced_sparse_varied:
      recommended correctness/demo pattern.  Each output filter receives a
      small deterministic set of nonzero taps spread over input channels and
      kernel positions, with signed values from {-max_abs..-1, +1..+max_abs}.
      This exercises signed weights, channel selection, kernel positions, and
      accumulation while keeping density low enough to reduce full saturation.

  - deterministic_small / random_small:
      denser stress patterns.  These may saturate after many layers.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

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


def pos_to_ckk(pos: int, k: int) -> Tuple[int, int, int]:
    """Map flattened C*K*K position into (c, ky, kx)."""
    kk = k * k
    c = pos // kk
    rem = pos % kk
    ky = rem // k
    kx = rem % k
    return c, ky, kx


def signed_value_from_index(layer_id: int, f: int, j: int, max_abs: int) -> int:
    """Deterministic signed nonzero value in [-max_abs,-1] U [1,max_abs]."""
    if max_abs < 1:
        raise ValueError("max_abs must be >= 1")
    mag = 1 + ((17 * layer_id + 7 * f + 5 * j) % max_abs)
    sign = -1 if ((layer_id + f + j) & 1) else 1
    return sign * mag


def make_center_tap_identity(layer_id: int, f_out: int, c_in: int, k: int) -> np.ndarray:
    w = np.zeros((f_out, c_in, k, k), dtype=np.int8)
    center = k // 2
    for f in range(f_out):
        c = f % c_in
        w[f, c, center, center] = np.int8(1)
    return w


def make_single_tap_varied(layer_id: int, f_out: int, c_in: int, k: int) -> np.ndarray:
    """One nonzero tap/filter, but channel, position, and sign vary."""
    w = np.zeros((f_out, c_in, k, k), dtype=np.int8)
    for f in range(f_out):
        c = (f * 3 + layer_id * 5) % c_in
        ky = (f + layer_id) % k
        kx = (2 * f + layer_id) % k
        # Mostly +1, with occasional -1 to exercise signed weights without
        # making the first correctness pattern too aggressive.
        val = -1 if ((f + layer_id) % 5 == 0) else 1
        w[f, c, ky, kx] = np.int8(val)
    return w


def make_two_tap_varied(layer_id: int, f_out: int, c_in: int, k: int) -> np.ndarray:
    """Two nonzero taps/filter, one positive and one negative, deterministic."""
    w = np.zeros((f_out, c_in, k, k), dtype=np.int8)
    for f in range(f_out):
        c0 = (f * 3 + layer_id * 5) % c_in
        ky0 = (f + layer_id) % k
        kx0 = (2 * f + layer_id) % k

        c1 = (f * 5 + layer_id * 7 + 1) % c_in
        ky1 = (2 * f + layer_id + 1) % k
        kx1 = (f + 2 * layer_id + 2) % k

        w[f, c0, ky0, kx0] = np.int8(1)
        # Avoid exact same slot cancellation. If it collides, move kx.
        if (c1, ky1, kx1) == (c0, ky0, kx0):
            kx1 = (kx1 + 1) % k
        w[f, c1, ky1, kx1] = np.int8(-1)
    return w


def make_balanced_sparse_varied(
    layer_id: int,
    f_out: int,
    c_in: int,
    k: int,
    seed: int,
    nonzero_per_filter: int,
    max_abs_weight: int,
) -> np.ndarray:
    """
    Sparse multi-tap deterministic weights.

    This is a better functional-verification pattern than single-tap identity:
    it uses multiple input channels, multiple kernel positions, positive and
    negative weights, and small magnitudes.  Density is intentionally low to
    avoid all outputs saturating to 0x7F after many layers.
    """
    if nonzero_per_filter <= 0:
        raise ValueError("nonzero_per_filter must be > 0")
    if max_abs_weight <= 0 or max_abs_weight > 7:
        raise ValueError("max_abs_weight must be in 1..7 for safe INT8 debug weights")

    w = np.zeros((f_out, c_in, k, k), dtype=np.int8)
    total_positions = c_in * k * k
    taps = min(nonzero_per_filter, total_positions)

    for f in range(f_out):
        # Deterministic per-filter RNG.  This makes every layer/filter different
        # but reproducible across machines.
        rng_seed = (int(seed) * 1_000_003 + layer_id * 10_007 + f * 97) & 0xFFFFFFFF
        rng = np.random.default_rng(rng_seed)
        positions = rng.choice(total_positions, size=taps, replace=False)

        # To limit DC growth, make signs roughly balanced.
        vals = []
        for j in range(taps):
            vals.append(signed_value_from_index(layer_id, f, j, max_abs_weight))

        # If taps is even, force exact sign balance by flipping excess signs.
        # This does not guarantee zero sum, but avoids all-positive filters.
        pos_count = sum(1 for v in vals if v > 0)
        neg_count = taps - pos_count
        if taps >= 2 and abs(pos_count - neg_count) > 1:
            for j, v in enumerate(vals):
                if pos_count > neg_count and v > 0:
                    vals[j] = -abs(v)
                    pos_count -= 1
                    neg_count += 1
                elif neg_count > pos_count and v < 0:
                    vals[j] = abs(v)
                    neg_count -= 1
                    pos_count += 1
                if abs(pos_count - neg_count) <= 1:
                    break

        for pos, val in zip(positions, vals):
            c, ky, kx = pos_to_ckk(int(pos), k)
            w[f, c, ky, kx] = np.int8(val)

    return w


def make_weight(
    layer_id: int,
    f_out: int,
    c_in: int,
    k: int,
    pattern: str,
    seed: int,
    nonzero_per_filter: int,
    max_abs_weight: int,
) -> np.ndarray:
    if pattern == "deterministic_small":
        w = np.zeros((f_out, c_in, k, k), dtype=np.int8)
        for f in range(f_out):
            for c in range(c_in):
                for ky in range(k):
                    for kx in range(k):
                        # Dense stress pattern; may saturate after many layers.
                        val = ((3 * f + 5 * c + 7 * ky + 11 * kx + 13 * layer_id) % 5) - 2
                        w[f, c, ky, kx] = np.int8(val)
        return w

    if pattern == "random_small":
        rng = np.random.default_rng(seed + layer_id)
        return rng.integers(-2, 3, size=(f_out, c_in, k, k), dtype=np.int8)

    if pattern == "center_tap_identity":
        return make_center_tap_identity(layer_id, f_out, c_in, k)

    if pattern == "single_tap_varied":
        return make_single_tap_varied(layer_id, f_out, c_in, k)

    if pattern == "two_tap_varied":
        return make_two_tap_varied(layer_id, f_out, c_in, k)

    if pattern == "sparse_3tap_varied":
        return make_balanced_sparse_varied(layer_id, f_out, c_in, k, seed, 3, max_abs_weight)

    if pattern == "sparse_6tap_varied":
        return make_balanced_sparse_varied(layer_id, f_out, c_in, k, seed, 6, max_abs_weight)

    if pattern == "balanced_sparse_varied":
        return make_balanced_sparse_varied(
            layer_id, f_out, c_in, k, seed, nonzero_per_filter, max_abs_weight
        )

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
    ap.add_argument(
        "--pattern",
        choices=[
            "deterministic_small",
            "random_small",
            "center_tap_identity",
            "single_tap_varied",
            "two_tap_varied",
            "sparse_3tap_varied",
            "sparse_6tap_varied",
            "balanced_sparse_varied",
        ],
        default="balanced_sparse_varied",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--nonzero-per-filter",
        type=int,
        default=6,
        help="Number of nonzero weights per output filter for balanced_sparse_varied.",
    )
    ap.add_argument(
        "--max-abs-weight",
        type=int,
        default=2,
        help="Max absolute nonzero weight for sparse varied patterns. Recommended: 1 or 2.",
    )
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
        "nonzero_per_filter": args.nonzero_per_filter,
        "max_abs_weight": args.max_abs_weight,
        "tensor_layout": "F,C,K,K",
        "dtype": "int8",
        "notes": [
            "Sparse varied patterns are for functional verification, not trained-model accuracy.",
            "They intentionally avoid dense accumulation to reduce full saturation after many layers.",
        ],
        "layers": []
    }

    for i, row in enumerate(rows):
        layer_id = parse_int(row.get("layer_id"), i)
        name = row.get("layer_name") or f"L{layer_id}"
        f_out = parse_int(row.get("f_out"))
        c_in = parse_int(row.get("c_in"))
        k = parse_int(row.get("k"), 3)

        w = make_weight(
            layer_id=layer_id,
            f_out=f_out,
            c_in=c_in,
            k=k,
            pattern=args.pattern,
            seed=args.seed,
            nonzero_per_filter=args.nonzero_per_filter,
            max_abs_weight=args.max_abs_weight,
        )

        npy_path = layers_dir / f"layer{layer_id:02d}_{name}_weight_i8.npy"
        hex_path = layers_dir / f"layer{layer_id:02d}_{name}_weight_i8_bytes.hex"
        np.save(npy_path, w)
        write_i8_byte_hex(hex_path, w)

        nonzero = int(np.count_nonzero(w))
        density = float(nonzero) / float(w.size) if w.size else 0.0

        metadata["layers"].append({
            "layer_id": layer_id,
            "layer_name": name,
            "shape_f_c_k_k": [int(x) for x in w.shape],
            "weight_i8_npy": str(npy_path),
            "weight_i8_bytes_hex": str(hex_path),
            "min": int(w.min()) if w.size else 0,
            "max": int(w.max()) if w.size else 0,
            "num_weights": int(w.size),
            "nonzero_weights": nonzero,
            "density": density,
            "unique_values": [int(v) for v in np.unique(w).tolist()],
        })
        print(
            f"[OK] L{layer_id:02d} {name}: shape={w.shape} "
            f"nonzero={nonzero}/{w.size} density={density:.4%} "
            f"unique={np.unique(w).tolist()} -> {npy_path}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Weight metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
