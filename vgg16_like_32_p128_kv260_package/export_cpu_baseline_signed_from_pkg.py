#!/usr/bin/env python3
"""
Export generated CNN_V4 package tensors into raw files for a SIGNED INT8 CPU baseline.

Logical convention for the exported CPU baseline:
  - input tensor: signed int8 HWC bytes
  - weights: signed int8 F,C,K,K bytes
  - bias: signed int32 per output channel, optional; zeros are generated if absent
  - output/golden: signed int8 HWC bytes
  - compute: int32 accumulation, optional OUT_SHIFT in C++ baseline, ReLU, signed saturation [-128,127]

This version is for the user's current package convention:
  - The files are still named input_uint8_hwc.npy and golden_final_ofm_uint8_hwc.npy.
  - Their contents should be treated as SIGNED INT8/two's-complement bytes.

Therefore the old *_uint8_hwc.npy names are searched FIRST and interpreted as
raw signed-int8 bytes by the C++ baseline. New *_i8_hwc.npy names are still
accepted as fallback for future packages.
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def read_layers_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_int(x: Any, default: int = 0) -> int:
    if x is None or str(x).strip() == "":
        return int(default)
    s = str(x).strip()
    return int(s, 16) if s.lower().startswith("0x") else int(float(s))


def first_existing(candidates: List[Path], desc: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    msg = "\n".join(f"  {p}" for p in candidates)
    raise FileNotFoundError(f"Cannot find {desc}. Tried:\n{msg}")


def load_3d_raw_i8_bytes(path: Path, desc: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"{desc}: expected 3-D HWC tensor, got shape={arr.shape} from {path}")

    if arr.dtype == np.int8 or arr.dtype == np.uint8:
        # Keep raw bytes exactly. uint8 bytes 128..255 are later interpreted as signed int8.
        return np.ascontiguousarray(arr)

    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{desc}: expected int8/uint8/integer tensor, got dtype={arr.dtype} from {path}")

    mn = int(arr.min())
    mx = int(arr.max())

    if -128 <= mn and mx <= 127:
        # Numeric signed values stored in a wider integer dtype.
        return np.ascontiguousarray(arr.astype(np.int8))

    if 0 <= mn and mx <= 255:
        # Raw two's-complement bytes stored in a wider unsigned/integer dtype.
        # Preserve byte pattern; C++ baseline interprets them as int8_t.
        return np.ascontiguousarray(arr.astype(np.uint8))

    raise ValueError(
        f"{desc}: dtype={arr.dtype} range=[{mn},{mx}] cannot be represented as signed int8 bytes"
    )


def write_raw_bytes(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # view(np.uint8) preserves two's-complement bytes for int8 arrays.
    np.ascontiguousarray(arr).view(np.uint8).tofile(path)


def find_weight(weights_dir: Path, layer_id: int, layer_name: str) -> Path:
    layers_dir = weights_dir / "layers"
    candidates = [
        layers_dir / f"layer{layer_id:02d}_{layer_name}_weight_i8.npy",
        layers_dir / f"layer{layer_id:02d}_L{layer_id}_weight_i8.npy",
    ]
    for c in candidates:
        if c.exists():
            return c
    hits = sorted(layers_dir.glob(f"layer{layer_id:02d}_*_weight_i8.npy"))
    if hits:
        return hits[0]
    hits = sorted(weights_dir.glob(f"**/layer{layer_id:02d}_*_weight_i8.npy"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Cannot find weight_i8 .npy for layer {layer_id} in {weights_dir}")


def find_bias(weights_dir: Path, layer_id: int, layer_name: str) -> Optional[Path]:
    layers_dir = weights_dir / "layers"
    patterns = [
        f"layer{layer_id:02d}_{layer_name}_bias_i32.npy",
        f"layer{layer_id:02d}_L{layer_id}_bias_i32.npy",
        f"layer{layer_id:02d}_*_bias_i32.npy",
        f"layer{layer_id:02d}_{layer_name}_bias.npy",
        f"layer{layer_id:02d}_*_bias.npy",
    ]
    for pat in patterns:
        hits = sorted(layers_dir.glob(pat)) if "*" in pat else [layers_dir / pat]
        for h in hits:
            if h.exists():
                return h
    hits = sorted(weights_dir.glob(f"**/layer{layer_id:02d}_*_bias*.npy"))
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg-root", required=True, help="Path to vgg16_like_32_p128_kv260_package")
    ap.add_argument("--bench", default="vgg16_like_32_p128")
    ap.add_argument("--image", default="img0000")
    ap.add_argument("--out", default="cpu_baseline_signed_run")
    ap.add_argument("--layers-csv", default=None, help="Optional explicit layers CSV path")
    ap.add_argument("--input", default=None, help="Optional explicit signed input .npy path")
    ap.add_argument("--golden", default=None, help="Optional explicit signed final OFM .npy path")
    ap.add_argument("--weights-dir", default=None, help="Optional explicit weights directory")
    args = ap.parse_args()

    root = Path(args.pkg_root).resolve()
    bench = args.bench
    image = args.image
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    layers_csv = Path(args.layers_csv).resolve() if args.layers_csv else root / "eval" / "configs" / f"{bench}_layers.csv"
    weights_dir = Path(args.weights_dir).resolve() if args.weights_dir else root / "eval" / "assets" / bench / "weights"

    input_dir = root / "eval" / "assets" / bench / "inputs" / image
    golden_dir = root / "eval" / "golden" / bench / image

    if args.input:
        input_npy = Path(args.input).resolve()
    else:
        # IMPORTANT: current local package keeps the old name, but the bytes are signed int8/two's-complement.
        # Search the old name first so the behavior is unambiguous.
        input_npy = first_existing([
            input_dir / "input_uint8_hwc.npy",
            input_dir / "input_i8_hwc.npy",
            input_dir / "input_int8_hwc.npy",
            input_dir / "input_signed_i8_hwc.npy",
            input_dir / "input_s8_hwc.npy",
        ], "signed input HWC .npy using the existing input_uint8_hwc.npy name")

    if args.golden:
        golden_npy = Path(args.golden).resolve()
    else:
        # IMPORTANT: current local package keeps the old name, but the bytes are signed int8/two's-complement.
        # Search the old name first so the behavior is unambiguous.
        golden_npy = first_existing([
            golden_dir / "golden_final_ofm_uint8_hwc.npy",
            golden_dir / "golden_final_ofm_i8_hwc.npy",
            golden_dir / "golden_final_ofm_int8_hwc.npy",
            golden_dir / "golden_final_ofm_signed_i8_hwc.npy",
            golden_dir / "golden_final_ofm_s8_hwc.npy",
            golden_dir / "expected_final_ofm_i8_hwc.npy",
        ], "signed final OFM golden .npy using the existing golden_final_ofm_uint8_hwc.npy name")

    missing = [p for p in [layers_csv, weights_dir, input_npy, golden_npy] if not p.exists()]
    if missing:
        print("ERROR: missing required files/directories:")
        for p in missing:
            print("  ", p)
        raise SystemExit(1)

    layers = read_layers_csv(layers_csv)

    x = load_3d_raw_i8_bytes(input_npy, "input")
    y = load_3d_raw_i8_bytes(golden_npy, "golden")

    write_raw_bytes(x, out / "ifm_l0_i8_hwc.bin")
    write_raw_bytes(y, out / "expected_final_ofm_i8_hwc.bin")

    weight_info = []
    bias_info = []
    for row in layers:
        layer_id = as_int(row.get("layer_id"), len(weight_info))
        layer_name = row.get("layer_name") or f"L{layer_id}"
        f_out = as_int(row.get("f_out"))
        c_in = as_int(row.get("c_in"))
        k = as_int(row.get("k"), 3)

        wp = find_weight(weights_dir, layer_id, layer_name)
        w = np.load(wp)
        exp_shape = (f_out, c_in, k, k)
        if w.dtype != np.int8 or tuple(w.shape) != exp_shape:
            raise ValueError(f"{wp}: expected int8 shape {exp_shape}, got dtype={w.dtype}, shape={w.shape}")
        write_raw_bytes(w, out / f"w_l{layer_id}_i8.bin")
        weight_info.append({"layer_id": layer_id, "path": str(wp), "shape": list(map(int, w.shape))})

        bp = find_bias(weights_dir, layer_id, layer_name)
        if bp is not None:
            b = np.load(bp)
            if b.shape != (f_out,):
                raise ValueError(f"{bp}: expected bias shape {(f_out,)}, got {b.shape}")
            if b.dtype != np.int32:
                if not np.issubdtype(b.dtype, np.integer):
                    raise ValueError(f"{bp}: expected integer bias, got {b.dtype}")
                b = b.astype(np.int32)
            bias_note = str(bp)
        else:
            b = np.zeros((f_out,), dtype=np.int32)
            bias_note = "zeros_generated"
        b.astype(np.int32).tofile(out / f"b_l{layer_id}_i32.bin")
        bias_info.append({"layer_id": layer_id, "path": bias_note, "shape": [int(f_out)]})

    # Compact layer CSV for C++.
    cpp_csv = out / "layers_cpu.csv"
    fields = [
        "layer_id", "layer_name", "h_in", "w_in", "c_in", "f_out", "k",
        "pad_top", "pad_bottom", "pad_left", "pad_right", "relu_en",
        "pool_en", "pool_k", "pool_stride", "h_out", "w_out",
    ]
    with cpp_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in layers:
            writer.writerow({k: row.get(k, "") for k in fields})

    meta = {
        "convention": "SIGNED_INT8_HWC, weights SIGNED_INT8 F,C,K,K, bias INT32, output SIGNED_INT8_HWC",
        "pkg_root": str(root),
        "bench": bench,
        "image": image,
        "layers_csv": str(layers_csv),
        "weights_dir": str(weights_dir),
        "input_npy": str(input_npy),
        "input_original_dtype": str(x.dtype),
        "input_shape_hwc": list(map(int, x.shape)),
        "input_signed_minmax": [int(x.view(np.int8).min()), int(x.view(np.int8).max())],
        "golden_npy": str(golden_npy),
        "golden_original_dtype": str(y.dtype),
        "golden_shape_hwc": list(map(int, y.shape)),
        "golden_signed_minmax": [int(y.view(np.int8).min()), int(y.view(np.int8).max())],
        "num_layers": len(layers),
        "weights": weight_info,
        "biases": bias_info,
        "note": "Current package may keep *_uint8_hwc.npy names, but raw bytes are exported and interpreted as signed int8 two's-complement values by the C++ baseline."
    }
    with (out / "cpu_baseline_signed_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[OK] exported SIGNED INT8 CPU baseline data to", out)
    print("     input :", tuple(x.shape), x.dtype, "signed range", meta["input_signed_minmax"])
    print("     golden:", tuple(y.shape), y.dtype, "signed range", meta["golden_signed_minmax"])
    print("     layers:", len(layers))
    print("Next:")
    print("  copy cpu_baseline_signed_i8.cpp into this directory")
    print("  g++ -O3 -std=c++17 cpu_baseline_signed_i8.cpp -o cpu_baseline_signed_i8")
    print("  ./cpu_baseline_signed_i8 1000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
