#!/usr/bin/env python3
"""
pack_weights_for_cnn_v4.py
--------------------------
Pack per-layer int8 Conv2d weight tensors into the CNN_V4 weight-DDR stream
expected by cnn_dma_direct -> weight_buffer.

This script is intended to run AFTER:
  1) export_torchvision_weights.py     -> convXX_*_weight_i8.npy
  2) dse_lite_v3_rtlmode.py            -> *_dse.csv, where mode uses RTL encoding
                                           0=MODE1, 1=MODE2

Supported CNN_V4 weight layouts
-------------------------------
Physical weight-buffer word:
  - PTOTAL int8 lanes.
  - cnn_dma_direct reads one or more DDR words and assembles one PTOTAL-lane
    physical weight-buffer word while preserving bit/lane order.
  - With the current KV260 XSCT runner, each DDR word is 32-bit = 4 int8 lanes.

Mode 1, RTL mode=0:
  - logical bundle order: (f_group, c, ky, kx)
  - each logical bundle contains Pf_cur weights:
      bundle[pf] = W[f_group*Pf_cur + pf][c][ky][kx]
  - one physical word packs Pv_cur logical bundles:
      lane = sub_bundle * Pf_cur + pf
      sub_bundle = logical_idx % Pv_cur
  - physical_addr = logical_idx // Pv_cur

Mode 2, RTL mode=1:
  - one physical word stores one (f_group, c_group, ky, kx) PFxPC block
  - lane = pf*PC + pc
  - low PF*PC lanes are active; tails are zero-padded.

The output hex file is a stream of DDR words. Every ceil(PTOTAL/ddr_word_lanes)
consecutive DDR words form one physical weight-buffer word.

Examples
--------
VGG16:
  python eval/scripts/pack_weights_for_cnn_v4.py \
    --weights-dir eval/assets/vgg16_conv13/weights \
    --descriptor eval/models/vgg16_conv13.json \
    --dse-csv eval/results/vgg16_conv13/dse/vgg16_dse.csv \
    --out eval/assets/vgg16_conv13/weights/all_weights_cnnv4.hex \
    --dse-out eval/results/vgg16_conv13/dse/vgg16_dse_with_wgt_base.csv \
    --ptotal 128 \
    --ddr-word-lanes 4 \
    --wgt-base 0x08000 \
    --overwrite

Then generate cnn_layer_desc_pkg.sv from the updated CSV so wgt_ddr_base fields
match the packed stream:
  python eval/scripts/dse_csv_to_layer_pkg_v2_rtlmode.py \
    --csv eval/results/vgg16_conv13/dse/vgg16_dse_with_wgt_base.csv \
    --out cnn_layer_desc_pkg.sv \
    --ptotal 128 --default-relu
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

RTL_MODE1 = 0
RTL_MODE2 = 1


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError(f"ceil_div divisor must be positive, got {b}")
    return (a + b - 1) // b


def is_blank(x: Any) -> bool:
    return x is None or str(x).strip() == ""


def norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def normalize_row(row: Dict[str, Any]) -> Dict[str, str]:
    return {norm_key(k): ("" if v is None else str(v)) for k, v in row.items()}


def as_int(value: Any, default: Optional[int] = None, field: str = "") -> int:
    if is_blank(value):
        if default is None:
            raise ValueError(f"Missing integer field: {field}")
        return int(default)
    s = str(value).strip().replace("_", "")
    try:
        if s.lower().startswith("0x"):
            return int(s, 16)
        return int(float(s))
    except ValueError as exc:
        raise ValueError(f"Cannot parse integer field {field!r}: {value!r}") from exc


def get_first(row: Dict[str, str], candidates: Sequence[str], default: Optional[Any] = None) -> Any:
    for c in candidates:
        k = norm_key(c)
        if k in row and not is_blank(row[k]):
            return row[k]
    return default


def read_descriptor(path: Optional[Path]) -> Optional[List[Dict[str, Any]]]:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("layers"), list):
        return list(data["layers"])
    if isinstance(data, list):
        return data
    raise ValueError(f"Descriptor {path} does not contain a layers list")


def read_dse_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"DSE CSV has no header: {path}")
        rows = [normalize_row(r) for r in reader]
        fieldnames = [norm_key(x) for x in reader.fieldnames]
    if not rows:
        raise ValueError(f"DSE CSV has no rows: {path}")
    return rows, fieldnames


def load_export_metadata(weights_dir: Path) -> Optional[Dict[str, Any]]:
    path = weights_dir / "metadata.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_weight_files(weights_dir: Path, expected_layers: int) -> List[Path]:
    """
    Prefer export_torchvision_weights.py metadata order. If metadata is absent,
    fall back to sorting *weight_i8.npy files under weights_dir/layers.
    """
    meta = load_export_metadata(weights_dir)
    files: List[Path] = []

    if meta is not None and isinstance(meta.get("layers"), list):
        for layer in meta["layers"]:
            p = Path(str(layer.get("weight_i8_npy", "")))
            if not p.is_absolute():
                # The export script writes paths relative to the current working directory.
                # First try CWD-relative, then weights_dir-relative fallback.
                if p.exists():
                    pass
                elif (weights_dir / p).exists():
                    p = weights_dir / p
                else:
                    # Common case: metadata has eval/assets/... path and user runs from repo root.
                    # If that does not exist, try just the basename inside layers/.
                    alt = weights_dir / "layers" / p.name
                    if alt.exists():
                        p = alt
            if not p.exists():
                raise FileNotFoundError(f"Weight file from metadata not found: {p}")
            files.append(p)
    else:
        layers_dir = weights_dir / "layers"
        search_dir = layers_dir if layers_dir.exists() else weights_dir
        files = sorted(search_dir.glob("*weight_i8.npy"))

    if len(files) < expected_layers:
        raise ValueError(
            f"Only found {len(files)} int8 weight files, but DSE/descriptor has {expected_layers} layers. "
            f"weights_dir={weights_dir}"
        )
    if len(files) > expected_layers:
        files = files[:expected_layers]

    return files


def load_weight_i8(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 4:
        raise ValueError(f"Weight tensor must be [F,C,K,K], got shape {arr.shape} from {path}")
    if arr.dtype != np.int8:
        if arr.min() < -128 or arr.max() > 127:
            raise ValueError(
                f"Weight tensor {path} dtype={arr.dtype}, range=[{arr.min()},{arr.max()}] cannot be cast to int8"
            )
        arr = arr.astype(np.int8)
    return np.ascontiguousarray(arr)


def lane_values_to_ddr_words(lanes: Sequence[int], ptotal: int, ddr_word_lanes: int) -> List[int]:
    """Split one PTOTAL-lane physical weight word into DDR words."""
    if len(lanes) != ptotal:
        raise ValueError(f"Physical lane vector length {len(lanes)} != PTOTAL {ptotal}")
    subwords = ceil_div(ptotal, ddr_word_lanes)
    words: List[int] = []
    for sw in range(subwords):
        word = 0
        base = sw * ddr_word_lanes
        for lane in range(ddr_word_lanes):
            src = base + lane
            b = int(lanes[src]) & 0xFF if src < ptotal else 0
            word |= b << (8 * lane)
        words.append(word)
    return words


def write_hex_words(path: Path, words: Iterable[int], ddr_word_lanes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    digits = max(2, ddr_word_lanes * 2)
    with path.open("w", encoding="utf-8") as f:
        for w in words:
            f.write(f"{int(w) & ((1 << (8*ddr_word_lanes)) - 1):0{digits}X}\n")


def get_layer_dims_from_row(row: Dict[str, str]) -> Tuple[int, int, int, int, int, int]:
    c = as_int(get_first(row, ["c_in", "c", "cin", "channels"], None), field="c_in")
    f = as_int(get_first(row, ["f_out", "f", "cout", "filters"], None), field="f_out")
    k = as_int(get_first(row, ["k", "kernel", "kernel_size"], None), field="k")
    mode = as_int(get_first(row, ["mode", "layer_mode"], None), field="mode")
    pv = as_int(get_first(row, ["pv_m1", "pv", "pv_cur"], 0), default=0, field="pv_m1")
    pf_m1 = as_int(get_first(row, ["pf_m1", "pf1", "pf_mode1"], 0), default=0, field="pf_m1")
    pc = as_int(get_first(row, ["pc_m2", "pc", "pc_mode2"], 0), default=0, field="pc_m2")
    pf_m2 = as_int(get_first(row, ["pf_m2", "pf2", "pf_mode2"], 0), default=0, field="pf_m2")
    return c, f, k, mode, (pv if mode == RTL_MODE1 else pc), (pf_m1 if mode == RTL_MODE1 else pf_m2)


def infer_ptotal(rows: Sequence[Dict[str, str]]) -> int:
    products: List[int] = []
    for i, r in enumerate(rows):
        mode = as_int(get_first(r, ["mode", "layer_mode"], None), field=f"mode row {i}")
        if mode == RTL_MODE1:
            pv = as_int(get_first(r, ["pv_m1", "pv", "pv_cur"], 0), default=0, field=f"pv row {i}")
            pf = as_int(get_first(r, ["pf_m1", "pf", "pf1", "pf_mode1"], 0), default=0, field=f"pf_m1 row {i}")
            if pv > 0 and pf > 0:
                products.append(pv * pf)
        elif mode == RTL_MODE2:
            pc = as_int(get_first(r, ["pc_m2", "pc", "pc_mode2"], 0), default=0, field=f"pc row {i}")
            pf = as_int(get_first(r, ["pf_m2", "pf", "pf2", "pf_mode2"], 0), default=0, field=f"pf_m2 row {i}")
            if pc > 0 and pf > 0:
                products.append(pc * pf)
        else:
            raise ValueError(f"DSE row {i}: mode must use RTL encoding 0=MODE1, 1=MODE2, got {mode}")
    if not products:
        raise ValueError("Cannot infer PTOTAL from DSE rows; pass --ptotal explicitly")
    uniq = sorted(set(products))
    if len(uniq) != 1:
        raise ValueError(f"DSE rows do not have a constant PTOTAL product: {uniq}. Pass/fix --ptotal.")
    return uniq[0]


def validate_weight_shape(layer_idx: int, w: np.ndarray, dse_row: Dict[str, str], desc_layer: Optional[Dict[str, Any]]) -> List[str]:
    warnings: List[str] = []
    f_w, c_w, ky_w, kx_w = map(int, w.shape)
    c_dse = as_int(get_first(dse_row, ["c_in", "c", "cin"], c_w), default=c_w, field="c_in")
    f_dse = as_int(get_first(dse_row, ["f_out", "f", "cout"], f_w), default=f_w, field="f_out")
    k_dse = as_int(get_first(dse_row, ["k", "kernel"], ky_w), default=ky_w, field="k")

    if (f_w, c_w, ky_w, kx_w) != (f_dse, c_dse, k_dse, k_dse):
        warnings.append(
            f"layer {layer_idx}: weight shape {tuple(w.shape)} != DSE shape "
            f"(F,C,K,K)=({f_dse},{c_dse},{k_dse},{k_dse})"
        )

    if desc_layer is not None:
        f_desc = int(desc_layer.get("f_out", f_dse))
        c_desc = int(desc_layer.get("c_in", c_dse))
        k_desc = int(desc_layer.get("k", k_dse))
        if (f_w, c_w, ky_w, kx_w) != (f_desc, c_desc, k_desc, k_desc):
            warnings.append(
                f"layer {layer_idx}: weight shape {tuple(w.shape)} != descriptor shape "
                f"(F,C,K,K)=({f_desc},{c_desc},{k_desc},{k_desc})"
            )
    return warnings


def pack_layer_mode1(weight_fckk: np.ndarray, pv: int, pf: int, ptotal: int, ddr_word_lanes: int) -> Tuple[List[int], Dict[str, Any]]:
    f_out, c_in, k_h, k_w = map(int, weight_fckk.shape)
    if k_h != k_w:
        raise ValueError("CNN_V4 packer expects square kernels")
    k = k_h
    if pv <= 0 or pf <= 0:
        raise ValueError(f"Mode 1 requires Pv/Pf > 0, got Pv={pv}, Pf={pf}")
    if pv * pf != ptotal:
        raise ValueError(f"Mode 1 requires Pv*Pf=PTOTAL, got {pv}*{pf} != {ptotal}")

    num_fgroup = ceil_div(f_out, pf)
    num_logical_bundles = num_fgroup * c_in * k * k
    num_physical_words = ceil_div(num_logical_bundles, pv)

    ddr_words: List[int] = []
    physical_word_count = 0

    logical_idx = 0
    for phys in range(num_physical_words):
        lanes = [0] * ptotal
        for sub_bundle in range(pv):
            li = phys * pv + sub_bundle
            if li >= num_logical_bundles:
                continue
            # Invert flat logical bundle order: (f_group, c, ky, kx)
            tmp = li
            kx = tmp % k
            tmp //= k
            ky = tmp % k
            tmp //= k
            c = tmp % c_in
            tmp //= c_in
            f_group = tmp

            base_lane = sub_bundle * pf
            for pf_i in range(pf):
                f = f_group * pf + pf_i
                lanes[base_lane + pf_i] = int(weight_fckk[f, c, ky, kx]) if f < f_out else 0

        ddr_words.extend(lane_values_to_ddr_words(lanes, ptotal=ptotal, ddr_word_lanes=ddr_word_lanes))
        physical_word_count += 1
        logical_idx += pv

    info = {
        "layout": "mode1_logical_bundle_fgroup_c_ky_kx_packed_by_pv",
        "mode": RTL_MODE1,
        "pv": pv,
        "pf": pf,
        "ptotal": ptotal,
        "f_out": f_out,
        "c_in": c_in,
        "k": k,
        "num_fgroup": num_fgroup,
        "num_logical_bundles": num_logical_bundles,
        "num_physical_words": physical_word_count,
        "num_ddr_words": len(ddr_words),
        "lane_formula": "lane = (logical_idx % Pv) * Pf + pf",
        "logical_order": "f_group, c, ky, kx",
    }
    return ddr_words, info


def pack_layer_mode2(weight_fckk: np.ndarray, pc: int, pf: int, ptotal: int, ddr_word_lanes: int) -> Tuple[List[int], Dict[str, Any]]:
    f_out, c_in, k_h, k_w = map(int, weight_fckk.shape)
    if k_h != k_w:
        raise ValueError("CNN_V4 packer expects square kernels")
    k = k_h
    if pc <= 0 or pf <= 0:
        raise ValueError(f"Mode 2 requires PC/PF > 0, got PC={pc}, PF={pf}")
    if pc * pf != ptotal:
        raise ValueError(f"Mode 2 requires PC*PF=PTOTAL, got {pc}*{pf} != {ptotal}")

    num_fgroup = ceil_div(f_out, pf)
    num_cgroup = ceil_div(c_in, pc)
    ddr_words: List[int] = []
    physical_word_count = 0

    for f_group in range(num_fgroup):
        for c_group in range(num_cgroup):
            for ky in range(k):
                for kx in range(k):
                    lanes = [0] * ptotal
                    for pf_i in range(pf):
                        f = f_group * pf + pf_i
                        for pc_i in range(pc):
                            c = c_group * pc + pc_i
                            lane = pf_i * pc + pc_i
                            lanes[lane] = int(weight_fckk[f, c, ky, kx]) if (f < f_out and c < c_in) else 0
                    ddr_words.extend(lane_values_to_ddr_words(lanes, ptotal=ptotal, ddr_word_lanes=ddr_word_lanes))
                    physical_word_count += 1

    info = {
        "layout": "mode2_physical_block_fgroup_cgroup_ky_kx_pf_pc",
        "mode": RTL_MODE2,
        "pc": pc,
        "pf": pf,
        "ptotal": ptotal,
        "f_out": f_out,
        "c_in": c_in,
        "k": k,
        "num_fgroup": num_fgroup,
        "num_cgroup": num_cgroup,
        "num_physical_words": physical_word_count,
        "num_ddr_words": len(ddr_words),
        "lane_formula": "lane = pf*PC + pc",
        "physical_order": "f_group, c_group, ky, kx",
    }
    return ddr_words, info


def write_offsets_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layer_id",
        "layer_name",
        "mode",
        "mode_name",
        "wgt_ddr_base_word",
        "offset_ddr_words",
        "offset_physical_words",
        "num_ddr_words",
        "num_physical_words",
        "ptotal",
        "ddr_word_lanes",
        "subwords_per_physical",
        "shape_f_c_k_k",
        "pv_m1",
        "pf_m1",
        "pc_m2",
        "pf_m2",
        "weight_i8_npy",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_updated_dse_csv(path: Path, original_rows: Sequence[Dict[str, str]], original_fieldnames: Sequence[str], offsets: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = [
        "wgt_ddr_base",
        "wgt_offset_ddr_words",
        "wgt_offset_physical_words",
        "wgt_num_ddr_words",
        "wgt_num_physical_words",
    ]
    fieldnames = list(original_fieldnames)
    for f in extra_fields:
        if f not in fieldnames:
            fieldnames.append(f)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, off in zip(original_rows, offsets):
            out = dict(row)
            out["wgt_ddr_base"] = str(off["wgt_ddr_base_word"])
            out["wgt_offset_ddr_words"] = str(off["offset_ddr_words"])
            out["wgt_offset_physical_words"] = str(off["offset_physical_words"])
            out["wgt_num_ddr_words"] = str(off["num_ddr_words"])
            out["wgt_num_physical_words"] = str(off["num_physical_words"])
            writer.writerow(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pack per-layer int8 weights into CNN_V4 weight DDR layout."
    )
    parser.add_argument("--weights-dir", required=True, help="Directory generated by export_torchvision_weights.py")
    parser.add_argument("--dse-csv", required=True, help="DSE CSV generated by dse_lite_v3_rtlmode.py")
    parser.add_argument("--descriptor", default=None, help="Optional descriptor JSON for shape checking")
    parser.add_argument("--out", required=True, help="Output packed CNN_V4 weight DDR hex")
    parser.add_argument("--metadata-out", default=None, help="Optional metadata JSON output")
    parser.add_argument("--offsets-csv", default=None, help="Optional per-layer offsets CSV output")
    parser.add_argument("--dse-out", default=None, help="Optional updated DSE CSV with wgt_ddr_base fields")
    parser.add_argument("--ptotal", type=int, default=None, help="Physical weight-buffer lanes. If omitted, infer from DSE products")
    parser.add_argument("--ddr-word-lanes", type=int, default=4, help="8-bit lanes per DDR word; KV260 XSCT runner default is 4")
    parser.add_argument("--wgt-base", type=lambda x: int(x, 0), default=0x08000, help="Absolute WGT DDR base word address for layer_desc_pkg fields")
    parser.add_argument("--strict-shapes", action="store_true", help="Treat descriptor/DSE/weight shape mismatch warnings as errors")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)

    weights_dir = Path(args.weights_dir)
    dse_path = Path(args.dse_csv)
    descriptor_layers = read_descriptor(Path(args.descriptor)) if args.descriptor else None
    dse_rows, dse_fieldnames = read_dse_csv(dse_path)
    n_layers = len(dse_rows)

    ptotal = int(args.ptotal) if args.ptotal is not None else infer_ptotal(dse_rows)
    if ptotal <= 0:
        raise ValueError(f"PTOTAL must be positive, got {ptotal}")
    if args.ddr_word_lanes <= 0:
        raise ValueError(f"--ddr-word-lanes must be positive, got {args.ddr_word_lanes}")

    subwords_per_physical = ceil_div(ptotal, args.ddr_word_lanes)

    weight_files = discover_weight_files(weights_dir, expected_layers=n_layers)

    out_path = Path(args.out)
    metadata_path = Path(args.metadata_out) if args.metadata_out else out_path.with_suffix(out_path.suffix + ".metadata.json")
    offsets_path = Path(args.offsets_csv) if args.offsets_csv else out_path.with_suffix(out_path.suffix + ".offsets.csv")
    dse_out_path = Path(args.dse_out) if args.dse_out else None

    for p in [out_path, metadata_path, offsets_path] + ([dse_out_path] if dse_out_path else []):
        if p is not None and p.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {p}. Use --overwrite to replace it.")

    all_ddr_words: List[int] = []
    offsets: List[Dict[str, Any]] = []
    layer_infos: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for i, (row, w_path) in enumerate(zip(dse_rows, weight_files)):
        w = load_weight_i8(w_path)
        desc_layer = descriptor_layers[i] if descriptor_layers is not None and i < len(descriptor_layers) else None
        shape_warnings = validate_weight_shape(i, w, row, desc_layer)
        warnings.extend(shape_warnings)

        mode = as_int(get_first(row, ["mode", "layer_mode"], None), field=f"mode row {i}")
        if mode not in (RTL_MODE1, RTL_MODE2):
            raise ValueError(f"row {i}: mode must use RTL encoding 0=MODE1, 1=MODE2; got {mode}")

        layer_name = str(get_first(row, ["layer_name", "name"], f"layer{i:02d}"))
        layer_id = as_int(get_first(row, ["layer_id", "id"], i), default=i, field=f"layer_id row {i}")

        offset_ddr_words = len(all_ddr_words)
        offset_physical_words = offset_ddr_words // subwords_per_physical

        if mode == RTL_MODE1:
            pv = as_int(get_first(row, ["pv_m1", "pv", "pv_cur"], None), field=f"pv_m1 row {i}")
            pf = as_int(get_first(row, ["pf_m1", "pf", "pf1", "pf_mode1"], None), field=f"pf_m1 row {i}")
            layer_words, info = pack_layer_mode1(w, pv=pv, pf=pf, ptotal=ptotal, ddr_word_lanes=args.ddr_word_lanes)
            mode_name = "MODE1"
            pc = 0
            pf_m2 = 0
        else:
            pc = as_int(get_first(row, ["pc_m2", "pc", "pc_mode2"], None), field=f"pc_m2 row {i}")
            pf_m2 = as_int(get_first(row, ["pf_m2", "pf", "pf2", "pf_mode2"], None), field=f"pf_m2 row {i}")
            layer_words, info = pack_layer_mode2(w, pc=pc, pf=pf_m2, ptotal=ptotal, ddr_word_lanes=args.ddr_word_lanes)
            mode_name = "MODE2"
            pv = 0
            pf = 0

        all_ddr_words.extend(layer_words)

        off = {
            "layer_id": layer_id,
            "layer_name": layer_name,
            "mode": mode,
            "mode_name": mode_name,
            "wgt_ddr_base_word": args.wgt_base + offset_ddr_words,
            "offset_ddr_words": offset_ddr_words,
            "offset_physical_words": offset_physical_words,
            "num_ddr_words": len(layer_words),
            "num_physical_words": info["num_physical_words"],
            "ptotal": ptotal,
            "ddr_word_lanes": args.ddr_word_lanes,
            "subwords_per_physical": subwords_per_physical,
            "shape_f_c_k_k": "x".join(str(x) for x in map(int, w.shape)),
            "pv_m1": pv,
            "pf_m1": pf,
            "pc_m2": pc,
            "pf_m2": pf_m2,
            "weight_i8_npy": str(w_path),
        }
        offsets.append(off)
        layer_infos.append({"offset": off, "layout": info})

        if not args.quiet:
            print(
                f"[OK] layer {i:02d} {layer_name}: {mode_name} shape={tuple(w.shape)} "
                f"base=0x{off['wgt_ddr_base_word']:X} ddr_words={len(layer_words)} phys_words={info['num_physical_words']}"
            )

    if warnings:
        print("\nShape/layout warnings:", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)
        if args.strict_shapes:
            print("--strict-shapes enabled; aborting.", file=sys.stderr)
            return 2

    write_hex_words(out_path, all_ddr_words, ddr_word_lanes=args.ddr_word_lanes)
    write_offsets_csv(offsets_path, offsets)
    if dse_out_path is not None:
        write_updated_dse_csv(dse_out_path, dse_rows, dse_fieldnames, offsets)

    metadata = {
        "script": "pack_weights_for_cnn_v4.py",
        "purpose": "Pack int8 Conv2d weights into CNN_V4 weight DDR layout",
        "mode_encoding": "RTL: 0=MODE1, 1=MODE2",
        "weights_dir": str(weights_dir),
        "dse_csv": str(dse_path),
        "descriptor": str(args.descriptor) if args.descriptor else None,
        "output_hex": str(out_path),
        "offsets_csv": str(offsets_path),
        "dse_out": str(dse_out_path) if dse_out_path else None,
        "ptotal": ptotal,
        "ddr_word_lanes": args.ddr_word_lanes,
        "subwords_per_physical": subwords_per_physical,
        "wgt_base_word": args.wgt_base,
        "total_ddr_words": len(all_ddr_words),
        "total_physical_words": len(all_ddr_words) // subwords_per_physical,
        "word_pack": {
            "data_width_bits": 8,
            "endianness": "little-endian lanes: lane0 -> bits[7:0]",
            "ddr_word_bits": args.ddr_word_lanes * 8,
            "physical_weight_word_lanes": ptotal,
            "physical_weight_word_bits": ptotal * 8,
        },
        "layout_summary": {
            "mode1": {
                "logical_order": "f_group, c, ky, kx",
                "logical_bundle_lanes": "pf_i -> W[f_group*Pf + pf_i][c][ky][kx]",
                "physical_lane_formula": "lane = (logical_idx % Pv) * Pf + pf_i",
            },
            "mode2": {
                "physical_order": "f_group, c_group, ky, kx",
                "physical_lane_formula": "lane = pf_i*PC + pc_i",
            },
        },
        "warnings": warnings,
        "layers": layer_infos,
        "notes": [
            "Use --dse-out and then generate cnn_layer_desc_pkg.sv from that updated CSV so wgt_ddr_base matches this packed file.",
            "run_kv260_eval.tcl writes this hex stream at the WGT DDR base address; layer descriptor bases should be absolute word addresses inside CNN_V4 DDR map.",
            "This script does not pack bias. Current CNN_V4 evaluation flow treats Conv/ReLU/Pooling only.",
        ],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if not args.quiet:
        print("\nDone.")
        print(f"  output hex          : {out_path}")
        print(f"  metadata            : {metadata_path}")
        print(f"  offsets csv         : {offsets_path}")
        if dse_out_path:
            print(f"  updated DSE csv     : {dse_out_path}")
        print(f"  total DDR words     : {len(all_ddr_words)}")
        print(f"  total physical words: {len(all_ddr_words) // subwords_per_physical}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
