#!/usr/bin/env python3
"""
pack_ifm_for_cnn_v4.py
-----------------------
Pack a logical input feature map tensor into the DDR IFM layout expected by the
current CNN_V4 DMA/IFM preload path.

This script is for the *initial input feature map* loaded from DDR into IFM
buffer before the first layer. Inter-layer OFM->IFM handoff is handled by RTL
and uses a different path.

Supported CNN_V4 IFM DDR layouts
--------------------------------
RTL mode convention:
  mode = 0 -> Mode 1, pixel/filter parallelism
  mode = 1 -> Mode 2, channel/filter parallelism

Mode 1 DDR layout:
  [channel][row][col_group]

  col_group = floor(x / Pv_cur)
  lane      = x % Pv_cur

  One DDR word contains up to Pv_cur pixel lanes for one channel and one row.
  Lanes beyond Pv_cur are zero-padded. Tail pixels at the end of a row are
  zero-padded.

Mode 2 DDR layout:
  [tile_x][row][cgrp][col_l]

  tile_x = floor(x / PC)
  col_l  = x % PC
  cgrp   = floor(channel / PC)
  lane   = channel % PC

  One DDR word contains up to PC channel lanes for one local column inside the
  horizontal tile. Channel tails and tile-width tails are zero-padded.

Word packing
------------
Each 8-bit lane is packed little-endian into the output hex word:
  lane 0 -> bits [7:0]
  lane 1 -> bits [15:8]
  lane 2 -> bits [23:16]
  lane 3 -> bits [31:24]
  ...

For the current KV260 32-bit runner, use --word-lanes 4. If you use a wider RTL
DDR word in simulation, you may set --word-lanes accordingly, but run_kv260_eval.tcl
currently writes 32-bit words only.

Examples
--------
Pack VGG16 layer-0 input using DSE CSV:

  python eval/scripts/pack_ifm_for_cnn_v4.py \
    --input eval/assets/vgg16_conv13/inputs/img0000/input_uint8_hwc.npy \
    --descriptor eval/models/vgg16_conv13.json \
    --dse-csv eval/results/vgg16_conv13/dse/vgg16_dse.csv \
    --out eval/assets/vgg16_conv13/inputs/img0000/ifm_ddr_cnnv4.hex \
    --word-lanes 4

Manual Mode 1 example:

  python eval/scripts/pack_ifm_for_cnn_v4.py \
    --input input_uint8_hwc.npy --out ifm.hex --mode 0 --pv 4 --word-lanes 4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


RTL_MODE1 = 0
RTL_MODE2 = 1


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError(f"ceil_div divisor must be positive, got {b}")
    return (a + b - 1) // b


def as_int_or_none(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"none", "null", "nan"}:
            return None
        return int(float(s))
    return int(x)


def read_descriptor_layer(path: Path, layer_index: int) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        model = json.load(f)
    layers = model.get("layers")
    if not isinstance(layers, list):
        raise ValueError(f"Descriptor {path} has no 'layers' list")
    if layer_index < 0 or layer_index >= len(layers):
        raise IndexError(f"Layer index {layer_index} out of range for {path}")
    return layers[layer_index]


def read_dse_row(path: Path, layer_index: int) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"DSE CSV {path} has no rows")

    # Prefer matching by row order; this is what dse_lite_v3_rtlmode.py emits.
    if 0 <= layer_index < len(rows):
        return rows[layer_index]

    # Fallback: match layer_id field.
    for row in rows:
        if as_int_or_none(row.get("layer_id")) == layer_index:
            return row

    raise IndexError(f"Layer index {layer_index} not found in DSE CSV {path}")


def load_ifm(path: Path, layout: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"Input tensor must be 3-D, got shape {arr.shape}")

    if layout.upper() == "HWC":
        hwc = arr
    elif layout.upper() == "CHW":
        hwc = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported input layout: {layout}; expected HWC or CHW")

    if hwc.dtype != np.uint8:
        # Be strict-ish: values must be valid u8, then cast.
        if hwc.min() < 0 or hwc.max() > 255:
            raise ValueError(
                f"Input dtype is {hwc.dtype} with range [{hwc.min()}, {hwc.max()}], cannot cast to uint8 safely"
            )
        hwc = hwc.astype(np.uint8)

    return np.ascontiguousarray(hwc)


def lane_values_to_word(lanes: List[int], word_lanes: int) -> int:
    if len(lanes) > word_lanes:
        raise ValueError(f"Too many lanes: {len(lanes)} > word_lanes {word_lanes}")
    word = 0
    for i in range(word_lanes):
        b = int(lanes[i]) & 0xFF if i < len(lanes) else 0
        word |= b << (8 * i)
    return word


def write_hex_words(path: Path, words: List[int], word_lanes: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hex_digits = max(2, word_lanes * 2)
    with path.open("w", encoding="utf-8") as f:
        for w in words:
            f.write(f"{w:0{hex_digits}X}\n")


def pack_mode1(ifm_hwc: np.ndarray, pv_cur: int, word_lanes: int) -> Tuple[List[int], Dict[str, Any]]:
    h, w, c = map(int, ifm_hwc.shape)
    if pv_cur <= 0:
        raise ValueError("Mode 1 requires pv_cur > 0")
    if pv_cur > word_lanes:
        raise ValueError(
            f"Mode 1 pv_cur={pv_cur} exceeds word_lanes={word_lanes}. "
            "word_lanes should normally equal RTL PV_MAX."
        )

    groups = ceil_div(w, pv_cur)
    words: List[int] = []

    # RTL mode-1 DDR layout: [channel][row][col_group].
    for ch in range(c):
        for yy in range(h):
            for g in range(groups):
                lanes: List[int] = []
                base_x = g * pv_cur
                for lane in range(pv_cur):
                    xx = base_x + lane
                    val = int(ifm_hwc[yy, xx, ch]) if xx < w else 0
                    lanes.append(val)
                # Remaining lanes beyond pv_cur are padded to zero in lane_values_to_word.
                words.append(lane_values_to_word(lanes, word_lanes))

    info = {
        "layout": "mode1_channel_row_colgroup",
        "mode": RTL_MODE1,
        "pv_cur": pv_cur,
        "word_lanes": word_lanes,
        "h": h,
        "w": w,
        "c": c,
        "col_groups": groups,
        "num_words": len(words),
        "formula": "word_index = ch * H * ceil(W/Pv) + row * ceil(W/Pv) + col_group",
    }
    return words, info


def pack_mode2(ifm_hwc: np.ndarray, pc: int, word_lanes: int) -> Tuple[List[int], Dict[str, Any]]:
    h, w, c = map(int, ifm_hwc.shape)
    if pc <= 0:
        raise ValueError("Mode 2 requires pc > 0")
    if pc > word_lanes:
        raise ValueError(
            f"Mode 2 pc={pc} exceeds word_lanes={word_lanes}. "
            "word_lanes should normally equal RTL PV_MAX and be >= PC."
        )

    num_tiles = ceil_div(w, pc)
    cgroups = ceil_div(c, pc)
    words: List[int] = []

    # Full IFM memory layout expected by addr_gen_ifm_ddr for mode 2:
    # [tile_x][row][cgrp][col_l].
    for tile_x in range(num_tiles):
        for yy in range(h):
            for cg in range(cgroups):
                for col_l in range(pc):
                    xx = tile_x * pc + col_l
                    lanes: List[int] = []
                    base_ch = cg * pc
                    for lane in range(pc):
                        ch = base_ch + lane
                        if xx < w and ch < c:
                            val = int(ifm_hwc[yy, xx, ch])
                        else:
                            val = 0
                        lanes.append(val)
                    # Remaining lanes beyond pc are padded to zero in lane_values_to_word.
                    words.append(lane_values_to_word(lanes, word_lanes))

    tile_words = h * cgroups * pc
    info = {
        "layout": "mode2_tile_row_cgrp_coll",
        "mode": RTL_MODE2,
        "pc": pc,
        "word_lanes": word_lanes,
        "h": h,
        "w": w,
        "c": c,
        "num_tiles": num_tiles,
        "cgroups": cgroups,
        "tile_words": tile_words,
        "num_words": len(words),
        "formula": "word_index = tile_x * (H*ceil(C/PC)*PC) + row * (ceil(C/PC)*PC) + cgrp * PC + col_l",
    }
    return words, info


def infer_config(args: argparse.Namespace, ifm_hwc: np.ndarray) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}

    # Shape from input is source of truth, descriptor is a guard.
    h, w, c = map(int, ifm_hwc.shape)
    cfg.update({"h": h, "w": w, "c": c})

    if args.descriptor:
        layer = read_descriptor_layer(Path(args.descriptor), args.layer_index)
        desc_h = as_int_or_none(layer.get("h_in"))
        desc_w = as_int_or_none(layer.get("w_in"))
        desc_c = as_int_or_none(layer.get("c_in"))
        if (desc_h, desc_w, desc_c) != (h, w, c):
            msg = (
                f"Input tensor shape HWC={(h, w, c)} does not match descriptor layer "
                f"{args.layer_index} h_in/w_in/c_in={(desc_h, desc_w, desc_c)}"
            )
            if args.allow_shape_mismatch:
                cfg.setdefault("warnings", []).append(msg)
            else:
                raise ValueError(msg)
        cfg["descriptor_layer_name"] = str(layer.get("name", f"layer{args.layer_index}"))

    dse_row: Optional[Dict[str, str]] = None
    if args.dse_csv:
        dse_row = read_dse_row(Path(args.dse_csv), args.layer_index)
        cfg["dse_layer_name"] = dse_row.get("layer_name")

    mode = args.mode
    if mode is None and dse_row is not None:
        mode = as_int_or_none(dse_row.get("mode"))
    if mode is None:
        raise ValueError("Cannot infer mode. Provide --mode 0|1 or --dse-csv.")
    if mode not in (RTL_MODE1, RTL_MODE2):
        raise ValueError(f"Mode must use RTL encoding 0=MODE1, 1=MODE2. Got {mode}")
    cfg["mode"] = mode
    cfg["mode_name"] = "MODE1" if mode == RTL_MODE1 else "MODE2"

    if mode == RTL_MODE1:
        pv = args.pv
        if pv is None and dse_row is not None:
            pv = as_int_or_none(dse_row.get("pv_m1"))
        if pv is None or pv <= 0:
            raise ValueError("Mode 1 requires --pv or a DSE CSV row with pv_m1 > 0")
        cfg["pv_cur"] = int(pv)
    else:
        pc = args.pc
        if pc is None and dse_row is not None:
            pc = as_int_or_none(dse_row.get("pc_m2"))
        if pc is None or pc <= 0:
            raise ValueError("Mode 2 requires --pc or a DSE CSV row with pc_m2 > 0")
        cfg["pc"] = int(pc)

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack logical HWC/CHW IFM tensor into CNN_V4 IFM DDR layout."
    )

    parser.add_argument("--input", required=True, help="Input IFM .npy tensor, normally uint8 HWC")
    parser.add_argument("--out", required=True, help="Output IFM DDR hex file")
    parser.add_argument("--metadata-out", default=None, help="Optional metadata JSON path")

    parser.add_argument("--descriptor", default=None, help="Benchmark descriptor JSON, used for shape checking")
    parser.add_argument("--dse-csv", default=None, help="DSE CSV generated by dse_lite_v3_rtlmode.py")
    parser.add_argument("--layer-index", type=int, default=0, help="Layer index to use, default 0")

    parser.add_argument("--mode", type=int, choices=[0, 1], default=None, help="RTL mode override: 0=MODE1, 1=MODE2")
    parser.add_argument("--pv", type=int, default=None, help="Runtime Pv for MODE1")
    parser.add_argument("--pc", type=int, default=None, help="Runtime PC for MODE2")
    parser.add_argument("--word-lanes", type=int, default=4, help="8-bit lanes per DDR word. KV260 runner default is 4")
    parser.add_argument("--input-layout", choices=["HWC", "CHW"], default="HWC")

    parser.add_argument("--allow-shape-mismatch", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.out)
    metadata_path = Path(args.metadata_out) if args.metadata_out else out_path.with_suffix(out_path.suffix + ".metadata.json")

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {out_path}. Use --overwrite to replace it.")
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"Metadata file already exists: {metadata_path}. Use --overwrite to replace it.")

    ifm_hwc = load_ifm(input_path, args.input_layout)
    cfg = infer_config(args, ifm_hwc)

    if cfg["mode"] == RTL_MODE1:
        words, layout_info = pack_mode1(ifm_hwc, pv_cur=int(cfg["pv_cur"]), word_lanes=args.word_lanes)
    else:
        words, layout_info = pack_mode2(ifm_hwc, pc=int(cfg["pc"]), word_lanes=args.word_lanes)

    write_hex_words(out_path, words, word_lanes=args.word_lanes)

    metadata: Dict[str, Any] = {
        "script": "pack_ifm_for_cnn_v4.py",
        "purpose": "Pack initial IFM tensor into CNN_V4 DDR layout for IFM preload",
        "input_npy": str(input_path),
        "output_hex": str(out_path),
        "input_layout": args.input_layout,
        "input_shape_hwc": list(map(int, ifm_hwc.shape)),
        "input_dtype": str(ifm_hwc.dtype),
        "mode_encoding": "RTL: 0=MODE1, 1=MODE2",
        "config": cfg,
        "layout": layout_info,
        "word_pack": {
            "data_width_bits": 8,
            "word_lanes": args.word_lanes,
            "word_bits": args.word_lanes * 8,
            "endianness": "little-endian lanes: lane0 -> bits[7:0]",
            "hex_digits_per_line": args.word_lanes * 2,
        },
        "notes": [
            "This packs only the external DDR initial IFM for layer 0 / selected layer.",
            "Inter-layer OFM->IFM refill/handoff is handled by RTL, not by this script.",
            "For run_kv260_eval.tcl, keep word_lanes=4 unless your runner supports wider words.",
        ],
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if not args.quiet:
        print("[OK] Packed IFM for CNN_V4")
        print(f"  input          : {input_path}")
        print(f"  shape HWC      : {tuple(ifm_hwc.shape)}")
        print(f"  mode           : {cfg['mode']} ({cfg['mode_name']})")
        if cfg["mode"] == RTL_MODE1:
            print(f"  Pv             : {cfg['pv_cur']}")
            print(f"  col groups     : {layout_info['col_groups']}")
        else:
            print(f"  PC             : {cfg['pc']}")
            print(f"  tiles          : {layout_info['num_tiles']}")
            print(f"  cgroups        : {layout_info['cgroups']}")
        print(f"  word lanes     : {args.word_lanes}")
        print(f"  output words   : {len(words)}")
        print(f"  output hex     : {out_path}")
        print(f"  metadata       : {metadata_path}")


if __name__ == "__main__":
    main()
