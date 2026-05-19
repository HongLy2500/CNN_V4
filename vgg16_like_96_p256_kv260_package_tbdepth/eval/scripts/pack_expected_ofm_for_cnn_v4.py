#!/usr/bin/env python3
"""
pack_expected_ofm_for_cnn_v4.py

Pack a golden OFM tensor produced by gen_fixedpoint_golden.py into the DDR
readback order used by CNN_V4's ofm_buffer -> cnn_dma_direct OFM DMA path.

The logical golden tensor is expected in HWC order:
    ofm[h][w][f]   dtype uint8 / integer-compatible

The packed DDR/readback order follows ofm_buffer's DMA linear readback:
    for channel in 0..F-1:
      for row in 0..H-1:
        for group in 0..ceil(W/store_pack)-1:
          word lanes contain consecutive spatial columns inside that channel

Each output line is one physical DDR word in hexadecimal. By default the word
contains 4 lanes x 8-bit = 32-bit, matching the current KV260 runner.

RTL mode convention:
    mode = 0 -> MODE1
    mode = 1 -> MODE2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


MODE1 = 0
MODE2 = 1


@dataclass
class LayerCfg:
    layer_id: int
    layer_name: str
    mode: int
    h_conv: Optional[int]
    w_conv: Optional[int]
    f_out: Optional[int]
    pool_enable: bool
    pv_m1: int
    pf_m1: int
    pc_m2: int
    pf_m2: int


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError(f"ceil_div denominator must be > 0, got {b}")
    return (a + b - 1) // b


def parse_int(value: Any, default: Optional[int] = None) -> int:
    if value is None or value == "":
        if default is None:
            raise ValueError("missing integer value")
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    s = str(value).strip()
    if not s:
        if default is None:
            raise ValueError("empty integer value")
        return default
    return int(s, 0)


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def first_present(mapping: Dict[str, Any], names: List[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] not in (None, ""):
            return mapping[name]
    return default


def parse_mode(value: Any) -> int:
    if value is None or value == "":
        raise ValueError("missing mode")
    if isinstance(value, str):
        s = value.strip().upper()
        if s in {"MODE1", "M1", "0"}:
            return MODE1
        if s in {"MODE2", "M2", "1"}:
            return MODE2
        return int(s, 0)
    v = int(value)
    if v not in (MODE1, MODE2):
        raise ValueError(f"mode must be 0/1 for RTL convention, got {value}")
    return v


def sanitize_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return name or "layer"


def read_dse_csv(path: Path) -> List[LayerCfg]:
    rows: List[LayerCfg] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            layer_id = parse_int(first_present(row, ["layer_id", "id", "idx"], i), default=i)
            layer_name = str(first_present(row, ["layer_name", "name"], f"layer{layer_id}")).strip()
            mode = parse_mode(first_present(row, ["mode", "layer_mode", "mode_name"], None))

            h_conv = first_present(row, ["h_out", "ho", "conv_h_out", "output_h", "hout"], None)
            w_conv = first_present(row, ["w_out", "wo", "conv_w_out", "output_w", "wout"], None)
            f_out = first_present(row, ["f_out", "f", "filters", "cout", "out_channels"], None)

            rows.append(
                LayerCfg(
                    layer_id=layer_id,
                    layer_name=layer_name,
                    mode=mode,
                    h_conv=parse_int(h_conv, default=0) if h_conv not in (None, "") else None,
                    w_conv=parse_int(w_conv, default=0) if w_conv not in (None, "") else None,
                    f_out=parse_int(f_out, default=0) if f_out not in (None, "") else None,
                    pool_enable=parse_bool(first_present(row, ["pool_enable", "pool_en", "pool", "maxpool"], False)),
                    pv_m1=parse_int(first_present(row, ["pv_m1", "pv", "pv_cur"], 0), default=0),
                    pf_m1=parse_int(first_present(row, ["pf_m1", "pf1", "pf_mode1"], 0), default=0),
                    pc_m2=parse_int(first_present(row, ["pc_m2", "pc", "pc_mode2"], 0), default=0),
                    pf_m2=parse_int(first_present(row, ["pf_m2", "pf2", "pf_mode2"], 0), default=0),
                )
            )
    if not rows:
        raise RuntimeError(f"No layers found in DSE CSV: {path}")
    return rows


def read_descriptor_layers(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as f:
        model = json.load(f)
    layers = model.get("layers")
    if not isinstance(layers, list) or not layers:
        raise RuntimeError(f"Descriptor has no non-empty 'layers': {path}")
    return layers


def layer_cfg_from_descriptor(layer: Dict[str, Any], idx: int) -> LayerCfg:
    pool = layer.get("pool") or {}
    dse = layer.get("dse") or {}
    return LayerCfg(
        layer_id=parse_int(layer.get("id", idx), default=idx),
        layer_name=str(layer.get("name", f"layer{idx}")),
        mode=parse_mode(dse.get("mode", layer.get("mode", 0))),
        h_conv=parse_int(layer.get("conv_h_out", layer.get("h_out", 0)), default=0),
        w_conv=parse_int(layer.get("conv_w_out", layer.get("w_out", 0)), default=0),
        f_out=parse_int(layer.get("f_out", 0), default=0),
        pool_enable=parse_bool(pool.get("enable", layer.get("pool_enable", False))),
        pv_m1=parse_int(dse.get("pv_m1", layer.get("pv_m1", 0)), default=0),
        pf_m1=parse_int(dse.get("pf_m1", layer.get("pf_m1", 0)), default=0),
        pc_m2=parse_int(dse.get("pc_m2", layer.get("pc_m2", 0)), default=0),
        pf_m2=parse_int(dse.get("pf_m2", layer.get("pf_m2", 0)), default=0),
    )


def select_layer(rows: List[LayerCfg], layer_index: Optional[int], layer_id: Optional[int]) -> Tuple[int, LayerCfg]:
    if layer_id is not None:
        for i, row in enumerate(rows):
            if row.layer_id == layer_id:
                return i, row
        raise RuntimeError(f"layer_id {layer_id} not found")
    if layer_index is None:
        layer_index = len(rows) - 1
    if layer_index < 0:
        layer_index = len(rows) + layer_index
    if layer_index < 0 or layer_index >= len(rows):
        raise RuntimeError(f"layer_index {layer_index} is outside 0..{len(rows)-1}")
    return layer_index, rows[layer_index]


def compute_ofm_store_params(
    cur: LayerCfg,
    next_cfg: Optional[LayerCfg],
    pc_mode2: int,
    explicit_store_pack: Optional[int] = None,
    final_layer_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Mirror the ofm_buffer storage-pack selection driven by control_unit_top:
      src_mode  = current layer mode
      next_mode = next layer mode if present else current layer mode
      pv_next   = next layer pv_m1 if present else 1

    ofm_buffer store_pack:
      M1 -> M1: pv_next
      M1 -> M2: pool ? max(pv_cur/2,1) : 1
      M2 -> M2: PC
    """
    if explicit_store_pack is not None:
        if explicit_store_pack <= 0:
            raise ValueError("--store-pack must be > 0")
        return {
            "src_mode": cur.mode,
            "next_mode": None,
            "pv_cur": cur.pv_m1,
            "pv_next": None,
            "pc": pc_mode2,
            "pool_enable": cur.pool_enable,
            "store_pack": explicit_store_pack,
            "reason": "explicit --store-pack",
        }

    src_mode = cur.mode
    if next_cfg is not None:
        next_mode = next_cfg.mode
        pv_next = next_cfg.pv_m1 if next_cfg.pv_m1 > 0 else 1
    else:
        # control_unit_top final-layer fallback: next_mode=current mode, pv_next=1.
        next_mode = cur.mode if final_layer_fallback else MODE1
        pv_next = 1

    pc = cur.pc_m2 if cur.pc_m2 > 0 else pc_mode2
    if pc <= 0:
        pc = pc_mode2

    if src_mode == MODE1 and next_mode == MODE1:
        store_pack = pv_next if pv_next > 0 else 1
        reason = "M1->M1 store: next layer Pv; final layer fallback uses Pv_next=1"
    elif src_mode == MODE1 and next_mode == MODE2:
        if cur.pool_enable:
            store_pack = max(cur.pv_m1 // 2, 1)
        else:
            store_pack = 1
        reason = "M1->M2 transition store: source layout, pooled Pv/2 or no-pool 1"
    elif src_mode == MODE2 and next_mode == MODE2:
        store_pack = pc
        reason = "M2->M2 store: PC lanes"
    else:
        raise RuntimeError("Unsupported OFM storage transition M2->M1 in current RTL")

    if store_pack <= 0:
        raise RuntimeError(f"Computed invalid store_pack={store_pack}")

    return {
        "src_mode": src_mode,
        "next_mode": next_mode,
        "pv_cur": cur.pv_m1,
        "pv_next": pv_next,
        "pc": pc,
        "pool_enable": cur.pool_enable,
        "store_pack": store_pack,
        "reason": reason,
    }


def pack_word_u8_lanes(lanes: List[int], word_lanes: int) -> int:
    if len(lanes) != word_lanes:
        raise ValueError("lanes length must equal word_lanes")
    word = 0
    for i, v in enumerate(lanes):
        word |= (int(v) & 0xFF) << (8 * i)
    return word


def write_u32_hex(path: Path, words: List[int], bits: int) -> None:
    if bits <= 0 or bits % 4 != 0:
        raise ValueError("word bits must be positive and multiple of 4")
    width = bits // 4
    mask = (1 << bits) - 1 if bits < 4096 else None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for w in words:
            if mask is not None:
                w &= mask
            f.write(f"{w:0{width}X}\n")


def pack_expected_ofm_hwc(
    ofm_hwc: np.ndarray,
    store_pack: int,
    word_lanes: int,
) -> Tuple[List[int], Dict[str, Any]]:
    if ofm_hwc.ndim != 3:
        raise RuntimeError(f"Expected HWC tensor with ndim=3, got shape={ofm_hwc.shape}")
    h, w, f = [int(x) for x in ofm_hwc.shape]
    if store_pack <= 0:
        raise ValueError("store_pack must be > 0")
    if word_lanes <= 0:
        raise ValueError("word_lanes must be > 0")
    if store_pack > word_lanes:
        raise RuntimeError(
            f"store_pack={store_pack} exceeds physical word_lanes={word_lanes}. "
            "Use --word-lanes equal to PV_MAX/DDR lanes for this build."
        )

    groups = ceil_div(w, store_pack)
    words: List[int] = []

    # ofm_buffer DMA readback order: ch -> row -> compact stored group.
    for ch in range(f):
        for row in range(h):
            for grp in range(groups):
                lanes = [0] * word_lanes
                col_base = grp * store_pack
                for lane in range(store_pack):
                    col = col_base + lane
                    if col < w:
                        lanes[lane] = int(ofm_hwc[row, col, ch]) & 0xFF
                words.append(pack_word_u8_lanes(lanes, word_lanes))

    info = {
        "h": h,
        "w": w,
        "f": f,
        "store_pack": store_pack,
        "word_lanes": word_lanes,
        "groups_per_row": groups,
        "num_words": len(words),
        "expected_layer_num_words": f * h * groups,
        "layout": "OFM DMA linear readback: [channel][row][group], word lanes are consecutive spatial columns",
    }
    return words, info


def load_ofm(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3:
        raise RuntimeError(f"Expected HWC OFM npy, got shape={arr.shape} from {path}")
    # Convert to uint8 exactly as DDR stores bytes.
    if arr.dtype != np.uint8:
        arr = np.asarray(arr, dtype=np.int64)
        arr = np.bitwise_and(arr, 0xFF).astype(np.uint8)
    return np.ascontiguousarray(arr)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pack golden final OFM tensor to CNN_V4 OFM DDR/readback layout."
    )
    p.add_argument("--ofm", required=True, help="Input golden OFM .npy in HWC uint8 layout")
    p.add_argument("--out", required=True, help="Output expected OFM hex in CNN_V4 OFM readback order")

    src = p.add_mutually_exclusive_group()
    src.add_argument("--dse-csv", help="DSE CSV with RTL mode convention 0=MODE1, 1=MODE2")
    src.add_argument("--descriptor", help="Descriptor JSON with layers; requires dse fields if no --dse-csv")

    p.add_argument("--layer-index", type=int, default=None, help="Layer index to pack; default last layer")
    p.add_argument("--layer-id", type=int, default=None, help="Layer id to pack instead of index")
    p.add_argument("--pc-mode2", type=int, default=4, help="Physical PC for mode2 if CSV row pc_m2 is 0; default 4")
    p.add_argument("--word-lanes", type=int, default=4, help="Physical OFM DDR word lanes = PV_MAX; default 4 for 32-bit KV260 runner")
    p.add_argument("--store-pack", type=int, default=None, help="Override computed OFM store_pack")
    p.add_argument("--metadata", default=None, help="Optional metadata JSON path; default <out>.metadata.json")
    p.add_argument("--word-count-out", default=None, help="Optional text file containing number of OFM words")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output")
    return p


def main() -> int:
    args = build_argparser().parse_args()

    ofm_path = Path(args.ofm)
    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        raise RuntimeError(f"Output exists: {out_path}. Use --overwrite to replace it.")

    ofm = load_ofm(ofm_path)

    rows: List[LayerCfg]
    src_path: Optional[str] = None
    if args.dse_csv:
        src_path = args.dse_csv
        rows = read_dse_csv(Path(args.dse_csv))
    elif args.descriptor:
        src_path = args.descriptor
        desc_layers = read_descriptor_layers(Path(args.descriptor))
        rows = [layer_cfg_from_descriptor(layer, i) for i, layer in enumerate(desc_layers)]
    else:
        # Minimal mode-free fallback: final layer M1 output, one pixel per word.
        h, w, f = [int(x) for x in ofm.shape]
        rows = [LayerCfg(0, "manual_final", MODE1, h, w, f, False, 1, 1, args.pc_mode2, 1)]

    layer_index, cur = select_layer(rows, args.layer_index, args.layer_id)
    next_cfg = rows[layer_index + 1] if layer_index + 1 < len(rows) else None

    store_info = compute_ofm_store_params(
        cur=cur,
        next_cfg=next_cfg,
        pc_mode2=args.pc_mode2,
        explicit_store_pack=args.store_pack,
    )

    h, w, f = [int(x) for x in ofm.shape]
    if cur.f_out is not None and cur.f_out not in (0, f):
        raise RuntimeError(f"OFM channel mismatch: tensor F={f}, layer f_out={cur.f_out}")

    words, pack_info = pack_expected_ofm_hwc(
        ofm_hwc=ofm,
        store_pack=int(store_info["store_pack"]),
        word_lanes=args.word_lanes,
    )

    write_u32_hex(out_path, words, bits=args.word_lanes * 8)

    meta_path = Path(args.metadata) if args.metadata else out_path.with_suffix(out_path.suffix + ".metadata.json")
    metadata = {
        "tool": "pack_expected_ofm_for_cnn_v4.py",
        "source_ofm": str(ofm_path),
        "source_layer_config": src_path,
        "output_hex": str(out_path),
        "layer": {
            "selected_index": layer_index,
            "layer_id": cur.layer_id,
            "layer_name": cur.layer_name,
            "mode": cur.mode,
            "mode_name": "MODE1" if cur.mode == MODE1 else "MODE2",
            "pool_enable": cur.pool_enable,
            "pv_m1": cur.pv_m1,
            "pf_m1": cur.pf_m1,
            "pc_m2": cur.pc_m2,
            "pf_m2": cur.pf_m2,
            "next_layer_mode": None if next_cfg is None else next_cfg.mode,
            "next_layer_name": None if next_cfg is None else next_cfg.layer_name,
        },
        "ofm_tensor_hwc": [h, w, f],
        "store_selection": store_info,
        "pack": pack_info,
        "notes": [
            "Expected hex matches ofm_buffer DMA linear readback, not simple HWC flatten.",
            "Invalid/tail lanes are zero-filled; OFM DDR should be cleared before hardware run.",
            "For final MODE1 layer, control_unit_top uses next_mode=current and pv_next=1 fallback, so store_pack=1 unless overridden.",
        ],
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w") as meta_f:
        json.dump(metadata, meta_f, indent=2)

    if args.word_count_out:
        wc_path = Path(args.word_count_out)
        wc_path.parent.mkdir(parents=True, exist_ok=True)
        wc_path.write_text(str(pack_info["num_words"]) + "\n")

    print("[OK] Packed expected OFM for CNN_V4")
    print(f"  input OFM      : {ofm_path}")
    print(f"  output hex     : {out_path}")
    print(f"  layer          : idx={layer_index} id={cur.layer_id} name={cur.layer_name}")
    print(f"  OFM shape HWC  : {h} x {w} x {f}")
    print(f"  store_pack     : {store_info['store_pack']} ({store_info['reason']})")
    print(f"  word_lanes     : {args.word_lanes}")
    print(f"  words          : {pack_info['num_words']}")
    print(f"  metadata       : {meta_path}")
    if args.word_count_out:
        print(f"  word count file: {args.word_count_out}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
