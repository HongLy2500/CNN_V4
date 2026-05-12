#!/usr/bin/env python3
"""
Convert a DSE result CSV file into cnn_layer_desc_pkg.sv.

This script is designed for the CNN_V4 flow:
  descriptor JSON -> dse_lite_v3_rtlmode.py -> *_dse.csv -> cnn_layer_desc_pkg.sv

It accepts both common DSE CSV variants:
  1) mode,pv,pf,pc
  2) mode,pv_m1,pf_m1,pc_m2,pf_m2

The generated package keeps the same layer_desc_t field names used by the
current CNN_V4 package and adds:
  - localparam int NUM_LAYERS
  - function automatic layer_desc_t get_layer_desc(input int unsigned idx)

In your testbench or config loader, call get_layer_desc(i) and write the
returned struct into layer_cfg_manager through cfg_wr_data.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def norm_key(key: str) -> str:
    """Normalize CSV header names for flexible matching."""
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


def is_blank(value: object) -> bool:
    return value is None or str(value).strip() == ""


def as_int(value: object, default: Optional[int] = None, field: str = "") -> int:
    """Parse decimal or hex integer. Blank uses default if provided."""
    if is_blank(value):
        if default is None:
            raise ValueError(f"Missing required integer field: {field}")
        return int(default)

    s = str(value).strip().replace("_", "")
    try:
        if s.lower().startswith("0x"):
            return int(s, 16)
        return int(float(s))
    except ValueError as exc:
        raise ValueError(f"Cannot parse integer field {field!r}: {value!r}") from exc


def as_bool(value: object, default: bool = False) -> bool:
    if is_blank(value):
        return default
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
        return True
    if s in {"0", "false", "no", "n", "off", "disable", "disabled"}:
        return False
    return bool(as_int(value, default=int(default)))


def get_first(row: Dict[str, str], candidates: Sequence[str], default: Optional[object] = None) -> object:
    """Get first non-blank value among normalized candidate column names."""
    for c in candidates:
        k = norm_key(c)
        if k in row and not is_blank(row[k]):
            return row[k]
    return default


def ceil_log2_unsigned_depth(max_value: int) -> int:
    """Minimum bit width to represent unsigned value max_value."""
    if max_value <= 0:
        return 1
    return max_value.bit_length()


def sv_cast(width_name: str, value: int) -> str:
    return f"{width_name}'({value})"


def sv_bool(value: bool) -> str:
    return "1'b1" if value else "1'b0"


def hex32(value: int) -> str:
    return f"32'h{value & 0xFFFFFFFF:08X}"


def sanitize_comment(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_./:+\- ]", "_", text)


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------


@dataclass
class LayerRow:
    layer_id: int
    layer_name: str
    mode_name: str  # MODE1 or MODE2

    h_in: int
    w_in: int
    c_in: int
    f_out: int
    k: int
    h_out: int
    w_out: int

    pv_m1: int
    pf_m1: int
    pc_m2: int
    pf_m2: int

    conv_stride: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int

    relu_en: bool
    pool_en: bool
    pool_k: int
    pool_stride: int

    ifm_ddr_base: int
    wgt_ddr_base: int
    ofm_ddr_base: int

    first_layer: bool
    last_layer: bool


def parse_mode(value: object) -> str:
    """
    Map DSE mode notation to package enum.

    Accepted mode-1 examples: 0, m1, mode1, MODE1, pixel_filter, pvpf
    Accepted mode-2 examples: 1, m2, mode2, MODE2, channel_filter, pcpf

    Current CNN_V4 RTL enum numeric mode is 0=MODE1 and 1=MODE2.
    The default parser policy is therefore --mode-numeric rtl. Use
    --mode-numeric dse only for old CSV files generated with 1=MODE1, 2=MODE2.
    """
    raise RuntimeError("parse_mode requires mode_numeric; use parse_mode_with_policy()")


def parse_mode_with_policy(value: object, mode_numeric: str) -> str:
    if is_blank(value):
        raise ValueError("Missing mode field")

    s_raw = str(value).strip()
    s = re.sub(r"[^a-z0-9]+", "", s_raw.lower())

    if s in {"mode1", "m1", "pvpf", "pixelfilter", "pixelparallel", "pfpv"}:
        return "MODE1"
    if s in {"mode2", "m2", "pcpf", "channelfilter", "channelparallel", "pfpc"}:
        return "MODE2"

    # Numeric modes can be ambiguous across DSE and RTL conventions.
    if re.fullmatch(r"[0-9]+", s):
        v = int(s)
        if mode_numeric == "dse":
            if v == 1:
                return "MODE1"
            if v == 2:
                return "MODE2"
            if v == 0:
                # tolerate 0 as MODE1 in DSE files generated from enum casts
                return "MODE1"
        elif mode_numeric == "rtl":
            if v == 0:
                return "MODE1"
            if v == 1:
                return "MODE2"
            if v == 2:
                return "MODE2"

    raise ValueError(
        f"Unsupported mode value {value!r}. Use MODE1/MODE2 or set --mode-numeric dse|rtl."
    )


def normalize_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        rows: List[Dict[str, str]] = []
        for raw in reader:
            rows.append({norm_key(k): (v.strip() if isinstance(v, str) else v) for k, v in raw.items()})
    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")
    return rows


def parse_layer_rows(
    csv_rows: List[Dict[str, str]],
    *,
    mode_numeric: str,
    default_relu: bool,
    default_pool_k: int,
    default_pool_stride: int,
    default_ifm_base: int,
    default_wgt_base: int,
    default_ofm_base: int,
    addr_step: int,
) -> List[LayerRow]:
    out: List[LayerRow] = []
    n = len(csv_rows)

    for i, row in enumerate(csv_rows):
        layer_id = as_int(get_first(row, ["layer_id", "id", "idx"], i), field="layer_id")
        layer_name = str(get_first(row, ["layer_name", "name"], f"layer{layer_id}")).strip()
        mode_name = parse_mode_with_policy(get_first(row, ["mode", "layer_mode"]), mode_numeric)

        h_in = as_int(get_first(row, ["h_in", "hi", "input_h", "hin", "h"], None), field="h_in/hi")
        w_in = as_int(get_first(row, ["w_in", "wi", "input_w", "win", "w"], None), field="w_in/wi")
        c_in = as_int(get_first(row, ["c_in", "c", "channels", "cin"], None), field="c_in/c")
        f_out = as_int(get_first(row, ["f_out", "f", "filters", "cout", "out_channels"], None), field="f_out/f")
        k = as_int(get_first(row, ["k", "kernel", "kernel_size"], None), field="k")
        h_out = as_int(get_first(row, ["h_out", "ho", "conv_h_out", "output_h", "hout"], None), field="h_out/ho")
        w_out = as_int(get_first(row, ["w_out", "wo", "conv_w_out", "output_w", "wout"], None), field="w_out/wo")

        # Parallelism: support both compact pv/pf/pc and explicit pv_m1/pf_m1/pc_m2/pf_m2.
        pv_raw = get_first(row, ["pv_m1", "pv", "pv_cur"], 0)
        pf_m1_raw = get_first(row, ["pf_m1", "pf1", "pf_mode1"], None)
        pf_m2_raw = get_first(row, ["pf_m2", "pf2", "pf_mode2"], None)
        pf_raw = get_first(row, ["pf", "pf_cur"], 0)
        pc_raw = get_first(row, ["pc_m2", "pc", "pc_mode2"], 0)

        if mode_name == "MODE1":
            pv_m1 = as_int(pv_raw, default=0, field="pv_m1/pv")
            pf_m1 = as_int(pf_m1_raw if pf_m1_raw is not None else pf_raw, default=0, field="pf_m1/pf")
            pc_m2 = as_int(get_first(row, ["pc_m2", "pc", "pc_mode2"], 0), default=0, field="pc_m2")
            pf_m2 = as_int(pf_m2_raw if pf_m2_raw is not None else 0, default=0, field="pf_m2")
        else:
            pv_m1 = as_int(get_first(row, ["pv_m1"], 0), default=0, field="pv_m1")
            pf_m1 = as_int(pf_m1_raw if pf_m1_raw is not None else 0, default=0, field="pf_m1")
            pc_m2 = as_int(pc_raw, default=0, field="pc_m2/pc")
            pf_m2 = as_int(pf_m2_raw if pf_m2_raw is not None else pf_raw, default=0, field="pf_m2/pf")

        conv_stride = as_int(get_first(row, ["conv_stride", "stride", "s"], 1), default=1, field="conv_stride")
        pad_top = as_int(get_first(row, ["pad_top", "padding_top", "pt", "pad", "padding"], 0), default=0, field="pad_top")
        pad_bottom = as_int(get_first(row, ["pad_bottom", "padding_bottom", "pb", "pad", "padding"], pad_top), default=pad_top, field="pad_bottom")
        pad_left = as_int(get_first(row, ["pad_left", "padding_left", "pl", "pad", "padding"], 0), default=0, field="pad_left")
        pad_right = as_int(get_first(row, ["pad_right", "padding_right", "pr", "pad", "padding"], pad_left), default=pad_left, field="pad_right")

        relu_en = as_bool(get_first(row, ["relu_en", "relu"], int(default_relu)), default=default_relu)
        pool_en = as_bool(get_first(row, ["pool_en", "pool", "maxpool"], 0), default=False)
        pool_k = as_int(get_first(row, ["pool_k", "pool_size", "pool_kernel"], default_pool_k if pool_en else 0), default=(default_pool_k if pool_en else 0), field="pool_k")
        pool_stride = as_int(get_first(row, ["pool_stride", "pool_s"], default_pool_stride if pool_en else 0), default=(default_pool_stride if pool_en else 0), field="pool_stride")

        ifm_ddr_base = as_int(get_first(row, ["ifm_ddr_base", "ifm_base"], default_ifm_base + i * addr_step), default=(default_ifm_base + i * addr_step), field="ifm_ddr_base")
        wgt_ddr_base = as_int(get_first(row, ["wgt_ddr_base", "weight_ddr_base", "wgt_base", "weight_base"], default_wgt_base + i * addr_step), default=(default_wgt_base + i * addr_step), field="wgt_ddr_base")
        ofm_ddr_base = as_int(get_first(row, ["ofm_ddr_base", "ofm_base"], default_ofm_base + i * addr_step), default=(default_ofm_base + i * addr_step), field="ofm_ddr_base")

        first_layer = as_bool(get_first(row, ["first_layer", "is_first"], int(i == 0)), default=(i == 0))
        last_layer = as_bool(get_first(row, ["last_layer", "is_last"], int(i == n - 1)), default=(i == n - 1))

        out.append(
            LayerRow(
                layer_id=layer_id,
                layer_name=layer_name,
                mode_name=mode_name,
                h_in=h_in,
                w_in=w_in,
                c_in=c_in,
                f_out=f_out,
                k=k,
                h_out=h_out,
                w_out=w_out,
                pv_m1=pv_m1,
                pf_m1=pf_m1,
                pc_m2=pc_m2,
                pf_m2=pf_m2,
                conv_stride=conv_stride,
                pad_top=pad_top,
                pad_bottom=pad_bottom,
                pad_left=pad_left,
                pad_right=pad_right,
                relu_en=relu_en,
                pool_en=pool_en,
                pool_k=pool_k,
                pool_stride=pool_stride,
                ifm_ddr_base=ifm_ddr_base,
                wgt_ddr_base=wgt_ddr_base,
                ofm_ddr_base=ofm_ddr_base,
                first_layer=first_layer,
                last_layer=last_layer,
            )
        )
    return out


def validate_rows(layers: List[LayerRow], args: argparse.Namespace) -> List[str]:
    warnings: List[str] = []

    def check_unsigned(width_name: str, width: int, value: int, ctx: str):
        if value < 0 or value >= (1 << width):
            warnings.append(f"{ctx}: value {value} exceeds {width_name}={width} unsigned range")

    for L in layers:
        ctx = f"layer {L.layer_id} {L.layer_name}"
        for field_name, value, width_name, width in [
            ("h_in", L.h_in, "DIM_W", args.dim_w),
            ("w_in", L.w_in, "DIM_W", args.dim_w),
            ("c_in", L.c_in, "DIM_W", args.dim_w),
            ("f_out", L.f_out, "DIM_W", args.dim_w),
            ("h_out", L.h_out, "DIM_W", args.dim_w),
            ("w_out", L.w_out, "DIM_W", args.dim_w),
            ("k", L.k, "K_W", args.k_w),
            ("pv_m1", L.pv_m1, "PV_W", args.pv_w),
            ("pf_m1", L.pf_m1, "PF1_W", args.pf1_w),
            ("pc_m2", L.pc_m2, "PC2_W", args.pc2_w),
            ("pf_m2", L.pf_m2, "PF2_W", args.pf2_w),
            ("conv_stride", L.conv_stride, "STRIDE_W", args.stride_w),
            ("pad_top", L.pad_top, "PAD_W", args.pad_w),
            ("pad_bottom", L.pad_bottom, "PAD_W", args.pad_w),
            ("pad_left", L.pad_left, "PAD_W", args.pad_w),
            ("pad_right", L.pad_right, "PAD_W", args.pad_w),
            ("pool_k", L.pool_k, "POOL_W", args.pool_w),
            ("pool_stride", L.pool_stride, "POOL_W", args.pool_w),
        ]:
            check_unsigned(width_name, width, value, f"{ctx}.{field_name}")

        if args.ptotal is not None:
            if L.mode_name == "MODE1":
                prod = L.pv_m1 * L.pf_m1
                if prod != args.ptotal:
                    warnings.append(f"{ctx}: MODE1 pv_m1*pf_m1={prod}, expected PTOTAL={args.ptotal}")
            else:
                prod = L.pc_m2 * L.pf_m2
                if prod != args.ptotal:
                    warnings.append(f"{ctx}: MODE2 pc_m2*pf_m2={prod}, expected PTOTAL={args.ptotal}")

        if L.conv_stride >= (1 << args.stride_w):
            warnings.append(
                f"{ctx}: conv_stride={L.conv_stride} does not fit STRIDE_W={args.stride_w}. "
                "For canonical AlexNet stride=4, use --stride-w 3, but ensure downstream RTL ports can accept it."
            )

    return warnings


# -----------------------------------------------------------------------------
# SystemVerilog generation
# -----------------------------------------------------------------------------


def sv_layer_literal(L: LayerRow, indent: str = "      ") -> str:
    fields = [
        ("layer_id", sv_cast("LAYER_ID_W", L.layer_id)),
        ("mode", L.mode_name),
        ("h_in", sv_cast("DIM_W", L.h_in)),
        ("w_in", sv_cast("DIM_W", L.w_in)),
        ("c_in", sv_cast("DIM_W", L.c_in)),
        ("f_out", sv_cast("DIM_W", L.f_out)),
        ("k", sv_cast("K_W", L.k)),
        ("h_out", sv_cast("DIM_W", L.h_out)),
        ("w_out", sv_cast("DIM_W", L.w_out)),
        ("pv_m1", sv_cast("PV_W", L.pv_m1)),
        ("pf_m1", sv_cast("PF1_W", L.pf_m1)),
        ("pc_m2", sv_cast("PC2_W", L.pc_m2)),
        ("pf_m2", sv_cast("PF2_W", L.pf_m2)),
        ("conv_stride", sv_cast("STRIDE_W", L.conv_stride)),
        ("pad_top", sv_cast("PAD_W", L.pad_top)),
        ("pad_bottom", sv_cast("PAD_W", L.pad_bottom)),
        ("pad_left", sv_cast("PAD_W", L.pad_left)),
        ("pad_right", sv_cast("PAD_W", L.pad_right)),
        ("relu_en", sv_bool(L.relu_en)),
        ("pool_en", sv_bool(L.pool_en)),
        ("pool_k", sv_cast("POOL_W", L.pool_k)),
        ("pool_stride", sv_cast("POOL_W", L.pool_stride)),
        ("ifm_ddr_base", hex32(L.ifm_ddr_base)),
        ("wgt_ddr_base", hex32(L.wgt_ddr_base)),
        ("ofm_ddr_base", hex32(L.ofm_ddr_base)),
        ("first_layer", sv_bool(L.first_layer)),
        ("last_layer", sv_bool(L.last_layer)),
    ]

    body = ",\n".join(f"{indent}{name}: {value}" for name, value in fields)
    return "'{\n" + body + "\n    }"


def generate_sv(layers: List[LayerRow], args: argparse.Namespace, csv_path: Path) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_note = csv_path.as_posix()

    lines: List[str] = []
    lines.append("// -----------------------------------------------------------------------------")
    lines.append("// Auto-generated by dse_csv_to_layer_pkg_v2_rtlmode.py")
    lines.append(f"// Source CSV : {source_note}")
    lines.append(f"// Generated  : {generated_at}")
    lines.append("// Do not edit manually; regenerate from *_dse.csv.")
    lines.append("// Mode encoding: MODE1=1\'b0, MODE2=1\'b1, matching CNN_V4 RTL.")
    lines.append("// -----------------------------------------------------------------------------")
    lines.append("")
    lines.append(f"package {args.pkg_name};")
    lines.append("")
    lines.append("  // Widths are kept compatible with the current CNN_V4 layer_desc_t fields.")
    lines.append(f"  localparam int DIM_W      = {args.dim_w};")
    lines.append(f"  localparam int K_W        = {args.k_w};")
    lines.append(f"  localparam int PV_W       = {args.pv_w};")
    lines.append(f"  localparam int PF1_W      = {args.pf1_w};")
    lines.append(f"  localparam int PC2_W      = {args.pc2_w};")
    lines.append(f"  localparam int PF2_W      = {args.pf2_w};")
    lines.append(f"  localparam int ADDR_W     = {args.addr_w};")
    lines.append(f"  localparam int STRIDE_W   = {args.stride_w};")
    lines.append(f"  localparam int PAD_W      = {args.pad_w};")
    lines.append(f"  localparam int POOL_W     = {args.pool_w};")
    lines.append(f"  localparam int LAYER_ID_W = {args.layer_id_w};")
    lines.append("")
    lines.append("  typedef enum logic [0:0] {")
    lines.append("    MODE1 = 1'b0,")
    lines.append("    MODE2 = 1'b1")
    lines.append("  } layer_mode_e;")
    lines.append("")
    lines.append("  typedef struct packed {")
    lines.append("    logic [LAYER_ID_W-1:0] layer_id;")
    lines.append("    layer_mode_e mode;")
    lines.append("")
    lines.append("    logic [DIM_W-1:0] h_in;")
    lines.append("    logic [DIM_W-1:0] w_in;")
    lines.append("    logic [DIM_W-1:0] c_in;")
    lines.append("    logic [DIM_W-1:0] f_out;")
    lines.append("    logic [K_W-1:0]   k;")
    lines.append("    logic [DIM_W-1:0] h_out;")
    lines.append("    logic [DIM_W-1:0] w_out;")
    lines.append("")
    lines.append("    logic [PV_W-1:0]  pv_m1;")
    lines.append("    logic [PF1_W-1:0] pf_m1;")
    lines.append("    logic [PC2_W-1:0] pc_m2;")
    lines.append("    logic [PF2_W-1:0] pf_m2;")
    lines.append("")
    lines.append("    logic [STRIDE_W-1:0] conv_stride;")
    lines.append("    logic [PAD_W-1:0]    pad_top;")
    lines.append("    logic [PAD_W-1:0]    pad_bottom;")
    lines.append("    logic [PAD_W-1:0]    pad_left;")
    lines.append("    logic [PAD_W-1:0]    pad_right;")
    lines.append("    logic relu_en;")
    lines.append("    logic pool_en;")
    lines.append("    logic [POOL_W-1:0] pool_k;")
    lines.append("    logic [POOL_W-1:0] pool_stride;")
    lines.append("")
    lines.append("    logic [ADDR_W-1:0] ifm_ddr_base;")
    lines.append("    logic [ADDR_W-1:0] wgt_ddr_base;")
    lines.append("    logic [ADDR_W-1:0] ofm_ddr_base;")
    lines.append("")
    lines.append("    logic first_layer;")
    lines.append("    logic last_layer;")
    lines.append("  } layer_desc_t;")
    lines.append("")
    lines.append(f"  localparam int NUM_LAYERS = {len(layers)};")
    lines.append("")
    lines.append("  function automatic layer_desc_t get_layer_desc(input int unsigned idx);")
    lines.append("    case (idx)")
    for L in layers:
        comment = sanitize_comment(L.layer_name)
        lines.append(f"      {L.layer_id}: begin // {comment}")
        lines.append(f"        get_layer_desc = {sv_layer_literal(L, indent='          ')};")
        lines.append("      end")
    lines.append("      default: begin")
    lines.append("        get_layer_desc = '0;")
    lines.append("      end")
    lines.append("    endcase")
    lines.append("  endfunction")
    lines.append("")

    if args.emit_array:
        # Package constant array is useful for some TBs, but the function above is the safer interface.
        lines.append("  // Optional constant table. If your Vivado version dislikes package arrays,")
        lines.append("  // regenerate with --no-emit-array and use get_layer_desc(idx) instead.")
        lines.append("  localparam layer_desc_t LAYER_DESC [0:NUM_LAYERS-1] = '{")
        for idx, L in enumerate(layers):
            comma = "," if idx != len(layers) - 1 else ""
            lines.append(f"    // {L.layer_id}: {sanitize_comment(L.layer_name)}")
            lit = sv_layer_literal(L, indent="      ")
            lines.append(f"    {lit}{comma}")
        lines.append("  };")
        lines.append("")

    lines.append(f"endpackage : {args.pkg_name}")
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert *_dse.csv into cnn_layer_desc_pkg.sv for CNN_V4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", required=True, type=Path, help="Input *_dse.csv file")
    p.add_argument("--out", required=True, type=Path, help="Output cnn_layer_desc_pkg.sv path")
    p.add_argument("--pkg-name", default="cnn_layer_desc_pkg", help="SystemVerilog package name")

    p.add_argument(
        "--mode-numeric",
        choices=["dse", "rtl"],
        default="rtl",
        help="Interpret numeric mode values. rtl: 0=MODE1,1=MODE2. dse: legacy 1=MODE1,2=MODE2.",
    )

    # Width defaults follow the current package in CNN_V4; PC2 can be overridden to 6 if the integrated RTL contract is narrowed.
    p.add_argument("--dim-w", type=int, default=16)
    p.add_argument("--k-w", type=int, default=4)
    p.add_argument("--pv-w", type=int, default=8)
    p.add_argument("--pf1-w", type=int, default=8)
    p.add_argument("--pc2-w", type=int, default=8)
    p.add_argument("--pf2-w", type=int, default=8)
    p.add_argument("--addr-w", type=int, default=32)
    p.add_argument("--stride-w", type=int, default=2)
    p.add_argument("--pad-w", type=int, default=4)
    p.add_argument("--pool-w", type=int, default=2)
    p.add_argument("--layer-id-w", type=int, default=8)

    p.add_argument("--ptotal", type=int, default=None, help="Optional PTOTAL sanity check")
    p.add_argument("--default-relu", action="store_true", help="Default relu_en=1 if CSV has no relu column")
    p.add_argument("--default-pool-k", type=int, default=2)
    p.add_argument("--default-pool-stride", type=int, default=2)

    p.add_argument("--ifm-base", type=lambda x: int(x, 0), default=0)
    p.add_argument("--wgt-base", type=lambda x: int(x, 0), default=0)
    p.add_argument("--ofm-base", type=lambda x: int(x, 0), default=0)
    p.add_argument(
        "--addr-step",
        type=lambda x: int(x, 0),
        default=0,
        help="If CSV lacks DDR base columns, add layer_id*addr_step to each default base.",
    )

    p.add_argument("--emit-array", dest="emit_array", action="store_true", default=False)
    p.add_argument("--no-emit-array", dest="emit_array", action="store_false")
    p.add_argument("--strict", action="store_true", help="Treat validation warnings as errors")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    csv_rows = normalize_csv_rows(args.csv)
    layers = parse_layer_rows(
        csv_rows,
        mode_numeric=args.mode_numeric,
        default_relu=args.default_relu,
        default_pool_k=args.default_pool_k,
        default_pool_stride=args.default_pool_stride,
        default_ifm_base=args.ifm_base,
        default_wgt_base=args.wgt_base,
        default_ofm_base=args.ofm_base,
        addr_step=args.addr_step,
    )

    warnings = validate_rows(layers, args)
    if warnings:
        print("Validation warnings:", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)
        if args.strict:
            print("Strict mode enabled; aborting.", file=sys.stderr)
            return 2

    sv_text = generate_sv(layers, args, args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(sv_text, encoding="utf-8")

    mode1_cnt = sum(1 for x in layers if x.mode_name == "MODE1")
    mode2_cnt = sum(1 for x in layers if x.mode_name == "MODE2")
    print(f"Wrote {args.out}")
    print(f"Layers: {len(layers)}  MODE1: {mode1_cnt}  MODE2: {mode2_cnt}")
    if warnings:
        print("Generated with warnings. Review them before simulation/synthesis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
