#!/usr/bin/env python3
"""
layer_csv_to_layer_pkg.py
Generate cnn_layer_desc_pkg.sv from a fixed layer CSV.

RTL convention:
  mode=0 -> MODE1
  mode=1 -> MODE2

This is intentionally not a DSE script. It only converts a fixed, reviewed
layer table into the SystemVerilog package consumed by the KV260 top.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Any


def is_blank(x: Any) -> bool:
    return x is None or str(x).strip() == ""


def parse_int(x: Any, default=None, field: str = "") -> int:
    if is_blank(x):
        if default is None:
            raise ValueError(f"Missing field {field}")
        return int(default)
    s = str(x).strip().replace("_", "")
    return int(s, 16) if s.lower().startswith("0x") else int(float(s))


def hex32(x: int) -> str:
    return f"32'h{x & 0xFFFFFFFF:08X}"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def sv_lit(width: int, value: int) -> str:
    return f"{width}'d{int(value)}"


def row_to_sv(row: Dict[str, str], widths: Dict[str, int]) -> str:
    lid = parse_int(row.get("layer_id"), field="layer_id")
    mode = parse_int(row.get("mode"), field="mode")
    if mode not in (0, 1):
        raise ValueError(f"Layer {lid}: mode must be 0 or 1, got {mode}")
    wgt = parse_int(row.get("wgt_ddr_base"), field="wgt_ddr_base")
    ifm = parse_int(row.get("ifm_ddr_base"), 0, "ifm_ddr_base")
    ofm = parse_int(row.get("ofm_ddr_base"), 0, "ofm_ddr_base")

    f = {
        "layer_id": sv_lit(widths["LAYER_ID_W"], lid),
        "mode": "MODE1" if mode == 0 else "MODE2",
        "h_in": sv_lit(widths["DIM_W"], parse_int(row.get("h_in"), field="h_in")),
        "w_in": sv_lit(widths["DIM_W"], parse_int(row.get("w_in"), field="w_in")),
        "c_in": sv_lit(widths["DIM_W"], parse_int(row.get("c_in"), field="c_in")),
        "f_out": sv_lit(widths["DIM_W"], parse_int(row.get("f_out"), field="f_out")),
        "k": sv_lit(widths["K_W"], parse_int(row.get("k"), field="k")),
        "h_out": sv_lit(widths["DIM_W"], parse_int(row.get("h_out"), field="h_out")),
        "w_out": sv_lit(widths["DIM_W"], parse_int(row.get("w_out"), field="w_out")),
        "pv_m1": sv_lit(widths["PV_W"], parse_int(row.get("pv_m1"), 0, "pv_m1")),
        "pf_m1": sv_lit(widths["PF1_W"], parse_int(row.get("pf_m1"), 0, "pf_m1")),
        "pc_m2": sv_lit(widths["PC2_W"], parse_int(row.get("pc_m2"), 0, "pc_m2")),
        "pf_m2": sv_lit(widths["PF2_W"], parse_int(row.get("pf_m2"), 0, "pf_m2")),
        "conv_stride": sv_lit(widths["STRIDE_W"], parse_int(row.get("conv_stride"), 1, "conv_stride")),
        "pad_top": sv_lit(widths["PAD_W"], parse_int(row.get("pad_top"), 0, "pad_top")),
        "pad_bottom": sv_lit(widths["PAD_W"], parse_int(row.get("pad_bottom"), 0, "pad_bottom")),
        "pad_left": sv_lit(widths["PAD_W"], parse_int(row.get("pad_left"), 0, "pad_left")),
        "pad_right": sv_lit(widths["PAD_W"], parse_int(row.get("pad_right"), 0, "pad_right")),
        "relu_en": "1'b1" if parse_int(row.get("relu_en"), 1, "relu_en") else "1'b0",
        "pool_en": "1'b1" if parse_int(row.get("pool_en"), 0, "pool_en") else "1'b0",
        "pool_k": sv_lit(widths["POOL_W"], parse_int(row.get("pool_k"), 2, "pool_k")),
        "pool_stride": sv_lit(widths["POOL_W"], parse_int(row.get("pool_stride"), 2, "pool_stride")),
        "ifm_ddr_base": hex32(ifm),
        "wgt_ddr_base": hex32(wgt),
        "ofm_ddr_base": hex32(ofm),
        "first_layer": "1'b1" if parse_int(row.get("first_layer"), 0, "first_layer") else "1'b0",
        "last_layer": "1'b1" if parse_int(row.get("last_layer"), 0, "last_layer") else "1'b0",
    }

    return """'{
      layer_id     : %(layer_id)s,
      mode         : %(mode)s,
      h_in         : %(h_in)s,
      w_in         : %(w_in)s,
      c_in         : %(c_in)s,
      f_out        : %(f_out)s,
      k            : %(k)s,
      h_out        : %(h_out)s,
      w_out        : %(w_out)s,
      pv_m1        : %(pv_m1)s,
      pf_m1        : %(pf_m1)s,
      pc_m2        : %(pc_m2)s,
      pf_m2        : %(pf_m2)s,
      conv_stride  : %(conv_stride)s,
      pad_top      : %(pad_top)s,
      pad_bottom   : %(pad_bottom)s,
      pad_left     : %(pad_left)s,
      pad_right    : %(pad_right)s,
      relu_en      : %(relu_en)s,
      pool_en      : %(pool_en)s,
      pool_k       : %(pool_k)s,
      pool_stride  : %(pool_stride)s,
      ifm_ddr_base : %(ifm_ddr_base)s,
      wgt_ddr_base : %(wgt_ddr_base)s,
      ofm_ddr_base : %(ofm_ddr_base)s,
      first_layer  : %(first_layer)s,
      last_layer   : %(last_layer)s
    }""" % f


def generate_pkg(rows: List[Dict[str, str]], widths: Dict[str, int], source: str) -> str:
    cases = []
    for i, row in enumerate(rows):
        lid = parse_int(row.get("layer_id"), i, "layer_id")
        cases.append(f"      {lid}: get_layer_desc = {row_to_sv(row, widths)};")
    case_body = "\n".join(cases)

    return f"""package cnn_layer_desc_pkg;
// Auto-generated by layer_csv_to_layer_pkg.py
// Source CSV: {source}
// RTL mode convention: MODE1=0, MODE2=1

localparam int DIM_W      = {widths['DIM_W']};
localparam int K_W        = {widths['K_W']};
localparam int PV_W       = {widths['PV_W']};
localparam int PF1_W      = {widths['PF1_W']};
localparam int PC2_W      = {widths['PC2_W']};
localparam int PF2_W      = {widths['PF2_W']};
localparam int ADDR_W     = {widths['ADDR_W']};
localparam int STRIDE_W   = {widths['STRIDE_W']};
localparam int PAD_W      = {widths['PAD_W']};
localparam int POOL_W     = {widths['POOL_W']};
localparam int LAYER_ID_W = {widths['LAYER_ID_W']};

localparam int NUM_LAYERS = {len(rows)};

typedef enum logic [0:0] {{
  MODE1 = 1'b0,
  MODE2 = 1'b1
}} layer_mode_e;

typedef struct packed {{
  logic [LAYER_ID_W-1:0] layer_id;
  layer_mode_e mode;

  logic [DIM_W-1:0] h_in;
  logic [DIM_W-1:0] w_in;
  logic [DIM_W-1:0] c_in;
  logic [DIM_W-1:0] f_out;
  logic [K_W-1:0]   k;
  logic [DIM_W-1:0] h_out;
  logic [DIM_W-1:0] w_out;

  logic [PV_W-1:0]  pv_m1;
  logic [PF1_W-1:0] pf_m1;
  logic [PC2_W-1:0] pc_m2;
  logic [PF2_W-1:0] pf_m2;

  logic [STRIDE_W-1:0] conv_stride;
  logic [PAD_W-1:0]    pad_top;
  logic [PAD_W-1:0]    pad_bottom;
  logic [PAD_W-1:0]    pad_left;
  logic [PAD_W-1:0]    pad_right;
  logic                relu_en;
  logic                pool_en;
  logic [POOL_W-1:0]   pool_k;
  logic [POOL_W-1:0]   pool_stride;

  logic [ADDR_W-1:0] ifm_ddr_base;
  logic [ADDR_W-1:0] wgt_ddr_base;
  logic [ADDR_W-1:0] ofm_ddr_base;

  logic first_layer;
  logic last_layer;
}} layer_desc_t;

function automatic layer_desc_t get_layer_desc(input int unsigned idx);
  begin
    get_layer_desc = '0;
    unique case (idx)
{case_body}
      default: get_layer_desc = '0;
    endcase
  end
endfunction

endpackage
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim-w", type=int, default=8)
    ap.add_argument("--k-w", type=int, default=4)
    ap.add_argument("--pv-w", type=int, default=8)
    ap.add_argument("--pf1-w", type=int, default=8)
    ap.add_argument("--pc2-w", type=int, default=8)
    ap.add_argument("--pf2-w", type=int, default=8)
    ap.add_argument("--addr-w", type=int, default=32)
    ap.add_argument("--stride-w", type=int, default=2)
    ap.add_argument("--pad-w", type=int, default=4)
    ap.add_argument("--pool-w", type=int, default=2)
    ap.add_argument("--layer-id-w", type=int, default=8)
    args = ap.parse_args()

    rows = read_csv(Path(args.csv))
    widths = {
        "DIM_W": args.dim_w,
        "K_W": args.k_w,
        "PV_W": args.pv_w,
        "PF1_W": args.pf1_w,
        "PC2_W": args.pc2_w,
        "PF2_W": args.pf2_w,
        "ADDR_W": args.addr_w,
        "STRIDE_W": args.stride_w,
        "PAD_W": args.pad_w,
        "POOL_W": args.pool_w,
        "LAYER_ID_W": args.layer_id_w,
    }

    text = generate_pkg(rows, widths, args.csv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"[OK] Wrote {out} from {args.csv} ({len(rows)} layers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
