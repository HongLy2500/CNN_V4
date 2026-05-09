#!/usr/bin/env python3
"""
Generate DDR images matching init_mem() in:
tb_cnn_top_9layer_m1_dcp_efficientnet_b0_tablevi_fullscale_with_expected_compare.sv

Outputs:
  ifm_l0.bin
  weights_9layer.bin
  ofm_zero.bin
  load_9layer_ddr_images.xsct.tcl
  read_9layer_ofm_head.xsct.tcl
  ddr_image_manifest.json

Physical DDR addresses assume:
  AXI_DDR_BASE_ADDR = 0x70000000
  DDR_WORD_W        = PV_MAX * DATA_W = 128 * 8 = 1024 bits
  word byte size    = 128 bytes

Therefore:
  DDR_IFM_BASE 0x00000 -> 0x70000000
  DDR_WGT_BASE 0x08000 -> 0x70400000
  DDR_OFM_BASE 0x10000 -> 0x70800000

Do not use the old small-smoke 0x70020000/0x70040000 addresses for this
full-scale 1024-bit-DDR-word top.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


DATA_W = 8
PTOTAL = 2048
PV_MAX = 128
PF_MAX = 128
PC_MODE2 = 32
PF_MODE2 = 64
C_MAX = 192
F_MAX = 192
W_MAX = 224
H_MAX = 224
HT = 4
K_MAX = 3
OFM_ROW_STRIDE = 16

DDR_ADDR_W = 20
DDR_IFM_BASE = 0x00000
DDR_IFM_SIZE = 0x08000
DDR_WGT_BASE = 0x08000
DDR_WGT_SIZE = 0x08000
DDR_OFM_BASE = 0x10000
DDR_OFM_SIZE = 0x08000
DDR_RSVD_BASE = 0x18000
DDR_RSVD_SIZE = 0x08000

DDR_WORD_W = PV_MAX * DATA_W
WORD_BYTES = DDR_WORD_W // 8
WGT_SUBWORDS = (PTOTAL + PV_MAX - 1) // PV_MAX

AXI_DDR_BASE_ADDR_DEFAULT = 0x70000000


def conv_out(h: int, w: int, k: int) -> tuple[int, int]:
    return h - k + 1, w - k + 1


def pooled(v: int, pool_en: int) -> int:
    return v // 2 if pool_en else v


# Layer table matching the testbench.
LAYERS: List[Dict[str, int]] = []

# L0
h_in, w_in, c_in, f_out, k, pool_en, pv, pf = 224, 224, 3, 32, 3, 1, 128, 16
h_conv, w_conv = conv_out(h_in, w_in, k)
LAYERS.append(dict(H_IN=h_in, W_IN=w_in, C_IN=c_in, F_OUT=f_out, K=k, POOL_EN=pool_en,
                   H_CONV_OUT=h_conv, W_CONV_OUT=w_conv, H_OUT=pooled(h_conv, pool_en),
                   W_OUT=pooled(w_conv, pool_en), PV=pv, PF=pf))

# L1..L8 derived from the testbench chain.
specs = [
    # F, K, pool, PV, PF
    (16, 3, 0, 128, 16),
    (24, 3, 1, 64, 32),
    (24, 3, 0, 64, 32),
    (40, 3, 1, 64, 32),
    (40, 3, 0, 32, 64),
    (80, 3, 1, 32, 64),
    (80, 3, 0, 32, 64),
    (192, 3, 0, 16, 128),
]
for f_out, k, pool_en, pv, pf in specs:
    prev = LAYERS[-1]
    h_in, w_in, c_in = prev["H_OUT"], prev["W_OUT"], prev["F_OUT"]
    h_conv, w_conv = conv_out(h_in, w_in, k)
    LAYERS.append(dict(H_IN=h_in, W_IN=w_in, C_IN=c_in, F_OUT=f_out, K=k, POOL_EN=pool_en,
                       H_CONV_OUT=h_conv, W_CONV_OUT=w_conv, H_OUT=pooled(h_conv, pool_en),
                       W_OUT=pooled(w_conv, pool_en), PV=pv, PF=pf))


def to_i8_byte(x: int) -> int:
    """Return x encoded as a signed int8 byte."""
    return x & 0xFF


def gen_l0_ifm(c: int, r: int, x: int) -> int:
    # gen_l0_ifm = signed(((c*7 + r*3 + x*5 + (r*x)%11) % 7) - 3)
    val = ((c * 7 + r * 3 + x * 5 + (r * x) % 11) % 7) - 3
    return to_i8_byte(val)


def gen_wgt(layer: int, f: int, c: int, ky: int, kx: int) -> int:
    # v = f*(7+2*layer) + c*(3+2*layer) + ky*(5+layer) + kx*(11+layer) + (1+layer)
    # gen_wgt = signed((v % 3) - 1)
    v = f * (7 + 2 * layer) + c * (3 + 2 * layer) + ky * (5 + layer) + kx * (11 + layer) + (1 + layer)
    val = (v % 3) - 1
    return to_i8_byte(val)


def layer_weight_words(layer: Dict[str, int]) -> dict:
    num_fgroup = (layer["F_OUT"] + layer["PF"] - 1) // layer["PF"]
    logical_bundles = num_fgroup * layer["C_IN"] * layer["K"] * layer["K"]
    phys_words = (logical_bundles + layer["PV"] - 1) // layer["PV"]
    ddr_words = phys_words * WGT_SUBWORDS
    return dict(num_fgroup=num_fgroup, logical_bundles=logical_bundles, phys_words=phys_words, ddr_words=ddr_words)


def compute_weight_bases() -> list[int]:
    bases = []
    base = DDR_WGT_BASE
    for layer in LAYERS:
        bases.append(base)
        base += layer_weight_words(layer)["ddr_words"]
    return bases


WGT_BASES = compute_weight_bases()
TOTAL_WGT_DDR_WORDS = (WGT_BASES[-1] - DDR_WGT_BASE) + layer_weight_words(LAYERS[-1])["ddr_words"]

FINAL_STORE_PACK = 1
FINAL_GROUPS = (LAYERS[8]["W_OUT"] + FINAL_STORE_PACK - 1) // FINAL_STORE_PACK
EXP_OFM_WORDS = LAYERS[8]["F_OUT"] * LAYERS[8]["H_OUT"] * FINAL_GROUPS


def write_byte(buf: bytearray, direct_word_index: int, lane: int, value_u8: int) -> None:
    """Write one 8-bit lane in one 1024-bit direct DDR word."""
    off = direct_word_index * WORD_BYTES + lane
    buf[off] = value_u8 & 0xFF


def generate_ifm() -> bytearray:
    l0 = LAYERS[0]
    words_per_row = (l0["W_IN"] + l0["PV"] - 1) // l0["PV"]
    total_words = l0["C_IN"] * l0["H_IN"] * words_per_row
    buf = bytearray(total_words * WORD_BYTES)

    for c in range(l0["C_IN"]):
        for r in range(l0["H_IN"]):
            for colg in range(words_per_row):
                word_idx = ((c * l0["H_IN"] + r) * words_per_row) + colg
                for lane in range(PV_MAX):
                    abs_col = colg * l0["PV"] + lane
                    if lane < l0["PV"] and abs_col < l0["W_IN"]:
                        write_byte(buf, word_idx, lane, gen_l0_ifm(c, r, abs_col))
    return buf


def write_wgt_lane(buf: bytearray, rel_base_words: int, phys_word: int, lane_abs: int, value_u8: int) -> None:
    ddr_subword = lane_abs // PV_MAX
    ddr_lane = lane_abs % PV_MAX
    word_idx = rel_base_words + phys_word * WGT_SUBWORDS + ddr_subword
    write_byte(buf, word_idx, ddr_lane, value_u8)


def generate_weights() -> bytearray:
    buf = bytearray(TOTAL_WGT_DDR_WORDS * WORD_BYTES)

    for layer_idx, layer in enumerate(LAYERS):
        info = layer_weight_words(layer)
        rel_base = WGT_BASES[layer_idx] - DDR_WGT_BASE
        logical_idx = 0
        for fg in range(info["num_fgroup"]):
            for c in range(layer["C_IN"]):
                for ky in range(layer["K"]):
                    for kx in range(layer["K"]):
                        phys_word = logical_idx // layer["PV"]
                        subword_idx = logical_idx % layer["PV"]
                        base_lane = subword_idx * layer["PF"]
                        for pf in range(layer["PF"]):
                            f = fg * layer["PF"] + pf
                            value = gen_wgt(layer_idx, f, c, ky, kx) if f < layer["F_OUT"] else 0
                            write_wgt_lane(buf, rel_base, phys_word, base_lane + pf, value)
                        logical_idx += 1
    return buf


def maybe_convert_expected_hex(expected_hex: Path, out_bin: Path) -> int:
    """
    Convert expected_final_ofm.hex into little-endian 1024-bit direct-DDR words.

    Each non-empty line is expected to be one DDR_WORD_W-wide hex word, as used
    by $readmemh into logic [DDR_WORD_W-1:0].
    """
    words = []
    for raw in expected_hex.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        line = line.replace("_", "")
        word_int = int(line, 16)
        words.append(word_int.to_bytes(WORD_BYTES, byteorder="little", signed=False))
    out_bin.write_bytes(b"".join(words))
    return len(words)


def make_tcl(out_dir: Path, ddr_base: int) -> str:
    ifm_path = (out_dir / "ifm_l0.bin").as_posix()
    wgt_path = (out_dir / "weights_9layer.bin").as_posix()
    ofm_zero_path = (out_dir / "ofm_zero.bin").as_posix()

    ifm_phys = ddr_base + DDR_IFM_BASE * WORD_BYTES
    wgt_phys = ddr_base + DDR_WGT_BASE * WORD_BYTES
    ofm_phys = ddr_base + DDR_OFM_BASE * WORD_BYTES

    return f"""# Auto-generated by gen_kv260_9layer_ddr_images.py
# Load DDR images for the 9-layer Mode-1 DCP/EfficientNet-B0-prefix test.
#
# Use after psu_init/psu_post_config and after programming the bitstream.
# Prefer the PSU target to avoid A53 cache effects.
#
# Current target number can change across sessions. If target 5 is your PSU,
# "targets -set 5" is also OK. The filter below is usually safer.

targets
catch {{ targets -set -filter {{name =~ "PSU"}} }}

set IFM_PHYS 0x{ifm_phys:08X}
set WGT_PHYS 0x{wgt_phys:08X}
set OFM_PHYS 0x{ofm_phys:08X}

puts "Loading IFM image to $IFM_PHYS ..."
dow -data {{{ifm_path}}} $IFM_PHYS

puts "Loading 9-layer weight image to $WGT_PHYS ..."
dow -data {{{wgt_path}}} $WGT_PHYS

puts "Clearing OFM region at $OFM_PHYS ..."
dow -data {{{ofm_zero_path}}} $OFM_PHYS

puts "Verify IFM first 16 x 32-bit words:"
mrd $IFM_PHYS 16

puts "Verify WGT first 16 x 32-bit words:"
mrd $WGT_PHYS 16

puts "Verify OFM first 16 x 32-bit words should be zero:"
mrd $OFM_PHYS 16

puts "DDR load complete."
puts "Now use VIO: soft_reset_n=0, run=0, abort=0; then soft_reset_n=1; wait cfg_done=1; pulse run."
"""


def make_read_tcl(ddr_base: int) -> str:
    ofm_phys = ddr_base + DDR_OFM_BASE * WORD_BYTES
    return f"""# Read a small head of final OFM region for the 9-layer test.
# For full compare, use a Linux/driver flow or dump a larger memory range.
targets
catch {{ targets -set -filter {{name =~ "PSU"}} }}

set OFM_PHYS 0x{ofm_phys:08X}

puts "OFM physical base = $OFM_PHYS"
puts "First 64 x 32-bit words at OFM base:"
mrd $OFM_PHYS 64

puts "Note: each RTL DDR word is 1024 bits = 128 bytes = 32 x 32-bit mrd words."
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="kv260_9layer_ddr_images", help="output directory")
    ap.add_argument("--ddr-base", default="0x70000000", help="AXI DDR physical base")
    ap.add_argument("--expected-hex", default=None, help="optional expected_final_ofm.hex to convert")
    ap.add_argument("--clear-ofm-full-region", action="store_true",
                    help="clear full DDR_OFM_SIZE instead of only expected final OFM words")
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ddr_base = int(args.ddr_base, 0)

    ifm = generate_ifm()
    wgt = generate_weights()
    clear_words = DDR_OFM_SIZE if args.clear_ofm_full_region else EXP_OFM_WORDS
    ofm_zero = bytearray(clear_words * WORD_BYTES)

    (out / "ifm_l0.bin").write_bytes(ifm)
    (out / "weights_9layer.bin").write_bytes(wgt)
    (out / "ofm_zero.bin").write_bytes(ofm_zero)
    (out / "load_9layer_ddr_images.xsct.tcl").write_text(make_tcl(out, ddr_base), encoding="utf-8")
    (out / "read_9layer_ofm_head.xsct.tcl").write_text(make_read_tcl(ddr_base), encoding="utf-8")

    expected_words = None
    if args.expected_hex:
        expected_words = maybe_convert_expected_hex(Path(args.expected_hex), out / "expected_final_ofm.bin")

    manifest = {
        "ddr_word_bits": DDR_WORD_W,
        "word_bytes": WORD_BYTES,
        "axi_ddr_base": f"0x{ddr_base:08X}",
        "ifm": {
            "rtl_word_base": f"0x{DDR_IFM_BASE:05X}",
            "physical_base": f"0x{ddr_base + DDR_IFM_BASE * WORD_BYTES:08X}",
            "direct_ddr_words": len(ifm) // WORD_BYTES,
            "bytes": len(ifm),
            "file": "ifm_l0.bin",
        },
        "weights": {
            "rtl_word_base": f"0x{DDR_WGT_BASE:05X}",
            "physical_base": f"0x{ddr_base + DDR_WGT_BASE * WORD_BYTES:08X}",
            "direct_ddr_words": len(wgt) // WORD_BYTES,
            "bytes": len(wgt),
            "file": "weights_9layer.bin",
        },
        "ofm_clear": {
            "rtl_word_base": f"0x{DDR_OFM_BASE:05X}",
            "physical_base": f"0x{ddr_base + DDR_OFM_BASE * WORD_BYTES:08X}",
            "direct_ddr_words": len(ofm_zero) // WORD_BYTES,
            "bytes": len(ofm_zero),
            "file": "ofm_zero.bin",
            "full_region_clear": bool(args.clear_ofm_full_region),
        },
        "expected_final_ofm_words": EXP_OFM_WORDS,
        "converted_expected_words": expected_words,
        "weight_bases": [
            {
                "layer": i,
                "rtl_word_base": f"0x{WGT_BASES[i]:05X}",
                "relative_words_from_wgt_base": WGT_BASES[i] - DDR_WGT_BASE,
                **layer_weight_words(LAYERS[i]),
            }
            for i in range(len(LAYERS))
        ],
        "layers": LAYERS,
    }
    (out / "ddr_image_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Generated into: {out}")
    print(f"IFM:     {len(ifm)} bytes at 0x{ddr_base + DDR_IFM_BASE * WORD_BYTES:08X}")
    print(f"Weights: {len(wgt)} bytes at 0x{ddr_base + DDR_WGT_BASE * WORD_BYTES:08X}")
    print(f"OFM clr: {len(ofm_zero)} bytes at 0x{ddr_base + DDR_OFM_BASE * WORD_BYTES:08X}")
    print("XSCT load script:", out / "load_9layer_ddr_images.xsct.tcl")


if __name__ == "__main__":
    main()
