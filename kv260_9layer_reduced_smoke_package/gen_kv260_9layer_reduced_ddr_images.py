#!/usr/bin/env python3
"""
Generate DDR images for kv260_cnn_smoke_top_9layer_reduced_kv260.sv.

Reduced 9-layer Mode-1 board smoke:
  DATA_W=8, PTOTAL=16, PV_MAX=4, PF_MAX=4, DDR_WORD_W=32
  IFM physical base = 0x70000000
  WGT physical base = 0x70020000
  OFM physical base = 0x70040000

Outputs:
  ifm_l0.bin
  weights_9layer_reduced.bin
  ofm_zero.bin
  load_9layer_reduced_ddr_images.xsct.tcl
  read_9layer_reduced_ofm.xsct.tcl
  manifest.json
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

DATA_W = 8
PTOTAL = 16
PV_MAX = 4
PF_MAX = 4
DDR_WORD_W = PV_MAX * DATA_W
WORD_BYTES = DDR_WORD_W // 8
WGT_SUBWORDS = (PTOTAL + PV_MAX - 1) // PV_MAX

DDR_IFM_BASE = 0x00000
DDR_WGT_BASE = 0x08000
DDR_OFM_BASE = 0x10000
AXI_BASE_DEFAULT = 0x70000000

def conv_out(h, w, k):
    return h-k+1, w-k+1

def pooled(v, en):
    return v//2 if en else v

LAYERS = []
h,w,c,f,k,pool,pv,pf = 144,144,3,8,3,1,4,4
hc,wc = conv_out(h,w,k)
LAYERS.append(dict(H_IN=h,W_IN=w,C_IN=c,F_OUT=f,K=k,POOL_EN=pool,H_CONV_OUT=hc,W_CONV_OUT=wc,H_OUT=pooled(hc,pool),W_OUT=pooled(wc,pool),PV=pv,PF=pf))
for f,k,pool,pv,pf in [
    (8,3,0,4,4),
    (8,3,1,4,4),
    (8,3,0,4,4),
    (16,3,1,4,4),
    (16,3,0,4,4),
    (16,3,1,4,4),
    (16,3,0,4,4),
    (16,3,0,4,4),
]:
    prev = LAYERS[-1]
    h,w,c = prev["H_OUT"], prev["W_OUT"], prev["F_OUT"]
    hc,wc = conv_out(h,w,k)
    LAYERS.append(dict(H_IN=h,W_IN=w,C_IN=c,F_OUT=f,K=k,POOL_EN=pool,H_CONV_OUT=hc,W_CONV_OUT=wc,H_OUT=pooled(hc,pool),W_OUT=pooled(wc,pool),PV=pv,PF=pf))

def i8(x):
    return x & 0xff

def gen_l0_ifm(c, r, x):
    # Same style as full-scale testbench generator.
    return i8(((c*7 + r*3 + x*5 + (r*x)%11) % 7) - 3)

def gen_wgt(layer, f, c, ky, kx):
    v = f*(7+2*layer) + c*(3+2*layer) + ky*(5+layer) + kx*(11+layer) + (1+layer)
    return i8((v % 3) - 1)

def layer_weight_info(layer):
    num_fgroup = (layer["F_OUT"] + layer["PF"] - 1) // layer["PF"]
    logical_bundles = num_fgroup * layer["C_IN"] * layer["K"] * layer["K"]
    phys_words = (logical_bundles + layer["PV"] - 1) // layer["PV"]
    ddr_words = phys_words * WGT_SUBWORDS
    return dict(num_fgroup=num_fgroup, logical_bundles=logical_bundles, phys_words=phys_words, ddr_words=ddr_words)

def weight_bases():
    bases = []
    base = DDR_WGT_BASE
    for l in LAYERS:
        bases.append(base)
        base += layer_weight_info(l)["ddr_words"]
    return bases

WGT_BASES = weight_bases()
TOTAL_WGT_WORDS = (WGT_BASES[-1] - DDR_WGT_BASE) + layer_weight_info(LAYERS[-1])["ddr_words"]
FINAL_WORDS = LAYERS[-1]["F_OUT"] * LAYERS[-1]["H_OUT"] * LAYERS[-1]["W_OUT"]

def put_lane(buf, word, lane, val):
    buf[word*WORD_BYTES + lane] = val & 0xff

def gen_ifm():
    l0 = LAYERS[0]
    words_per_row = (l0["W_IN"] + l0["PV"] - 1) // l0["PV"]
    total_words = l0["C_IN"] * l0["H_IN"] * words_per_row
    buf = bytearray(total_words * WORD_BYTES)
    for c in range(l0["C_IN"]):
        for r in range(l0["H_IN"]):
            for colg in range(words_per_row):
                word = ((c*l0["H_IN"] + r) * words_per_row) + colg
                for lane in range(PV_MAX):
                    x = colg*l0["PV"] + lane
                    if lane < l0["PV"] and x < l0["W_IN"]:
                        put_lane(buf, word, lane, gen_l0_ifm(c, r, x))
    return buf

def write_wgt_lane(buf, rel_base_words, phys_word, lane_abs, val):
    subword = lane_abs // PV_MAX
    lane = lane_abs % PV_MAX
    word = rel_base_words + phys_word * WGT_SUBWORDS + subword
    put_lane(buf, word, lane, val)

def gen_weights():
    buf = bytearray(TOTAL_WGT_WORDS * WORD_BYTES)
    for li, layer in enumerate(LAYERS):
        info = layer_weight_info(layer)
        rel_base = WGT_BASES[li] - DDR_WGT_BASE
        logical_idx = 0
        for fg in range(info["num_fgroup"]):
            for c in range(layer["C_IN"]):
                for ky in range(layer["K"]):
                    for kx in range(layer["K"]):
                        phys_word = logical_idx // layer["PV"]
                        subword_idx = logical_idx % layer["PV"]
                        base_lane = subword_idx * layer["PF"]
                        for pf in range(layer["PF"]):
                            f = fg*layer["PF"] + pf
                            val = gen_wgt(li, f, c, ky, kx) if f < layer["F_OUT"] else 0
                            write_wgt_lane(buf, rel_base, phys_word, base_lane + pf, val)
                        logical_idx += 1
    return buf

def make_load_tcl(out_dir, ddr_base):
    ifm_phys = ddr_base + DDR_IFM_BASE * WORD_BYTES
    wgt_phys = ddr_base + DDR_WGT_BASE * WORD_BYTES
    ofm_phys = ddr_base + DDR_OFM_BASE * WORD_BYTES
    return f"""# Auto-generated reduced 9-layer DDR load script.
# Prefer PSU target to avoid A53 cache effects.
targets
catch {{ targets -set -filter {{name =~ "PSU"}} }}

set IFM_PHYS 0x{ifm_phys:08X}
set WGT_PHYS 0x{wgt_phys:08X}
set OFM_PHYS 0x{ofm_phys:08X}

puts "Loading IFM to $IFM_PHYS"
dow -data {{{(out_dir/'ifm_l0.bin').as_posix()}}} $IFM_PHYS

puts "Loading weights to $WGT_PHYS"
dow -data {{{(out_dir/'weights_9layer_reduced.bin').as_posix()}}} $WGT_PHYS

puts "Clearing OFM at $OFM_PHYS"
dow -data {{{(out_dir/'ofm_zero.bin').as_posix()}}} $OFM_PHYS

puts "IFM check:"
mrd $IFM_PHYS 16
puts "WGT check:"
mrd $WGT_PHYS 16
puts "OFM check:"
mrd $OFM_PHYS 16
"""

def make_read_tcl(ddr_base):
    ofm_phys = ddr_base + DDR_OFM_BASE * WORD_BYTES
    return f"""targets
catch {{ targets -set -filter {{name =~ "PSU"}} }}
set OFM_PHYS 0x{ofm_phys:08X}
puts "OFM final base = $OFM_PHYS"
mrd $OFM_PHYS 64
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="kv260_9layer_reduced_ddr_images")
    ap.add_argument("--ddr-base", default="0x70000000")
    ap.add_argument("--clear-words", type=int, default=1024, help="OFM words to clear; default 1024")
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ddr_base = int(args.ddr_base, 0)

    ifm = gen_ifm()
    wgt = gen_weights()
    ofm_zero = bytearray(max(args.clear_words, FINAL_WORDS) * WORD_BYTES)

    (out/"ifm_l0.bin").write_bytes(ifm)
    (out/"weights_9layer_reduced.bin").write_bytes(wgt)
    (out/"ofm_zero.bin").write_bytes(ofm_zero)
    (out/"load_9layer_reduced_ddr_images.xsct.tcl").write_text(make_load_tcl(out, ddr_base), encoding="utf-8")
    (out/"read_9layer_reduced_ofm.xsct.tcl").write_text(make_read_tcl(ddr_base), encoding="utf-8")

    manifest = {
        "params": dict(DATA_W=DATA_W, PTOTAL=PTOTAL, PV_MAX=PV_MAX, PF_MAX=PF_MAX, DDR_WORD_W=DDR_WORD_W, WORD_BYTES=WORD_BYTES),
        "physical_addresses": {
            "IFM": f"0x{ddr_base + DDR_IFM_BASE*WORD_BYTES:08X}",
            "WGT": f"0x{ddr_base + DDR_WGT_BASE*WORD_BYTES:08X}",
            "OFM": f"0x{ddr_base + DDR_OFM_BASE*WORD_BYTES:08X}",
        },
        "files": {
            "ifm_l0.bin": len(ifm),
            "weights_9layer_reduced.bin": len(wgt),
            "ofm_zero.bin": len(ofm_zero),
        },
        "layers": LAYERS,
        "weight_bases": [
            {"layer": i, "rtl_word_base": f"0x{WGT_BASES[i]:05X}", **layer_weight_info(LAYERS[i])}
            for i in range(len(LAYERS))
        ],
        "final_ofm_words": FINAL_WORDS,
    }
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated into {out}")
    print(f"IFM bytes: {len(ifm)} @ 0x{ddr_base + DDR_IFM_BASE*WORD_BYTES:08X}")
    print(f"WGT bytes: {len(wgt)} @ 0x{ddr_base + DDR_WGT_BASE*WORD_BYTES:08X}")
    print(f"OFM clear bytes: {len(ofm_zero)} @ 0x{ddr_base + DDR_OFM_BASE*WORD_BYTES:08X}")
    print(f"Final OFM words expected to be produced: {FINAL_WORDS}")

if __name__ == "__main__":
    main()
