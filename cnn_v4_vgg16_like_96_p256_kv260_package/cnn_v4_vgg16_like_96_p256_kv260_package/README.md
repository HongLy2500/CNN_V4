# CNN_V4 KV260 test package: VGG16-like 96x96 P256

This package is for the fixed benchmark you selected:

```text
PV_MAX = 16
PF_MAX = 16
PC     = 16
PTOTAL = 256
HT     = 3

Mode1: L0-L6,  Pv=16, Pf=16
Mode2: L7-L12, PC=16, Pf=16

Input : 96x96x3
K     : 3
stride: 1
pad   : 1
pool  : 2x2 on L1, L3, L6, L9, L12
Final output tensor: 3x3x128
```

This is a **VGG16-like scaled workload**, not canonical VGG16. It uses deterministic INT8
weights because the channel counts are scaled to 1/4 and do not match TorchVision
VGG16 pretrained weights.

## Files included

```text
rtl/
  cnn_ddr_defs.svh
  cnn_layer_desc_pkg.sv
  kv260_cnn_eval_top_pkgcfg_vgg16_like_96_p256_defaults.sv
  kv260_cnn_smoke_top_vgg16_like_96_p256.sv

eval/configs/
  vgg16_like_96_p256_hw.json
  vgg16_like_96_p256_layers.csv

eval/models/
  vgg16_like_96_p256.json

eval/scripts/
  prepare_imagenet_inputs.py
  gen_vgg16_like_weights.py
  pack_ifm_for_cnn_v4.py
  pack_weights_for_cnn_v4.py
  layer_csv_to_layer_pkg.py
  gen_fixedpoint_golden.py
  pack_expected_ofm_for_cnn_v4.py
  compare_ofm.py

eval/board/
  run_kv260_eval.tcl

eval/reports/
  collect_vivado_reports.tcl
```

## How to run software preparation

Copy this package into the `CNN_V4` repository root.

Install dependencies:

```powershell
pip install -r requirements.txt
```

Edit the ImageNet path in:

```text
run_vgg16_like_96_p256_flow.ps1
```

Then run:

```powershell
.\run_vgg16_like_96_p256_flow.ps1
```

The flow generates:

```text
cnn_layer_desc_pkg.sv

eval/assets/vgg16_like_96_p256/inputs/img0000/ifm_ddr_cnnv4.hex
eval/assets/vgg16_like_96_p256/weights/all_weights_cnnv4.hex
eval/golden/vgg16_like_96_p256/img0000/expected_final_ofm_cnnv4.hex
eval/golden/vgg16_like_96_p256/img0000/expected_ofm_words.txt
```

## Vivado build

Add these files to the Vivado project:

```text
cnn_layer_desc_pkg.sv
rtl/cnn_ddr_defs.svh
rtl/kv260_cnn_eval_top_pkgcfg_vgg16_like_96_p256_defaults.sv
rtl/kv260_cnn_smoke_top_vgg16_like_96_p256.sv
```

Set top module:

```text
kv260_cnn_smoke_top_vgg16_like_96_p256
```

The top uses `cnn_layer_desc_pkg::get_layer_desc()` to load the 13 fixed layers.

## KV260 run

After programming the bitstream, run:

```powershell
$OFM_WORDS = Get-Content eval\golden\vgg16_like_96_p256\img0000\expected_ofm_words.txt

xsct eval\board\run_kv260_eval.tcl `
  -ifm eval\assets\vgg16_like_96_p256\inputs\img0000\ifm_ddr_cnnv4.hex `
  -wgt eval\assets\vgg16_like_96_p256\weights\all_weights_cnnv4.hex `
  -out eval\results\vgg16_like_96_p256\img0000\fpga_ofm_readback.hex `
  -ofm-words $OFM_WORDS `
  -word-bytes 16 `
  -ifm-base-word 0x000000 `
  -wgt-base-word 0x010000 `
  -ofm-base-word 0x020000 `
  -wgt-size-words 0x010000 `
  -ctrl-mode manual
```

Manual mode means the TCL script preloads IFM/WGT and clears OFM, then waits while
you drive `soft_reset_n`, `run`, and `abort` through VIO/GPIO.

## Compare

```powershell
python eval\scripts\compare_ofm.py `
  --expected eval\golden\vgg16_like_96_p256\img0000\expected_final_ofm_cnnv4.hex `
  --actual eval\results\vgg16_like_96_p256\img0000\fpga_ofm_readback.hex `
  --word-bits 128 `
  --out-report eval\results\vgg16_like_96_p256\img0000\compare_report.txt
```

## Notes

- `vgg16_like_96_p256_layers.csv` is the single source of truth for layer shapes, modes and parallelism.
- `pack_weights_for_cnn_v4.py` writes `vgg16_like_96_p256_layers_with_wgt_base.csv`.
- `layer_csv_to_layer_pkg.py` generates `cnn_layer_desc_pkg.sv` from the CSV that contains the correct `wgt_ddr_base`.
- Do not use TorchVision VGG16 pretrained weights for this scaled benchmark.
