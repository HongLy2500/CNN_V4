# CNN_V4 KV260 test package: VGG16-like 32x32 P128

This package is the board-safe replacement for the earlier 96x96 package. It keeps the VGG16-like 13-convolution block structure but uses a smaller hardware profile:

```text
Input  = 32x32x3
Layers = 13 conv layers, VGG16-like blocks 2-2-3-3-3
PV_MAX = 8
PC     = 8
PF     = 16
PTOTAL = 128
HT     = 4
C_MAX/F_MAX = 64
OFM_ROW_STRIDE = 4
WGT_DEPTH = 512
DDR word = 64 bits = 8 bytes
```

Layer descriptors are **not** placed in `cnn_layer_desc_pkg.sv`. The package remains the stable type-only package (`DIM_W=16`). The fixed descriptors are generated inside `rtl/kv260_cnn_eval_top_fixedcfg_vgg16_like_32_p128_defaults.sv`.

## RTL files

```text
rtl/cnn_ddr_defs.svh
rtl/cnn_layer_desc_pkg.sv
rtl/kv260_cnn_eval_top_fixedcfg_vgg16_like_32_p128_defaults.sv
rtl/kv260_cnn_smoke_top_vgg16_like_32_p128.sv
```

Set the Vivado top to:

```text
kv260_cnn_smoke_top_vgg16_like_32_p128
```

## Important AXI/DDR note

This profile uses `DDR_WORD_W = PV_MAX*DATA_W = 64` bits. The repository bridge `cnn_dma_to_axi_bridge_kv260` requires `AXI_DATA_W == DDR_WORD_W`, so this top exposes a 64-bit AXI data bus. If your Block Design/IP instance was previously 128-bit, refresh/repackage the IP and reconnect/regenerate the BD so SmartConnect/PS DDR sees the updated width.

## Software generation

Edit `$IMAGENET_VAL_DIR` in `run_vgg16_like_32_p128_flow.ps1`, then run:

```powershell
.\run_vgg16_like_32_p128_flow.ps1
```

Generated files include:

```text
eval/assets/vgg16_like_32_p128/inputs/img0000/ifm_ddr_cnnv4.hex
eval/assets/vgg16_like_32_p128/weights/all_weights_cnnv4.hex
eval/golden/vgg16_like_32_p128/img0000/expected_final_ofm_cnnv4.hex
eval/golden/vgg16_like_32_p128/img0000/expected_ofm_words.txt
```

## KV260 run

```powershell
$OFM_WORDS = Get-Content eval\golden\vgg16_like_32_p128\img0000\expected_ofm_words.txt

xsct eval\board\run_kv260_eval.tcl `
  -ifm eval\assets\vgg16_like_32_p128\inputs\img0000\ifm_ddr_cnnv4.hex `
  -wgt eval\assets\vgg16_like_32_p128\weights\all_weights_cnnv4.hex `
  -out eval\results\vgg16_like_32_p128\img0000\fpga_ofm_readback.hex `
  -ofm-words $OFM_WORDS `
  -word-bytes 8 `
  -ifm-base-word 0x00000 `
  -wgt-base-word 0x08000 `
  -ofm-base-word 0xF0000 `
  -wgt-size-words 0xE8000 `
  -ctrl-mode manual
```

## Compare

```powershell
python eval\scripts\compare_ofm.py `
  --expected eval\golden\vgg16_like_32_p128\img0000\expected_final_ofm_cnnv4.hex `
  --actual eval\results\vgg16_like_32_p128\img0000\fpga_ofm_readback.hex `
  --word-bits 64 `
  --out-report eval\results\vgg16_like_32_p128\img0000\compare_report.txt
```
