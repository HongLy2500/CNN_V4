# VGG16-like 32x32 P128 board-safe software preparation flow.
# Run this from the CNN_V4 repository root after copying this package into the repo.
# Edit this path first:
$IMAGENET_VAL_DIR = "D:\sdh\ILSVRC2012_img_val"

$BENCH = "vgg16_like_32_p128"
$WORD_LANES = 8

New-Item -ItemType Directory -Force eval\assets\$BENCH\inputs | Out-Null
New-Item -ItemType Directory -Force eval\assets\$BENCH\weights | Out-Null
New-Item -ItemType Directory -Force eval\golden\$BENCH\img0000 | Out-Null
New-Item -ItemType Directory -Force eval\results\$BENCH\img0000 | Out-Null

Write-Host "== 1. Prepare ImageNet input resized to 32x32 =="
python eval\scripts\prepare_imagenet_inputs.py `
  --imagenet-val-dir $IMAGENET_VAL_DIR `
  --out-dir eval\assets\$BENCH\inputs `
  --height 32 `
  --width 32 `
  --count 5

Write-Host "== 2. Generate deterministic INT8 weights =="
python eval\scripts\gen_vgg16_like_weights.py `
  --layers-csv eval\configs\vgg16_like_32_p128_layers.csv `
  --out-dir eval\assets\$BENCH\weights `
  --pattern balanced_sparse_varied `
  --nonzero-per-filter 6 `
  --max-abs-weight 2 `
  --overwrite

Write-Host "== 3. Pack initial IFM for CNN_V4 =="
python eval\scripts\pack_ifm_for_cnn_v4.py `
  --input eval\assets\$BENCH\inputs\img0000\input_uint8_hwc.npy `
  --dse-csv eval\configs\vgg16_like_32_p128_layers.csv `
  --layer-index 0 `
  --out eval\assets\$BENCH\inputs\img0000\ifm_ddr_cnnv4.hex `
  --word-lanes $WORD_LANES `
  --overwrite

Write-Host "== 4. Pack weights for CNN_V4 =="
python eval\scripts\pack_weights_for_cnn_v4.py `
  --weights-dir eval\assets\$BENCH\weights `
  --dse-csv eval\configs\vgg16_like_32_p128_layers.csv `
  --out eval\assets\$BENCH\weights\all_weights_cnnv4.hex `
  --offsets-csv eval\assets\$BENCH\weights\weight_offsets_cnnv4.csv `
  --dse-out eval\configs\vgg16_like_32_p128_layers_with_wgt_base.csv `
  --ptotal 128 `
  --ddr-word-lanes $WORD_LANES `
  --wgt-base 0x08000 `
  --overwrite

Write-Host "== 5. Generate PyTorch fixed-point golden output =="
python eval\scripts\gen_fixedpoint_golden.py `
  --descriptor eval\models\vgg16_like_32_p128.json `
  --input eval\assets\$BENCH\inputs\img0000\input_uint8_hwc.npy `
  --weights-dir eval\assets\$BENCH\weights `
  --out-dir eval\golden\$BENCH\img0000 `
  --store-policy saturate_s8 `
  --input-interpretation auto `
  --strict-shapes `
  --overwrite

Write-Host "== 6. Pack expected OFM for CNN_V4 OFM readback layout =="
python eval\scripts\pack_expected_ofm_for_cnn_v4.py `
  --ofm eval\golden\$BENCH\img0000\golden_final_ofm_uint8_hwc.npy `
  --dse-csv eval\configs\vgg16_like_32_p128_layers_with_wgt_base.csv `
  --out eval\golden\$BENCH\img0000\expected_final_ofm_cnnv4.hex `
  --word-lanes $WORD_LANES `
  --pc-mode2 8 `
  --word-count-out eval\golden\$BENCH\img0000\expected_ofm_words.txt `
  --overwrite


Write-Host "== 7. Sanity check final golden distribution =="
python -c "import numpy as np; p=r'eval\golden\vgg16_like_32_p128\img0000\golden_final_ofm_uint8_hwc.npy'; a=np.load(p); u,c=np.unique(a, return_counts=True); print('shape=', a.shape); print('unique_count=', len(u)); print('unique_sample=', list(zip(u[:32].tolist(), c[:32].tolist()))); print('first64=', a.reshape(-1)[:64].tolist())"

Write-Host "Done."
Write-Host "Vivado RTL files: rtl/cnn_ddr_defs.svh, rtl/cnn_layer_desc_pkg.sv,"
Write-Host "  rtl/kv260_cnn_eval_top_fixedcfg_vgg16_like_32_p128_defaults.sv,"
Write-Host "  rtl/kv260_cnn_smoke_top_vgg16_like_32_p128.sv"
Write-Host "Board run: use -word-bytes 8 -wgt-base-word 0x08000 -ofm-base-word 0xF0000"
