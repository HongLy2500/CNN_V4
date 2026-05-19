# VGG16-like 96x96 P256 fixed-config software preparation flow.
# Run this from the CNN_V4 repository root after copying this package into the repo.
# Edit this path first:
$IMAGENET_VAL_DIR = "D:\sdh\ILSVRC2012_img_val"

$BENCH = "vgg16_like_96_p256"
$WORD_LANES = 16

New-Item -ItemType Directory -Force eval\assets\$BENCH\inputs | Out-Null
New-Item -ItemType Directory -Force eval\assets\$BENCH\weights | Out-Null
New-Item -ItemType Directory -Force eval\golden\$BENCH\img0000 | Out-Null
New-Item -ItemType Directory -Force eval\results\$BENCH | Out-Null

Write-Host "== 1. Prepare ImageNet inputs 96x96 =="
python eval\scripts\prepare_imagenet_inputs.py `
  --imagenet-val-dir $IMAGENET_VAL_DIR `
  --out-dir eval\assets\$BENCH\inputs `
  --height 96 `
  --width 96 `
  --count 5

Write-Host "== 2. Generate deterministic INT8 weights =="
python eval\scripts\gen_vgg16_like_weights.py `
  --layers-csv eval\configs\vgg16_like_96_p256_layers.csv `
  --out-dir eval\assets\$BENCH\weights `
  --pattern deterministic_small `
  --overwrite

Write-Host "== 3. Pack initial IFM for CNN_V4 =="
python eval\scripts\pack_ifm_for_cnn_v4.py `
  --input eval\assets\$BENCH\inputs\img0000\input_uint8_hwc.npy `
  --dse-csv eval\configs\vgg16_like_96_p256_layers.csv `
  --layer-index 0 `
  --out eval\assets\$BENCH\inputs\img0000\ifm_ddr_cnnv4.hex `
  --word-lanes $WORD_LANES `
  --overwrite

Write-Host "== 4. Pack weights for CNN_V4 and emit layer CSV with weight bases =="
python eval\scripts\pack_weights_for_cnn_v4.py `
  --weights-dir eval\assets\$BENCH\weights `
  --dse-csv eval\configs\vgg16_like_96_p256_layers.csv `
  --out eval\assets\$BENCH\weights\all_weights_cnnv4.hex `
  --offsets-csv eval\assets\$BENCH\weights\weight_offsets_cnnv4.csv `
  --dse-out eval\configs\vgg16_like_96_p256_layers_with_wgt_base.csv `
  --ptotal 256 `
  --ddr-word-lanes $WORD_LANES `
  --wgt-base 0x010000 `
  --overwrite

Write-Host "== 5. Generate cnn_layer_desc_pkg.sv =="
python eval\scripts\layer_csv_to_layer_pkg.py `
  --csv eval\configs\vgg16_like_96_p256_layers_with_wgt_base.csv `
  --out cnn_layer_desc_pkg.sv

Write-Host "== 6. Generate PyTorch fixed-point golden output =="
python eval\scripts\gen_fixedpoint_golden.py `
  --descriptor eval\models\vgg16_like_96_p256.json `
  --input eval\assets\$BENCH\inputs\img0000\input_uint8_hwc.npy `
  --weights-dir eval\assets\$BENCH\weights `
  --out-dir eval\golden\$BENCH\img0000 `
  --store-policy saturate_u8 `
  --strict-shapes `
  --overwrite

Write-Host "== 7. Pack expected OFM for CNN_V4 OFM readback layout =="
python eval\scripts\pack_expected_ofm_for_cnn_v4.py `
  --ofm eval\golden\$BENCH\img0000\golden_final_ofm_uint8_hwc.npy `
  --dse-csv eval\configs\vgg16_like_96_p256_layers_with_wgt_base.csv `
  --out eval\golden\$BENCH\img0000\expected_final_ofm_cnnv4.hex `
  --word-lanes $WORD_LANES `
  --pc-mode2 16 `
  --word-count-out eval\golden\$BENCH\img0000\expected_ofm_words.txt `
  --overwrite

Write-Host "Done."
Write-Host "Next: add rtl/cnn_ddr_defs.svh, rtl/kv260_cnn_eval_top_pkgcfg_vgg16_like_96_p256_defaults.sv,"
Write-Host "      rtl/kv260_cnn_smoke_top_vgg16_like_96_p256.sv, and generated cnn_layer_desc_pkg.sv to Vivado."
