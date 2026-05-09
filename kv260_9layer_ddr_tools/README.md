# KV260 9-layer DDR image tools

These scripts generate the same IFM/weight DDR contents as `init_mem()` in:

`tb_cnn_top_9layer_m1_dcp_efficientnet_b0_tablevi_fullscale_with_expected_compare.sv`

## Generate images on Windows/PC

```powershell
python D:\path\to\gen_kv260_9layer_ddr_images.py --out-dir D:\sdh\CNN_V4\kv260_9layer_ddr_images
```

Optional, if you have `expected_final_ofm.hex`:

```powershell
python D:\path\to\gen_kv260_9layer_ddr_images.py `
  --out-dir D:\sdh\CNN_V4\kv260_9layer_ddr_images `
  --expected-hex D:\sdh\CNN_V4\expected_final_ofm.hex
```

## Load images into DDR using XSCT

After `psu_init`, `psu_post_config`, and programming the `.bit + .ltx`:

```tcl
source D:/sdh/CNN_V4/kv260_9layer_ddr_images/load_9layer_ddr_images.xsct.tcl
```

Then use VIO:

```text
soft_reset_n = 0, run = 0, abort = 0
soft_reset_n = 1
wait cfg_done = 1
run = 1
run = 0
wait done pulse, error = 0
```

Read first OFM words:

```tcl
source D:/sdh/CNN_V4/kv260_9layer_ddr_images/read_9layer_ofm_head.xsct.tcl
```

## Important physical addresses

Because this 9-layer top uses `DDR_WORD_W=1024` bits, one RTL DDR word is 128 bytes.

```text
IFM physical base = 0x70000000
WGT physical base = 0x70400000
OFM physical base = 0x70800000
```

Do not use the old small-smoke `0x70020000` / `0x70040000` addresses for this top.
