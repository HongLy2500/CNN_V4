# HT=4 changelog

This revision changes the VGG16-like 96×96 P256 KV260 package from `HT=3` to `HT=4`.

Updated files:

- `rtl/kv260_cnn_eval_top_pkgcfg_vgg16_like_96_p256_defaults.sv`
- `rtl/kv260_cnn_smoke_top_vgg16_like_96_p256.sv`
- `eval/configs/vgg16_like_96_p256_hw.json`
- `docs/RTL_CHECKLIST.md`
- `README.md`

Layer shapes, modes, PV/PF/PC, input/weight/golden scripts, DDR map, and expected output size are unchanged.
