# KV260 reduced 9-layer smoke tools

Files:
- `kv260_cnn_smoke_top_9layer_reduced_kv260.sv`
- `gen_kv260_9layer_reduced_ddr_images.py`

This reduced smoke keeps 9 layers and Mode 1, but uses:
- PTOTAL=16
- PV_MAX=4
- PF_MAX=4
- H_MAX/W_MAX=144
- C_MAX/F_MAX=16
- DDR_WORD_W=32
- AXI_DATA_W=32

DDR physical addresses:
- IFM: 0x70000000
- WGT: 0x70020000
- OFM: 0x70040000
