# RTL checklist for vgg16_like_32_p128

- Use stable `cnn_layer_desc_pkg.sv` with `DIM_W=16`; do not put layer ROM in this package.
- Use stable IFM/OFM buffers unless intentionally testing a RAM-backend change.
- Parameters: PV_MAX=8, PF_MAX=16, PC_MODE2=8, PF_MODE2=16, PTOTAL=128.
- H_MAX=W_MAX=32, C_MAX=F_MAX=64, HT=4.
- OFM_ROW_STRIDE=4, OFM_BANK_DEPTH=128, OFM_LINEAR_DEPTH=8192.
- WGT_DEPTH=512.
- DDR_WORD_W=64 and AXI_DATA_W=64 in this top.
