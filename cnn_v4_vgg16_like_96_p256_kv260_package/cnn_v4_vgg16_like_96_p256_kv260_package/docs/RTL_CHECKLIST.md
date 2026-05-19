# RTL checklist for VGG16-like 96x96 P256 on KV260

Required build/profile values:

```text
DATA_W     = 8
PSUM_W     = 32
PTOTAL     = 256
PV_MAX     = 16
PF_MAX     = 16
PC_MODE2   = 16
PF_MODE2   = 16
H_MAX      = 96
W_MAX      = 96
C_MAX      = 128
F_MAX      = 128
K_MAX      = 3
HT         = 3
CFG_DEPTH  = 16
DDR_WORD_W = 128
AXI_DATA_W = 128
```

Important checks:

1. Padding must be enabled and consistent with descriptor: K=3, stride=1, pad=1.
2. `C_MAX=128` and `F_MAX=128`; config fields of 8 bits are sufficient for this benchmark.
3. `PV_MAX=16` and `PC=16`, so DDR words are 128-bit.
4. Vivado block design must connect a 128-bit AXI data bus.
5. Use the provided `cnn_ddr_defs.svh` DDR map:
   - IFM: 0x000000
   - WGT: 0x010000
   - OFM: 0x020000
6. The weight stream is 57,488 wide words, so WGT region [0x010000,0x020000) is sufficient.
7. Expected final OFM tensor shape is 3x3x128, but OFM readback layout uses 384 wide words
   because final Mode2 stores one channel-major row word per channel/row group.
