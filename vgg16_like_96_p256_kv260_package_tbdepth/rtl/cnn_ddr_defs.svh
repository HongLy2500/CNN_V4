`ifndef CNN_DDR_DEFS_SVH
`define CNN_DDR_DEFS_SVH

// CNN_V4 DDR word-address map for VGG16-like 96x96 P256/PV16 evaluation.
// Address unit is one RTL DDR word, not a byte.
// For this profile: DDR_WORD_W = PV_MAX * DATA_W = 16 * 8 = 128 bits = 16 bytes.
//
// Byte addresses with AXI_DDR_BASE_ADDR=0x70000000:
//   IFM  : 0x70000000 + 0x000000 * 16 = 0x70000000
//   WGT  : 0x70000000 + 0x010000 * 16 = 0x70100000
//   OFM  : 0x70000000 + 0x020000 * 16 = 0x70200000
//   RSVD : 0x70000000 + 0x030000 * 16 = 0x70300000
//
// Weight stream for this benchmark is 57,488 128-bit words, so the WGT region
// [0x010000,0x020000) is sufficient.

`define CNN_DDR_ADDR_W 32

`define DDR_IFM_BASE  32'h000000
`define DDR_WGT_BASE  32'h010000
`define DDR_OFM_BASE  32'h020000
`define DDR_RSVD_BASE 32'h030000

`endif // CNN_DDR_DEFS_SVH
