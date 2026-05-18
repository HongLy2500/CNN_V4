`ifndef CNN_DDR_DEFS_SVH
`define CNN_DDR_DEFS_SVH

//==================================================
// Global DDR model
// Address unit = 1 DDR word, not byte address
//==================================================
`define CNN_DDR_ADDR_W 20
`define CNN_DDR_DEPTH  (1 << `CNN_DDR_ADDR_W)
`define CNN_DDR_INIT_HEX ""

//==================================================
// DDR region map for VGG16 conv-only regression
// Total 20-bit word address space: 0x00000 .. 0xFFFFF
//==================================================

// IFM input image / initial feature map.
// Full 224x224x3 with PV=16 needs only 224*ceil(224/16)*3 = 9408 words.
`define DDR_IFM_BASE 20'h00000
`define DDR_IFM_SIZE 20'h08000

// Weights.
// VGG16 conv13 with PF=16 needs about 0xE076C words.
// Allocate 0xE8000 words: 0x08000 .. 0xEFFFF.
`define DDR_WGT_BASE 20'h08000
`define DDR_WGT_SIZE 20'hE8000

// Final OFM output.
// Put this after the large weight region.
// 0xF0000 .. 0xFFFFF = 0x10000 words.
`define DDR_OFM_BASE 20'hF0000
`define DDR_OFM_SIZE 20'h10000

// Reserved region is disabled/zero-sized in 20-bit map.
`define DDR_RSVD_BASE 20'h00000
`define DDR_RSVD_SIZE 20'h00000

`endif