`ifndef CNN_DDR_DEFS_SVH
`define CNN_DDR_DEFS_SVH

//==================================================
// Global DDR model
// Address unit = 1 DDR word
// Not byte address
//==================================================
`define CNN_DDR_ADDR_W   20
`define CNN_DDR_DEPTH    (1 << `CNN_DDR_ADDR_W)
`define CNN_DDR_INIT_HEX ""

//==================================================
// DDR region map (word address space)
//==================================================
`define DDR_IFM_BASE     20'h00000
`define DDR_IFM_SIZE     20'h08000

`define DDR_WGT_BASE     20'h08000
`define DDR_WGT_SIZE     20'h08000

`define DDR_OFM_BASE     20'h10000
`define DDR_OFM_SIZE     20'h08000

`define DDR_RSVD_BASE    20'h18000
`define DDR_RSVD_SIZE    20'h08000

`endif