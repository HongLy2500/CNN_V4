package cnn_layer_desc_pkg;
  localparam int DIM_W      = 16;
  localparam int K_W        = 4;
  localparam int PV_W       = 8;
  localparam int PF1_W      = 8;
  localparam int PC2_W      = 8;
  localparam int PF2_W      = 8;
  localparam int ADDR_W     = 32;
  localparam int STRIDE_W   = 2;
  localparam int PAD_W      = 4;
  localparam int POOL_W     = 2;
  localparam int LAYER_ID_W = 8;

  typedef enum logic [0:0] {
    MODE1 = 1'b0,
    MODE2 = 1'b1
  } layer_mode_e;

  typedef struct packed {
    logic [LAYER_ID_W-1:0] layer_id;
    layer_mode_e           mode;
    logic [DIM_W-1:0]      h_in;
    logic [DIM_W-1:0]      w_in;
    logic [DIM_W-1:0]      c_in;
    logic [DIM_W-1:0]      f_out;
    logic [K_W-1:0]        k;
    logic [DIM_W-1:0]      h_out;
    logic [DIM_W-1:0]      w_out;
    logic [PV_W-1:0]       pv_m1;
    logic [PF1_W-1:0]      pf_m1;
    logic [PC2_W-1:0]      pc_m2;
    logic [PF2_W-1:0]      pf_m2;
    logic [STRIDE_W-1:0]   conv_stride;
    logic [PAD_W-1:0]      pad_top;
    logic [PAD_W-1:0]      pad_bottom;
    logic [PAD_W-1:0]      pad_left;
    logic [PAD_W-1:0]      pad_right;
    logic                  relu_en;
    logic                  pool_en;
    logic [POOL_W-1:0]     pool_k;
    logic [POOL_W-1:0]     pool_stride;
    logic [ADDR_W-1:0]     ifm_ddr_base;
    logic [ADDR_W-1:0]     wgt_ddr_base;
    logic [ADDR_W-1:0]     ofm_ddr_base;
    logic                  first_layer;
    logic                  last_layer;
  } layer_desc_t;
endpackage
