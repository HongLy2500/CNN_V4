package cnn_layer_desc_pkg;

  // ============================================================
  // Tham số giả định tham khảo
  // Có thể chỉnh lại theo hệ thống thực tế
  // ============================================================
  localparam int DIM_W        = 16;  // H, W, C, F, Hout, Wout
  localparam int K_W          = 4;   // kernel size K
  localparam int PV_W         = 8;   // Pv_mode1
  localparam int PF1_W        = 8;   // Pf_mode1
  localparam int PC2_W        = 8;   // Pc_mode2
  localparam int PF2_W        = 8;   // Pf_mode2
  localparam int ADDR_W       = 32;  // DDR base address
  localparam int STRIDE_W     = 2;   // stride
  localparam int PAD_W        = 4;   // padding
  localparam int POOL_W       = 2;   // pool size / stride nếu cần
  localparam int LAYER_ID_W   = 8;

  // ============================================================
  // Mode của layer
  // 0: mode 1
  // 1: mode 2
  // ============================================================
  typedef enum logic [0:0] {
    MODE1 = 1'b0,
    MODE2 = 1'b1
  } layer_mode_e;

  // ============================================================
  // Descriptor của 1 layer
  // ============================================================
  typedef struct packed {
    // ----- Thông tin nhận dạng -----
    logic [LAYER_ID_W-1:0] layer_id;
    layer_mode_e           mode;

    // ----- Kích thước IFM / OFM / kernel -----
    logic [DIM_W-1:0] h_in;      // IFM height
    logic [DIM_W-1:0] w_in;      // IFM width
    logic [DIM_W-1:0] c_in;      // IFM channels

    logic [DIM_W-1:0] f_out;     // OFM channels / filters
    logic [K_W-1:0]   k;         // kernel size

    logic [DIM_W-1:0] h_out;     // OFM height
    logic [DIM_W-1:0] w_out;     // OFM width

    // ----- Parallelism runtime -----
    // mode 1
    logic [PV_W-1:0]  pv_m1;
    logic [PF1_W-1:0] pf_m1;

    // mode 2
    logic [PC2_W-1:0] pc_m2;
    logic [PF2_W-1:0] pf_m2;

    // ----- CNN options -----
    logic [STRIDE_W-1:0] conv_stride;
    logic [PAD_W-1:0]    pad_top;
    logic [PAD_W-1:0]    pad_bottom;
    logic [PAD_W-1:0]    pad_left;
    logic [PAD_W-1:0]    pad_right;

    logic                relu_en;
    logic                pool_en;
    logic [POOL_W-1:0]   pool_k;
    logic [POOL_W-1:0]   pool_stride;

    // ----- Địa chỉ DDR -----
    logic [ADDR_W-1:0] ifm_ddr_base;
    logic [ADDR_W-1:0] wgt_ddr_base;
    logic [ADDR_W-1:0] ofm_ddr_base;

    // ----- Cờ phụ trợ cho scheduler -----
    logic first_layer;
    logic last_layer;
  } layer_desc_t;

endpackage