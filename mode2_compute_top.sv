module mode2_compute_top #(
  parameter int DATA_W    = 8,
  parameter int PSUM_W    = 32,
  parameter int K_MAX     = 7,
  parameter int PC        = 8,
  parameter int PF        = 4,
  parameter int HOUT_MAX  = 224,
  parameter int WOUT_MAX  = 224,
  parameter int WB_LANES  = PF*PC,
  parameter int WB_ADDR_W = 12
)(
  input  logic clk,
  input  logic rst_n,

  input  logic start,
  input  logic step_en,

  input  logic [15:0] K_cur,
  input  logic [15:0] C_cur,
  input  logic [15:0] F_cur,
  input  logic [15:0] Hout_cur,
  input  logic [15:0] Wout_cur,

  input  logic                     dr_write_en,
  input  logic [$clog2(K_MAX)-1:0] dr_write_row_idx,
  input  logic [PC*DATA_W-1:0]     dr_write_data,

  input  logic                       weight_bank_sel,
  input  logic                       weight_bank_ready,
  output logic                       wb_rd_en,
  output logic                       wb_rd_buf_sel,
  output logic [WB_ADDR_W-1:0]       wb_rd_addr,
  input  logic [WB_LANES*DATA_W-1:0] wb_rd_data,
  input  logic                       wb_rd_valid,

  // --------------------------------------------------------------------------
  // Backward-compatible loop outputs
  // IMPORTANT: out_row/out_col are GLOBAL OFM coordinates.
  // --------------------------------------------------------------------------
  output logic [15:0]              out_row,
  output logic [15:0]              out_col,
  output logic [15:0]              f_group,
  output logic [15:0]              c_group,
  output logic [$clog2(K_MAX)-1:0] ky,
  output logic [$clog2(K_MAX)-1:0] kx,

  // --------------------------------------------------------------------------
  // Explicit contract outputs
  // These make the global/local split visible without breaking the old ports.
  // - out_row_g / out_col_g are GLOBAL OFM coordinates from controller.
  // - No local tile coordinate is generated here; downstream logic must derive:
  //       out_col_l = out_col_g - tile_col_base_g
  // --------------------------------------------------------------------------
  output logic [15:0]              out_row_g,
  output logic [15:0]              out_col_g,

  output logic                     mac_en,
  output logic                     clear_psum,
  output logic                     ce_out_valid,
  output logic                     pass_start_pulse,
  output logic                     group_start_pulse,
  output logic                     row_done_pulse,
  output logic [$clog2(K_MAX)-1:0] row_done_ky,
  output logic                     c_group_done_pulse,
  output logic                     pixel_done_pulse,
  output logic                     f_group_done_pulse,
  output logic                     done,
  output logic                     busy,

  output logic [PC*DATA_W-1:0]     ce_data_out_logic,
  output logic [PF*PC*DATA_W-1:0]  ce_weight_out,
  output logic [PF*PSUM_W-1:0]     ce_mac_data_out,
  output logic                     ce_mac_data_out_valid,
  output logic [PF*PSUM_W-1:0]     relu_data_out,
  output logic                     relu_data_out_valid,
  output logic                     relu_group_start,
  output logic [15:0]              relu_f_base,

  output logic                     ofm_wr_en,
  output logic [15:0]              ofm_wr_row,
  output logic [15:0]              ofm_wr_col,
  output logic [15:0]              ofm_wr_f_base,
  output logic [PF*DATA_W-1:0]     ofm_wr_data,

  // Explicit aliases for OFM write coordinates (global by contract)
  output logic [15:0]              ofm_wr_row_g,
  output logic [15:0]              ofm_wr_col_g
);

  logic        ce_mac_group_start;
  logic [15:0] ce_mac_f_base;

  logic [15:0] ce_out_row_g;
  logic [15:0] ce_out_col_g;

  ce_mode2_top #(
    .DATA_W   (DATA_W),
    .PSUM_W   (PSUM_W),
    .K_MAX    (K_MAX),
    .PC       (PC),
    .PF       (PF),
    .HOUT_MAX (HOUT_MAX),
    .WOUT_MAX (WOUT_MAX),
    .WB_LANES (WB_LANES),
    .WB_ADDR_W(WB_ADDR_W)
  ) u_ce_mode2_top (
    .clk               (clk),
    .rst_n             (rst_n),
    .start             (start),
    .step_en           (step_en),
    .K_cur             (K_cur),
    .C_cur             (C_cur),
    .F_cur             (F_cur),
    .Hout_cur          (Hout_cur),
    .Wout_cur          (Wout_cur),
    .dr_write_en       (dr_write_en),
    .dr_write_row_idx  (dr_write_row_idx),
    .dr_write_data     (dr_write_data),
    .weight_bank_sel   (weight_bank_sel),
    .weight_bank_ready (weight_bank_ready),
    .wb_rd_en          (wb_rd_en),
    .wb_rd_buf_sel     (wb_rd_buf_sel),
    .wb_rd_addr        (wb_rd_addr),
    .wb_rd_data        (wb_rd_data),
    .wb_rd_valid       (wb_rd_valid),
    .out_row           (ce_out_row_g),
    .out_col           (ce_out_col_g),
    .f_group           (f_group),
    .c_group           (c_group),
    .ky                (ky),
    .kx                (kx),
    .mac_en            (mac_en),
    .clear_psum        (clear_psum),
    .out_valid         (ce_out_valid),
    .pass_start_pulse  (pass_start_pulse),
    .group_start_pulse (group_start_pulse),
    .row_done_pulse    (row_done_pulse),
    .row_done_ky       (row_done_ky),
    .c_group_done_pulse(c_group_done_pulse),
    .pixel_done_pulse  (pixel_done_pulse),
    .f_group_done_pulse(f_group_done_pulse),
    .done              (done),
    .busy              (busy),
    .data_out_logic    (ce_data_out_logic),
    .weight_out        (ce_weight_out),
    .mac_data_out      (ce_mac_data_out),
    .mac_data_out_valid(ce_mac_data_out_valid),
    .mac_group_start   (ce_mac_group_start),
    .mac_f_base        (ce_mac_f_base)
  );

  // Export global coordinates explicitly, while preserving the old outputs.
  assign out_row_g = ce_out_row_g;
  assign out_col_g = ce_out_col_g;
  assign out_row   = ce_out_row_g;
  assign out_col   = ce_out_col_g;

  relu_mode2 #(
    .PSUM_W (PSUM_W),
    .PF     (PF)
  ) u_relu_mode2 (
    .clk           (clk),
    .rst_n         (rst_n),
    .data_in       (ce_mac_data_out),
    .data_in_valid (ce_mac_data_out_valid),
    .data_out      (relu_data_out),
    .data_out_valid(relu_data_out_valid)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      relu_group_start <= 1'b0;
      relu_f_base      <= '0;
    end
    else begin
      relu_group_start <= ce_mac_group_start;
      if (ce_mac_data_out_valid)
        relu_f_base <= ce_mac_f_base;
    end
  end

  pooling_mode2 #(
    .DATA_W (PSUM_W),
    .OUT_W  (DATA_W),
    .PF     (PF),
    .W_MAX  (WOUT_MAX),
    .H_MAX  (HOUT_MAX)
  ) u_pooling_mode2 (
    .clk           (clk),
    .rst_n         (rst_n),
    .W_cur         (Wout_cur),
    .H_cur         (Hout_cur),
    .data_in       (relu_data_out),
    .data_in_valid (relu_data_out_valid),
    .in_group_start(relu_group_start),
    .in_f_base     (relu_f_base),
    .ofm_wr_en     (ofm_wr_en),
    .ofm_wr_row    (ofm_wr_row),
    .ofm_wr_col    (ofm_wr_col),
    .ofm_wr_f_base (ofm_wr_f_base),
    .ofm_wr_data   (ofm_wr_data)
  );

  // By contract, pooling/ofm write coordinates are GLOBAL OFM coordinates.
  assign ofm_wr_row_g = ofm_wr_row;
  assign ofm_wr_col_g = ofm_wr_col;

endmodule
