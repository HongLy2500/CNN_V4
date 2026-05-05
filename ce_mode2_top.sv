module ce_mode2_top #(
  parameter int DATA_W    = 8,
  parameter int PSUM_W    = 32,
  parameter int K_MAX     = 7,
  parameter int PC        = 8,
  parameter int PF        = 4,
  parameter int HOUT_MAX  = 224,
  parameter int WOUT_MAX  = 224,
  // Physical word width shared with weight_buffer.WORD_LANES.
  // Mode 2 consumes the low PF*PC logical lanes from this physical word.
  parameter int WB_LANES  = 32,
  parameter int WB_ADDR_W = 12
)(
  input  logic clk,
  input  logic rst_n,

  input  logic start,
  input  logic step_en,

  input  logic [3:0] K_cur,
  input  logic [7:0] C_cur,
  input  logic [7:0] F_cur,
  input  logic [15:0] Hout_cur,
  input  logic [15:0] Wout_cur,

  input  logic                     dr_write_en,
  input  logic [$clog2(K_MAX)-1:0] dr_write_row_idx,
  input  logic [PC*DATA_W-1:0]     dr_write_data,

  input  logic                     weight_bank_sel,
  input  logic                     weight_bank_ready,
  output logic                     wb_rd_en,
  output logic                     wb_rd_buf_sel,
  output logic [WB_ADDR_W-1:0]     wb_rd_addr,
  input  logic [WB_LANES*DATA_W-1:0] wb_rd_data,
  input  logic                     wb_rd_valid,

  output logic [15:0]              out_row,
  output logic [15:0]              out_col,
  output logic [15:0]              f_group,
  output logic [15:0]              c_group,
  output logic [$clog2(K_MAX)-1:0] ky,
  output logic [$clog2(K_MAX)-1:0] kx,

  output logic                     mac_en,
  output logic                     clear_psum,
  output logic                     out_valid,
  output logic                     pass_start_pulse,
  output logic                     group_start_pulse,
  output logic                     row_done_pulse,
  output logic [$clog2(K_MAX)-1:0] row_done_ky,
  output logic                     c_group_done_pulse,
  output logic                     pixel_done_pulse,
  output logic                     f_group_done_pulse,
  output logic                     done,
  output logic                     busy,

  output logic [PC*DATA_W-1:0]     data_out_logic,
  output logic [PF*PC*DATA_W-1:0]  weight_out,
  output logic [PF*PSUM_W-1:0]     mac_data_out,
  output logic                     mac_data_out_valid,
  output logic                     mac_group_start,
  output logic [15:0]              mac_f_base
);

  logic [15:0] f_base_cur;
  logic        weight_write_en;
  logic [PF*PC*DATA_W-1:0] weight_write_data;

  always_comb begin
    f_base_cur = f_group * PF;
  end

  ce_controller_mode2 #(
    .K_MAX    (K_MAX),
    .HOUT_MAX (HOUT_MAX),
    .WOUT_MAX (WOUT_MAX),
    .PC       (PC),
    .PF       (PF)
  ) u_ce_controller_mode2 (
    .clk               (clk),
    .rst_n             (rst_n),
    .start             (start),
    .step_en           (step_en),
    .K_cur             (K_cur),
    .C_cur             (C_cur),
    .F_cur             (F_cur),
    .Hout_cur          (Hout_cur),
    .Wout_cur          (Wout_cur),
    .out_row           (out_row),
    .out_col           (out_col),
    .f_group           (f_group),
    .c_group           (c_group),
    .ky                (ky),
    .kx                (kx),
    .mac_en            (mac_en),
    .clear_psum        (clear_psum),
    .out_valid         (out_valid),
    .pass_start_pulse  (pass_start_pulse),
    .group_start_pulse (group_start_pulse),
    .row_done_pulse    (row_done_pulse),
    .row_done_ky       (row_done_ky),
    .c_group_done_pulse(c_group_done_pulse),
    .pixel_done_pulse  (pixel_done_pulse),
    .f_group_done_pulse(f_group_done_pulse),
    .done              (done),
    .busy              (busy)
  );

  data_register_mode2 #(
    .K_MAX  (K_MAX),
    .DATA_W (DATA_W),
    .PC     (PC)
  ) u_data_register_mode2 (
    .clk          (clk),
    .rst_n        (rst_n),
    .K_cur        (K_cur),
    .write_en     (dr_write_en),
    .write_row_idx(dr_write_row_idx),
    .write_data   (dr_write_data),
    .read_row_idx (ky),
    .data_out     (data_out_logic)
  );

  weight_read_ctrl_mode2 #(
    .DATA_W   (DATA_W),
    .PC       (PC),
    .PF       (PF),
    .WB_LANES (WB_LANES),
    .WB_ADDR_W(WB_ADDR_W)
  ) u_weight_read_ctrl_mode2 (
    .clk             (clk),
    .rst_n           (rst_n),
    .K_cur           (K_cur),
    .C_cur           (C_cur),
    .F_cur           (F_cur),
    .Hout_cur        (Hout_cur),
    .Wout_cur        (Wout_cur),
    .start           (start),
    .pass_start_pulse(pass_start_pulse),
    .mac_en          (mac_en),
    .out_valid       (out_valid),
    .f_group         (f_group),
    .out_row         (out_row),
    .out_col         (out_col),
    .wb_bank_sel     (weight_bank_sel),
    .wb_bank_ready   (weight_bank_ready),
    .wb_rd_en        (wb_rd_en),
    .wb_rd_buf_sel   (wb_rd_buf_sel),
    .wb_rd_addr      (wb_rd_addr),
    .wb_rd_data      (wb_rd_data),
    .wb_rd_valid     (wb_rd_valid),
    .weight_write_en (weight_write_en),
    .weight_write_data(weight_write_data)
  );

  weight_register_mode2 #(
    .DATA_W (DATA_W),
    .PC     (PC),
    .PF     (PF)
  ) u_weight_register_mode2 (
    .clk       (clk),
    .rst_n     (rst_n),
    .write_en  (weight_write_en),
    .write_data(weight_write_data),
    .weight_out(weight_out)
  );

  mac_array_mode2 #(
    .DATA_W (DATA_W),
    .PSUM_W (PSUM_W),
    .PC     (PC),
    .PF     (PF)
  ) u_mac_array_mode2 (
    .clk           (clk),
    .rst_n         (rst_n),
    .mac_en        (mac_en),
    .out_valid     (out_valid),
    .data_in       (data_out_logic),
    .weight_in     (weight_out),
    .data_out      (mac_data_out),
    .data_out_valid(mac_data_out_valid)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      mac_group_start <= 1'b0;
      mac_f_base      <= '0;
    end
    else begin
      mac_group_start <= group_start_pulse;
      if (out_valid)
        mac_f_base <= f_base_cur;
    end
  end

endmodule
