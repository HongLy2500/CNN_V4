module ce_mode1_top #(
  parameter int DATA_W     = 8,
  parameter int PSUM_W     = 32,
  parameter int K_MAX      = 7,
  parameter int W_MAX      = 224,
  parameter int PV_MAX     = 8,
  parameter int PF_MAX     = 8,
  parameter int PTOTAL     = 16,
  parameter int HOUT_MAX   = 224,
  parameter int WOUT_MAX   = 224,
  // Mode-1 weight path:
  // - weight_buffer stores physical PTOTAL-lane words per address
  // - the mode-1 read port returns one Pf-wide logical chunk per cycle
  parameter int WB_ADDR_W  = 12
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
  input  logic [15:0] W_cur,
  input  logic [7:0] Pv_cur,
  input  logic [7:0] Pf_cur,

  input  logic                     dr_write_en,
  input  logic [$clog2(K_MAX)-1:0] dr_write_row_idx,
  input  logic [15:0]              dr_write_x_base,
  // Mode-1 data-register write path is Pv-wide, not Ptotal-wide.
  input  logic [PV_MAX*DATA_W-1:0] dr_write_data,

  // Read control to weight_buffer
  input  logic                     weight_bank_sel,
  input  logic                     weight_bank_ready,
  output logic                     wb_rd_en,
  output logic                     wb_rd_buf_sel,
  output logic [WB_ADDR_W-1:0]     wb_rd_addr,
  output logic [($clog2(PTOTAL) > 0 ? $clog2(PTOTAL) : 1)-1:0] wb_rd_base_lane,
  input  logic [PF_MAX*DATA_W-1:0] wb_rd_data,
  input  logic                     wb_rd_valid,

  output logic [15:0]              out_row,
  output logic [15:0]              out_col,
  output logic [15:0]              f_group,
  output logic [15:0]              c_iter,
  output logic [$clog2(K_MAX)-1:0] ky,
  output logic [$clog2(K_MAX)-1:0] kx,

  output logic                     mac_en,
  output logic                     clear_psum,
  output logic                     out_valid,
  output logic                     pass_start_pulse,
  output logic                     row_done_pulse,
  output logic [$clog2(K_MAX)-1:0] row_done_ky,
  output logic                     chan_done_pulse,
  output logic                     f_group_done_pulse,
  output logic                     out_row_done_pulse,
  output logic                     done,
  output logic                     busy,

  output logic [PV_MAX*DATA_W-1:0] data_out_logic,
  output logic signed [DATA_W-1:0] weight_out_lane [0:PTOTAL-1],
  output logic signed [PSUM_W-1:0] psum_out_lane [0:PTOTAL-1]
);

  logic                     weight_load_en;
  logic                     weight_clear;
  logic signed [DATA_W-1:0] weight_in_logic [0:PF_MAX-1];

  // Internal timing-aligned control and handshake
  logic                     ctrl_step_en;
  logic                     ctrl_mac_en;
  logic                     ctrl_clear_psum;
  logic                     ctrl_out_valid;
  logic                     ctrl_pass_start_pulse;
  logic                     ctrl_row_done_pulse;
  logic [$clog2(K_MAX)-1:0] ctrl_row_done_ky;
  logic                     ctrl_chan_done_pulse;
  logic                     ctrl_f_group_done_pulse;
  logic                     ctrl_out_row_done_pulse;
  logic                     ctrl_done;
  logic                     ctrl_busy;

  logic                     data_ready_q;
  logic [1:0]               data_warmup_q;
  logic                     weight_valid_q;
  logic                     weight_req_inflight_q;
  logic                     wb_bank_ready_sync;

  logic                     wb_rd_en_i;
  logic                     wb_rd_buf_sel_i;
  logic [WB_ADDR_W-1:0]     wb_rd_addr_i;
  logic [($clog2(PTOTAL) > 0 ? $clog2(PTOTAL) : 1)-1:0] wb_rd_base_lane_i;

  // Data path warm-up: IFM read + data-register write/visibility take two cycles
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      data_ready_q  <= 1'b0;
      data_warmup_q <= 2'd0;
    end
    else begin
      if (start || ctrl_pass_start_pulse) begin
        data_ready_q  <= 1'b0;
        data_warmup_q <= 2'd2;
      end
      else if (data_warmup_q != 0) begin
        data_warmup_q <= data_warmup_q - 1'b1;
        if (data_warmup_q == 2'd1)
          data_ready_q <= 1'b1;
      end
    end
  end

  // Weight path bookkeeping: allow only one in-flight bundle and one active
  // bundle. A new bundle can be requested on the same cycle the current one is
  // consumed by MAC, but not earlier.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      weight_valid_q        <= 1'b0;
      weight_req_inflight_q <= 1'b0;
    end
    else begin
      if (start) begin
        weight_valid_q        <= 1'b0;
        weight_req_inflight_q <= 1'b0;
      end
      else begin
        if (wb_rd_en_i)
          weight_req_inflight_q <= 1'b1;
        if (weight_load_en) begin
          weight_req_inflight_q <= 1'b0;
          weight_valid_q        <= 1'b1;
        end
        else if (ctrl_step_en) begin
          weight_valid_q <= 1'b0;
        end
      end
    end
  end

  assign ctrl_step_en      = step_en & data_ready_q & weight_valid_q;
  // At a new layer/sweep start, allow the first weight request even if
  // weight_valid_q is still high from the previous layer in this
  // combinational cycle. Otherwise the one-cycle start pulse can be missed.
  assign wb_bank_ready_sync =
      weight_bank_ready & ~weight_req_inflight_q & (start | ~weight_valid_q | ctrl_step_en);

  ce_controller_mode1 #(
    .K_MAX    (K_MAX),
    .HOUT_MAX (HOUT_MAX),
    .WOUT_MAX (WOUT_MAX)
  ) u_ce_controller_mode1 (
    .clk               (clk),
    .rst_n             (rst_n),
    .start             (start),
    .step_en           (ctrl_step_en),
    .K_cur             (K_cur),
    .C_cur             (C_cur),
    .F_cur             (F_cur),
    .Hout_cur          (Hout_cur),
    .Wout_cur          (Wout_cur),
    .Pv_cur            (Pv_cur),
    .Pf_cur            (Pf_cur),
    .out_row           (out_row),
    .out_col           (out_col),
    .f_group           (f_group),
    .c_iter            (c_iter),
    .ky                (ky),
    .kx                (kx),
    .mac_en            (ctrl_mac_en),
    .clear_psum        (ctrl_clear_psum),
    .out_valid         (ctrl_out_valid),
    .pass_start_pulse  (ctrl_pass_start_pulse),
    .row_done_pulse    (ctrl_row_done_pulse),
    .row_done_ky       (ctrl_row_done_ky),
    .chan_done_pulse   (ctrl_chan_done_pulse),
    .f_group_done_pulse(ctrl_f_group_done_pulse),
    .out_row_done_pulse(ctrl_out_row_done_pulse),
    .done              (ctrl_done),
    .busy              (ctrl_busy)
  );

  data_register_mode1 #(
    .K_MAX   (K_MAX),
    .W_MAX   (W_MAX),
    .DATA_W  (DATA_W),
    .PV_MAX  (PV_MAX),
    .PTOTAL  (PTOTAL)
  ) u_data_register_mode1 (
    .clk           (clk),
    .rst_n         (rst_n),
    .K_cur         (K_cur),
    .W_cur         (W_cur),
    .Pv_cur        (Pv_cur),
    .write_en      (dr_write_en),
    .write_row_idx (dr_write_row_idx),
    .write_x_base  (dr_write_x_base),
    .write_data    (dr_write_data),
    .ky            (ky),
    .kx            (kx),
    .out_col       (out_col),
    .data_out_logic(data_out_logic)
  );

  weight_read_ctrl_mode1 #(
    .DATA_W   (DATA_W),
    .PF_MAX   (PF_MAX),
    .PTOTAL   (PTOTAL),
    .WB_ADDR_W(WB_ADDR_W)
  ) u_weight_read_ctrl_mode1 (
    .clk             (clk),
    .rst_n           (rst_n),
    .K_cur           (K_cur),
    .C_cur           (C_cur),
    .F_cur           (F_cur),
    .Pv_cur          (Pv_cur),
    .Pf_cur          (Pf_cur),
    .Wout_cur        (Wout_cur),
    .start           (start),
    .pass_start_pulse(ctrl_pass_start_pulse),
    .consume_en      (ctrl_step_en),
    .out_valid       (ctrl_out_valid),
    .f_group         (f_group),
    .out_col         (out_col),
    .wb_bank_sel     (weight_bank_sel),
    .wb_bank_ready   (wb_bank_ready_sync),
    .wb_rd_en        (wb_rd_en_i),
    .wb_rd_buf_sel   (wb_rd_buf_sel_i),
    .wb_rd_addr      (wb_rd_addr_i),
    .wb_rd_base_lane (wb_rd_base_lane_i),
    .wb_rd_data      (wb_rd_data),
    .wb_rd_valid     (wb_rd_valid),
    .weight_load_en  (weight_load_en),
    .weight_clear    (weight_clear),
    .weight_in_logic (weight_in_logic)
  );

  weight_register_mode1 #(
    .DATA_W (DATA_W),
    .PF_MAX (PF_MAX),
    .PTOTAL (PTOTAL)
  ) u_weight_register_mode1 (
    .clk            (clk),
    .rst_n          (rst_n),
    .Pv_cur         (Pv_cur),
    .Pf_cur         (Pf_cur),
    .load_en        (weight_load_en),
    .clear          (weight_clear),
    .weight_in_logic(weight_in_logic),
    .weight_out_lane(weight_out_lane)
  );

  mac_array_mode1 #(
    .DATA_W (DATA_W),
    .PSUM_W (PSUM_W),
    .PV_MAX (PV_MAX),
    .PTOTAL (PTOTAL)
  ) u_mac_array_mode1 (
    .clk           (clk),
    .rst_n         (rst_n),
    .Pv_cur        (Pv_cur),
    .Pf_cur        (Pf_cur),
    .mac_en        (ctrl_step_en),
    .clear_psum    (ctrl_clear_psum),
    .data_in_logic (data_out_logic),
    .weight_in_lane(weight_out_lane),
    .psum_out_lane (psum_out_lane)
  );

  assign wb_rd_en            = wb_rd_en_i;
  assign wb_rd_buf_sel       = wb_rd_buf_sel_i;
  assign wb_rd_addr          = wb_rd_addr_i;
  assign wb_rd_base_lane     = wb_rd_base_lane_i;

  assign mac_en              = ctrl_step_en;
  assign clear_psum          = ctrl_clear_psum;
  assign out_valid           = ctrl_out_valid;
  assign pass_start_pulse    = ctrl_pass_start_pulse;
  assign row_done_pulse      = ctrl_row_done_pulse;
  assign row_done_ky         = ctrl_row_done_ky;
  assign chan_done_pulse     = ctrl_chan_done_pulse;
  assign f_group_done_pulse  = ctrl_f_group_done_pulse;
  assign out_row_done_pulse  = ctrl_out_row_done_pulse;
  assign done                = ctrl_done;
  assign busy                = ctrl_busy;

endmodule
