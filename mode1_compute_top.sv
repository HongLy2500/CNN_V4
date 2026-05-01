module mode1_compute_top #(
  parameter int DATA_W    = 8,
  parameter int PSUM_W    = 32,
  parameter int K_MAX     = 7,
  parameter int W_MAX     = 224,
  parameter int PV_MAX    = 8,
  parameter int PF_MAX    = 8,
  parameter int PTOTAL    = 16,
  parameter int F_MAX     = 128,
  parameter int HOUT_MAX  = 224,
  parameter int WOUT_MAX  = 224,
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
  input  logic [15:0] W_cur,
  input  logic       pool_en,
  input  logic [7:0] Pv_cur,
  input  logic [7:0] Pf_cur,

  input  logic                     dr_write_en,
  input  logic [$clog2(K_MAX)-1:0] dr_write_row_idx,
  input  logic [15:0]              dr_write_x_base,
  input  logic [PV_MAX*DATA_W-1:0] dr_write_data,

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
  output logic                     ce_out_valid,
  output logic                     pass_start_pulse,
  output logic                     row_done_pulse,
  output logic [$clog2(K_MAX)-1:0] row_done_ky,
  output logic                     chan_done_pulse,
  output logic                     f_group_done_pulse,
  output logic                     out_row_done_pulse,
  output logic                     done,
  output logic                     busy,

  output logic [PV_MAX*DATA_W-1:0]      ce_data_out_logic,
  output logic signed [DATA_W-1:0]      ce_weight_out_lane [0:PTOTAL-1],
  output logic signed [PSUM_W-1:0]      ce_psum_out_lane   [0:PTOTAL-1],
  output logic signed [PSUM_W-1:0]      relu_out_lane      [0:PTOTAL-1],

  output logic                          ofm_write_en,
  output logic [15:0]                   ofm_write_filter_base,
  output logic [15:0]                   ofm_write_row,
  output logic [15:0]                   ofm_write_col_base,
  output logic [15:0]                   ofm_write_count,
  output logic signed [DATA_W-1:0]      ofm_write_data [0:PTOTAL-1]
);

  localparam int PV_IDX_W = (PV_MAX <= 1) ? 1 : $clog2(PV_MAX);

  logic ce_done_int, ce_busy_int;
  logic ce_step_en_s;
  logic ce_done_pending_q;
  logic pool_write_seen_q;
  logic pool_output_expected_s;
  logic compute_done_s;
  logic pool_en_eff_s;

  logic                          pool_ofm_write_en_s;
  logic [15:0]                   pool_ofm_write_filter_base_s;
  logic [15:0]                   pool_ofm_write_row_s;
  logic [15:0]                   pool_ofm_write_col_base_s;
  logic [15:0]                   pool_ofm_write_count_s;
  logic signed [DATA_W-1:0]      pool_ofm_write_data_s [0:PTOTAL-1];

  logic                          direct_valid_q;
  logic [15:0]                   direct_row_q;
  logic [15:0]                   direct_col_base_q;
  logic [15:0]                   direct_filter_base_q;
  logic [15:0]                   direct_pv_stride_q;
  logic [15:0]                   direct_pv_count_q;
  logic [15:0]                   direct_pf_count_q;
  logic [15:0]                   direct_emit_idx_q;
  logic signed [PSUM_W-1:0]      direct_relu_lane_q [0:PTOTAL-1];

  logic                          direct_ofm_write_en_s;
  logic [15:0]                   direct_ofm_write_filter_base_s;
  logic [15:0]                   direct_ofm_write_row_s;
  logic [15:0]                   direct_ofm_write_col_base_s;
  logic [15:0]                   direct_ofm_write_count_s;
  logic signed [DATA_W-1:0]      direct_ofm_write_data_s [0:PTOTAL-1];

  logic                          direct_capture_s;
  logic [15:0]                   direct_capture_pv_stride_s;
  logic [15:0]                   direct_capture_pv_count_s;
  logic [15:0]                   direct_capture_pf_count_s;
  logic [15:0]                   direct_capture_filter_base_s;

  function automatic logic signed [DATA_W-1:0] sat_relu_to_data(input logic signed [PSUM_W-1:0] v);
    logic signed [PSUM_W-1:0] max_v;
    begin
      max_v = $signed((1 << (DATA_W-1)) - 1);
      if (v <= '0)
        sat_relu_to_data = '0;
      else if (v > max_v)
        sat_relu_to_data = max_v[DATA_W-1:0];
      else
        sat_relu_to_data = v[DATA_W-1:0];
    end
  endfunction

  assign pool_en_eff_s          = (pool_en !== 1'b0);
  assign pool_output_expected_s = pool_en_eff_s && (F_cur != 0) && (Pf_cur != 0) && (Pv_cur > 1) && (Hout_cur > 1) && (Wout_cur > 1);
  assign compute_done_s         = ce_done_pending_q &&
                                  (pool_en_eff_s ?
                                   ((!pool_output_expected_s) || pool_write_seen_q || pool_ofm_write_en_s) :
                                   (!direct_valid_q && !direct_capture_s));

  assign ce_step_en_s           = step_en && (pool_en_eff_s || !direct_valid_q);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ce_done_pending_q <= 1'b0;
      pool_write_seen_q <= 1'b0;
    end
    else begin
      if (start) begin
        ce_done_pending_q <= 1'b0;
        pool_write_seen_q <= 1'b0;
      end
      else begin
        if (pool_ofm_write_en_s)
          pool_write_seen_q <= 1'b1;

        if (ce_done_int)
          ce_done_pending_q <= 1'b1;

        if (compute_done_s)
          ce_done_pending_q <= 1'b0;
      end
    end
  end

  assign done = compute_done_s;
  assign busy = ce_busy_int || ce_done_pending_q || direct_valid_q;

  ce_mode1_top #(
    .DATA_W   (DATA_W),
    .PSUM_W   (PSUM_W),
    .K_MAX    (K_MAX),
    .W_MAX    (W_MAX),
    .PV_MAX   (PV_MAX),
    .PF_MAX   (PF_MAX),
    .PTOTAL   (PTOTAL),
    .HOUT_MAX (HOUT_MAX),
    .WOUT_MAX (WOUT_MAX),
    .WB_ADDR_W(WB_ADDR_W)
  ) u_ce_mode1_top (
    .clk               (clk),
    .rst_n             (rst_n),
    .start             (start),
    .step_en           (ce_step_en_s),
    .K_cur             (K_cur),
    .C_cur             (C_cur),
    .F_cur             (F_cur),
    .Hout_cur          (Hout_cur),
    .Wout_cur          (Wout_cur),
    .W_cur             (W_cur),
    .Pv_cur            (Pv_cur),
    .Pf_cur            (Pf_cur),
    .dr_write_en       (dr_write_en),
    .dr_write_row_idx  (dr_write_row_idx),
    .dr_write_x_base   (dr_write_x_base),
    .dr_write_data     (dr_write_data),
    .weight_bank_sel   (weight_bank_sel),
    .weight_bank_ready (weight_bank_ready),
    .wb_rd_en          (wb_rd_en),
    .wb_rd_buf_sel     (wb_rd_buf_sel),
    .wb_rd_addr        (wb_rd_addr),
    .wb_rd_base_lane   (wb_rd_base_lane),
    .wb_rd_data        (wb_rd_data),
    .wb_rd_valid       (wb_rd_valid),
    .out_row           (out_row),
    .out_col           (out_col),
    .f_group           (f_group),
    .c_iter            (c_iter),
    .ky                (ky),
    .kx                (kx),
    .mac_en            (mac_en),
    .clear_psum        (clear_psum),
    .out_valid         (ce_out_valid),
    .pass_start_pulse  (pass_start_pulse),
    .row_done_pulse    (row_done_pulse),
    .row_done_ky       (row_done_ky),
    .chan_done_pulse   (chan_done_pulse),
    .f_group_done_pulse(f_group_done_pulse),
    .out_row_done_pulse(out_row_done_pulse),
    .done              (ce_done_int),
    .busy              (ce_busy_int),
    .data_out_logic    (ce_data_out_logic),
    .weight_out_lane   (ce_weight_out_lane),
    .psum_out_lane     (ce_psum_out_lane)
  );

  relu_mode1 #(
    .PSUM_W (PSUM_W),
    .PTOTAL (PTOTAL)
  ) u_relu_mode1 (
    .in_data  (ce_psum_out_lane),
    .out_data (relu_out_lane)
  );

  pooling_mode1 #(
    .DATA_W   (PSUM_W),
    .OUT_W    (DATA_W),
    .PV_MAX   (PV_MAX),
    .PF_MAX   (PF_MAX),
    .PTOTAL   (PTOTAL),
    .F_MAX    (F_MAX),
    .WOUT_MAX (WOUT_MAX)
  ) u_pooling_mode1 (
    .clk                  (clk),
    .rst_n                (rst_n),
    .pool_en              (pool_en_eff_s),
    .Pv_cur               (Pv_cur),
    .Pf_cur               (Pf_cur),
    .F_cur                (F_cur),
    .Wout_cur             (Wout_cur),
    .f_group              (f_group),
    .in_valid             (pool_en_eff_s && ce_out_valid),
    .in_row               (out_row),
    .in_col               (out_col),
    .in_data              (relu_out_lane),
    .ofm_write_en         (pool_ofm_write_en_s),
    .ofm_write_filter_base(pool_ofm_write_filter_base_s),
    .ofm_write_row        (pool_ofm_write_row_s),
    .ofm_write_col_base   (pool_ofm_write_col_base_s),
    .ofm_write_count      (pool_ofm_write_count_s),
    .ofm_write_data       (pool_ofm_write_data_s)
  );

  always_comb begin
    integer i;
    integer pf;
    integer lane_idx;

    direct_ofm_write_en_s          = direct_valid_q;
    direct_ofm_write_filter_base_s = direct_filter_base_q;
    direct_ofm_write_row_s         = direct_row_q;
    direct_ofm_write_col_base_s    = direct_col_base_q + direct_emit_idx_q;
    direct_ofm_write_count_s       = direct_pf_count_q;

    for (i = 0; i < PTOTAL; i++) begin
      direct_ofm_write_data_s[i] = '0;
    end

    for (pf = 0; pf < PTOTAL; pf++) begin
      lane_idx = (pf * direct_pv_stride_q) + direct_emit_idx_q;
      if ((pf < direct_pf_count_q) && (lane_idx >= 0) && (lane_idx < PTOTAL)) begin
        direct_ofm_write_data_s[pf] = sat_relu_to_data(direct_relu_lane_q[lane_idx]);
      end
    end
  end

  always_comb begin
    if (Pv_cur == 0)
      direct_capture_pv_stride_s = 16'd1;
    else
      direct_capture_pv_stride_s = Pv_cur;

    if (out_col >= Wout_cur)
      direct_capture_pv_count_s = 16'd0;
    else if ((Wout_cur - out_col) < direct_capture_pv_stride_s)
      direct_capture_pv_count_s = Wout_cur - out_col;
    else
      direct_capture_pv_count_s = direct_capture_pv_stride_s;

    direct_capture_filter_base_s = f_group * Pf_cur;

    if (direct_capture_filter_base_s >= F_cur)
      direct_capture_pf_count_s = 16'd0;
    else if ((F_cur - direct_capture_filter_base_s) < Pf_cur)
      direct_capture_pf_count_s = F_cur - direct_capture_filter_base_s;
    else
      direct_capture_pf_count_s = Pf_cur;
  end

  assign direct_capture_s = (!pool_en_eff_s) && ce_out_valid &&
                            (direct_capture_pv_count_s != 0) &&
                            (direct_capture_pf_count_s != 0) &&
                            !direct_valid_q;

  always_ff @(posedge clk or negedge rst_n) begin
    integer i;

    if (!rst_n) begin
      direct_valid_q       <= 1'b0;
      direct_row_q         <= '0;
      direct_col_base_q    <= '0;
      direct_filter_base_q <= '0;
      direct_pv_stride_q   <= 16'd1;
      direct_pv_count_q    <= '0;
      direct_pf_count_q    <= '0;
      direct_emit_idx_q    <= '0;
      for (i = 0; i < PTOTAL; i++) begin
        direct_relu_lane_q[i] <= '0;
      end
    end
    else begin
      if (start) begin
        direct_valid_q       <= 1'b0;
        direct_row_q         <= '0;
        direct_col_base_q    <= '0;
        direct_filter_base_q <= '0;
        direct_pv_stride_q   <= 16'd1;
        direct_pv_count_q    <= '0;
        direct_pf_count_q    <= '0;
        direct_emit_idx_q    <= '0;
      end
      else begin
        if (direct_valid_q) begin
          if ((direct_emit_idx_q + 16'd1) < direct_pv_count_q) begin
            direct_emit_idx_q <= direct_emit_idx_q + 1'b1;
          end
          else begin
            direct_valid_q    <= 1'b0;
            direct_emit_idx_q <= '0;
          end
        end
        else if (direct_capture_s) begin
          direct_valid_q       <= 1'b1;
          direct_row_q         <= out_row;
          direct_col_base_q    <= out_col;
          direct_filter_base_q <= direct_capture_filter_base_s;
          direct_pv_stride_q   <= direct_capture_pv_stride_s;
          direct_pv_count_q    <= direct_capture_pv_count_s;
          direct_pf_count_q    <= direct_capture_pf_count_s;
          direct_emit_idx_q    <= '0;
          for (i = 0; i < PTOTAL; i++) begin
            direct_relu_lane_q[i] <= relu_out_lane[i];
          end
        end
      end
    end
  end

  always_comb begin
    integer i;

    if (pool_en_eff_s) begin
      ofm_write_en          = pool_ofm_write_en_s;
      ofm_write_filter_base = pool_ofm_write_filter_base_s;
      ofm_write_row         = pool_ofm_write_row_s;
      ofm_write_col_base    = pool_ofm_write_col_base_s;
      ofm_write_count       = pool_ofm_write_count_s;
      for (i = 0; i < PTOTAL; i++) begin
        ofm_write_data[i] = pool_ofm_write_data_s[i];
      end
    end
    else begin
      ofm_write_en          = direct_ofm_write_en_s;
      ofm_write_filter_base = direct_ofm_write_filter_base_s;
      ofm_write_row         = direct_ofm_write_row_s;
      ofm_write_col_base    = direct_ofm_write_col_base_s;
      ofm_write_count       = direct_ofm_write_count_s;
      for (i = 0; i < PTOTAL; i++) begin
        ofm_write_data[i] = direct_ofm_write_data_s[i];
      end
    end
  end

endmodule
