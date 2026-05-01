module pooling_mode1 #(
  parameter int DATA_W   = 32,
  parameter int OUT_W    = DATA_W,
  parameter int PV_MAX   = 8,
  parameter int PF_MAX   = 8,
  parameter int PTOTAL   = 16,
  parameter int F_MAX    = 128,
  parameter int WOUT_MAX = 224
)(
  input  logic clk,
  input  logic rst_n,
  input  logic pool_en,

  // =====================================================
  // Runtime config
  // =====================================================
  // When pool_en=0 this module is completely idle. The no-pool
  // bypass path is handled outside this module by mode1_compute_top.
  input  logic [7:0] Pv_cur,
  input  logic [7:0] Pf_cur,
  input  logic [7:0] F_cur,
  input  logic [15:0] Wout_cur,
  input  logic [7:0] f_group,

  // =====================================================
  // Input stream from relu_mode1
  // Each in_valid corresponds to one block:
  //   - row      = in_row
  //   - col base = in_col
  //   - filters  = current f_group
  //
  // Lane mapping:
  //   lane = pf * Pv_cur + pv
  // =====================================================
  input  logic                     in_valid,
  input  logic [15:0]              in_row,
  input  logic [15:0]              in_col,
  input  logic signed [DATA_W-1:0] in_data [0:PTOTAL-1],

  // =====================================================
  // Output write interface to OFM buffer
  // For maxpool 2x2 stride 2:
  //   pooled horizontal count = Pv_cur / 2
  //   pooled total count      = Pf_cur * (Pv_cur/2)
  //
  // Output lane packing:
  //   out_lane = pf * (Pv_cur/2) + pool_x
  //
  // NOTE:
  //   OUT_W may be smaller than DATA_W. In that case the pooled
  //   result is saturated to signed OUT_W range before being driven
  //   out. With ReLU in front, this effectively becomes [0, 2^(OUT_W-1)-1]
  //   for signed outputs.
  // =====================================================
  output logic                     ofm_write_en,
  output logic [15:0]              ofm_write_filter_base,
  output logic [15:0]              ofm_write_row,
  output logic [15:0]              ofm_write_col_base,
  output logic [15:0]              ofm_write_count,
  output logic signed [OUT_W-1:0]  ofm_write_data [0:PTOTAL-1]
);

  localparam int signed OUT_MAX = (1 <<< (OUT_W-1)) - 1;
  localparam int signed OUT_MIN = -(1 <<< (OUT_W-1));

  // =====================================================
  // Two-row buffer for all output filters
  // row_buf[row_slot][filter][col]
  // row_slot = in_row[0]
  // =====================================================
  logic signed [DATA_W-1:0] row_buf [0:1][0:F_MAX-1][0:WOUT_MAX-1];


  function automatic logic signed [OUT_W-1:0] sat_to_out(
    input logic signed [DATA_W-1:0] din
  );
    integer signed din_i;
    integer signed sat_i;
    begin
      din_i = din;

      if (din_i > OUT_MAX)
        sat_i = OUT_MAX;
      else if (din_i < OUT_MIN)
        sat_i = OUT_MIN;
      else
        sat_i = din_i;

      sat_to_out = sat_i[OUT_W-1:0];
    end
  endfunction

  // =====================================================
  // Write current ReLU block into row buffer
  // =====================================================
  always_ff @(posedge clk or negedge rst_n) begin
    int pf;
    int pv;
    int filter_idx;
    int top_col0;
    int lane_idx;

    if (!rst_n) begin
      // optional clear omitted for large memories
    end
    else if (pool_en && in_valid) begin
      for (pf = 0; pf < PF_MAX; pf++) begin
        if (pf < Pf_cur) begin
          filter_idx = f_group * Pf_cur + pf;

          if (filter_idx < F_cur && filter_idx < F_MAX) begin
            for (pv = 0; pv < PV_MAX; pv++) begin
              if (pv < Pv_cur) begin
                top_col0 = in_col + pv;
                lane_idx = pf * Pv_cur + pv;

                if (top_col0 < Wout_cur && top_col0 < WOUT_MAX) begin
                  if (lane_idx < PTOTAL) begin
                    row_buf[in_row[0]][filter_idx][top_col0] <= in_data[lane_idx];
                  end
                end
              end
            end
          end
        end
      end
    end
  end

  // =====================================================
  // Pooling output generation
  // Pool only when current row is odd:
  //   use previous even row from row_buf[row_slot=0]
  //   use current odd row from in_data
  // =====================================================
  always_comb begin
    int i;
    int pf;
    int pv;
    int filter_idx;
    int top_col0;
    int top_col1;
    int out_lane;
    int pooled_pv;
    int valid_count;
    int lane0_idx;
    int lane1_idx;
    logic signed [DATA_W-1:0] top0, top1, bot0, bot1;
    logic signed [DATA_W-1:0] max_top, max_bot, max_all;

    ofm_write_en          = 1'b0;
    ofm_write_filter_base = f_group * Pf_cur;
    ofm_write_row         = in_row >> 1;
    ofm_write_col_base    = in_col >> 1;
    ofm_write_count       = 16'd0;
    pooled_pv             = 0;
    valid_count           = 0;
    top0                  = '0;
    top1                  = '0;
    bot0                  = '0;
    bot1                  = '0;
    max_top               = '0;
    max_bot               = '0;
    max_all               = '0;

    for (i = 0; i < PTOTAL; i++) begin
      ofm_write_data[i] = '0;
    end

    if (pool_en && in_valid && in_row[0]) begin
      pooled_pv = Pv_cur >> 1;

      for (pf = 0; pf < PF_MAX; pf++) begin
        if (pf < Pf_cur) begin
          filter_idx = f_group * Pf_cur + pf;

          if (filter_idx < F_cur && filter_idx < F_MAX) begin
            for (pv = 0; pv < PV_MAX; pv = pv + 2) begin
              top_col0 = in_col + pv;
              top_col1 = in_col + pv + 1;
              lane0_idx = pf * Pv_cur + pv;
              lane1_idx = pf * Pv_cur + pv + 1;

              if (((pv + 1) < Pv_cur) &&
                  (top_col0 < Wout_cur) && (top_col1 < Wout_cur) &&
                  (top_col0 < WOUT_MAX) && (top_col1 < WOUT_MAX) &&
                  (lane1_idx < PTOTAL)) begin

                // top row comes from row_buf of previous even row
                top0 = row_buf[1'b0][filter_idx][top_col0];
                top1 = row_buf[1'b0][filter_idx][top_col1];

                // bottom row comes from current input block
                bot0 = in_data[lane0_idx];
                bot1 = in_data[lane1_idx];

                max_top = (top0 > top1) ? top0 : top1;
                max_bot = (bot0 > bot1) ? bot0 : bot1;
                max_all = (max_top > max_bot) ? max_top : max_bot;

                out_lane = pf * pooled_pv + (pv >> 1);
                if (out_lane < PTOTAL) begin
                  ofm_write_data[out_lane] = sat_to_out(max_all);
                  valid_count = valid_count + 1;
                end
              end
            end
          end
        end
      end

      ofm_write_en    = (valid_count != 0);
      ofm_write_count = valid_count[15:0];
    end
  end

endmodule
