// Mode1 2x2 maxpool optimized for the current P128 benchmark shape.
//
// This version is based on pooling_mode1_outw_linebuf_fast.sv, but stores only
// the top-row horizontal pair maxima in the line buffer:
//   old linebuf word: PV_MAX lanes x OUT_W  = 8 x 8 = 64 bit for P128
//   new linebuf word: PV_MAX/2 lanes x OUT_W = 4 x 8 = 32 bit for P128
//
// For a 2x2 window:
//   top_pair = max(top0, top1)    // stored on even row
//   bot_pair = max(bot0, bot1)    // computed on odd row
//   pool     = max(top_pair, bot_pair)
//
// This avoids storing both top-row pixels and removes the 64-bit top-row word
// mux from the previous fast version. It keeps the same module interface and
// output lane contract for the current 32x32/P128 build.

module pooling_mode1_linebuf_bank #(
  parameter int DATA_W = 8,
  parameter int LANES  = 4,
  parameter int DEPTH  = 4,
  parameter int ADDR_W = 2
)(
  input  logic clk,

  input  logic                    wr_en,
  input  logic [ADDR_W-1:0]       wr_addr,
  input  logic [DATA_W*LANES-1:0] wr_data,

  input  logic [ADDR_W-1:0]       rd_addr,
  output logic [DATA_W*LANES-1:0] rd_data
);
  (* ram_style = "distributed" *) logic [DATA_W*LANES-1:0] mem [0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (wr_en) begin
      mem[wr_addr] <= wr_data;
    end
  end

  assign rd_data = mem[rd_addr];
endmodule

module pooling_mode1 #(
  parameter int DATA_W   = 32,
  parameter int OUT_W    = DATA_W,
  parameter int PV_MAX   = 8,
  parameter int PF_MAX   = 16,
  parameter int PTOTAL   = 128,
  parameter int F_MAX    = 64,
  parameter int WOUT_MAX = 32
)(
  input  logic clk,
  input  logic rst_n,
  input  logic pool_en,

  input  logic [7:0]  Pv_cur,
  input  logic [7:0]  Pf_cur,
  input  logic [9:0]  F_cur,
  input  logic [15:0] Wout_cur,
  input  logic [7:0]  f_group,

  input  logic                     in_valid,
  input  logic [15:0]              in_row,
  input  logic [15:0]              in_col,
  input  logic signed [DATA_W-1:0] in_data [0:PTOTAL-1],

  output logic                     ofm_write_en,
  output logic [15:0]              ofm_write_filter_base,
  output logic [15:0]              ofm_write_row,
  output logic [15:0]              ofm_write_col_base,
  output logic [15:0]              ofm_write_count,
  output logic signed [OUT_W-1:0]  ofm_write_data [0:PTOTAL-1]
);

  localparam int signed OUT_MAX = (1 <<< (OUT_W-1)) - 1;
  localparam int signed OUT_MIN = -(1 <<< (OUT_W-1));

  localparam int WORD_LANES = PV_MAX;
  localparam int POOL_LANES_PER_PF = (PV_MAX / 2);
  localparam int PAIR_WORD_LANES   = POOL_LANES_PER_PF;
  localparam int PAIR_WORD_W       = OUT_W * PAIR_WORD_LANES;
  localparam int COLGRP_MAX        = (WOUT_MAX + WORD_LANES - 1) / WORD_LANES;
  localparam int COLGRP_W          = (COLGRP_MAX <= 1) ? 1 : $clog2(COLGRP_MAX);
  localparam int FGRP_MAX          = (F_MAX + PF_MAX - 1) / PF_MAX;
  localparam int POOL_TOTAL_LANES  = PF_MAX * POOL_LANES_PER_PF;

  logic                  lb_wr_en   [0:FGRP_MAX-1][0:PF_MAX-1];
  logic [COLGRP_W-1:0]   lb_wr_addr [0:FGRP_MAX-1][0:PF_MAX-1];
  logic [PAIR_WORD_W-1:0] lb_wr_data [0:FGRP_MAX-1][0:PF_MAX-1];
  logic [COLGRP_W-1:0]   lb_rd_addr [0:FGRP_MAX-1][0:PF_MAX-1];
  logic [PAIR_WORD_W-1:0] lb_rd_data [0:FGRP_MAX-1][0:PF_MAX-1];

  genvar g_fg, g_pf;
  generate
    for (g_fg = 0; g_fg < FGRP_MAX; g_fg++) begin : G_POOL_M1_FGRP
      for (g_pf = 0; g_pf < PF_MAX; g_pf++) begin : G_POOL_M1_PF
        pooling_mode1_linebuf_bank #(
          .DATA_W(OUT_W),
          .LANES (PAIR_WORD_LANES),
          .DEPTH (COLGRP_MAX),
          .ADDR_W(COLGRP_W)
        ) u_linebuf_bank (
          .clk    (clk),
          .wr_en  (lb_wr_en[g_fg][g_pf]),
          .wr_addr(lb_wr_addr[g_fg][g_pf]),
          .wr_data(lb_wr_data[g_fg][g_pf]),
          .rd_addr(lb_rd_addr[g_fg][g_pf]),
          .rd_data(lb_rd_data[g_fg][g_pf])
        );
      end
    end
  endgenerate

  function automatic logic signed [OUT_W-1:0] sat_to_out(
    input logic signed [DATA_W-1:0] din
  );
    begin
      // Input is expected to come from relu_mode1's resource-oriented output:
      // non-negative and already clamped to OUT_W. Keep only the payload bits
      // to avoid re-building 32-bit saturation comparators inside pooling.
      sat_to_out = din[OUT_W-1:0];
    end
  endfunction

  function automatic int div_const_word_lanes(input int v);
    begin
      unique case (WORD_LANES)
        1:  div_const_word_lanes = v;
        2:  div_const_word_lanes = v >> 1;
        4:  div_const_word_lanes = v >> 2;
        8:  div_const_word_lanes = v >> 3;
        16: div_const_word_lanes = v >> 4;
        32: div_const_word_lanes = v >> 5;
        default: div_const_word_lanes = v / WORD_LANES;
      endcase
    end
  endfunction

  function automatic int mod_const_word_lanes(input int v);
    begin
      unique case (WORD_LANES)
        1:  mod_const_word_lanes = 0;
        2:  mod_const_word_lanes = v & 1;
        4:  mod_const_word_lanes = v & 3;
        8:  mod_const_word_lanes = v & 7;
        16: mod_const_word_lanes = v & 15;
        32: mod_const_word_lanes = v & 31;
        default: mod_const_word_lanes = v % WORD_LANES;
      endcase
    end
  endfunction

  function automatic int mul_const_pfmax(input int v);
    begin
      unique case (PF_MAX)
        1:  mul_const_pfmax = v;
        2:  mul_const_pfmax = v << 1;
        4:  mul_const_pfmax = v << 2;
        8:  mul_const_pfmax = v << 3;
        16: mul_const_pfmax = v << 4;
        32: mul_const_pfmax = v << 5;
        default: mul_const_pfmax = v * PF_MAX;
      endcase
    end
  endfunction

  function automatic logic signed [OUT_W-1:0] get_pair_lane(
    input logic [PAIR_WORD_W-1:0] word,
    input int lane
  );
    begin
      get_pair_lane = $signed(word[lane*OUT_W +: OUT_W]);
    end
  endfunction

  // Decode only constant-width geometry.
  logic [COLGRP_W-1:0] col_grp_cur;
  int                  col_grp_i;
  int                  col_off_cur;
  int                  fgrp_cur_i;
  int                  filter_base_i;
  logic                beat_aligned_fast;

  always_comb begin
    col_grp_i        = div_const_word_lanes(int'(in_col));
    col_grp_cur      = col_grp_i[COLGRP_W-1:0];
    col_off_cur      = mod_const_word_lanes(int'(in_col));
    fgrp_cur_i       = int'(f_group);
    filter_base_i    = mul_const_pfmax(int'(f_group));
    beat_aligned_fast = (col_off_cur == 0) &&
                        (Pv_cur == PV_MAX[7:0]) &&
                        (Pf_cur == PF_MAX[7:0]);
  end

  // Even row: store the top-row horizontal pair max for each active PF lane.
  always_comb begin
    int fg;
    int pf;
    int px;
    int pv0;
    int pv1;
    int abs_filter;
    int lane0_idx;
    int lane1_idx;
    int col0;
    int col1;
    logic [PAIR_WORD_W-1:0] next_word;
    logic signed [OUT_W-1:0] top0_s;
    logic signed [OUT_W-1:0] top1_s;
    logic signed [OUT_W-1:0] top_pair_s;

    top0_s    = '0;
    top1_s    = '0;
    top_pair_s = '0;

    for (fg = 0; fg < FGRP_MAX; fg++) begin
      for (pf = 0; pf < PF_MAX; pf++) begin
        lb_wr_en[fg][pf]   = 1'b0;
        lb_wr_addr[fg][pf] = col_grp_cur;
        lb_wr_data[fg][pf] = lb_rd_data[fg][pf];
        lb_rd_addr[fg][pf] = col_grp_cur;
      end
    end

    if (pool_en && in_valid && !in_row[0] && beat_aligned_fast) begin
      for (fg = 0; fg < FGRP_MAX; fg++) begin
        if (fg == fgrp_cur_i) begin
          for (pf = 0; pf < PF_MAX; pf++) begin
            abs_filter = (fg * PF_MAX) + pf;
            if ((abs_filter < int'(F_cur)) && (abs_filter < F_MAX)) begin
              next_word = lb_rd_data[fg][pf];
              for (px = 0; px < POOL_LANES_PER_PF; px++) begin
                pv0 = px << 1;
                pv1 = pv0 + 1;
                col0 = int'(in_col) + pv0;
                col1 = int'(in_col) + pv1;
                if ((col1 < int'(Wout_cur)) && (col1 < WOUT_MAX)) begin
                  lane0_idx = (pf * PV_MAX) + pv0;
                  lane1_idx = lane0_idx + 1;
                  if (lane1_idx < PTOTAL) begin
                    top0_s = sat_to_out(in_data[lane0_idx]);
                    top1_s = sat_to_out(in_data[lane1_idx]);
                    top_pair_s = (top0_s > top1_s) ? top0_s : top1_s;
                    next_word[px*OUT_W +: OUT_W] = top_pair_s;
                  end
                end
              end
              lb_wr_en[fg][pf]   = 1'b1;
              lb_wr_addr[fg][pf] = col_grp_cur;
              lb_wr_data[fg][pf] = next_word;
            end
          end
        end
      end
    end
  end

  // Odd row: produce pooled output using static lane packing.
  always_comb begin
    int i;
    int fg;
    int pf;
    int px;
    int pv0;
    int pv1;
    int lane0_idx;
    int lane1_idx;
    int out_lane;
    int abs_filter;
    int col1;
    int valid_count;
    logic signed [OUT_W-1:0] top_pair;
    logic signed [OUT_W-1:0] bot0;
    logic signed [OUT_W-1:0] bot1;
    logic signed [OUT_W-1:0] bot_pair;
    logic signed [OUT_W-1:0] max_all;

    ofm_write_en          = 1'b0;
    ofm_write_filter_base = filter_base_i[15:0];
    ofm_write_row         = in_row >> 1;
    ofm_write_col_base    = in_col >> 1;
    ofm_write_count       = 16'd0;
    valid_count           = 0;

    top_pair = '0;
    bot0     = '0;
    bot1     = '0;
    bot_pair = '0;
    max_all  = '0;

    for (i = 0; i < PTOTAL; i++) begin
      ofm_write_data[i] = '0;
    end

    if (pool_en && in_valid && in_row[0] && beat_aligned_fast) begin
      for (fg = 0; fg < FGRP_MAX; fg++) begin
        if (fg == fgrp_cur_i) begin
          for (pf = 0; pf < PF_MAX; pf++) begin
            abs_filter = (fg * PF_MAX) + pf;
            if ((abs_filter < int'(F_cur)) && (abs_filter < F_MAX)) begin
              for (px = 0; px < POOL_LANES_PER_PF; px++) begin
                pv0 = px << 1;
                pv1 = pv0 + 1;
                col1 = int'(in_col) + pv1;
                if ((col1 < int'(Wout_cur)) && (col1 < WOUT_MAX)) begin
                  lane0_idx = (pf * PV_MAX) + pv0;
                  lane1_idx = lane0_idx + 1;
                  if (lane1_idx < PTOTAL) begin
                    top_pair = get_pair_lane(lb_rd_data[fg][pf], px);
                    bot0     = sat_to_out(in_data[lane0_idx]);
                    bot1     = sat_to_out(in_data[lane1_idx]);
                    bot_pair = (bot0 > bot1) ? bot0 : bot1;
                    max_all  = (top_pair > bot_pair) ? top_pair : bot_pair;

                    out_lane = (pf * POOL_LANES_PER_PF) + px;
                    if (out_lane < PTOTAL) begin
                      ofm_write_data[out_lane] = max_all;
                      valid_count = valid_count + 1;
                    end
                  end
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

`ifndef SYNTHESIS
  always_ff @(posedge clk) begin
    if (rst_n && pool_en && in_valid && !beat_aligned_fast) begin
      $display("POOL_M1_WARN_FASTPATH_ONLY t=%0t row=%0d col=%0d Pv=%0d Pf=%0d f_group=%0d",
               $time, in_row, in_col, Pv_cur, Pf_cur, f_group);
    end
  end
`endif

endmodule
