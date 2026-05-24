// -----------------------------------------------------------------------------
// pooling_mode2.sv
//
// Synth-friendly Mode-2 pooling / bypass stage.
//
// Main changes versus the original row_buf implementation:
//   * No full row-buffer reset.
//   * No 4-D row_buf with dynamic multi-port access.
//   * Top row is stored as saturated horizontal pair-max per filter group.
//   * Current odd-row left sample is held after saturation in a small per-filter-group register.
//   * Debug monitors are guarded with `ifndef SYNTHESIS.
//   * f_base/PF conversion uses shift/case for common power-of-two PF values.
//
// Functional contract is preserved:
//   - input is one global spatial sample and one PF filter group per valid beat
//   - pool_en=1 emits one 2x2 max-pooled output at odd row/odd col
//   - pool_en=0 bypasses every valid sample
//   - OFM coordinates are global output coordinates
// -----------------------------------------------------------------------------

module pooling_mode2_pair_linebuf_bank #(
  parameter int DATA_W = 32,
  parameter int PF     = 8,
  parameter int DEPTH  = 16,
  parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
)(
  input  logic clk,

  input  logic                  wr_en,
  input  logic [ADDR_W-1:0]     wr_addr,
  input  logic [PF*DATA_W-1:0] wr_data,

  input  logic [ADDR_W-1:0]     rd_addr,
  output logic [PF*DATA_W-1:0] rd_data
);
  (* ram_style = "distributed" *)
  logic [PF*DATA_W-1:0] ram [0:DEPTH-1];

  // FPGA init + simulation init. Do not reset RAM in the clocked reset path.
  initial begin
    for (int i = 0; i < DEPTH; i++) begin
      ram[i] = '0;
    end
  end

  always_ff @(posedge clk) begin
    if (wr_en) begin
      ram[wr_addr] <= wr_data;
    end
  end

  // Async read preserves the original single-cycle pooling behavior.
  assign rd_data = ram[rd_addr];
endmodule


module pooling_mode2 #(
  parameter int DATA_W = 32,   // width after ReLU / MAC output
  parameter int OUT_W  = DATA_W,
  parameter int PF     = 8,    // fixed filter parallelism in mode 2
  parameter int W_MAX  = 32,
  parameter int H_MAX  = 32,
  parameter int F_MAX  = 64
)(
  input  logic clk,
  input  logic rst_n,

  // Runtime config. W_cur/H_cur are the full pre-pool spatial dimensions.
  input  logic [15:0] W_cur,
  input  logic [15:0] H_cur,
  input  logic        pool_en,

  // Global coordinate of input sample.
  input  logic [15:0] in_row_g,
  input  logic [15:0] in_col_g,
  input  logic [15:0] tile_col_base_g,

  // Input stream from ReLU mode 2.
  input  logic [PF*DATA_W-1:0] data_in,
  input  logic                 data_in_valid,
  input  logic                 in_group_start,
  input  logic [15:0]          in_f_base,

  // Output write interface to OFM buffer.
  output logic                 ofm_wr_en,
  output logic [15:0]          ofm_wr_row,
  output logic [15:0]          ofm_wr_col,
  output logic [15:0]          ofm_wr_f_base,
  output logic [PF*OUT_W-1:0]  ofm_wr_data
);

  localparam int PF_SAFE      = (PF > 0) ? PF : 1;
  localparam int FGRP_MAX     = (F_MAX + PF_SAFE - 1) / PF_SAFE;
  localparam int PAIR_DEPTH   = (W_MAX + 1) / 2;
  localparam int PAIR_ADDR_W  = (PAIR_DEPTH <= 1) ? 1 : $clog2(PAIR_DEPTH);
  localparam int IN_WORD_W    = PF * DATA_W;
  localparam int OUT_WORD_W   = PF * OUT_W;

  localparam int signed OUT_MAX = (1 <<< (OUT_W-1)) - 1;
  localparam int signed OUT_MIN = -(1 <<< (OUT_W-1));

  // --------------------------------------------------
  // Input staging
  // --------------------------------------------------
  logic                 in_valid_q;
  logic [IN_WORD_W-1:0] in_data_q;
  logic                 in_group_start_q;
  logic [15:0]          in_f_base_q;
  logic [15:0]          in_row_g_q;
  logic [15:0]          in_col_g_q;
  logic [15:0]          tile_col_base_g_q;
  logic [15:0]          W_cur_q;
  logic [15:0]          H_cur_q;
  logic                 pool_en_q;
  logic [15:0]          in_fgrp_idx_q;

  // --------------------------------------------------
  // Pair-based line buffer.
  //
  // For even input rows:
  //   even col: hold left sample for this fgrp
  //   odd  col: write max(sat(right=current), sat(left=held)) to line buffer
  //
  // For odd input rows:
  //   even col: hold bottom-left sample for this fgrp
  //   odd  col: read top pair-max from line buffer, combine with bottom-left hold
  //             and current bottom-right sample, then emit pooled result.
  // --------------------------------------------------
  // Hold and line-buffer data are stored after ReLU/saturation to OUT_W.
  // This keeps the 2x2 pooling result identical for monotonic saturation, while
  // reducing the line-buffer width from 2*PF*DATA_W to PF*OUT_W.
  logic [OUT_WORD_W-1:0] top_left_hold [0:FGRP_MAX-1];
  logic [OUT_WORD_W-1:0] bot_left_hold [0:FGRP_MAX-1];

  logic [FGRP_MAX-1:0]          linebuf_wr_en;
  logic [PAIR_ADDR_W-1:0]       linebuf_wr_addr [0:FGRP_MAX-1];
  logic [OUT_WORD_W-1:0]        linebuf_wr_data [0:FGRP_MAX-1];
  logic [PAIR_ADDR_W-1:0]       linebuf_rd_addr [0:FGRP_MAX-1];
  logic [OUT_WORD_W-1:0]        linebuf_rd_data [0:FGRP_MAX-1];

  logic [PAIR_ADDR_W-1:0] pair_idx_s;
  logic [OUT_WORD_W-1:0]  selected_top_pair_s;
  logic [OUT_WORD_W-1:0]  selected_bot_left_s;
  logic [OUT_WORD_W-1:0]  in_sat_word_s;
  logic [OUT_WORD_W-1:0]  top_pair_word_s [0:FGRP_MAX-1];
  logic [PF*OUT_W-1:0]    pool_word_s;
  logic [PF*OUT_W-1:0]    bypass_word_s;

  logic coord_valid_s;
  logic fgrp_valid_s;
  logic pool_window_close_s;
  logic top_pair_store_s;
  logic bot_left_store_s;

  // --------------------------------------------------
  // Helper functions
  // --------------------------------------------------
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

  function automatic [15:0] fbase_to_fgrp(input logic [15:0] fbase);
    begin
      case (PF_SAFE)
        1:       fbase_to_fgrp = fbase;
        2:       fbase_to_fgrp = fbase >> 1;
        4:       fbase_to_fgrp = fbase >> 2;
        8:       fbase_to_fgrp = fbase >> 3;
        16:      fbase_to_fgrp = fbase >> 4;
        32:      fbase_to_fgrp = fbase >> 5;
        64:      fbase_to_fgrp = fbase >> 6;
        default: fbase_to_fgrp = fbase / PF_SAFE;
      endcase
    end
  endfunction

  // --------------------------------------------------
  // Line buffer banks: one top-row-pair buffer per filter group.
  // --------------------------------------------------
  genvar g_fgrp;
  generate
    for (g_fgrp = 0; g_fgrp < FGRP_MAX; g_fgrp++) begin : G_TOP_PAIR_BUF
      pooling_mode2_pair_linebuf_bank #(
        .DATA_W(OUT_W),
        .PF    (PF),
        .DEPTH (PAIR_DEPTH),
        .ADDR_W(PAIR_ADDR_W)
      ) u_pair_linebuf (
        .clk    (clk),
        .wr_en  (linebuf_wr_en[g_fgrp]),
        .wr_addr(linebuf_wr_addr[g_fgrp]),
        .wr_data(linebuf_wr_data[g_fgrp]),
        .rd_addr(linebuf_rd_addr[g_fgrp]),
        .rd_data(linebuf_rd_data[g_fgrp])
      );
    end
  endgenerate

  // --------------------------------------------------
  // Coordinate / group validity for the staged incoming sample.
  // --------------------------------------------------
  always_comb begin
    coord_valid_s = (in_row_g_q < H_cur_q) &&
                    (in_col_g_q < W_cur_q) &&
                    (in_row_g_q < H_MAX)  &&
                    (in_col_g_q < W_MAX);

    fgrp_valid_s = (in_fgrp_idx_q < FGRP_MAX);

    pool_window_close_s = coord_valid_s &&
                          in_row_g_q[0] &&
                          in_col_g_q[0];

    top_pair_store_s = coord_valid_s &&
                       !in_row_g_q[0] &&
                       in_col_g_q[0];

    bot_left_store_s = coord_valid_s &&
                       in_row_g_q[0] &&
                       !in_col_g_q[0];

    pair_idx_s = in_col_g_q >> 1;
  end

  // --------------------------------------------------
  // Saturate current PF word once and pre-compute the top-row horizontal max.
  // The line buffer stores top_pair_max per PF lane, not the raw top-left/right
  // 32-bit samples. This reduces the line-buffer word width substantially.
  // --------------------------------------------------
  always_comb begin
    in_sat_word_s = '0;

    for (int pf = 0; pf < PF; pf++) begin
      logic signed [DATA_W-1:0] in_raw;
      in_raw = signed'(in_data_q[pf*DATA_W +: DATA_W]);
      in_sat_word_s[pf*OUT_W +: OUT_W] = sat_to_out(in_raw);
    end
  end

  always_comb begin
    for (int fg = 0; fg < FGRP_MAX; fg++) begin
      top_pair_word_s[fg] = '0;

      for (int pf = 0; pf < PF; pf++) begin
        logic signed [OUT_W-1:0] top_left_s;
        logic signed [OUT_W-1:0] top_right_s;
        top_left_s  = signed'(top_left_hold[fg][pf*OUT_W +: OUT_W]);
        top_right_s = signed'(in_sat_word_s[pf*OUT_W +: OUT_W]);
        top_pair_word_s[fg][pf*OUT_W +: OUT_W] =
          (top_right_s > top_left_s) ? top_right_s : top_left_s;
      end
    end
  end

  // --------------------------------------------------
  // Line-buffer write/read command generation.
  // --------------------------------------------------
  always_comb begin
    for (int fg = 0; fg < FGRP_MAX; fg++) begin
      linebuf_wr_en[fg]   = 1'b0;
      linebuf_wr_addr[fg] = pair_idx_s;
      linebuf_wr_data[fg] = top_pair_word_s[fg];

      linebuf_rd_addr[fg] = pair_idx_s;
    end

    if (in_valid_q && pool_en_q && fgrp_valid_s && top_pair_store_s) begin
      for (int fg = 0; fg < FGRP_MAX; fg++) begin
        if (in_fgrp_idx_q == fg) begin
          linebuf_wr_en[fg]   = 1'b1;
          linebuf_wr_addr[fg] = pair_idx_s;
          linebuf_wr_data[fg] = top_pair_word_s[fg];
        end
      end
    end
  end

  // --------------------------------------------------
  // Select current filter-group top pair and bottom-left hold.
  // FGRP_MAX is small, so this is only a small mux, not a 64-way filter mux.
  // --------------------------------------------------
  always_comb begin
    selected_top_pair_s = '0;
    selected_bot_left_s = '0;

    for (int fg = 0; fg < FGRP_MAX; fg++) begin
      if (in_fgrp_idx_q == fg) begin
        selected_top_pair_s = linebuf_rd_data[fg];
        selected_bot_left_s = bot_left_hold[fg];
      end
    end
  end

  // --------------------------------------------------
  // Combinational output word generation.
  // Pooling is performed in OUT_W domain after monotonic saturation:
  //   max(sat(a), sat(b), sat(c), sat(d)) == sat(max(a,b,c,d)).
  // --------------------------------------------------
  always_comb begin
    pool_word_s   = '0;
    bypass_word_s = '0;

    for (int pf = 0; pf < PF; pf++) begin
      logic signed [OUT_W-1:0] top_pair;
      logic signed [OUT_W-1:0] bot_left;
      logic signed [OUT_W-1:0] bot_right;
      logic signed [OUT_W-1:0] bot_pair;
      logic signed [OUT_W-1:0] final_max;

      top_pair = signed'(selected_top_pair_s[pf*OUT_W +: OUT_W]);
      bot_left = signed'(selected_bot_left_s[pf*OUT_W +: OUT_W]);
      bot_right = signed'(in_sat_word_s[pf*OUT_W +: OUT_W]);

      bot_pair = (bot_right > bot_left) ? bot_right : bot_left;
      final_max = (bot_pair > top_pair) ? bot_pair : top_pair;

      pool_word_s[pf*OUT_W +: OUT_W]   = final_max;
      bypass_word_s[pf*OUT_W +: OUT_W] = bot_right;
    end
  end

  // --------------------------------------------------
  // Sequential behavior
  // --------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      in_valid_q        <= 1'b0;
      in_data_q         <= '0;
      in_group_start_q  <= 1'b0;
      in_f_base_q       <= '0;
      in_row_g_q        <= '0;
      in_col_g_q        <= '0;
      tile_col_base_g_q <= '0;
      W_cur_q           <= '0;
      H_cur_q           <= '0;
      pool_en_q         <= 1'b0;
      in_fgrp_idx_q     <= '0;

      ofm_wr_en     <= 1'b0;
      ofm_wr_row    <= '0;
      ofm_wr_col    <= '0;
      ofm_wr_f_base <= '0;
      ofm_wr_data   <= '0;

      // Do not reset RAM contents. Small hold registers may be reset safely.
      for (int fg = 0; fg < FGRP_MAX; fg++) begin
        top_left_hold[fg] <= '0;
        bot_left_hold[fg] <= '0;
      end
    end
    else begin
      ofm_wr_en <= 1'b0;

      // Consume the previously staged sample.
      if (in_valid_q && fgrp_valid_s) begin
        if (pool_en_q) begin
          if (coord_valid_s) begin
            if (!in_row_g_q[0] && !in_col_g_q[0]) begin
              // Even row, even col: remember top-left candidate.
              top_left_hold[in_fgrp_idx_q] <= in_sat_word_s;
            end

            if (bot_left_store_s) begin
              // Odd row, even col: remember bottom-left candidate.
              bot_left_hold[in_fgrp_idx_q] <= in_sat_word_s;
            end
          end

          // Emit pooled output when current sample closes a 2x2 window.
          if (pool_window_close_s) begin
            ofm_wr_en     <= 1'b1;
            ofm_wr_row    <= in_row_g_q >> 1;
            ofm_wr_col    <= in_col_g_q >> 1;
            ofm_wr_f_base <= in_f_base_q;
            ofm_wr_data   <= pool_word_s;
          end
        end
        else begin
          // No-pool/bypass path.
          if (coord_valid_s) begin
            ofm_wr_en     <= 1'b1;
            ofm_wr_row    <= in_row_g_q;
            ofm_wr_col    <= in_col_g_q;
            ofm_wr_f_base <= in_f_base_q;
            ofm_wr_data   <= bypass_word_s;
          end
        end
      end

      // Stage current raw input for consumption next cycle.
      in_valid_q <= data_in_valid;
      if (data_in_valid) begin
        in_data_q         <= data_in;
        in_group_start_q  <= in_group_start;
        in_f_base_q       <= in_f_base;
        in_row_g_q        <= in_row_g;
        in_col_g_q        <= in_col_g;
        tile_col_base_g_q <= tile_col_base_g;
        W_cur_q           <= W_cur;
        H_cur_q           <= H_cur;
        pool_en_q         <= pool_en;
        in_fgrp_idx_q     <= fbase_to_fgrp(in_f_base);
      end
    end
  end

`ifndef SYNTHESIS
  function automatic logic dbg_pool_col_focus(input logic [15:0] c);
    begin
      dbg_pool_col_focus =
        (c < 16'd40) ||
        ((c >= 16'd60) && (c <= 16'd68));
    end
  endfunction

  logic [31:0] dbg_pool_m2_evt_q;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      dbg_pool_m2_evt_q <= 32'd0;
    end
    else begin
      if ((data_in_valid &&
           (in_row_g < 16'd4) &&
           (in_f_base == 16'd0) &&
           dbg_pool_col_focus(in_col_g)) ||

          (in_valid_q &&
           (in_row_g_q < 16'd4) &&
           (in_f_base_q == 16'd0) &&
           dbg_pool_col_focus(in_col_g_q)) ||

          (ofm_wr_en &&
           (ofm_wr_row < 16'd4) &&
           (ofm_wr_f_base == 16'd0) &&
           (ofm_wr_col < 16'd48))) begin

        dbg_pool_m2_evt_q <= dbg_pool_m2_evt_q + 32'd1;

        $display("DBG_POOL_M2_GLOBAL t=%0t evt=%0d pool=%0b W=%0d H=%0d raw_v=%0b raw_row=%0d raw_col=%0d raw_tile_base=%0d raw_fbase=%0d raw_data0=%0d staged_v=%0b st_row=%0d st_col=%0d st_tile_base=%0d st_fbase=%0d st_fgrp=%0d st_data0=%0d close=%0b wr=%0b wr_row=%0d wr_col=%0d wr_fbase=%0d wr_data0=%0d",
          $time,
          dbg_pool_m2_evt_q,
          pool_en,
          W_cur,
          H_cur,

          data_in_valid,
          in_row_g,
          in_col_g,
          tile_col_base_g,
          in_f_base,
          $signed(data_in[0*DATA_W +: DATA_W]),

          in_valid_q,
          in_row_g_q,
          in_col_g_q,
          tile_col_base_g_q,
          in_f_base_q,
          in_fgrp_idx_q,
          $signed(in_data_q[0*DATA_W +: DATA_W]),

          pool_window_close_s,

          ofm_wr_en,
          ofm_wr_row,
          ofm_wr_col,
          ofm_wr_f_base,
          $signed(ofm_wr_data[0*OUT_W +: OUT_W])
        );
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // no-op
    end else begin
      if (ofm_wr_en) begin
        $display("DBG_M2_FINAL_WRITE t=%0t pool_en_q=%0b W_q=%0d H_q=%0d st_row=%0d st_col=%0d st_tile_base=%0d st_fbase=%0d close=%0b wr_row=%0d wr_col=%0d wr_fbase=%0d wr_data0=%0d",
                 $time,
                 pool_en_q,
                 W_cur_q,
                 H_cur_q,
                 in_row_g_q,
                 in_col_g_q,
                 tile_col_base_g_q,
                 in_f_base_q,
                 pool_window_close_s,
                 ofm_wr_row,
                 ofm_wr_col,
                 ofm_wr_f_base,
                 $signed(ofm_wr_data[0*OUT_W +: OUT_W]));
      end
    end
  end
`endif

endmodule
