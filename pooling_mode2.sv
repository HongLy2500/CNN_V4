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

  // --------------------------------------------------
  // Runtime config
  // W_cur/H_cur are the full pre-pool spatial dimensions seen by this module.
  // For pool_en=1, output coordinates are in_row_g/2 and in_col_g/2.
  // For pool_en=0, output coordinates are in_row_g and in_col_g.
  // --------------------------------------------------
  input  logic [15:0] W_cur,
  input  logic [15:0] H_cur,
  input  logic        pool_en,

  // --------------------------------------------------
  // GLOBAL coordinate of the input sample, aligned with data_in_valid.
  // This is the same contract used by Mode 1: CE/controller is the single
  // source of global spatial coordinates, and pooling only applies stride.
  // tile_col_base_g is provided for visibility/debug and compatibility with
  // the Mode-2 tile-window path; this module does not use it to form OFM col.
  // --------------------------------------------------
  input  logic [15:0] in_row_g,
  input  logic [15:0] in_col_g,
  input  logic [15:0] tile_col_base_g,

  // --------------------------------------------------
  // Input stream from ReLU mode 2
  // - one valid sample = one GLOBAL spatial location and one PF filter group
  // - data_in contains PF lanes for one filter-group
  // - in_f_base identifies that filter-group
  // - in_group_start is kept for compatibility/debug; coordinates no longer
  //   depend on internal raster counters.
  // --------------------------------------------------
  input  logic [PF*DATA_W-1:0] data_in,
  input  logic                 data_in_valid,
  input  logic                 in_group_start,
  input  logic [15:0]          in_f_base,

  // --------------------------------------------------
  // Output write interface to OFM buffer
  // Each write stores PF pooled/no-pool values of one output pixel.
  // OFM row/col are GLOBAL coordinates by contract.
  // --------------------------------------------------
  output logic                 ofm_wr_en,
  output logic [15:0]          ofm_wr_row,
  output logic [15:0]          ofm_wr_col,
  output logic [15:0]          ofm_wr_f_base,
  output logic [PF*OUT_W-1:0]  ofm_wr_data
);

  localparam int PF_SAFE   = (PF > 0) ? PF : 1;
  localparam int FGRP_MAX  = (F_MAX + PF_SAFE - 1) / PF_SAFE;

  localparam int signed OUT_MAX = (1 <<< (OUT_W-1)) - 1;
  localparam int signed OUT_MIN = -(1 <<< (OUT_W-1));

  // --------------------------------------------------
  // Input staging
  // --------------------------------------------------
  // ReLU mode2 is registered. The sample, global coordinates, and filter-base
  // are captured together, then consumed one cycle later.
  // --------------------------------------------------
  logic                 in_valid_q;
  logic [PF*DATA_W-1:0] in_data_q;
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
  // Two-row buffer per filter group.
  //
  // Mode 2 now runs pixel-major like Mode 1:
  //   row -> col -> f_group
  // so different f_groups for the same spatial pixel may be interleaved.
  // The row buffer therefore needs a filter-group dimension; otherwise one
  // f_group would overwrite another f_group's top-row data.
  //
  // row_buf[fgrp][buf_sel][pf][global_col]
  // --------------------------------------------------
  logic signed [DATA_W-1:0] row_buf [0:FGRP_MAX-1][0:1][0:PF-1][0:W_MAX-1];

  logic signed [DATA_W-1:0] in_lane   [0:PF-1];
  logic signed [DATA_W-1:0] pool_lane [0:PF-1];

  logic        coord_valid_s;
  logic        fgrp_valid_s;
  logic        pool_window_close_s;
  logic        buf_sel_s;
  logic        prev_buf_sel_s;

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

  // --------------------------------------------------
  // Unpack staged input lanes
  // --------------------------------------------------
  always_comb begin
    for (int pf = 0; pf < PF; pf++) begin
      in_lane[pf] = signed'(in_data_q[pf*DATA_W +: DATA_W]);
    end
  end

  // --------------------------------------------------
  // Coordinate / group validity for the staged incoming sample.
  // Coordinates are GLOBAL; no internal raster counter is used.
  // --------------------------------------------------
  always_comb begin
    coord_valid_s = (in_row_g_q < H_cur_q) &&
                    (in_col_g_q < W_cur_q) &&
                    (in_row_g_q < H_MAX)  &&
                    (in_col_g_q < W_MAX);

    fgrp_valid_s  = (in_fgrp_idx_q < FGRP_MAX);

    buf_sel_s      = in_row_g_q[0];
    prev_buf_sel_s = ~in_row_g_q[0];

    pool_window_close_s = coord_valid_s &&
                          in_row_g_q[0] &&
                          in_col_g_q[0];
  end

  // --------------------------------------------------
  // Combinational 2x2 max-pooling, stride 2.
  // Uses the STAGED current sample.
  // Valid only when current sample closes a 2x2 window:
  //   global row is odd and global col is odd
  // --------------------------------------------------
  always_comb begin
    for (int pf = 0; pf < PF; pf++) begin
      pool_lane[pf] = '0;
    end

    if (fgrp_valid_s && pool_window_close_s) begin
      for (int pf = 0; pf < PF; pf++) begin : GEN_POOL
        logic signed [DATA_W-1:0] top_max;
        logic signed [DATA_W-1:0] bot_max;
        logic signed [DATA_W-1:0] final_max;

        top_max = row_buf[in_fgrp_idx_q][prev_buf_sel_s][pf][in_col_g_q - 1];
        if (row_buf[in_fgrp_idx_q][prev_buf_sel_s][pf][in_col_g_q] > top_max)
          top_max = row_buf[in_fgrp_idx_q][prev_buf_sel_s][pf][in_col_g_q];

        bot_max = row_buf[in_fgrp_idx_q][buf_sel_s][pf][in_col_g_q - 1];
        if (in_lane[pf] > bot_max)
          bot_max = in_lane[pf];

        final_max = top_max;
        if (bot_max > final_max)
          final_max = bot_max;

        pool_lane[pf] = final_max;
      end
    end
  end

  // --------------------------------------------------
  // Sequential behavior
  // - consume one staged sample per cycle when in_valid_q=1
  // - capture the raw input sample and its GLOBAL coordinate for consumption
  //   on the next cycle
  // - store incoming sample into row buffer only for pooling mode
  // - when a 2x2 window closes, emit pooled output to GLOBAL OFM coordinate
  // - in no-pool mode, emit every staged sample directly to GLOBAL OFM coord
  // --------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      in_valid_q         <= 1'b0;
      in_data_q          <= '0;
      in_group_start_q   <= 1'b0;
      in_f_base_q        <= '0;
      in_row_g_q         <= '0;
      in_col_g_q         <= '0;
      tile_col_base_g_q  <= '0;
      W_cur_q            <= '0;
      H_cur_q            <= '0;
      pool_en_q          <= 1'b0;
      in_fgrp_idx_q      <= '0;

      ofm_wr_en     <= 1'b0;
      ofm_wr_row    <= '0;
      ofm_wr_col    <= '0;
      ofm_wr_f_base <= '0;
      ofm_wr_data   <= '0;

      for (int fgi = 0; fgi < FGRP_MAX; fgi++) begin
        for (int c = 0; c < W_MAX; c++) begin
          for (int pf = 0; pf < PF; pf++) begin
            row_buf[fgi][0][pf][c] <= '0;
            row_buf[fgi][1][pf][c] <= '0;
          end
        end
      end
    end
    else begin
      ofm_wr_en <= 1'b0;

      // ------------------------------------------------
      // Consume the previously staged sample.
      // ------------------------------------------------
      if (in_valid_q && fgrp_valid_s) begin
        if (pool_en_q) begin
          // Store current sample into row buffer only in pooling mode.
          if (coord_valid_s) begin
            for (int pf = 0; pf < PF; pf++) begin
              row_buf[in_fgrp_idx_q][buf_sel_s][pf][in_col_g_q] <= in_lane[pf];
            end
          end

          // Emit pooled output when a 2x2 window closes.
          // IMPORTANT: OFM coordinates are GLOBAL, exactly like Mode 1:
          //   pooled row = global input row / 2
          //   pooled col = global input col / 2
          if (pool_window_close_s) begin
            ofm_wr_en     <= 1'b1;
            ofm_wr_row    <= in_row_g_q >> 1;
            ofm_wr_col    <= in_col_g_q >> 1;
            ofm_wr_f_base <= in_f_base_q;

            for (int pf = 0; pf < PF; pf++) begin
              ofm_wr_data[pf*OUT_W +: OUT_W] <= sat_to_out(pool_lane[pf]);
            end
          end
        end
        else begin
          // No-pool/bypass path: one staged ReLU sample produces exactly one
          // OFM write at the same GLOBAL spatial coordinate.
          if (coord_valid_s) begin
            ofm_wr_en     <= 1'b1;
            ofm_wr_row    <= in_row_g_q;
            ofm_wr_col    <= in_col_g_q;
            ofm_wr_f_base <= in_f_base_q;

            for (int pf = 0; pf < PF; pf++) begin
              ofm_wr_data[pf*OUT_W +: OUT_W] <= sat_to_out(in_lane[pf]);
            end
          end
        end
      end

      // ------------------------------------------------
      // Stage current raw input for consumption next cycle.
      // Capture metadata only when the sample is valid.
      // ------------------------------------------------
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
        in_fgrp_idx_q     <= in_f_base / PF_SAFE;
      end
    end
  end
  
  


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
        $signed(in_lane[0]),

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

// -----------------------------------------------------------------------------
// DEBUG: Mode2 final OFM write monitor with staged coordinate context
// Place inside pooling_mode2.sv, before endmodule.
// -----------------------------------------------------------------------------

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


endmodule
