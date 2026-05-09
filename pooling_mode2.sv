module pooling_mode2 #(
  parameter int DATA_W = 32,   // width after ReLU / MAC output
  parameter int OUT_W  = DATA_W,
  parameter int PF     = 4,    // fixed filter parallelism in mode 2
  parameter int W_MAX  = 224,
  parameter int H_MAX  = 224
)(
  input  logic clk,
  input  logic rst_n,

  // --------------------------------------------------
  // Runtime config
  // pool_en=1: 2x2 max-pooling, stride 2
  // pool_en=0: bypass/no-pool, one input pixel -> one OFM write
  // --------------------------------------------------
  input  logic [15:0] W_cur,
  input  logic [15:0] H_cur,
  input  logic        pool_en,

  // --------------------------------------------------
  // Input stream from ReLU mode 2
  // Assumption:
  // - one valid sample = one spatial location
  // - data_in contains PF lanes for one filter-group
  // - samples arrive in raster order for the current group
  // - in_group_start=1 on the first sample of a new group
  // --------------------------------------------------
  input  logic [PF*DATA_W-1:0] data_in,
  input  logic                 data_in_valid,
  input  logic                 in_group_start,
  input  logic [15:0]          in_f_base,

  // --------------------------------------------------
  // Output write interface to OFM buffer
  // Each write stores PF pooled/no-pool values of one output pixel.
  // OUT_W may be smaller than DATA_W; results are saturated to signed
  // OUT_W range before being driven out.
  // --------------------------------------------------
  output logic                 ofm_wr_en,
  output logic [15:0]          ofm_wr_row,
  output logic [15:0]          ofm_wr_col,
  output logic [15:0]          ofm_wr_f_base,
  output logic [PF*OUT_W-1:0]  ofm_wr_data
);

  localparam int signed OUT_MAX = (1 <<< (OUT_W-1)) - 1;
  localparam int signed OUT_MIN = -(1 <<< (OUT_W-1));

  // --------------------------------------------------
  // Input staging
  // --------------------------------------------------
  // ReLU mode2 is registered. If this module consumes data_in directly in the
  // same always_ff edge as data_in_valid, it can sample the previous data value
  // while seeing the current valid pulse. That creates exactly the kind of
  // first-pixel error seen in the 9-layer Mode2 test: the first no-pool output
  // of a filter group is written as 0, while following pixels are correct.
  //
  // Therefore every input sample and its metadata are captured first, then the
  // pooling/bypass state machine consumes the staged sample one cycle later.
  // This adds one cycle of latency but keeps valid/data/coords/f_base aligned.
  // --------------------------------------------------
  logic                 in_valid_q;
  logic [PF*DATA_W-1:0] in_data_q;
  logic                 in_group_start_q;
  logic [15:0]          in_f_base_q;
  logic [15:0]          W_cur_q;
  logic [15:0]          H_cur_q;
  logic                 pool_en_q;

  // --------------------------------------------------
  // Two-row buffer:
  // row_buf[buf_sel][pf][col]
  // buf_sel = 0/1, selected by input row parity
  // --------------------------------------------------
  logic signed [DATA_W-1:0] row_buf [0:1][0:PF-1][0:W_MAX-1];

  logic signed [DATA_W-1:0] in_lane   [0:PF-1];
  logic signed [DATA_W-1:0] pool_lane [0:PF-1];

  logic [15:0] cur_row, cur_col;
  logic [15:0] eff_row, eff_col;
  logic        eff_buf_sel, eff_prev_buf_sel;

  integer c;

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
  // Effective coordinates for the staged incoming sample
  // If a new filter-group starts, that staged sample is treated as (row=0,col=0).
  // --------------------------------------------------
  always_comb begin
    if (in_group_start_q) begin
      eff_row = 16'd0;
      eff_col = 16'd0;
    end
    else begin
      eff_row = cur_row;
      eff_col = cur_col;
    end

    eff_buf_sel      = eff_row[0];
    eff_prev_buf_sel = ~eff_row[0];
  end

  // --------------------------------------------------
  // Combinational 2x2 max-pooling, stride 2.
  // Uses the STAGED current sample.
  // Valid only when current sample closes a 2x2 window:
  //   eff_row is odd and eff_col is odd
  // --------------------------------------------------
  always_comb begin
    for (int pf = 0; pf < PF; pf++) begin
      pool_lane[pf] = '0;
    end

    if ((eff_row < H_cur_q) &&
        (eff_col < W_cur_q) &&
        (eff_row < H_MAX) &&
        (eff_col < W_MAX) &&
        eff_row[0] &&
        eff_col[0]) begin
      for (int pf = 0; pf < PF; pf++) begin : GEN_POOL
        logic signed [DATA_W-1:0] top_max;
        logic signed [DATA_W-1:0] bot_max;
        logic signed [DATA_W-1:0] final_max;

        top_max = row_buf[eff_prev_buf_sel][pf][eff_col - 1];
        if (row_buf[eff_prev_buf_sel][pf][eff_col] > top_max)
          top_max = row_buf[eff_prev_buf_sel][pf][eff_col];

        bot_max = row_buf[eff_buf_sel][pf][eff_col - 1];
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
  // - capture the raw input sample for consumption on the next cycle
  // - store incoming sample into current row buffer only for pooling mode
  // - when a 2x2 window closes, emit pooled output to OFM
  // - in no-pool mode, emit every staged sample directly to OFM
  // --------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      in_valid_q       <= 1'b0;
      in_data_q        <= '0;
      in_group_start_q <= 1'b0;
      in_f_base_q      <= '0;
      W_cur_q          <= '0;
      H_cur_q          <= '0;
      pool_en_q        <= 1'b0;

      cur_row       <= '0;
      cur_col       <= '0;
      ofm_wr_en     <= 1'b0;
      ofm_wr_row    <= '0;
      ofm_wr_col    <= '0;
      ofm_wr_f_base <= '0;
      ofm_wr_data   <= '0;

      for (c = 0; c < W_MAX; c++) begin
        for (int pf = 0; pf < PF; pf++) begin
          row_buf[0][pf][c] <= '0;
          row_buf[1][pf][c] <= '0;
        end
      end
    end
    else begin
      ofm_wr_en <= 1'b0;

      // ------------------------------------------------
      // Consume the previously staged sample.
      // ------------------------------------------------
      if (in_valid_q) begin
        if (pool_en_q) begin
          // Store current sample into row buffer only in pooling mode.
          for (int pf = 0; pf < PF; pf++) begin
            if (eff_col < W_MAX)
              row_buf[eff_buf_sel][pf][eff_col] <= in_lane[pf];
          end

          // Emit pooled output when a 2x2 window closes.
          if ((eff_row < H_cur_q) &&
              (eff_col < W_cur_q) &&
              (eff_row < H_MAX) &&
              (eff_col < W_MAX) &&
              eff_row[0] &&
              eff_col[0]) begin
            ofm_wr_en     <= 1'b1;
            ofm_wr_row    <= eff_row >> 1;
            ofm_wr_col    <= eff_col >> 1;
            ofm_wr_f_base <= in_f_base_q;

            for (int pf = 0; pf < PF; pf++) begin
              ofm_wr_data[pf*OUT_W +: OUT_W] <= sat_to_out(pool_lane[pf]);
            end
          end
        end
        else begin
          // No-pool/bypass path:
          // one staged ReLU sample produces exactly one OFM write at the same
          // global spatial coordinate. Do NOT divide row/col by 2 and do NOT
          // wait for a 2x2 window.
          if ((eff_row < H_cur_q) &&
              (eff_col < W_cur_q) &&
              (eff_row < H_MAX) &&
              (eff_col < W_MAX)) begin
            ofm_wr_en     <= 1'b1;
            ofm_wr_row    <= eff_row;
            ofm_wr_col    <= eff_col;
            ofm_wr_f_base <= in_f_base_q;

            for (int pf = 0; pf < PF; pf++) begin
              ofm_wr_data[pf*OUT_W +: OUT_W] <= sat_to_out(in_lane[pf]);
            end
          end
        end

        // Advance raster counters for the staged filter-group sample.
        if (in_group_start_q) begin
          if (W_cur_q == 16'd1) begin
            cur_col <= '0;
            if (H_cur_q == 16'd1)
              cur_row <= '0;
            else
              cur_row <= 16'd1;
          end
          else begin
            cur_col <= 16'd1;
            cur_row <= 16'd0;
          end
        end
        else begin
          if (cur_col == (W_cur_q - 1)) begin
            cur_col <= '0;
            if (cur_row == (H_cur_q - 1))
              cur_row <= '0;
            else
              cur_row <= cur_row + 1'b1;
          end
          else begin
            cur_col <= cur_col + 1'b1;
          end
        end
      end

      // ------------------------------------------------
      // Stage current raw input for consumption next cycle.
      // Capture metadata only when the sample is valid.
      // ------------------------------------------------
      in_valid_q <= data_in_valid;
      if (data_in_valid) begin
        in_data_q        <= data_in;
        in_group_start_q <= in_group_start;
        in_f_base_q      <= in_f_base;
        W_cur_q          <= W_cur;
        H_cur_q          <= H_cur;
        pool_en_q        <= pool_en;
      end
    end
  end

endmodule
