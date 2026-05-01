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
  // --------------------------------------------------
  input  logic [15:0] W_cur,
  input  logic [15:0] H_cur,

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
  // Each write stores PF pooled values of one pooled pixel.
  // OUT_W may be smaller than DATA_W; pooled results are
  // saturated to signed OUT_W range before being driven out.
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
  // Unpack input lanes
  // --------------------------------------------------
  always_comb begin
    for (int pf = 0; pf < PF; pf++) begin
      in_lane[pf] = signed'(data_in[pf*DATA_W +: DATA_W]);
    end
  end

  // --------------------------------------------------
  // Effective coordinates for the current incoming sample
  // If a new filter-group starts, that current sample is
  // treated as (row=0, col=0).
  // --------------------------------------------------
  always_comb begin
    if (in_group_start) begin
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
  // Combinational 2x2 max-pooling, stride 2
  // Valid only when current sample closes a 2x2 window:
  //   eff_row is odd and eff_col is odd
  //
  // Window:
  //   top-left     = previous row, col-1
  //   top-right    = previous row, col
  //   bottom-left  = current row,  col-1
  //   bottom-right = current input
  // --------------------------------------------------
  always_comb begin
    for (int pf = 0; pf < PF; pf++) begin
      pool_lane[pf] = '0;
    end

    if ((eff_row < H_cur) &&
        (eff_col < W_cur) &&
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
  // - store incoming sample into current row buffer
  // - when a 2x2 window closes, emit pooled output to OFM
  // - maintain raster counters for the current filter-group
  // --------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
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

      if (data_in_valid) begin
        // Store current sample into row buffer
        for (int pf = 0; pf < PF; pf++) begin
          if (eff_col < W_MAX)
            row_buf[eff_buf_sel][pf][eff_col] <= in_lane[pf];
        end

        // Emit pooled output when a 2x2 window closes
        if ((eff_row < H_cur) &&
            (eff_col < W_cur) &&
            (eff_row < H_MAX) &&
            (eff_col < W_MAX) &&
            eff_row[0] &&
            eff_col[0]) begin
          ofm_wr_en     <= 1'b1;
          ofm_wr_row    <= eff_row >> 1;
          ofm_wr_col    <= eff_col >> 1;
          ofm_wr_f_base <= in_f_base;

          for (int pf = 0; pf < PF; pf++) begin
            ofm_wr_data[pf*OUT_W +: OUT_W] <= sat_to_out(pool_lane[pf]);
          end
        end

        // Advance raster counters for current filter-group
        if (in_group_start) begin
          if (W_cur == 16'd1) begin
            cur_col <= '0;
            if (H_cur == 16'd1)
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
          if (cur_col == (W_cur - 1)) begin
            cur_col <= '0;
            if (cur_row == (H_cur - 1))
              cur_row <= '0;
            else
              cur_row <= cur_row + 1'b1;
          end
          else begin
            cur_col <= cur_col + 1'b1;
          end
        end
      end
    end
  end

endmodule
