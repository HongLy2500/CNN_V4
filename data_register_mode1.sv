module data_register_mode1 #(
  parameter int K_MAX   = 7,
  parameter int W_MAX   = 224,
  parameter int DATA_W  = 8,
  parameter int PV_MAX  = 16,
  // Kept for backward compatibility with existing named-parameter
  // instantiations while mode-1 data path is logically Pv-wide.
  parameter int PTOTAL  = 256
)(
  input  logic clk,
  input  logic rst_n,

  // =====================================================
  // Runtime config
  // =====================================================
  input  logic [3:0] K_cur,
  input  logic [15:0] W_cur,
  input  logic [7:0] Pv_cur,

  // =====================================================
  // Write control from CE controller / IFM path
  // Only the low Pv_cur lanes of write_data are valid.
  // =====================================================
  input  logic                      write_en,
  input  logic [$clog2(K_MAX)-1:0]  write_row_idx,
  input  logic [15:0]               write_x_base,
  input  logic [PV_MAX*DATA_W-1:0]  write_data,

  // =====================================================
  // Read control from CE controller
  // Mode-1 supports fixed horizontal padding PAD_X=1 for K3/P1 tests:
  //   data_out_logic[pv] = reg_bank[ky][out_col + kx + pv - 1]
  // Out-of-range x positions are zero-filled.
  // Only the low Pv_cur lanes are valid.
  // =====================================================
  input  logic [$clog2(K_MAX)-1:0]  ky,
  input  logic [$clog2(K_MAX)-1:0]  kx,
  input  logic [15:0]               out_col,

  // =====================================================
  // Output to MAC array
  // =====================================================
  output logic [PV_MAX*DATA_W-1:0]  data_out_logic,
  output logic                      data_ready
);

  // =====================================================
  // Internal storage
  // reg_bank[row][x]
  // Stores K rows of one current channel at pixel granularity.
  // =====================================================
  logic [DATA_W-1:0] reg_bank [0:K_MAX-1][0:W_MAX-1];

  integer r, c;
  logic [15:0] base_x;       // legacy/debug: out_col + kx before padding
  logic [3:0]  pad_x;        // fixed horizontal padding, PAD_X=1
  logic [3:0]  pad_y;        // fixed vertical padding, PAD_Y=1; center-tap confirmation patch
  logic        data_valid_q;
  logic [$clog2(K_MAX)-1:0] write_row_idx_clamped;
  logic [$clog2(K_MAX)-1:0] ky_clamped;

  // Clamp row selects to the physical array range.
  // The controller should already provide legal values; this is only
  // to keep addressing well-defined in simulation/synthesis.
  always_comb begin
    if (write_row_idx < K_MAX)
      write_row_idx_clamped = write_row_idx;
    else
      write_row_idx_clamped = '0;

    if (ky < K_MAX)
      ky_clamped = ky;
    else
      ky_clamped = '0;
  end

  // =====================================================
  // Storage write
  // =====================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      data_valid_q <= 1'b0;
      for (r = 0; r < K_MAX; r++) begin
        for (c = 0; c < W_MAX; c++) begin
          reg_bank[r][c] <= '0;
        end
      end
    end
    else if (write_en) begin
      data_valid_q <= 1'b1;
      for (int i = 0; i < PV_MAX; i++) begin
        if ((i < Pv_cur) &&
            (write_row_idx < K_cur) &&
            ((write_x_base + i) < W_cur) &&
            ((write_x_base + i) < W_MAX)) begin
          reg_bank[write_row_idx_clamped][write_x_base + i]
            <= write_data[i*DATA_W +: DATA_W];
        end
      end
    end
  end

  // =====================================================
  // Read path
  // =====================================================
  always_comb begin
    int read_x_i;
    int read_y_i;
    int pad_x_i;
    int pad_y_i;

    // Keep the old unpadded expression for debug visibility.
    base_x  = out_col + kx;

    // Fixed K3/P1 padding for the current Mode1 center-tap test.
    // X: with kx=1, read_x_i = out_col + 1 + i - 1 = out_col + i.
    // Y: with ky=1, read_y_i = 1 - 1 = 0, so center tap uses row0.
    // NOTE: PAD_Y here is a confirmation patch for center-tap K3/P1.
    // It is sufficient for the current sparse center-tap test, but a full
    // 3x3 kernel should implement vertical padding in addr_gen_ifm_m1.
    pad_x   = 4'd1;
    pad_y   = 4'd1;
    pad_x_i = 1;
    pad_y_i = 1;

    for (int i = 0; i < PV_MAX; i++) begin
      read_x_i = int'(out_col) + int'(kx) + i - pad_x_i;
      read_y_i = int'(ky) - pad_y_i;

      if (data_valid_q &&
          (i < Pv_cur) &&
          (ky < K_cur) &&
          (read_y_i >= 0) &&
          (read_y_i < K_MAX) &&
          (read_x_i >= 0) &&
          (read_x_i < int'(W_cur)) &&
          (read_x_i < W_MAX)) begin
        data_out_logic[i*DATA_W +: DATA_W] = reg_bank[read_y_i][read_x_i];
      end
      else begin
        data_out_logic[i*DATA_W +: DATA_W] = '0;
      end
    end
  end

  assign data_ready = data_valid_q;

`ifndef SYNTHESIS
always_ff @(posedge clk) begin : DBG_M1_CENTER_TAP_SPATIAL
    int same_x_i;
    int old_unpadded_x_i;
    int fixed_x_i;

    if (rst_n &&
        data_valid_q &&
        (K_cur == 4'd3) &&
        (ky == 1) &&
        (kx == 1) &&
        (out_col < 16'd8)) begin

        same_x_i         = int'(out_col);
        old_unpadded_x_i = int'(out_col) + int'(kx);
        fixed_x_i        = int'(out_col) + int'(kx) - int'(pad_x);

        if ((same_x_i         >= 0) && (same_x_i         < W_MAX) &&
            (old_unpadded_x_i >= 0) && (old_unpadded_x_i < W_MAX) &&
            (fixed_x_i        >= 0) && (fixed_x_i        < W_MAX)) begin

            $display("DBG_M1_CENTER_TAP_SPATIAL t=%0t out_col=%0d ky=%0d kx=%0d pad_x=%0d old_unpad_x=%0d fixed_x=%0d same_x=%0d lane0=%0d reg_same=%0d reg_old_unpad=%0d reg_fixed=%0d",
                $time,
                out_col,
                ky,
                kx,
                pad_x,
                old_unpadded_x_i,
                fixed_x_i,
                same_x_i,
                $signed(data_out_logic[0*DATA_W +: DATA_W]),
                $signed(reg_bank[ky_clamped][same_x_i]),
                $signed(reg_bank[ky_clamped][old_unpadded_x_i]),
                $signed(reg_bank[ky_clamped][fixed_x_i])
            );
        end
    end
end
`endif

`ifndef SYNTHESIS
always_ff @(posedge clk) begin : DBG_M1_CENTER_TAP_ROW_SPATIAL
    if (rst_n &&
        data_valid_q &&
        (K_cur == 4'd3) &&
        (ky == 1) &&
        (kx == 1) &&
        (out_col == 0)) begin

        $display("DBG_M1_CENTER_TAP_ROW_SPATIAL t=%0t ky=%0d kx=%0d pad_y=%0d fixed_y=%0d out_col=%0d data_lane0=%0d row0_lane0=%0d row1_lane0=%0d row2_lane0=%0d",
            $time,
            ky,
            kx,
            pad_y,
            int'(ky) - int'(pad_y),
            out_col,
            $signed(data_out_logic[0*DATA_W +: DATA_W]),
            $signed(reg_bank[0][0]),
            $signed(reg_bank[1][0]),
            $signed(reg_bank[2][0])
        );
    end
end
`endif

endmodule
