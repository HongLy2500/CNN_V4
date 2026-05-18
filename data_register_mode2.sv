module data_register_mode2 #(
  parameter int K_MAX   = 7,
  parameter int DATA_W  = 8,
  parameter int PC      = 16
)(
  input  logic clk,
  input  logic rst_n,

  // Runtime config
  input  logic [3:0]  K_cur,
  input  logic [9:0]  C_cur,
  input  logic [15:0] c_group,

  // ---------------------------------------------
  // Write side from control unit
  // write one PC-wide vector into one row
  // ---------------------------------------------
  input  logic                     write_en,
  input  logic [$clog2(K_MAX)-1:0] write_row_idx,
  input  logic [PC*DATA_W-1:0]     write_data,

  // ---------------------------------------------
  // Read side to MAC array
  // read one PC-wide vector from one row
  // ---------------------------------------------
  input  logic [$clog2(K_MAX)-1:0] read_row_idx,
  output logic [PC*DATA_W-1:0]     data_out
);

  logic [DATA_W-1:0] reg_bank [0:K_MAX-1][0:PC-1];

  logic [$clog2(K_MAX)-1:0] write_row_idx_clamped;
  logic [$clog2(K_MAX)-1:0] read_row_idx_clamped;
  logic [15:0]              c_base_cur;
  logic [PC-1:0]            pc_lane_valid_s;

  integer r;

  // ---------------------------------------------
  // Clamp row indices and compute current channel base/lane mask
  // ---------------------------------------------
  always_comb begin
    if (write_row_idx < K_MAX)
      write_row_idx_clamped = write_row_idx;
    else
      write_row_idx_clamped = '0;

    if (read_row_idx < K_MAX)
      read_row_idx_clamped = read_row_idx;
    else
      read_row_idx_clamped = '0;

    c_base_cur = c_group * PC;

    // Mode 2 reduces across PC channel lanes.  For the last C group,
    // only lanes where (c_group*PC + lane) < C_cur are valid.
    // All invalid lanes must be forced to zero before reaching the MAC
    // reduction tree; otherwise stale/X lanes can poison the whole sum.
    pc_lane_valid_s = '0;
    for (int pc_i = 0; pc_i < PC; pc_i++) begin
      if ((c_base_cur + pc_i) < C_cur)
        pc_lane_valid_s[pc_i] = 1'b1;
    end
  end

  // ---------------------------------------------
  // Write path
  // ---------------------------------------------
  // Robust invalid-lane policy:
  //   * On every cycle, lanes that are invalid for the current C group are
  //     scrubbed to zero in every stored row.
  //   * When a row is written, only valid lanes capture write_data; invalid
  //     lanes are written as zero.
  //
  // This is intentionally stronger than only masking data_out.  It keeps
  // reg_bank itself clean in waveform/debug and prevents stale/X values from
  // being observed if a raw/internal signal is inspected or reused.
  // ---------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (r = 0; r < K_MAX; r++) begin
        for (int pc_i = 0; pc_i < PC; pc_i++) begin
          reg_bank[r][pc_i] <= '0;
        end
      end
    end
    else begin
      // Scrub invalid PC lanes for the current channel group in all rows.
      // This guarantees that partial-C groups such as C=40, PC=32,
      // c_group=1 keep lanes 8..31 at zero even when the current row is not
      // being written in this cycle.
      for (r = 0; r < K_MAX; r++) begin
        for (int pc_i = 0; pc_i < PC; pc_i++) begin
          if (!pc_lane_valid_s[pc_i]) begin
            reg_bank[r][pc_i] <= '0;
          end
        end
      end

      if (write_en && (write_row_idx < K_cur)) begin
        for (int pc_i = 0; pc_i < PC; pc_i++) begin
          if (pc_lane_valid_s[pc_i]) begin
            reg_bank[write_row_idx_clamped][pc_i]
              <= write_data[pc_i*DATA_W +: DATA_W];
          end
          else begin
            reg_bank[write_row_idx_clamped][pc_i] <= '0;
          end
        end
      end
    end
  end

  // ---------------------------------------------
  // Read path
  // Return zero for invalid K rows or invalid channel lanes.
  // Every lane is assigned explicitly so invalid lanes cannot retain X.
  // ---------------------------------------------
  always_comb begin
    for (int pc_i = 0; pc_i < PC; pc_i++) begin
      data_out[pc_i*DATA_W +: DATA_W] = '0;

      if ((read_row_idx < K_cur) && pc_lane_valid_s[pc_i]) begin
        data_out[pc_i*DATA_W +: DATA_W]
          = reg_bank[read_row_idx_clamped][pc_i];
      end
    end
  end


always @(posedge clk) begin
  #1;
  if (rst_n) begin
    for (int pc_i = 0; pc_i < PC; pc_i++) begin
      if (!pc_lane_valid_s[pc_i]) begin
        if (data_out[pc_i*DATA_W +: DATA_W] !== '0) begin
          $display("DRM2_ASSERT_FAIL t=%0t c_group=%0d C_cur=%0d pc=%0d valid=%0b data_out=%h reg_bank0=%h",
            $time, c_group, C_cur, pc_i, pc_lane_valid_s[pc_i],
            data_out[pc_i*DATA_W +: DATA_W],
            reg_bank[0][pc_i]);
          $fatal;
        end
      end
    end
  end
end


endmodule
