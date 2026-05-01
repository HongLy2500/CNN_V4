module data_register_mode1 #(
  parameter int K_MAX   = 7,
  parameter int W_MAX   = 224,
  parameter int DATA_W  = 8,
  parameter int PV_MAX  = 8,
  // Kept for backward compatibility with existing named-parameter
  // instantiations while mode-1 data path is logically Pv-wide.
  parameter int PTOTAL  = 16
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
  // data_out_logic[pv] = reg_bank[ky][out_col + kx + pv]
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
  logic [15:0] base_x;
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
    base_x = out_col + kx;

    for (int i = 0; i < PV_MAX; i++) begin
      if (data_valid_q &&
          (i < Pv_cur) &&
          (ky < K_cur) &&
          ((base_x + i) < W_cur) &&
          ((base_x + i) < W_MAX)) begin
        data_out_logic[i*DATA_W +: DATA_W] = reg_bank[ky_clamped][base_x + i];
      end
      else begin
        data_out_logic[i*DATA_W +: DATA_W] = '0;
      end
    end
  end

  assign data_ready = data_valid_q;

endmodule
