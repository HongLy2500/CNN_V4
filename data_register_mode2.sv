module data_register_mode2 #(
  parameter int K_MAX   = 7,
  parameter int DATA_W  = 8,
  parameter int PC      = 8
)(
  input  logic clk,
  input  logic rst_n,

  // Runtime config
  input  logic [3:0]  K_cur,
  input  logic [7:0]  C_cur,
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

  integer r;

  // ---------------------------------------------
  // Clamp row indices and compute current channel base
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
  end

  // ---------------------------------------------
  // Write path
  // Invalid channel lanes in a partial C group are explicitly cleared.
  // This prevents X/rubbish lanes from being stored and later summed by
  // the Mode-2 PC reduction MAC array.
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
      if (write_en && (write_row_idx < K_cur)) begin
        for (int pc_i = 0; pc_i < PC; pc_i++) begin
          if ((c_base_cur + pc_i) < C_cur) begin
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
  // ---------------------------------------------
  always_comb begin
    for (int pc_i = 0; pc_i < PC; pc_i++) begin
      if ((read_row_idx < K_cur) && ((c_base_cur + pc_i) < C_cur)) begin
        data_out[pc_i*DATA_W +: DATA_W]
          = reg_bank[read_row_idx_clamped][pc_i];
      end
      else begin
        data_out[pc_i*DATA_W +: DATA_W] = '0;
      end
    end
  end

endmodule
