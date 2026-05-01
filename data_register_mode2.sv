module data_register_mode2 #(
  parameter int K_MAX   = 7,
  parameter int DATA_W  = 8,
  parameter int PC      = 8
)(
  input  logic clk,
  input  logic rst_n,

  // Runtime config
  input  logic [15:0] K_cur,

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

  integer r;

  // ---------------------------------------------
  // Clamp row indices
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
  end

  // ---------------------------------------------
  // Write path
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
          reg_bank[write_row_idx_clamped][pc_i]
            <= write_data[pc_i*DATA_W +: DATA_W];
        end
      end
    end
  end

  // ---------------------------------------------
  // Read path
  // ---------------------------------------------
  always_comb begin
    for (int pc_i = 0; pc_i < PC; pc_i++) begin
      if (read_row_idx < K_cur)
        data_out[pc_i*DATA_W +: DATA_W]
          = reg_bank[read_row_idx_clamped][pc_i];
      else
        data_out[pc_i*DATA_W +: DATA_W] = '0;
    end
  end

endmodule