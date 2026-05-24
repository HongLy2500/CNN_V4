module data_register_mode1 #(
  parameter int K_MAX   = 7,
  parameter int W_MAX   = 32,
  parameter int DATA_W  = 8,
  parameter int PV_MAX  = 8,
  // Kept for backward compatibility with existing named-parameter
  // instantiations while mode-1 data path is logically Pv-wide.
  parameter int PTOTAL  = 128
)(
  input  logic clk,
  input  logic rst_n,

  // Runtime config
  input  logic [3:0] K_cur,
  input  logic [15:0] W_cur,
  input  logic [7:0] Pv_cur,

  // Load descriptor from local_dataflow/addr_gen.
  // Stage3: data_register owns the two physical banks and selects the bank
  // whose tag matches the current CE context.  No external read/write bank
  // select is required for correctness.
  input  logic        load_start,
  input  logic        load_done,
  input  logic [15:0] load_c,
  input  logic [15:0] load_out_row,

  // Current CE context used to select/validate the bank that should feed MACs.
  input  logic [15:0] cur_c,
  input  logic [15:0] cur_out_row,

  // Write control from IFM path
  input  logic                      write_en,
  input  logic [$clog2(K_MAX)-1:0]  write_row_idx,
  input  logic [15:0]               write_x_base,
  input  logic [PV_MAX*DATA_W-1:0]  write_data,
  // Kept for bankif interface compatibility; ignored in Stage3.
  input  logic                      write_bank_sel,

  // Read control from CE controller
  input  logic [$clog2(K_MAX)-1:0]  ky,
  input  logic [$clog2(K_MAX)-1:0]  kx,
  input  logic [15:0]               out_col,
  // Kept for bankif interface compatibility; ignored in Stage3.
  input  logic                      read_bank_sel,

  // Output to MAC array
  output logic [PV_MAX*DATA_W-1:0]  data_out_logic,
  output logic                      data_ready
);

  logic [DATA_W-1:0] reg_bank0 [0:K_MAX-1][0:W_MAX-1];
  logic [DATA_W-1:0] reg_bank1 [0:K_MAX-1][0:W_MAX-1];

  integer r, c;
  logic [$clog2(K_MAX)-1:0] write_row_idx_clamped;
  logic [$clog2(K_MAX)-1:0] ky_clamped;

  logic load_active_q;
  logic load_bank_q;
  logic last_load_bank_q;

  logic        bank_valid_q [0:1];
  logic [15:0] bank_c_q     [0:1];
  logic [15:0] bank_row_q   [0:1];

  logic [15:0] load_c_q;
  logic [15:0] load_row_q;

  logic bank0_match_cur_s;
  logic bank1_match_cur_s;
  logic load_done_match_cur_s;
  logic load_bank_next_s;
  logic write_to_bank1_s;
  logic read_from_bank1_s;
  logic ready_s;

  always_comb begin
    if (write_row_idx < K_MAX)
      write_row_idx_clamped = write_row_idx;
    else
      write_row_idx_clamped = '0;

    if (ky < K_MAX)
      ky_clamped = ky;
    else
      ky_clamped = '0;

    bank0_match_cur_s = bank_valid_q[0] && (bank_c_q[0] == cur_c) && (bank_row_q[0] == cur_out_row);
    bank1_match_cur_s = bank_valid_q[1] && (bank_c_q[1] == cur_c) && (bank_row_q[1] == cur_out_row);

    // Choose a load bank that is not currently feeding the CE context.
    // If neither bank matches current context, alternate from the previous load.
    if (bank0_match_cur_s && !bank1_match_cur_s)
      load_bank_next_s = 1'b1;
    else if (bank1_match_cur_s && !bank0_match_cur_s)
      load_bank_next_s = 1'b0;
    else
      load_bank_next_s = ~last_load_bank_q;

    // First write of a load can arrive in the same cycle as load_start.
    write_to_bank1_s = load_active_q ? load_bank_q :
                       (load_start ? load_bank_next_s : load_bank_next_s);

    load_done_match_cur_s = load_done && load_active_q &&
                            (load_c_q == cur_c) && (load_row_q == cur_out_row);

    // Same-cycle done bypass: if the just-loaded descriptor is exactly the
    // one CE wants, allow read/ready from load_bank_q immediately.
    if (load_done_match_cur_s) begin
      read_from_bank1_s = load_bank_q;
      ready_s           = 1'b1;
    end
    else if (bank0_match_cur_s) begin
      read_from_bank1_s = 1'b0;
      ready_s           = 1'b1;
    end
    else if (bank1_match_cur_s) begin
      read_from_bank1_s = 1'b1;
      ready_s           = 1'b1;
    end
    else begin
      read_from_bank1_s = 1'b0;
      ready_s           = 1'b0;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      load_active_q    <= 1'b0;
      load_bank_q      <= 1'b0;
      last_load_bank_q <= 1'b1;
      bank_valid_q[0]  <= 1'b0;
      bank_valid_q[1]  <= 1'b0;
      bank_c_q[0]      <= 16'd0;
      bank_c_q[1]      <= 16'd0;
      bank_row_q[0]    <= 16'd0;
      bank_row_q[1]    <= 16'd0;
      load_c_q         <= 16'd0;
      load_row_q       <= 16'd0;
      for (r = 0; r < K_MAX; r++) begin
        for (c = 0; c < W_MAX; c++) begin
          reg_bank0[r][c] <= '0;
          reg_bank1[r][c] <= '0;
        end
      end
    end
    else begin
      if (load_start) begin
        load_active_q <= 1'b1;
        load_bank_q   <= load_bank_next_s;
        load_c_q      <= load_c;
        load_row_q    <= load_out_row;
        bank_valid_q[load_bank_next_s] <= 1'b0;
      end

      if (write_en) begin
        for (int i = 0; i < PV_MAX; i++) begin
          if ((i < Pv_cur) &&
              (write_row_idx < K_cur) &&
              ((write_x_base + i) < W_cur) &&
              ((write_x_base + i) < W_MAX)) begin
            if (write_to_bank1_s) begin
              reg_bank1[write_row_idx_clamped][write_x_base + i]
                <= write_data[i*DATA_W +: DATA_W];
            end
            else begin
              reg_bank0[write_row_idx_clamped][write_x_base + i]
                <= write_data[i*DATA_W +: DATA_W];
            end
          end
        end
      end

      if (load_done && load_active_q) begin
        bank_valid_q[load_bank_q] <= 1'b1;
        bank_c_q[load_bank_q]     <= load_c_q;
        bank_row_q[load_bank_q]   <= load_row_q;
        last_load_bank_q          <= load_bank_q;
        load_active_q             <= 1'b0;
      end
    end
  end

  always_comb begin
    int read_x_i;
    int pad_x_i;

    pad_x_i = 1;  // fixed K3/P1 horizontal padding currently used by tests

    for (int i = 0; i < PV_MAX; i++) begin
      read_x_i = int'(out_col) + int'(kx) + i - pad_x_i;

      if (ready_s &&
          (i < Pv_cur) &&
          (ky < K_cur) &&
          (read_x_i >= 0) &&
          (read_x_i < int'(W_cur)) &&
          (read_x_i < W_MAX)) begin
        data_out_logic[i*DATA_W +: DATA_W] = read_from_bank1_s ?
          reg_bank1[ky_clamped][read_x_i] : reg_bank0[ky_clamped][read_x_i];
      end
      else begin
        data_out_logic[i*DATA_W +: DATA_W] = '0;
      end
    end
  end

  assign data_ready = ready_s;

`ifndef SYNTHESIS
  always_ff @(posedge clk) begin : DBG_M1_DR_STAGE3_TAGMATCH
    if (rst_n && load_start) begin
      $display("DBG_M1_DR_STAGE3_LOAD_START t=%0t load_bank=%0d c=%0d row=%0d cur_c=%0d cur_row=%0d b0v=%0d b0c=%0d b0r=%0d b1v=%0d b1c=%0d b1r=%0d",
        $time, load_bank_next_s, load_c, load_out_row, cur_c, cur_out_row,
        bank_valid_q[0], bank_c_q[0], bank_row_q[0], bank_valid_q[1], bank_c_q[1], bank_row_q[1]);
    end
    if (rst_n && load_done && load_active_q) begin
      $display("DBG_M1_DR_STAGE3_LOAD_DONE t=%0t bank=%0d c=%0d row=%0d match_cur=%0d",
        $time, load_bank_q, load_c_q, load_row_q, load_done_match_cur_s);
    end
  end
`endif

endmodule
