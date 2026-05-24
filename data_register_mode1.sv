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

  // --------------------------------------------------------------------------
  // Resource-oriented storage.
  //
  // Previous version stored scalar pixels:
  //   bank[row][col] : DATA_W
  // and reset the whole memory. That created many dynamic scalar muxes and
  // prevented a RAM-friendly implementation.
  //
  // This version stores one physical Mode1 beat per word:
  //   bank[row][col_group] : PV_MAX * DATA_W
  //
  // For the current P128 build PV_MAX=8. Writes are expected to be aligned to
  // PV_MAX pixels; invalid/right-edge lanes are allowed to contain don't-care
  // data because reads are masked by W_cur.
  // --------------------------------------------------------------------------
  localparam int WORD_W      = PV_MAX * DATA_W;
  localparam int PV_LG       = (PV_MAX <= 1) ? 1 : $clog2(PV_MAX);
  localparam int WORD_GROUPS = (W_MAX + PV_MAX - 1) / PV_MAX;
  localparam int GROUP_AW    = (WORD_GROUPS <= 1) ? 1 : $clog2(WORD_GROUPS);

  (* ram_style = "distributed" *) logic [WORD_W-1:0] reg_bank0 [0:K_MAX-1][0:WORD_GROUPS-1];
  (* ram_style = "distributed" *) logic [WORD_W-1:0] reg_bank1 [0:K_MAX-1][0:WORD_GROUPS-1];

  logic [$clog2(K_MAX)-1:0] write_row_idx_clamped;
  logic [$clog2(K_MAX)-1:0] ky_clamped;

  logic [GROUP_AW-1:0] write_group_s;
  logic                write_group_valid_s;

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

  function automatic logic [DATA_W-1:0] get_lane_from_word(
    input logic [WORD_W-1:0] word,
    input int lane
  );
    begin
      if ((lane >= 0) && (lane < PV_MAX))
        get_lane_from_word = word[lane*DATA_W +: DATA_W];
      else
        get_lane_from_word = '0;
    end
  endfunction

  always_comb begin
    int write_group_i;

    if (write_row_idx < K_MAX)
      write_row_idx_clamped = write_row_idx;
    else
      write_row_idx_clamped = '0;

    if (ky < K_MAX)
      ky_clamped = ky;
    else
      ky_clamped = '0;

    write_group_i       = int'(write_x_base >> PV_LG);
    write_group_valid_s = (write_x_base < W_MAX) && (write_group_i < WORD_GROUPS);
    write_group_s       = '0;
    if (write_group_valid_s)
      write_group_s = write_group_i[GROUP_AW-1:0];

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
      // Do not reset reg_bank0/reg_bank1. Valid tags gate all legal reads,
      // and avoiding memory reset keeps the arrays LUTRAM-friendly.
    end
    else begin
      if (load_start) begin
        load_active_q <= 1'b1;
        load_bank_q   <= load_bank_next_s;
        load_c_q      <= load_c;
        load_row_q    <= load_out_row;
        bank_valid_q[load_bank_next_s] <= 1'b0;
      end

      if (write_en &&
          (write_row_idx < K_cur) &&
          (write_row_idx < K_MAX) &&
          write_group_valid_s) begin
        if (write_to_bank1_s) begin
          reg_bank1[write_row_idx_clamped][write_group_s] <= write_data;
        end
        else begin
          reg_bank0[write_row_idx_clamped][write_group_s] <= write_data;
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
    int read_group_i;
    int read_lane_i;
    int pad_x_i;
    logic [WORD_W-1:0] read_word_s;

    pad_x_i = 1;  // fixed K3/P1 horizontal padding currently used by tests

    for (int i = 0; i < PV_MAX; i++) begin
      read_x_i     = int'(out_col) + int'(kx) + i - pad_x_i;
      read_group_i = read_x_i >> PV_LG;
      read_lane_i  = read_x_i & (PV_MAX - 1);
      read_word_s  = '0;

      if (ready_s &&
          (i < Pv_cur) &&
          (ky < K_cur) &&
          (read_x_i >= 0) &&
          (read_x_i < int'(W_cur)) &&
          (read_x_i < W_MAX) &&
          (read_group_i >= 0) &&
          (read_group_i < WORD_GROUPS)) begin
        read_word_s = read_from_bank1_s ?
          reg_bank1[ky_clamped][read_group_i[GROUP_AW-1:0]] :
          reg_bank0[ky_clamped][read_group_i[GROUP_AW-1:0]];

        data_out_logic[i*DATA_W +: DATA_W] = get_lane_from_word(read_word_s, read_lane_i);
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

    if (rst_n && write_en) begin
      if (Pv_cur != PV_MAX[7:0]) begin
        $display("M1_DATAREG_WARN_WORD_FASTPATH t=%0t Pv_cur=%0d PV_MAX=%0d: word-banked data register is optimized for full-PV writes",
                 $time, Pv_cur, PV_MAX);
      end
      if ((write_x_base & (PV_MAX - 1)) != 0) begin
        $display("M1_DATAREG_WARN_UNALIGNED_WRITE t=%0t write_x_base=%0d PV_MAX=%0d",
                 $time, write_x_base, PV_MAX);
      end
    end
  end
`endif

endmodule
