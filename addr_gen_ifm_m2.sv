module addr_gen_ifm_m2 #(
  parameter int DATA_W = 8,
  parameter int PV_MAX = 8,
  parameter int PC     = 8,
  parameter int PF     = 4,
  parameter int C_MAX  = 64,
  parameter int W_MAX  = 224,
  parameter int H_MAX  = 224,
  parameter int K_MAX  = 7
)(
  input  logic clk,
  input  logic rst_n,

  // --------------------------------------------------
  // Runtime configuration for current mode-2 layer / tile
  //
  // UPDATED CONTRACT:
  // - ifm_buffer is configured in mode 2
  // - input out_row/out_col are GLOBAL output coordinates
  // - local tile coordinate is derived internally from the global column
  //   using the current fixed-width tile model:
  //       tile_col_base_g = floor(out_col_g / PC) * PC
  //       out_col_l       = out_col_g - tile_col_base_g
  //
  // This keeps the interface backward-compatible while removing the old
  // ambiguity where out_col was treated as local.
  // --------------------------------------------------
  input  logic [15:0] K_cur,
  input  logic [15:0] C_cur,
  input  logic [15:0] F_cur,
  input  logic [15:0] H_in,
  input  logic [15:0] W_in,
  input  logic [15:0] Hout_cur,
  input  logic [15:0] Wout_cur,

  // --------------------------------------------------
  // Triggers / loop position from ce_controller_mode2
  //
  // start:
  //   start of the whole mode-2 workload. Generator prefetches the first
  //   tuple of the first output block.
  //
  // out_valid:
  //   flush/output cycle of the current output block. Generator prefetches
  //   the first tuple of the NEXT output block (if any).
  //
  // pass_start_pulse / mac_en:
  //   advance IFM prefetch stream inside the current output block.
  //   This mirrors the successor-issue style used by weight_read_ctrl_mode2.
  //
  // NOTE:
  // - out_row/out_col below are GLOBAL coordinates by contract.
  // --------------------------------------------------
  input  logic        start,
  input  logic        pass_start_pulse,
  input  logic        mac_en,
  input  logic        out_valid,
  input  logic [15:0] out_row,
  input  logic [15:0] out_col,
  input  logic [15:0] f_group,

  // --------------------------------------------------
  // Read port from ifm_buffer (configured in mode 2)
  // --------------------------------------------------
  output logic                     ifm_rd_en,
  output logic [$clog2(C_MAX)-1:0] ifm_rd_bank_base,
  output logic [$clog2(H_MAX)-1:0] ifm_rd_row_idx,
  output logic [$clog2(W_MAX)-1:0] ifm_rd_col_idx,
  input  logic                     ifm_rd_valid,
  input  logic [PV_MAX*DATA_W-1:0] ifm_rd_data,

  // --------------------------------------------------
  // Write port to data_register_mode2
  // --------------------------------------------------
  output logic                     dr_write_en,
  output logic [$clog2(K_MAX)-1:0] dr_write_row_idx,
  output logic [PC*DATA_W-1:0]     dr_write_data,

  // --------------------------------------------------
  // Status back to control_unit
  // --------------------------------------------------
  output logic                     busy,
  output logic                     done,
  output logic                     error,

  // Optional debug / visibility
  output logic [15:0]              dbg_num_fgroup,
  output logic [15:0]              dbg_num_cgroup,
  output logic [15:0]              dbg_block_row,
  output logic [15:0]              dbg_block_col,   // GLOBAL output column
  output logic [15:0]              dbg_issue_cgroup,
  output logic [$clog2(K_MAX)-1:0] dbg_issue_ky,
  output logic [$clog2(K_MAX)-1:0] dbg_issue_kx,
  output logic                     dbg_waiting_for_return
);

  localparam int C_BANK_W = (C_MAX <= 1) ? 1 : $clog2(C_MAX);
  localparam int H_ROW_W  = (H_MAX <= 1) ? 1 : $clog2(H_MAX);
  localparam int W_COL_W  = (W_MAX <= 1) ? 1 : $clog2(W_MAX);
  localparam int K_ROW_W  = (K_MAX <= 1) ? 1 : $clog2(K_MAX);

  logic [15:0] num_fgroup;
  logic [15:0] num_cgroup;
  logic        cfg_valid;

  logic        last_col;
  logic        last_row;
  logic        last_fgroup;
  logic        have_next_block;
  logic [15:0] next_block_row;
  logic [15:0] next_block_col;

  logic [15:0] block_row_q;
  logic [15:0] block_col_q;   // GLOBAL output column
  logic [15:0] issue_cgroup_q;
  logic [K_ROW_W-1:0] issue_ky_q;
  logic [K_ROW_W-1:0] issue_kx_q;
  logic               stream_active_q;

  logic [15:0] succ_cgroup;
  logic [K_ROW_W-1:0] succ_ky;
  logic [K_ROW_W-1:0] succ_kx;
  logic               last_issue;

  logic issue_first;
  logic issue_succ;
  logic issue_any;

  logic [15:0] issue_block_row;
  logic [15:0] issue_block_col;      // GLOBAL output column
  logic [15:0] issue_cgroup;
  logic [K_ROW_W-1:0] issue_ky;
  logic [K_ROW_W-1:0] issue_kx;

  logic [15:0] issue_bank_base16;
  logic [15:0] issue_abs_row16;
  logic [15:0] issue_abs_col_g16;    // GLOBAL IFM input column
  logic [15:0] issue_tile_base_g16;  // inferred tile base (GLOBAL)
  logic [15:0] issue_col_sel_l16;    // LOCAL IFM column inside current tile
  logic        issue_addr_valid;
  logic        final_out_valid;

  // Metadata delayed to align with 1-cycle ifm_buffer read latency.
  logic               ret_valid_q;
  logic [K_ROW_W-1:0] ret_row_q;

  integer lane_i;

  // --------------------------------------------------
  // Derived runtime values
  // --------------------------------------------------
  always_comb begin
    if (PF != 0)
      num_fgroup = (F_cur + PF - 1) / PF;
    else
      num_fgroup = 16'd0;

    if (PC != 0)
      num_cgroup = (C_cur + PC - 1) / PC;
    else
      num_cgroup = 16'd0;
  end

  always_comb begin
    cfg_valid = 1'b0;
    if ((K_cur != 0) && (K_cur <= K_MAX) &&
        (C_cur != 0) && (C_cur <= C_MAX) &&
        (F_cur != 0) &&
        (H_in != 0) && (H_in <= H_MAX) &&
        (W_in != 0) && (W_in <= W_MAX) &&
        (Hout_cur != 0) &&
        (Wout_cur != 0) &&
        (num_cgroup != 0) &&
        (num_fgroup != 0)) begin
      cfg_valid = 1'b1;
    end
  end

  always_comb begin
    last_col    = (Wout_cur == 0) ? 1'b1 : (out_col == (Wout_cur - 1));
    last_row    = (Hout_cur == 0) ? 1'b1 : (out_row == (Hout_cur - 1));
    last_fgroup = (num_fgroup == 0) ? 1'b1 : (f_group == (num_fgroup - 1));

    have_next_block = !(last_col && last_row && last_fgroup);

    next_block_row = out_row;
    next_block_col = out_col;
    if (!last_col) begin
      next_block_col = out_col + 16'd1;
    end
    else begin
      next_block_col = 16'd0;
      if (!last_row) begin
        next_block_row = out_row + 16'd1;
      end
      else begin
        next_block_row = 16'd0;
      end
    end
  end

  // --------------------------------------------------
  // Issue-state successor within one output block
  // tuple order inside one block:
  //   for c_group
  //     for ky
  //       for kx
  // --------------------------------------------------
  always_comb begin
    last_issue = (issue_cgroup_q == (num_cgroup - 1)) &&
                 (issue_ky_q     == (K_cur - 1))       &&
                 (issue_kx_q     == (K_cur - 1));

    succ_cgroup = issue_cgroup_q;
    succ_ky     = issue_ky_q;
    succ_kx     = issue_kx_q;

    if (issue_kx_q != (K_cur - 1)) begin
      succ_kx = issue_kx_q + 1'b1;
    end
    else begin
      succ_kx = '0;
      if (issue_ky_q != (K_cur - 1)) begin
        succ_ky = issue_ky_q + 1'b1;
      end
      else begin
        succ_ky     = '0;
        succ_cgroup = issue_cgroup_q + 16'd1;
      end
    end
  end

  // --------------------------------------------------
  // Issue scheduling
  // --------------------------------------------------
  always_comb begin
    issue_first = 1'b0;
    issue_succ  = 1'b0;

    if (cfg_valid) begin
      if (start) begin
        issue_first = 1'b1;
      end
      else if (stream_active_q && out_valid && have_next_block) begin
        issue_first = 1'b1;
      end
      else if (stream_active_q && (pass_start_pulse || mac_en) && !last_issue) begin
        issue_succ = 1'b1;
      end
    end
  end

  assign issue_any = issue_first || issue_succ;

  always_comb begin
    issue_block_row = block_row_q;
    issue_block_col = block_col_q;
    issue_cgroup    = issue_cgroup_q;
    issue_ky        = issue_ky_q;
    issue_kx        = issue_kx_q;

    if (issue_first) begin
      if (start) begin
        issue_block_row = 16'd0;
        issue_block_col = 16'd0;
      end
      else begin
        issue_block_row = next_block_row;
        issue_block_col = next_block_col;
      end
      issue_cgroup = 16'd0;
      issue_ky     = '0;
      issue_kx     = '0;
    end
    else if (issue_succ) begin
      issue_block_row = block_row_q;
      issue_block_col = block_col_q;
      issue_cgroup    = succ_cgroup;
      issue_ky        = succ_ky;
      issue_kx        = succ_kx;
    end
  end

  // --------------------------------------------------
  // Address mapping into ifm_buffer mode 2
  //
  // For one issued tuple:
  //   bank_base      = c_group * PC
  //   abs_row_g      = out_row_g(block) + ky
  //   abs_col_g      = out_col_g(block) + kx
  //   tile_base_g    = floor(abs_col_g / PC) * PC
  //   col_sel_local  = abs_col_g - tile_base_g
  //
  // This keeps GLOBAL and LOCAL meanings separate:
  // - abs_row_g / abs_col_g are feature-map coordinates
  // - col_sel_local is the address used inside the currently addressed PC-wide
  //   segment
  //
  // Boundary / tile rollover policy:
  // - local tile coordinate is derived from the ABSOLUTE input column of each
  //   issued tuple, not from the output block origin.
  // - when kx pushes the window across a PC-wide boundary, the generator
  //   automatically rolls over to the next segment by recomputing tile_base_g
  //   from abs_col_g.
  // - control/refill logic must still ensure that the segment containing the
  //   requested abs_col_g is resident in ifm_buffer.
  // --------------------------------------------------
  always_comb begin
    issue_bank_base16 = issue_cgroup * PC;
    issue_abs_row16   = issue_block_row + issue_ky;
    issue_abs_col_g16 = issue_block_col + issue_kx;

    if (PC != 0)
      issue_tile_base_g16 = (issue_abs_col_g16 / PC) * PC;
    else
      issue_tile_base_g16 = 16'd0;

    issue_col_sel_l16 = issue_abs_col_g16 - issue_tile_base_g16;

    issue_addr_valid  = 1'b1;
    if (!issue_any)
      issue_addr_valid = 1'b0;
    else if (issue_bank_base16 >= C_MAX)
      issue_addr_valid = 1'b0;
    else if (issue_abs_row16 >= H_in)
      issue_addr_valid = 1'b0;
    else if (issue_abs_col_g16 >= W_in)
      issue_addr_valid = 1'b0;
    else if ((PC == 0) || (issue_col_sel_l16 >= PC))
      issue_addr_valid = 1'b0;
  end

  assign ifm_rd_en        = issue_any && issue_addr_valid;
  assign ifm_rd_bank_base = issue_bank_base16[C_BANK_W-1:0];
  assign ifm_rd_row_idx   = issue_abs_row16[H_ROW_W-1:0];
  assign ifm_rd_col_idx   = issue_col_sel_l16[W_COL_W-1:0];

  // --------------------------------------------------
  // data_register_mode2 write side
  // low PC lanes from ifm_buffer are meaningful in mode 2
  // --------------------------------------------------
  assign dr_write_en      = ret_valid_q && ifm_rd_valid;
  assign dr_write_row_idx = ret_row_q;

  always_comb begin
    dr_write_data = '0;
    for (lane_i = 0; lane_i < PC; lane_i++) begin
      dr_write_data[lane_i*DATA_W +: DATA_W] = ifm_rd_data[lane_i*DATA_W +: DATA_W];
    end
  end

  // --------------------------------------------------
  // Status / debug
  // --------------------------------------------------
  assign busy                   = stream_active_q;
  assign dbg_num_fgroup         = num_fgroup;
  assign dbg_num_cgroup         = num_cgroup;
  assign dbg_block_row          = block_row_q;
  assign dbg_block_col          = block_col_q; // GLOBAL output column
  assign dbg_issue_cgroup       = issue_cgroup_q;
  assign dbg_issue_ky           = issue_ky_q;
  assign dbg_issue_kx           = issue_kx_q;
  assign dbg_waiting_for_return = stream_active_q && ret_valid_q && !ifm_rd_valid;

  assign final_out_valid = stream_active_q && out_valid && !have_next_block;

  // --------------------------------------------------
  // State / sequencing
  // --------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      block_row_q     <= 16'd0;
      block_col_q     <= 16'd0;
      issue_cgroup_q  <= 16'd0;
      issue_ky_q      <= '0;
      issue_kx_q      <= '0;
      stream_active_q <= 1'b0;
      ret_valid_q     <= 1'b0;
      ret_row_q       <= '0;
      done            <= 1'b0;
      error           <= 1'b0;
    end
    else begin
      done  <= 1'b0;
      error <= 1'b0;

      // Delay metadata for the read issued in this cycle.
      ret_valid_q <= ifm_rd_en;
      ret_row_q   <= issue_ky;

      if (start) begin
        if (!cfg_valid) begin
          stream_active_q <= 1'b0;
          error           <= 1'b1;
        end
        else if (!issue_addr_valid) begin
          stream_active_q <= 1'b0;
          error           <= 1'b1;
        end
        else begin
          stream_active_q <= 1'b1;
        end
      end

      if (issue_any) begin
        if (!issue_addr_valid) begin
          stream_active_q <= 1'b0;
          error           <= 1'b1;
        end
        else begin
          block_row_q    <= issue_block_row;
          block_col_q    <= issue_block_col;
          issue_cgroup_q <= issue_cgroup;
          issue_ky_q     <= issue_ky;
          issue_kx_q     <= issue_kx;

          if (!stream_active_q)
            stream_active_q <= 1'b1;
        end
      end

      if (final_out_valid) begin
        stream_active_q <= 1'b0;
        done            <= 1'b1;
      end
    end
  end

endmodule
