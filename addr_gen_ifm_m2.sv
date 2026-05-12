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
  // - local tile coordinate is derived from the explicit resident tile
  //   base supplied by local_dataflow_manager/control:
  //       col_l = global_input_col - tile_col_base_g
  // - the generator rejects reads outside the currently resident WT=PC tile;
  //   this prevents the old behavior where global_col % PC silently wrapped
  //   to tile 0 when compute crossed a tile boundary.
  // --------------------------------------------------
  input  logic [3:0] K_cur,
  input  logic [7:0] C_cur,
  input  logic [7:0] F_cur,
  input  logic [15:0] H_in,
  input  logic [15:0] W_in,
  input  logic [15:0] Hout_cur,
  input  logic [15:0] Wout_cur,

  // GLOBAL column base of the horizontal Mode-2 tile currently resident
  // in IFM buffer. Mode 2 IFM buffer stores only one WT=PC tile at a time:
  //   bank = local column col_l = global_col - tile_col_base_g
  //   addr = row*C_GRP_MAX + cgrp
  //   lane = channel within PC group
  input  logic [15:0] tile_col_base_g,

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

  // Col-pair successor helpers for the updated Mode-2 traversal order.
  // These affect only the prefetch target of the next output block; they do
  // not change tuple order inside one output block or resident-tile validity.
  logic [15:0] out_col_pair_base_s;
  logic        out_col_pair_first_s;
  logic        out_col_pair_has_second_s;
  logic        out_col_pair_has_next_s;

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

    // Match ce_controller_mode2 col-pair-major, row-inner traversal:
    //   (row, pair+0) -> (row, pair+1 if valid)
    //   then next row at pair+0
    //   after the last row, advance to pair+2.
    // This changes only successor prefetch order.  The resident-tile read
    // validity below remains the fixed [tile_col_base_g, tile_col_base_g+PC)
    // contract from the Test-A-passing design.
    out_col_pair_base_s       = {out_col[15:1], 1'b0};
    out_col_pair_first_s      = (out_col == out_col_pair_base_s);
    out_col_pair_has_second_s = ((out_col_pair_base_s + 16'd1) < Wout_cur);
    out_col_pair_has_next_s   = ((out_col_pair_base_s + 16'd2) < Wout_cur);

    have_next_block = !(last_col && last_row && last_fgroup);

    next_block_row = out_row;
    next_block_col = out_col;

    if (out_col_pair_first_s && out_col_pair_has_second_s) begin
      // Same row, second column of the current pair.
      next_block_col = out_col_pair_base_s + 16'd1;
    end
    else if (!last_row) begin
      // Next row, restart at the first column of the same pair.
      next_block_row = out_row + 16'd1;
      next_block_col = out_col_pair_base_s;
    end
    else if (out_col_pair_has_next_s) begin
      // Finished all rows of this pair; move to the next pair.
      next_block_row = 16'd0;
      next_block_col = out_col_pair_base_s + 16'd2;
    end
    else begin
      // Finished the spatial raster for this f_group.  If another f_group
      // remains, have_next_block is still true and the next block starts at
      // row 0 / col 0, matching the CE controller.
      next_block_row = 16'd0;
      next_block_col = 16'd0;
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
  //   tile_base_g    = explicit tile_col_base_g from local_dataflow_manager
  //   col_sel_local  = abs_col_g - tile_col_base_g
  //
  // This keeps GLOBAL and LOCAL meanings separate:
  // - abs_row_g / abs_col_g are feature-map coordinates
  // - col_sel_local is the IFM-buffer physical bank index for mode 2
  //
  // Tile-resident policy:
  // - only columns inside [tile_col_base_g, tile_col_base_g + PC) may be read.
  // - do NOT use abs_col_g % PC here, because that would silently read from
  //   the wrong resident tile for W > PC.
  // - K>1 halo/cross-tile behavior must be handled by tile scheduling/control;
  //   this module intentionally flags such out-of-resident-tile accesses.
  // --------------------------------------------------
  always_comb begin
    issue_bank_base16 = issue_cgroup * PC;
    issue_abs_row16   = issue_block_row + issue_ky;
    issue_abs_col_g16 = issue_block_col + issue_kx;

    issue_tile_base_g16 = tile_col_base_g;

    if (issue_abs_col_g16 >= tile_col_base_g)
      issue_col_sel_l16 = issue_abs_col_g16 - tile_col_base_g;
    else
      issue_col_sel_l16 = 16'hffff;

    issue_addr_valid  = 1'b1;
    if (!issue_any)
      issue_addr_valid = 1'b0;
    else if (issue_bank_base16 >= C_MAX)
      issue_addr_valid = 1'b0;
    else if (issue_abs_row16 >= H_in)
      issue_addr_valid = 1'b0;
    else if (issue_abs_col_g16 >= W_in)
      issue_addr_valid = 1'b0;
    else if (tile_col_base_g >= W_in)
      issue_addr_valid = 1'b0;
    else if ((PC == 0) || ((tile_col_base_g % PC) != 0))
      issue_addr_valid = 1'b0;
    else if (issue_abs_col_g16 < tile_col_base_g)
      issue_addr_valid = 1'b0;
    else if (issue_abs_col_g16 >= (tile_col_base_g + PC))
      issue_addr_valid = 1'b0;
    else if (issue_col_sel_l16 >= PC)
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

`ifndef SYNTHESIS
  // Simulation-only sanity checks for the explicit resident tile contract.
  always_ff @(posedge clk) begin
    if (rst_n && cfg_valid && start) begin
      if ((PC != 0) && ((tile_col_base_g % PC) != 0)) begin
        $display("ERROR: addr_gen_ifm_m2 tile_col_base_g=%0d is not PC-aligned PC=%0d at t=%0t",
                 tile_col_base_g, PC, $time);
      end
      if (tile_col_base_g >= W_in) begin
        $display("ERROR: addr_gen_ifm_m2 tile_col_base_g=%0d outside W_in=%0d at t=%0t",
                 tile_col_base_g, W_in, $time);
      end
    end
  end
`endif

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


always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    // no-op
  end else begin

    // Print near the right boundary of resident PC window, and print all invalid issues.
    if (issue_any &&
        ((!issue_addr_valid) ||
         (issue_abs_col_g16 + 16'd2 >= (tile_col_base_g + PC)))) begin

      $display("DBG_M2_AG_ISSUE t=%0t start=%0b pass=%0b mac=%0b out_v=%0b fgrp=%0d block_row=%0d block_col=%0d issue_cgrp=%0d ky=%0d kx=%0d abs_row=%0d abs_col=%0d tile_base=%0d PC=%0d col_l_calc=%0d addr_valid=%0b ifm_rd_en=%0b ifm_rd_valid=%0b dr_wr=%0b",
               $time,
               start,
               pass_start_pulse,
               mac_en,
               out_valid,
               f_group,
               issue_block_row,
               issue_block_col,
               issue_cgroup,
               issue_ky,
               issue_kx,
               issue_abs_row16,
               issue_abs_col_g16,
               tile_col_base_g,
               PC,
               issue_col_sel_l16,
               issue_addr_valid,
               ifm_rd_en,
               ifm_rd_valid,
               dr_write_en);
    end

    if (issue_any && !issue_addr_valid) begin
      if (!cfg_valid) begin
        $display("DBG_M2_AG_INVALID_REASON t=%0t reason=CFG_INVALID K=%0d C=%0d F=%0d H_in=%0d W_in=%0d Hout=%0d Wout=%0d num_cgroup=%0d num_fgroup=%0d",
                 $time, K_cur, C_cur, F_cur, H_in, W_in, Hout_cur, Wout_cur,
                 num_cgroup, num_fgroup);
      end

      if (issue_bank_base16 >= C_MAX) begin
        $display("DBG_M2_AG_INVALID_REASON t=%0t reason=BANK_BASE_RANGE bank_base=%0d C_MAX=%0d",
                 $time, issue_bank_base16, C_MAX);
      end

      if (issue_abs_row16 >= H_in) begin
        $display("DBG_M2_AG_INVALID_REASON t=%0t reason=ROW_RANGE abs_row=%0d H_in=%0d",
                 $time, issue_abs_row16, H_in);
      end

      if (issue_abs_col_g16 >= W_in) begin
        $display("DBG_M2_AG_INVALID_REASON t=%0t reason=COL_RANGE abs_col=%0d W_in=%0d",
                 $time, issue_abs_col_g16, W_in);
      end

      if (tile_col_base_g >= W_in) begin
        $display("DBG_M2_AG_INVALID_REASON t=%0t reason=TILE_BASE_RANGE tile_base=%0d W_in=%0d",
                 $time, tile_col_base_g, W_in);
      end

      if ((PC == 0) || ((tile_col_base_g % PC) != 0)) begin
        $display("DBG_M2_AG_INVALID_REASON t=%0t reason=TILE_BASE_ALIGN tile_base=%0d PC=%0d",
                 $time, tile_col_base_g, PC);
      end

      if (issue_abs_col_g16 < tile_col_base_g) begin
        $display("DBG_M2_AG_INVALID_REASON t=%0t reason=COL_BEFORE_TILE abs_col=%0d tile_base=%0d",
                 $time, issue_abs_col_g16, tile_col_base_g);
      end

      if (issue_abs_col_g16 >= (tile_col_base_g + PC)) begin
        $display("DBG_M2_AG_INVALID_REASON t=%0t reason=COL_AFTER_RESIDENT abs_col=%0d tile_base=%0d PC=%0d expected_col_l_mod=%0d",
                 $time,
                 issue_abs_col_g16,
                 tile_col_base_g,
                 PC,
                 (PC == 0) ? 16'd0 : (issue_abs_col_g16 % PC));
      end

      if (issue_col_sel_l16 >= PC) begin
        $display("DBG_M2_AG_INVALID_REASON t=%0t reason=LOCAL_COL_RANGE col_l=%0d PC=%0d",
                 $time, issue_col_sel_l16, PC);
      end
    end

    if (error) begin
      $display("DBG_M2_AG_ERROR t=%0t block_row_q=%0d block_col_q=%0d issue_cgroup_q=%0d issue_ky_q=%0d issue_kx_q=%0d waiting=%0b",
               $time,
               block_row_q,
               block_col_q,
               issue_cgroup_q,
               issue_ky_q,
               issue_kx_q,
               dbg_waiting_for_return);
    end
  end
end

// -----------------------------------------------------------------------------
// DEBUG: Mode2 addr_gen focused issue monitor
// Purpose:
//   Show exact requested abs_col and whether addr_gen blocks it before IFM read.
// -----------------------------------------------------------------------------
// Enable with: +define+DBG_M2_ADDRGEN_FOCUS
// -----------------------------------------------------------------------------

always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    // no-op
  end else begin
    if (issue_any &&
        ((!issue_addr_valid) ||
         (issue_abs_col_g16 >= 16'd28))) begin

      $display("DBG_M2_AG_FOCUS t=%0t issue_any=%0b addr_valid=%0b out_block_row=%0d out_block_col=%0d issue_cgrp=%0d ky=%0d kx=%0d abs_row=%0d abs_col=%0d tile_base=%0d PC=%0d col_l_calc=%0d expected_bank_mod=%0d ifm_rd_en=%0b ifm_rd_valid=%0b dr_wr=%0b",
               $time,
               issue_any,
               issue_addr_valid,
               issue_block_row,
               issue_block_col,
               issue_cgroup,
               issue_ky,
               issue_kx,
               issue_abs_row16,
               issue_abs_col_g16,
               tile_col_base_g,
               PC,
               issue_col_sel_l16,
               (PC == 0) ? 16'd0 : (issue_abs_col_g16 % PC),
               ifm_rd_en,
               ifm_rd_valid,
               dr_write_en);
    end

    if (issue_any && !issue_addr_valid) begin
      if (issue_abs_col_g16 >= (tile_col_base_g + PC)) begin
        $display("DBG_M2_AG_BLOCKED_AFTER_RESIDENT t=%0t abs_col=%0d tile_base=%0d PC=%0d expected_bank_mod=%0d",
                 $time,
                 issue_abs_col_g16,
                 tile_col_base_g,
                 PC,
                 (PC == 0) ? 16'd0 : (issue_abs_col_g16 % PC));
      end

      if (issue_col_sel_l16 >= PC) begin
        $display("DBG_M2_AG_BLOCKED_LOCAL_COL_RANGE t=%0t col_l_calc=%0d PC=%0d",
                 $time,
                 issue_col_sel_l16,
                 PC);
      end
    end
  end
end

endmodule
