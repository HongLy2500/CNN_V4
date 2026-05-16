module ce_controller_mode2 #(
  parameter int K_MAX    = 7,
  parameter int HOUT_MAX = 224,
  parameter int WOUT_MAX = 224,
  parameter int PC       = 8,
  parameter int PF       = 4
)(
  input  logic clk,
  input  logic rst_n,

  // =====================================================
  // Global control
  // =====================================================
  input  logic start,
  input  logic step_en,

  // Operand readiness from ce_mode2_top.
  //
  // After the Mode2 consume-qualified fix, step_en is no longer a raw
  // free-running cadence.  ce_mode2_top must drive step_en only when the
  // current IFM tuple and current weight tuple are both valid in their
  // registers.  tuple_ready is kept as the block-start readiness guard for
  // S_CLEAR; in S_RUN, all loop counters advance only on qualified step_en.
  input  logic tuple_ready,

  // =====================================================
  // Runtime config
  // =====================================================
  input  logic [3:0] K_cur,
  input  logic [7:0] C_cur,
  input  logic [7:0] F_cur,
  input  logic [15:0] Hout_cur,
  input  logic [15:0] Wout_cur,

  // =====================================================
  // Mode-2 resident-tile contract
  // =====================================================
  // IFM buffer Mode 2 holds one horizontal tile at a time:
  //   bank = local column col_l inside the resident tile
  //   lane = channel inside PC group
  //
  // This controller computes exactly ONE resident output tile per start:
  //   global out_col = tile_col_base_g + local_col
  //   local_col      = 0 .. tile_col_count-1
  //
  // The outer control_unit_top is responsible for:
  //   load/refill tile -> start CE -> wait done -> next tile.
  input  logic [15:0] tile_col_base_g,
  input  logic [15:0] tile_col_count,

  // =====================================================
  // Loop outputs
  //
  // IMPORTANT CONTRACT:
  // - out_row / out_col are GLOBAL output-feature-map coordinates.
  // - Internally this controller scans tile-local columns only.
  // - Downstream addr_gen_ifm_m2 derives IFM local column as:
  //       ifm_col_l = out_col_global - tile_col_base_g
  //
  // Mode 2 tile-local loop order per start after the col-pair update:
  //   for out_col_pair_l = 0, 2, 4, ... within resident tile
  //     for out_row_g
  //       compute out_col_pair_l
  //       compute out_col_pair_l + 1 if it is still inside the tile
  //         for f_group
  //           for c_group
  //             for ky
  //               for kx
  //
  // This changes only the spatial traversal order.  It does not change
  // the CE handshake, f_group/c_group order, K loops, resident-tile
  // contract, IFM/OFM layout, or any refill/stream mechanism.
  //
  // In other words, CE is the single source of GLOBAL spatial
  // coordinates. Downstream ReLU/pooling/OFM write must use out_row/out_col
  // as global coordinates; only IFM address generation derives a tile-local
  // column by subtracting tile_col_base_g.
  // =====================================================
  output logic [15:0] out_row,
  output logic [15:0] out_col,
  output logic [15:0] f_group,
  output logic [15:0] c_group,
  output logic [$clog2(K_MAX)-1:0] ky,
  output logic [$clog2(K_MAX)-1:0] kx,

  // =====================================================
  // Core CE control
  // =====================================================
  output logic mac_en,
  output logic clear_psum,
  output logic out_valid,

  // =====================================================
  // Pulses for outer control_unit / dataflow control
  // =====================================================
  // Start of a new accumulation sweep for one GLOBAL output pixel.
  output logic pass_start_pulse,

  // Marker of the first GLOBAL output pixel of a new filter-group.
  // ce_mode2_top should delay this pulse to line up with ReLU output valid.
  output logic group_start_pulse,

  // Finished all kx of one ky row for current c_group and current GLOBAL pixel.
  output logic                     row_done_pulse,
  output logic [$clog2(K_MAX)-1:0] row_done_ky,

  // Finished all ky/kx for one c_group at current GLOBAL pixel.
  output logic c_group_done_pulse,

  // Finished one GLOBAL output pixel of current filter-group.
  output logic pixel_done_pulse,

  // Finished an entire GLOBAL raster map of current filter-group for this tile.
  output logic f_group_done_pulse,

  // Finished entire configured tile workload.
  output logic done,
  output logic busy
);

  typedef enum logic [2:0] {
    S_IDLE,
    S_CLEAR,
    S_RUN,
    S_FLUSH,
    S_ADVANCE
  } state_t;

  state_t state, next_state;

  // out_col_l_r is local to the currently resident Mode-2 tile.
  // out_row_g_r and out_col output remain global externally.
  logic [15:0] out_row_g_r, out_col_l_r, f_group_r, c_group_r;
  logic [$clog2(K_MAX)-1:0] ky_r, kx_r;

  logic [15:0] num_fgroup;
  logic [15:0] num_cgroup;

  logic [15:0] tile_remaining_cols;
  logic [15:0] tile_col_count_req;
  logic [15:0] tile_col_count_eff;
  logic        tile_cfg_valid;
  logic        workload_valid;

  logic last_kx;
  logic last_ky;
  logic last_cgroup;
  logic last_col;
  logic last_row;
  logic last_fgroup;

  logic block_start_fire;

  // Col-pair traversal helpers.  These are local to the resident tile.
  // pair_base_l is always even: 0, 2, 4, ...
  logic [15:0] out_col_pair_base_l_s;
  logic        out_col_pair_first_s;
  logic        out_col_pair_has_second_s;
  logic        out_col_pair_has_next_s;

  // =====================================================
  // Derived runtime values
  // =====================================================
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

  // Effective tile width is clipped to remaining output width.
  // A zero tile_col_count means "use all remaining columns" for safety,
  // but normal control_unit_top should drive min(PC, Wout - tile_base).
  always_comb begin
    if (tile_col_base_g < Wout_cur)
      tile_remaining_cols = Wout_cur - tile_col_base_g;
    else
      tile_remaining_cols = 16'd0;

    tile_col_count_req = (tile_col_count != 16'd0) ? tile_col_count
                                                   : tile_remaining_cols;

    if (tile_col_count_req > tile_remaining_cols)
      tile_col_count_eff = tile_remaining_cols;
    else
      tile_col_count_eff = tile_col_count_req;

    tile_cfg_valid = (tile_remaining_cols != 16'd0) &&
                     (tile_col_count_eff != 16'd0);

    workload_valid = tile_cfg_valid &&
                     (K_cur != 4'd0) &&
                     (Hout_cur != 16'd0) &&
                     (Wout_cur != 16'd0) &&
                     (num_fgroup != 16'd0) &&
                     (num_cgroup != 16'd0);
  end

  always_comb begin
    last_kx     = (kx_r == K_cur - 1);
    last_ky     = (ky_r == K_cur - 1);
    last_cgroup = (c_group_r == num_cgroup - 1);
    last_col    = (tile_col_count_eff == 16'd0) ||
                  (out_col_l_r == tile_col_count_eff - 1);
    last_row    = (out_row_g_r == Hout_cur - 1);
    last_fgroup = (f_group_r == num_fgroup - 1);
  end

  always_comb begin
    // Floor current local column to an even column-pair base.
    out_col_pair_base_l_s   = {out_col_l_r[15:1], 1'b0};
    out_col_pair_first_s    = (out_col_l_r == out_col_pair_base_l_s);
    out_col_pair_has_second_s =
        ((out_col_pair_base_l_s + 16'd1) < tile_col_count_eff);
    out_col_pair_has_next_s =
        ((out_col_pair_base_l_s + 16'd2) < tile_col_count_eff);
  end

  // step_en is consume-qualified by ce_mode2_top.  In S_CLEAR this also
  // means the first tuple is ready to leave the clear phase.
  assign block_start_fire = step_en && tuple_ready;

  // =====================================================
  // FSM state register
  // =====================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      state <= S_IDLE;
    else
      state <= next_state;
  end

  // =====================================================
  // FSM next-state logic
  // =====================================================
  always_comb begin
    next_state = state;

    case (state)
      S_IDLE: begin
        if (start && workload_valid)
          next_state = S_CLEAR;
      end

      S_CLEAR: begin
        // Wait for the first IFM/weight tuple of this output block.
        // block_start_fire is qualified by ce_mode2_top's consume-ready
        // contract, so S_RUN never starts before the first weight is
        // registered and visible to the MAC.
        if (block_start_fire)
          next_state = S_RUN;
      end

      S_RUN: begin
        if (step_en && last_kx && last_ky && last_cgroup)
          next_state = S_FLUSH;
      end

      S_FLUSH: begin
        next_state = S_ADVANCE;
      end

      S_ADVANCE: begin
        if (last_fgroup && last_row && last_col)
          next_state = S_IDLE;
        else
          next_state = S_CLEAR;
      end

      default: begin
        next_state = S_IDLE;
      end
    endcase
  end

  // =====================================================
  // Counter update
  // =====================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_row_g_r <= 16'd0;
      out_col_l_r <= 16'd0;
      f_group_r   <= 16'd0;
      c_group_r   <= 16'd0;
      ky_r        <= '0;
      kx_r        <= '0;
    end
    else begin
      case (state)
        S_IDLE: begin
          if (start && workload_valid) begin
            out_row_g_r <= 16'd0;
            out_col_l_r <= 16'd0;
            f_group_r   <= 16'd0;
            c_group_r   <= 16'd0;
            ky_r        <= '0;
            kx_r        <= '0;
          end
        end

        S_CLEAR: begin
          c_group_r <= 16'd0;
          ky_r      <= '0;
          kx_r      <= '0;
        end

        S_RUN: begin
          if (step_en) begin
            if (!last_kx) begin
              kx_r <= kx_r + 1'b1;
            end
            else begin
              kx_r <= '0;

              if (!last_ky) begin
                ky_r <= ky_r + 1'b1;
              end
              else begin
                ky_r <= '0;

                if (!last_cgroup)
                  c_group_r <= c_group_r + 1'b1;
              end
            end
          end
        end

        S_FLUSH: begin
          // Hold loop counters stable while MAC array emits the completed pixel.
        end

        S_ADVANCE: begin
          // Spatial order after finishing one GLOBAL output pixel for the
          // current filter group:
          //   f_group -> col inside current pair -> row -> next col pair
          //
          // This keeps all PF groups for the same spatial pixel adjacent,
          // preserves the existing f_group/c_group/K-loop semantics, and
          // changes only the row/column traversal inside the resident tile.
          //
          // Example for one tile:
          //   (row0,col0), (row0,col1), (row1,col0), (row1,col1), ...
          //   (row0,col2), (row0,col3), ...
          if (!last_fgroup) begin
            f_group_r <= f_group_r + 1'b1;
          end
          else begin
            f_group_r <= 16'd0;

            if (out_col_pair_first_s && out_col_pair_has_second_s) begin
              // Finish the second column of the current column pair for
              // the same row before moving to the next row.
              out_col_l_r <= out_col_pair_base_l_s + 16'd1;
            end
            else if (!last_row) begin
              // Move to the next row and return to the first column of
              // the current pair.
              out_row_g_r <= out_row_g_r + 1'b1;
              out_col_l_r <= out_col_pair_base_l_s;
            end
            else begin
              // Finished all rows for this column pair.  Move to the next
              // pair if one exists; otherwise wrap to idle/done on the next
              // FSM transition.
              out_row_g_r <= 16'd0;

              if (out_col_pair_has_next_s) begin
                out_col_l_r <= out_col_pair_base_l_s + 16'd2;
              end
              else begin
                out_col_l_r <= 16'd0;
              end
            end
          end
        end

        default: begin
        end
      endcase
    end
  end

  // =====================================================
  // Outputs
  // =====================================================
  always_comb begin
    // Export GLOBAL coordinates.
    out_row = out_row_g_r;
    out_col = tile_col_base_g + out_col_l_r;
    f_group = f_group_r;
    c_group = c_group_r;
    ky      = ky_r;
    kx      = kx_r;

    busy = (state != S_IDLE);

    // Kept for interface symmetry/debug. mac_array_mode2 clears internally.
    clear_psum = (state == S_CLEAR);

    // Accumulate only on consume-qualified RUN cycles.  step_en is supplied
    // by ce_mode2_top as the Mode2 equivalent of Mode1's ctrl_step_en.
    // Therefore mac_en and loop-counter advance are aligned to a real
    // data+weight consume event, not to raw controller cadence.
    mac_en = (state == S_RUN) && step_en;

    // Separate flush cycle so the last MAC accumulation is not lost.
    out_valid = (state == S_FLUSH);

    // Start of accumulation for the current GLOBAL pixel block.
    pass_start_pulse = (state == S_CLEAR) && block_start_fire;

    // First GLOBAL pixel result of one filter-group in this tile.
    group_start_pulse = (state == S_FLUSH) &&
                        (out_row_g_r == 16'd0) &&
                        (out_col_l_r == 16'd0);

    row_done_pulse = (state == S_RUN) && step_en && last_kx;
    row_done_ky    = ky_r;

    c_group_done_pulse = (state == S_RUN) && step_en &&
                         last_kx && last_ky;

    pixel_done_pulse = (state == S_FLUSH);

    f_group_done_pulse = (state == S_ADVANCE) && last_row && last_col;

    done = (state == S_ADVANCE) &&
           last_fgroup && last_row && last_col;
  end



endmodule
