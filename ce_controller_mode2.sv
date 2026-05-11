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
  // This is used only to prime the first tuple of a new output block.
  // Once S_RUN starts, the existing Mode-2 prefetch pipeline advances
  // with step_en/mac_en exactly as before.
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
  // Mode 2 tile-local loop order per start, aligned with Mode 1:
  //   for out_row_g
  //     for out_col_l within resident tile
  //       for f_group
  //         for c_group
  //           for ky
  //             for kx
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
        // Wait only for the first IFM/weight tuple of this output block.
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
          // Mode-1-style order after finishing one GLOBAL output pixel
          // for the current filter group:
          //   f_group -> out_col_l -> out_row_g
          //
          // This keeps all Pf groups for the same spatial pixel adjacent,
          // then advances to the next pixel. The exported out_col remains
          // GLOBAL because it is tile_col_base_g + out_col_l_r.
          if (!last_fgroup) begin
            f_group_r <= f_group_r + 1'b1;
          end
          else begin
            f_group_r <= 16'd0;

            if (!last_col) begin
              out_col_l_r <= out_col_l_r + 1'b1;
            end
            else begin
              out_col_l_r <= 16'd0;

              if (!last_row) begin
                out_row_g_r <= out_row_g_r + 1'b1;
              end
              else begin
                out_row_g_r <= 16'd0;
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

    // Accumulate in RUN cycles as in the original Mode-2 pipeline.
    // tuple_ready only gates the transition out of S_CLEAR.
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
