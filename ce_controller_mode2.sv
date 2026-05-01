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

  // =====================================================
  // Runtime config
  // =====================================================
  input  logic [15:0] K_cur,
  input  logic [15:0] C_cur,
  input  logic [15:0] F_cur,
  input  logic [15:0] Hout_cur,
  input  logic [15:0] Wout_cur,

  // =====================================================
  // Loop outputs
  //
  // IMPORTANT CONTRACT:
  // - out_row / out_col are GLOBAL output-feature-map coordinates.
  // - ce_controller_mode2 does NOT generate any tile-local coordinate.
  // - Any local column used to index data already loaded in ifm_buffer
  //   must be derived downstream from:
  //       out_col_local = out_col_global - tile_col_base_global
  //
  // Mode 2 loop order:
  //   for f_group
  //     for out_row_g
  //       for out_col_g
  //         for c_group
  //           for ky
  //             for kx
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

  // Finished an entire GLOBAL raster map of current filter-group.
  output logic f_group_done_pulse,

  // Finished entire configured workload.
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

  // These registers hold GLOBAL coordinates.
  logic [15:0] out_row_g_r, out_col_g_r, f_group_r, c_group_r;
  logic [$clog2(K_MAX)-1:0] ky_r, kx_r;

  logic [15:0] num_fgroup;
  logic [15:0] num_cgroup;

  logic last_kx;
  logic last_ky;
  logic last_cgroup;
  logic last_col;
  logic last_row;
  logic last_fgroup;

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

  always_comb begin
    last_kx     = (kx_r == K_cur - 1);
    last_ky     = (ky_r == K_cur - 1);
    last_cgroup = (c_group_r == num_cgroup - 1);
    last_col    = (out_col_g_r == Wout_cur - 1);
    last_row    = (out_row_g_r == Hout_cur - 1);
    last_fgroup = (f_group_r == num_fgroup - 1);
  end

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
  //
  // Important for mac_array_mode2:
  // - The final multiply-accumulate must happen in S_RUN.
  // - A separate S_FLUSH cycle then asserts out_valid to emit the result.
  // =====================================================
  always_comb begin
    next_state = state;

    case (state)
      S_IDLE: begin
        if (start)
          next_state = S_CLEAR;
      end

      S_CLEAR: begin
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
      out_col_g_r <= 16'd0;
      f_group_r   <= 16'd0;
      c_group_r   <= 16'd0;
      ky_r        <= '0;
      kx_r        <= '0;
    end
    else begin
      case (state)
        S_IDLE: begin
          if (start) begin
            out_row_g_r <= 16'd0;
            out_col_g_r <= 16'd0;
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
          // Next order after finishing one GLOBAL output pixel:
          //   out_col_g -> out_row_g -> f_group
          if (!last_col) begin
            out_col_g_r <= out_col_g_r + 1'b1;
          end
          else begin
            out_col_g_r <= 16'd0;

            if (!last_row) begin
              out_row_g_r <= out_row_g_r + 1'b1;
            end
            else begin
              out_row_g_r <= 16'd0;

              if (!last_fgroup)
                f_group_r <= f_group_r + 1'b1;
              else
                f_group_r <= 16'd0;
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
    out_col = out_col_g_r;
    f_group = f_group_r;
    c_group = c_group_r;
    ky      = ky_r;
    kx      = kx_r;

    busy = (state != S_IDLE);

    // Kept for interface symmetry/debug. mac_array_mode2 clears internally.
    clear_psum = (state == S_CLEAR);

    // Accumulate only in RUN cycles.
    mac_en = (state == S_RUN) && step_en;

    // Separate flush cycle so the last MAC accumulation is not lost.
    out_valid = (state == S_FLUSH);

    // Start of accumulation for the current GLOBAL pixel block.
    pass_start_pulse = (state == S_CLEAR);

    // First GLOBAL pixel result of one filter-group.
    // Top-level should delay this to match relu_out_valid timing.
    group_start_pulse = (state == S_FLUSH) &&
                        (out_row_g_r == 16'd0) &&
                        (out_col_g_r == 16'd0);

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
