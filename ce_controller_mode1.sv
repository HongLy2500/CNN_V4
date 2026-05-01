module ce_controller_mode1 #(
  parameter int K_MAX    = 7,
  parameter int HOUT_MAX = 224,
  parameter int WOUT_MAX = 224
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
  input  logic [3:0] K_cur,
  input  logic [7:0] C_cur,
  input  logic [7:0] F_cur,
  input  logic [15:0] Hout_cur,
  input  logic [15:0] Wout_cur,
  input  logic [7:0] Pv_cur,
  input  logic [7:0] Pf_cur,

  // =====================================================
  // Loop outputs
  // =====================================================
  output logic [15:0] out_row,
  output logic [15:0] out_col,
  output logic [15:0] f_group,
  output logic [15:0] c_iter,
  output logic [$clog2(K_MAX)-1:0] ky,
  output logic [$clog2(K_MAX)-1:0] kx,

  // =====================================================
  // Core CE control
  // =====================================================
  output logic mac_en,
  output logic clear_psum,
  output logic out_valid,

  // =====================================================
  // Pulses for outer control_unit / transfer control
  // =====================================================
  // Start of a new compute sweep for the current output block:
  //   (out_row, f_group, out_col)
  // This pulse occurs in S_CLEAR, before MAC accumulation begins.
  // External control can use it to start / align IFM -> data_register writes
  // for this specific output block, not for an entire row-level pass.
  output logic pass_start_pulse,

  // Finished all kx of one ky row
  output logic                     row_done_pulse,
  output logic [$clog2(K_MAX)-1:0] row_done_ky,

  // Finished all ky/kx for one channel
  output logic chan_done_pulse,

  // Finished the last out_col block of the current f_group in the current out_row.
  // In other words, this pulses when the controller completes the sweep for:
  //   current out_row, current f_group, and the final out_col of that row.
  // It does NOT pulse for every (out_row, out_col, f_group) block.
  output logic f_group_done_pulse,

  // Finished all f_groups for one out_row
  output logic out_row_done_pulse,

  // Finished entire configured workload
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

  logic [15:0] out_row_r, out_col_r, f_group_r, c_iter_r;
  logic [$clog2(K_MAX)-1:0] ky_r, kx_r;

  logic [15:0] num_fgroup;

  logic last_kx;
  logic last_ky;
  logic last_c;
  logic last_fgroup;
  logic last_col;
  logic last_row;

  // =====================================================
  // Derived runtime values
  // =====================================================
  always_comb begin
    if (Pf_cur != 0)
      num_fgroup = (F_cur + Pf_cur - 1) / Pf_cur;
    else
      num_fgroup = 16'd0;
  end

  always_comb begin
    last_kx     = (kx_r == K_cur - 1);
    last_ky     = (ky_r == K_cur - 1);
    last_c      = (c_iter_r == C_cur - 1);
    last_fgroup = (f_group_r == num_fgroup - 1);
    last_col    = (out_col_r + Pv_cur >= Wout_cur);
    last_row    = (out_row_r == Hout_cur - 1);
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
        if (step_en && last_kx && last_ky && last_c)
          next_state = S_FLUSH;
      end

      S_FLUSH: begin
        next_state = S_ADVANCE;
      end

      S_ADVANCE: begin
        if (last_row && last_col && last_fgroup)
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
  //
  // Loop order agreed most recently:
  //   for out_row
  //     for f_group
  //       for out_col (step = Pv_cur)
  //         for c_iter
  //           for ky
  //             for kx
  //
  // Important timing for mac_array_mode1:
  // - The final multiply-accumulate happens in S_RUN.
  // - A separate S_FLUSH cycle then asserts out_valid,
  //   after the final psum has been committed.
  // =====================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_row_r <= 16'd0;
      out_col_r <= 16'd0;
      f_group_r <= 16'd0;
      c_iter_r  <= 16'd0;
      ky_r      <= '0;
      kx_r      <= '0;
    end
    else begin
      case (state)
        S_IDLE: begin
          if (start) begin
            out_row_r <= 16'd0;
            out_col_r <= 16'd0;
            f_group_r <= 16'd0;
            c_iter_r  <= 16'd0;
            ky_r      <= '0;
            kx_r      <= '0;
          end
        end

        S_CLEAR: begin
          // New compute sweep starts here for the current output block
          // (out_row, f_group, out_col).
          c_iter_r <= 16'd0;
          ky_r     <= '0;
          kx_r     <= '0;
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

                if (!last_c) begin
                  c_iter_r <= c_iter_r + 1'b1;
                end
              end
            end
          end
        end

        S_FLUSH: begin
          // Hold loop counters stable while the final psum becomes visible.
        end

        S_ADVANCE: begin
          // Finished current (out_row, f_group, out_col) sweep.
          // Next order is:
          //   out_col -> f_group -> out_row
          if (!last_col) begin
            out_col_r <= out_col_r + Pv_cur;
          end
          else begin
            out_col_r <= 16'd0;

            if (!last_fgroup) begin
              f_group_r <= f_group_r + 1'b1;
            end
            else begin
              f_group_r <= 16'd0;
              if (!last_row)
                out_row_r <= out_row_r + 1'b1;
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
    out_row = out_row_r;
    out_col = out_col_r;
    f_group = f_group_r;
    c_iter  = c_iter_r;
    ky      = ky_r;
    kx      = kx_r;

    busy = (state != S_IDLE);

    // clear accumulator when a new sweep starts
    clear_psum = (state == S_CLEAR);

    // MAC request is active for the whole RUN phase.
    // The actual qualified consume event is provided back through step_en
    // from ce_mode1_top after operand readiness is checked.
    mac_en = (state == S_RUN);

    // one output block (for current f_group/current out_col/current out_row)
    // becomes valid in a dedicated flush cycle, after the final MAC step
    // has already updated psum_out_lane
    out_valid = (state == S_FLUSH);

    // pulse at the start of every new compute sweep for the current
    // output block (out_row, f_group, out_col)
    // external control_unit can use this to start / align
    // IFM -> data_register writes for that block
    pass_start_pulse = (state == S_CLEAR);

    // finished all kx of current ky row
    row_done_pulse = (state == S_RUN) && step_en && last_kx;
    row_done_ky    = ky_r;

    // finished all ky/kx of current channel
    chan_done_pulse = (state == S_RUN) && step_en &&
                      last_kx && last_ky;

    // finished the final out_col block of the current f_group
    // within the current out_row
    f_group_done_pulse = (state == S_RUN) && step_en &&
                         last_kx && last_ky && last_c && last_col;

    // finished all f_groups of current output row
    out_row_done_pulse = (state == S_ADVANCE) &&
                         last_col && last_fgroup;

    // all work finished
    done = (state == S_ADVANCE) &&
           last_row && last_col && last_fgroup;
  end

endmodule