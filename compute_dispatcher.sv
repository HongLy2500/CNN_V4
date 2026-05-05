module compute_dispatcher
  import cnn_layer_desc_pkg::*;
#(
  parameter int PC_MODE2 = 8
)(
  input  logic clk,
  input  logic rst_n,

  input  layer_desc_t cur_cfg,
  input  logic        compute_bank_sel,
  input  logic        compute_bank_ready,

  input  logic        kick_compute,
  input  logic        hold_compute,
  input  logic        cur_mode,   // 0: mode1, 1: mode2

  input  logic        m1_done,
  input  logic        m1_busy,
  input  logic        m2_done,
  input  logic        m2_busy,

  output logic        m1_start,
  output logic        m1_step_en,
  output logic [7:0]  m1_k_cur,
  output logic [15:0] m1_c_cur,
  output logic [15:0] m1_f_cur,
  output logic [15:0] m1_hout_cur,
  output logic [15:0] m1_wout_cur,
  output logic [15:0] m1_w_cur,
  output logic [7:0]  m1_pv_cur,
  output logic [7:0]  m1_pf_cur,
  output logic        m1_weight_bank_sel,
  output logic        m1_weight_bank_ready,

  output logic        m2_start,
  output logic        m2_step_en,
  output logic [7:0]  m2_k_cur,
  output logic [15:0] m2_c_cur,
  output logic [15:0] m2_f_cur,
  output logic [15:0] m2_hout_cur,
  output logic [15:0] m2_wout_cur,
  output logic        m2_weight_bank_sel,
  output logic        m2_weight_bank_ready,

  output logic        compute_done,
  output logic        compute_busy
);

  // --------------------------------------------------------------------------
  // Internal state
  // --------------------------------------------------------------------------
  logic active_q, active_d;
  logic active_mode_q, active_mode_d;  // 0: mode1, 1: mode2

  logic selected_busy_cur;
  logic selected_done_q;
  logic start_fire;

  // --------------------------------------------------------------------------
  // Current selected-mode status
  // --------------------------------------------------------------------------
  always_comb begin
    selected_busy_cur = (cur_mode == 1'b0) ? m1_busy : m2_busy;
    selected_done_q   = (active_mode_q == 1'b0) ? m1_done : m2_done;
  end

  // Start only when:
  // - scheduler requests it
  // - the chosen compute bank is ready
  // - no previously dispatched compute is still active
  // - target compute block is not still reporting busy
  assign start_fire =
      kick_compute
    && compute_bank_ready
    && !active_q
    && !selected_busy_cur;

  // --------------------------------------------------------------------------
  // Active-dispatch bookkeeping
  // --------------------------------------------------------------------------
  always_comb begin
    active_d      = active_q;
    active_mode_d = active_mode_q;

    if (start_fire) begin
      active_d      = 1'b1;
      active_mode_d = cur_mode;
    end
    else if (active_q && selected_done_q) begin
      active_d      = 1'b0;
      active_mode_d = active_mode_q;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      active_q      <= 1'b0;
      active_mode_q <= 1'b0;
    end
    else begin
      active_q      <= active_d;
      active_mode_q <= active_mode_d;
    end
  end

  // --------------------------------------------------------------------------
  // Runtime configuration passthrough
  // These are always driven from cur_cfg; the inactive mode simply ignores them.
  // --------------------------------------------------------------------------
  assign m1_k_cur             = cur_cfg.k[3:0];
  assign m1_c_cur             = cur_cfg.c_in;
  assign m1_f_cur             = cur_cfg.f_out;
  assign m1_hout_cur          = cur_cfg.h_out;
  assign m1_wout_cur          = cur_cfg.w_out;
  assign m1_w_cur             = cur_cfg.w_in;
  assign m1_pv_cur            = cur_cfg.pv_m1;
  assign m1_pf_cur            = cur_cfg.pf_m1;
  assign m1_weight_bank_sel   = compute_bank_sel;
  assign m1_weight_bank_ready = compute_bank_ready;

  assign m2_k_cur             = cur_cfg.k[3:0];
  assign m2_c_cur             = cur_cfg.c_in;
  assign m2_f_cur             = cur_cfg.f_out;
  assign m2_hout_cur          = cur_cfg.h_out;
  assign m2_wout_cur          = cur_cfg.w_out;
  assign m2_weight_bank_sel   = compute_bank_sel;
  assign m2_weight_bank_ready = compute_bank_ready;

  // --------------------------------------------------------------------------
  // Start pulses
  // --------------------------------------------------------------------------
  assign m1_start = start_fire && (cur_mode == 1'b0);
  assign m2_start = start_fire && (cur_mode == 1'b1);

  // --------------------------------------------------------------------------
  // Step enable
  // Keep the selected compute path running while a dispatched layer is active,
  // unless higher-level control wants to hold it while local dataflow catches up.
  // --------------------------------------------------------------------------
  assign m1_step_en = active_q && (active_mode_q == 1'b0) && !hold_compute;
  assign m2_step_en = active_q && (active_mode_q == 1'b1) && !hold_compute;

  // --------------------------------------------------------------------------
  // Scheduler-visible status
  // compute_done pulses when the currently active mode finishes.
  // compute_busy is asserted from the dispatch cycle onward.
  // --------------------------------------------------------------------------
  assign compute_done = active_q && selected_done_q;
  assign compute_busy = active_q || start_fire;

`ifndef SYNTHESIS
  // Mode-2 Pc is assumed fixed by hardware resources.
  // Keep this as a simulation-time guard only.
  always_ff @(posedge clk) begin
    if (rst_n && start_fire && (cur_mode == 1'b1)) begin
      if (cur_cfg.pc_m2 != PC_MODE2) begin
        $error("compute_dispatcher: cur_cfg.pc_m2 (%0d) != PC_MODE2 parameter (%0d).", cur_cfg.pc_m2, PC_MODE2);
      end
    end
  end
`endif

endmodule
