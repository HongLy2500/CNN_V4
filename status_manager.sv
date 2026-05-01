module status_manager #(
  parameter int LAYER_IDX_W = 8
)(
  input  logic clk,
  input  logic rst_n,

  // From global scheduler
  input  logic sched_busy,
  input  logic sched_done,
  input  logic sched_error,

  // Aggregated sub-block errors
  input  logic dma_error,
  input  logic ofm_error,
  input  logic local_error,
  input  logic transition_error,

  // Current execution context
  input  logic [LAYER_IDX_W-1:0] cur_layer_idx,
  input  logic                   cur_mode,         // 0: mode1, 1: mode2
  input  logic                   compute_bank_sel, // active compute bank

  // Top-level visible status
  output logic busy,
  output logic done,
  output logic error,

  // Debug / visibility
  output logic [LAYER_IDX_W-1:0] dbg_layer_idx,
  output logic                   dbg_mode,
  output logic                   dbg_weight_bank,
  output logic [3:0]             dbg_error_vec
);

  // --------------------------------------------------------------------------
  // Error aggregation
  // Bit order of dbg_error_vec:
  //   [0] dma_error
  //   [1] ofm_error
  //   [2] local_error
  //   [3] transition_error
  // --------------------------------------------------------------------------
  assign dbg_error_vec = {
    transition_error,
    local_error,
    ofm_error,
    dma_error
  };

  // --------------------------------------------------------------------------
  // Top-level visible status
  //
  // - busy follows the scheduler busy state.
  // - done follows scheduler done, but is suppressed if any error source is high.
  // - error is asserted if either the scheduler entered ERROR state or any
  //   sub-block reports an error.
  // --------------------------------------------------------------------------
  assign error = sched_error | (|dbg_error_vec);
  assign busy  = sched_busy;
  assign done  = sched_done & ~error;

  // --------------------------------------------------------------------------
  // Debug visibility
  // These are direct pass-throughs of the current scheduler context.
  // Keeping them combinational makes debug waveforms easy to interpret.
  // --------------------------------------------------------------------------
  assign dbg_layer_idx   = cur_layer_idx;
  assign dbg_mode        = cur_mode;
  assign dbg_weight_bank = compute_bank_sel;

endmodule
