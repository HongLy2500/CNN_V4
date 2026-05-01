module transition_manager
  import cnn_layer_desc_pkg::*;
#(
  parameter int PV_MAX = 8,
  parameter int PC     = 8,
  parameter int F_MAX  = 512,
  parameter int H_MAX  = 224,
  parameter int W_MAX  = 224
)(
  input  logic clk,
  input  logic rst_n,

  // Current / next layer descriptors
  input  layer_desc_t cur_cfg,
  input  layer_desc_t next_cfg,
  input  logic        next_valid,

  // Requests from scheduler
  // CONTRACT:
  // - transition_manager ONLY services mode1 -> mode2 handoff.
  // - kick_same_mode_stream is ignored here and must not be treated as an error.
  input  logic kick_same_mode_stream,
  input  logic kick_transition_stream,

  // Geometry of the requested OFM->IFM stream window
  // req_col_base is the GLOBAL starting column of the first mode-2 tile to load.
  input  logic [$clog2(H_MAX+1)-1:0] req_row_base,
  input  logic [$clog2(H_MAX+1)-1:0] req_num_rows,
  input  logic [$clog2(W_MAX+1)-1:0] req_col_base,

  // OFM buffer stream status
  input  logic ofm_layer_write_done,
  input  logic ofm_ifm_stream_busy,
  input  logic ofm_ifm_stream_done,
  input  logic ofm_error,

  // Command to ofm_buffer
  output logic                       ofm_ifm_stream_start,
  output logic [$clog2(H_MAX+1)-1:0] ofm_ifm_stream_row_base,
  output logic [$clog2(H_MAX+1)-1:0] ofm_ifm_stream_num_rows,
  output logic [$clog2(W_MAX+1)-1:0] ofm_ifm_stream_col_base,

  // Status back to scheduler
  output logic transition_done,
  output logic transition_busy,
  output logic transition_error,

  // Debug / visibility
  output logic [1:0] dbg_req_kind,   // 0:none, 2:m1->m2 transition, 3:invalid
  output logic       dbg_need_full_store,
  output logic       dbg_waiting_for_layer,
  output logic       dbg_waiting_for_stream
);

  localparam int ROW_W = (H_MAX <= 1) ? 1 : $clog2(H_MAX+1);
  localparam int COL_W = (W_MAX <= 1) ? 1 : $clog2(W_MAX+1);

  typedef enum logic [1:0] {
    REQ_NONE     = 2'd0,
    REQ_RESERVED = 2'd1,  // same-mode no longer supported here
    REQ_M1_TO_M2 = 2'd2,
    REQ_INVALID  = 2'd3
  } req_kind_t;

  logic pending_q, pending_d;
  logic [ROW_W-1:0] row_base_q, row_base_d;
  logic [ROW_W-1:0] num_rows_q, num_rows_d;
  logic [COL_W-1:0] col_base_q, col_base_d;
  logic [COL_W-1:0] active_col_q, active_col_d;
  req_kind_t        req_kind_q, req_kind_d;

  logic start_subgen;
  logic sub_busy, sub_done, sub_error;
  logic invalid_req_pulse;

  logic cfg_src_mode;
  logic cfg_next_mode;
  logic [ROW_W-1:0]                 cfg_h_out_s;
  logic [COL_W-1:0]                 cfg_w_out_s;
  logic [$clog2(F_MAX+1)-1:0]       cfg_f_out_s;
  logic [$clog2(PV_MAX+1)-1:0]      cfg_pv_next_s;

  logic transition_valid;
  logic req_fire;
  req_kind_t req_kind_new;
  logic more_tiles_after_done;

  // --------------------------------------------------
  // Valid transition handled by this manager:
  //   mode1 -> mode2
  // --------------------------------------------------
  assign transition_valid =
      next_valid
    && (cur_cfg.mode == MODE1)
    && (next_cfg.mode == MODE2);

  always_comb begin
    req_kind_new = REQ_NONE;

    if (kick_transition_stream) begin
      if (transition_valid)
        req_kind_new = REQ_M1_TO_M2;
      else
        req_kind_new = REQ_INVALID;
    end
    // kick_same_mode_stream is intentionally ignored here. Same-mode refill is
    // handled elsewhere in the control path and must not raise an error pulse.
  end

  assign req_fire          = kick_transition_stream;
  assign invalid_req_pulse = kick_transition_stream && (req_kind_new == REQ_INVALID);

  // --------------------------------------------------
  // Accept one pending request only when idle.
  // --------------------------------------------------
  always_comb begin
    pending_d    = pending_q;
    row_base_d   = row_base_q;
    num_rows_d   = num_rows_q;
    col_base_d   = col_base_q;
    active_col_d = active_col_q;
    req_kind_d   = req_kind_q;

    if (!pending_q && req_fire && (req_kind_new == REQ_M1_TO_M2)) begin
      pending_d    = 1'b1;
      row_base_d   = req_row_base;
      num_rows_d   = req_num_rows;
      col_base_d   = req_col_base;
      active_col_d = req_col_base;
      req_kind_d   = REQ_M1_TO_M2;
    end
    else if (sub_done) begin
      if ((req_kind_q == REQ_M1_TO_M2) && more_tiles_after_done) begin
        active_col_d = active_col_q + PC[COL_W-1:0];
      end
      else begin
        pending_d    = 1'b0;
        req_kind_d   = REQ_NONE;
      end
    end
    else if (sub_error) begin
      pending_d    = 1'b0;
      req_kind_d   = REQ_NONE;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pending_q    <= 1'b0;
      row_base_q   <= '0;
      num_rows_q   <= '0;
      col_base_q   <= '0;
      active_col_q <= '0;
      req_kind_q   <= REQ_NONE;
    end
    else begin
      pending_q    <= pending_d;
      row_base_q   <= row_base_d;
      num_rows_q   <= num_rows_d;
      col_base_q   <= col_base_d;
      active_col_q <= active_col_d;
      req_kind_q   <= req_kind_d;
    end
  end

  // --------------------------------------------------
  // addr_gen_ofm_to_ifm configuration
  // - src mode is current layer mode
  // - next mode is forced to MODE2 here
  // - this manager never serves same-mode anymore
  // --------------------------------------------------
  assign start_subgen = pending_q && !sub_busy && (req_kind_q == REQ_M1_TO_M2);

  assign cfg_src_mode  = (cur_cfg.mode == MODE2);
  assign cfg_next_mode = 1'b1;  // only mode1 -> mode2 handled here

  assign cfg_h_out_s   = cur_cfg.h_out[ROW_W-1:0];
  assign cfg_w_out_s   = cur_cfg.w_out[COL_W-1:0];
  assign cfg_f_out_s   = cur_cfg.f_out[$clog2(F_MAX+1)-1:0];
  assign cfg_pv_next_s = '0; // unused when next mode is mode2

  // Continue across all horizontal mode-2 tiles of the requested row window.
  always_comb begin
    more_tiles_after_done = 1'b0;
    if ((req_kind_q == REQ_M1_TO_M2) && sub_done) begin
      more_tiles_after_done = ((active_col_q + PC[COL_W-1:0]) < cur_cfg.w_out[COL_W-1:0]);
    end
  end

  addr_gen_ofm_to_ifm #(
    .PV_MAX(PV_MAX),
    .PC    (PC),
    .F_MAX (F_MAX),
    .H_MAX (H_MAX),
    .W_MAX (W_MAX)
  ) u_addr_gen_ofm_to_ifm (
    .clk                 (clk),
    .rst_n               (rst_n),

    .start               (start_subgen),
    .cfg_src_mode        (cfg_src_mode),
    .cfg_next_mode       (cfg_next_mode),
    .cfg_h_out           (cfg_h_out_s),
    .cfg_w_out           (cfg_w_out_s),
    .cfg_f_out           (cfg_f_out_s),
    .cfg_pv_next         (cfg_pv_next_s),

    .req_row_base        (row_base_q),
    .req_num_rows        (num_rows_q),
    .req_col_base        (active_col_q),

    .ofm_stream_busy     (ofm_ifm_stream_busy),
    .ofm_stream_done     (ofm_ifm_stream_done),
    .ofm_layer_write_done(ofm_layer_write_done),
    .ofm_error           (ofm_error),

    .ifm_stream_start    (ofm_ifm_stream_start),
    .ifm_stream_row_base (ofm_ifm_stream_row_base),
    .ifm_stream_num_rows (ofm_ifm_stream_num_rows),
    .ifm_stream_col_base (ofm_ifm_stream_col_base),

    .busy                (sub_busy),
    .done                (sub_done),
    .error               (sub_error),

    .dbg_num_mode1_groups(),
    .dbg_num_mode2_tiles (),
    .dbg_last_row        (),
    .dbg_last_col        (),
    .dbg_need_full_store (dbg_need_full_store),
    .dbg_waiting_for_layer(dbg_waiting_for_layer),
    .dbg_waiting_for_stream(dbg_waiting_for_stream)
  );

  // --------------------------------------------------
  // Status back to scheduler
  // --------------------------------------------------
  assign transition_done  = sub_done && !more_tiles_after_done;
  assign transition_busy  = pending_q || sub_busy;
  assign transition_error = sub_error || invalid_req_pulse;

  assign dbg_req_kind = pending_q ? req_kind_q :
                        (invalid_req_pulse ? REQ_INVALID : REQ_NONE);

endmodule
