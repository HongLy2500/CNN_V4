module same_mode_refill_manager_m2 #(
  parameter int H_W         = 16,
  parameter int COLG_W      = 16,
  parameter int COLL_W      = 8,
  parameter int CGRP_W      = 16,
  parameter int TOKEN_DEPTH = 16
)(
  input  logic clk,
  input  logic rst_n,

  // --------------------------------------------------
  // Config of next layer (kept for contract visibility / future assertions)
  // --------------------------------------------------
  input  logic [15:0] cfg_cur_h_out,
  input  logic [15:0] cfg_cur_w_out,
  input  logic [15:0] cfg_cur_f_out,

  input  logic [15:0] cfg_next_h_in,
  input  logic [15:0] cfg_next_w_in,
  input  logic [15:0] cfg_next_c_in,
  input  logic [7:0] cfg_next_pc,
  input  logic [7:0] cfg_next_pf,

  // --------------------------------------------------
  // FREE token from IFM side
  // --------------------------------------------------
  input  logic                    free_valid,
  input  logic [H_W-1:0]          free_row_g,
  input  logic [COLG_W-1:0]       free_col_g,
  input  logic [COLL_W-1:0]       free_col_l,
  input  logic [CGRP_W-1:0]       free_cgrp_g,

  // --------------------------------------------------
  // READY token from OFM side
  // --------------------------------------------------
  input  logic                    ready_valid,
  input  logic [H_W-1:0]          ready_row_g,
  input  logic [COLG_W-1:0]       ready_col_g,
  input  logic [CGRP_W-1:0]       ready_cgrp_g,

  // --------------------------------------------------
  // Refill request downstream
  // --------------------------------------------------
  output logic                    refill_req_valid,
  input  logic                    refill_req_ready,

  output logic [H_W-1:0]          refill_row_g,
  output logic [COLG_W-1:0]       refill_col_g,
  output logic [COLL_W-1:0]       refill_col_l,
  output logic [CGRP_W-1:0]       refill_cgrp_g,

  // --------------------------------------------------
  // Status / error
  // --------------------------------------------------
  output logic                    busy,
  output logic                    error,
  output logic                    free_fifo_full,
  output logic                    ready_fifo_full
);

  localparam int DEPTH_W = (TOKEN_DEPTH <= 1) ? 1 : $clog2(TOKEN_DEPTH+1);
  localparam int IDX_W   = (TOKEN_DEPTH <= 1) ? 1 : $clog2(TOKEN_DEPTH);

  // --------------------------------------------------------------------------
  // Token storage
  // --------------------------------------------------------------------------
  // Do not use array-of-struct for the stored tokens.  Vivado can report
  // "both Set and reset with same priority" on individual struct fields after
  // expanding dynamic-index writes.  Payload is split into simple field arrays
  // and is intentionally NOT reset; valid bits/counts define whether a slot is
  // meaningful.

  logic [H_W-1:0]     free_row_g_q   [0:TOKEN_DEPTH-1];
  logic [COLG_W-1:0]  free_col_g_q   [0:TOKEN_DEPTH-1];
  logic [COLL_W-1:0]  free_col_l_q   [0:TOKEN_DEPTH-1];
  logic [CGRP_W-1:0]  free_cgrp_g_q  [0:TOKEN_DEPTH-1];

  logic [H_W-1:0]     ready_row_g_q  [0:TOKEN_DEPTH-1];
  logic [COLG_W-1:0]  ready_col_g_q  [0:TOKEN_DEPTH-1];
  logic [CGRP_W-1:0]  ready_cgrp_g_q [0:TOKEN_DEPTH-1];

  logic [TOKEN_DEPTH-1:0] free_v_q,  free_v_d;
  logic [TOKEN_DEPTH-1:0] ready_v_q, ready_v_d;

  logic [DEPTH_W-1:0] free_count_q,  free_count_d;
  logic [DEPTH_W-1:0] ready_count_q, ready_count_d;

  logic req_pending_q, req_pending_d;
  logic [H_W-1:0]     req_row_g_q,   req_row_g_d;
  logic [COLG_W-1:0]  req_col_g_q,   req_col_g_d;
  logic [COLL_W-1:0]  req_col_l_q,   req_col_l_d;
  logic [CGRP_W-1:0]  req_cgrp_g_q,  req_cgrp_g_d;

  logic overflow_err_q, overflow_err_d;

  integer scan_i;
  integer seq_i;

  // Match / empty search results.
  logic found_ready_match_free;
  logic found_free_match_ready;
  logic [TOKEN_DEPTH-1:0] free_match_mask, ready_match_mask;
  logic [IDX_W-1:0] free_match_idx, ready_match_idx;

  logic free_has_empty, ready_has_empty;
  logic [IDX_W-1:0] free_empty_idx, ready_empty_idx;

  // Transaction decisions.
  logic free_to_req;
  logic ready_to_req;
  logic free_enqueue;
  logic ready_enqueue;
  logic free_store_has_space;
  logic ready_store_has_space;
  logic [IDX_W-1:0] free_store_idx;
  logic [IDX_W-1:0] ready_store_idx;

  // Valid-bit update masks.  This avoids multiple procedural assignments to the
  // same valid vector elements in different branches.
  logic [TOKEN_DEPTH-1:0] free_set_mask,  free_clr_mask;
  logic [TOKEN_DEPTH-1:0] ready_set_mask, ready_clr_mask;

  // Payload write enables.  The actual stored payload fields are written only
  // in one sequential loop below, one clear priority point per field.
  logic [TOKEN_DEPTH-1:0] free_payload_we;
  logic [TOKEN_DEPTH-1:0] ready_payload_we;

  // --------------------------------------------------------------------------
  // Associative search: stored FREE vs incoming READY, stored READY vs incoming FREE.
  // M2 match identity is {row_g, col_g, cgrp_g}; col_l belongs to the free token
  // and is forwarded to the refill request when the matched free token is used.
  // --------------------------------------------------------------------------
  always_comb begin
    free_match_mask        = '0;
    ready_match_mask       = '0;
    free_match_idx         = '0;
    ready_match_idx        = '0;
    found_free_match_ready = 1'b0;
    found_ready_match_free = 1'b0;

    free_has_empty         = 1'b0;
    ready_has_empty        = 1'b0;
    free_empty_idx         = '0;
    ready_empty_idx        = '0;

    for (scan_i = 0; scan_i < TOKEN_DEPTH; scan_i = scan_i + 1) begin
      if (free_v_q[scan_i] &&
          (free_row_g_q[scan_i]  == ready_row_g) &&
          (free_col_g_q[scan_i]  == ready_col_g) &&
          (free_cgrp_g_q[scan_i] == ready_cgrp_g)) begin
        free_match_mask[scan_i] = 1'b1;
      end

      if (ready_v_q[scan_i] &&
          (ready_row_g_q[scan_i]  == free_row_g) &&
          (ready_col_g_q[scan_i]  == free_col_g) &&
          (ready_cgrp_g_q[scan_i] == free_cgrp_g)) begin
        ready_match_mask[scan_i] = 1'b1;
      end

      if (!free_has_empty && !free_v_q[scan_i]) begin
        free_has_empty = 1'b1;
        free_empty_idx = scan_i;
      end

      if (!ready_has_empty && !ready_v_q[scan_i]) begin
        ready_has_empty = 1'b1;
        ready_empty_idx = scan_i;
      end
    end

    for (scan_i = 0; scan_i < TOKEN_DEPTH; scan_i = scan_i + 1) begin
      if (!found_free_match_ready && free_match_mask[scan_i]) begin
        found_free_match_ready = 1'b1;
        free_match_idx         = scan_i;
      end
      if (!found_ready_match_free && ready_match_mask[scan_i]) begin
        found_ready_match_free = 1'b1;
        ready_match_idx        = scan_i;
      end
    end
  end

  // --------------------------------------------------------------------------
  // Next-state control.  Payload arrays are not written here; this block only
  // creates payload write-enable masks and valid-bit masks.
  // --------------------------------------------------------------------------
  always_comb begin
    free_count_d       = free_count_q;
    ready_count_d      = ready_count_q;
    req_pending_d      = req_pending_q;
    req_row_g_d        = req_row_g_q;
    req_col_g_d        = req_col_g_q;
    req_col_l_d        = req_col_l_q;
    req_cgrp_g_d       = req_cgrp_g_q;
    overflow_err_d     = overflow_err_q;

    free_to_req           = 1'b0;
    ready_to_req          = 1'b0;
    free_enqueue          = 1'b0;
    ready_enqueue         = 1'b0;
    free_store_has_space  = 1'b0;
    ready_store_has_space = 1'b0;
    free_store_idx        = free_empty_idx;
    ready_store_idx       = ready_empty_idx;

    free_set_mask         = '0;
    free_clr_mask         = '0;
    ready_set_mask        = '0;
    ready_clr_mask        = '0;
    free_payload_we       = '0;
    ready_payload_we      = '0;

    // Consume a pending request on handshake first.  This permits same-cycle
    // accept of an old request and generation of a new request.
    if (req_pending_q && refill_req_ready) begin
      req_pending_d = 1'b0;
    end

    // Incoming FREE token.
    if (free_valid) begin
      if (!req_pending_d && found_ready_match_free) begin
        free_to_req        = 1'b1;
        req_pending_d      = 1'b1;
        req_row_g_d        = free_row_g;
        req_col_g_d        = free_col_g;
        req_col_l_d        = free_col_l;
        req_cgrp_g_d       = free_cgrp_g;
        ready_clr_mask[ready_match_idx] = 1'b1;
        if (ready_count_d != 0)
          ready_count_d = ready_count_d - 1'b1;
      end else begin
        free_enqueue = 1'b1;
      end
    end

    // Incoming READY token.
    if (ready_valid) begin
      if (!req_pending_d && found_free_match_ready) begin
        ready_to_req       = 1'b1;
        req_pending_d      = 1'b1;
        req_row_g_d        = free_row_g_q[free_match_idx];
        req_col_g_d        = free_col_g_q[free_match_idx];
        req_col_l_d        = free_col_l_q[free_match_idx];
        req_cgrp_g_d       = free_cgrp_g_q[free_match_idx];
        free_clr_mask[free_match_idx] = 1'b1;
        if (free_count_d != 0)
          free_count_d = free_count_d - 1'b1;
      end else begin
        ready_enqueue = 1'b1;
      end
    end

    // Direct same-cycle FREE/READY match if neither matched existing storage.
    if (free_enqueue && ready_enqueue && !req_pending_d &&
        (free_row_g  == ready_row_g) &&
        (free_col_g  == ready_col_g) &&
        (free_cgrp_g == ready_cgrp_g)) begin
      free_enqueue    = 1'b0;
      ready_enqueue   = 1'b0;
      req_pending_d   = 1'b1;
      req_row_g_d     = free_row_g;
      req_col_g_d     = free_col_g;
      req_col_l_d     = free_col_l;
      req_cgrp_g_d    = free_cgrp_g;
    end

    // Reuse a slot that was consumed earlier in this cycle, if no empty slot
    // existed before the cycle.
    free_store_has_space  = free_has_empty  || ready_to_req;
    ready_store_has_space = ready_has_empty || free_to_req;
    free_store_idx        = free_has_empty  ? free_empty_idx  : free_match_idx;
    ready_store_idx       = ready_has_empty ? ready_empty_idx : ready_match_idx;

    if (free_enqueue) begin
      if (free_store_has_space) begin
        free_set_mask[free_store_idx] = 1'b1;
        free_payload_we[free_store_idx] = 1'b1;
        free_count_d = free_count_d + 1'b1;
      end else begin
        overflow_err_d = 1'b1;
      end
    end

    if (ready_enqueue) begin
      if (ready_store_has_space) begin
        ready_set_mask[ready_store_idx] = 1'b1;
        ready_payload_we[ready_store_idx] = 1'b1;
        ready_count_d = ready_count_d + 1'b1;
      end else begin
        overflow_err_d = 1'b1;
      end
    end

    free_v_d  = (free_v_q  & ~free_clr_mask)  | free_set_mask;
    ready_v_d = (ready_v_q & ~ready_clr_mask) | ready_set_mask;
  end

  // --------------------------------------------------------------------------
  // Registers.
  //
  // IMPORTANT for Vivado synthesis:
  //   - Control/valid/count/output registers are in the async-reset process.
  //   - Token payload arrays are in a separate no-reset process.
  //
  // Payload arrays are only meaningful when the corresponding valid bit is set,
  // so resetting them is unnecessary.  Keeping payload writes out of the
  // reset-sensitive process avoids "both Set and reset with same priority"
  // warnings on payload fields.
  // --------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      free_v_q       <= '0;
      ready_v_q      <= '0;
      free_count_q   <= '0;
      ready_count_q  <= '0;
      req_pending_q  <= 1'b0;
      req_row_g_q    <= '0;
      req_col_g_q    <= '0;
      req_col_l_q    <= '0;
      req_cgrp_g_q   <= '0;
      overflow_err_q <= 1'b0;
    end else begin
      free_v_q       <= free_v_d;
      ready_v_q      <= ready_v_d;
      free_count_q   <= free_count_d;
      ready_count_q  <= ready_count_d;
      req_pending_q  <= req_pending_d;
      req_row_g_q    <= req_row_g_d;
      req_col_g_q    <= req_col_g_d;
      req_col_l_q    <= req_col_l_d;
      req_cgrp_g_q   <= req_cgrp_g_d;
      overflow_err_q <= overflow_err_d;
    end
  end

  // Payload token storage: no reset path.
  always_ff @(posedge clk) begin
    for (seq_i = 0; seq_i < TOKEN_DEPTH; seq_i = seq_i + 1) begin
      if (free_payload_we[seq_i]) begin
        free_row_g_q[seq_i]  <= free_row_g;
        free_col_g_q[seq_i]  <= free_col_g;
        free_col_l_q[seq_i]  <= free_col_l;
        free_cgrp_g_q[seq_i] <= free_cgrp_g;
      end

      if (ready_payload_we[seq_i]) begin
        ready_row_g_q[seq_i]  <= ready_row_g;
        ready_col_g_q[seq_i]  <= ready_col_g;
        ready_cgrp_g_q[seq_i] <= ready_cgrp_g;
      end
    end
  end

  assign refill_req_valid = req_pending_q;
  assign refill_row_g     = req_row_g_q;
  assign refill_col_g     = req_col_g_q;
  assign refill_col_l     = req_col_l_q;
  assign refill_cgrp_g    = req_cgrp_g_q;

  assign free_fifo_full   = (free_count_q >= TOKEN_DEPTH);
  assign ready_fifo_full  = (ready_count_q >= TOKEN_DEPTH);
  assign busy             = req_pending_q || (free_count_q != 0) || (ready_count_q != 0);
  assign error            = overflow_err_q;

`ifndef SYNTHESIS
  always_ff @(posedge clk) begin
    if (rst_n && free_valid) begin
      if ((free_row_g >= cfg_next_h_in) || (free_col_g >= cfg_next_w_in) || (free_cgrp_g * cfg_next_pc >= cfg_next_c_in)) begin
        $error("same_mode_refill_manager_m2: free token out of next-layer range.");
      end
    end
    if (rst_n && ready_valid) begin
      if ((ready_row_g >= cfg_cur_h_out) || (ready_col_g >= cfg_cur_w_out) || (ready_cgrp_g * cfg_next_pc >= cfg_cur_f_out)) begin
        $error("same_mode_refill_manager_m2: ready token out of current-layer range.");
      end
    end
  end
`endif

endmodule
