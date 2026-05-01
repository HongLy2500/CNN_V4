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

  typedef struct packed {
    logic [H_W-1:0]    row_g;
    logic [COLG_W-1:0] col_g;
    logic [COLL_W-1:0] col_l;
    logic [CGRP_W-1:0] cgrp_g;
  } free_tok_t;

  typedef struct packed {
    logic [H_W-1:0]    row_g;
    logic [COLG_W-1:0] col_g;
    logic [CGRP_W-1:0] cgrp_g;
  } ready_tok_t;

  free_tok_t  free_mem  [0:TOKEN_DEPTH-1];
  ready_tok_t ready_mem [0:TOKEN_DEPTH-1];

  logic [TOKEN_DEPTH-1:0] free_v_q,  free_v_d;
  logic [TOKEN_DEPTH-1:0] ready_v_q, ready_v_d;

  logic [DEPTH_W-1:0] free_count_q,  free_count_d;
  logic [DEPTH_W-1:0] ready_count_q, ready_count_d;

  logic req_pending_q, req_pending_d;
  free_tok_t req_tok_q, req_tok_d;

  logic overflow_err_q, overflow_err_d;

  integer i;

  function automatic logic id_match_fr(
    input free_tok_t f,
    input ready_tok_t r
  );
    begin
      id_match_fr = (f.row_g  == r.row_g) &&
                    (f.col_g  == r.col_g) &&
                    (f.cgrp_g == r.cgrp_g);
    end
  endfunction

  logic found_ready_match_free;
  logic found_free_match_ready;
  logic [TOKEN_DEPTH-1:0] free_match_mask, ready_match_mask;
  logic [$clog2(TOKEN_DEPTH)-1:0] free_match_idx, ready_match_idx;

  logic free_has_empty, ready_has_empty;
  logic [$clog2(TOKEN_DEPTH)-1:0] free_empty_idx, ready_empty_idx;

  free_tok_t  in_free_tok;
  ready_tok_t in_ready_tok;

  assign in_free_tok.row_g  = free_row_g;
  assign in_free_tok.col_g  = free_col_g;
  assign in_free_tok.col_l  = free_col_l;
  assign in_free_tok.cgrp_g = free_cgrp_g;

  assign in_ready_tok.row_g  = ready_row_g;
  assign in_ready_tok.col_g  = ready_col_g;
  assign in_ready_tok.cgrp_g = ready_cgrp_g;

  always_comb begin
    free_match_mask         = '0;
    ready_match_mask        = '0;
    free_match_idx          = '0;
    ready_match_idx         = '0;
    found_free_match_ready  = 1'b0;
    found_ready_match_free  = 1'b0;

    free_has_empty          = 1'b0;
    ready_has_empty         = 1'b0;
    free_empty_idx          = '0;
    ready_empty_idx         = '0;

    for (i = 0; i < TOKEN_DEPTH; i++) begin
      if (free_v_q[i] && id_match_fr(free_mem[i], in_ready_tok))
        free_match_mask[i] = 1'b1;
      if (ready_v_q[i] && id_match_fr(in_free_tok, ready_mem[i]))
        ready_match_mask[i] = 1'b1;

      if (!free_has_empty && !free_v_q[i]) begin
        free_has_empty = 1'b1;
        free_empty_idx = i[$clog2(TOKEN_DEPTH)-1:0];
      end
      if (!ready_has_empty && !ready_v_q[i]) begin
        ready_has_empty = 1'b1;
        ready_empty_idx = i[$clog2(TOKEN_DEPTH)-1:0];
      end
    end

    for (i = 0; i < TOKEN_DEPTH; i++) begin
      if (!found_free_match_ready && free_match_mask[i]) begin
        found_free_match_ready = 1'b1;
        free_match_idx         = i[$clog2(TOKEN_DEPTH)-1:0];
      end
      if (!found_ready_match_free && ready_match_mask[i]) begin
        found_ready_match_free = 1'b1;
        ready_match_idx        = i[$clog2(TOKEN_DEPTH)-1:0];
      end
    end
  end

  always_comb begin
    free_v_d       = free_v_q;
    ready_v_d      = ready_v_q;
    free_count_d   = free_count_q;
    ready_count_d  = ready_count_q;
    req_pending_d  = req_pending_q;
    req_tok_d      = req_tok_q;
    overflow_err_d = overflow_err_q;

    if (req_pending_q && refill_req_ready) begin
      req_pending_d = 1'b0;
    end

    if (free_valid) begin
      if (!req_pending_d && found_ready_match_free) begin
        req_pending_d                 = 1'b1;
        req_tok_d.row_g               = free_row_g;
        req_tok_d.col_g               = free_col_g;
        req_tok_d.col_l               = free_col_l;
        req_tok_d.cgrp_g              = free_cgrp_g;
        ready_v_d[ready_match_idx]    = 1'b0;
        if (ready_count_q != 0)
          ready_count_d = ready_count_q - 1'b1;
      end
      else if (free_has_empty) begin
        free_v_d[free_empty_idx]      = 1'b1;
        free_count_d                  = free_count_q + 1'b1;
      end
      else begin
        overflow_err_d = 1'b1;
      end
    end

    if (ready_valid) begin
      if (!req_pending_d && found_free_match_ready) begin
        req_pending_d                 = 1'b1;
        req_tok_d                     = free_mem[free_match_idx];
        free_v_d[free_match_idx]      = 1'b0;
        if (free_count_d != 0)
          free_count_d = free_count_d - 1'b1;
      end
      else if (ready_has_empty) begin
        ready_v_d[ready_empty_idx]    = 1'b1;
        ready_count_d                 = ready_count_q + 1'b1;
      end
      else begin
        overflow_err_d = 1'b1;
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      free_v_q       <= '0;
      ready_v_q      <= '0;
      free_count_q   <= '0;
      ready_count_q  <= '0;
      req_pending_q  <= 1'b0;
      req_tok_q      <= '0;
      overflow_err_q <= 1'b0;
    end
    else begin
      free_v_q       <= free_v_d;
      ready_v_q      <= ready_v_d;
      free_count_q   <= free_count_d;
      ready_count_q  <= ready_count_d;
      req_pending_q  <= req_pending_d;
      req_tok_q      <= req_tok_d;
      overflow_err_q <= overflow_err_d;

      if (free_valid && (!( !req_pending_d && found_ready_match_free)) && free_has_empty) begin
        free_mem[free_empty_idx] <= in_free_tok;
      end

      if (ready_valid && (!( !req_pending_d && found_free_match_ready)) && ready_has_empty) begin
        ready_mem[ready_empty_idx] <= in_ready_tok;
      end
    end
  end

  assign refill_req_valid = req_pending_q;
  assign refill_row_g     = req_tok_q.row_g;
  assign refill_col_g     = req_tok_q.col_g;
  assign refill_col_l     = req_tok_q.col_l;
  assign refill_cgrp_g    = req_tok_q.cgrp_g;

  assign free_fifo_full   = (free_count_q == TOKEN_DEPTH[DEPTH_W-1:0]);
  assign ready_fifo_full  = (ready_count_q == TOKEN_DEPTH[DEPTH_W-1:0]);
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
