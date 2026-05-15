module global_scheduler_fsm (
  input  logic clk,
  input  logic rst_n,

  input  logic start,
  input  logic abort,

  input  logic cur_valid,
  input  logic next_valid,
  input  logic cur_first_layer,
  input  logic cur_last_layer,
  input  logic cur_mode,   // 0: mode1, 1: mode2
  input  logic next_mode,  // valid only when next_valid=1

  // Current compute-bank readiness (from weight_bank_manager)
  input  logic bank_compute_ready,

  // Phase-completion inputs
  input  logic ifm_load_done,
  input  logic wgt_load_done,
  input  logic compute_done,
  input  logic compute_busy,
  input  logic ofm_layer_write_done,
  input  logic ofm_ifm_stream_done,
  input  logic ofm_store_done,

  input  logic any_error,

  // Kick/pulse outputs to sub-managers
  output logic kick_ifm_load,
  output logic kick_wgt_preload,
  output logic kick_compute,
  output logic kick_same_mode_stream,
  output logic kick_transition_stream,
  output logic kick_ofm_store,

  // Coarse hold to compute_dispatcher.
  // Same-mode refill is NOT scheduler-driven anymore.
  output logic hold_compute,

  // Layer / bank progression
  output logic advance_layer,
  output logic swap_weight_bank,

  // Scheduler status
  output logic sched_busy,
  output logic sched_done,
  output logic sched_error
);

  typedef enum logic [3:0] {
    S_IDLE           = 4'd0,
    S_PREP           = 4'd1,
    S_COMPUTE        = 4'd2,
    S_WAIT_NEXT      = 4'd3,
    S_WAIT_STORE     = 4'd4,
    S_ADVANCE        = 4'd5,
    S_DONE           = 4'd6,
    S_ERROR          = 4'd7,
    // M1->M2 transition must not be kicked at compute_done alone.
    // Wait here until the producer OFM layer is actually closed.
    S_WAIT_TRANS_OFM = 4'd8
  } sched_state_t;

  sched_state_t state_q, state_d;

  logic cur_ifm_req_sent_q,  cur_ifm_req_sent_d;
  logic cur_ifm_ready_q,     cur_ifm_ready_d;
  logic cur_wgt_req_sent_q,  cur_wgt_req_sent_d;

  logic next_wgt_req_sent_q, next_wgt_req_sent_d;
  logic next_wgt_ready_q,    next_wgt_ready_d;

  logic transition_req_sent_q, transition_req_sent_d;
  logic store_req_sent_q,      store_req_sent_d;

  // M1->M2 transition must wait for an OFM-done event that belongs to
  // the current producer layer, not a stale high level from the previous
  // layer.  This flag is reset when a new compute transaction is kicked
  // and is set only after ofm_layer_write_done has been observed low
  // during that transaction/wait phase.
  logic trans_ofm_seen_low_q, trans_ofm_seen_low_d;

  logic need_initial_ifm;
  logic need_same_mode_refill;
  logic need_transition_stream;
  logic unsupported_transition;

  assign need_initial_ifm       = (cur_valid === 1'b1) && (cur_first_layer === 1'b1);
  assign need_same_mode_refill  = next_valid && (cur_mode == next_mode);
  assign need_transition_stream = next_valid && (cur_mode == 1'b0) && (next_mode == 1'b1);
  assign unsupported_transition = next_valid && (cur_mode == 1'b1) && (next_mode == 1'b0);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state_q                <= S_IDLE;
      cur_ifm_req_sent_q     <= 1'b0;
      cur_ifm_ready_q        <= 1'b0;
      cur_wgt_req_sent_q     <= 1'b0;
      next_wgt_req_sent_q    <= 1'b0;
      next_wgt_ready_q       <= 1'b0;
      transition_req_sent_q  <= 1'b0;
      store_req_sent_q       <= 1'b0;
      trans_ofm_seen_low_q   <= 1'b0;
    end
    else begin
      state_q                <= state_d;
      cur_ifm_req_sent_q     <= cur_ifm_req_sent_d;
      cur_ifm_ready_q        <= cur_ifm_ready_d;
      cur_wgt_req_sent_q     <= cur_wgt_req_sent_d;
      next_wgt_req_sent_q    <= next_wgt_req_sent_d;
      next_wgt_ready_q       <= next_wgt_ready_d;
      transition_req_sent_q  <= transition_req_sent_d;
      store_req_sent_q       <= store_req_sent_d;
      trans_ofm_seen_low_q   <= trans_ofm_seen_low_d;
    end
  end

  always_comb begin
    state_d               = state_q;

    cur_ifm_req_sent_d    = cur_ifm_req_sent_q;
    cur_ifm_ready_d       = cur_ifm_ready_q;
    cur_wgt_req_sent_d    = cur_wgt_req_sent_q;
    next_wgt_req_sent_d   = next_wgt_req_sent_q;
    next_wgt_ready_d      = next_wgt_ready_q;
    transition_req_sent_d = transition_req_sent_q;
    store_req_sent_d      = store_req_sent_q;
    trans_ofm_seen_low_d  = trans_ofm_seen_low_q;

    kick_ifm_load          = 1'b0;
    kick_wgt_preload       = 1'b0;
    kick_compute           = 1'b0;
    kick_same_mode_stream  = 1'b0;
    kick_transition_stream = 1'b0;
    kick_ofm_store         = 1'b0;

    hold_compute           = 1'b1;

    advance_layer          = 1'b0;
    swap_weight_bank       = 1'b0;

    sched_busy             = 1'b1;
    sched_done             = 1'b0;
    sched_error            = 1'b0;

    if (!rst_n) begin
      state_d               = S_IDLE;
      cur_ifm_req_sent_d    = 1'b0;
      cur_ifm_ready_d       = 1'b0;
      cur_wgt_req_sent_d    = 1'b0;
      next_wgt_req_sent_d   = 1'b0;
      next_wgt_ready_d      = 1'b0;
      transition_req_sent_d = 1'b0;
      store_req_sent_d      = 1'b0;
      trans_ofm_seen_low_d  = 1'b0;

      hold_compute          = 1'b1;
      sched_busy            = 1'b0;
    end
    else begin
      // Latch completions into per-layer coarse flags
      if (ifm_load_done)
        cur_ifm_ready_d = 1'b1;

      if (wgt_load_done) begin
        if (next_wgt_req_sent_q)
          next_wgt_ready_d = 1'b1;
      end

      case (state_q)
        S_IDLE: begin
          sched_busy = 1'b0;
          hold_compute = 1'b1;

          cur_ifm_req_sent_d    = 1'b0;
          cur_ifm_ready_d       = 1'b0;
          cur_wgt_req_sent_d    = 1'b0;
          next_wgt_req_sent_d   = 1'b0;
          next_wgt_ready_d      = 1'b0;
          transition_req_sent_d = 1'b0;
          store_req_sent_d      = 1'b0;
          trans_ofm_seen_low_d  = 1'b0;

          if (start) begin
            state_d = S_PREP;
          end
        end

        S_PREP: begin
          hold_compute = 1'b1;
          if (!cur_valid) begin
            state_d = S_PREP;
          end else begin
            if (abort || any_error) begin
              state_d = S_ERROR;
            end else begin
              if (need_initial_ifm && !cur_ifm_req_sent_q) begin
                kick_ifm_load      = 1'b1;
                cur_ifm_req_sent_d = 1'b1;
              end

              if (cur_first_layer && !cur_wgt_req_sent_q) begin
                kick_wgt_preload   = 1'b1;
                cur_wgt_req_sent_d = 1'b1;
              end

              if (((!need_initial_ifm) || (cur_ifm_ready_q === 1'b1)) &&
                  (bank_compute_ready === 1'b1)) begin
                kick_compute          = 1'b1;
                trans_ofm_seen_low_d = 1'b0;
                state_d               = S_COMPUTE;
              end
            end
          end
        end

        S_COMPUTE: begin
          hold_compute = 1'b0;

          if (abort || any_error) begin
            state_d = S_ERROR;
          end
          else begin
            if (next_valid && !next_wgt_req_sent_q) begin
              kick_wgt_preload    = 1'b1;
              next_wgt_req_sent_d = 1'b1;
            end

            // For M1->M2, qualify the producer OFM-done level by first
            // observing it low during the current compute/wait phase.
            // This prevents a stale high from the previous layer from
            // authorizing an early transition.
            if (need_transition_stream && !ofm_layer_write_done) begin
              trans_ofm_seen_low_d = 1'b1;
            end

            if (compute_done) begin
              hold_compute = 1'b1;

              if (cur_last_layer) begin
                if (!store_req_sent_q) begin
                  kick_ofm_store   = 1'b1;
                  store_req_sent_d = 1'b1;
                end
                state_d = S_WAIT_STORE;
              end
              else if (!next_valid) begin
                state_d = S_ERROR;
              end
              else if (unsupported_transition) begin
                state_d = S_ERROR;
              end
              else if (need_transition_stream) begin
                // M1->M2 is a layout-changing transition.  compute_done only
                // ends the compute phase; the transition must be kicked from
                // S_WAIT_TRANS_OFM after a current-layer OFM-done event is
                // observed.  Do not sample raw ofm_layer_write_done here,
                // because it may still be a stale level from the previous
                // layer in the compute_done cycle.
                if (!ofm_layer_write_done) begin
                  trans_ofm_seen_low_d = 1'b1;
                end
                state_d = S_WAIT_TRANS_OFM;
              end
              else if (need_same_mode_refill) begin
                state_d = S_WAIT_NEXT;
              end
              else begin
                state_d = S_ERROR;
              end
            end
          end
        end

        S_WAIT_TRANS_OFM: begin
          hold_compute = 1'b1;

          if (abort || any_error) begin
            state_d = S_ERROR;
          end
          else if (!need_transition_stream) begin
            state_d = S_ERROR;
          end
          else begin
            if (!ofm_layer_write_done) begin
              trans_ofm_seen_low_d = 1'b1;
            end

            // Accept OFM done only after it has been observed low in this
            // producer layer.  A high level seen before that may belong to
            // the previous layer and must not kick M1->M2 transition.
            if (trans_ofm_seen_low_q && ofm_layer_write_done) begin
              if (!transition_req_sent_q) begin
                kick_transition_stream = 1'b1;
                transition_req_sent_d  = 1'b1;
              end
              state_d = S_WAIT_NEXT;
            end
          end
        end

        S_WAIT_NEXT: begin
          hold_compute = 1'b1;

          if (abort || any_error) begin
            state_d = S_ERROR;
          end
          else if (need_transition_stream) begin
            if (ofm_ifm_stream_done && next_wgt_ready_q) begin
              swap_weight_bank = 1'b1;
              advance_layer    = 1'b1;
              state_d          = S_ADVANCE;
            end
          end
          else if (need_same_mode_refill) begin
            if (ofm_ifm_stream_done && next_wgt_ready_q) begin
              swap_weight_bank = 1'b1;
              advance_layer    = 1'b1;
              state_d          = S_ADVANCE;
            end
          end
          else begin
            state_d = S_ERROR;
          end
        end

        S_WAIT_STORE: begin
          hold_compute = 1'b1;

          if (abort || any_error) begin
            state_d = S_ERROR;
          end
          else if (ofm_store_done) begin
            state_d = S_DONE;
          end
        end

        S_ADVANCE: begin
          hold_compute = 1'b1;

          cur_ifm_req_sent_d    = 1'b0;
          cur_ifm_ready_d       = 1'b1;
          cur_wgt_req_sent_d    = 1'b0;
          next_wgt_req_sent_d   = 1'b0;
          next_wgt_ready_d      = 1'b0;
          transition_req_sent_d = 1'b0;
          store_req_sent_d      = 1'b0;
          trans_ofm_seen_low_d  = 1'b0;

          state_d = S_PREP;
        end

        S_DONE: begin
          sched_busy   = 1'b0;
          sched_done   = 1'b1;
          hold_compute = 1'b1;

          if (!start)
            state_d = S_IDLE;
        end

        S_ERROR: begin
          sched_busy   = 1'b0;
          sched_error  = 1'b1;
          hold_compute = 1'b1;

          if (!start)
            state_d = S_IDLE;
        end

        default: begin
          state_d = S_ERROR;
        end
      endcase

      if (compute_busy) begin
        // no-op
      end
      if (ofm_layer_write_done) begin
        // no-op
      end
    end
  end

endmodule
