module addr_gen_ifm_m1 #(
  parameter int DATA_W = 8,
  parameter int PV_MAX = 8,
  parameter int C_MAX  = 64,
  parameter int W_MAX  = 224,
  parameter int H_MAX  = 224,
  parameter int K_MAX  = 7
)(
  input  logic clk,
  input  logic rst_n,

  // --------------------------------------------------
  // Runtime configuration for the current layer / block
  // --------------------------------------------------
  input  logic [3:0] K_cur,
  input  logic [7:0] C_cur,
  input  logic [15:0] W_cur,
  input  logic [7:0] Pv_cur,

  // --------------------------------------------------
  // Triggers from ce_controller_mode1
  //
  // pass_start_pulse:
  //   start loading channel 0 for a new output block
  //   (out_row, f_group, out_col)
  //
  // chan_done_pulse:
  //   current channel finished for this block.
  //   If there is another channel, start loading c_iter+1.
  //
  // IMPORTANT integration rule:
  //   control_unit should hold mode1 step_en = 0 while busy=1,
  //   so the CE waits until data_register has been refilled for
  //   the required channel.
  // --------------------------------------------------
  input  logic        pass_start_pulse,
  input  logic        chan_done_pulse,
  input  logic [15:0] c_iter,

  // --------------------------------------------------
  // Read port from ifm_buffer (mode 1)
  // --------------------------------------------------
  output logic                     ifm_rd_en,
  output logic [$clog2(C_MAX)-1:0] ifm_rd_bank_base,
  output logic [$clog2(H_MAX)-1:0] ifm_rd_row_idx,
  output logic [$clog2(W_MAX)-1:0] ifm_rd_col_idx,
  input  logic                     ifm_rd_valid,
  input  logic [PV_MAX*DATA_W-1:0] ifm_rd_data,

  // --------------------------------------------------
  // Write port to data_register_mode1
  // --------------------------------------------------
  output logic                     dr_write_en,
  output logic [$clog2(K_MAX)-1:0] dr_write_row_idx,
  output logic [15:0]              dr_write_x_base,
  output logic [PV_MAX*DATA_W-1:0] dr_write_data,

  // --------------------------------------------------
  // Status back to control_unit
  // --------------------------------------------------
  output logic                     busy,
  output logic                     done,
  output logic                     error,

  // Optional debug / visibility
  output logic [15:0]              dbg_target_channel,
  output logic [15:0]              dbg_words_per_row,
  output logic [$clog2(K_MAX)-1:0] dbg_issue_row,
  output logic [$clog2(W_MAX)-1:0] dbg_issue_col,
  output logic                     dbg_waiting_for_return
);

  localparam int C_BANK_W = (C_MAX <= 1) ? 1 : $clog2(C_MAX);
  localparam int H_ROW_W  = (H_MAX <= 1) ? 1 : $clog2(H_MAX);
  localparam int W_COL_W  = (W_MAX <= 1) ? 1 : $clog2(W_MAX);
  localparam int K_ROW_W  = (K_MAX <= 1) ? 1 : $clog2(K_MAX);

  typedef enum logic [1:0] {
    ST_IDLE,
    ST_LOAD,
    ST_DONE,
    ST_ERROR
  } state_t;

  state_t state_q;

  logic [15:0] target_channel_q;
  logic [15:0] words_per_row;
  logic        cfg_valid;

  logic [K_ROW_W-1:0] issue_row_q;
  logic [W_COL_W-1:0] issue_col_q;
  logic               issued_all_q;

  // Metadata delayed by one cycle to match ifm_buffer read latency.
  logic               ret_valid_q;
  logic [K_ROW_W-1:0] ret_row_q;
  logic [15:0]        ret_x_base_q;
  logic               ret_last_q;

  logic load_req;
  logic [15:0] load_req_channel;
  logic issue_fire;
  logic issue_is_last;
  logic issue_next_col_wrap;

  // --------------------------------------------------
  // Derived values / legality
  // --------------------------------------------------
  always_comb begin
    if (Pv_cur != 0)
      words_per_row = (W_cur + Pv_cur - 1) / Pv_cur;
    else
      words_per_row = 16'd0;
  end

  always_comb begin
    cfg_valid = 1'b0;
    if ((K_cur != 0) && (K_cur <= K_MAX) &&
        (C_cur != 0) && (C_cur <= C_MAX) &&
        (W_cur != 0) && (W_cur <= W_MAX) &&
        (Pv_cur != 0) && (Pv_cur <= PV_MAX) &&
        (words_per_row != 0)) begin
      cfg_valid = 1'b1;
    end
  end

  // --------------------------------------------------
  // Request decode
  // --------------------------------------------------
  always_comb begin
    load_req         = 1'b0;
    load_req_channel = 16'd0;

    // Start of a new output block: load channel 0.
    if (pass_start_pulse) begin
      load_req         = 1'b1;
      load_req_channel = 16'd0;
    end
    // End of one channel inside the same output block: load next channel.
    else if (chan_done_pulse && ((c_iter + 16'd1) < C_cur)) begin
      load_req         = 1'b1;
      load_req_channel = c_iter + 16'd1;
    end
  end

  // --------------------------------------------------
  // Read issuing side
  // --------------------------------------------------
  assign issue_fire          = (state_q == ST_LOAD) && !issued_all_q;
  assign issue_is_last       = issue_fire &&
                               (issue_row_q == K_cur[K_ROW_W-1:0] - 1'b1) &&
                               (issue_col_q == words_per_row[W_COL_W-1:0] - 1'b1);
  assign issue_next_col_wrap = (issue_col_q == words_per_row[W_COL_W-1:0] - 1'b1);

  assign ifm_rd_en        = issue_fire;
  assign ifm_rd_bank_base = target_channel_q[C_BANK_W-1:0];
  assign ifm_rd_row_idx   = issue_row_q;
  assign ifm_rd_col_idx   = issue_col_q;

  // --------------------------------------------------
  // data_register write side
  // --------------------------------------------------
  assign dr_write_en      = (state_q == ST_LOAD) && ret_valid_q && ifm_rd_valid;
  assign dr_write_row_idx = ret_row_q;
  assign dr_write_x_base  = ret_x_base_q;
  assign dr_write_data    = ifm_rd_data;

  // --------------------------------------------------
  // Status / debug
  // --------------------------------------------------
  assign busy               = (state_q == ST_LOAD);
  assign dbg_target_channel = target_channel_q;
  assign dbg_words_per_row  = words_per_row;
  assign dbg_issue_row      = issue_row_q;
  assign dbg_issue_col      = issue_col_q;
  assign dbg_waiting_for_return = (state_q == ST_LOAD) && issued_all_q && ret_valid_q && !ifm_rd_valid;

  // --------------------------------------------------
  // State / sequencing
  // --------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state_q          <= ST_IDLE;
      target_channel_q <= 16'd0;
      issue_row_q      <= '0;
      issue_col_q      <= '0;
      issued_all_q     <= 1'b0;
      ret_valid_q      <= 1'b0;
      ret_row_q        <= '0;
      ret_x_base_q     <= 16'd0;
      ret_last_q       <= 1'b0;
      done             <= 1'b0;
      error            <= 1'b0;
    end
    else begin
      done  <= 1'b0;
      error <= 1'b0;

      case (state_q)
        ST_IDLE: begin
          ret_valid_q <= 1'b0;
          ret_last_q  <= 1'b0;

          if (load_req) begin
            if (!cfg_valid || (load_req_channel >= C_cur)) begin
              state_q <= ST_ERROR;
            end
            else begin
              state_q          <= ST_LOAD;
              target_channel_q <= load_req_channel;
              issue_row_q      <= '0;
              issue_col_q      <= '0;
              issued_all_q     <= 1'b0;
            end
          end
        end

        ST_LOAD: begin
          // Capture metadata for the read that is being issued in this cycle.
          ret_valid_q  <= issue_fire;
          ret_row_q    <= issue_row_q;
          ret_x_base_q <= issue_col_q * Pv_cur;
          ret_last_q   <= issue_is_last;

          if (issue_fire) begin
            if (issue_is_last) begin
              issued_all_q <= 1'b1;
            end

            if (!issue_is_last) begin
              if (!issue_next_col_wrap) begin
                issue_col_q <= issue_col_q + 1'b1;
              end
              else begin
                issue_col_q <= '0;
                issue_row_q <= issue_row_q + 1'b1;
              end
            end
          end

          // Last returned read completes the channel load.
          if (ret_valid_q && ifm_rd_valid && ret_last_q) begin
            state_q <= ST_DONE;
          end
        end

        ST_DONE: begin
          done     <= 1'b1;
          state_q  <= ST_IDLE;
          ret_valid_q <= 1'b0;
          ret_last_q  <= 1'b0;
        end

        ST_ERROR: begin
          error    <= 1'b1;
          state_q  <= ST_IDLE;
          ret_valid_q <= 1'b0;
          ret_last_q  <= 1'b0;
        end

        default: begin
          state_q <= ST_IDLE;
          ret_valid_q <= 1'b0;
          ret_last_q  <= 1'b0;
        end
      endcase
    end
  end

endmodule
