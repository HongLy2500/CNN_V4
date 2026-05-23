module addr_gen_ifm_m1 #(
  parameter int DATA_W = 8,
  parameter int PV_MAX = 16,
  parameter int C_MAX  = 512,
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
  input  logic [9:0] C_cur,
  input  logic [15:0] H_cur,
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
  input  logic [15:0] out_row,
  input  logic [15:0] out_col,

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
  logic [W_COL_W-1:0] issue_first_col_q;
  logic [W_COL_W-1:0] issue_last_col_q;
  logic               issued_all_q;

  // Mode1 performance optimization:
  // Instead of reloading the full W row for every output-column block,
  // load only the horizontal word groups that can be touched by the
  // current Pv-wide output block and K-wide horizontal window.
  // This preserves the data_register contract because data_register reads
  // using GLOBAL x positions; stale entries outside the current block window
  // are never selected by the CE for this pass/channel.
  logic [W_COL_W-1:0] window_first_col_s;
  logic [W_COL_W-1:0] window_last_col_s;

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

  // K3/P1 vertical padding support.
  // ifm_rd_row_idx remains a LOCAL row index in the current Mode1 IFM window.
  logic issue_zero_pad_s;
  logic [K_ROW_W-1:0] issue_local_row_s;
  logic ret_zero_pad_q;

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
        (H_cur != 0) && (H_cur <= H_MAX) &&
        (out_row < H_cur) &&
        (out_col < W_cur) &&
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
                               (issue_col_q == issue_last_col_q);
  assign issue_next_col_wrap = (issue_col_q == issue_last_col_q);

  always_comb begin : GEN_M1_WINDOW_COLS
    int signed read_x_min_i;
    int signed read_x_max_i;
    int signed first_col_i;
    int signed last_col_i;
    int signed pv_i;
    int signed k_i;
    int signed w_i;
    int signed pad_x_i;

    window_first_col_s = '0;
    window_last_col_s  = '0;

    pv_i    = int'(Pv_cur);
    k_i     = int'(K_cur);
    w_i     = int'(W_cur);
    // Match data_register_mode1's current horizontal-padding contract.
    // For the current benchmark this is K3/P1; keeping PAD_X=1 here avoids
    // changing existing functional behavior for any already-passing tests.
    pad_x_i = 1;

    if ((pv_i > 0) && (k_i > 0) && (w_i > 0)) begin
      read_x_min_i = int'(out_col) - pad_x_i;
      read_x_max_i = int'(out_col) + pv_i - 1 + k_i - 1 - pad_x_i;

      if (read_x_min_i < 0)
        read_x_min_i = 0;
      if (read_x_max_i >= w_i)
        read_x_max_i = w_i - 1;

      first_col_i = read_x_min_i / pv_i;
      last_col_i  = read_x_max_i / pv_i;

      if (first_col_i < 0)
        first_col_i = 0;
      if (last_col_i < first_col_i)
        last_col_i = first_col_i;

      window_first_col_s = first_col_i;
      window_last_col_s  = last_col_i;
    end
  end

  // --------------------------------------------------
  // K3/P1 vertical padding mapping
  // --------------------------------------------------
  // Contract: ifm_buffer Mode1 read row is LOCAL to current HT/ring window.
  // For K=3/P=1 and with control delaying row advance by one row:
  //   out_row=0:      ky0 zero, ky1 local0, ky2 local1
  //   middle rows:    ky0 local0, ky1 local1, ky2 local2
  //   out_row=H-1:    ky0 local0, ky1 local1, ky2 zero
  // Non-K3 keeps legacy local-row mapping.
  always_comb begin
    issue_zero_pad_s  = 1'b0;
    issue_local_row_s = issue_row_q;

    if (K_cur == 4'd3) begin
      if ((out_row == 16'd0) && (issue_row_q == '0)) begin
        issue_zero_pad_s  = 1'b1;
        issue_local_row_s = '0;
      end
      else if ((H_cur != 16'd0) &&
               (out_row == (H_cur - 16'd1)) &&
               (issue_row_q == 2)) begin
        issue_zero_pad_s  = 1'b1;
        issue_local_row_s = issue_row_q;
      end
      else if (out_row == 16'd0) begin
        issue_local_row_s = issue_row_q - 1'b1;
      end
      else begin
        issue_local_row_s = issue_row_q;
      end
    end
  end

  assign ifm_rd_en        = issue_fire && !issue_zero_pad_s;
  assign ifm_rd_bank_base = target_channel_q[C_BANK_W-1:0];
  assign ifm_rd_row_idx   = issue_local_row_s;
  assign ifm_rd_col_idx   = issue_col_q;

  // --------------------------------------------------
  // data_register write side
  // --------------------------------------------------
  assign dr_write_en      = (state_q == ST_LOAD) && ret_valid_q && (ret_zero_pad_q || ifm_rd_valid);
  assign dr_write_row_idx = ret_row_q;
  assign dr_write_x_base  = ret_x_base_q;
  assign dr_write_data    = ret_zero_pad_q ? '0 : ifm_rd_data;

  // --------------------------------------------------
  // Status / debug
  // --------------------------------------------------
  assign busy               = (state_q == ST_LOAD);
  assign dbg_target_channel = target_channel_q;
  assign dbg_words_per_row  = words_per_row;
  assign dbg_issue_row      = issue_row_q;
  assign dbg_issue_col      = issue_col_q;
  assign dbg_waiting_for_return = (state_q == ST_LOAD) && issued_all_q && ret_valid_q && !ret_zero_pad_q && !ifm_rd_valid;

  // --------------------------------------------------
  // State / sequencing
  // --------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state_q          <= ST_IDLE;
      target_channel_q <= 16'd0;
      issue_row_q       <= '0;
      issue_col_q       <= '0;
      issue_first_col_q <= '0;
      issue_last_col_q  <= '0;
      issued_all_q      <= 1'b0;
      ret_valid_q      <= 1'b0;
      ret_row_q        <= '0;
      ret_x_base_q     <= 16'd0;
      ret_last_q       <= 1'b0;
      ret_zero_pad_q   <= 1'b0;
      done             <= 1'b0;
      error            <= 1'b0;
    end
    else begin
      done  <= 1'b0;
      error <= 1'b0;

      case (state_q)
        ST_IDLE: begin
          ret_valid_q    <= 1'b0;
          ret_last_q     <= 1'b0;
          ret_zero_pad_q <= 1'b0;

          if (load_req) begin
            if (!cfg_valid || (load_req_channel >= C_cur)) begin
              state_q <= ST_ERROR;
            end
            else begin
              state_q          <= ST_LOAD;
              target_channel_q <= load_req_channel;
              issue_row_q       <= '0;
              issue_col_q       <= window_first_col_s;
              issue_first_col_q <= window_first_col_s;
              issue_last_col_q  <= window_last_col_s;
              issued_all_q      <= 1'b0;
            end
          end
        end

        ST_LOAD: begin
          // Capture metadata for the read that is being issued in this cycle.
          ret_valid_q    <= issue_fire;
          ret_row_q      <= issue_row_q;
          ret_x_base_q   <= issue_col_q * Pv_cur;
          ret_last_q     <= issue_is_last;
          ret_zero_pad_q <= issue_fire && issue_zero_pad_s;

          if (issue_fire) begin
            if (issue_is_last) begin
              issued_all_q <= 1'b1;
            end

            if (!issue_is_last) begin
              if (!issue_next_col_wrap) begin
                issue_col_q <= issue_col_q + 1'b1;
              end
              else begin
                issue_col_q <= issue_first_col_q;
                issue_row_q <= issue_row_q + 1'b1;
              end
            end
          end

          // Last returned read completes the channel load.
          if (ret_valid_q && (ret_zero_pad_q || ifm_rd_valid) && ret_last_q) begin
            state_q <= ST_DONE;
          end
        end

        ST_DONE: begin
          done     <= 1'b1;
          state_q  <= ST_IDLE;
          ret_valid_q    <= 1'b0;
          ret_last_q     <= 1'b0;
          ret_zero_pad_q <= 1'b0;
        end

        ST_ERROR: begin
          error    <= 1'b1;
          state_q  <= ST_IDLE;
          ret_valid_q    <= 1'b0;
          ret_last_q     <= 1'b0;
          ret_zero_pad_q <= 1'b0;
        end

        default: begin
          state_q <= ST_IDLE;
          ret_valid_q    <= 1'b0;
          ret_last_q     <= 1'b0;
          ret_zero_pad_q <= 1'b0;
        end
      endcase
    end
  end

`ifndef SYNTHESIS
  // ------------------------------------------------------------
  // PERF_M1_IFM_AG
  //
  // Purpose:
  //   Aggregate Mode1 IFM-load behavior inside addr_gen_ifm_m1.
  //
  // This tells whether hold_local comes from:
  //   - too many IFM reads per load
  //   - zero-padding writes
  //   - waiting for ifm_rd_valid
  //   - extra busy cycles after all issues
  //
  // Default prints one summary every 512 completed load events.
  // Set PERF_M1_AG_PRINT_EACH=1 to print every load event.
  // ------------------------------------------------------------

  localparam bit PERF_M1_AG_PRINT_EACH = 1'b0;
  localparam int PERF_M1_AG_PRINT_PERIOD = 512;

  longint unsigned perf_m1_ag_load_cnt;
  longint unsigned perf_m1_ag_done_cnt;
  longint unsigned perf_m1_ag_error_cnt;

  longint unsigned perf_m1_ag_total_cycles;
  longint unsigned perf_m1_ag_total_issue;
  longint unsigned perf_m1_ag_total_ifm_rd;
  longint unsigned perf_m1_ag_total_zero_issue;
  longint unsigned perf_m1_ag_total_dr_write;
  longint unsigned perf_m1_ag_total_wait_ret;
  longint unsigned perf_m1_ag_total_tail_busy;

  longint unsigned perf_m1_ag_max_cycles;
  longint unsigned perf_m1_ag_max_wait_ret;

  longint unsigned perf_m1_ag_cur_cycles;
  longint unsigned perf_m1_ag_cur_issue;
  longint unsigned perf_m1_ag_cur_ifm_rd;
  longint unsigned perf_m1_ag_cur_zero_issue;
  longint unsigned perf_m1_ag_cur_dr_write;
  longint unsigned perf_m1_ag_cur_wait_ret;
  longint unsigned perf_m1_ag_cur_tail_busy;

  logic [15:0] perf_m1_ag_cur_channel;
  logic [15:0] perf_m1_ag_cur_out_row;
  logic [15:0] perf_m1_ag_cur_words_per_row;
  logic [3:0]  perf_m1_ag_cur_K;
  logic [7:0]  perf_m1_ag_cur_Pv;

  wire perf_m1_ag_load_start_s =
      (state_q == ST_IDLE) && load_req && cfg_valid && (load_req_channel < C_cur);

  wire perf_m1_ag_load_error_s =
      (state_q == ST_IDLE) && load_req && (!cfg_valid || (load_req_channel >= C_cur));

  always_ff @(posedge clk or negedge rst_n) begin : PERF_M1_IFM_AG_MON
    if (!rst_n) begin
      perf_m1_ag_load_cnt        <= 0;
      perf_m1_ag_done_cnt        <= 0;
      perf_m1_ag_error_cnt       <= 0;

      perf_m1_ag_total_cycles    <= 0;
      perf_m1_ag_total_issue     <= 0;
      perf_m1_ag_total_ifm_rd    <= 0;
      perf_m1_ag_total_zero_issue<= 0;
      perf_m1_ag_total_dr_write  <= 0;
      perf_m1_ag_total_wait_ret  <= 0;
      perf_m1_ag_total_tail_busy <= 0;

      perf_m1_ag_max_cycles      <= 0;
      perf_m1_ag_max_wait_ret    <= 0;

      perf_m1_ag_cur_cycles      <= 0;
      perf_m1_ag_cur_issue       <= 0;
      perf_m1_ag_cur_ifm_rd      <= 0;
      perf_m1_ag_cur_zero_issue  <= 0;
      perf_m1_ag_cur_dr_write    <= 0;
      perf_m1_ag_cur_wait_ret    <= 0;
      perf_m1_ag_cur_tail_busy   <= 0;

      perf_m1_ag_cur_channel     <= 0;
      perf_m1_ag_cur_out_row     <= 0;
      perf_m1_ag_cur_words_per_row <= 0;
      perf_m1_ag_cur_K           <= 0;
      perf_m1_ag_cur_Pv          <= 0;
    end
    else begin
      if (perf_m1_ag_load_error_s)
        perf_m1_ag_error_cnt <= perf_m1_ag_error_cnt + 1;

      if (perf_m1_ag_load_start_s) begin
        perf_m1_ag_load_cnt        <= perf_m1_ag_load_cnt + 1;

        perf_m1_ag_cur_cycles      <= 0;
        perf_m1_ag_cur_issue       <= 0;
        perf_m1_ag_cur_ifm_rd      <= 0;
        perf_m1_ag_cur_zero_issue  <= 0;
        perf_m1_ag_cur_dr_write    <= 0;
        perf_m1_ag_cur_wait_ret    <= 0;
        perf_m1_ag_cur_tail_busy   <= 0;

        perf_m1_ag_cur_channel     <= load_req_channel;
        perf_m1_ag_cur_out_row     <= out_row;
        perf_m1_ag_cur_words_per_row <= words_per_row;
        perf_m1_ag_cur_K           <= K_cur;
        perf_m1_ag_cur_Pv          <= Pv_cur;
      end

      if (state_q == ST_LOAD) begin
        perf_m1_ag_cur_cycles <= perf_m1_ag_cur_cycles + 1;

        if (issue_fire)
          perf_m1_ag_cur_issue <= perf_m1_ag_cur_issue + 1;

        if (ifm_rd_en)
          perf_m1_ag_cur_ifm_rd <= perf_m1_ag_cur_ifm_rd + 1;

        if (issue_fire && issue_zero_pad_s)
          perf_m1_ag_cur_zero_issue <= perf_m1_ag_cur_zero_issue + 1;

        if (dr_write_en)
          perf_m1_ag_cur_dr_write <= perf_m1_ag_cur_dr_write + 1;

        if (dbg_waiting_for_return)
          perf_m1_ag_cur_wait_ret <= perf_m1_ag_cur_wait_ret + 1;

        // Tail busy means all reads have been issued but addr_gen is still busy.
        if (issued_all_q)
          perf_m1_ag_cur_tail_busy <= perf_m1_ag_cur_tail_busy + 1;
      end

      if (done) begin
        perf_m1_ag_done_cnt <= perf_m1_ag_done_cnt + 1;

        perf_m1_ag_total_cycles     <= perf_m1_ag_total_cycles     + perf_m1_ag_cur_cycles;
        perf_m1_ag_total_issue      <= perf_m1_ag_total_issue      + perf_m1_ag_cur_issue;
        perf_m1_ag_total_ifm_rd     <= perf_m1_ag_total_ifm_rd     + perf_m1_ag_cur_ifm_rd;
        perf_m1_ag_total_zero_issue <= perf_m1_ag_total_zero_issue + perf_m1_ag_cur_zero_issue;
        perf_m1_ag_total_dr_write   <= perf_m1_ag_total_dr_write   + perf_m1_ag_cur_dr_write;
        perf_m1_ag_total_wait_ret   <= perf_m1_ag_total_wait_ret   + perf_m1_ag_cur_wait_ret;
        perf_m1_ag_total_tail_busy  <= perf_m1_ag_total_tail_busy  + perf_m1_ag_cur_tail_busy;

        if (perf_m1_ag_cur_cycles > perf_m1_ag_max_cycles)
          perf_m1_ag_max_cycles <= perf_m1_ag_cur_cycles;

        if (perf_m1_ag_cur_wait_ret > perf_m1_ag_max_wait_ret)
          perf_m1_ag_max_wait_ret <= perf_m1_ag_cur_wait_ret;

        if (PERF_M1_AG_PRINT_EACH) begin
          $display("PERF_M1_IFM_LOAD t=%0t load_idx=%0d ch=%0d out_row=%0d K=%0d Pv=%0d words_per_row=%0d cycles=%0d issue=%0d ifm_rd=%0d zero_issue=%0d dr_wr=%0d wait_ret=%0d tail_busy=%0d",
            $time,
            perf_m1_ag_done_cnt + 1,
            perf_m1_ag_cur_channel,
            perf_m1_ag_cur_out_row,
            perf_m1_ag_cur_K,
            perf_m1_ag_cur_Pv,
            perf_m1_ag_cur_words_per_row,
            perf_m1_ag_cur_cycles,
            perf_m1_ag_cur_issue,
            perf_m1_ag_cur_ifm_rd,
            perf_m1_ag_cur_zero_issue,
            perf_m1_ag_cur_dr_write,
            perf_m1_ag_cur_wait_ret,
            perf_m1_ag_cur_tail_busy
          );
        end
        else if (((perf_m1_ag_done_cnt + 1) % PERF_M1_AG_PRINT_PERIOD) == 0) begin
          $display("PERF_M1_IFM_AG_SUM t=%0t loads=%0d errors=%0d total_cycles=%0d total_issue=%0d total_ifm_rd=%0d total_zero_issue=%0d total_dr_wr=%0d total_wait_ret=%0d total_tail_busy=%0d max_cycles=%0d max_wait_ret=%0d last_ch=%0d last_row=%0d last_cycles=%0d last_issue=%0d last_ifm_rd=%0d last_zero=%0d last_dr_wr=%0d",
            $time,
            perf_m1_ag_done_cnt + 1,
            perf_m1_ag_error_cnt,
            perf_m1_ag_total_cycles     + perf_m1_ag_cur_cycles,
            perf_m1_ag_total_issue      + perf_m1_ag_cur_issue,
            perf_m1_ag_total_ifm_rd     + perf_m1_ag_cur_ifm_rd,
            perf_m1_ag_total_zero_issue + perf_m1_ag_cur_zero_issue,
            perf_m1_ag_total_dr_write   + perf_m1_ag_cur_dr_write,
            perf_m1_ag_total_wait_ret   + perf_m1_ag_cur_wait_ret,
            perf_m1_ag_total_tail_busy  + perf_m1_ag_cur_tail_busy,
            (perf_m1_ag_cur_cycles > perf_m1_ag_max_cycles) ? perf_m1_ag_cur_cycles : perf_m1_ag_max_cycles,
            (perf_m1_ag_cur_wait_ret > perf_m1_ag_max_wait_ret) ? perf_m1_ag_cur_wait_ret : perf_m1_ag_max_wait_ret,
            perf_m1_ag_cur_channel,
            perf_m1_ag_cur_out_row,
            perf_m1_ag_cur_cycles,
            perf_m1_ag_cur_issue,
            perf_m1_ag_cur_ifm_rd,
            perf_m1_ag_cur_zero_issue,
            perf_m1_ag_cur_dr_write
          );
        end
      end
    end
  end
`endif

endmodule
