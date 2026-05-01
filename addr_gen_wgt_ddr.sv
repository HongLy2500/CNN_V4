module addr_gen_wgt_ddr #(
  parameter int PTOTAL      = 16,
  parameter int PC_MODE2    = 8,
  parameter int WGT_DEPTH   = 4096,
  parameter int DDR_ADDR_W  = 32
)(
  input  logic clk,
  input  logic rst_n,

  // --------------------------------------------------
  // Request from control unit / scheduler
  // --------------------------------------------------
  input  logic        start,
  input  logic        cfg_mode,          // 0: mode1, 1: mode2
  input  logic        cfg_buf_sel,       // target weight-buffer bank for preload
  input  logic [3:0]  cfg_k_cur,
  input  logic [7:0]  cfg_c_cur,
  input  logic [7:0]  cfg_f_cur,
  input  logic [7:0]  cfg_pv_mode1_cur,
  input  logic [7:0]  cfg_pf_mode1_cur,
  input  logic [7:0]  cfg_pf_mode2,
  input  logic [DDR_ADDR_W-1:0] cfg_wgt_ddr_base,

  // --------------------------------------------------
  // Handshake/status from DMA
  // --------------------------------------------------
  input  logic dma_busy,
  input  logic dma_done_wgt,
  input  logic dma_error,

  // --------------------------------------------------
  // Command to cnn_dma_direct
  // --------------------------------------------------
  output logic                           wgt_cmd_start,
  output logic                           wgt_cmd_buf_sel,
  output logic [DDR_ADDR_W-1:0]          wgt_cmd_ddr_base,
  output logic [$clog2(WGT_DEPTH+1)-1:0] wgt_cmd_num_words,

  // --------------------------------------------------
  // Status back to control unit
  // --------------------------------------------------
  output logic busy,
  output logic done,
  output logic error,

  // Optional debug / visibility
  output logic [31:0] dbg_num_fgroup,
  output logic [31:0] dbg_num_cgroup,
  output logic [31:0] dbg_num_logical_bundles,
  output logic [31:0] dbg_num_physical_words
);

  localparam int CMD_W = (WGT_DEPTH <= 1) ? 1 : $clog2(WGT_DEPTH+1);

  // Timing note:
  // The original implementation calculated validity, all ceil-div/multiply
  // terms, state transition, and DMA command payload in one combinational
  // path.  In implementation this created a very long path from layer cfg
  // bits to addr_gen_wgt_ddr state_q CE and cnn_dma_direct registers.
  //
  // This version keeps the same count equations and command semantics, but
  // pipelines the calculation before issuing the DMA command:
  //   ST_IDLE   : accept start when DMA is not busy, latch cfg
  //   ST_CALC1  : compute basic group counts and validation
  //   ST_CALC2  : compute products using registered group counts
  //   ST_CALC3  : compute final physical-word count and fit flag
  //   ST_CHECK  : decide error/issue using registered flags only
  //   ST_ISSUE  : emit one-cycle command pulse with registered payload
  //   ST_WAIT_DMA/ST_DONE/ST_ERROR: same completion/error behavior
  //
  // The functional count equations are unchanged.  The observable difference
  // is a few extra cycles of latency from start to wgt_cmd_start.

  typedef enum logic [3:0] {
    ST_IDLE,
    ST_CALC1,
    ST_CALC2,
    ST_CALC3,
    ST_CHECK,
    ST_ISSUE,
    ST_WAIT_DMA,
    ST_DONE,
    ST_ERROR
  } state_t;

  state_t state_q, state_d;

  // Captured configuration for the accepted request.
  logic                  mode_q;
  logic                  buf_sel_q;
  logic [DDR_ADDR_W-1:0] ddr_base_q;
  logic [3:0]            k_q;
  logic [7:0]            c_q;
  logic [7:0]            f_q;
  logic [7:0]            pv_m1_q;
  logic [7:0]            pf_m1_q;
  logic [7:0]            pf_m2_q;

  // Stage 1: group counts / coarse validation.
  logic [31:0] num_fgroup_m1_q;
  logic [31:0] num_fgroup_m2_q;
  logic [31:0] num_cgroup_m2_q;
  logic [31:0] k_sq_q;
  logic        cfg_valid_q;

  logic [31:0] num_fgroup_m1_calc;
  logic [31:0] num_fgroup_m2_calc;
  logic [31:0] num_cgroup_m2_calc;
  logic [31:0] k_sq_calc;
  logic        cfg_valid_calc;

  // Stage 2: products that do not require the final mode1 ceil-div.
  logic [31:0] logical_bundles_m1_q;
  logic [31:0] physical_words_m2_q;

  logic [31:0] logical_bundles_m1_calc;
  logic [31:0] physical_words_m2_calc;

  // Stage 3: final selected physical word count and fit flag.
  logic [31:0] physical_words_m1_q;
  logic [31:0] calc_num_words_q;
  logic        count_fits_q;

  logic [31:0] physical_words_m1_calc;
  logic [31:0] calc_num_words_calc;
  logic        count_fits_calc;

  logic [CMD_W-1:0] num_words_q;

  // --------------------------------------------------
  // Stage 1 calculation from captured cfg
  // --------------------------------------------------
  always_comb begin
    num_fgroup_m1_calc = 32'd0;
    num_fgroup_m2_calc = 32'd0;
    num_cgroup_m2_calc = 32'd0;
    k_sq_calc          = 32'd0;
    cfg_valid_calc     = 1'b0;

    if (k_q != 4'd0) begin
      k_sq_calc = 32'(k_q) * 32'(k_q);
    end

    if (!mode_q) begin
      // Mode 1:
      //   ceil(F / Pf_mode1) * C * K * K logical bundles
      //   ceil(logical_bundles / Pv_mode1) physical words
      if ((k_q != 4'd0) && (c_q != 8'd0) && (f_q != 8'd0) &&
          (pv_m1_q != 8'd0) && (pf_m1_q != 8'd0) &&
          ((32'(pv_m1_q) * 32'(pf_m1_q)) == 32'(PTOTAL))) begin
        num_fgroup_m1_calc = (32'(f_q) + 32'(pf_m1_q) - 32'd1) / 32'(pf_m1_q);
        cfg_valid_calc     = 1'b1;
      end
    end
    else begin
      // Mode 2:
      //   ceil(F / Pf_mode2) * ceil(C / Pc_mode2) * K * K physical words
      if ((k_q != 4'd0) && (c_q != 8'd0) && (f_q != 8'd0) &&
          (pf_m2_q != 8'd0) && ((32'(PC_MODE2) * 32'(pf_m2_q)) == 32'(PTOTAL))) begin
        num_fgroup_m2_calc = (32'(f_q) + 32'(pf_m2_q) - 32'd1) / 32'(pf_m2_q);
        num_cgroup_m2_calc = (32'(c_q) + 32'(PC_MODE2) - 32'd1) / 32'(PC_MODE2);
        cfg_valid_calc     = 1'b1;
      end
    end
  end

  // --------------------------------------------------
  // Stage 2 calculation from registered Stage 1 results
  // --------------------------------------------------
  always_comb begin
    logical_bundles_m1_calc = 32'd0;
    physical_words_m2_calc  = 32'd0;

    if (!mode_q) begin
      logical_bundles_m1_calc = num_fgroup_m1_q * 32'(c_q) * k_sq_q;
    end
    else begin
      physical_words_m2_calc = num_fgroup_m2_q * num_cgroup_m2_q * k_sq_q;
    end
  end

  // --------------------------------------------------
  // Stage 3 calculation from registered Stage 2 results
  // --------------------------------------------------
  always_comb begin
    physical_words_m1_calc = 32'd0;
    calc_num_words_calc    = 32'd0;
    count_fits_calc        = 1'b0;

    if (!mode_q) begin
      if (pv_m1_q != 8'd0) begin
        physical_words_m1_calc = (logical_bundles_m1_q + 32'(pv_m1_q) - 32'd1) / 32'(pv_m1_q);
      end
      calc_num_words_calc = physical_words_m1_calc;
    end
    else begin
      calc_num_words_calc = physical_words_m2_q;
    end

    count_fits_calc = (calc_num_words_calc <= 32'(WGT_DEPTH));
  end

  // Registered debug visibility for the accepted command.
  assign dbg_num_fgroup          = (!mode_q) ? num_fgroup_m1_q      : num_fgroup_m2_q;
  assign dbg_num_cgroup          = (!mode_q) ? 32'd0                 : num_cgroup_m2_q;
  assign dbg_num_logical_bundles = (!mode_q) ? logical_bundles_m1_q : 32'd0;
  assign dbg_num_physical_words  = calc_num_words_q;

  // --------------------------------------------------
  // Next-state / command pulse generation
  // --------------------------------------------------
  always_comb begin
    state_d = state_q;

    wgt_cmd_start     = 1'b0;
    wgt_cmd_buf_sel   = buf_sel_q;
    wgt_cmd_ddr_base  = ddr_base_q;
    wgt_cmd_num_words = num_words_q;

    done  = 1'b0;
    error = 1'b0;

    unique case (state_q)
      ST_IDLE: begin
        // Preserve the original behavior: a start that arrives while dma_busy
        // is ignored rather than queued.  Otherwise, latch cfg and pipeline
        // the count generation before issuing the DMA command.
        if (start && !dma_busy) begin
          state_d = ST_CALC1;
        end
      end

      ST_CALC1: begin
        state_d = ST_CALC2;
      end

      ST_CALC2: begin
        state_d = ST_CALC3;
      end

      ST_CALC3: begin
        state_d = ST_CHECK;
      end

      ST_CHECK: begin
        if (!cfg_valid_q || !count_fits_q) begin
          state_d = ST_ERROR;
        end
        else begin
          state_d = ST_ISSUE;
        end
      end

      ST_ISSUE: begin
        if (dma_error) begin
          state_d = ST_ERROR;
        end
        else if (!dma_busy) begin
          wgt_cmd_start = 1'b1;
          state_d       = ST_WAIT_DMA;
        end
      end

      ST_WAIT_DMA: begin
        if (dma_error) begin
          state_d = ST_ERROR;
        end
        else if (dma_done_wgt) begin
          state_d = ST_DONE;
        end
      end

      ST_DONE: begin
        done    = 1'b1;
        state_d = ST_IDLE;
      end

      ST_ERROR: begin
        error   = 1'b1;
        state_d = ST_IDLE;
      end

      default: begin
        state_d = ST_IDLE;
      end
    endcase
  end

  assign busy = (state_q != ST_IDLE);

  // --------------------------------------------------
  // Sequential state / calculation pipeline
  // --------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state_q <= ST_IDLE;

      mode_q     <= 1'b0;
      buf_sel_q  <= 1'b0;
      ddr_base_q <= '0;
      k_q        <= '0;
      c_q        <= '0;
      f_q        <= '0;
      pv_m1_q    <= '0;
      pf_m1_q    <= '0;
      pf_m2_q    <= '0;

      num_fgroup_m1_q <= 32'd0;
      num_fgroup_m2_q <= 32'd0;
      num_cgroup_m2_q <= 32'd0;
      k_sq_q          <= 32'd0;
      cfg_valid_q     <= 1'b0;

      logical_bundles_m1_q <= 32'd0;
      physical_words_m2_q  <= 32'd0;

      physical_words_m1_q <= 32'd0;
      calc_num_words_q    <= 32'd0;
      count_fits_q        <= 1'b0;
      num_words_q         <= '0;
    end
    else begin
      state_q <= state_d;

      // Capture request configuration only when the original implementation
      // would have accepted a start for command generation, i.e. start in IDLE
      // while DMA is not busy.
      if ((state_q == ST_IDLE) && start && !dma_busy) begin
        mode_q     <= cfg_mode;
        buf_sel_q  <= cfg_buf_sel;
        ddr_base_q <= cfg_wgt_ddr_base;
        k_q        <= cfg_k_cur;
        c_q        <= cfg_c_cur;
        f_q        <= cfg_f_cur;
        pv_m1_q    <= cfg_pv_mode1_cur;
        pf_m1_q    <= cfg_pf_mode1_cur;
        pf_m2_q    <= cfg_pf_mode2;
      end

      if (state_q == ST_CALC1) begin
        num_fgroup_m1_q <= num_fgroup_m1_calc;
        num_fgroup_m2_q <= num_fgroup_m2_calc;
        num_cgroup_m2_q <= num_cgroup_m2_calc;
        k_sq_q          <= k_sq_calc;
        cfg_valid_q     <= cfg_valid_calc;
      end

      if (state_q == ST_CALC2) begin
        logical_bundles_m1_q <= logical_bundles_m1_calc;
        physical_words_m2_q  <= physical_words_m2_calc;
      end

      if (state_q == ST_CALC3) begin
        physical_words_m1_q <= physical_words_m1_calc;
        calc_num_words_q    <= calc_num_words_calc;
        count_fits_q        <= count_fits_calc;
        num_words_q         <= calc_num_words_calc[CMD_W-1:0];
      end
    end
  end

endmodule
