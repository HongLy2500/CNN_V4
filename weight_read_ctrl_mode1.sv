module weight_read_ctrl_mode1 #(
  parameter int DATA_W    = 8,
  parameter int PF_MAX    = 16,
  // Physical width of one weight_buffer word. In the integrated design this
  // should be PTOTAL. One physical word in mode 1 packs Pv_cur logical bundles,
  // each logical bundle being Pf_cur weights wide.
  parameter int PTOTAL    = 256,
  parameter int WB_ADDR_W = 12,
  // Number of prefetched Pf-bundles kept locally. This is intentionally small:
  // it only needs to cover the address-pipeline + weight-buffer latency, not a
  // whole layer. Keep it power-of-two for simple pointer wrap.
  parameter int PREFETCH_DEPTH = 16
)(
  input  logic clk,
  input  logic rst_n,

  // =====================================================
  // Runtime config
  // Mode-1 packing assumption in weight_buffer:
  //   physical word width = PTOTAL lanes
  //   each logical mode-1 bundle = Pf_cur lanes
  //   therefore one physical word packs Pv_cur logical bundles
  //   because Pv_cur * Pf_cur = PTOTAL
  // Logical bundle order is the flat order of (f_group, c, ky, kx).
  // =====================================================
  input  logic [3:0] K_cur,
  input  logic [9:0] C_cur,
  input  logic [9:0] F_cur,
  input  logic [7:0] Pv_cur,
  input  logic [7:0] Pf_cur,
  input  logic [15:0] Wout_cur,

  // Current controller position / pulses
  input  logic        start,
  input  logic        pass_start_pulse,
  // Qualified MAC consume pulse from ce_mode1_top.
  // This must be asserted only when the MAC really consumes the current
  // data/weight bundle, not merely when the controller is in S_RUN.
  input  logic        consume_en,
  input  logic        out_valid,
  input  logic [15:0] f_group,
  input  logic [15:0] out_col,

  // Active bank selection from system control
  input  logic        wb_bank_sel,
  input  logic        wb_bank_ready,

  // Read side to weight_buffer mode-1 logical port
  output logic        wb_rd_en,
  output logic        wb_rd_buf_sel,
  output logic [WB_ADDR_W-1:0] wb_rd_addr,
  output logic [($clog2(PTOTAL) > 0 ? $clog2(PTOTAL) : 1)-1:0] wb_rd_base_lane,
  input  logic [PF_MAX*DATA_W-1:0] wb_rd_data,
  input  logic        wb_rd_valid,

  // To weight_register_mode1
  output logic        weight_load_en,
  output logic        weight_clear,
  output logic signed [DATA_W-1:0] weight_in_logic [0:PF_MAX-1]
);

  localparam int BASE_W       = (PTOTAL > 1) ? $clog2(PTOTAL) : 1;
  localparam int FIFO_PTR_W   = (PREFETCH_DEPTH > 1) ? $clog2(PREFETCH_DEPTH) : 1;
  localparam int FIFO_CNT_W   = (PREFETCH_DEPTH > 1) ? $clog2(PREFETCH_DEPTH + 1) : 1;
  localparam int PREFETCH_MAX = (PREFETCH_DEPTH > 4) ? (PREFETCH_DEPTH - 2) : 1;
  localparam int FIFO_DATA_W  = PF_MAX * DATA_W;
  localparam int OCC_W        = FIFO_CNT_W + 4;
  localparam logic [OCC_W-1:0] PREFETCH_MAX_C = PREFETCH_MAX;

  // --------------------------------------------------------------------------
  // Local registered config / derived config
  // --------------------------------------------------------------------------
  // These registers reduce fanout from layer_cfg_manager.cur_cfg_q and prevent
  // cur_cfg bits from feeding the weight-buffer read-address register through a
  // very long combinational path.
  logic [3:0]  K_q;
  logic [9:0]  C_q;
  logic [7:0]  Pv_q;
  logic [7:0]  Pf_q;
  logic [15:0] Wout_q;
  logic [15:0] num_fgroup_q;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      K_q          <= '0;
      C_q          <= '0;
      Pv_q         <= '0;
      Pf_q         <= '0;
      Wout_q       <= '0;
      num_fgroup_q <= '0;
    end
    else begin
      K_q    <= K_cur;
      C_q    <= C_cur;
      Pv_q   <= Pv_cur;
      Pf_q   <= Pf_cur;
      Wout_q <= Wout_cur;

      if (Pf_cur != 0)
        num_fgroup_q <= (F_cur + Pf_cur - 1'b1) / Pf_cur;
      else
        num_fgroup_q <= 16'd0;
    end
  end

  // --------------------------------------------------------------------------
  // Sweep / issue pointer
  // --------------------------------------------------------------------------
  // The old implementation issued the next bundle only after the current bundle
  // was consumed. That serialized this address pipeline into every MAC step.
  // This revision keeps a separate issue pointer so the address pipeline can run
  // ahead and fill a small Pf-bundle FIFO.
  logic        sweep_active_q;
  logic        issue_done_q;
  logic [15:0] issue_fgroup_q;
  logic [15:0] issue_c_q;
  logic [7:0]  issue_ky_q;
  logic [7:0]  issue_kx_q;

  logic        last_col;
  logic        last_fgroup;
  logic [15:0] next_sweep_fgroup;
  logic        issue_last_bundle;
  logic        issue_new;
  logic        cfg_nonzero;

  always_comb begin
    last_col    = (Wout_q == 0) ? 1'b1 : ((out_col + Pv_q) >= Wout_q);
    last_fgroup = (num_fgroup_q == 0) ? 1'b1 : (f_group == (num_fgroup_q - 1'b1));

    if (!last_col) begin
      next_sweep_fgroup = f_group;
    end
    else if (!last_fgroup) begin
      next_sweep_fgroup = f_group + 1'b1;
    end
    else begin
      next_sweep_fgroup = 16'd0;
    end

    cfg_nonzero = (K_q != 0) && (C_q != 0) && (Pv_q != 0) && (Pf_q != 0);

    issue_last_bundle = (issue_c_q  == (C_q - 1'b1)) &&
                        (issue_ky_q == (K_q - 1'b1)) &&
                        (issue_kx_q == (K_q - 1'b1));
  end

  // --------------------------------------------------------------------------
  // Pipelined mode-1 address calculation
  // --------------------------------------------------------------------------
  // Original formula preserved:
  //   logical_idx = ((((f_group * C) + c) * K + ky) * K + kx)
  //   phys_addr   = logical_idx / Pv
  //   subword     = logical_idx % Pv
  //   base_lane   = subword * Pf
  // The calculation is split across stages so the weight_buffer input registers
  // do not see a long path from cur_cfg_q/config through multiply/divide logic.

  // Stage 0: selected coordinates and config snapshot
  logic [15:0] s0_fgroup_q, s0_c_q;
  logic [7:0]  s0_ky_q, s0_kx_q;
  logic [7:0]  s0_K_q, s0_Pv_q, s0_Pf_q;
  logic [9:0]  s0_C_q;
  logic        s0_buf_sel_q;

  // Stage 1: f_group*C + c
  logic [31:0] s1_fc_q;
  logic [7:0]  s1_ky_q, s1_kx_q;
  logic [7:0]  s1_K_q, s1_Pv_q, s1_Pf_q;
  logic [15:0] s1_fgroup_q, s1_c_q;
  logic        s1_buf_sel_q;

  // Stage 2: (f_group*C+c)*K + ky
  logic [31:0] s2_fck_q;
  logic [7:0]  s2_kx_q;
  logic [7:0]  s2_K_q, s2_Pv_q, s2_Pf_q;
  logic [15:0] s2_fgroup_q, s2_c_q;
  logic [7:0]  s2_ky_q;
  logic        s2_buf_sel_q;

  // Stage 3: full logical index
  logic [31:0] s3_logical_idx_q;
  logic [7:0]  s3_Pv_q, s3_Pf_q;
  logic [15:0] s3_fgroup_q, s3_c_q;
  logic [7:0]  s3_ky_q, s3_kx_q;
  logic        s3_buf_sel_q;

  // Stage 4: physical address and base lane
  logic [31:0] s4_phys_addr_q;
  logic [31:0] s4_base_lane_q;
  logic [15:0] s4_fgroup_q, s4_c_q;
  logic [7:0]  s4_ky_q, s4_kx_q;
  logic        s4_buf_sel_q;

  // Registered command output to weight_buffer.
  logic [WB_ADDR_W-1:0] cmd_addr_q;
  logic [BASE_W-1:0]    cmd_base_lane_q;
  logic                 cmd_buf_sel_q;

  // Address-generation pipeline valid bits.
  logic s0_valid_q;
  logic s1_valid_q;
  logic s2_valid_q;
  logic s3_valid_q;
  logic s4_valid_q;
  logic cmd_valid_q;

  logic [OCC_W-1:0]      pipe_count;
  logic [FIFO_CNT_W-1:0] outstanding_count_q;
  logic [FIFO_CNT_W-1:0] data_count_q;
  logic [OCC_W-1:0]      prefetch_occupancy;
  logic                  prefetch_room;

  always_comb begin
    pipe_count = '0;
    pipe_count = pipe_count + s0_valid_q;
    pipe_count = pipe_count + s1_valid_q;
    pipe_count = pipe_count + s2_valid_q;
    pipe_count = pipe_count + s3_valid_q;
    pipe_count = pipe_count + s4_valid_q;
    pipe_count = pipe_count + cmd_valid_q;

    prefetch_occupancy = data_count_q + outstanding_count_q + pipe_count;
    prefetch_room      = (prefetch_occupancy < PREFETCH_MAX_C);

    issue_new = sweep_active_q && !issue_done_q && cfg_nonzero &&
                wb_bank_ready && prefetch_room && !start && !out_valid;
  end

  // --------------------------------------------------------------------------
  // Prefetch FIFO for returned Pf-bundles
  // --------------------------------------------------------------------------
  // Keep FIFO payload storage out of the async-reset control block so Vivado
  // can infer it as RAM/LUTRAM instead of dissolving it into registers.
  (* ram_style = "distributed" *) logic [FIFO_DATA_W-1:0] data_fifo_q [0:PREFETCH_DEPTH-1];
  logic [FIFO_PTR_W-1:0]    data_wr_ptr_q;
  logic [FIFO_PTR_W-1:0]    data_rd_ptr_q;
  logic [PF_MAX*DATA_W-1:0] fifo_head_data;
  logic [PF_MAX*DATA_W-1:0] load_data_mux;
  logic                     fifo_has_data;
  logic                     slot_can_load;
  logic                     load_from_fifo;
  logic                     load_bypass;
  logic                     push_to_fifo;
  logic                     pop_from_fifo;
  logic                     cmd_fire;

  assign fifo_head_data = data_fifo_q[data_rd_ptr_q];
  assign fifo_has_data  = (data_count_q != '0);

  // Internal mirror of the active bundle in weight_register_mode1.  It is not a
  // second source of truth for CE execution; it only tells this prefetch block
  // whether the CE-side weight slot can accept a new bundle.
  logic bundle_valid_q;

  assign slot_can_load  = (!bundle_valid_q) || consume_en;
  assign load_from_fifo = slot_can_load && fifo_has_data;
  assign load_bypass    = slot_can_load && !fifo_has_data && wb_rd_valid && !start;
  assign push_to_fifo   = wb_rd_valid && !start && !load_bypass;
  assign pop_from_fifo  = load_from_fifo;
  assign load_data_mux  = load_from_fifo ? fifo_head_data : wb_rd_data;

  assign weight_load_en = load_from_fifo || load_bypass;
  assign weight_clear   = start;

  // Suppress any command on the same cycle a new layer starts.  The sequential
  // logic also flushes the local pipeline on start.
  assign cmd_fire        = cmd_valid_q && !start;
  assign wb_rd_en        = cmd_fire;
  assign wb_rd_buf_sel   = cmd_buf_sel_q;
  assign wb_rd_addr      = cmd_addr_q;
  assign wb_rd_base_lane = cmd_base_lane_q;

  integer i;
  always_comb begin
    for (i = 0; i < PF_MAX; i++) begin
      if (i < Pf_q)
        weight_in_logic[i] = load_data_mux[i*DATA_W +: DATA_W];
      else
        weight_in_logic[i] = '0;
    end
  end

  // Payload RAM write port.  This block intentionally has no reset;
  // data_count_q/data_{wr,rd}_ptr_q define FIFO validity.  Keeping the memory
  // in a clock-only process avoids Synth 8-4767/Synth 8-7137 and lets Vivado
  // map the FIFO payload as distributed RAM.
  always_ff @(posedge clk) begin
    if (push_to_fifo) begin
      data_fifo_q[data_wr_ptr_q] <= wb_rd_data;
    end
  end

  // --------------------------------------------------------------------------
  // Sequential control
  // --------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sweep_active_q     <= 1'b0;
      issue_done_q       <= 1'b0;
      issue_fgroup_q     <= 16'd0;
      issue_c_q          <= 16'd0;
      issue_ky_q         <= '0;
      issue_kx_q         <= '0;

      s0_valid_q         <= 1'b0;
      s1_valid_q         <= 1'b0;
      s2_valid_q         <= 1'b0;
      s3_valid_q         <= 1'b0;
      s4_valid_q         <= 1'b0;
      cmd_valid_q        <= 1'b0;

      s0_fgroup_q        <= '0;
      s0_c_q             <= '0;
      s0_ky_q            <= '0;
      s0_kx_q            <= '0;
      s0_K_q             <= '0;
      s0_C_q             <= '0;
      s0_Pv_q            <= '0;
      s0_Pf_q            <= '0;
      s0_buf_sel_q       <= 1'b0;

      s1_fc_q            <= '0;
      s1_ky_q            <= '0;
      s1_kx_q            <= '0;
      s1_K_q             <= '0;
      s1_Pv_q            <= '0;
      s1_Pf_q            <= '0;
      s1_fgroup_q        <= '0;
      s1_c_q             <= '0;
      s1_buf_sel_q       <= 1'b0;

      s2_fck_q           <= '0;
      s2_kx_q            <= '0;
      s2_K_q             <= '0;
      s2_Pv_q            <= '0;
      s2_Pf_q            <= '0;
      s2_fgroup_q        <= '0;
      s2_c_q             <= '0;
      s2_ky_q            <= '0;
      s2_buf_sel_q       <= 1'b0;

      s3_logical_idx_q   <= '0;
      s3_Pv_q            <= '0;
      s3_Pf_q            <= '0;
      s3_fgroup_q        <= '0;
      s3_c_q             <= '0;
      s3_ky_q            <= '0;
      s3_kx_q            <= '0;
      s3_buf_sel_q       <= 1'b0;

      s4_phys_addr_q     <= '0;
      s4_base_lane_q     <= '0;
      s4_fgroup_q        <= '0;
      s4_c_q             <= '0;
      s4_ky_q            <= '0;
      s4_kx_q            <= '0;
      s4_buf_sel_q       <= 1'b0;

      cmd_addr_q         <= '0;
      cmd_base_lane_q    <= '0;
      cmd_buf_sel_q      <= 1'b0;

      data_wr_ptr_q      <= '0;
      data_rd_ptr_q      <= '0;
      data_count_q       <= '0;
      outstanding_count_q <= '0;
      bundle_valid_q     <= 1'b0;
    end
    else begin
      // Default pipeline advance.  Unlike the old design, stage 0 can accept a
      // new request every cycle while the prefetch FIFO has space.
      s1_valid_q  <= s0_valid_q;
      s2_valid_q  <= s1_valid_q;
      s3_valid_q  <= s2_valid_q;
      s4_valid_q  <= s3_valid_q;
      cmd_valid_q <= s4_valid_q;
      s0_valid_q  <= issue_new;

      // Stage 0 load.
      if (issue_new) begin
        s0_fgroup_q  <= issue_fgroup_q;
        s0_c_q       <= issue_c_q;
        s0_ky_q      <= issue_ky_q;
        s0_kx_q      <= issue_kx_q;
        s0_K_q       <= {4'd0, K_q};
        s0_C_q       <= C_q;
        s0_Pv_q      <= Pv_q;
        s0_Pf_q      <= Pf_q;
        s0_buf_sel_q <= wb_bank_sel;
      end

      // Stage 1.
      s1_fc_q      <= (s0_fgroup_q * s0_C_q) + s0_c_q;
      s1_ky_q      <= s0_ky_q;
      s1_kx_q      <= s0_kx_q;
      s1_K_q       <= s0_K_q;
      s1_Pv_q      <= s0_Pv_q;
      s1_Pf_q      <= s0_Pf_q;
      s1_fgroup_q  <= s0_fgroup_q;
      s1_c_q       <= s0_c_q;
      s1_buf_sel_q <= s0_buf_sel_q;

      // Stage 2.
      s2_fck_q     <= (s1_fc_q * s1_K_q) + s1_ky_q;
      s2_kx_q      <= s1_kx_q;
      s2_K_q       <= s1_K_q;
      s2_Pv_q      <= s1_Pv_q;
      s2_Pf_q      <= s1_Pf_q;
      s2_fgroup_q  <= s1_fgroup_q;
      s2_c_q       <= s1_c_q;
      s2_ky_q      <= s1_ky_q;
      s2_buf_sel_q <= s1_buf_sel_q;

      // Stage 3.
      s3_logical_idx_q <= (s2_fck_q * s2_K_q) + s2_kx_q;
      s3_Pv_q          <= s2_Pv_q;
      s3_Pf_q          <= s2_Pf_q;
      s3_fgroup_q      <= s2_fgroup_q;
      s3_c_q           <= s2_c_q;
      s3_ky_q          <= s2_ky_q;
      s3_kx_q          <= s2_kx_q;
      s3_buf_sel_q     <= s2_buf_sel_q;

      // Stage 4. Preserve original dynamic division/modulo semantics.
      if (s3_Pv_q != 0) begin
        s4_phys_addr_q <= s3_logical_idx_q / s3_Pv_q;
        s4_base_lane_q <= (s3_logical_idx_q % s3_Pv_q) * s3_Pf_q;
      end
      else begin
        s4_phys_addr_q <= '0;
        s4_base_lane_q <= '0;
      end
      s4_fgroup_q  <= s3_fgroup_q;
      s4_c_q       <= s3_c_q;
      s4_ky_q      <= s3_ky_q;
      s4_kx_q      <= s3_kx_q;
      s4_buf_sel_q <= s3_buf_sel_q;

      // Registered output command.
      cmd_addr_q      <= s4_phys_addr_q[WB_ADDR_W-1:0];
      cmd_base_lane_q <= s4_base_lane_q[BASE_W-1:0];
      cmd_buf_sel_q   <= s4_buf_sel_q;

      // Command/read outstanding counter.  This is used only for back-pressure
      // into the prefetch pipeline, so in-order metadata is not required.
      if (cmd_fire && !wb_rd_valid)
        outstanding_count_q <= outstanding_count_q + 1'b1;
      else if (!cmd_fire && wb_rd_valid && (outstanding_count_q != '0))
        outstanding_count_q <= outstanding_count_q - 1'b1;

      // Returned-data FIFO update.  The bypass path lets a returning bundle load
      // the CE weight slot directly when the FIFO is empty.
      if (push_to_fifo) begin
        data_wr_ptr_q <= data_wr_ptr_q + 1'b1;
      end
      if (pop_from_fifo) begin
        data_rd_ptr_q <= data_rd_ptr_q + 1'b1;
      end

      if (push_to_fifo && !pop_from_fifo)
        data_count_q <= data_count_q + 1'b1;
      else if (!push_to_fifo && pop_from_fifo && (data_count_q != '0))
        data_count_q <= data_count_q - 1'b1;

      // Mirror CE-side active weight validity.
      if (weight_load_en)
        bundle_valid_q <= 1'b1;
      else if (consume_en && bundle_valid_q)
        bundle_valid_q <= 1'b0;

      // Advance issue coordinate after launching a request into stage 0.
      if (issue_new) begin
        if (issue_last_bundle) begin
          issue_done_q   <= 1'b1;
          sweep_active_q <= 1'b0;
        end
        else if (issue_kx_q != (K_q - 1'b1)) begin
          issue_kx_q <= issue_kx_q + 1'b1;
        end
        else begin
          issue_kx_q <= '0;
          if (issue_ky_q != (K_q - 1'b1)) begin
            issue_ky_q <= issue_ky_q + 1'b1;
          end
          else begin
            issue_ky_q <= '0;
            issue_c_q  <= issue_c_q + 1'b1;
          end
        end
      end

      // New layer or new output-block sweep.  A new sweep starts issuing from
      // the next clock; this avoids mixing a flush and a new request in the same
      // pipeline cycle.
      if (start) begin
        sweep_active_q      <= 1'b1;
        issue_done_q        <= 1'b0;
        issue_fgroup_q      <= 16'd0;
        issue_c_q           <= 16'd0;
        issue_ky_q          <= '0;
        issue_kx_q          <= '0;

        s0_valid_q          <= 1'b0;
        s1_valid_q          <= 1'b0;
        s2_valid_q          <= 1'b0;
        s3_valid_q          <= 1'b0;
        s4_valid_q          <= 1'b0;
        cmd_valid_q         <= 1'b0;

        data_wr_ptr_q       <= '0;
        data_rd_ptr_q       <= '0;
        data_count_q        <= '0;
        outstanding_count_q <= '0;
        bundle_valid_q      <= 1'b0;
      end
      else if (out_valid) begin
        sweep_active_q <= 1'b1;
        issue_done_q   <= 1'b0;
        issue_fgroup_q <= next_sweep_fgroup;
        issue_c_q      <= 16'd0;
        issue_ky_q     <= '0;
        issue_kx_q     <= '0;
      end
    end
  end

  // pass_start_pulse is intentionally unused in this module revision. The
  // actual consume event is already qualified by consume_en from ce_mode1_top.
  // Keep a dummy reference to avoid lint-only unused-input noise in some flows.
  logic unused_pass_start_pulse;
  assign unused_pass_start_pulse = pass_start_pulse;

endmodule
