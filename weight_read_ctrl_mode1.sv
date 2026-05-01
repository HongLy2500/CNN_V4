module weight_read_ctrl_mode1 #(
  parameter int DATA_W    = 8,
  parameter int PF_MAX    = 8,
  // Physical width of one weight_buffer word. In the integrated design this
  // should be PTOTAL. One physical word in mode 1 packs Pv_cur logical bundles,
  // each logical bundle being Pf_cur weights wide.
  parameter int PTOTAL    = 16,
  parameter int WB_ADDR_W = 12
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
  input  logic [7:0] C_cur,
  input  logic [7:0] F_cur,
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

  localparam int BASE_W = (PTOTAL > 1) ? $clog2(PTOTAL) : 1;

  // --------------------------------------------------------------------------
  // Local registered config / derived config
  // --------------------------------------------------------------------------
  // These registers reduce fanout from layer_cfg_manager.cur_cfg_q and prevent
  // cur_cfg bits from feeding the weight-buffer read-address register through a
  // very long combinational path.
  logic [3:0]  K_q;
  logic [7:0]  C_q;
  logic [7:0]  F_q;
  logic [7:0]  Pv_q;
  logic [7:0]  Pf_q;
  logic [15:0] Wout_q;
  logic [15:0] num_fgroup_q;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      K_q          <= '0;
      C_q          <= '0;
      F_q          <= '0;
      Pv_q         <= '0;
      Pf_q         <= '0;
      Wout_q       <= '0;
      num_fgroup_q <= '0;
    end
    else begin
      K_q    <= K_cur;
      C_q    <= C_cur;
      F_q    <= F_cur;
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
  // Current / requested bundle bookkeeping
  // --------------------------------------------------------------------------
  // Requested bundle currently assigned to the weight-buffer request in flight.
  logic [15:0] req_fgroup_r, req_c_r;
  logic [7:0]  req_ky_r, req_kx_r;

  // Current bundle loaded in weight_register and waiting to be consumed.
  logic [15:0] cur_fgroup_r, cur_c_r;
  logic [7:0]  cur_ky_r, cur_kx_r;

  logic        req_inflight_q;
  logic        bundle_valid_q;

  // --------------------------------------------------------------------------
  // Issue decision logic
  // --------------------------------------------------------------------------
  logic        cur_last_issue;
  logic [15:0] succ_fgroup, succ_c;
  logic [7:0]  succ_ky, succ_kx;
  logic [15:0] next_sweep_fgroup;
  logic        last_col;
  logic        last_fgroup;
  logic        pipe_busy;
  logic        issue_first;
  logic        issue_succ;
  logic        issue_new;

  // Selected logical bundle coordinate to push into the address pipeline.
  logic [15:0] issue_fgroup;
  logic [15:0] issue_c;
  logic [7:0]  issue_ky;
  logic [7:0]  issue_kx;

  // Address-generation pipeline valid bits.
  logic s0_valid_q;
  logic s1_valid_q;
  logic s2_valid_q;
  logic s3_valid_q;
  logic s4_valid_q;
  logic cmd_valid_q;

  assign pipe_busy = s0_valid_q | s1_valid_q | s2_valid_q |
                     s3_valid_q | s4_valid_q | cmd_valid_q;

  always_comb begin
    cur_last_issue = (cur_c_r  == (C_q - 1'b1)) &&
                     (cur_ky_r == (K_q - 1'b1)) &&
                     (cur_kx_r == (K_q - 1'b1));

    last_col    = (Wout_q == 0) ? 1'b1 : ((out_col + Pv_q) >= Wout_q);
    last_fgroup = (num_fgroup_q == 0) ? 1'b1 : (f_group == (num_fgroup_q - 1'b1));

    succ_fgroup = cur_fgroup_r;
    succ_c      = cur_c_r;
    succ_ky     = cur_ky_r;
    succ_kx     = cur_kx_r;

    if (cur_kx_r != (K_q - 1'b1)) begin
      succ_kx = cur_kx_r + 1'b1;
    end
    else begin
      succ_kx = '0;
      if (cur_ky_r != (K_q - 1'b1)) begin
        succ_ky = cur_ky_r + 1'b1;
      end
      else begin
        succ_ky = '0;
        succ_c  = cur_c_r + 1'b1;
      end
    end

    if (!last_col) begin
      next_sweep_fgroup = f_group;
    end
    else if (!last_fgroup) begin
      next_sweep_fgroup = f_group + 1'b1;
    end
    else begin
      next_sweep_fgroup = 16'd0;
    end

    // Only one request may be in the local address pipeline or in flight to the
    // weight buffer. This keeps the original request/consume ordering but moves
    // the expensive address calculation away from the weight_buffer input path.
    //
    // `start` is a new layer/sweep boundary. It must be able to issue the first
    // request even if bundle_valid_q or the local address pipeline is still high
    // from the previous layer before the sequential start-flush takes effect.
    // For out_valid-driven sweeps, keep the original idle-pipeline/no-bundle
    // requirements.
    issue_first = wb_bank_ready && !req_inflight_q &&
                  (start || (!pipe_busy && !bundle_valid_q && out_valid));

    issue_succ  = wb_bank_ready && !pipe_busy && !req_inflight_q && bundle_valid_q &&
                  consume_en && !cur_last_issue;

    issue_new = issue_first || issue_succ;

    issue_fgroup = 16'd0;
    issue_c      = 16'd0;
    issue_ky     = '0;
    issue_kx     = '0;

    if (issue_first) begin
      issue_fgroup = start ? 16'd0 : next_sweep_fgroup;
      issue_c      = 16'd0;
      issue_ky     = '0;
      issue_kx     = '0;
    end
    else if (issue_succ) begin
      issue_fgroup = succ_fgroup;
      issue_c      = succ_c;
      issue_ky     = succ_ky;
      issue_kx     = succ_kx;
    end
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
  logic [7:0]  s0_K_q, s0_C_q, s0_Pv_q, s0_Pf_q;
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

  // Metadata aligned with cmd_valid_q.
  logic [15:0] cmd_fgroup_q, cmd_c_q;
  logic [7:0]  cmd_ky_q, cmd_kx_q;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s0_valid_q      <= 1'b0;
      s1_valid_q      <= 1'b0;
      s2_valid_q      <= 1'b0;
      s3_valid_q      <= 1'b0;
      s4_valid_q      <= 1'b0;
      cmd_valid_q     <= 1'b0;

      s0_fgroup_q     <= '0;
      s0_c_q          <= '0;
      s0_ky_q         <= '0;
      s0_kx_q         <= '0;
      s0_K_q          <= '0;
      s0_C_q          <= '0;
      s0_Pv_q         <= '0;
      s0_Pf_q         <= '0;
      s0_buf_sel_q    <= 1'b0;

      s1_fc_q         <= '0;
      s1_ky_q         <= '0;
      s1_kx_q         <= '0;
      s1_K_q          <= '0;
      s1_Pv_q         <= '0;
      s1_Pf_q         <= '0;
      s1_fgroup_q     <= '0;
      s1_c_q          <= '0;
      s1_buf_sel_q    <= 1'b0;

      s2_fck_q        <= '0;
      s2_kx_q         <= '0;
      s2_K_q          <= '0;
      s2_Pv_q         <= '0;
      s2_Pf_q         <= '0;
      s2_fgroup_q     <= '0;
      s2_c_q          <= '0;
      s2_ky_q         <= '0;
      s2_buf_sel_q    <= 1'b0;

      s3_logical_idx_q <= '0;
      s3_Pv_q          <= '0;
      s3_Pf_q          <= '0;
      s3_fgroup_q      <= '0;
      s3_c_q           <= '0;
      s3_ky_q          <= '0;
      s3_kx_q          <= '0;
      s3_buf_sel_q     <= 1'b0;

      s4_phys_addr_q  <= '0;
      s4_base_lane_q  <= '0;
      s4_fgroup_q     <= '0;
      s4_c_q          <= '0;
      s4_ky_q         <= '0;
      s4_kx_q         <= '0;
      s4_buf_sel_q    <= 1'b0;

      cmd_addr_q      <= '0;
      cmd_base_lane_q <= '0;
      cmd_buf_sel_q   <= 1'b0;
      cmd_fgroup_q    <= '0;
      cmd_c_q         <= '0;
      cmd_ky_q        <= '0;
      cmd_kx_q        <= '0;

      req_fgroup_r    <= 16'd0;
      req_c_r         <= 16'd0;
      req_ky_r        <= '0;
      req_kx_r        <= '0;
      cur_fgroup_r    <= 16'd0;
      cur_c_r         <= 16'd0;
      cur_ky_r        <= '0;
      cur_kx_r        <= '0;
      req_inflight_q  <= 1'b0;
      bundle_valid_q  <= 1'b0;
    end
    else begin
      // Default pipeline advance.
      s1_valid_q  <= s0_valid_q;
      s2_valid_q  <= s1_valid_q;
      s3_valid_q  <= s2_valid_q;
      s4_valid_q  <= s3_valid_q;
      cmd_valid_q <= s4_valid_q;
      s0_valid_q  <= 1'b0;

      // Flush active-bundle state and outstanding bookkeeping on layer start.
      // A new first request may still be loaded into stage 0 below.
      if (start) begin
        req_inflight_q <= 1'b0;
        bundle_valid_q <= 1'b0;

        s1_valid_q     <= 1'b0;
        s2_valid_q     <= 1'b0;
        s3_valid_q     <= 1'b0;
        s4_valid_q     <= 1'b0;
        cmd_valid_q    <= 1'b0;
      end

      // Stage 0 load.
      if (issue_new) begin
        s0_valid_q   <= 1'b1;
        s0_fgroup_q  <= issue_fgroup;
        s0_c_q       <= issue_c;
        s0_ky_q      <= issue_ky;
        s0_kx_q      <= issue_kx;
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
      cmd_fgroup_q    <= s4_fgroup_q;
      cmd_c_q         <= s4_c_q;
      cmd_ky_q        <= s4_ky_q;
      cmd_kx_q        <= s4_kx_q;

      // Mark request in-flight when the command is launched from stage 4 into
      // the registered output. The weight buffer captures cmd_valid_q one cycle
      // later, and wb_rd_valid eventually returns for the same metadata.
      if (s4_valid_q) begin
        req_inflight_q <= 1'b1;
        req_fgroup_r   <= s4_fgroup_q;
        req_c_r        <= s4_c_q;
        req_ky_r       <= s4_ky_q;
        req_kx_r       <= s4_kx_q;
      end

      if (wb_rd_valid) begin
        req_inflight_q <= 1'b0;
        bundle_valid_q <= 1'b1;
        cur_fgroup_r   <= req_fgroup_r;
        cur_c_r        <= req_c_r;
        cur_ky_r       <= req_ky_r;
        cur_kx_r       <= req_kx_r;
      end
      else if (consume_en && bundle_valid_q) begin
        bundle_valid_q <= 1'b0;
      end
    end
  end

  assign wb_rd_en        = cmd_valid_q;
  assign wb_rd_buf_sel   = cmd_buf_sel_q;
  assign wb_rd_addr      = cmd_addr_q;
  assign wb_rd_base_lane = cmd_base_lane_q;

  assign weight_load_en = wb_rd_valid;
  assign weight_clear   = start;

  integer i;
  always_comb begin
    for (i = 0; i < PF_MAX; i++) begin
      if (i < Pf_q)
        weight_in_logic[i] = wb_rd_data[i*DATA_W +: DATA_W];
      else
        weight_in_logic[i] = '0;
    end
  end

  // pass_start_pulse is intentionally unused in this module revision. The
  // actual consume event is already qualified by consume_en from ce_mode1_top.
  // Keep a dummy reference to avoid lint-only unused-input noise in some flows.
  logic unused_pass_start_pulse;
  assign unused_pass_start_pulse = pass_start_pulse;

endmodule
