module weight_read_ctrl_mode2 #(
  parameter int DATA_W    = 8,
  parameter int PC        = 8,
  parameter int PF        = 4,
  // Physical word width from weight_buffer. Must match weight_buffer.WORD_LANES.
  // In mode 2, this controller consumes the low PF*PC lanes from that physical word.
  parameter int WB_LANES  = 32,
  parameter int WB_ADDR_W = 12
)(
  input  logic clk,
  input  logic rst_n,

  // =====================================================
  // Runtime config
  // Layout assumption in weight_buffer for mode 2:
  // each physical word stores one (f_group, c_group, ky, kx) PFxPC block
  // in the low PF*PC lanes.
  // lane = pf*PC + pc
  // =====================================================
  input  logic [3:0] K_cur,
  input  logic [7:0] C_cur,
  input  logic [7:0] F_cur,
  input  logic [15:0] Hout_cur,
  input  logic [15:0] Wout_cur,

  // Current controller position / pulses
  // IMPORTANT CONTRACT AFTER THE Mode2 timing fix:
  //   mac_en must be the consume-qualified MAC fire from ce_mode2_top
  //   (i.e. the controller only asserts it when current IFM and current
  //   weight bundles are already valid in their registers).  This matches
  //   Mode1's consume_en/ctrl_step_en contract.
  input  logic        start,
  input  logic        pass_start_pulse,
  input  logic        mac_en,
  input  logic        out_valid,
  input  logic [15:0] f_group,
  input  logic [15:0] out_row,
  input  logic [15:0] out_col,

  // Active bank selection from system control
  input  logic        wb_bank_sel,
  input  logic        wb_bank_ready,

  // Read side to weight_buffer
  output logic        wb_rd_en,
  output logic        wb_rd_buf_sel,
  output logic [WB_ADDR_W-1:0] wb_rd_addr,
  input  logic [WB_LANES*DATA_W-1:0] wb_rd_data,
  input  logic        wb_rd_valid,

  // To weight_register_mode2
  output logic        weight_write_en,
  output logic [PF*PC*DATA_W-1:0] weight_write_data
);

  logic [15:0] num_fgroup, num_cgroup;
  logic        last_issue;
  logic        last_col, last_row, last_fgroup;

  logic [15:0] issue_fgroup_r, issue_cgroup_r;
  logic [7:0]  issue_ky_r, issue_kx_r;
  logic [15:0] succ_fgroup, succ_cgroup;
  logic [7:0]  succ_ky, succ_kx;
  logic [15:0] next_sweep_fgroup;

  logic issue_first;
  logic issue_succ;
  logic req_inflight_q;
  logic [31:0] flat_addr32;

  // Metadata for the weight-buffer read request that is expected to return
  // with wb_rd_valid.  The weight-buffer response is registered, so these
  // registers align wb_rd_data with the f/c group that generated the request.
  logic [15:0] req_fgroup_s, req_cgroup_s;
  logic [15:0] resp_fgroup_q, resp_cgroup_q;
  logic [PF*PC*DATA_W-1:0] weight_write_data_masked;

  // pass_start_pulse is intentionally not used as an issue trigger anymore.
  // It used to prefetch the successor before the current registered weight
  // bundle was consumed.  That allowed mac_en to see stale weight_out_raw.
  // Keep a harmless reference so lint does not complain about an unused input.
  logic unused_pass_start_pulse;
  assign unused_pass_start_pulse = pass_start_pulse;

  always_comb begin
    if (PF != 0)
      num_fgroup = (F_cur + PF - 1) / PF;
    else
      num_fgroup = 16'd0;

    if (PC != 0)
      num_cgroup = (C_cur + PC - 1) / PC;
    else
      num_cgroup = 16'd0;

    last_issue  = (num_cgroup == 0) ? 1'b1 :
                  ((issue_cgroup_r == (num_cgroup - 1)) &&
                   (issue_ky_r     == (K_cur - 1)) &&
                   (issue_kx_r     == (K_cur - 1)));

    last_col    = (Wout_cur == 0) ? 1'b1 : (out_col == (Wout_cur - 1));
    last_row    = (Hout_cur == 0) ? 1'b1 : (out_row == (Hout_cur - 1));
    last_fgroup = (num_fgroup == 0) ? 1'b1 : (f_group == (num_fgroup - 1));

    // Successor of the last issued/consumed weight tuple in the order:
    //   kx -> ky -> c_group.
    succ_fgroup = issue_fgroup_r;
    succ_cgroup = issue_cgroup_r;
    succ_ky     = issue_ky_r;
    succ_kx     = issue_kx_r;

    if (issue_kx_r != (K_cur - 1)) begin
      succ_kx = issue_kx_r + 1'b1;
    end
    else begin
      succ_kx = '0;
      if (issue_ky_r != (K_cur - 1)) begin
        succ_ky = issue_ky_r + 1'b1;
      end
      else begin
        succ_ky = '0;
        if (num_cgroup == 0) begin
          succ_cgroup = 16'd0;
        end
        else begin
          succ_cgroup = issue_cgroup_r + 1'b1;
        end
      end
    end

    // First tuple for the next output block.  Keep the old sweep-level f_group
    // behavior so this controller remains compatible with the existing Mode2
    // controller loop ordering.
    if (!last_col) begin
      next_sweep_fgroup = f_group;
    end
    else if (!last_row) begin
      next_sweep_fgroup = f_group;
    end
    else if (!last_fgroup) begin
      next_sweep_fgroup = f_group + 1'b1;
    end
    else begin
      next_sweep_fgroup = 16'd0;
    end

    // Issue policy after the fix:
    //   - issue_first requests the first weight tuple of a layer/new output block;
    //   - issue_succ requests the next tuple only after a real MAC consume.
    // No request is issued while a request is already in flight.
    issue_first = wb_bank_ready && !req_inflight_q && (start || out_valid);
    issue_succ  = wb_bank_ready && !req_inflight_q && mac_en && !last_issue;

    wb_rd_en      = 1'b0;
    wb_rd_buf_sel = wb_bank_sel;
    wb_rd_addr    = '0;
    flat_addr32   = 32'd0;
    req_fgroup_s  = 16'd0;
    req_cgroup_s  = 16'd0;

    if (issue_first) begin
      req_fgroup_s = start ? 16'd0 : next_sweep_fgroup;
      req_cgroup_s = 16'd0;
      flat_addr32  = ((((req_fgroup_s * num_cgroup) + req_cgroup_s) * K_cur) + 16'd0) * K_cur + 16'd0;
      if (start) begin
        flat_addr32 = 32'd0;
      end
      wb_rd_en   = 1'b1;
      wb_rd_addr = flat_addr32[WB_ADDR_W-1:0];
    end
    else if (issue_succ) begin
      req_fgroup_s = succ_fgroup;
      req_cgroup_s = succ_cgroup;
      flat_addr32  = ((((req_fgroup_s * num_cgroup) + req_cgroup_s) * K_cur) + succ_ky) * K_cur + succ_kx;
      wb_rd_en   = 1'b1;
      wb_rd_addr = flat_addr32[WB_ADDR_W-1:0];
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      issue_fgroup_r <= 16'd0;
      issue_cgroup_r <= 16'd0;
      issue_ky_r     <= '0;
      issue_kx_r     <= '0;
      resp_fgroup_q  <= 16'd0;
      resp_cgroup_q  <= 16'd0;
      req_inflight_q <= 1'b0;
    end
    else begin
      // Track one outstanding weight-buffer read.  If a read returns and a new
      // read is issued in the same cycle, keep in-flight asserted for the new
      // request.
      if (wb_rd_valid) begin
        req_inflight_q <= 1'b0;
      end
      if (wb_rd_en) begin
        req_inflight_q <= 1'b1;
      end

      if (wb_rd_en) begin
        // Capture the request metadata for the corresponding wb_rd_valid data.
        resp_fgroup_q <= req_fgroup_s;
        resp_cgroup_q <= req_cgroup_s;

        if (issue_first) begin
          issue_fgroup_r <= req_fgroup_s;
          issue_cgroup_r <= 16'd0;
          issue_ky_r     <= '0;
          issue_kx_r     <= '0;
        end
        else begin
          issue_fgroup_r <= succ_fgroup;
          issue_cgroup_r <= succ_cgroup;
          issue_ky_r     <= succ_ky;
          issue_kx_r     <= succ_kx;
        end
      end
    end
  end

  always_comb begin
    weight_write_data_masked = '0;

    for (int pf_i = 0; pf_i < PF; pf_i++) begin
      for (int pc_i = 0; pc_i < PC; pc_i++) begin
        if (((resp_fgroup_q * PF + pf_i) < F_cur) &&
            ((resp_cgroup_q * PC + pc_i) < C_cur)) begin
          weight_write_data_masked[(pf_i*PC + pc_i)*DATA_W +: DATA_W] =
              wb_rd_data[(pf_i*PC + pc_i)*DATA_W +: DATA_W];
        end
      end
    end
  end

  assign weight_write_en   = wb_rd_valid;
  assign weight_write_data = weight_write_data_masked;

endmodule
