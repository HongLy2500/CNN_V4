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
  logic [31:0] flat_addr32;

  always_comb begin
    if (PF != 0)
      num_fgroup = (F_cur + PF - 1) / PF;
    else
      num_fgroup = 16'd0;

    if (PC != 0)
      num_cgroup = (C_cur + PC - 1) / PC;
    else
      num_cgroup = 16'd0;

    last_issue  = (issue_cgroup_r == (num_cgroup - 1)) &&
                  (issue_ky_r     == (K_cur - 1)) &&
                  (issue_kx_r     == (K_cur - 1));

    last_col    = (Wout_cur == 0) ? 1'b1 : (out_col == (Wout_cur - 1));
    last_row    = (Hout_cur == 0) ? 1'b1 : (out_row == (Hout_cur - 1));
    last_fgroup = (num_fgroup == 0) ? 1'b1 : (f_group == (num_fgroup - 1));

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
        succ_ky     = '0;
        succ_cgroup = issue_cgroup_r + 1'b1;
      end
    end

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

    issue_first = wb_bank_ready && (start || out_valid);
    issue_succ  = wb_bank_ready && (pass_start_pulse || mac_en) && !last_issue;

    wb_rd_en      = 1'b0;
    wb_rd_buf_sel = wb_bank_sel;
    wb_rd_addr    = '0;
    flat_addr32   = 32'd0;

    if (issue_first) begin
      flat_addr32 = ((((next_sweep_fgroup * num_cgroup) + 16'd0) * K_cur) + 16'd0) * K_cur + 16'd0;
      if (start)
        flat_addr32 = 32'd0;
      wb_rd_en   = 1'b1;
      wb_rd_addr = flat_addr32[WB_ADDR_W-1:0];
    end
    else if (issue_succ) begin
      flat_addr32 = ((((succ_fgroup * num_cgroup) + succ_cgroup) * K_cur) + succ_ky) * K_cur + succ_kx;
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
    end
    else if (wb_rd_en) begin
      if (issue_first) begin
        if (start)
          issue_fgroup_r <= 16'd0;
        else
          issue_fgroup_r <= next_sweep_fgroup;
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

  assign weight_write_en   = wb_rd_valid;
  assign weight_write_data = wb_rd_data[PF*PC*DATA_W-1:0];

endmodule
