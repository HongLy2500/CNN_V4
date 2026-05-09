module ce_mode2_top #(
  parameter int DATA_W    = 8,
  parameter int PSUM_W    = 32,
  parameter int K_MAX     = 7,
  parameter int PC        = 8,
  parameter int PF        = 4,
  parameter int HOUT_MAX  = 224,
  parameter int WOUT_MAX  = 224,
  // Physical word width shared with weight_buffer.WORD_LANES.
  // Mode 2 consumes the low PF*PC logical lanes from this physical word.
  parameter int WB_LANES  = 32,
  parameter int WB_ADDR_W = 12
)(
  input  logic clk,
  input  logic rst_n,

  input  logic start,
  input  logic step_en,

  input  logic [3:0] K_cur,
  input  logic [7:0] C_cur,
  input  logic [7:0] F_cur,
  input  logic [15:0] Hout_cur,
  input  logic [15:0] Wout_cur,

  input  logic                     dr_write_en,
  input  logic [$clog2(K_MAX)-1:0] dr_write_row_idx,
  input  logic [PC*DATA_W-1:0]     dr_write_data,

  input  logic                     weight_bank_sel,
  input  logic                     weight_bank_ready,
  output logic                     wb_rd_en,
  output logic                     wb_rd_buf_sel,
  output logic [WB_ADDR_W-1:0]     wb_rd_addr,
  input  logic [WB_LANES*DATA_W-1:0] wb_rd_data,
  input  logic                     wb_rd_valid,

  output logic [15:0]              out_row,
  output logic [15:0]              out_col,
  output logic [15:0]              f_group,
  output logic [15:0]              c_group,
  output logic [$clog2(K_MAX)-1:0] ky,
  output logic [$clog2(K_MAX)-1:0] kx,

  output logic                     mac_en,
  output logic                     clear_psum,
  output logic                     out_valid,
  output logic                     pass_start_pulse,
  output logic                     group_start_pulse,
  output logic                     row_done_pulse,
  output logic [$clog2(K_MAX)-1:0] row_done_ky,
  output logic                     c_group_done_pulse,
  output logic                     pixel_done_pulse,
  output logic                     f_group_done_pulse,
  output logic                     done,
  output logic                     busy,

  output logic [PC*DATA_W-1:0]     data_out_logic,
  output logic [PF*PC*DATA_W-1:0]  weight_out,
  output logic [PF*PSUM_W-1:0]     mac_data_out,
  output logic                     mac_data_out_valid,
  output logic                     mac_group_start,
  output logic [15:0]              mac_f_base
);

  logic [15:0] f_base_cur;
  logic [15:0] c_base_cur;
  logic        weight_write_en;
  logic [PF*PC*DATA_W-1:0] weight_write_data;

  // Raw outputs from the internal registers. These may contain X/rubbish
  // on inactive lanes in partial C/F groups, so they are masked before MAC.
  logic [PC*DATA_W-1:0]       data_out_raw;
  logic [PF*PC*DATA_W-1:0]    weight_out_raw;

  // Block-start operand readiness for Mode 2.
  // The CE waits in S_CLEAR until the first IFM/weight tuple of the
  // current output block is loaded. After entering S_RUN, the existing
  // prefetch-ahead cadence driven by pass_start_pulse/mac_en is preserved.
  logic ifm_tuple_ready_q;
  logic wgt_tuple_ready_q;
  logic tuple_ready_s;

  always_comb begin
    f_base_cur = f_group * PF;
    c_base_cur = c_group * PC;
  end

  assign tuple_ready_s = ifm_tuple_ready_q && wgt_tuple_ready_q;

  ce_controller_mode2 #(
    .K_MAX    (K_MAX),
    .HOUT_MAX (HOUT_MAX),
    .WOUT_MAX (WOUT_MAX),
    .PC       (PC),
    .PF       (PF)
  ) u_ce_controller_mode2 (
    .clk               (clk),
    .rst_n             (rst_n),
    .start             (start),
    .step_en           (step_en),
    .tuple_ready       (tuple_ready_s),
    .K_cur             (K_cur),
    .C_cur             (C_cur),
    .F_cur             (F_cur),
    .Hout_cur          (Hout_cur),
    .Wout_cur          (Wout_cur),
    .out_row           (out_row),
    .out_col           (out_col),
    .f_group           (f_group),
    .c_group           (c_group),
    .ky                (ky),
    .kx                (kx),
    .mac_en            (mac_en),
    .clear_psum        (clear_psum),
    .out_valid         (out_valid),
    .pass_start_pulse  (pass_start_pulse),
    .group_start_pulse (group_start_pulse),
    .row_done_pulse    (row_done_pulse),
    .row_done_ky       (row_done_ky),
    .c_group_done_pulse(c_group_done_pulse),
    .pixel_done_pulse  (pixel_done_pulse),
    .f_group_done_pulse(f_group_done_pulse),
    .done              (done),
    .busy              (busy)
  );

  data_register_mode2 #(
    .K_MAX  (K_MAX),
    .DATA_W (DATA_W),
    .PC     (PC)
  ) u_data_register_mode2 (
    .clk          (clk),
    .rst_n        (rst_n),
    .K_cur        (K_cur),
    .C_cur        (C_cur),
    .c_group      (c_group),
    .write_en     (dr_write_en),
    .write_row_idx(dr_write_row_idx),
    .write_data   (dr_write_data),
    .read_row_idx (ky),
    .data_out     (data_out_raw)
  );

  weight_read_ctrl_mode2 #(
    .DATA_W   (DATA_W),
    .PC       (PC),
    .PF       (PF),
    .WB_LANES (WB_LANES),
    .WB_ADDR_W(WB_ADDR_W)
  ) u_weight_read_ctrl_mode2 (
    .clk             (clk),
    .rst_n           (rst_n),
    .K_cur           (K_cur),
    .C_cur           (C_cur),
    .F_cur           (F_cur),
    .Hout_cur        (Hout_cur),
    .Wout_cur        (Wout_cur),
    .start           (start),
    .pass_start_pulse(pass_start_pulse),
    .mac_en          (mac_en),
    .out_valid       (out_valid),
    .f_group         (f_group),
    .out_row         (out_row),
    .out_col         (out_col),
    .wb_bank_sel     (weight_bank_sel),
    .wb_bank_ready   (weight_bank_ready),
    .wb_rd_en        (wb_rd_en),
    .wb_rd_buf_sel   (wb_rd_buf_sel),
    .wb_rd_addr      (wb_rd_addr),
    .wb_rd_data      (wb_rd_data),
    .wb_rd_valid     (wb_rd_valid),
    .weight_write_en (weight_write_en),
    .weight_write_data(weight_write_data)
  );

  weight_register_mode2 #(
    .DATA_W (DATA_W),
    .PC     (PC),
    .PF     (PF)
  ) u_weight_register_mode2 (
    .clk       (clk),
    .rst_n     (rst_n),
    .write_en  (weight_write_en),
    .write_data(weight_write_data),
    .weight_out(weight_out_raw)
  );

  // ------------------------------------------------------------
  // Tuple-ready tracking for Mode 2.
  //
  // start/out_valid begin a new output block and invalidate the old
  // first-tuple readiness. The CE controller uses tuple_ready only to
  // release S_CLEAR. Do NOT clear readiness on every mac_en; otherwise
  // Mode 2 is converted into a per-MAC handshake and can stall on layers
  // with multiple C/F groups. Register writes have priority over clears
  // so same-cycle return at a block boundary is not lost.
  // ------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ifm_tuple_ready_q <= 1'b0;
      wgt_tuple_ready_q <= 1'b0;
    end
    else begin
      if (start || out_valid) begin
        ifm_tuple_ready_q <= 1'b0;
        wgt_tuple_ready_q <= 1'b0;
      end

      // Writes have priority over the block-boundary clears above.
      if (dr_write_en) begin
        ifm_tuple_ready_q <= 1'b1;
      end

      if (weight_write_en) begin
        wgt_tuple_ready_q <= 1'b1;
      end
    end
  end

  // ------------------------------------------------------------
  // Final Mode-2 lane mask before the MAC array.
  //
  // Mode 2 reduces across all PC lanes. For partial C groups
  // (for example C_cur=3, PC=32), inactive channel lanes must be
  // zero, otherwise a single X on pc>=C_cur poisons the whole sum.
  // Also mask partial F groups so inactive filters cannot consume
  // stale/rubbish weights.
  // ------------------------------------------------------------
  always_comb begin
    data_out_logic = '0;
    weight_out     = '0;

    for (int pc_i = 0; pc_i < PC; pc_i++) begin
      if ((c_base_cur + pc_i) < C_cur) begin
        data_out_logic[pc_i*DATA_W +: DATA_W]
          = data_out_raw[pc_i*DATA_W +: DATA_W];
      end
    end

    for (int pf_i = 0; pf_i < PF; pf_i++) begin
      for (int pc_i = 0; pc_i < PC; pc_i++) begin
        if (((c_base_cur + pc_i) < C_cur) &&
            ((f_base_cur + pf_i) < F_cur)) begin
          weight_out[(pf_i*PC + pc_i)*DATA_W +: DATA_W]
            = weight_out_raw[(pf_i*PC + pc_i)*DATA_W +: DATA_W];
        end
      end
    end
  end

  mac_array_mode2 #(
    .DATA_W (DATA_W),
    .PSUM_W (PSUM_W),
    .PC     (PC),
    .PF     (PF)
  ) u_mac_array_mode2 (
    .clk           (clk),
    .rst_n         (rst_n),
    .mac_en        (mac_en),
    .out_valid     (out_valid),
    .data_in       (data_out_logic),
    .weight_in     (weight_out),
    .data_out      (mac_data_out),
    .data_out_valid(mac_data_out_valid)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      mac_group_start <= 1'b0;
      mac_f_base      <= '0;
    end
    else begin
      mac_group_start <= group_start_pulse;
      if (out_valid)
        mac_f_base <= f_base_cur;
    end
  end

endmodule
