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

  // Mode-2 resident-tile window. One start computes one horizontal tile.
  // out_col exported by the controller remains GLOBAL:
  //   out_col = tile_col_base_g + local_col
  input  logic [15:0] tile_col_base_g,
  input  logic [15:0] tile_col_count,

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

  // Consume-qualified operand readiness for Mode 2.
  // This mirrors Mode 1's ctrl_step_en contract:
  // the controller/MAC may advance only after both the current IFM tuple
  // and the current weight tuple are already valid in their registers.
  logic ifm_tuple_ready_q;
  logic wgt_tuple_ready_q;
  logic tuple_ready_s;
  logic ctrl_step_en;

  // One outstanding Mode-2 weight request at a time.  This prevents the
  // weight register from being overwritten by a prefetched bundle before
  // the current registered bundle is consumed by the MAC.
  logic weight_req_inflight_q;
  logic weight_bank_ready_m2_s;

  always_comb begin
    f_base_cur = f_group * PF;
    c_base_cur = c_group * PC;
  end

  assign tuple_ready_s = ifm_tuple_ready_q && wgt_tuple_ready_q;
  assign ctrl_step_en = step_en && tuple_ready_s;

  // Present the weight buffer as ready only when Mode 2 is allowed to
  // issue a request for the current/next consumed tuple.  In particular,
  // do not let pass_start_pulse prefetch over a still-unconsumed current
  // weight bundle.
  assign weight_bank_ready_m2_s = weight_bank_ready &&
                                  !weight_req_inflight_q &&
                                  (!wgt_tuple_ready_q || mac_en || start);

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
    .step_en           (ctrl_step_en),
    .tuple_ready       (tuple_ready_s),
    .K_cur             (K_cur),
    .C_cur             (C_cur),
    .F_cur             (F_cur),
    .Hout_cur          (Hout_cur),
    .Wout_cur          (Wout_cur),
    .tile_col_base_g   (tile_col_base_g),
    .tile_col_count    (tile_col_count),
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
    .wb_bank_ready   (weight_bank_ready_m2_s),
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
  // Consume-qualified tuple tracking for Mode 2.
  //
  // This is the Mode-2 counterpart of Mode1's data_ready_q /
  // weight_valid_q / ctrl_step_en contract.  A returned weight bundle
  // first loads weight_register_mode2.  Only on the following cycle does
  // wgt_tuple_ready_q allow ce_controller_mode2 to assert mac_en and
  // advance the loop counters.  Therefore MAC never consumes the same
  // cycle as weight_write_en.
  // ------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ifm_tuple_ready_q     <= 1'b0;
      wgt_tuple_ready_q     <= 1'b0;
      weight_req_inflight_q <= 1'b0;
    end
    else begin
      if (start) begin
        ifm_tuple_ready_q     <= 1'b0;
        wgt_tuple_ready_q     <= 1'b0;
        weight_req_inflight_q <= wb_rd_en;
      end
      else begin
        // IFM readiness is kept at output-block granularity, matching the
        // existing Mode2 data-register lifecycle.  A write has priority
        // over the block-boundary clear.
        if (out_valid) begin
          ifm_tuple_ready_q <= 1'b0;
        end
        if (dr_write_en) begin
          ifm_tuple_ready_q <= 1'b1;
        end

        // Current weight bundle validity.  A real MAC consume invalidates
        // the current registered weight.  A same-cycle return/write has
        // priority and becomes consumable on the next cycle.
        if (mac_en) begin
          wgt_tuple_ready_q <= 1'b0;
        end
        if (weight_write_en) begin
          wgt_tuple_ready_q <= 1'b1;
        end

        // Track one outstanding weight request.  If a return and a new
        // request ever happen in the same cycle, keep the request in-flight.
        if (weight_write_en) begin
          weight_req_inflight_q <= 1'b0;
        end
        if (wb_rd_en) begin
          weight_req_inflight_q <= 1'b1;
        end
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
  
  
  `ifndef SYNTHESIS
always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    // debug only
  end else begin
    if ((K_cur == 4'd3) && (C_cur == 8'd24) && (F_cur == 8'd24) && (Hout_cur == 16'd52) && (Wout_cur == 16'd84) && (wb_rd_en || wb_rd_valid || weight_write_en || mac_en || out_valid)) begin
      $display("DBG_M2_WGT_L5 t=%0t start=%0b pass=%0b mac=%0b out_v=%0b clr=%0b row=%0d col=%0d fg=%0d cg=%0d ky=%0d kx=%0d wb_en=%0b wb_addr=%0d wb_valid=%0b wwe=%0b wb0=%0d ww0=%0d wraw0=%0d wmask0=%0d ifm0=%0d mac0=%0d", $time, start, pass_start_pulse, mac_en, out_valid, clear_psum, out_row, out_col, f_group, c_group, ky, kx, wb_rd_en, wb_rd_addr, wb_rd_valid, weight_write_en, $signed(wb_rd_data[0*DATA_W +: DATA_W]), $signed(weight_write_data[0*DATA_W +: DATA_W]), $signed(weight_out_raw[0*DATA_W +: DATA_W]), $signed(weight_out[0*DATA_W +: DATA_W]), $signed(data_out_logic[0*DATA_W +: DATA_W]), $signed(mac_data_out[0*PSUM_W +: PSUM_W]));
    end
  end
end
`endif

`ifndef SYNTHESIS

logic [31:0] dbg_l8_ce_evt_q;

function automatic logic dbg_l8_ce_focus_coord(input logic [15:0] row_g, input logic [15:0] col_g, input logic [15:0] fg_g, input logic [15:0] cg_g);
begin
  dbg_l8_ce_focus_coord = (fg_g == 16'd0) && (cg_g == 16'd0) && (row_g < 16'd4) && (col_g < 16'd24);
end
endfunction

logic dbg_l8_ce_layer_s;
assign dbg_l8_ce_layer_s = (K_cur == 4'd3) && (F_cur == 8'd16) && (Hout_cur == 16'd46) && (Wout_cur == 16'd78);

always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    dbg_l8_ce_evt_q <= 32'd0;
  end else begin
    if (dbg_l8_ce_layer_s && dbg_l8_ce_focus_coord(out_row, out_col, f_group, c_group) && (start || pass_start_pulse || wb_rd_en || wb_rd_valid || weight_write_en || ctrl_step_en || mac_en || out_valid || mac_data_out_valid)) begin
      dbg_l8_ce_evt_q <= dbg_l8_ce_evt_q + 32'd1;
      $display("DBG_L8_CE_TAP t=%0t evt=%0d start=%0b step_raw=%0b ctrl_step=%0b tuple=%0b ifm_rdy=%0b wgt_rdy=%0b inflight=%0b pass=%0b mac=%0b clr=%0b out_v=%0b mac_v=%0b row=%0d col=%0d fg=%0d cg=%0d ky=%0d kx=%0d wb_en=%0b wb_addr=%0d wb_valid=%0b wwe=%0b wb0=%0d wb1=%0d ww0=%0d ww1=%0d data_raw0=%0d data_raw1=%0d data0=%0d data1=%0d wraw0=%0d wraw1=%0d wmask0=%0d wmask1=%0d mac0=%0d mac1=%0d mac_f_base=%0d", $time, dbg_l8_ce_evt_q + 32'd1, start, step_en, ctrl_step_en, tuple_ready_s, ifm_tuple_ready_q, wgt_tuple_ready_q, weight_req_inflight_q, pass_start_pulse, mac_en, clear_psum, out_valid, mac_data_out_valid, out_row, out_col, f_group, c_group, ky, kx, wb_rd_en, wb_rd_addr, wb_rd_valid, weight_write_en, $signed(wb_rd_data[0*DATA_W +: DATA_W]), $signed(wb_rd_data[1*DATA_W +: DATA_W]), $signed(weight_write_data[0*DATA_W +: DATA_W]), $signed(weight_write_data[1*DATA_W +: DATA_W]), $signed(data_out_raw[0*DATA_W +: DATA_W]), $signed(data_out_raw[1*DATA_W +: DATA_W]), $signed(data_out_logic[0*DATA_W +: DATA_W]), $signed(data_out_logic[1*DATA_W +: DATA_W]), $signed(weight_out_raw[0*DATA_W +: DATA_W]), $signed(weight_out_raw[1*DATA_W +: DATA_W]), $signed(weight_out[0*DATA_W +: DATA_W]), $signed(weight_out[1*DATA_W +: DATA_W]), $signed(mac_data_out[0*PSUM_W +: PSUM_W]), $signed(mac_data_out[1*PSUM_W +: PSUM_W]), mac_f_base);
    end
  end
end

`endif

endmodule
