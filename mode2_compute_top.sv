module mode2_compute_top #(
  parameter int DATA_W    = 8,
  parameter int PSUM_W    = 32,
  parameter int K_MAX     = 7,
  parameter int PC        = 16,
  parameter int PF        = 16,
  parameter int HOUT_MAX  = 224,
  parameter int WOUT_MAX  = 224,
  // Maximum filter count visible to pooling_mode2 row buffers.
  // Keep this >= largest F_cur used by Mode 2 benchmarks.
  parameter int F_MAX     = 512,
  parameter int WB_LANES  = PF*PC,
  parameter int WB_ADDR_W = 12
)(
  input  logic clk,
  input  logic rst_n,

  input  logic start,
  input  logic step_en,
  input  logic pool_en,

  input  logic [3:0] K_cur,
  input  logic [9:0] C_cur,
  input  logic [9:0] F_cur,
  input  logic [15:0] Hout_cur,
  input  logic [15:0] Wout_cur,

  // --------------------------------------------------------------------------
  // Mode-2 resident-tile window. One start of the CE computes one horizontal
  // tile only. The CE still exports GLOBAL out_col coordinates:
  //   out_col = tile_col_base_g + local_col
  // --------------------------------------------------------------------------
  input  logic [15:0] tile_col_base_g,
  input  logic [15:0] tile_col_count,

  input  logic                     dr_write_en,
  input  logic [$clog2(K_MAX)-1:0] dr_write_row_idx,
  input  logic [PC*DATA_W-1:0]     dr_write_data,

  input  logic                       weight_bank_sel,
  input  logic                       weight_bank_ready,
  output logic                       wb_rd_en,
  output logic                       wb_rd_buf_sel,
  output logic [WB_ADDR_W-1:0]       wb_rd_addr,
  input  logic [WB_LANES*DATA_W-1:0] wb_rd_data,
  input  logic                       wb_rd_valid,

  // --------------------------------------------------------------------------
  // Backward-compatible loop outputs
  // IMPORTANT: out_row/out_col are GLOBAL OFM coordinates.
  // --------------------------------------------------------------------------
  output logic [15:0]              out_row,
  output logic [15:0]              out_col,
  output logic [15:0]              f_group,
  output logic [15:0]              c_group,
  output logic [$clog2(K_MAX)-1:0] ky,
  output logic [$clog2(K_MAX)-1:0] kx,

  // --------------------------------------------------------------------------
  // Explicit contract outputs
  // These make the global/local split visible without breaking the old ports.
  // - out_row_g / out_col_g are GLOBAL OFM coordinates from controller.
  // - tile_col_base_g/tile_col_count define the resident horizontal tile.
  // - Downstream logic derives local tile column as:
  //       out_col_l = out_col_g - tile_col_base_g
  // --------------------------------------------------------------------------
  output logic [15:0]              out_row_g,
  output logic [15:0]              out_col_g,

  output logic                     mac_en,
  output logic                     clear_psum,
  output logic                     ce_out_valid,
  output logic                     pass_start_pulse,
  output logic                     group_start_pulse,
  output logic                     row_done_pulse,
  output logic [$clog2(K_MAX)-1:0] row_done_ky,
  output logic                     c_group_done_pulse,
  output logic                     pixel_done_pulse,
  output logic                     f_group_done_pulse,
  output logic                     done,
  output logic                     busy,

  output logic [PC*DATA_W-1:0]     ce_data_out_logic,
  output logic [PF*PC*DATA_W-1:0]  ce_weight_out,
  output logic [PF*PSUM_W-1:0]     ce_mac_data_out,
  output logic                     ce_mac_data_out_valid,
  output logic [PF*PSUM_W-1:0]     relu_data_out,
  output logic                     relu_data_out_valid,
  output logic                     relu_group_start,
  output logic [15:0]              relu_f_base,

  output logic                     ofm_wr_en,
  output logic [15:0]              ofm_wr_row,
  output logic [15:0]              ofm_wr_col,
  output logic [15:0]              ofm_wr_f_base,
  output logic [PF*DATA_W-1:0]     ofm_wr_data,

  // Explicit aliases for OFM write coordinates (global by contract)
  output logic [15:0]              ofm_wr_row_g,
  output logic [15:0]              ofm_wr_col_g
);

  logic        ce_mac_group_start;
  logic [15:0] ce_mac_f_base;

  logic [15:0] ce_out_row_g;
  logic [15:0] ce_out_col_g;

  // Coordinates/metadata aligned with relu_data_out_valid.
  // relu_mode2 registers the MAC output, so these are delayed together with
  // relu_f_base and relu_group_start before entering pooling_mode2.
  logic [15:0] relu_row_g;
  logic [15:0] relu_col_g;
  logic [15:0] relu_tile_col_base_g;

  ce_mode2_top #(
    .DATA_W   (DATA_W),
    .PSUM_W   (PSUM_W),
    .K_MAX    (K_MAX),
    .PC       (PC),
    .PF       (PF),
    .HOUT_MAX (HOUT_MAX),
    .WOUT_MAX (WOUT_MAX),
    .WB_LANES (WB_LANES),
    .WB_ADDR_W(WB_ADDR_W)
  ) u_ce_mode2_top (
    .clk               (clk),
    .rst_n             (rst_n),
    .start             (start),
    .step_en           (step_en),
    .K_cur             (K_cur),
    .C_cur             (C_cur),
    .F_cur             (F_cur),
    .Hout_cur          (Hout_cur),
    .Wout_cur          (Wout_cur),
    .tile_col_base_g   (tile_col_base_g),
    .tile_col_count    (tile_col_count),
    .dr_write_en       (dr_write_en),
    .dr_write_row_idx  (dr_write_row_idx),
    .dr_write_data     (dr_write_data),
    .weight_bank_sel   (weight_bank_sel),
    .weight_bank_ready (weight_bank_ready),
    .wb_rd_en          (wb_rd_en),
    .wb_rd_buf_sel     (wb_rd_buf_sel),
    .wb_rd_addr        (wb_rd_addr),
    .wb_rd_data        (wb_rd_data),
    .wb_rd_valid       (wb_rd_valid),
    .out_row           (ce_out_row_g),
    .out_col           (ce_out_col_g),
    .f_group           (f_group),
    .c_group           (c_group),
    .ky                (ky),
    .kx                (kx),
    .mac_en            (mac_en),
    .clear_psum        (clear_psum),
    .out_valid         (ce_out_valid),
    .pass_start_pulse  (pass_start_pulse),
    .group_start_pulse (group_start_pulse),
    .row_done_pulse    (row_done_pulse),
    .row_done_ky       (row_done_ky),
    .c_group_done_pulse(c_group_done_pulse),
    .pixel_done_pulse  (pixel_done_pulse),
    .f_group_done_pulse(f_group_done_pulse),
    .done              (done),
    .busy              (busy),
    .data_out_logic    (ce_data_out_logic),
    .weight_out        (ce_weight_out),
    .mac_data_out      (ce_mac_data_out),
    .mac_data_out_valid(ce_mac_data_out_valid),
    .mac_group_start   (ce_mac_group_start),
    .mac_f_base        (ce_mac_f_base)
  );

  // Export global coordinates explicitly, while preserving the old outputs.
  assign out_row_g = ce_out_row_g;
  assign out_col_g = ce_out_col_g;
  assign out_row   = ce_out_row_g;
  assign out_col   = ce_out_col_g;

  relu_mode2 #(
    .PSUM_W (PSUM_W),
    .PF     (PF)
  ) u_relu_mode2 (
    .clk           (clk),
    .rst_n         (rst_n),
    .data_in       (ce_mac_data_out),
    .data_in_valid (ce_mac_data_out_valid),
    .data_out      (relu_data_out),
    .data_out_valid(relu_data_out_valid)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      relu_group_start    <= 1'b0;
      relu_f_base         <= '0;
      relu_row_g          <= '0;
      relu_col_g          <= '0;
      relu_tile_col_base_g <= '0;
    end
    else begin
      relu_group_start <= ce_mac_group_start;
      if (ce_mac_data_out_valid) begin
        relu_f_base          <= ce_mac_f_base;
        relu_row_g           <= ce_out_row_g;
        relu_col_g           <= ce_out_col_g;
        relu_tile_col_base_g <= tile_col_base_g;
      end
    end
  end

  pooling_mode2 #(
    .DATA_W (PSUM_W),
    .OUT_W  (DATA_W),
    .PF     (PF),
    .W_MAX  (WOUT_MAX),
    .H_MAX  (HOUT_MAX),
    .F_MAX  (F_MAX)
  ) u_pooling_mode2 (
    .clk           (clk),
    .rst_n         (rst_n),
    .W_cur         (Wout_cur),
    .H_cur         (Hout_cur),
    .pool_en       (pool_en),
    .in_row_g      (relu_row_g),
    .in_col_g      (relu_col_g),
    .tile_col_base_g(relu_tile_col_base_g),
    .data_in       (relu_data_out),
    .data_in_valid (relu_data_out_valid),
    .in_group_start(relu_group_start),
    .in_f_base     (relu_f_base),
    .ofm_wr_en     (ofm_wr_en),
    .ofm_wr_row    (ofm_wr_row),
    .ofm_wr_col    (ofm_wr_col),
    .ofm_wr_f_base (ofm_wr_f_base),
    .ofm_wr_data   (ofm_wr_data)
  );

  // By contract, pooling/ofm write coordinates are GLOBAL OFM coordinates.
  assign ofm_wr_row_g = ofm_wr_row;
  assign ofm_wr_col_g = ofm_wr_col;


function automatic logic dbg_m2_col_focus(input logic [15:0] c);
  begin
    dbg_m2_col_focus =
      (c < 16'd40) ||
      ((c >= 16'd60) && (c <= 16'd68));
  end
endfunction

logic [31:0] dbg_m2_coord_evt_q;

always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    dbg_m2_coord_evt_q <= 32'd0;
  end
  else begin
    if ((ce_mac_data_out_valid &&
         (ce_out_row_g < 16'd4) &&
         (ce_mac_f_base == 16'd0) &&
         dbg_m2_col_focus(ce_out_col_g)) ||

        (relu_data_out_valid &&
         (relu_row_g < 16'd4) &&
         (relu_f_base == 16'd0) &&
         dbg_m2_col_focus(relu_col_g)) ||

        (ofm_wr_en &&
         (ofm_wr_row < 16'd4) &&
         (ofm_wr_f_base == 16'd0) &&
         (ofm_wr_col < 16'd48))) begin

      dbg_m2_coord_evt_q <= dbg_m2_coord_evt_q + 32'd1;

      $display("DBG_M2_COORD_PIPE t=%0t evt=%0d pool=%0b tile_base=%0d tile_count=%0d ce_v=%0b ce_row=%0d ce_col=%0d ce_fbase=%0d ce_data0=%0d relu_v=%0b relu_row=%0d relu_col=%0d relu_fbase=%0d relu_data0=%0d pool_wr=%0b wr_row=%0d wr_col=%0d wr_fbase=%0d wr_data0=%0d",
        $time,
        dbg_m2_coord_evt_q,
        pool_en,
        tile_col_base_g,
        tile_col_count,

        ce_mac_data_out_valid,
        ce_out_row_g,
        ce_out_col_g,
        ce_mac_f_base,
        $signed(ce_mac_data_out[0*PSUM_W +: PSUM_W]),

        relu_data_out_valid,
        relu_row_g,
        relu_col_g,
        relu_f_base,
        $signed(relu_data_out[0*PSUM_W +: PSUM_W]),

        ofm_wr_en,
        ofm_wr_row,
        ofm_wr_col,
        ofm_wr_f_base,
        $signed(ofm_wr_data[0*DATA_W +: DATA_W])
      );
    end
  end
end

`ifndef SYNTHESIS

logic [31:0] dbg_m2_pipe_evt_q;

logic dbg_m2_pipe_l5_s;
assign dbg_m2_pipe_l5_s =
       (K_cur     == 4'd3)  &&
       (C_cur     == 8'd24) &&
       (F_cur     == 8'd24) &&
       (Hout_cur  == 16'd52) &&
       (Wout_cur  == 16'd84);

function automatic logic dbg_m2_pipe_col_focus(input logic [15:0] col_g);
begin
  dbg_m2_pipe_col_focus =
      (col_g < 16'd4) ||
      ((col_g >= 16'd30) && (col_g <= 16'd34)) ||
      ((col_g >= 16'd62) && (col_g <= 16'd66)) ||
      ((col_g >= 16'd80) && (col_g <= 16'd83));
end
endfunction

always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    dbg_m2_pipe_evt_q <= 32'd0;
  end else begin
    if (dbg_m2_pipe_l5_s &&
        (dr_write_en || mac_en || ce_mac_data_out_valid || relu_data_out_valid || ofm_wr_en) &&
        (dbg_m2_pipe_col_focus(ce_out_col_g) ||
         dbg_m2_pipe_col_focus(relu_col_g) ||
         dbg_m2_pipe_col_focus(ofm_wr_col) ||
         (ce_mac_data_out_valid && (ce_mac_data_out[0*PSUM_W +: PSUM_W] == '0)) ||
         (relu_data_out_valid && (relu_data_out[0*PSUM_W +: PSUM_W] == '0)) ||
         (ofm_wr_en && (ofm_wr_data[0*DATA_W +: DATA_W] == '0)))) begin

      dbg_m2_pipe_evt_q <= dbg_m2_pipe_evt_q + 32'd1;

      $display("DBG_M2_PIPE_L5 t=%0t evt=%0d start=%0b busy=%0b done=%0b pool=%0b tile_base=%0d tile_count=%0d dr_wr=%0b dr_row=%0d dr0=%0d mac_en=%0b clr=%0b ce_v=%0b row=%0d col=%0d cgrp=%0d fgrp=%0d ky=%0d kx=%0d ifm0=%0d w0=%0d mac0=%0d relu_v=%0b relu_row=%0d relu_col=%0d relu_fbase=%0d relu0=%0d wr=%0b wr_row=%0d wr_col=%0d wr_fbase=%0d wr0=%0d", $time, dbg_m2_pipe_evt_q + 32'd1, start, busy, done, pool_en, tile_col_base_g, tile_col_count, dr_write_en, dr_write_row_idx, $signed(dr_write_data[0*DATA_W +: DATA_W]), mac_en, clear_psum, ce_mac_data_out_valid, ce_out_row_g, ce_out_col_g, c_group, f_group, ky, kx, $signed(ce_data_out_logic[0*DATA_W +: DATA_W]), $signed(ce_weight_out[0*DATA_W +: DATA_W]), $signed(ce_mac_data_out[0*PSUM_W +: PSUM_W]), relu_data_out_valid, relu_row_g, relu_col_g, relu_f_base, $signed(relu_data_out[0*PSUM_W +: PSUM_W]), ofm_wr_en, ofm_wr_row, ofm_wr_col, ofm_wr_f_base, $signed(ofm_wr_data[0*DATA_W +: DATA_W]));
    end
  end
end

`endif

`ifndef SYNTHESIS

logic [31:0] dbg_l8_pipe_evt_q;

function automatic logic dbg_l8_focus_coord(input logic [15:0] row_g, input logic [15:0] col_g, input logic [15:0] fbase_g);
begin
  dbg_l8_focus_coord = (fbase_g == 16'd0) && (row_g < 16'd4) && ((col_g < 16'd40) || ((col_g >= 16'd60) && (col_g < 16'd78)));
end
endfunction

logic dbg_l8_layer_s;
assign dbg_l8_layer_s = (K_cur == 4'd3) && (F_cur == 8'd16) && (Hout_cur == 16'd46) && (Wout_cur == 16'd78);

always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    dbg_l8_pipe_evt_q <= 32'd0;
  end else begin
    if (dbg_l8_layer_s && ((ce_mac_data_out_valid && dbg_l8_focus_coord(ce_out_row_g, ce_out_col_g, ce_mac_f_base)) || (relu_data_out_valid && dbg_l8_focus_coord(relu_row_g, relu_col_g, relu_f_base)) || (ofm_wr_en && dbg_l8_focus_coord(ofm_wr_row, ofm_wr_col, ofm_wr_f_base)))) begin
      dbg_l8_pipe_evt_q <= dbg_l8_pipe_evt_q + 32'd1;
      $display("DBG_L8_PIPE_ALIGN t=%0t evt=%0d pool=%0b tile_base=%0d tile_count=%0d mac=%0b clr=%0b ky=%0d kx=%0d cg=%0d fg=%0d ce_v=%0b ce_row=%0d ce_col=%0d ce_fbase=%0d ce0=%0d ce1=%0d ce2=%0d ce3=%0d relu_v=%0b relu_row=%0d relu_col=%0d relu_fbase=%0d relu0=%0d relu1=%0d relu2=%0d relu3=%0d wr=%0b wr_row=%0d wr_col=%0d wr_fbase=%0d wr0=%0d wr1=%0d wr2=%0d wr3=%0d", $time, dbg_l8_pipe_evt_q + 32'd1, pool_en, tile_col_base_g, tile_col_count, mac_en, clear_psum, ky, kx, c_group, f_group, ce_mac_data_out_valid, ce_out_row_g, ce_out_col_g, ce_mac_f_base, $signed(ce_mac_data_out[0*PSUM_W +: PSUM_W]), $signed(ce_mac_data_out[1*PSUM_W +: PSUM_W]), $signed(ce_mac_data_out[2*PSUM_W +: PSUM_W]), $signed(ce_mac_data_out[3*PSUM_W +: PSUM_W]), relu_data_out_valid, relu_row_g, relu_col_g, relu_f_base, $signed(relu_data_out[0*PSUM_W +: PSUM_W]), $signed(relu_data_out[1*PSUM_W +: PSUM_W]), $signed(relu_data_out[2*PSUM_W +: PSUM_W]), $signed(relu_data_out[3*PSUM_W +: PSUM_W]), ofm_wr_en, ofm_wr_row, ofm_wr_col, ofm_wr_f_base, $signed(ofm_wr_data[0*DATA_W +: DATA_W]), $signed(ofm_wr_data[1*DATA_W +: DATA_W]), $signed(ofm_wr_data[2*DATA_W +: DATA_W]), $signed(ofm_wr_data[3*DATA_W +: DATA_W]));
    end
  end
end

`endif

`ifndef SYNTHESIS

logic [31:0] dbg_m2_vgg_evt_q;

always_ff @(posedge clk or negedge rst_n) begin : DBG_M2_COMPUTE_ZERO_PATH_MON
  if (!rst_n) begin
    dbg_m2_vgg_evt_q <= 32'd0;
  end else begin
    if ((K_cur == 4'd3) &&
        (Hout_cur <= 16'd8) &&
        (Wout_cur <= 16'd8) &&
        (F_cur <= 8'd64) &&
        (dr_write_en || mac_en || ce_mac_data_out_valid || relu_data_out_valid || ofm_wr_en)) begin

      dbg_m2_vgg_evt_q <= dbg_m2_vgg_evt_q + 32'd1;

      $display("DBG_M2_ZERO_PATH t=%0t evt=%0d start=%0b busy=%0b done=%0b pool=%0b H=%0d W=%0d C=%0d F=%0d dr_wr=%0b dr_row=%0d dr0=%0d dr1=%0d dr2=%0d dr3=%0d mac=%0b clr=%0b ky=%0d kx=%0d row=%0d col=%0d cgrp=%0d fgrp=%0d ifm0=%0d ifm1=%0d w0=%0d w1=%0d ce_v=%0b ce_row=%0d ce_col=%0d ce_fbase=%0d mac0=%0d mac1=%0d relu_v=%0b relu0=%0d relu1=%0d wr=%0b wr_row=%0d wr_col=%0d wr_fbase=%0d wr0=%0d wr1=%0d", $time, dbg_m2_vgg_evt_q + 32'd1, start, busy, done, pool_en, Hout_cur, Wout_cur, C_cur, F_cur, dr_write_en, dr_write_row_idx, $signed(dr_write_data[0*DATA_W +: DATA_W]), $signed(dr_write_data[1*DATA_W +: DATA_W]), $signed(dr_write_data[2*DATA_W +: DATA_W]), $signed(dr_write_data[3*DATA_W +: DATA_W]), mac_en, clear_psum, ky, kx, out_row_g, out_col_g, c_group, f_group, $signed(ce_data_out_logic[0*DATA_W +: DATA_W]), $signed(ce_data_out_logic[1*DATA_W +: DATA_W]), $signed(ce_weight_out[0*DATA_W +: DATA_W]), $signed(ce_weight_out[1*DATA_W +: DATA_W]), ce_mac_data_out_valid, ce_out_row_g, ce_out_col_g, ce_mac_f_base, $signed(ce_mac_data_out[0*PSUM_W +: PSUM_W]), $signed(ce_mac_data_out[1*PSUM_W +: PSUM_W]), relu_data_out_valid, $signed(relu_data_out[0*PSUM_W +: PSUM_W]), $signed(relu_data_out[1*PSUM_W +: PSUM_W]), ofm_wr_en, ofm_wr_row, ofm_wr_col, ofm_wr_f_base, $signed(ofm_wr_data[0*DATA_W +: DATA_W]), $signed(ofm_wr_data[1*DATA_W +: DATA_W]));
    end
  end
end

`endif


endmodule
