`timescale 1ns/1ps
`include "cnn_ddr_defs.svh"

// ============================================================================
// KV260 board wrapper for CNN_V4 VGG16-like 32x32 P128 benchmark
//
// Purpose:
//   - Keep the same board-facing interface as kv260_cnn_smoke_top.sv.
//   - Do NOT hardcode per-layer descriptors in this top.
//   - Keep cnn_layer_desc_pkg.sv as type-only stable package.
//   - Fixed layer descriptors are local to this top, not stored in the package.
//   - cfg_num_layers is driven by NUM_LAYERS_LOCAL.
//
// DDR preload:
//   This wrapper does not initialize DDR. Before asserting run, preload:
//     - IFM  at `DDR_IFM_BASE
//     - WGT  at `DDR_WGT_BASE
//     - clear/read OFM at `DDR_OFM_BASE
//   through XSCT/Linux using files packed for CNN_V4 layout.
// ============================================================================

module kv260_cnn_eval_top_fixedcfg_vgg16_like_32_p128_defaults
  import cnn_layer_desc_pkg::*;
#(
  parameter int DATA_W = 8,
  parameter int PSUM_W = 32,

  // Hardware profile parameters. Set these per benchmark/testbench.
  parameter int PTOTAL = 128,
  parameter int PV_MAX = 8,
  parameter int PF_MAX = 16,
  parameter int PC_MODE2 = 8,
  parameter int PF_MODE2 = 16,

  parameter int C_MAX = 64,
  parameter int F_MAX = 64,
  parameter int W_MAX = 32,
  parameter int H_MAX = 32,
  parameter int HT = 4,
  parameter int K_MAX = 3,

  // Match the passing 96x96 VGG16-like simulation compact OFM profile.
  // One OFM word stores PV_MAX (=8) scalar lanes. The maximum row word count
  // in this benchmark is 4: W=32 with pack=8, or pooled W=16 with pack=4.
  parameter int OFM_ROW_STRIDE = 4,
  parameter int WGT_DEPTH = 512,
  parameter int OFM_BANK_DEPTH = H_MAX * OFM_ROW_STRIDE,
  parameter int OFM_LINEAR_DEPTH = F_MAX * OFM_BANK_DEPTH,
  parameter int CFG_DEPTH = 16,

  parameter int DDR_ADDR_W = `CNN_DDR_ADDR_W,
  parameter int DDR_WORD_W = PV_MAX * DATA_W,  // 8 lanes * 8 = 64 bits for this profile

  // AXI master side.
  parameter int AXI_ADDR_W = 40,
  parameter int AXI_DATA_W = DDR_WORD_W,
  parameter int AXI_ID_W = 1,
  parameter logic [AXI_ADDR_W-1:0] AXI_DDR_BASE_ADDR = 40'h0000_7000_0000,
  parameter int WR_FIFO_DEPTH = 64
)(
  input  logic clk,
  input  logic rst_n,

  // Drive these from VIO/GPIO.
  input  logic soft_reset_n,
  input  logic run,
  input  logic abort,

  // Status for VIO/ILA.
  output logic cfg_done,
  output logic start_pulse,
  output logic busy,
  output logic done,
  output logic error,
  output logic core_busy,
  output logic core_done,
  output logic core_error,
  output logic bridge_busy,
  output logic bridge_error,
  output logic wr_fifo_overflow,
  output logic [$clog2(CFG_DEPTH)-1:0] dbg_layer_idx,
  output logic dbg_mode,
  output logic dbg_weight_bank,
  output logic [3:0] dbg_error_vec,

  // Optional direct-DDR side visibility for ILA.
  output logic dbg_ddr_rd_req,
  output logic [DDR_ADDR_W-1:0] dbg_ddr_rd_addr,
  output logic dbg_ddr_rd_valid,
  output logic [DDR_WORD_W-1:0] dbg_ddr_rd_data,
  output logic dbg_ddr_wr_en,
  output logic [DDR_ADDR_W-1:0] dbg_ddr_wr_addr,
  output logic [DDR_WORD_W-1:0] dbg_ddr_wr_data,
  output logic [(DDR_WORD_W/8)-1:0] dbg_ddr_wr_be,

  // AXI4 master write address channel.
  output logic [AXI_ID_W-1:0] m_axi_awid,
  output logic [AXI_ADDR_W-1:0] m_axi_awaddr,
  output logic [7:0] m_axi_awlen,
  output logic [2:0] m_axi_awsize,
  output logic [1:0] m_axi_awburst,
  output logic m_axi_awlock,
  output logic [3:0] m_axi_awcache,
  output logic [2:0] m_axi_awprot,
  output logic [3:0] m_axi_awqos,
  output logic [3:0] m_axi_awregion,
  output logic m_axi_awvalid,
  input  logic m_axi_awready,

  // AXI4 master write data channel.
  output logic [AXI_DATA_W-1:0] m_axi_wdata,
  output logic [(AXI_DATA_W/8)-1:0] m_axi_wstrb,
  output logic m_axi_wlast,
  output logic m_axi_wvalid,
  input  logic m_axi_wready,

  // AXI4 master write response channel.
  input  logic [AXI_ID_W-1:0] m_axi_bid,
  input  logic [1:0] m_axi_bresp,
  input  logic m_axi_bvalid,
  output logic m_axi_bready,

  // AXI4 master read address channel.
  output logic [AXI_ID_W-1:0] m_axi_arid,
  output logic [AXI_ADDR_W-1:0] m_axi_araddr,
  output logic [7:0] m_axi_arlen,
  output logic [2:0] m_axi_arsize,
  output logic [1:0] m_axi_arburst,
  output logic m_axi_arlock,
  output logic [3:0] m_axi_arcache,
  output logic [2:0] m_axi_arprot,
  output logic [3:0] m_axi_arqos,
  output logic [3:0] m_axi_arregion,
  output logic m_axi_arvalid,
  input  logic m_axi_arready,

  // AXI4 master read data channel.
  input  logic [AXI_ID_W-1:0] m_axi_rid,
  input  logic [AXI_DATA_W-1:0] m_axi_rdata,
  input  logic [1:0] m_axi_rresp,
  input  logic m_axi_rlast,
  input  logic m_axi_rvalid,
  output logic m_axi_rready
);

  localparam int CFG_AW = (CFG_DEPTH <= 1) ? 1 : $clog2(CFG_DEPTH);
  localparam int NUM_LAYERS_W = (CFG_DEPTH <= 1) ? 1 : $clog2(CFG_DEPTH + 1);

  // Combined reset for VIO-friendly bring-up.
  logic rst_core_n;
  assign rst_core_n = rst_n & soft_reset_n;

  // --------------------------------------------------------------------------
  // cnn_top configuration/start wires.
  // --------------------------------------------------------------------------
  logic cfg_wr_en_s;
  logic [CFG_AW-1:0] cfg_wr_addr_s;
  layer_desc_t cfg_wr_data_s;
  logic [NUM_LAYERS_W-1:0] cfg_num_layers_s;
  logic start_s;

  localparam int NUM_LAYERS_LOCAL = 13;
  localparam logic [NUM_LAYERS_W-1:0] NUM_LAYERS_CFG = NUM_LAYERS_LOCAL;
  assign cfg_num_layers_s = NUM_LAYERS_CFG;

  function automatic layer_desc_t make_desc(
    input logic [LAYER_ID_W-1:0] layer_id,
    input layer_mode_e mode,
    input logic [DIM_W-1:0] h_in,
    input logic [DIM_W-1:0] w_in,
    input logic [DIM_W-1:0] c_in,
    input logic [DIM_W-1:0] f_out,
    input logic [K_W-1:0] k,
    input logic [DIM_W-1:0] h_out,
    input logic [DIM_W-1:0] w_out,
    input logic [PV_W-1:0] pv_m1,
    input logic [PF1_W-1:0] pf_m1,
    input logic [PC2_W-1:0] pc_m2,
    input logic [PF2_W-1:0] pf_m2,
    input logic pool_en,
    input logic [ADDR_W-1:0] ifm_base,
    input logic [ADDR_W-1:0] wgt_base,
    input logic [ADDR_W-1:0] ofm_base,
    input logic first_layer,
    input logic last_layer
  );
    layer_desc_t d;
    begin
      d = '0;
      d.layer_id     = layer_id;
      d.mode         = mode;
      d.h_in         = h_in;
      d.w_in         = w_in;
      d.c_in         = c_in;
      d.f_out        = f_out;
      d.k            = k;
      d.h_out        = h_out;
      d.w_out        = w_out;
      d.pv_m1        = pv_m1;
      d.pf_m1        = pf_m1;
      d.pc_m2        = pc_m2;
      d.pf_m2        = pf_m2;
      d.conv_stride  = 2'd1;
      d.pad_top      = 4'd1;
      d.pad_bottom   = 4'd1;
      d.pad_left     = 4'd1;
      d.pad_right    = 4'd1;
      d.relu_en      = 1'b1;
      d.pool_en      = pool_en;
      d.pool_k       = 2'd2;
      d.pool_stride  = 2'd2;
      d.ifm_ddr_base = ifm_base;
      d.wgt_ddr_base = wgt_base;
      d.ofm_ddr_base = ofm_base;
      d.first_layer  = first_layer;
      d.last_layer   = last_layer;
      return d;
    end
  endfunction

  function automatic layer_desc_t get_layer_desc_local(input int unsigned idx);
    layer_desc_t d;
    begin
      d = '0;
      case (idx)
      0: d = make_desc(8'd0, MODE1, 16'd32, 16'd32, 16'd3, 16'd16, 4'd3, 16'd32, 16'd32, 8'd8, 8'd16, 8'd0, 8'd0, 1'b0, 32'h00000, 32'h08000, 32'hF0000, 1'b1, 1'b0);
      1: d = make_desc(8'd1, MODE1, 16'd32, 16'd32, 16'd16, 16'd16, 4'd3, 16'd32, 16'd32, 8'd8, 8'd16, 8'd0, 8'd0, 1'b1, 32'h00000, 32'h08040, 32'hF0000, 1'b0, 1'b0);
      2: d = make_desc(8'd2, MODE1, 16'd16, 16'd16, 16'd16, 16'd32, 4'd3, 16'd16, 16'd16, 8'd8, 8'd16, 8'd0, 8'd0, 1'b0, 32'h00000, 32'h08160, 32'hF0000, 1'b0, 1'b0);
      3: d = make_desc(8'd3, MODE1, 16'd16, 16'd16, 16'd32, 16'd32, 4'd3, 16'd16, 16'd16, 8'd8, 8'd16, 8'd0, 8'd0, 1'b1, 32'h00000, 32'h083A0, 32'hF0000, 1'b0, 1'b0);
      4: d = make_desc(8'd4, MODE2, 16'd8, 16'd8, 16'd32, 16'd64, 4'd3, 16'd8, 16'd8, 8'd0, 8'd0, 8'd8, 8'd16, 1'b0, 32'h00000, 32'h08820, 32'hF0000, 1'b0, 1'b0);
      5: d = make_desc(8'd5, MODE2, 16'd8, 16'd8, 16'd64, 16'd64, 4'd3, 16'd8, 16'd8, 8'd0, 8'd0, 8'd8, 8'd16, 1'b0, 32'h00000, 32'h09120, 32'hF0000, 1'b0, 1'b0);
      6: d = make_desc(8'd6, MODE2, 16'd8, 16'd8, 16'd64, 16'd64, 4'd3, 16'd8, 16'd8, 8'd0, 8'd0, 8'd8, 8'd16, 1'b1, 32'h00000, 32'h0A320, 32'hF0000, 1'b0, 1'b0);
      7: d = make_desc(8'd7, MODE2, 16'd4, 16'd4, 16'd64, 16'd64, 4'd3, 16'd4, 16'd4, 8'd0, 8'd0, 8'd8, 8'd16, 1'b0, 32'h00000, 32'h0B520, 32'hF0000, 1'b0, 1'b0);
      8: d = make_desc(8'd8, MODE2, 16'd4, 16'd4, 16'd64, 16'd64, 4'd3, 16'd4, 16'd4, 8'd0, 8'd0, 8'd8, 8'd16, 1'b0, 32'h00000, 32'h0C720, 32'hF0000, 1'b0, 1'b0);
      9: d = make_desc(8'd9, MODE2, 16'd4, 16'd4, 16'd64, 16'd64, 4'd3, 16'd4, 16'd4, 8'd0, 8'd0, 8'd8, 8'd16, 1'b1, 32'h00000, 32'h0D920, 32'hF0000, 1'b0, 1'b0);
      10: d = make_desc(8'd10, MODE2, 16'd2, 16'd2, 16'd64, 16'd64, 4'd3, 16'd2, 16'd2, 8'd0, 8'd0, 8'd8, 8'd16, 1'b0, 32'h00000, 32'h0EB20, 32'hF0000, 1'b0, 1'b0);
      11: d = make_desc(8'd11, MODE2, 16'd2, 16'd2, 16'd64, 16'd64, 4'd3, 16'd2, 16'd2, 8'd0, 8'd0, 8'd8, 8'd16, 1'b0, 32'h00000, 32'h0FD20, 32'hF0000, 1'b0, 1'b0);
      12: d = make_desc(8'd12, MODE2, 16'd2, 16'd2, 16'd64, 16'd64, 4'd3, 16'd2, 16'd2, 8'd0, 8'd0, 8'd8, 8'd16, 1'b1, 32'h00000, 32'h10F20, 32'hF0000, 1'b0, 1'b1);
        default: d = '0;
      endcase
      return d;
    end
  endfunction

  initial begin
    if (NUM_LAYERS_LOCAL > CFG_DEPTH) begin
      $error("NUM_LAYERS_LOCAL (%0d) exceeds CFG_DEPTH (%0d)", NUM_LAYERS_LOCAL, CFG_DEPTH);
    end
  end

  // Direct DDR interface between cnn_top and AXI bridge.
  logic ddr_rd_req_s;
  logic [DDR_ADDR_W-1:0] ddr_rd_addr_s;
  logic ddr_rd_valid_s;
  logic [DDR_WORD_W-1:0] ddr_rd_data_s;
  logic ddr_wr_en_s;
  logic [DDR_ADDR_W-1:0] ddr_wr_addr_s;
  logic [DDR_WORD_W-1:0] ddr_wr_data_s;
  logic [(DDR_WORD_W/8)-1:0] ddr_wr_be_s;

  // Same-mode refill hooks follow the existing board smoke-top tie-offs.
  logic [15:0] m1_free_col_blk_g_s;
  logic [15:0] m1_free_ch_blk_g_s;
  logic m1_sm_refill_req_ready_s;
  logic m1_sm_refill_req_valid_s;
  logic [$clog2(HT)-1:0] m1_sm_refill_row_slot_l_s;
  logic [15:0] m1_sm_refill_row_g_s;
  logic [15:0] m1_sm_refill_col_blk_g_s;
  logic [15:0] m1_sm_refill_ch_blk_g_s;

  logic m2_sm_refill_req_ready_s;
  logic m2_sm_refill_req_valid_s;
  logic [15:0] m2_sm_refill_row_g_s;
  logic [15:0] m2_sm_refill_col_g_s;
  logic [15:0] m2_sm_refill_col_l_s;
  logic [15:0] m2_sm_refill_cgrp_g_s;

  logic ifm_m1_free_valid_s;
  logic [$clog2(HT)-1:0] ifm_m1_free_row_slot_l_s;
  logic [15:0] ifm_m1_free_row_g_s;

  assign m1_free_col_blk_g_s = '0;
  assign m1_free_ch_blk_g_s  = '0;
  assign m1_sm_refill_req_ready_s = 1'b1;
  assign m2_sm_refill_req_ready_s = 1'b1;

  // --------------------------------------------------------------------------
  // Internal config loader and one-shot run controller.
  // --------------------------------------------------------------------------
  logic cfg_done_q;
  logic run_armed_q;
  logic [CFG_AW-1:0] cfg_load_idx_q;

  always_ff @(posedge clk) begin
    if (!rst_core_n) begin
      cfg_done_q      <= 1'b0;
      cfg_load_idx_q  <= '0;
      cfg_wr_en_s     <= 1'b0;
      cfg_wr_addr_s   <= '0;
      cfg_wr_data_s   <= '0;
      start_s         <= 1'b0;
      run_armed_q     <= 1'b0;
    end else begin
      cfg_wr_en_s <= 1'b0;
      start_s     <= 1'b0;

      if (!cfg_done_q) begin
        cfg_wr_en_s   <= 1'b1;
        cfg_wr_addr_s <= cfg_load_idx_q;
        cfg_wr_data_s <= get_layer_desc_local(int'(cfg_load_idx_q));

        if (cfg_load_idx_q == CFG_AW'(NUM_LAYERS_LOCAL - 1)) begin
          cfg_done_q <= 1'b1;
        end else begin
          cfg_load_idx_q <= cfg_load_idx_q + 1'b1;
        end
      end

      // One start pulse per run-high interval.
      if (!run) begin
        run_armed_q <= 1'b0;
      end else if (cfg_done_q && !run_armed_q && !core_busy && !core_done) begin
        start_s     <= 1'b1;
        run_armed_q <= 1'b1;
      end
    end
  end


logic core_done_seen_q;
logic done_q;

always_ff @(posedge clk or negedge rst_core_n) begin
  if (!rst_core_n) begin
    core_done_seen_q <= 1'b0;
    done_q           <= 1'b0;
  end else begin
    if (start_s) begin
      core_done_seen_q <= 1'b0;
      done_q           <= 1'b0;
    end else begin
      if (core_done) begin
        core_done_seen_q <= 1'b1;
      end

      if ((core_done_seen_q || core_done) && !bridge_busy) begin
        done_q <= 1'b1;
      end
    end
  end
end


  assign cfg_done    = cfg_done_q;
  assign start_pulse = start_s;
  assign busy        = (core_busy || bridge_busy || core_done_seen_q) && !done_q;
  assign done        = done_q;
  assign error       = core_error | bridge_error | wr_fifo_overflow;

  assign dbg_ddr_rd_req   = ddr_rd_req_s;
  assign dbg_ddr_rd_addr  = ddr_rd_addr_s;
  assign dbg_ddr_rd_valid = ddr_rd_valid_s;
  assign dbg_ddr_rd_data  = ddr_rd_data_s;
  assign dbg_ddr_wr_en    = ddr_wr_en_s;
  assign dbg_ddr_wr_addr  = ddr_wr_addr_s;
  assign dbg_ddr_wr_data  = ddr_wr_data_s;
  assign dbg_ddr_wr_be    = ddr_wr_be_s;

  // --------------------------------------------------------------------------
  // CNN accelerator core.
  // --------------------------------------------------------------------------
  cnn_top #(
    .DATA_W          (DATA_W),
    .PSUM_W          (PSUM_W),
    .PTOTAL          (PTOTAL),
    .PV_MAX          (PV_MAX),
    .PF_MAX          (PF_MAX),
    .PC_MODE2        (PC_MODE2),
    .PF_MODE2        (PF_MODE2),
    .C_MAX           (C_MAX),
    .F_MAX           (F_MAX),
    .W_MAX           (W_MAX),
    .H_MAX           (H_MAX),
    .HT              (HT),
    .K_MAX           (K_MAX),
    .WGT_DEPTH       (WGT_DEPTH),
    .OFM_BANK_DEPTH  (OFM_BANK_DEPTH),
    .OFM_LINEAR_DEPTH(OFM_LINEAR_DEPTH),
    .CFG_DEPTH       (CFG_DEPTH),
    .DDR_ADDR_W      (DDR_ADDR_W),
    .DDR_WORD_W      (DDR_WORD_W)
  ) u_cnn_top (
    .clk                    (clk),
    .rst_n                  (rst_core_n),
    .start                  (start_s),
    .abort                  (abort),
    .cfg_wr_en              (cfg_wr_en_s),
    .cfg_wr_addr            (cfg_wr_addr_s),
    .cfg_wr_data            (cfg_wr_data_s),
    .cfg_num_layers         (cfg_num_layers_s),
    .ddr_rd_req             (ddr_rd_req_s),
    .ddr_rd_addr            (ddr_rd_addr_s),
    .ddr_rd_valid           (ddr_rd_valid_s),
    .ddr_rd_data            (ddr_rd_data_s),
    .ddr_wr_en              (ddr_wr_en_s),
    .ddr_wr_addr            (ddr_wr_addr_s),
    .ddr_wr_data            (ddr_wr_data_s),
    .ddr_wr_be              (ddr_wr_be_s),
    .m1_free_col_blk_g      (m1_free_col_blk_g_s),
    .m1_free_ch_blk_g       (m1_free_ch_blk_g_s),
    .m1_sm_refill_req_ready (m1_sm_refill_req_ready_s),
    .m1_sm_refill_req_valid (m1_sm_refill_req_valid_s),
    .m1_sm_refill_row_slot_l(m1_sm_refill_row_slot_l_s),
    .m1_sm_refill_row_g     (m1_sm_refill_row_g_s),
    .m1_sm_refill_col_blk_g (m1_sm_refill_col_blk_g_s),
    .m1_sm_refill_ch_blk_g  (m1_sm_refill_ch_blk_g_s),
    .m2_sm_refill_req_ready (m2_sm_refill_req_ready_s),
    .m2_sm_refill_req_valid (m2_sm_refill_req_valid_s),
    .m2_sm_refill_row_g     (m2_sm_refill_row_g_s),
    .m2_sm_refill_col_g     (m2_sm_refill_col_g_s),
    .m2_sm_refill_col_l     (m2_sm_refill_col_l_s),
    .m2_sm_refill_cgrp_g    (m2_sm_refill_cgrp_g_s),
    .ifm_m1_free_valid      (ifm_m1_free_valid_s),
    .ifm_m1_free_row_slot_l (ifm_m1_free_row_slot_l_s),
    .ifm_m1_free_row_g      (ifm_m1_free_row_g_s),
    .busy                   (core_busy),
    .done                   (core_done),
    .error                  (core_error),
    .dbg_layer_idx          (dbg_layer_idx),
    .dbg_mode               (dbg_mode),
    .dbg_weight_bank        (dbg_weight_bank),
    .dbg_error_vec          (dbg_error_vec)
  );

  // --------------------------------------------------------------------------
  // Direct DDR -> AXI4 bridge for PS DDR access.
  // --------------------------------------------------------------------------
  cnn_dma_to_axi_bridge_kv260 #(
    .DDR_ADDR_W       (DDR_ADDR_W),
    .DDR_WORD_W       (DDR_WORD_W),
    .AXI_ADDR_W       (AXI_ADDR_W),
    .AXI_DATA_W       (AXI_DATA_W),
    .AXI_ID_W         (AXI_ID_W),
    .AXI_DDR_BASE_ADDR(AXI_DDR_BASE_ADDR),
    .WR_FIFO_DEPTH    (WR_FIFO_DEPTH)
  ) u_dma_to_axi_bridge (
    .clk            (clk),
    .rst_n          (rst_core_n),
    .ddr_rd_req     (ddr_rd_req_s),
    .ddr_rd_addr    (ddr_rd_addr_s),
    .ddr_rd_valid   (ddr_rd_valid_s),
    .ddr_rd_data    (ddr_rd_data_s),
    .ddr_wr_en      (ddr_wr_en_s),
    .ddr_wr_addr    (ddr_wr_addr_s),
    .ddr_wr_data    (ddr_wr_data_s),
    .ddr_wr_be      (ddr_wr_be_s),
    .busy           (bridge_busy),
    .error          (bridge_error),
    .wr_fifo_overflow(wr_fifo_overflow),
    .m_axi_awid     (m_axi_awid),
    .m_axi_awaddr   (m_axi_awaddr),
    .m_axi_awlen    (m_axi_awlen),
    .m_axi_awsize   (m_axi_awsize),
    .m_axi_awburst  (m_axi_awburst),
    .m_axi_awlock   (m_axi_awlock),
    .m_axi_awcache  (m_axi_awcache),
    .m_axi_awprot   (m_axi_awprot),
    .m_axi_awqos    (m_axi_awqos),
    .m_axi_awregion (m_axi_awregion),
    .m_axi_awvalid  (m_axi_awvalid),
    .m_axi_awready  (m_axi_awready),
    .m_axi_wdata    (m_axi_wdata),
    .m_axi_wstrb    (m_axi_wstrb),
    .m_axi_wlast    (m_axi_wlast),
    .m_axi_wvalid   (m_axi_wvalid),
    .m_axi_wready   (m_axi_wready),
    .m_axi_bid      (m_axi_bid),
    .m_axi_bresp    (m_axi_bresp),
    .m_axi_bvalid   (m_axi_bvalid),
    .m_axi_bready   (m_axi_bready),
    .m_axi_arid     (m_axi_arid),
    .m_axi_araddr   (m_axi_araddr),
    .m_axi_arlen    (m_axi_arlen),
    .m_axi_arsize   (m_axi_arsize),
    .m_axi_arburst  (m_axi_arburst),
    .m_axi_arlock   (m_axi_arlock),
    .m_axi_arcache  (m_axi_arcache),
    .m_axi_arprot   (m_axi_arprot),
    .m_axi_arqos    (m_axi_arqos),
    .m_axi_arregion (m_axi_arregion),
    .m_axi_arvalid  (m_axi_arvalid),
    .m_axi_arready  (m_axi_arready),
    .m_axi_rid      (m_axi_rid),
    .m_axi_rdata    (m_axi_rdata),
    .m_axi_rresp    (m_axi_rresp),
    .m_axi_rlast    (m_axi_rlast),
    .m_axi_rvalid   (m_axi_rvalid),
    .m_axi_rready   (m_axi_rready)
  );

endmodule
