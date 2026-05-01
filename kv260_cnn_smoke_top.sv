`timescale 1ns/1ps
`include "cnn_ddr_defs.svh"

// ============================================================================
// KV260 smoke-test wrapper for the current CNN accelerator RTL
//
// Purpose:
// - Run the same small single-layer mode-1 test as tb_cnn_top.sv, but using
//   real PS DDR through an AXI4 master port.
// - PS/Linux initializes the reserved DDR window before run is asserted.
// - This wrapper internally programs one layer descriptor, pulses cnn_top.start,
//   and exposes status/debug signals for VIO/ILA.
//
// Expected DDR layout inside the reserved window pointed to by
// AXI_DDR_BASE_ADDR. The addresses below are WORD addresses from
// cnn_ddr_defs.svh, not byte addresses.
//
//   IFM: `DDR_IFM_BASE
//   WGT: `DDR_WGT_BASE
//   OFM: `DDR_OFM_BASE
//
// For the default smoke-test config:
//   DATA_W     = 8
//   PV_MAX     = 4
//   DDR_WORD_W = PV_MAX * DATA_W = 32 bits = 4 bytes
//
// Therefore byte offsets from AXI_DDR_BASE_ADDR are:
//   IFM offset = `DDR_IFM_BASE * 4 = 0x00000
//   WGT offset = `DDR_WGT_BASE * 4 = 0x20000
//   OFM offset = `DDR_OFM_BASE * 4 = 0x40000
//
// Vivado block design connection:
//   kv260_cnn_smoke_top.m_axi_* -> SmartConnect -> ZynqMP S_AXI_HP/HPC_FPD
// ============================================================================

module kv260_cnn_smoke_top
  import cnn_layer_desc_pkg::*;
#(
  // --------------------------------------------------------------------------
  // Keep these defaults aligned with tb_cnn_top.sv for the first hardware test.
  // Do not start with the large canonical project parameters; the simple bridge
  // expects AXI_DATA_W == DDR_WORD_W and this smoke wrapper defaults to 32-bit.
  // --------------------------------------------------------------------------
  parameter int DATA_W           = 8,
  parameter int PSUM_W           = 32,
  parameter int PTOTAL           = 4,
  parameter int PV_MAX           = 4,
  parameter int PF_MAX           = 4,
  parameter int PC_MODE2         = 2,
  parameter int PF_MODE2         = 2,
  parameter int C_MAX            = 4,
  parameter int F_MAX            = 4,
  parameter int W_MAX            = 4,
  parameter int H_MAX            = 4,
  parameter int HT               = 4,
  parameter int K_MAX            = 3,
  parameter int WGT_DEPTH        = 16,
  parameter int OFM_BANK_DEPTH   = H_MAX * W_MAX,
  parameter int OFM_LINEAR_DEPTH = C_MAX * OFM_BANK_DEPTH,
  parameter int CFG_DEPTH        = 4,
  parameter int DDR_ADDR_W       = `CNN_DDR_ADDR_W,
  parameter int DDR_WORD_W       = PV_MAX * DATA_W,

  // AXI master side. AXI_DATA_W is intentionally tied to DDR_WORD_W for the
  // current simple bridge. Use SmartConnect/NoC width conversion externally if
  // the PS HP/HPC port is configured wider than this.
  parameter int AXI_ADDR_W        = 40,
  parameter int AXI_DATA_W        = DDR_WORD_W,
  parameter int AXI_ID_W          = 1,
  parameter logic [AXI_ADDR_W-1:0] AXI_DDR_BASE_ADDR = 40'h0000_7000_0000,
  parameter int WR_FIFO_DEPTH     = 16
)(
  input  logic clk,
  input  logic rst_n,

  // Drive these from VIO/GPIO for the first board test.
  // rst_n and soft_reset_n are ANDed internally.
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
  output logic                         dbg_mode,
  output logic                         dbg_weight_bank,
  output logic [3:0]                   dbg_error_vec,

  // Optional direct-DDR side visibility for ILA.
  output logic                      dbg_ddr_rd_req,
  output logic [DDR_ADDR_W-1:0]     dbg_ddr_rd_addr,
  output logic                      dbg_ddr_rd_valid,
  output logic [DDR_WORD_W-1:0]     dbg_ddr_rd_data,
  output logic                      dbg_ddr_wr_en,
  output logic [DDR_ADDR_W-1:0]     dbg_ddr_wr_addr,
  output logic [DDR_WORD_W-1:0]     dbg_ddr_wr_data,
  output logic [(DDR_WORD_W/8)-1:0] dbg_ddr_wr_be,

  // AXI4 master write address channel.
  output logic [AXI_ID_W-1:0]       m_axi_awid,
  output logic [AXI_ADDR_W-1:0]     m_axi_awaddr,
  output logic [7:0]                m_axi_awlen,
  output logic [2:0]                m_axi_awsize,
  output logic [1:0]                m_axi_awburst,
  output logic                      m_axi_awlock,
  output logic [3:0]                m_axi_awcache,
  output logic [2:0]                m_axi_awprot,
  output logic [3:0]                m_axi_awqos,
  output logic [3:0]                m_axi_awregion,
  output logic                      m_axi_awvalid,
  input  logic                      m_axi_awready,

  // AXI4 master write data channel.
  output logic [AXI_DATA_W-1:0]     m_axi_wdata,
  output logic [(AXI_DATA_W/8)-1:0] m_axi_wstrb,
  output logic                      m_axi_wlast,
  output logic                      m_axi_wvalid,
  input  logic                      m_axi_wready,

  // AXI4 master write response channel.
  input  logic [AXI_ID_W-1:0]       m_axi_bid,
  input  logic [1:0]                m_axi_bresp,
  input  logic                      m_axi_bvalid,
  output logic                      m_axi_bready,

  // AXI4 master read address channel.
  output logic [AXI_ID_W-1:0]       m_axi_arid,
  output logic [AXI_ADDR_W-1:0]     m_axi_araddr,
  output logic [7:0]                m_axi_arlen,
  output logic [2:0]                m_axi_arsize,
  output logic [1:0]                m_axi_arburst,
  output logic                      m_axi_arlock,
  output logic [3:0]                m_axi_arcache,
  output logic [2:0]                m_axi_arprot,
  output logic [3:0]                m_axi_arqos,
  output logic [3:0]                m_axi_arregion,
  output logic                      m_axi_arvalid,
  input  logic                      m_axi_arready,

  // AXI4 master read data channel.
  input  logic [AXI_ID_W-1:0]       m_axi_rid,
  input  logic [AXI_DATA_W-1:0]     m_axi_rdata,
  input  logic [1:0]                m_axi_rresp,
  input  logic                      m_axi_rlast,
  input  logic                      m_axi_rvalid,
  output logic                      m_axi_rready
);

  localparam int CFG_AW = (CFG_DEPTH <= 1) ? 1 : $clog2(CFG_DEPTH);

  // Combined reset for VIO-friendly bring-up.
  logic rst_core_n;
  assign rst_core_n = rst_n & soft_reset_n;

  // --------------------------------------------------------------------------
  // cnn_top configuration/start wires
  // --------------------------------------------------------------------------
  logic                  cfg_wr_en_s;
  logic [CFG_AW-1:0]     cfg_wr_addr_s;
  layer_desc_t           cfg_wr_data_s;
  logic [$clog2(CFG_DEPTH+1)-1:0] cfg_num_layers_s;
  logic                  start_s;

  // Direct DDR interface between cnn_top and AXI bridge.
  logic                      ddr_rd_req_s;
  logic [DDR_ADDR_W-1:0]     ddr_rd_addr_s;
  logic                      ddr_rd_valid_s;
  logic [DDR_WORD_W-1:0]     ddr_rd_data_s;
  logic                      ddr_wr_en_s;
  logic [DDR_ADDR_W-1:0]     ddr_wr_addr_s;
  logic [DDR_WORD_W-1:0]     ddr_wr_data_s;
  logic [(DDR_WORD_W/8)-1:0] ddr_wr_be_s;

  // Same-mode refill hooks are tied off for this one-layer smoke test.
  logic [15:0] m1_free_col_blk_g_s;
  logic [15:0] m1_free_ch_blk_g_s;
  logic        m1_sm_refill_req_ready_s;
  logic        m1_sm_refill_req_valid_s;
  logic [$clog2(HT)-1:0] m1_sm_refill_row_slot_l_s;
  logic [15:0] m1_sm_refill_row_g_s;
  logic [15:0] m1_sm_refill_col_blk_g_s;
  logic [15:0] m1_sm_refill_ch_blk_g_s;
  logic        m2_sm_refill_req_ready_s;
  logic        m2_sm_refill_req_valid_s;
  logic [15:0] m2_sm_refill_row_g_s;
  logic [15:0] m2_sm_refill_col_g_s;
  logic [15:0] m2_sm_refill_col_l_s;
  logic [15:0] m2_sm_refill_cgrp_g_s;

  logic                    ifm_m1_free_valid_s;
  logic [$clog2(HT)-1:0]   ifm_m1_free_row_slot_l_s;
  logic [15:0]             ifm_m1_free_row_g_s;

  assign m1_free_col_blk_g_s      = '0;
  assign m1_free_ch_blk_g_s       = '0;
  assign m1_sm_refill_req_ready_s = 1'b1;
  assign m2_sm_refill_req_ready_s = 1'b1;

  assign cfg_num_layers_s = 'd1;

  // --------------------------------------------------------------------------
  // Build the same single mode-1 descriptor as tb_cnn_top.sv.
  // PS/Linux must initialize DDR before run is asserted:
  //   DDR_IFM_BASE + i = 32'h1000_0000 + i
  //   DDR_WGT_BASE + i = 32'h2000_0000 + i
  //   DDR_OFM_BASE region cleared to zero
  // --------------------------------------------------------------------------
  function automatic layer_desc_t make_single_mode1_layer();
    layer_desc_t cfg;
    begin
      cfg = '0;
      cfg.layer_id      = 0;
      cfg.mode          = MODE1;
      cfg.h_in          = 16'd4;
      cfg.w_in          = 16'd4;
      cfg.c_in          = 16'd1;
      cfg.f_out         = 16'd2;
      cfg.k             = 4'd3;
      cfg.h_out         = 16'd2;
      cfg.w_out         = 16'd2;
      cfg.pv_m1         = 8'd2;
      cfg.pf_m1         = 8'd2;
      cfg.pc_m2         = 8'd2;
      cfg.pf_m2         = 8'd2;
      cfg.conv_stride   = 2'd1;
      cfg.pad_top       = 4'd0;
      cfg.pad_bottom    = 4'd0;
      cfg.pad_left      = 4'd0;
      cfg.pad_right     = 4'd0;
      cfg.relu_en       = 1'b0;
      cfg.pool_en       = 1'b0;
      cfg.pool_k        = 2'd0;
      cfg.pool_stride   = 2'd0;
      cfg.ifm_ddr_base  = `DDR_IFM_BASE;
      cfg.wgt_ddr_base  = `DDR_WGT_BASE;
      cfg.ofm_ddr_base  = `DDR_OFM_BASE;
      cfg.first_layer   = 1'b1;
      cfg.last_layer    = 1'b1;
      return cfg;
    end
  endfunction

  // --------------------------------------------------------------------------
  // Internal config loader and one-shot run controller.
  // - After reset, write cfg[0] for one cycle.
  // - run is treated as a level from VIO/GPIO. A new run starts once per high
  //   assertion, then run must be lowered before another attempt.
  // --------------------------------------------------------------------------
  logic cfg_done_q;
  logic run_armed_q;

  always_ff @(posedge clk) begin
    if (!rst_core_n) begin
      cfg_done_q    <= 1'b0;
      cfg_wr_en_s   <= 1'b0;
      cfg_wr_addr_s <= '0;
      cfg_wr_data_s <= '0;
      start_s       <= 1'b0;
      run_armed_q   <= 1'b0;
    end else begin
      cfg_wr_en_s <= 1'b0;
      start_s     <= 1'b0;

      // Program one descriptor immediately after reset release.
      if (!cfg_done_q) begin
        cfg_wr_en_s   <= 1'b1;
        cfg_wr_addr_s <= '0;
        cfg_wr_data_s <= make_single_mode1_layer();
        cfg_done_q    <= 1'b1;
      end

      // Allow one start per run-high interval.
      if (!run) begin
        run_armed_q <= 1'b0;
      end else if (cfg_done_q && !run_armed_q && !core_busy && !core_done) begin
        start_s     <= 1'b1;
        run_armed_q <= 1'b1;
      end
    end
  end

  assign cfg_done    = cfg_done_q;
  assign start_pulse = start_s;

  // Combined public status.
  assign busy  = core_busy | bridge_busy;
  assign done  = core_done;
  assign error = core_error | bridge_error | wr_fifo_overflow;

  // ILA helper signals.
  assign dbg_ddr_rd_req   = ddr_rd_req_s;
  assign dbg_ddr_rd_addr  = ddr_rd_addr_s;
  assign dbg_ddr_rd_valid = ddr_rd_valid_s;
  assign dbg_ddr_rd_data  = ddr_rd_data_s;
  assign dbg_ddr_wr_en    = ddr_wr_en_s;
  assign dbg_ddr_wr_addr  = ddr_wr_addr_s;
  assign dbg_ddr_wr_data  = ddr_wr_data_s;
  assign dbg_ddr_wr_be    = ddr_wr_be_s;

  // --------------------------------------------------------------------------
  // CNN accelerator core, still using the direct DDR request/response interface.
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
    .clk             (clk),
    .rst_n           (rst_core_n),
    .ddr_rd_req      (ddr_rd_req_s),
    .ddr_rd_addr     (ddr_rd_addr_s),
    .ddr_rd_valid    (ddr_rd_valid_s),
    .ddr_rd_data     (ddr_rd_data_s),
    .ddr_wr_en       (ddr_wr_en_s),
    .ddr_wr_addr     (ddr_wr_addr_s),
    .ddr_wr_data     (ddr_wr_data_s),
    .ddr_wr_be       (ddr_wr_be_s),
    .busy            (bridge_busy),
    .error           (bridge_error),
    .wr_fifo_overflow(wr_fifo_overflow),
    .m_axi_awid      (m_axi_awid),
    .m_axi_awaddr    (m_axi_awaddr),
    .m_axi_awlen     (m_axi_awlen),
    .m_axi_awsize    (m_axi_awsize),
    .m_axi_awburst   (m_axi_awburst),
    .m_axi_awlock    (m_axi_awlock),
    .m_axi_awcache   (m_axi_awcache),
    .m_axi_awprot    (m_axi_awprot),
    .m_axi_awqos     (m_axi_awqos),
    .m_axi_awregion  (m_axi_awregion),
    .m_axi_awvalid   (m_axi_awvalid),
    .m_axi_awready   (m_axi_awready),
    .m_axi_wdata     (m_axi_wdata),
    .m_axi_wstrb     (m_axi_wstrb),
    .m_axi_wlast     (m_axi_wlast),
    .m_axi_wvalid    (m_axi_wvalid),
    .m_axi_wready    (m_axi_wready),
    .m_axi_bid       (m_axi_bid),
    .m_axi_bresp     (m_axi_bresp),
    .m_axi_bvalid    (m_axi_bvalid),
    .m_axi_bready    (m_axi_bready),
    .m_axi_arid      (m_axi_arid),
    .m_axi_araddr    (m_axi_araddr),
    .m_axi_arlen     (m_axi_arlen),
    .m_axi_arsize    (m_axi_arsize),
    .m_axi_arburst   (m_axi_arburst),
    .m_axi_arlock    (m_axi_arlock),
    .m_axi_arcache   (m_axi_arcache),
    .m_axi_arprot    (m_axi_arprot),
    .m_axi_arqos     (m_axi_arqos),
    .m_axi_arregion  (m_axi_arregion),
    .m_axi_arvalid   (m_axi_arvalid),
    .m_axi_arready   (m_axi_arready),
    .m_axi_rid       (m_axi_rid),
    .m_axi_rdata     (m_axi_rdata),
    .m_axi_rresp     (m_axi_rresp),
    .m_axi_rlast     (m_axi_rlast),
    .m_axi_rvalid    (m_axi_rvalid),
    .m_axi_rready    (m_axi_rready)
  );

endmodule
