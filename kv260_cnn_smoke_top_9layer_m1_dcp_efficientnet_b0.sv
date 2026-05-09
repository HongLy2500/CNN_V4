`timescale 1ns/1ps
`include "cnn_ddr_defs.svh"

// ============================================================================
// KV260 board wrapper for:
// tb_cnn_top_9layer_m1_dcp_efficientnet_b0_tablevi_fullscale_with_expected_compare.sv
//
// This wrapper keeps the same layer table/configuration intent as the 9-layer
// full-scale Mode-1 DCP/EfficientNet-B0-prefix testbench:
//
//   L0: 224x224x3   -> conv 222x222x32,  K=3, pool -> 111x111x32
//   L1: 111x111x32  -> conv 109x109x16,  K=3
//   L2: 109x109x16  -> conv 107x107x24,  K=3, pool -> 53x53x24
//   L3: 53x53x24    -> conv 51x51x24,    K=3
//   L4: 51x51x24    -> conv 49x49x40,    K=3, pool -> 24x24x40
//   L5: 24x24x40    -> conv 22x22x40,    K=3
//   L6: 22x22x40    -> conv 20x20x80,    K=3, pool -> 10x10x80
//   L7: 10x10x80    -> conv 8x8x80,      K=3
//   L8: 8x8x80      -> conv 6x6x192,     K=3
//
// IMPORTANT:
// - This wrapper does NOT auto-initialize DDR. Load IFM/weights and clear OFM
//   from XSCT/Linux before asserting run.
// - This wrapper is intentionally drop-in compatible with the existing KV260
//   Block Design ports, but its default data width is much larger than the
//   small smoke test:
//      DDR_WORD_W = PV_MAX*DATA_W = 128*8 = 1024 bits.
//   The current cnn_dma_to_axi_bridge_kv260 requires AXI_DATA_W == DDR_WORD_W.
//   Therefore the BD/SmartConnect/PS port must accept this AXI width or be
//   reworked with a proper width-converting bridge.
// ============================================================================

module kv260_cnn_smoke_top
  import cnn_layer_desc_pkg::*;
#(
  parameter int DATA_W           = 8,
  parameter int PSUM_W           = 32,

  // DCP-CNN Table VI Mode-1 prefix: Pv*Pf = 2048.
  parameter int PTOTAL           = 2048,
  parameter int PV_MAX           = 128,
  parameter int PF_MAX           = 128,

  // Mode-2 parameters are kept for descriptor compatibility; this test is M1.
  parameter int PC_MODE2         = 32,
  parameter int PF_MODE2         = 64,

  // EfficientNet-B0 table-derived Mode-1 prefix scale.
  parameter int C_MAX            = 192,
  parameter int F_MAX            = 192,
  parameter int W_MAX            = 224,
  parameter int H_MAX            = 224,
  parameter int HT               = 4,
  parameter int K_MAX            = 3,

  // Match the testbench row-aligned OFM storage.
  parameter int OFM_ROW_STRIDE   = 16,
  parameter int WGT_DEPTH        = 8192,
  parameter int OFM_BANK_DEPTH   = H_MAX * OFM_ROW_STRIDE,
  parameter int OFM_LINEAR_DEPTH = C_MAX * OFM_BANK_DEPTH,
  parameter int CFG_DEPTH        = 16,

  parameter int DDR_ADDR_W       = `CNN_DDR_ADDR_W,
  parameter int DDR_WORD_W       = PV_MAX * DATA_W,

  // AXI master side.
  parameter int AXI_ADDR_W        = 40,
  parameter int AXI_DATA_W        = DDR_WORD_W,
  parameter int AXI_ID_W          = 1,
  parameter logic [AXI_ADDR_W-1:0] AXI_DDR_BASE_ADDR = 40'h0000_7000_0000,
  parameter int WR_FIFO_DEPTH     = 16
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

  // --------------------------------------------------------------------------
  // Layer geometry copied from the full-scale 9-layer Mode-1 testbench.
  // h_out/w_out descriptor fields are conv-output dimensions; pooling is
  // described by pool_en/pool_k/pool_stride.
  // --------------------------------------------------------------------------
  localparam int L0_H_IN=224, L0_W_IN=224, L0_C_IN=3,  L0_F_OUT=32,  L0_K=3, L0_POOL_EN=1;
  localparam int L0_H_CONV_OUT=L0_H_IN-L0_K+1, L0_W_CONV_OUT=L0_W_IN-L0_K+1;
  localparam int L0_H_OUT=(L0_POOL_EN ? (L0_H_CONV_OUT/2) : L0_H_CONV_OUT);
  localparam int L0_W_OUT=(L0_POOL_EN ? (L0_W_CONV_OUT/2) : L0_W_CONV_OUT);

  localparam int L1_H_IN=L0_H_OUT, L1_W_IN=L0_W_OUT, L1_C_IN=L0_F_OUT, L1_F_OUT=16, L1_K=3, L1_POOL_EN=0;
  localparam int L1_H_CONV_OUT=L1_H_IN-L1_K+1, L1_W_CONV_OUT=L1_W_IN-L1_K+1;
  localparam int L1_H_OUT=(L1_POOL_EN ? (L1_H_CONV_OUT/2) : L1_H_CONV_OUT);
  localparam int L1_W_OUT=(L1_POOL_EN ? (L1_W_CONV_OUT/2) : L1_W_CONV_OUT);

  localparam int L2_H_IN=L1_H_OUT, L2_W_IN=L1_W_OUT, L2_C_IN=L1_F_OUT, L2_F_OUT=24, L2_K=3, L2_POOL_EN=1;
  localparam int L2_H_CONV_OUT=L2_H_IN-L2_K+1, L2_W_CONV_OUT=L2_W_IN-L2_K+1;
  localparam int L2_H_OUT=(L2_POOL_EN ? (L2_H_CONV_OUT/2) : L2_H_CONV_OUT);
  localparam int L2_W_OUT=(L2_POOL_EN ? (L2_W_CONV_OUT/2) : L2_W_CONV_OUT);

  localparam int L3_H_IN=L2_H_OUT, L3_W_IN=L2_W_OUT, L3_C_IN=L2_F_OUT, L3_F_OUT=24, L3_K=3, L3_POOL_EN=0;
  localparam int L3_H_CONV_OUT=L3_H_IN-L3_K+1, L3_W_CONV_OUT=L3_W_IN-L3_K+1;
  localparam int L3_H_OUT=(L3_POOL_EN ? (L3_H_CONV_OUT/2) : L3_H_CONV_OUT);
  localparam int L3_W_OUT=(L3_POOL_EN ? (L3_W_CONV_OUT/2) : L3_W_CONV_OUT);

  localparam int L4_H_IN=L3_H_OUT, L4_W_IN=L3_W_OUT, L4_C_IN=L3_F_OUT, L4_F_OUT=40, L4_K=3, L4_POOL_EN=1;
  localparam int L4_H_CONV_OUT=L4_H_IN-L4_K+1, L4_W_CONV_OUT=L4_W_IN-L4_K+1;
  localparam int L4_H_OUT=(L4_POOL_EN ? (L4_H_CONV_OUT/2) : L4_H_CONV_OUT);
  localparam int L4_W_OUT=(L4_POOL_EN ? (L4_W_CONV_OUT/2) : L4_W_CONV_OUT);

  localparam int L5_H_IN=L4_H_OUT, L5_W_IN=L4_W_OUT, L5_C_IN=L4_F_OUT, L5_F_OUT=40, L5_K=3, L5_POOL_EN=0;
  localparam int L5_H_CONV_OUT=L5_H_IN-L5_K+1, L5_W_CONV_OUT=L5_W_IN-L5_K+1;
  localparam int L5_H_OUT=(L5_POOL_EN ? (L5_H_CONV_OUT/2) : L5_H_CONV_OUT);
  localparam int L5_W_OUT=(L5_POOL_EN ? (L5_W_CONV_OUT/2) : L5_W_CONV_OUT);

  localparam int L6_H_IN=L5_H_OUT, L6_W_IN=L5_W_OUT, L6_C_IN=L5_F_OUT, L6_F_OUT=80, L6_K=3, L6_POOL_EN=1;
  localparam int L6_H_CONV_OUT=L6_H_IN-L6_K+1, L6_W_CONV_OUT=L6_W_IN-L6_K+1;
  localparam int L6_H_OUT=(L6_POOL_EN ? (L6_H_CONV_OUT/2) : L6_H_CONV_OUT);
  localparam int L6_W_OUT=(L6_POOL_EN ? (L6_W_CONV_OUT/2) : L6_W_CONV_OUT);

  localparam int L7_H_IN=L6_H_OUT, L7_W_IN=L6_W_OUT, L7_C_IN=L6_F_OUT, L7_F_OUT=80, L7_K=3, L7_POOL_EN=0;
  localparam int L7_H_CONV_OUT=L7_H_IN-L7_K+1, L7_W_CONV_OUT=L7_W_IN-L7_K+1;
  localparam int L7_H_OUT=(L7_POOL_EN ? (L7_H_CONV_OUT/2) : L7_H_CONV_OUT);
  localparam int L7_W_OUT=(L7_POOL_EN ? (L7_W_CONV_OUT/2) : L7_W_CONV_OUT);

  localparam int L8_H_IN=L7_H_OUT, L8_W_IN=L7_W_OUT, L8_C_IN=L7_F_OUT, L8_F_OUT=192, L8_K=3, L8_POOL_EN=0;
  localparam int L8_H_CONV_OUT=L8_H_IN-L8_K+1, L8_W_CONV_OUT=L8_W_IN-L8_K+1;
  localparam int L8_H_OUT=(L8_POOL_EN ? (L8_H_CONV_OUT/2) : L8_H_CONV_OUT);
  localparam int L8_W_OUT=(L8_POOL_EN ? (L8_W_CONV_OUT/2) : L8_W_CONV_OUT);

  // DCP Table-VI expanded Mode-1 Pv/Pf.
  localparam int L0_PV=128, L0_PF=16;
  localparam int L1_PV=128, L1_PF=16;
  localparam int L2_PV=64,  L2_PF=32;
  localparam int L3_PV=64,  L3_PF=32;
  localparam int L4_PV=64,  L4_PF=32;
  localparam int L5_PV=32,  L5_PF=64;
  localparam int L6_PV=32,  L6_PF=64;
  localparam int L7_PV=32,  L7_PF=64;
  localparam int L8_PV=16,  L8_PF=128;

  // Weight DDR packing/bases copied from the testbench.
  localparam int WGT_SUBWORDS=(PTOTAL+PV_MAX-1)/PV_MAX;

  localparam int L0_NUM_FGROUP=(L0_F_OUT+L0_PF-1)/L0_PF;
  localparam int L0_M1_LOGICAL_BUNDLES=L0_NUM_FGROUP*L0_C_IN*L0_K*L0_K;
  localparam int L0_M1_PHYS_WORDS=(L0_M1_LOGICAL_BUNDLES+L0_PV-1)/L0_PV;
  localparam int L0_M1_WGT_DDR_WORDS=L0_M1_PHYS_WORDS*WGT_SUBWORDS;

  localparam int L1_NUM_FGROUP=(L1_F_OUT+L1_PF-1)/L1_PF;
  localparam int L1_M1_LOGICAL_BUNDLES=L1_NUM_FGROUP*L1_C_IN*L1_K*L1_K;
  localparam int L1_M1_PHYS_WORDS=(L1_M1_LOGICAL_BUNDLES+L1_PV-1)/L1_PV;
  localparam int L1_M1_WGT_DDR_WORDS=L1_M1_PHYS_WORDS*WGT_SUBWORDS;

  localparam int L2_NUM_FGROUP=(L2_F_OUT+L2_PF-1)/L2_PF;
  localparam int L2_M1_LOGICAL_BUNDLES=L2_NUM_FGROUP*L2_C_IN*L2_K*L2_K;
  localparam int L2_M1_PHYS_WORDS=(L2_M1_LOGICAL_BUNDLES+L2_PV-1)/L2_PV;
  localparam int L2_M1_WGT_DDR_WORDS=L2_M1_PHYS_WORDS*WGT_SUBWORDS;

  localparam int L3_NUM_FGROUP=(L3_F_OUT+L3_PF-1)/L3_PF;
  localparam int L3_M1_LOGICAL_BUNDLES=L3_NUM_FGROUP*L3_C_IN*L3_K*L3_K;
  localparam int L3_M1_PHYS_WORDS=(L3_M1_LOGICAL_BUNDLES+L3_PV-1)/L3_PV;
  localparam int L3_M1_WGT_DDR_WORDS=L3_M1_PHYS_WORDS*WGT_SUBWORDS;

  localparam int L4_NUM_FGROUP=(L4_F_OUT+L4_PF-1)/L4_PF;
  localparam int L4_M1_LOGICAL_BUNDLES=L4_NUM_FGROUP*L4_C_IN*L4_K*L4_K;
  localparam int L4_M1_PHYS_WORDS=(L4_M1_LOGICAL_BUNDLES+L4_PV-1)/L4_PV;
  localparam int L4_M1_WGT_DDR_WORDS=L4_M1_PHYS_WORDS*WGT_SUBWORDS;

  localparam int L5_NUM_FGROUP=(L5_F_OUT+L5_PF-1)/L5_PF;
  localparam int L5_M1_LOGICAL_BUNDLES=L5_NUM_FGROUP*L5_C_IN*L5_K*L5_K;
  localparam int L5_M1_PHYS_WORDS=(L5_M1_LOGICAL_BUNDLES+L5_PV-1)/L5_PV;
  localparam int L5_M1_WGT_DDR_WORDS=L5_M1_PHYS_WORDS*WGT_SUBWORDS;

  localparam int L6_NUM_FGROUP=(L6_F_OUT+L6_PF-1)/L6_PF;
  localparam int L6_M1_LOGICAL_BUNDLES=L6_NUM_FGROUP*L6_C_IN*L6_K*L6_K;
  localparam int L6_M1_PHYS_WORDS=(L6_M1_LOGICAL_BUNDLES+L6_PV-1)/L6_PV;
  localparam int L6_M1_WGT_DDR_WORDS=L6_M1_PHYS_WORDS*WGT_SUBWORDS;

  localparam int L7_NUM_FGROUP=(L7_F_OUT+L7_PF-1)/L7_PF;
  localparam int L7_M1_LOGICAL_BUNDLES=L7_NUM_FGROUP*L7_C_IN*L7_K*L7_K;
  localparam int L7_M1_PHYS_WORDS=(L7_M1_LOGICAL_BUNDLES+L7_PV-1)/L7_PV;
  localparam int L7_M1_WGT_DDR_WORDS=L7_M1_PHYS_WORDS*WGT_SUBWORDS;

  localparam int L8_NUM_FGROUP=(L8_F_OUT+L8_PF-1)/L8_PF;
  localparam int L8_M1_LOGICAL_BUNDLES=L8_NUM_FGROUP*L8_C_IN*L8_K*L8_K;
  localparam int L8_M1_PHYS_WORDS=(L8_M1_LOGICAL_BUNDLES+L8_PV-1)/L8_PV;
  localparam int L8_M1_WGT_DDR_WORDS=L8_M1_PHYS_WORDS*WGT_SUBWORDS;

  localparam int L0_WGT_DDR_BASE=`DDR_WGT_BASE;
  localparam int L1_WGT_DDR_BASE=L0_WGT_DDR_BASE+L0_M1_WGT_DDR_WORDS;
  localparam int L2_WGT_DDR_BASE=L1_WGT_DDR_BASE+L1_M1_WGT_DDR_WORDS;
  localparam int L3_WGT_DDR_BASE=L2_WGT_DDR_BASE+L2_M1_WGT_DDR_WORDS;
  localparam int L4_WGT_DDR_BASE=L3_WGT_DDR_BASE+L3_M1_WGT_DDR_WORDS;
  localparam int L5_WGT_DDR_BASE=L4_WGT_DDR_BASE+L4_M1_WGT_DDR_WORDS;
  localparam int L6_WGT_DDR_BASE=L5_WGT_DDR_BASE+L5_M1_WGT_DDR_WORDS;
  localparam int L7_WGT_DDR_BASE=L6_WGT_DDR_BASE+L6_M1_WGT_DDR_WORDS;
  localparam int L8_WGT_DDR_BASE=L7_WGT_DDR_BASE+L7_M1_WGT_DDR_WORDS;

  localparam int FINAL_STORE_PACK = 1;
  localparam int FINAL_GROUPS     = (L8_W_OUT + FINAL_STORE_PACK - 1) / FINAL_STORE_PACK;
  localparam int EXP_OFM_WORDS    = L8_F_OUT * L8_H_OUT * FINAL_GROUPS;

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

  assign cfg_num_layers_s = 'd9;

  // Direct DDR interface between cnn_top and AXI bridge.
  logic                      ddr_rd_req_s;
  logic [DDR_ADDR_W-1:0]     ddr_rd_addr_s;
  logic                      ddr_rd_valid_s;
  logic [DDR_WORD_W-1:0]     ddr_rd_data_s;
  logic                      ddr_wr_en_s;
  logic [DDR_ADDR_W-1:0]     ddr_wr_addr_s;
  logic [DDR_WORD_W-1:0]     ddr_wr_data_s;
  logic [(DDR_WORD_W/8)-1:0] ddr_wr_be_s;

  // Same-mode refill hooks use the same tie-offs as the testbench.
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

  function automatic layer_desc_t make_layer_desc(input logic [CFG_AW-1:0] idx);
    layer_desc_t cfg;
    begin
      cfg = '0;
      cfg.mode          = MODE1;
      cfg.pc_m2         = PC_MODE2[7:0];
      cfg.pf_m2         = PF_MODE2[7:0];
      cfg.conv_stride   = 2'd1;
      cfg.pad_top       = 4'd0;
      cfg.pad_bottom    = 4'd0;
      cfg.pad_left      = 4'd0;
      cfg.pad_right     = 4'd0;
      cfg.relu_en       = 1'b1;
      cfg.pool_k        = 2'd2;
      cfg.pool_stride   = 2'd2;
      cfg.ifm_ddr_base  = `DDR_IFM_BASE;
      cfg.ofm_ddr_base  = `DDR_OFM_BASE;
      cfg.first_layer   = 1'b0;
      cfg.last_layer    = 1'b0;

      unique case (idx)
        4'd0: begin
          cfg.layer_id     = 0;
          cfg.h_in         = L0_H_IN[15:0];
          cfg.w_in         = L0_W_IN[15:0];
          cfg.c_in         = L0_C_IN[15:0];
          cfg.f_out        = L0_F_OUT[15:0];
          cfg.k            = L0_K[3:0];
          cfg.h_out        = L0_H_CONV_OUT[15:0];
          cfg.w_out        = L0_W_CONV_OUT[15:0];
          cfg.pv_m1        = L0_PV[7:0];
          cfg.pf_m1        = L0_PF[7:0];
          cfg.pool_en      = (L0_POOL_EN != 0);
          cfg.wgt_ddr_base = L0_WGT_DDR_BASE[DDR_ADDR_W-1:0];
          cfg.first_layer  = 1'b1;
        end
        4'd1: begin
          cfg.layer_id     = 1;
          cfg.h_in         = L1_H_IN[15:0];
          cfg.w_in         = L1_W_IN[15:0];
          cfg.c_in         = L1_C_IN[15:0];
          cfg.f_out        = L1_F_OUT[15:0];
          cfg.k            = L1_K[3:0];
          cfg.h_out        = L1_H_CONV_OUT[15:0];
          cfg.w_out        = L1_W_CONV_OUT[15:0];
          cfg.pv_m1        = L1_PV[7:0];
          cfg.pf_m1        = L1_PF[7:0];
          cfg.pool_en      = (L1_POOL_EN != 0);
          cfg.wgt_ddr_base = L1_WGT_DDR_BASE[DDR_ADDR_W-1:0];
        end
        4'd2: begin
          cfg.layer_id     = 2;
          cfg.h_in         = L2_H_IN[15:0];
          cfg.w_in         = L2_W_IN[15:0];
          cfg.c_in         = L2_C_IN[15:0];
          cfg.f_out        = L2_F_OUT[15:0];
          cfg.k            = L2_K[3:0];
          cfg.h_out        = L2_H_CONV_OUT[15:0];
          cfg.w_out        = L2_W_CONV_OUT[15:0];
          cfg.pv_m1        = L2_PV[7:0];
          cfg.pf_m1        = L2_PF[7:0];
          cfg.pool_en      = (L2_POOL_EN != 0);
          cfg.wgt_ddr_base = L2_WGT_DDR_BASE[DDR_ADDR_W-1:0];
        end
        4'd3: begin
          cfg.layer_id     = 3;
          cfg.h_in         = L3_H_IN[15:0];
          cfg.w_in         = L3_W_IN[15:0];
          cfg.c_in         = L3_C_IN[15:0];
          cfg.f_out        = L3_F_OUT[15:0];
          cfg.k            = L3_K[3:0];
          cfg.h_out        = L3_H_CONV_OUT[15:0];
          cfg.w_out        = L3_W_CONV_OUT[15:0];
          cfg.pv_m1        = L3_PV[7:0];
          cfg.pf_m1        = L3_PF[7:0];
          cfg.pool_en      = (L3_POOL_EN != 0);
          cfg.wgt_ddr_base = L3_WGT_DDR_BASE[DDR_ADDR_W-1:0];
        end
        4'd4: begin
          cfg.layer_id     = 4;
          cfg.h_in         = L4_H_IN[15:0];
          cfg.w_in         = L4_W_IN[15:0];
          cfg.c_in         = L4_C_IN[15:0];
          cfg.f_out        = L4_F_OUT[15:0];
          cfg.k            = L4_K[3:0];
          cfg.h_out        = L4_H_CONV_OUT[15:0];
          cfg.w_out        = L4_W_CONV_OUT[15:0];
          cfg.pv_m1        = L4_PV[7:0];
          cfg.pf_m1        = L4_PF[7:0];
          cfg.pool_en      = (L4_POOL_EN != 0);
          cfg.wgt_ddr_base = L4_WGT_DDR_BASE[DDR_ADDR_W-1:0];
        end
        4'd5: begin
          cfg.layer_id     = 5;
          cfg.h_in         = L5_H_IN[15:0];
          cfg.w_in         = L5_W_IN[15:0];
          cfg.c_in         = L5_C_IN[15:0];
          cfg.f_out        = L5_F_OUT[15:0];
          cfg.k            = L5_K[3:0];
          cfg.h_out        = L5_H_CONV_OUT[15:0];
          cfg.w_out        = L5_W_CONV_OUT[15:0];
          cfg.pv_m1        = L5_PV[7:0];
          cfg.pf_m1        = L5_PF[7:0];
          cfg.pool_en      = (L5_POOL_EN != 0);
          cfg.wgt_ddr_base = L5_WGT_DDR_BASE[DDR_ADDR_W-1:0];
        end
        4'd6: begin
          cfg.layer_id     = 6;
          cfg.h_in         = L6_H_IN[15:0];
          cfg.w_in         = L6_W_IN[15:0];
          cfg.c_in         = L6_C_IN[15:0];
          cfg.f_out        = L6_F_OUT[15:0];
          cfg.k            = L6_K[3:0];
          cfg.h_out        = L6_H_CONV_OUT[15:0];
          cfg.w_out        = L6_W_CONV_OUT[15:0];
          cfg.pv_m1        = L6_PV[7:0];
          cfg.pf_m1        = L6_PF[7:0];
          cfg.pool_en      = (L6_POOL_EN != 0);
          cfg.wgt_ddr_base = L6_WGT_DDR_BASE[DDR_ADDR_W-1:0];
        end
        4'd7: begin
          cfg.layer_id     = 7;
          cfg.h_in         = L7_H_IN[15:0];
          cfg.w_in         = L7_W_IN[15:0];
          cfg.c_in         = L7_C_IN[15:0];
          cfg.f_out        = L7_F_OUT[15:0];
          cfg.k            = L7_K[3:0];
          cfg.h_out        = L7_H_CONV_OUT[15:0];
          cfg.w_out        = L7_W_CONV_OUT[15:0];
          cfg.pv_m1        = L7_PV[7:0];
          cfg.pf_m1        = L7_PF[7:0];
          cfg.pool_en      = (L7_POOL_EN != 0);
          cfg.wgt_ddr_base = L7_WGT_DDR_BASE[DDR_ADDR_W-1:0];
        end
        4'd8: begin
          cfg.layer_id     = 8;
          cfg.h_in         = L8_H_IN[15:0];
          cfg.w_in         = L8_W_IN[15:0];
          cfg.c_in         = L8_C_IN[15:0];
          cfg.f_out        = L8_F_OUT[15:0];
          cfg.k            = L8_K[3:0];
          cfg.h_out        = L8_H_CONV_OUT[15:0];
          cfg.w_out        = L8_W_CONV_OUT[15:0];
          cfg.pv_m1        = L8_PV[7:0];
          cfg.pf_m1        = L8_PF[7:0];
          cfg.pool_en      = (L8_POOL_EN != 0);
          cfg.wgt_ddr_base = L8_WGT_DDR_BASE[DDR_ADDR_W-1:0];
          cfg.last_layer   = 1'b1;
        end
        default: begin
          cfg = '0;
        end
      endcase

      return cfg;
    end
  endfunction

  // --------------------------------------------------------------------------
  // Internal config loader and one-shot run controller.
  // - After reset release, write cfg[0..8].
  // - run is treated as a level from VIO/GPIO. A new run starts once per high
  //   assertion, then run must be lowered before another attempt.
  // --------------------------------------------------------------------------
  logic cfg_done_q;
  logic run_armed_q;
  logic [CFG_AW-1:0] cfg_load_idx_q;

  always_ff @(posedge clk) begin
    if (!rst_core_n) begin
      cfg_done_q     <= 1'b0;
      cfg_load_idx_q <= '0;
      cfg_wr_en_s    <= 1'b0;
      cfg_wr_addr_s  <= '0;
      cfg_wr_data_s  <= '0;
      start_s        <= 1'b0;
      run_armed_q    <= 1'b0;
    end else begin
      cfg_wr_en_s <= 1'b0;
      start_s     <= 1'b0;

      if (!cfg_done_q) begin
        cfg_wr_en_s   <= 1'b1;
        cfg_wr_addr_s <= cfg_load_idx_q;
        cfg_wr_data_s <= make_layer_desc(cfg_load_idx_q);

        if (cfg_load_idx_q == 4'd8) begin
          cfg_done_q <= 1'b1;
        end else begin
          cfg_load_idx_q <= cfg_load_idx_q + 1'b1;
        end
      end

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

  assign busy  = core_busy | bridge_busy;
  assign done  = core_done;
  assign error = core_error | bridge_error | wr_fifo_overflow;

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
