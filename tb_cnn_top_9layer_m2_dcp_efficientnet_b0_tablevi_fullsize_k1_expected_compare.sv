`timescale 1ns/1ps
`include "cnn_ddr_defs.svh"

module tb_cnn_top_9layer_m2_testA_64x96x32_k1_multitile_expected_compare;
  import cnn_layer_desc_pkg::*;

  // --------------------------------------------------------------------------
  // 9-layer Mode-2 DCP/EfficientNet-B0-style Test-A large-shape test.
  //
  // Purpose:
  //   This is the Test-A companion to the W32 Mode-2 regression.
  //   It uses a Mode-2-friendly initial IFM shape, 64x96x32, to stress
  //   W > PC multi-horizontal-tile scheduling without forcing the unrealistic
  //   224x224x3 first image directly into Mode 2.
  //
  // Important scope note:
  //   K is intentionally kept at 1 for all layers. This test validates the
  //   corrected Mode-2 WT=PC IFM layout and multi-horizontal-tile handling
  //   for W=96 without mixing in K>1 cross-tile/halo behavior. A later K=3
  //   test should be added after the Mode-2 halo/window-crossing contract is
  //   implemented and verified.
  //
  // Data model:
  //   First-layer IFM is tile-coded, not uniform:
  //     IFM[row][tile_x*PC + col_l][valid channel] = tile_x + 1.
  //   All weights are 1. Expected final OFM is still checked exactly.
  //   With K=1 and ReLU/signed-8 saturation, L2 onward saturates to 127,
  //   while an additional L0->L1 stream checker verifies that different
  //   horizontal tiles are actually consumed, not silently aliased to tile 0.
  //
  // This stresses:
  //   - Mode-2 first-layer DDR->IFM preload over 3 horizontal tiles
  //   - W=96 with WT=PC=32, i.e. ceil(96/32)=3 col tiles
  //   - Mode2->Mode2 deterministic OFM->IFM handoff over 8 transitions
  //   - partial C groups, especially C=16,24,40,80 with PC=32
  //   - partial F groups for F=16,24,40,80,192 with PF=64
  //   - L0->L1 stream data differs by horizontal tile
  //   - final DDR OFM compare
  // --------------------------------------------------------------------------

  localparam int DATA_W = 8;
  localparam int PSUM_W = 32;

  // DCP-CNN Table VI Mode-2 point: Pc=32, Pf=64 => PTOTAL=2048.
  localparam int PC = 32;
  localparam int PF = 64;
  localparam int PTOTAL = PC * PF;

  // Keep project-wide runtime maxima consistent with the current contract.
  // DDR word width is PV_MAX*DATA_W; weight DMA assembles one PTOTAL word from
  // WGT_SUBWORDS DDR words when PTOTAL > PV_MAX.
  localparam int PV_MAX = 128;
  localparam int PF_MAX = 128;

  localparam int C_MAX = 192;
  localparam int F_MAX = 192;
  localparam int W_MAX = 96;
  localparam int H_MAX = 64;
  localparam int HT = 4;
  localparam int K_MAX = 3;

  localparam int WGT_DEPTH = 512;
  // OFM storage in this all-Mode2 K1 test packs spatial columns in PC-wide words.
  // Do not use W_MAX as the physical row stride here: W_MAX=96 would make
  // DEPTH=64*ceil(96/32) per bank and can crash XSim with huge unpacked arrays.
  // The required physical groups per row are ceil(W_MAX/PC)=3.
  localparam int OFM_ROW_STRIDE = (W_MAX + PC - 1) / PC;
  localparam int OFM_BANK_DEPTH = H_MAX * OFM_ROW_STRIDE;
  localparam int OFM_LINEAR_DEPTH = C_MAX * OFM_BANK_DEPTH;
  localparam int CFG_DEPTH = 16;

  localparam int DDR_ADDR_W = `CNN_DDR_ADDR_W;
  localparam int DDR_WORD_W = PV_MAX * DATA_W;
  localparam int DDR_LANES = PV_MAX;
  localparam int WGT_SUBWORDS = (PTOTAL + PV_MAX - 1) / PV_MAX;
  localparam int MEM_DEPTH = (`DDR_RSVD_BASE + `DDR_RSVD_SIZE);
  localparam int CLK_PERIOD_NS = 10;
  localparam int MAX_CYCLES = 50000000;

  // 9-layer EfficientNet-B0 channel trend used in the Mode-1 Table-VI test.
  // K is set to 1 here so the Test-A W96 regression can validate
  // multi-tile Mode-2 storage without mixing in K>1 halo behavior yet.
  localparam int L0_H_IN=64, L0_W_IN=96, L0_C_IN=32, L0_F_OUT=32, L0_K=1, L0_POOL_EN=1;
  localparam int L0_H_CONV_OUT=L0_H_IN-L0_K+1, L0_W_CONV_OUT=L0_W_IN-L0_K+1;
  localparam int L0_H_OUT=(L0_POOL_EN ? (L0_H_CONV_OUT/2) : L0_H_CONV_OUT);
  localparam int L0_W_OUT=(L0_POOL_EN ? (L0_W_CONV_OUT/2) : L0_W_CONV_OUT);

  localparam int L1_H_IN=L0_H_OUT, L1_W_IN=L0_W_OUT, L1_C_IN=L0_F_OUT, L1_F_OUT=16, L1_K=1, L1_POOL_EN=0;
  localparam int L1_H_CONV_OUT=L1_H_IN-L1_K+1, L1_W_CONV_OUT=L1_W_IN-L1_K+1;
  localparam int L1_H_OUT=(L1_POOL_EN ? (L1_H_CONV_OUT/2) : L1_H_CONV_OUT);
  localparam int L1_W_OUT=(L1_POOL_EN ? (L1_W_CONV_OUT/2) : L1_W_CONV_OUT);

  localparam int L2_H_IN=L1_H_OUT, L2_W_IN=L1_W_OUT, L2_C_IN=L1_F_OUT, L2_F_OUT=24, L2_K=1, L2_POOL_EN=1;
  localparam int L2_H_CONV_OUT=L2_H_IN-L2_K+1, L2_W_CONV_OUT=L2_W_IN-L2_K+1;
  localparam int L2_H_OUT=(L2_POOL_EN ? (L2_H_CONV_OUT/2) : L2_H_CONV_OUT);
  localparam int L2_W_OUT=(L2_POOL_EN ? (L2_W_CONV_OUT/2) : L2_W_CONV_OUT);

  localparam int L3_H_IN=L2_H_OUT, L3_W_IN=L2_W_OUT, L3_C_IN=L2_F_OUT, L3_F_OUT=24, L3_K=1, L3_POOL_EN=0;
  localparam int L3_H_CONV_OUT=L3_H_IN-L3_K+1, L3_W_CONV_OUT=L3_W_IN-L3_K+1;
  localparam int L3_H_OUT=(L3_POOL_EN ? (L3_H_CONV_OUT/2) : L3_H_CONV_OUT);
  localparam int L3_W_OUT=(L3_POOL_EN ? (L3_W_CONV_OUT/2) : L3_W_CONV_OUT);

  localparam int L4_H_IN=L3_H_OUT, L4_W_IN=L3_W_OUT, L4_C_IN=L3_F_OUT, L4_F_OUT=40, L4_K=1, L4_POOL_EN=1;
  localparam int L4_H_CONV_OUT=L4_H_IN-L4_K+1, L4_W_CONV_OUT=L4_W_IN-L4_K+1;
  localparam int L4_H_OUT=(L4_POOL_EN ? (L4_H_CONV_OUT/2) : L4_H_CONV_OUT);
  localparam int L4_W_OUT=(L4_POOL_EN ? (L4_W_CONV_OUT/2) : L4_W_CONV_OUT);

  localparam int L5_H_IN=L4_H_OUT, L5_W_IN=L4_W_OUT, L5_C_IN=L4_F_OUT, L5_F_OUT=40, L5_K=1, L5_POOL_EN=0;
  localparam int L5_H_CONV_OUT=L5_H_IN-L5_K+1, L5_W_CONV_OUT=L5_W_IN-L5_K+1;
  localparam int L5_H_OUT=(L5_POOL_EN ? (L5_H_CONV_OUT/2) : L5_H_CONV_OUT);
  localparam int L5_W_OUT=(L5_POOL_EN ? (L5_W_CONV_OUT/2) : L5_W_CONV_OUT);

  localparam int L6_H_IN=L5_H_OUT, L6_W_IN=L5_W_OUT, L6_C_IN=L5_F_OUT, L6_F_OUT=80, L6_K=1, L6_POOL_EN=1;
  localparam int L6_H_CONV_OUT=L6_H_IN-L6_K+1, L6_W_CONV_OUT=L6_W_IN-L6_K+1;
  localparam int L6_H_OUT=(L6_POOL_EN ? (L6_H_CONV_OUT/2) : L6_H_CONV_OUT);
  localparam int L6_W_OUT=(L6_POOL_EN ? (L6_W_CONV_OUT/2) : L6_W_CONV_OUT);

  localparam int L7_H_IN=L6_H_OUT, L7_W_IN=L6_W_OUT, L7_C_IN=L6_F_OUT, L7_F_OUT=80, L7_K=1, L7_POOL_EN=0;
  localparam int L7_H_CONV_OUT=L7_H_IN-L7_K+1, L7_W_CONV_OUT=L7_W_IN-L7_K+1;
  localparam int L7_H_OUT=(L7_POOL_EN ? (L7_H_CONV_OUT/2) : L7_H_CONV_OUT);
  localparam int L7_W_OUT=(L7_POOL_EN ? (L7_W_CONV_OUT/2) : L7_W_CONV_OUT);

  localparam int L8_H_IN=L7_H_OUT, L8_W_IN=L7_W_OUT, L8_C_IN=L7_F_OUT, L8_F_OUT=192, L8_K=1, L8_POOL_EN=0;
  localparam int L8_H_CONV_OUT=L8_H_IN-L8_K+1, L8_W_CONV_OUT=L8_W_IN-L8_K+1;
  localparam int L8_H_OUT=(L8_POOL_EN ? (L8_H_CONV_OUT/2) : L8_H_CONV_OUT);
  localparam int L8_W_OUT=(L8_POOL_EN ? (L8_W_CONV_OUT/2) : L8_W_CONV_OUT);

  localparam int L0_NUM_CGROUP=(L0_C_IN+PC-1)/PC, L0_NUM_FGROUP=(L0_F_OUT+PF-1)/PF;
  localparam int L1_NUM_CGROUP=(L1_C_IN+PC-1)/PC, L1_NUM_FGROUP=(L1_F_OUT+PF-1)/PF;
  localparam int L2_NUM_CGROUP=(L2_C_IN+PC-1)/PC, L2_NUM_FGROUP=(L2_F_OUT+PF-1)/PF;
  localparam int L3_NUM_CGROUP=(L3_C_IN+PC-1)/PC, L3_NUM_FGROUP=(L3_F_OUT+PF-1)/PF;
  localparam int L4_NUM_CGROUP=(L4_C_IN+PC-1)/PC, L4_NUM_FGROUP=(L4_F_OUT+PF-1)/PF;
  localparam int L5_NUM_CGROUP=(L5_C_IN+PC-1)/PC, L5_NUM_FGROUP=(L5_F_OUT+PF-1)/PF;
  localparam int L6_NUM_CGROUP=(L6_C_IN+PC-1)/PC, L6_NUM_FGROUP=(L6_F_OUT+PF-1)/PF;
  localparam int L7_NUM_CGROUP=(L7_C_IN+PC-1)/PC, L7_NUM_FGROUP=(L7_F_OUT+PF-1)/PF;
  localparam int L8_NUM_CGROUP=(L8_C_IN+PC-1)/PC, L8_NUM_FGROUP=(L8_F_OUT+PF-1)/PF;

  localparam int L0_WGT_WORDS=L0_NUM_FGROUP*L0_NUM_CGROUP*L0_K*L0_K;
  localparam int L1_WGT_WORDS=L1_NUM_FGROUP*L1_NUM_CGROUP*L1_K*L1_K;
  localparam int L2_WGT_WORDS=L2_NUM_FGROUP*L2_NUM_CGROUP*L2_K*L2_K;
  localparam int L3_WGT_WORDS=L3_NUM_FGROUP*L3_NUM_CGROUP*L3_K*L3_K;
  localparam int L4_WGT_WORDS=L4_NUM_FGROUP*L4_NUM_CGROUP*L4_K*L4_K;
  localparam int L5_WGT_WORDS=L5_NUM_FGROUP*L5_NUM_CGROUP*L5_K*L5_K;
  localparam int L6_WGT_WORDS=L6_NUM_FGROUP*L6_NUM_CGROUP*L6_K*L6_K;
  localparam int L7_WGT_WORDS=L7_NUM_FGROUP*L7_NUM_CGROUP*L7_K*L7_K;
  localparam int L8_WGT_WORDS=L8_NUM_FGROUP*L8_NUM_CGROUP*L8_K*L8_K;

  localparam int L0_WGT_DDR_WORDS=L0_WGT_WORDS*WGT_SUBWORDS;
  localparam int L1_WGT_DDR_WORDS=L1_WGT_WORDS*WGT_SUBWORDS;
  localparam int L2_WGT_DDR_WORDS=L2_WGT_WORDS*WGT_SUBWORDS;
  localparam int L3_WGT_DDR_WORDS=L3_WGT_WORDS*WGT_SUBWORDS;
  localparam int L4_WGT_DDR_WORDS=L4_WGT_WORDS*WGT_SUBWORDS;
  localparam int L5_WGT_DDR_WORDS=L5_WGT_WORDS*WGT_SUBWORDS;
  localparam int L6_WGT_DDR_WORDS=L6_WGT_WORDS*WGT_SUBWORDS;
  localparam int L7_WGT_DDR_WORDS=L7_WGT_WORDS*WGT_SUBWORDS;
  localparam int L8_WGT_DDR_WORDS=L8_WGT_WORDS*WGT_SUBWORDS;

  localparam int L0_WGT_DDR_BASE=`DDR_WGT_BASE;
  localparam int L1_WGT_DDR_BASE=L0_WGT_DDR_BASE+L0_WGT_DDR_WORDS;
  localparam int L2_WGT_DDR_BASE=L1_WGT_DDR_BASE+L1_WGT_DDR_WORDS;
  localparam int L3_WGT_DDR_BASE=L2_WGT_DDR_BASE+L2_WGT_DDR_WORDS;
  localparam int L4_WGT_DDR_BASE=L3_WGT_DDR_BASE+L3_WGT_DDR_WORDS;
  localparam int L5_WGT_DDR_BASE=L4_WGT_DDR_BASE+L4_WGT_DDR_WORDS;
  localparam int L6_WGT_DDR_BASE=L5_WGT_DDR_BASE+L5_WGT_DDR_WORDS;
  localparam int L7_WGT_DDR_BASE=L6_WGT_DDR_BASE+L6_WGT_DDR_WORDS;
  localparam int L8_WGT_DDR_BASE=L7_WGT_DDR_BASE+L7_WGT_DDR_WORDS;

  localparam int L0_COLBLKS=(L0_W_IN+PC-1)/PC;
  localparam int L1_COLBLKS=(L1_W_IN+PC-1)/PC;
  localparam int L2_COLBLKS=(L2_W_IN+PC-1)/PC;
  localparam int L3_COLBLKS=(L3_W_IN+PC-1)/PC;
  localparam int L4_COLBLKS=(L4_W_IN+PC-1)/PC;
  localparam int L5_COLBLKS=(L5_W_IN+PC-1)/PC;
  localparam int L6_COLBLKS=(L6_W_IN+PC-1)/PC;
  localparam int L7_COLBLKS=(L7_W_IN+PC-1)/PC;
  localparam int L8_COLBLKS=(L8_W_IN+PC-1)/PC;

  localparam int EXP_OFM2IFM_STREAMS =
      (L1_H_IN*L1_COLBLKS*L1_NUM_CGROUP) +
      (L2_H_IN*L2_COLBLKS*L2_NUM_CGROUP) +
      (L3_H_IN*L3_COLBLKS*L3_NUM_CGROUP) +
      (L4_H_IN*L4_COLBLKS*L4_NUM_CGROUP) +
      (L5_H_IN*L5_COLBLKS*L5_NUM_CGROUP) +
      (L6_H_IN*L6_COLBLKS*L6_NUM_CGROUP) +
      (L7_H_IN*L7_COLBLKS*L7_NUM_CGROUP) +
      (L8_H_IN*L8_COLBLKS*L8_NUM_CGROUP);

  // L0->L1 Mode-2 stream writes one packed PC-lane word per output channel
  // bank, row and horizontal tile.  The lanes inside each word are spatial
  // columns and must be checked lane-by-lane.
  localparam int EXPECTED_L0_STREAM_WORDS = L1_H_IN * L1_COLBLKS * L1_C_IN;

  // Corrected Mode-2 IFM DDR preload layout:
  //   [tile_x][row][cgrp][col_l], lanes = PC channels.
  // For first-layer DDR->IFM, one tile contains H * ceil(C/PC) * PC words.
  localparam int EXPECTED_IFM_DDR_READS = L0_COLBLKS * L0_H_IN * L0_NUM_CGROUP * PC;
  localparam int EXPECTED_WGT_DDR_READS = L0_WGT_DDR_WORDS+L1_WGT_DDR_WORDS+L2_WGT_DDR_WORDS+L3_WGT_DDR_WORDS+L4_WGT_DDR_WORDS+L5_WGT_DDR_WORDS+L6_WGT_DDR_WORDS+L7_WGT_DDR_WORDS+L8_WGT_DDR_WORDS;

  // OFM buffer Mode-2 final DMA layout is NOT flat H*W*F words.
  // It stores one word per {output channel, row, spatial column group}.
  // In Mode 2, store_pack = PC, so groups per row = ceil(W_out / PC).
  localparam int L8_STORED_GROUPS = (L8_W_OUT + PC - 1) / PC;
  localparam int EXPECTED_FINAL_ELEMENTS = L8_F_OUT * L8_H_OUT * L8_W_OUT;
  localparam int EXPECTED_OFM_DDR_WORDS = L8_F_OUT * L8_H_OUT * L8_STORED_GROUPS;

  // Derived from this testbench, not assumed:
  //   init_mem() writes first-layer IFM with tile pattern tile_x+1 and all
  //   layer weights=1. K=1 for every layer.
  //   L0/L1 stream values vary by tile, and L2 onward saturates to 127.
  localparam int EXPECTED_FINAL_VALUE = 127;

  logic clk, rst_n, start, abort;
  logic cfg_wr_en;
  logic [$clog2(CFG_DEPTH)-1:0] cfg_wr_addr;
  layer_desc_t cfg_wr_data;
  logic [$clog2(CFG_DEPTH+1)-1:0] cfg_num_layers;

  logic ddr_rd_req, ddr_rd_valid, ddr_wr_en;
  logic [DDR_ADDR_W-1:0] ddr_rd_addr, ddr_wr_addr;
  logic [DDR_WORD_W-1:0] ddr_rd_data, ddr_wr_data;
  logic [(DDR_WORD_W/8)-1:0] ddr_wr_be;

  logic [15:0] m1_free_col_blk_g, m1_free_ch_blk_g;
  logic m1_sm_refill_req_ready, m1_sm_refill_req_valid;
  logic [$clog2(HT)-1:0] m1_sm_refill_row_slot_l;
  logic [15:0] m1_sm_refill_row_g, m1_sm_refill_col_blk_g, m1_sm_refill_ch_blk_g;
  logic m2_sm_refill_req_ready, m2_sm_refill_req_valid;
  logic [15:0] m2_sm_refill_row_g, m2_sm_refill_col_g, m2_sm_refill_col_l, m2_sm_refill_cgrp_g;
  logic ifm_m1_free_valid;
  logic [$clog2(HT)-1:0] ifm_m1_free_row_slot_l;
  logic [15:0] ifm_m1_free_row_g;

  logic busy, done, error;
  logic [$clog2(CFG_DEPTH)-1:0] dbg_layer_idx;
  logic dbg_mode, dbg_weight_bank;
  logic [3:0] dbg_error_vec;

  logic [DDR_WORD_W-1:0] ddr_mem [0:MEM_DEPTH-1];
  logic rd_pending_q;
  logic [DDR_ADDR_W-1:0] rd_addr_q;

  integer cycle_count;
  integer ddr_ifm_read_count;
  integer ddr_wgt_read_count;
  integer ddr_ofm_write_count;
  integer ofm_ifm_stream_start_count;
  integer ofm_ifm_stream_done_count;
  integer m2_sm_refill_req_count;
  integer l0_stream_checked_count;
  integer l0_stream_mismatch_count;

  // Latched command context for OFM->IFM stream checking.
  // The stream command ports are only guaranteed to be meaningful on
  // ofm_ifm_stream_start_s; during the later IFM write beats, the live
  // col_base port may already have returned to its default value.
  logic        tb_l0_stream_cmd_valid_q;
  logic [15:0] tb_l0_stream_col_base_q;
  logic [15:0] tb_l0_stream_row_base_q;
  logic [15:0] tb_l0_stream_cgrp_q;

  integer i, b;

  logic done_seen;
  logic error_seen;
  logic [3:0] first_error_vec;
  logic [$clog2(CFG_DEPTH)-1:0] first_error_layer;
  logic first_error_mode;
  integer first_error_cycle;

  cnn_top #(
    .DATA_W(DATA_W), .PSUM_W(PSUM_W), .PTOTAL(PTOTAL),
    .PV_MAX(PV_MAX), .PF_MAX(PF_MAX), .PC_MODE2(PC), .PF_MODE2(PF),
    .C_MAX(C_MAX), .F_MAX(F_MAX), .W_MAX(W_MAX), .H_MAX(H_MAX), .HT(HT), .K_MAX(K_MAX),
    .WGT_DEPTH(WGT_DEPTH), .OFM_BANK_DEPTH(OFM_BANK_DEPTH), .OFM_LINEAR_DEPTH(OFM_LINEAR_DEPTH),
    .CFG_DEPTH(CFG_DEPTH), .DDR_ADDR_W(DDR_ADDR_W), .DDR_WORD_W(DDR_WORD_W)
  ) dut (
    .clk(clk), .rst_n(rst_n), .start(start), .abort(abort),
    .cfg_wr_en(cfg_wr_en), .cfg_wr_addr(cfg_wr_addr), .cfg_wr_data(cfg_wr_data), .cfg_num_layers(cfg_num_layers),
    .ddr_rd_req(ddr_rd_req), .ddr_rd_addr(ddr_rd_addr), .ddr_rd_valid(ddr_rd_valid), .ddr_rd_data(ddr_rd_data),
    .ddr_wr_en(ddr_wr_en), .ddr_wr_addr(ddr_wr_addr), .ddr_wr_data(ddr_wr_data), .ddr_wr_be(ddr_wr_be),
    .m1_free_col_blk_g(m1_free_col_blk_g), .m1_free_ch_blk_g(m1_free_ch_blk_g),
    .m1_sm_refill_req_ready(m1_sm_refill_req_ready), .m1_sm_refill_req_valid(m1_sm_refill_req_valid),
    .m1_sm_refill_row_slot_l(m1_sm_refill_row_slot_l), .m1_sm_refill_row_g(m1_sm_refill_row_g),
    .m1_sm_refill_col_blk_g(m1_sm_refill_col_blk_g), .m1_sm_refill_ch_blk_g(m1_sm_refill_ch_blk_g),
    .m2_sm_refill_req_ready(m2_sm_refill_req_ready), .m2_sm_refill_req_valid(m2_sm_refill_req_valid),
    .m2_sm_refill_row_g(m2_sm_refill_row_g), .m2_sm_refill_col_g(m2_sm_refill_col_g),
    .m2_sm_refill_col_l(m2_sm_refill_col_l), .m2_sm_refill_cgrp_g(m2_sm_refill_cgrp_g),
    .ifm_m1_free_valid(ifm_m1_free_valid), .ifm_m1_free_row_slot_l(ifm_m1_free_row_slot_l), .ifm_m1_free_row_g(ifm_m1_free_row_g),
    .busy(busy), .done(done), .error(error),
    .dbg_layer_idx(dbg_layer_idx), .dbg_mode(dbg_mode), .dbg_weight_bank(dbg_weight_bank), .dbg_error_vec(dbg_error_vec)
  );

  initial clk = 1'b0;
  always #(CLK_PERIOD_NS/2) clk = ~clk;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      ddr_rd_valid <= 1'b0;
      ddr_rd_data <= '0;
      rd_pending_q <= 1'b0;
      rd_addr_q <= '0;
      cycle_count <= 0;
      ddr_ifm_read_count <= 0;
      ddr_wgt_read_count <= 0;
      ddr_ofm_write_count <= 0;
      ofm_ifm_stream_start_count <= 0;
      ofm_ifm_stream_done_count <= 0;
      m2_sm_refill_req_count <= 0;
      l0_stream_checked_count <= 0;
      l0_stream_mismatch_count <= 0;
      tb_l0_stream_cmd_valid_q <= 1'b0;
      tb_l0_stream_col_base_q  <= 16'd0;
      tb_l0_stream_row_base_q  <= 16'd0;
      tb_l0_stream_cgrp_q      <= 16'd0;
      done_seen <= 1'b0;
      error_seen <= 1'b0;
      first_error_vec <= '0;
      first_error_layer <= '0;
      first_error_mode <= 1'b0;
      first_error_cycle <= 0;
    end else begin
      cycle_count <= cycle_count + 1;
      ddr_rd_valid <= 1'b0;

      if (done) done_seen <= 1'b1;

      if (error && !error_seen) begin
        error_seen <= 1'b1;
        first_error_vec <= dbg_error_vec;
        first_error_layer <= dbg_layer_idx;
        first_error_mode <= dbg_mode;
        first_error_cycle <= cycle_count;
        $display("DBG_FIRST_ERROR t=%0t cycle=%0d dbg_error_vec=%04b layer=%0d mode=%0d busy=%0b done=%0b error=%0b", $time, cycle_count, dbg_error_vec, dbg_layer_idx, dbg_mode, busy, done, error);
        $display("DBG_ERROR_MAP bit0=dma_error bit1=ofm_error bit2=local_error bit3=transition_error");
      end

      if (rd_pending_q) begin
        ddr_rd_valid <= 1'b1;
        if (rd_addr_q < MEM_DEPTH) ddr_rd_data <= ddr_mem[rd_addr_q];
        else ddr_rd_data <= '0;
      end

      if (ddr_wr_en) begin
        if (ddr_wr_addr < MEM_DEPTH) begin
          for (b = 0; b < (DDR_WORD_W/8); b = b + 1) begin
            if (ddr_wr_be[b]) ddr_mem[ddr_wr_addr][8*b +: 8] <= ddr_wr_data[8*b +: 8];
          end
        end
        if ((ddr_wr_addr >= `DDR_OFM_BASE) && (ddr_wr_addr < (`DDR_OFM_BASE + `DDR_OFM_SIZE))) begin
          ddr_ofm_write_count <= ddr_ofm_write_count + 1;
        end
      end

      if (ddr_rd_req) begin
        rd_pending_q <= 1'b1;
        rd_addr_q <= ddr_rd_addr;
        if ((ddr_rd_addr >= `DDR_IFM_BASE) && (ddr_rd_addr < (`DDR_IFM_BASE + `DDR_IFM_SIZE))) ddr_ifm_read_count <= ddr_ifm_read_count + 1;
        if ((ddr_rd_addr >= `DDR_WGT_BASE) && (ddr_rd_addr < (`DDR_WGT_BASE + `DDR_WGT_SIZE))) ddr_wgt_read_count <= ddr_wgt_read_count + 1;
      end else if (rd_pending_q) begin
        rd_pending_q <= 1'b0;
      end

      if (dut.ofm_ifm_stream_start_s) begin
        ofm_ifm_stream_start_count <= ofm_ifm_stream_start_count + 1;

        // Latch stream command context.  Do not use ofm_ifm_stream_col_base_s
        // live in the data-beat checker because control may deassert/clear it
        // after the start pulse, while the corresponding IFM write arrives
        // later.
        tb_l0_stream_cmd_valid_q <= 1'b1;
        tb_l0_stream_col_base_q  <= dut.ofm_ifm_stream_col_base_s;
        tb_l0_stream_row_base_q  <= dut.ofm_ifm_stream_row_base_s;
        tb_l0_stream_cgrp_q      <= dut.ofm_ifm_stream_m2_cgrp_g_s;
      end
      if (dut.ofm_ifm_stream_done_s) ofm_ifm_stream_done_count <= ofm_ifm_stream_done_count + 1;
      if (m2_sm_refill_req_valid && m2_sm_refill_req_ready) m2_sm_refill_req_count <= m2_sm_refill_req_count + 1;
    end
  end

  always_ff @(posedge clk) begin
    if (rst_n) begin
      if (done || error || ((cycle_count % 50000) == 0)) begin
        $display("DBG_TOP_STATUS t=%0t cycle=%0d busy=%0b done=%0b error=%0b vec=%04b layer=%0d mode=%0d ifm_rd=%0d wgt_rd=%0d ofm_wr=%0d stream_start=%0d stream_done=%0d legacy_m2_req=%0d", $time, cycle_count, busy, done, error, dbg_error_vec, dbg_layer_idx, dbg_mode, ddr_ifm_read_count, ddr_wgt_read_count, ddr_ofm_write_count, ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, m2_sm_refill_req_count);
      end
    end
  end

  always_ff @(posedge clk) begin
    if (rst_n) begin
      // Check only the first L0->L1 OFM->IFM stream words.  After the
      // control fixes, L0->L1 may continue after dbg_layer_idx has already
      // advanced to layer 1, so this checker must not be gated by
      // dbg_layer_idx == 0.
      if (dut.ifm_ofm_wr_en_s &&
          dut.ifm_ofm_wr_ready_s &&
          tb_l0_stream_cmd_valid_q &&
          (l0_stream_checked_count < EXPECTED_L0_STREAM_WORDS)) begin
        int col_base;
        int lane;
        int lane_col;
        int bad_lane;
        int bad_col;
        logic word_bad;
        logic expected_keep_lane;
        logic [PC-1:0] expected_keep_pc;
        logic [DATA_W-1:0] got_bad;
        logic [DATA_W-1:0] exp_bad;

        // Use the latched stream command context, not the live command
        // port.  Tile-1 data beats may arrive while live col_base has already
        // returned to 0, which made the old checker expect tile-0 values and
        // full keep=ffffffff for the partial final tile.
        col_base = int'(tb_l0_stream_col_base_q);
        bad_lane = -1;
        bad_col  = -1;
        word_bad = 1'b0;
        expected_keep_pc = '0;
        got_bad = '0;
        exp_bad = '0;

        // Structural checks for the packed Mode-2 L0->L1 word.
        // bank = source/output channel, lanes = spatial columns.
        if ((dut.ifm_ofm_wr_row_idx_s >= L1_H_IN) ||
            (dut.ifm_ofm_wr_bank_s >= L1_C_IN) ||
            (dut.ifm_ofm_wr_col_idx_s != 0) ||
            (col_base >= L1_W_IN)) begin
          word_bad = 1'b1;
        end

        // Value check: each active lane has its own global output column and
        // therefore its own expected tile-pattern value.  The previous checker
        // compared got0/got1/got31 against one scalar expected value, which is
        // wrong for PC-packed Mode-2 words spanning multiple pooled columns.
        for (lane = 0; lane < PC; lane = lane + 1) begin
          lane_col = col_base + lane;
          expected_keep_lane = (lane_col < L1_W_IN);
          expected_keep_pc[lane] = expected_keep_lane;

          if (expected_keep_lane) begin
            if (!dut.ifm_ofm_wr_keep_s[lane] ||
                (dut.ifm_ofm_wr_data_s[lane*DATA_W +: DATA_W] !== expected_l0_stream_value(lane_col))) begin
              if (!word_bad) begin
                bad_lane = lane;
                bad_col  = lane_col;
                got_bad  = dut.ifm_ofm_wr_data_s[lane*DATA_W +: DATA_W];
                exp_bad  = expected_l0_stream_value(lane_col);
              end
              word_bad = 1'b1;
            end
          end
          else begin
            if (dut.ifm_ofm_wr_keep_s[lane]) begin
              if (!word_bad) begin
                bad_lane = lane;
                bad_col  = lane_col;
                got_bad  = dut.ifm_ofm_wr_data_s[lane*DATA_W +: DATA_W];
                exp_bad  = '0;
              end
              word_bad = 1'b1;
            end
          end
        end

        l0_stream_checked_count <= l0_stream_checked_count + 1;

        if (word_bad) begin
          l0_stream_mismatch_count <= l0_stream_mismatch_count + 1;
          if (l0_stream_mismatch_count < 16) begin
            $display("TB_MISMATCH_L0_M2_STREAM t=%0t cycle=%0d word=%0d row=%0d ch_bank=%0d col_base=%0d live_col_base=%0d cmd_row_base=%0d cmd_cgrp=%0d bad_lane=%0d bad_col=%0d got_bad=%0d exp_bad=%0d got0=%0d got1=%0d got15=%0d got16=%0d got31=%0d keep=%h exp_keep=%h",
              $time,
              cycle_count,
              l0_stream_checked_count,
              dut.ifm_ofm_wr_row_idx_s,
              dut.ifm_ofm_wr_bank_s,
              col_base,
              dut.ofm_ifm_stream_col_base_s,
              tb_l0_stream_row_base_q,
              tb_l0_stream_cgrp_q,
              bad_lane,
              bad_col,
              $signed(got_bad),
              $signed(exp_bad),
              $signed(dut.ifm_ofm_wr_data_s[0*DATA_W +: DATA_W]),
              $signed(dut.ifm_ofm_wr_data_s[1*DATA_W +: DATA_W]),
              $signed(dut.ifm_ofm_wr_data_s[15*DATA_W +: DATA_W]),
              $signed(dut.ifm_ofm_wr_data_s[16*DATA_W +: DATA_W]),
              $signed(dut.ifm_ofm_wr_data_s[31*DATA_W +: DATA_W]),
              dut.ifm_ofm_wr_keep_s[31:0],
              expected_keep_pc
            );
          end
        end
      end
    end
  end


  function automatic logic [DDR_WORD_W-1:0] pack_ddr_ones_word;
    logic [DDR_WORD_W-1:0] word;
    begin
      word = '0;
      for (int lane = 0; lane < DDR_LANES; lane++) begin
        word[lane*DATA_W +: DATA_W] = 8'sd1;
      end
      return word;
    end
  endfunction

  function automatic logic [DDR_WORD_W-1:0] pack_m2_ifm_tile_word;
    input int tile_x;
    input int cgrp;
    input int col_l;
    logic [DDR_WORD_W-1:0] word;
    int ch;
    logic [DATA_W-1:0] val;
    begin
      word = '0;
      val = tile_x + 1;
      for (int lane = 0; lane < DDR_LANES; lane++) begin
        ch = cgrp * PC + lane;
        if ((lane < PC) && (ch < L0_C_IN) && ((tile_x * PC + col_l) < L0_W_IN)) begin
          word[lane*DATA_W +: DATA_W] = val;
        end
      end
      return word;
    end
  endfunction

  function automatic logic [DATA_W-1:0] expected_l0_stream_value;
    input int out_col;
    int src_input_col;
    int src_tile;
    int val_i;
    begin
      // L0 uses K=1 and pool_stride=2. Pool output column out_col consumes
      // input columns {2*out_col, 2*out_col+1}. PC-aligned tiles make both
      // columns belong to the same tile for this W96/PC32 regression.
      src_input_col = out_col * 2;
      src_tile = src_input_col / PC;
      val_i = (src_tile + 1) * L0_C_IN;
      if (val_i > 127) val_i = 127;
      return val_i;
    end
  endfunction

  function automatic logic [DDR_WORD_W-1:0] expected_final_word;
    input int word_idx;
    logic [DDR_WORD_W-1:0] word;
    logic [DATA_W-1:0] exp_val;
    int words_per_ch;
    int ch;
    int rem;
    int row;
    int grp;
    int col_g;
    begin
      word = '0;
      exp_val = EXPECTED_FINAL_VALUE;

      // DMA read order from ofm_buffer for Mode-2 stored data:
      //   word_idx -> channel -> row -> column group
      words_per_ch = L8_H_OUT * L8_STORED_GROUPS;
      ch  = word_idx / words_per_ch;
      rem = word_idx % words_per_ch;
      row = rem / L8_STORED_GROUPS;
      grp = rem % L8_STORED_GROUPS;

      for (int lane = 0; lane < DDR_LANES; lane++) begin
        col_g = grp * PC + lane;
        if ((ch < L8_F_OUT) &&
            (row < L8_H_OUT) &&
            (lane < PC) &&
            (col_g < L8_W_OUT)) begin
          word[lane*DATA_W +: DATA_W] = exp_val;
        end
      end

      return word;
    end
  endfunction

  task automatic init_mem;
    int row;
    int ch;
    int word_idx;
    begin
      for (i = 0; i < MEM_DEPTH; i = i + 1) ddr_mem[i] = '0;

      // Corrected Mode-2 first-layer IFM DDR layout:
      //   [tile_x][row][cgrp][col_l], lanes = PC channels.
      // This Test-A W96 regression has 3 tile_x blocks when PC=32.
      // Each valid IFM lane is tile-coded as tile_x+1 so the test can catch
      // accidental reuse of tile 0 when computing columns from later tiles.
      // Expected first-layer IFM reads = 3 * 64 * ceil(32/32) * 32 = 6144.
      word_idx = 0;
      for (int tile_x = 0; tile_x < L0_COLBLKS; tile_x = tile_x + 1) begin
        for (row = 0; row < L0_H_IN; row = row + 1) begin
          for (int cgrp = 0; cgrp < L0_NUM_CGROUP; cgrp = cgrp + 1) begin
            for (int col_l = 0; col_l < PC; col_l = col_l + 1) begin
              ddr_mem[`DDR_IFM_BASE + word_idx] = pack_m2_ifm_tile_word(tile_x, cgrp, col_l);
              word_idx = word_idx + 1;
            end
          end
        end
      end

      for (i = 0; i < L0_WGT_DDR_WORDS; i = i + 1) ddr_mem[L0_WGT_DDR_BASE + i] = pack_ddr_ones_word();
      for (i = 0; i < L1_WGT_DDR_WORDS; i = i + 1) ddr_mem[L1_WGT_DDR_BASE + i] = pack_ddr_ones_word();
      for (i = 0; i < L2_WGT_DDR_WORDS; i = i + 1) ddr_mem[L2_WGT_DDR_BASE + i] = pack_ddr_ones_word();
      for (i = 0; i < L3_WGT_DDR_WORDS; i = i + 1) ddr_mem[L3_WGT_DDR_BASE + i] = pack_ddr_ones_word();
      for (i = 0; i < L4_WGT_DDR_WORDS; i = i + 1) ddr_mem[L4_WGT_DDR_BASE + i] = pack_ddr_ones_word();
      for (i = 0; i < L5_WGT_DDR_WORDS; i = i + 1) ddr_mem[L5_WGT_DDR_BASE + i] = pack_ddr_ones_word();
      for (i = 0; i < L6_WGT_DDR_WORDS; i = i + 1) ddr_mem[L6_WGT_DDR_BASE + i] = pack_ddr_ones_word();
      for (i = 0; i < L7_WGT_DDR_WORDS; i = i + 1) ddr_mem[L7_WGT_DDR_BASE + i] = pack_ddr_ones_word();
      for (i = 0; i < L8_WGT_DDR_WORDS; i = i + 1) ddr_mem[L8_WGT_DDR_BASE + i] = pack_ddr_ones_word();
    end
  endtask

  task automatic write_cfg(input logic [$clog2(CFG_DEPTH)-1:0] addr, input layer_desc_t cfg);
    begin
      @(posedge clk);
      cfg_wr_en <= 1'b1;
      cfg_wr_addr <= addr;
      cfg_wr_data <= cfg;
      @(posedge clk);
      cfg_wr_en <= 1'b0;
      cfg_wr_addr <= '0;
      cfg_wr_data <= '0;
    end
  endtask

  task automatic fill_m2_cfg(
    output layer_desc_t cfg,
    input int layer_id,
    input int h_in,
    input int w_in,
    input int c_in,
    input int f_out,
    input int k,
    input int h_out,
    input int w_out,
    input bit pool_en,
    input int wgt_base,
    input bit first_layer,
    input bit last_layer
  );
    begin
      cfg = '0;
      cfg.layer_id = layer_id;
      cfg.mode = MODE2;
      cfg.h_in = h_in;
      cfg.w_in = w_in;
      cfg.c_in = c_in;
      cfg.f_out = f_out;
      cfg.k = k;
      cfg.h_out = h_out;
      cfg.w_out = w_out;
      cfg.pv_m1 = PV_MAX;
      cfg.pf_m1 = PF_MAX;
      cfg.pc_m2 = PC;
      cfg.pf_m2 = PF;
      cfg.conv_stride = 1;
      cfg.pad_top = 0;
      cfg.pad_bottom = 0;
      cfg.pad_left = 0;
      cfg.pad_right = 0;
      cfg.relu_en = 1'b1;
      cfg.pool_en = pool_en;
      cfg.pool_k = 2;
      cfg.pool_stride = 2;
      cfg.ifm_ddr_base = `DDR_IFM_BASE;
      cfg.wgt_ddr_base = wgt_base;
      cfg.ofm_ddr_base = `DDR_OFM_BASE;
      cfg.first_layer = first_layer;
      cfg.last_layer = last_layer;
    end
  endtask

  task automatic program_layers;
    layer_desc_t cfg0, cfg1, cfg2, cfg3, cfg4, cfg5, cfg6, cfg7, cfg8;
    begin
      fill_m2_cfg(cfg0, 0, L0_H_IN, L0_W_IN, L0_C_IN, L0_F_OUT, L0_K, L0_H_CONV_OUT, L0_W_CONV_OUT, L0_POOL_EN, L0_WGT_DDR_BASE, 1'b1, 1'b0);
      fill_m2_cfg(cfg1, 1, L1_H_IN, L1_W_IN, L1_C_IN, L1_F_OUT, L1_K, L1_H_CONV_OUT, L1_W_CONV_OUT, L1_POOL_EN, L1_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_m2_cfg(cfg2, 2, L2_H_IN, L2_W_IN, L2_C_IN, L2_F_OUT, L2_K, L2_H_CONV_OUT, L2_W_CONV_OUT, L2_POOL_EN, L2_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_m2_cfg(cfg3, 3, L3_H_IN, L3_W_IN, L3_C_IN, L3_F_OUT, L3_K, L3_H_CONV_OUT, L3_W_CONV_OUT, L3_POOL_EN, L3_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_m2_cfg(cfg4, 4, L4_H_IN, L4_W_IN, L4_C_IN, L4_F_OUT, L4_K, L4_H_CONV_OUT, L4_W_CONV_OUT, L4_POOL_EN, L4_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_m2_cfg(cfg5, 5, L5_H_IN, L5_W_IN, L5_C_IN, L5_F_OUT, L5_K, L5_H_CONV_OUT, L5_W_CONV_OUT, L5_POOL_EN, L5_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_m2_cfg(cfg6, 6, L6_H_IN, L6_W_IN, L6_C_IN, L6_F_OUT, L6_K, L6_H_CONV_OUT, L6_W_CONV_OUT, L6_POOL_EN, L6_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_m2_cfg(cfg7, 7, L7_H_IN, L7_W_IN, L7_C_IN, L7_F_OUT, L7_K, L7_H_CONV_OUT, L7_W_CONV_OUT, L7_POOL_EN, L7_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_m2_cfg(cfg8, 8, L8_H_IN, L8_W_IN, L8_C_IN, L8_F_OUT, L8_K, L8_H_CONV_OUT, L8_W_CONV_OUT, L8_POOL_EN, L8_WGT_DDR_BASE, 1'b0, 1'b1);

      write_cfg(0, cfg0);
      write_cfg(1, cfg1);
      write_cfg(2, cfg2);
      write_cfg(3, cfg3);
      write_cfg(4, cfg4);
      write_cfg(5, cfg5);
      write_cfg(6, cfg6);
      write_cfg(7, cfg7);
      write_cfg(8, cfg8);
    end
  endtask

  task automatic pulse_start;
    begin
      @(posedge clk);
      start <= 1'b1;
      @(posedge clk);
      start <= 1'b0;
    end
  endtask

  task automatic dump_ofm_region;
    int j;
    begin
      $display("--- OFM DDR region dump ---");
      for (j = 0; j < 32; j = j + 1) begin
        $display("OFM[%0d] @0x%05h = 0x%0h", j, (`DDR_OFM_BASE + j), ddr_mem[`DDR_OFM_BASE + j]);
      end
    end
  endtask

  task automatic check_final_ofm;
    int mismatch;
    logic [DDR_WORD_W-1:0] exp_word;
    begin
      mismatch = 0;
      for (int j = 0; j < EXPECTED_OFM_DDR_WORDS; j = j + 1) begin
        exp_word = expected_final_word(j);
        if (ddr_mem[`DDR_OFM_BASE + j] !== exp_word) begin
          if (mismatch < 40) begin
            $display("TB_MISMATCH_M2_9L_FULLSIZE_K1 word=%0d got=0x%0h exp=0x%0h", j, ddr_mem[`DDR_OFM_BASE + j], exp_word);
          end
          mismatch = mismatch + 1;
        end
      end
      if (mismatch != 0) begin
        dump_ofm_region();
        $fatal(1, "TB_FAIL: 9-layer Mode2 Test-A K1 final OFM mismatch count=%0d", mismatch);
      end
    end
  endtask

  task automatic print_banner;
    begin
      $display("TB_INFO: 9-layer Mode2 DCP/EfficientNet-B0-style Test-A 64x96x32 K1 expected-compare test");
      $display("TB_INFO: Test-A Mode2 multi-tile regression; IFM tile pattern checks true tile scheduling");
      $display("TB_INFO: this test uses input %0dx%0dx%0d; K=1 isolates multi-tile WT=PC from K>1 halo behavior", L0_H_IN, L0_W_IN, L0_C_IN);
      $display("TB_INFO: Mode2 PC=%0d PF=%0d PTOTAL=%0d PV_MAX=%0d WGT_SUBWORDS=%0d", PC, PF, PTOTAL, PV_MAX, WGT_SUBWORDS);
      $display("TB_INFO: channel trend: 32 -> 32 -> 16 -> 24 -> 24 -> 40 -> 40 -> 80 -> 80 -> 192");
      $display("TB_INFO: L0 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> %s %0dx%0dx%0d", L0_H_IN, L0_W_IN, L0_C_IN, L0_H_CONV_OUT, L0_W_CONV_OUT, L0_F_OUT, L0_K, (L0_POOL_EN ? "pool" : "nopool"), L0_H_OUT, L0_W_OUT, L0_F_OUT);
      $display("TB_INFO: L1 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> %s %0dx%0dx%0d", L1_H_IN, L1_W_IN, L1_C_IN, L1_H_CONV_OUT, L1_W_CONV_OUT, L1_F_OUT, L1_K, (L1_POOL_EN ? "pool" : "nopool"), L1_H_OUT, L1_W_OUT, L1_F_OUT);
      $display("TB_INFO: L2 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> %s %0dx%0dx%0d", L2_H_IN, L2_W_IN, L2_C_IN, L2_H_CONV_OUT, L2_W_CONV_OUT, L2_F_OUT, L2_K, (L2_POOL_EN ? "pool" : "nopool"), L2_H_OUT, L2_W_OUT, L2_F_OUT);
      $display("TB_INFO: L3 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> %s %0dx%0dx%0d", L3_H_IN, L3_W_IN, L3_C_IN, L3_H_CONV_OUT, L3_W_CONV_OUT, L3_F_OUT, L3_K, (L3_POOL_EN ? "pool" : "nopool"), L3_H_OUT, L3_W_OUT, L3_F_OUT);
      $display("TB_INFO: L4 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> %s %0dx%0dx%0d", L4_H_IN, L4_W_IN, L4_C_IN, L4_H_CONV_OUT, L4_W_CONV_OUT, L4_F_OUT, L4_K, (L4_POOL_EN ? "pool" : "nopool"), L4_H_OUT, L4_W_OUT, L4_F_OUT);
      $display("TB_INFO: L5 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> %s %0dx%0dx%0d", L5_H_IN, L5_W_IN, L5_C_IN, L5_H_CONV_OUT, L5_W_CONV_OUT, L5_F_OUT, L5_K, (L5_POOL_EN ? "pool" : "nopool"), L5_H_OUT, L5_W_OUT, L5_F_OUT);
      $display("TB_INFO: L6 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> %s %0dx%0dx%0d", L6_H_IN, L6_W_IN, L6_C_IN, L6_H_CONV_OUT, L6_W_CONV_OUT, L6_F_OUT, L6_K, (L6_POOL_EN ? "pool" : "nopool"), L6_H_OUT, L6_W_OUT, L6_F_OUT);
      $display("TB_INFO: L7 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> %s %0dx%0dx%0d", L7_H_IN, L7_W_IN, L7_C_IN, L7_H_CONV_OUT, L7_W_CONV_OUT, L7_F_OUT, L7_K, (L7_POOL_EN ? "pool" : "nopool"), L7_H_OUT, L7_W_OUT, L7_F_OUT);
      $display("TB_INFO: L8 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> %s %0dx%0dx%0d", L8_H_IN, L8_W_IN, L8_C_IN, L8_H_CONV_OUT, L8_W_CONV_OUT, L8_F_OUT, L8_K, (L8_POOL_EN ? "pool" : "nopool"), L8_H_OUT, L8_W_OUT, L8_F_OUT);
      $display("TB_INFO: expected DDR->IFM reads=%0d", EXPECTED_IFM_DDR_READS);
      $display("TB_INFO: expected WGT DDR reads=%0d physical_words=%0d", EXPECTED_WGT_DDR_READS, L0_WGT_WORDS+L1_WGT_WORDS+L2_WGT_WORDS+L3_WGT_WORDS+L4_WGT_WORDS+L5_WGT_WORDS+L6_WGT_WORDS+L7_WGT_WORDS+L8_WGT_WORDS);
      $display("TB_INFO: expected deterministic OFM->IFM Mode2 stream commands >= %0d", EXP_OFM2IFM_STREAMS);
      $display("TB_INFO: expected final OFM logical elements=%0d", EXPECTED_FINAL_ELEMENTS);
      $display("TB_INFO: expected final OFM DDR words=%0d using Mode2 layout F*H*ceil(W/PC) = %0d*%0d*ceil(%0d/%0d)", EXPECTED_OFM_DDR_WORDS, L8_F_OUT, L8_H_OUT, L8_W_OUT, PC);
      $display("TB_INFO: expected L0->L1 tile-pattern stream words=%0d", EXPECTED_L0_STREAM_WORDS);
      $display("TB_INFO: expected final value per valid element=%0d", EXPECTED_FINAL_VALUE);
    end
  endtask

  initial begin
    rst_n = 1'b0;
    start = 1'b0;
    abort = 1'b0;
    cfg_wr_en = 1'b0;
    cfg_wr_addr = '0;
    cfg_wr_data = '0;
    cfg_num_layers = 9;
    m1_free_col_blk_g = '0;
    m1_free_ch_blk_g = '0;
    m1_sm_refill_req_ready = 1'b1;
    m2_sm_refill_req_ready = 1'b1;

    init_mem();
    print_banner();

    repeat (5) @(posedge clk);
    rst_n = 1'b1;
    program_layers();
    repeat (5) @(posedge clk);
    pulse_start();

    wait (done_seen || error_seen || (cycle_count > MAX_CYCLES));
    repeat (2) @(posedge clk);

    if (error_seen) begin
      $display("TB_FAIL: DUT asserted error. first_error_vec=%04b layer=%0d mode=%0d first_error_cycle=%0d", first_error_vec, first_error_layer, first_error_mode, first_error_cycle);
      $display("TB_ERROR_DECODE: bit0=dma_error bit1=ofm_error bit2=local_error bit3=transition_error");
      $display("DDR counts at stop: ifm_reads=%0d wgt_reads=%0d ofm_writes=%0d done_seen=%0b busy=%0b", ddr_ifm_read_count, ddr_wgt_read_count, ddr_ofm_write_count, done_seen, busy);
      $display("Stream counts at stop: start=%0d done=%0d legacy_m2_req=%0d expected>=%0d", ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, m2_sm_refill_req_count, EXP_OFM2IFM_STREAMS);
      dump_ofm_region();
      $fatal(1, "TB_FAIL: DUT error before successful completion");
    end

    if (cycle_count > MAX_CYCLES) begin
      $display("TB_FAIL: timeout after %0d cycles. busy=%0b done=%0b error=%0b layer=%0d mode=%0d vec=%04b", cycle_count, busy, done, error, dbg_layer_idx, dbg_mode, dbg_error_vec);
      $display("DDR counts at timeout: ifm_reads=%0d wgt_reads=%0d ofm_writes=%0d", ddr_ifm_read_count, ddr_wgt_read_count, ddr_ofm_write_count);
      $display("Stream counts at timeout: start=%0d done=%0d legacy_m2_req=%0d expected>=%0d", ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, m2_sm_refill_req_count, EXP_OFM2IFM_STREAMS);
      dump_ofm_region();
      $fatal(1, "TB_FAIL: timeout");
    end

    if (!done_seen) $fatal(1, "TB_FAIL: stopped without done_seen");

    $display("TB_INFO: 9-layer Mode2 done after %0d cycles", cycle_count);
    $display("TB_INFO: DDR counts: ifm_reads=%0d expected=%0d, wgt_reads=%0d expected=%0d, ofm_writes=%0d expected=%0d", ddr_ifm_read_count, EXPECTED_IFM_DDR_READS, ddr_wgt_read_count, EXPECTED_WGT_DDR_READS, ddr_ofm_write_count, EXPECTED_OFM_DDR_WORDS);
    $display("TB_INFO: OFM->IFM stream starts=%0d done=%0d expected>=%0d, legacy_m2_refill_req=%0d", ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, EXP_OFM2IFM_STREAMS, m2_sm_refill_req_count);
    $display("TB_INFO: L0 stream tile-pattern checks=%0d mismatch=%0d", l0_stream_checked_count, l0_stream_mismatch_count);

    if (ddr_ifm_read_count != EXPECTED_IFM_DDR_READS) $fatal(1, "TB_FAIL: unexpected IFM DDR read count");
    if (l0_stream_checked_count != EXPECTED_L0_STREAM_WORDS) $fatal(1, "TB_FAIL: unexpected L0->L1 stream data word count");
    if (l0_stream_mismatch_count != 0) $fatal(1, "TB_FAIL: L0->L1 stream tile-pattern mismatch count=%0d", l0_stream_mismatch_count);
    if (ddr_wgt_read_count != EXPECTED_WGT_DDR_READS) $fatal(1, "TB_FAIL: unexpected WGT DDR read count");
    if (ddr_ofm_write_count != EXPECTED_OFM_DDR_WORDS) begin
      dump_ofm_region();
      $fatal(1, "TB_FAIL: unexpected OFM DDR write count");
    end
    if (ofm_ifm_stream_done_count < EXP_OFM2IFM_STREAMS) $fatal(1, "TB_FAIL: insufficient OFM->IFM stream done count");
    if (m2_sm_refill_req_count != 0) $display("TB_WARN: legacy M2 refill request count is nonzero: %0d", m2_sm_refill_req_count);

    check_final_ofm();
    $display("TB_PASS: 9-layer Mode2 Test-A 64x96x32 K1 multi-tile pattern expected-compare passed. final_value=%0d elements=%0d ddr_words=%0d", EXPECTED_FINAL_VALUE, EXPECTED_FINAL_ELEMENTS, EXPECTED_OFM_DDR_WORDS);
    $finish;
  end
endmodule
