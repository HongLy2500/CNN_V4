`timescale 1ns/1ps
`include "cnn_ddr_defs.svh"

module tb_cnn_top_9layer_m1first_m2_testB_160x192x3_k3_expected_compare;
  import cnn_layer_desc_pkg::*;

  // --------------------------------------------------------------------------
  // 9-layer mixed Mode1->Mode2 DCP/EfficientNet-B0-style Test-B K=3 halo/window test.
  //
  // Purpose:
  //   Companion to Test-A K=1.  Test-A validated the corrected Mode-2
  //   WT=PC resident IFM layout and Mode1-style active-context OFM->IFM
  //   refill without K>1 halo/window pressure.  This Test-B deliberately
  //   uses a Mode1 first layer followed by K=3 Mode2 layers to stress the parts Test-A did not touch without requiring a non-realistic Mode2 DDR->IFM first-layer refill path:
  //
  //   - Mode1 first-layer producer and Mode1->Mode2 transition
  //   - 3x3 sliding windows in Mode 2
  //   - horizontal halo crossing across PC-wide resident tiles
  //   - vertical halo crossing across rows
  //   - pooling after K=3 convolution
  //   - later-layer partial C groups and partial F groups
  //   - final Mode-2 OFM DDR readback with small final W
  //
  // Shape choice:
  //   K=3 shrinks H/W by 2 at every layer, and layers 0/2/4/6 pool by 2.
  //   The spatial shape is kept at 160x192 so all 9 layers remain valid.
  //   L0 uses a realistic Mode1/RGB-like C=3 input and produces 32 channels
  //   for the subsequent Mode2 layers.
  //
  // Data/weight model:
  //   First-layer IFM is all ones and L0 runs in Mode1:
  //     every DDR IFM word lane is 1, avoiding any Mode2 DDR->IFM preload dependency.
  //   L0 weights are all ones in Mode1, so L0 output after 3x3 over 3 channels is 27.
  //   Layers L1..L8 run in Mode2 with all-ones weights; from L1 onward values
  //   saturate to 127, so the final OFM is still checked exactly against 127.
  // --------------------------------------------------------------------------

  localparam int DATA_W = 8;
  localparam int PSUM_W = 32;

  // DCP-CNN Table VI Mode-2 point: Pc=32, Pf=64 => PTOTAL=2048.
  localparam int PC = 32;
  localparam int PF = 64;
  localparam int PTOTAL = PC * PF;

  // Mode1 first-layer schedule.  Keep PV_M1*PF_M1 == PTOTAL.
  localparam int PV_M1 = 32;
  localparam int PF_M1 = 64;

  // Keep project-wide runtime maxima consistent with the current contract.
  // DDR word width is PV_MAX*DATA_W; weight DMA assembles one PTOTAL word from
  // WGT_SUBWORDS DDR words when PTOTAL > PV_MAX.
  localparam int PV_MAX = 128;
  localparam int PF_MAX = 128;

  localparam int C_MAX = 192;
  localparam int F_MAX = 192;
  localparam int W_MAX = 192;
  localparam int H_MAX = 160;
  localparam int HT = 4;
  localparam int K_MAX = 3;

  localparam int WGT_DEPTH = 512;
  // OFM storage in this all-Mode2 K3 test packs spatial columns in PC-wide words.
  // Do not use W_MAX as the physical row stride here: W_MAX=192 would make
  // DEPTH=160*ceil(192/32) per bank and can crash XSim with huge unpacked arrays.
  // The required physical groups per row are ceil(W_MAX/PC)=6.
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
  localparam int MAX_CYCLES = 120000000;

  // 9-layer EfficientNet-B0 channel trend used in the Mode-1 Table-VI test.
  // K is set to 3 here to validate Mode-2 halo/window behavior after
  // the Test-A K1 multi-tile storage regression has passed.
  localparam int L0_H_IN=160, L0_W_IN=192, L0_C_IN=3,  L0_F_OUT=32, L0_K=3, L0_POOL_EN=1;
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

  localparam int L0_NUM_CGROUP=(L0_C_IN+PV_M1-1)/PV_M1, L0_NUM_FGROUP=(L0_F_OUT+PF_M1-1)/PF_M1;
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

  localparam int L0_COLBLKS=(L0_W_IN+PV_M1-1)/PV_M1;
  localparam int L1_COLBLKS=(L1_W_IN+PC-1)/PC;
  localparam int L2_COLBLKS=(L2_W_IN+PC-1)/PC;
  localparam int L3_COLBLKS=(L3_W_IN+PC-1)/PC;
  localparam int L4_COLBLKS=(L4_W_IN+PC-1)/PC;
  localparam int L5_COLBLKS=(L5_W_IN+PC-1)/PC;
  localparam int L6_COLBLKS=(L6_W_IN+PC-1)/PC;
  localparam int L7_COLBLKS=(L7_W_IN+PC-1)/PC;
  localparam int L8_COLBLKS=(L8_W_IN+PC-1)/PC;

  // Mode1-style active-context Mode-2 OFM->IFM refill streams one IFM resident entry per
  // command/write in this regression:
  //   IFM entry = {row_g, global_col_g, cgrp}, data lanes = PC channels.
  // This is intentionally different from the OFM storage/DMA readback layout
  // {channel,row,col_group}.
  localparam int EXP_OFM2IFM_STREAMS =
      (L1_H_IN*L1_W_IN*L1_NUM_CGROUP) +
      (L2_H_IN*L2_W_IN*L2_NUM_CGROUP) +
      (L3_H_IN*L3_W_IN*L3_NUM_CGROUP) +
      (L4_H_IN*L4_W_IN*L4_NUM_CGROUP) +
      (L5_H_IN*L5_W_IN*L5_NUM_CGROUP) +
      (L6_H_IN*L6_W_IN*L6_NUM_CGROUP) +
      (L7_H_IN*L7_W_IN*L7_NUM_CGROUP) +
      (L8_H_IN*L8_W_IN*L8_NUM_CGROUP);

  // L0->L1 checker observes the IFM write port after ofm_buffer has converted
  // final OFM data into the IFM Mode-2 resident-tile layout:
  //   ifm_ofm_wr_bank    = local column col_l = global_col % PC
  //   ifm_ofm_wr_col_idx = cgrp
  //   ifm_ofm_wr_data    = PC channel lanes at {row, global_col, cgrp}
  localparam int EXPECTED_L0_STREAM_IFM_WRITES = L1_H_IN * L1_W_IN * L1_NUM_CGROUP;

  // Mode1 first-layer IFM DDR read estimate.  The test does not assert this
  // count because this mixed test is meant to avoid imposing any Mode2
  // DDR->IFM refill behavior.  It is printed for visibility only.
  localparam int EXPECTED_IFM_DDR_READS = L0_COLBLKS * L0_H_IN * L0_C_IN;
  localparam int EXPECTED_WGT_DDR_READS = L0_WGT_DDR_WORDS+L1_WGT_DDR_WORDS+L2_WGT_DDR_WORDS+L3_WGT_DDR_WORDS+L4_WGT_DDR_WORDS+L5_WGT_DDR_WORDS+L6_WGT_DDR_WORDS+L7_WGT_DDR_WORDS+L8_WGT_DDR_WORDS;

  // OFM buffer Mode-2 final DMA layout is NOT flat H*W*F words.
  // It stores one word per {output channel, row, spatial column group}.
  // In Mode 2, store_pack = PC, so groups per row = ceil(W_out / PC).
  localparam int L8_STORED_GROUPS = (L8_W_OUT + PC - 1) / PC;
  localparam int EXPECTED_FINAL_ELEMENTS = L8_F_OUT * L8_H_OUT * L8_W_OUT;
  localparam int EXPECTED_OFM_DDR_WORDS = L8_F_OUT * L8_H_OUT * L8_STORED_GROUPS;

  // Derived from this testbench, not assumed:
  //   init_mem() writes first-layer IFM with tile pattern tile_x+1 and all
  //   layer weights=1. K=3 for every layer.
  //   L0 Mode1 output is 27, and L1 onward saturates to 127.
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
  integer legacy_m2_mgr_req_count;
  logic   legacy_m2_mgr_req_seen_q;
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
      legacy_m2_mgr_req_count <= 0;
      legacy_m2_mgr_req_seen_q <= 1'b0;
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
        if (ofm_ifm_stream_start_count < 64) begin
          $display("DBG_TB_M2_STREAM_START t=%0t cycle=%0d count=%0d layer=%0d mode=%0d row=%0d col=%0d cgrp=%0d legacy_mgr_req_count=%0d legacy_mgr_req_now=%0b",
                   $time, cycle_count, ofm_ifm_stream_start_count,
                   dbg_layer_idx, dbg_mode,
                   dut.ofm_ifm_stream_row_base_s,
                   dut.ofm_ifm_stream_col_base_s,
                   dut.ofm_ifm_stream_m2_cgrp_g_s,
                   legacy_m2_mgr_req_count,
                   (m2_sm_refill_req_valid && m2_sm_refill_req_ready));
        end
      end
      if (dut.ofm_ifm_stream_done_s) ofm_ifm_stream_done_count <= ofm_ifm_stream_done_count + 1;
      if (m2_sm_refill_req_valid && m2_sm_refill_req_ready) begin
        legacy_m2_mgr_req_count <= legacy_m2_mgr_req_count + 1;
        if (!legacy_m2_mgr_req_seen_q) begin
          legacy_m2_mgr_req_seen_q <= 1'b1;
          $display("TB_WARN_LEGACY_M2_MANAGER_REQ t=%0t cycle=%0d: legacy same-mode-refill manager request fired. This should remain 0 for Mode1-style Mode2 refill (sm_m2_mgr_active_s must be 1'b0).",
                   $time, cycle_count);
        end
      end
    end
  end

  always_ff @(posedge clk) begin
    if (rst_n) begin
      if (done || error || ((cycle_count % 50000) == 0)) begin
        $display("DBG_TOP_STATUS t=%0t cycle=%0d busy=%0b done=%0b error=%0b vec=%04b layer=%0d mode=%0d ifm_rd=%0d wgt_rd=%0d ofm_wr=%0d stream_start=%0d stream_done=%0d legacy_m2_mgr_req=%0d", $time, cycle_count, busy, done, error, dbg_error_vec, dbg_layer_idx, dbg_mode, ddr_ifm_read_count, ddr_wgt_read_count, ddr_ofm_write_count, ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, legacy_m2_mgr_req_count);
      end
    end
  end

  always_ff @(posedge clk) begin
    if (rst_n) begin
      // Check only the first L0->L1 OFM->IFM IFM-entry writes.
      // This checker observes ifm_ofm_wr_* after ofm_buffer has converted
      // producer final OFM storage into the IFM Mode-2 resident-tile format.
      //
      // Correct IFM Mode-2 write contract:
      //   ifm_ofm_wr_bank    = col_l = global_col % PC
      //   ifm_ofm_wr_col_idx = cgrp
      //   ifm_ofm_wr_data    = PC channel lanes at {row, global_col, cgrp}
      //
      // Do NOT interpret bank as output channel or lane as spatial column.
      if (dut.ifm_ofm_wr_en_s &&
          dut.ifm_ofm_wr_ready_s &&
          tb_l0_stream_cmd_valid_q &&
          (l0_stream_checked_count < EXPECTED_L0_STREAM_IFM_WRITES)) begin
        int col_l;
        int global_col;
        int cgrp;
        int lane;
        int ch;
        int bad_lane;
        logic word_bad;
        logic expected_keep_lane;
        logic [PC-1:0] expected_keep_pc;
        logic [DATA_W-1:0] got_bad;
        logic [DATA_W-1:0] exp_bad;
        logic [DATA_W-1:0] exp_val;

        col_l      = int'(dut.ifm_ofm_wr_bank_s);
        cgrp       = int'(dut.ifm_ofm_wr_col_idx_s);
              // Mode2 active-context stream commands carry the exact global column.
        // ofm_buffer derives destination col_l as global_col % PC.
        global_col = int'(tb_l0_stream_col_base_q);
        bad_lane   = -1;
        word_bad   = 1'b0;
        expected_keep_pc = '0;
        got_bad = '0;
        exp_bad = '0;
        exp_val = expected_l0_stream_value(int'(dut.ifm_ofm_wr_row_idx_s), global_col);

        // Structural checks for IFM Mode-2 resident-tile write.
        if ((dut.ifm_ofm_wr_row_idx_s >= L1_H_IN) ||
            (dut.ifm_ofm_wr_row_idx_s != tb_l0_stream_row_base_q) ||
            (col_l < 0) || (col_l >= PC) ||
            (global_col >= L1_W_IN) ||
            (col_l != (global_col % PC)) ||
            (cgrp >= L1_NUM_CGROUP) ||
            (cgrp != int'(tb_l0_stream_cgrp_q))) begin
          word_bad = 1'b1;
        end

        // Value check: lanes are channel lanes within cgrp. For the L0->L1
        // row+tile halo check, all valid channels at the same global output
        // column have the same expected value.
        for (lane = 0; lane < PC; lane = lane + 1) begin
          ch = cgrp * PC + lane;
          expected_keep_lane = (ch < L1_C_IN) && (global_col < L1_W_IN) &&
                               (dut.ifm_ofm_wr_row_idx_s < L1_H_IN);
          expected_keep_pc[lane] = expected_keep_lane;

          if (expected_keep_lane) begin
            if (!dut.ifm_ofm_wr_keep_s[lane] ||
                (dut.ifm_ofm_wr_data_s[lane*DATA_W +: DATA_W] !== exp_val)) begin
              if (!word_bad) begin
                bad_lane = lane;
                got_bad  = dut.ifm_ofm_wr_data_s[lane*DATA_W +: DATA_W];
                exp_bad  = exp_val;
              end
              word_bad = 1'b1;
            end
          end
          else begin
            if (dut.ifm_ofm_wr_keep_s[lane]) begin
              if (!word_bad) begin
                bad_lane = lane;
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
            $display("TB_MISMATCH_L0_M1_TO_M2_IFM_WR t=%0t cycle=%0d word=%0d row=%0d cmd_row=%0d col_l=%0d global_col=%0d cgrp=%0d cmd_col=%0d live_col=%0d cmd_cgrp=%0d bad_lane=%0d got_bad=%0d exp_bad=%0d got0=%0d got1=%0d got15=%0d got16=%0d got31=%0d keep=%h exp_keep=%h",
              $time,
              cycle_count,
              l0_stream_checked_count,
              dut.ifm_ofm_wr_row_idx_s,
              tb_l0_stream_row_base_q,
              col_l,
              global_col,
              cgrp,
              tb_l0_stream_col_base_q,
              dut.ofm_ifm_stream_col_base_s,
              tb_l0_stream_cgrp_q,
              bad_lane,
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
    input int row;
    input int cgrp;
    input int col_l;
    logic [DDR_WORD_W-1:0] word;
    int ch;
    int val_i;
    logic [DATA_W-1:0] val;
    begin
      word = '0;
      // Row+tile pattern keeps L0 K=3 checker sensitive to both horizontal
      // tile halo and vertical row halo without saturating when L0 sparse
      // weights are used.
      val_i = tile_x + 1 + (row % 4);
      val = val_i[DATA_W-1:0];
      for (int lane = 0; lane < DDR_LANES; lane++) begin
        ch = cgrp * PC + lane;
        if ((lane < PC) && (ch < L0_C_IN) && ((tile_x * PC + col_l) < L0_W_IN)) begin
          word[lane*DATA_W +: DATA_W] = val;
        end
      end
      return word;
    end
  endfunction

  function automatic logic [DDR_WORD_W-1:0] pack_m2_weight_ch0_subword;
    input int subword_idx;
    logic [DDR_WORD_W-1:0] word;
    int global_w_lane;
    begin
      word = '0;
      // PTOTAL weights are ordered as PF filters x PC lanes.  For L0 only,
      // every filter keeps only PC lane 0 enabled for every K=3 tap.  This
      // verifies spatial K=3 halo/window behavior while avoiding immediate
      // L0 saturation.  Later layers use all-ones weights.
      for (int lane = 0; lane < DDR_LANES; lane++) begin
        global_w_lane = subword_idx * DDR_LANES + lane;
        if ((global_w_lane < PTOTAL) && ((global_w_lane % PC) == 0)) begin
          word[lane*DATA_W +: DATA_W] = 8'sd1;
        end
      end
      return word;
    end
  endfunction

  function automatic int l0_ifm_pattern_val;
    input int row;
    input int col;
    int tile_x;
    begin
      tile_x = col / PC;
      return tile_x + 1 + (row % 4);
    end
  endfunction

  function automatic int l0_conv3_sparse_sum;
    input int conv_row;
    input int conv_col;
    int acc;
    begin
      acc = 0;
      for (int ky = 0; ky < L0_K; ky++) begin
        for (int kx = 0; kx < L0_K; kx++) begin
          acc += l0_ifm_pattern_val(conv_row + ky, conv_col + kx);
        end
      end
      return acc;
    end
  endfunction

  function automatic logic [DATA_W-1:0] expected_l0_stream_value;
    input int out_row;
    input int out_col;
    begin
      // L0 is now Mode1 with all-one IFM and all-one weights.
      // K=3, C=3 => raw conv sum = 3*3*3 = 27. Pooling max keeps 27.
      // All valid L0 filters therefore produce the same value for L1 IFM.
      return DATA_W'(27);
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

      // Mode1 first-layer IFM: avoid depending on exact shallow-layer DDR
      // packing by filling the IFM region with all-one words.  The first
      // layer is Mode1, so this test intentionally does not require Mode2
      // DDR->IFM runtime refill.
      for (i = 0; i < 8192; i = i + 1) begin
        ddr_mem[`DDR_IFM_BASE + i] = pack_ddr_ones_word();
      end

      // All weights are one.  L0 Mode1 output is 27 for K=3,C=3; L1+ Mode2
      // layers saturate to 127.
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

  task automatic fill_layer_cfg(
    output layer_desc_t cfg,
    input int layer_id,
    input bit use_mode2,
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
      cfg.mode = use_mode2 ? MODE2 : MODE1;
      cfg.h_in = h_in;
      cfg.w_in = w_in;
      cfg.c_in = c_in;
      cfg.f_out = f_out;
      cfg.k = k;
      cfg.h_out = h_out;
      cfg.w_out = w_out;
      cfg.pv_m1 = PV_M1;
      cfg.pf_m1 = PF_M1;
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
      fill_layer_cfg(cfg0, 0, 1'b0, L0_H_IN, L0_W_IN, L0_C_IN, L0_F_OUT, L0_K, L0_H_CONV_OUT, L0_W_CONV_OUT, L0_POOL_EN, L0_WGT_DDR_BASE, 1'b1, 1'b0);
      fill_layer_cfg(cfg1, 1, 1'b1, L1_H_IN, L1_W_IN, L1_C_IN, L1_F_OUT, L1_K, L1_H_CONV_OUT, L1_W_CONV_OUT, L1_POOL_EN, L1_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg2, 2, 1'b1, L2_H_IN, L2_W_IN, L2_C_IN, L2_F_OUT, L2_K, L2_H_CONV_OUT, L2_W_CONV_OUT, L2_POOL_EN, L2_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg3, 3, 1'b1, L3_H_IN, L3_W_IN, L3_C_IN, L3_F_OUT, L3_K, L3_H_CONV_OUT, L3_W_CONV_OUT, L3_POOL_EN, L3_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg4, 4, 1'b1, L4_H_IN, L4_W_IN, L4_C_IN, L4_F_OUT, L4_K, L4_H_CONV_OUT, L4_W_CONV_OUT, L4_POOL_EN, L4_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg5, 5, 1'b1, L5_H_IN, L5_W_IN, L5_C_IN, L5_F_OUT, L5_K, L5_H_CONV_OUT, L5_W_CONV_OUT, L5_POOL_EN, L5_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg6, 6, 1'b1, L6_H_IN, L6_W_IN, L6_C_IN, L6_F_OUT, L6_K, L6_H_CONV_OUT, L6_W_CONV_OUT, L6_POOL_EN, L6_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg7, 7, 1'b1, L7_H_IN, L7_W_IN, L7_C_IN, L7_F_OUT, L7_K, L7_H_CONV_OUT, L7_W_CONV_OUT, L7_POOL_EN, L7_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg8, 8, 1'b1, L8_H_IN, L8_W_IN, L8_C_IN, L8_F_OUT, L8_K, L8_H_CONV_OUT, L8_W_CONV_OUT, L8_POOL_EN, L8_WGT_DDR_BASE, 1'b0, 1'b1);

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
            $display("TB_MISMATCH_M2_9L_FULLSIZE_K3 word=%0d got=0x%0h exp=0x%0h", j, ddr_mem[`DDR_OFM_BASE + j], exp_word);
          end
          mismatch = mismatch + 1;
        end
      end
      if (mismatch != 0) begin
        dump_ofm_region();
        $fatal(1, "TB_FAIL: 9-layer Mode2 Test-B K3 final OFM mismatch count=%0d", mismatch);
      end
    end
  endtask

  task automatic print_banner;
    begin
      $display("TB_INFO: 9-layer mixed Mode1->Mode2 Test-B 160x192x3 K3 expected-compare test");
      $display("TB_INFO: Test-B uses L0 Mode1 then L1-L8 Mode2 K=3; no Mode2 DDR->IFM first-layer refill required");
      $display("TB_INFO: this test uses Mode1 input %0dx%0dx%0d; Mode2 starts at L1 and stresses K=3 OFM->IFM refill", L0_H_IN, L0_W_IN, L0_C_IN);
      $display("TB_INFO: Mode2 PC=%0d PF=%0d PTOTAL=%0d PV_MAX=%0d WGT_SUBWORDS=%0d", PC, PF, PTOTAL, PV_MAX, WGT_SUBWORDS);
      $display("TB_INFO: channel trend: 3 -> 32 -> 16 -> 24 -> 24 -> 40 -> 40 -> 80 -> 80 -> 192");
      $display("TB_INFO: L0 MODE1 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> %s %0dx%0dx%0d", L0_H_IN, L0_W_IN, L0_C_IN, L0_H_CONV_OUT, L0_W_CONV_OUT, L0_F_OUT, L0_K, (L0_POOL_EN ? "pool" : "nopool"), L0_H_OUT, L0_W_OUT, L0_F_OUT);
      $display("TB_INFO: L1 MODE2 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> %s %0dx%0dx%0d", L1_H_IN, L1_W_IN, L1_C_IN, L1_H_CONV_OUT, L1_W_CONV_OUT, L1_F_OUT, L1_K, (L1_POOL_EN ? "pool" : "nopool"), L1_H_OUT, L1_W_OUT, L1_F_OUT);
      $display("TB_INFO: L2 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> %s %0dx%0dx%0d", L2_H_IN, L2_W_IN, L2_C_IN, L2_H_CONV_OUT, L2_W_CONV_OUT, L2_F_OUT, L2_K, (L2_POOL_EN ? "pool" : "nopool"), L2_H_OUT, L2_W_OUT, L2_F_OUT);
      $display("TB_INFO: L3 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> %s %0dx%0dx%0d", L3_H_IN, L3_W_IN, L3_C_IN, L3_H_CONV_OUT, L3_W_CONV_OUT, L3_F_OUT, L3_K, (L3_POOL_EN ? "pool" : "nopool"), L3_H_OUT, L3_W_OUT, L3_F_OUT);
      $display("TB_INFO: L4 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> %s %0dx%0dx%0d", L4_H_IN, L4_W_IN, L4_C_IN, L4_H_CONV_OUT, L4_W_CONV_OUT, L4_F_OUT, L4_K, (L4_POOL_EN ? "pool" : "nopool"), L4_H_OUT, L4_W_OUT, L4_F_OUT);
      $display("TB_INFO: L5 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> %s %0dx%0dx%0d", L5_H_IN, L5_W_IN, L5_C_IN, L5_H_CONV_OUT, L5_W_CONV_OUT, L5_F_OUT, L5_K, (L5_POOL_EN ? "pool" : "nopool"), L5_H_OUT, L5_W_OUT, L5_F_OUT);
      $display("TB_INFO: L6 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> %s %0dx%0dx%0d", L6_H_IN, L6_W_IN, L6_C_IN, L6_H_CONV_OUT, L6_W_CONV_OUT, L6_F_OUT, L6_K, (L6_POOL_EN ? "pool" : "nopool"), L6_H_OUT, L6_W_OUT, L6_F_OUT);
      $display("TB_INFO: L7 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> %s %0dx%0dx%0d", L7_H_IN, L7_W_IN, L7_C_IN, L7_H_CONV_OUT, L7_W_CONV_OUT, L7_F_OUT, L7_K, (L7_POOL_EN ? "pool" : "nopool"), L7_H_OUT, L7_W_OUT, L7_F_OUT);
      $display("TB_INFO: L8 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> %s %0dx%0dx%0d", L8_H_IN, L8_W_IN, L8_C_IN, L8_H_CONV_OUT, L8_W_CONV_OUT, L8_F_OUT, L8_K, (L8_POOL_EN ? "pool" : "nopool"), L8_H_OUT, L8_W_OUT, L8_F_OUT);
      $display("TB_INFO: estimated Mode1 DDR->IFM reads=%0d (visibility only; not asserted)", EXPECTED_IFM_DDR_READS);
      $display("TB_INFO: estimated WGT DDR reads=%0d physical_words=%0d (visibility only; not asserted)", EXPECTED_WGT_DDR_READS, L0_WGT_WORDS+L1_WGT_WORDS+L2_WGT_WORDS+L3_WGT_WORDS+L4_WGT_WORDS+L5_WGT_WORDS+L6_WGT_WORDS+L7_WGT_WORDS+L8_WGT_WORDS);
      $display("TB_INFO: expected Mode1/Mode2 OFM->IFM IFM-entry streams >= %0d", EXP_OFM2IFM_STREAMS);
      $display("TB_INFO: expected final OFM logical elements=%0d", EXPECTED_FINAL_ELEMENTS);
      $display("TB_INFO: expected final OFM DDR words=%0d using Mode2 layout F*H*ceil(W/PC) = %0d*%0d*ceil(%0d/%0d)", EXPECTED_OFM_DDR_WORDS, L8_F_OUT, L8_H_OUT, L8_W_OUT, PC);
      $display("TB_INFO: expected L0(M1)->L1(M2) transition IFM writes=%0d", EXPECTED_L0_STREAM_IFM_WRITES);
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
      $display("Stream counts at stop: start=%0d done=%0d legacy_m2_mgr_req=%0d expected>=%0d", ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, legacy_m2_mgr_req_count, EXP_OFM2IFM_STREAMS);
      dump_ofm_region();
      $fatal(1, "TB_FAIL: DUT error before successful completion");
    end

    if (cycle_count > MAX_CYCLES) begin
      $display("TB_FAIL: timeout after %0d cycles. busy=%0b done=%0b error=%0b layer=%0d mode=%0d vec=%04b", cycle_count, busy, done, error, dbg_layer_idx, dbg_mode, dbg_error_vec);
      $display("DDR counts at timeout: ifm_reads=%0d wgt_reads=%0d ofm_writes=%0d", ddr_ifm_read_count, ddr_wgt_read_count, ddr_ofm_write_count);
      $display("Stream counts at timeout: start=%0d done=%0d legacy_m2_mgr_req=%0d expected>=%0d", ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, legacy_m2_mgr_req_count, EXP_OFM2IFM_STREAMS);
      dump_ofm_region();
      $fatal(1, "TB_FAIL: timeout");
    end

    if (!done_seen) $fatal(1, "TB_FAIL: stopped without done_seen");

    $display("TB_INFO: 9-layer mixed Mode1->Mode2 done after %0d cycles", cycle_count);
    $display("TB_INFO: DDR counts: ifm_reads=%0d est=%0d, wgt_reads=%0d est=%0d, ofm_writes=%0d expected=%0d", ddr_ifm_read_count, EXPECTED_IFM_DDR_READS, ddr_wgt_read_count, EXPECTED_WGT_DDR_READS, ddr_ofm_write_count, EXPECTED_OFM_DDR_WORDS);
    $display("TB_INFO: OFM->IFM stream starts=%0d done=%0d expected>=%0d, legacy_m2_mgr_req=%0d", ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, EXP_OFM2IFM_STREAMS, legacy_m2_mgr_req_count);
    $display("TB_INFO: L0(M1)->L1(M2) transition IFM-write checks=%0d mismatch=%0d", l0_stream_checked_count, l0_stream_mismatch_count);

    // Do not assert exact IFM DDR read count in this mixed test: L0 is Mode1
    // and the goal is to avoid any Mode2 DDR->IFM first-layer refill dependency.
    if (l0_stream_checked_count != EXPECTED_L0_STREAM_IFM_WRITES) $fatal(1, "TB_FAIL: unexpected L0->L1 IFM stream write count");
    if (l0_stream_mismatch_count != 0) $fatal(1, "TB_FAIL: L0(M1)->L1(M2) IFM transition mismatch count=%0d", l0_stream_mismatch_count);
    // Do not assert exact WGT DDR read count in the mixed test; L0 is Mode1
    // and physical preload accounting can differ from the all-Mode2 case.
    if (ddr_ofm_write_count != EXPECTED_OFM_DDR_WORDS) begin
      dump_ofm_region();
      $fatal(1, "TB_FAIL: unexpected OFM DDR write count");
    end
    if (ofm_ifm_stream_done_count < EXP_OFM2IFM_STREAMS) $fatal(1, "TB_FAIL: insufficient OFM->IFM stream done count");
    if (legacy_m2_mgr_req_count != 0) $fatal(1, "TB_FAIL: legacy Mode2 refill-manager request fired count=%0d", legacy_m2_mgr_req_count);

    check_final_ofm();
    $display("TB_PASS: 9-layer mixed Mode1->Mode2 Test-B 160x192x3 K3 expected-compare passed. final_value=%0d elements=%0d ddr_words=%0d", EXPECTED_FINAL_VALUE, EXPECTED_FINAL_ELEMENTS, EXPECTED_OFM_DDR_WORDS);
    $finish;
  end
  
`ifdef DBG_TB_M2_READY_TOKEN
  // Optional visibility-only monitor.  The Mode2 main refill path must not rely
  // on m2_sm_ready_* tokens; they are printed here only to debug OFM final-entry
  // readiness.  Index the arrays explicitly to avoid packed/unpacked display
  // artifacts that look like huge bogus row/col values.
  integer tb_m2_ready_i;
  always @(posedge clk) begin
    if (rst_n) begin
      for (tb_m2_ready_i = 0; tb_m2_ready_i < PF; tb_m2_ready_i = tb_m2_ready_i + 1) begin
        if (dut.u_ofm_buffer.m2_sm_ready_valid[tb_m2_ready_i]) begin
          $display("DBG_TB_M2_READY_TOKEN t=%0t idx=%0d layer=%0d pool_en=%0b row=%0d col=%0d cgrp=%0d final_h=%0d final_w=%0d",
                   $time,
                   tb_m2_ready_i,
                   dut.u_control_unit_top.cur_cfg_s.layer_id,
                   dut.u_control_unit_top.cur_cfg_s.pool_en,
                   dut.u_ofm_buffer.m2_sm_ready_row_g[tb_m2_ready_i],
                   dut.u_ofm_buffer.m2_sm_ready_colbase_g[tb_m2_ready_i],
                   dut.u_ofm_buffer.m2_sm_ready_bank[tb_m2_ready_i],
                   dut.u_control_unit_top.ofm_cfg_h_out,
                   dut.u_control_unit_top.ofm_cfg_w_out);
        end
      end
    end
  end
`endif

// -----------------------------------------------------------------------------
// DEBUG: Mode2 OFM->IFM stream tag monitor
// Purpose:
//   Distinguish whether col32/33 have actually been written into IFM bank0/bank1
//   before addr_gen tries to read them.
// -----------------------------------------------------------------------------
// Enable with: +define+DBG_M2_STREAM_REFILL_TAG
// -----------------------------------------------------------------------------

`define DBG_CU   dut.u_control_unit_top
`define DBG_IFM  dut.u_ifm_buffer
`define DBG_LDM  dut.u_control_unit_top.u_local_dataflow_manager

function automatic bit dbg_m2_focus_col(input [15:0] col);
  begin
    dbg_m2_focus_col =
      (col == 16'd0 ) ||
      (col == 16'd1 ) ||
      (col == 16'd2 ) ||
      (col == 16'd30) ||
      (col == 16'd31) ||
      (col == 16'd32) ||
      (col == 16'd33) ||
      (col == 16'd34) ||
      (col == 16'd35) ||
      (col == 16'd64);
  end
endfunction

function automatic bit dbg_m2_focus_row(input [15:0] row);
  begin
    // For K=3, when out_row=0 and out_col=30,
    // the critical rows are 0,1,2.
    // Also print row 3 for margin.
    dbg_m2_focus_row = (row <= 16'd3);
  end
endfunction

logic        dbg_m2_stream_active_q;
logic [15:0] dbg_m2_stream_row_q;
logic [15:0] dbg_m2_stream_col_q;
logic [15:0] dbg_m2_stream_cgrp_q;

logic [2:0]  dbg_m2_col32_rows_seen_q;
logic [2:0]  dbg_m2_col33_rows_seen_q;
logic [2:0]  dbg_m2_col34_rows_seen_q;

always @(posedge clk) begin
  if (!rst_n) begin
    dbg_m2_stream_active_q   <= 1'b0;
    dbg_m2_stream_row_q      <= 16'd0;
    dbg_m2_stream_col_q      <= 16'd0;
    dbg_m2_stream_cgrp_q     <= 16'd0;
    dbg_m2_col32_rows_seen_q <= 3'b000;
    dbg_m2_col33_rows_seen_q <= 3'b000;
    dbg_m2_col34_rows_seen_q <= 3'b000;
  end else begin

    // Capture every OFM->IFM stream command.
    if (`DBG_CU.ofm_ifm_stream_start) begin
      dbg_m2_stream_active_q <= 1'b1;
      dbg_m2_stream_row_q    <= `DBG_CU.ofm_ifm_stream_row_base;
      dbg_m2_stream_col_q    <= `DBG_CU.ofm_ifm_stream_col_base;
      dbg_m2_stream_cgrp_q   <= `DBG_CU.ofm_ifm_stream_m2_cgrp_g;

      if (dbg_m2_focus_col(`DBG_CU.ofm_ifm_stream_col_base)) begin
        $display("DBG_M2_STREAM_CMD_TAG t=%0t layer=%0d mode=%0d start row_base=%0d global_col=%0d expected_bank=%0d cgrp=%0d active_seen32=%b seen33=%b seen34=%b",
                 $time,
                 `DBG_CU.cur_cfg_s.layer_id,
                 `DBG_CU.cur_cfg_s.mode,
                 `DBG_CU.ofm_ifm_stream_row_base,
                 `DBG_CU.ofm_ifm_stream_col_base,
                 (`DBG_CU.ofm_ifm_stream_col_base % 32),
                 `DBG_CU.ofm_ifm_stream_m2_cgrp_g,
                 dbg_m2_col32_rows_seen_q,
                 dbg_m2_col33_rows_seen_q,
                 dbg_m2_col34_rows_seen_q);
      end
    end

    // Tag actual IFM writes coming from OFM stream.
    // This is the important part: it links physical bank write to global stream col.
    if (`DBG_IFM.ofm_wr_en && `DBG_IFM.ofm_wr_ready) begin
      if (dbg_m2_stream_active_q &&
          dbg_m2_focus_col(dbg_m2_stream_col_q) &&
          dbg_m2_focus_row(`DBG_IFM.ofm_wr_row_idx)) begin

        $display("DBG_M2_IFM_OFM_WR_TAG t=%0t src_global_col=%0d expected_bank=%0d actual_bank=%0d row=%0d cgrp=%0d keep=%h data0=%0d data1=%0d active=%0b",
                 $time,
                 dbg_m2_stream_col_q,
                 (dbg_m2_stream_col_q % 32),
                 `DBG_IFM.ofm_wr_bank,
                 `DBG_IFM.ofm_wr_row_idx,
                 `DBG_IFM.ofm_wr_col_idx,
                 `DBG_IFM.ofm_wr_keep,
                 $signed(`DBG_IFM.ofm_wr_data[0*8 +: 8]),
                 $signed(`DBG_IFM.ofm_wr_data[1*8 +: 8]),
                 dbg_m2_stream_active_q);
      end

      // Record whether col32/33/34 have actually reached rows 0,1,2.
      if (dbg_m2_stream_active_q &&
          (`DBG_IFM.ofm_wr_col_idx == 0) &&
          (`DBG_IFM.ofm_wr_row_idx < 3)) begin

        if (dbg_m2_stream_col_q == 16'd32) begin
          dbg_m2_col32_rows_seen_q[`DBG_IFM.ofm_wr_row_idx] <= 1'b1;
        end

        if (dbg_m2_stream_col_q == 16'd33) begin
          dbg_m2_col33_rows_seen_q[`DBG_IFM.ofm_wr_row_idx] <= 1'b1;
        end

        if (dbg_m2_stream_col_q == 16'd34) begin
          dbg_m2_col34_rows_seen_q[`DBG_IFM.ofm_wr_row_idx] <= 1'b1;
        end
      end
    end

    if (`DBG_CU.ofm_ifm_stream_done) begin
      if (dbg_m2_focus_col(dbg_m2_stream_col_q)) begin
        $display("DBG_M2_STREAM_DONE_TAG t=%0t done_for_global_col=%0d expected_bank=%0d cgrp=%0d seen32=%b seen33=%b seen34=%b",
                 $time,
                 dbg_m2_stream_col_q,
                 (dbg_m2_stream_col_q % 32),
                 dbg_m2_stream_cgrp_q,
                 dbg_m2_col32_rows_seen_q,
                 dbg_m2_col33_rows_seen_q,
                 dbg_m2_col34_rows_seen_q);
      end

      dbg_m2_stream_active_q <= 1'b0;
    end

    // Print free-token requests that should lead to runtime refill.
    // For L1 K=3, these are important once Mode2 compute starts.
    if (`DBG_LDM.m2_free_valid &&
        (dbg_m2_focus_col(`DBG_LDM.m2_free_col_g))) begin
      $display("DBG_M2_FREE_TOKEN_TAG t=%0t layer=%0d free_row=%0d refill_global_col=%0d expected_bank=%0d col_l=%0d cgrp=%0d seen32=%b seen33=%b seen34=%b",
               $time,
               `DBG_CU.cur_cfg_s.layer_id,
               `DBG_LDM.m2_free_row_g,
               `DBG_LDM.m2_free_col_g,
               (`DBG_LDM.m2_free_col_g % 32),
               `DBG_LDM.m2_free_col_l,
               `DBG_LDM.m2_free_cgrp_g,
               dbg_m2_col32_rows_seen_q,
               dbg_m2_col33_rows_seen_q,
               dbg_m2_col34_rows_seen_q);
    end

    // Print final status exactly when DUT errors.
    if (dut.error) begin
      $display("DBG_M2_REFILL_STATUS_ON_ERROR t=%0t vec=%04b bit0_dma=%0b bit1_ofm=%0b bit2_local=%0b bit3_transition=%0b layer=%0d mode=%0d seen_col32_rows012=%b seen_col33_rows012=%b seen_col34_rows012=%b active_stream=%0b active_col=%0d active_cgrp=%0d",
               $time,
               dut.dbg_error_vec,
               dut.dbg_error_vec[0],
               dut.dbg_error_vec[1],
               dut.dbg_error_vec[2],
               dut.dbg_error_vec[3],
               `DBG_CU.cur_cfg_s.layer_id,
               `DBG_CU.cur_cfg_s.mode,
               dbg_m2_col32_rows_seen_q,
               dbg_m2_col33_rows_seen_q,
               dbg_m2_col34_rows_seen_q,
               dbg_m2_stream_active_q,
               dbg_m2_stream_col_q,
               dbg_m2_stream_cgrp_q);
    end
  end
end

`undef DBG_CU
`undef DBG_IFM
`undef DBG_LDM



endmodule
