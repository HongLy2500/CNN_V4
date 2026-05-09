`timescale 1ns/1ps
`include "cnn_ddr_defs.svh"

module tb_cnn_top_9layer_m2_dcp_efficientnet_b0_tablevi_w32_with_expected_compare;
  import cnn_layer_desc_pkg::*;

  // --------------------------------------------------------------------------
  // 9-layer Mode-2 DCP/EfficientNet-B0 Table-VI-style test with expected compare.
  //
  // Why W0=32 instead of 224:
  //   The current cnn_dma_direct Mode-2 first-layer IFM DMA path only accepts
  //   cfg_w_in <= PC. Therefore this test uses W0=PC=32 so it runs on the
  //   current RTL without changing DMA/control. It keeps the 9-layer channel
  //   trend and Table-VI Mode-2 parallelism point: Pc=32, Pf=64.
  //
  // Data model:
  //   IFM = 1, all weights = 1. Expected final OFM is checked exactly.
  //   With K=1, the values are:
  //     L0: 1 * C0(3)  = 3
  //     L1: 3 * C1(32) = 96
  //     L2 onward saturates to signed 8-bit max = 127
  //
  // This stresses:
  //   - Mode-2 first-layer DDR->IFM preload
  //   - Mode2->Mode2 deterministic OFM->IFM handoff over 8 transitions
  //   - partial C groups, especially C=3,16,24,40,80 with PC=32
  //   - partial F groups for F=16,24,40,80,192 with PF=64
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
  localparam int W_MAX = 32;
  localparam int H_MAX = 64;
  localparam int HT = 4;
  localparam int K_MAX = 3;

  localparam int WGT_DEPTH = 512;
  localparam int OFM_ROW_STRIDE = W_MAX;
  localparam int OFM_BANK_DEPTH = H_MAX * OFM_ROW_STRIDE;
  localparam int OFM_LINEAR_DEPTH = C_MAX * OFM_BANK_DEPTH;
  localparam int CFG_DEPTH = 16;

  localparam int DDR_ADDR_W = `CNN_DDR_ADDR_W;
  localparam int DDR_WORD_W = PV_MAX * DATA_W;
  localparam int DDR_LANES = PV_MAX;
  localparam int WGT_SUBWORDS = (PTOTAL + PV_MAX - 1) / PV_MAX;
  localparam int MEM_DEPTH = (`DDR_RSVD_BASE + `DDR_RSVD_SIZE);
  localparam int CLK_PERIOD_NS = 10;
  localparam int MAX_CYCLES = 20000000;

  // 9-layer EfficientNet-B0 channel trend used in the Mode-1 Table-VI test.
  // K is set to 1 here so current Mode-2 W0=PC constraint can still exercise
  // all 9 layers without shrinking width below 1.
  localparam int L0_H_IN=64, L0_W_IN=32, L0_C_IN=3,  L0_F_OUT=32,  L0_K=1, L0_POOL_EN=1;
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

  localparam int EXPECTED_IFM_DDR_READS = L0_C_IN * L0_H_IN;
  localparam int EXPECTED_WGT_DDR_READS = L0_WGT_DDR_WORDS+L1_WGT_DDR_WORDS+L2_WGT_DDR_WORDS+L3_WGT_DDR_WORDS+L4_WGT_DDR_WORDS+L5_WGT_DDR_WORDS+L6_WGT_DDR_WORDS+L7_WGT_DDR_WORDS+L8_WGT_DDR_WORDS;

  // OFM buffer Mode-2 final DMA layout is NOT flat H*W*F words.
  // It stores one word per {output channel, row, spatial column group}.
  // In Mode 2, store_pack = PC, so groups per row = ceil(W_out / PC).
  localparam int L8_STORED_GROUPS = (L8_W_OUT + PC - 1) / PC;
  localparam int EXPECTED_FINAL_ELEMENTS = L8_F_OUT * L8_H_OUT * L8_W_OUT;
  localparam int EXPECTED_OFM_DDR_WORDS = L8_F_OUT * L8_H_OUT * L8_STORED_GROUPS;

  // Derived from this testbench, not assumed:
  //   init_mem() writes IFM=1 and all layer weights=1.
  //   K=1 for every layer.
  //   L0 = 3, L1 = 96, and L2 onward saturates to signed 8-bit max 127.
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

integer dbg_l8_wr_cnt;

always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    dbg_l8_wr_cnt <= 0;
  end else begin
    if (dbg_layer_idx == 8 && dut.u_mode2_compute_top.ofm_wr_en && dbg_l8_wr_cnt < 80) begin
      dbg_l8_wr_cnt <= dbg_l8_wr_cnt + 1;

      $display("DBG_L8_M2_OFM_IN t=%0t cycle=%0d row=%0d col=%0d fbase=%0d data0=%0d data1=%0d data2=%0d data63=%0d",
        $time,
        cycle_count,
        dut.u_mode2_compute_top.ofm_wr_row,
        dut.u_mode2_compute_top.ofm_wr_col,
        dut.u_mode2_compute_top.ofm_wr_f_base,
        $signed(dut.u_mode2_compute_top.ofm_wr_data[0*DATA_W +: DATA_W]),
        $signed(dut.u_mode2_compute_top.ofm_wr_data[1*DATA_W +: DATA_W]),
        $signed(dut.u_mode2_compute_top.ofm_wr_data[2*DATA_W +: DATA_W]),
        $signed(dut.u_mode2_compute_top.ofm_wr_data[63*DATA_W +: DATA_W])
      );
    end
  end
end

integer dbg_l8_ofm_buf_cnt;

always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    dbg_l8_ofm_buf_cnt <= 0;
  end else begin
    if (dbg_layer_idx == 8 && dut.u_ofm_buffer.m2_wr_en && dbg_l8_ofm_buf_cnt < 80) begin
      dbg_l8_ofm_buf_cnt <= dbg_l8_ofm_buf_cnt + 1;

      $display("DBG_L8_OFM_BUF_IN t=%0t cycle=%0d row=%0d col=%0d fbase=%0d data0=%0d data1=%0d data2=%0d data63=%0d",
        $time,
        cycle_count,
        dut.u_ofm_buffer.m2_wr_row,
        dut.u_ofm_buffer.m2_wr_col,
        dut.u_ofm_buffer.m2_wr_f_base,
        $signed(dut.u_ofm_buffer.m2_wr_data[0*DATA_W +: DATA_W]),
        $signed(dut.u_ofm_buffer.m2_wr_data[1*DATA_W +: DATA_W]),
        $signed(dut.u_ofm_buffer.m2_wr_data[2*DATA_W +: DATA_W]),
        $signed(dut.u_ofm_buffer.m2_wr_data[63*DATA_W +: DATA_W])
      );
    end
  end
end

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

      if (dut.ofm_ifm_stream_start_s) ofm_ifm_stream_start_count <= ofm_ifm_stream_start_count + 1;
      if (dut.ofm_ifm_stream_done_s) ofm_ifm_stream_done_count <= ofm_ifm_stream_done_count + 1;
      if (m2_sm_refill_req_valid && m2_sm_refill_req_ready) m2_sm_refill_req_count <= m2_sm_refill_req_count + 1;
    end
  end

  always_ff @(posedge clk) begin
    if (rst_n) begin
      if (dut.ofm_ifm_stream_start_s || dut.ofm_ifm_stream_done_s || done || error || ((cycle_count % 20000) == 0)) begin
        $display("DBG_TOP_STATUS t=%0t cycle=%0d busy=%0b done=%0b error=%0b vec=%04b layer=%0d mode=%0d ifm_rd=%0d wgt_rd=%0d ofm_wr=%0d stream_start=%0d stream_done=%0d legacy_m2_req=%0d", $time, cycle_count, busy, done, error, dbg_error_vec, dbg_layer_idx, dbg_mode, ddr_ifm_read_count, ddr_wgt_read_count, ddr_ofm_write_count, ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, m2_sm_refill_req_count);
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

      // Mode-2 first-layer IFM DDR layout for current DMA: [channel][row].
      // One DDR word holds PC pixels. W0 is exactly PC in this test.
      word_idx = 0;
      for (ch = 0; ch < L0_C_IN; ch = ch + 1) begin
        for (row = 0; row < L0_H_IN; row = row + 1) begin
          ddr_mem[`DDR_IFM_BASE + word_idx] = pack_ddr_ones_word();
          word_idx = word_idx + 1;
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
            $display("TB_MISMATCH_M2_9L word=%0d got=0x%0h exp=0x%0h", j, ddr_mem[`DDR_OFM_BASE + j], exp_word);
          end
          mismatch = mismatch + 1;
        end
      end
      if (mismatch != 0) begin
        dump_ofm_region();
        $fatal(1, "TB_FAIL: 9-layer Mode2 final OFM mismatch count=%0d", mismatch);
      end
    end
  endtask

  task automatic print_banner;
    begin
      $display("TB_INFO: 9-layer Mode2 DCP/EfficientNet-B0 Table-VI-style W32 expected-compare test");
      $display("TB_INFO: current RTL Mode2 DMA requires W_in<=PC, so this test uses input %0dx%0dx%0d instead of 224x224x3", L0_H_IN, L0_W_IN, L0_C_IN);
      $display("TB_INFO: Mode2 PC=%0d PF=%0d PTOTAL=%0d PV_MAX=%0d WGT_SUBWORDS=%0d", PC, PF, PTOTAL, PV_MAX, WGT_SUBWORDS);
      $display("TB_INFO: channel trend: 3 -> 32 -> 16 -> 24 -> 24 -> 40 -> 40 -> 80 -> 80 -> 192");
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

    if (ddr_ifm_read_count != EXPECTED_IFM_DDR_READS) $fatal(1, "TB_FAIL: unexpected IFM DDR read count");
    if (ddr_wgt_read_count != EXPECTED_WGT_DDR_READS) $fatal(1, "TB_FAIL: unexpected WGT DDR read count");
    if (ddr_ofm_write_count != EXPECTED_OFM_DDR_WORDS) begin
      dump_ofm_region();
      $fatal(1, "TB_FAIL: unexpected OFM DDR write count");
    end
    if (ofm_ifm_stream_done_count < EXP_OFM2IFM_STREAMS) $fatal(1, "TB_FAIL: insufficient OFM->IFM stream done count");
    if (m2_sm_refill_req_count != 0) $display("TB_WARN: legacy M2 refill request count is nonzero: %0d", m2_sm_refill_req_count);

    check_final_ofm();
    $display("TB_PASS: 9-layer Mode2 DCP/EfficientNet-B0 Table-VI-style W32 expected-compare passed. final_value=%0d elements=%0d ddr_words=%0d", EXPECTED_FINAL_VALUE, EXPECTED_FINAL_ELEMENTS, EXPECTED_OFM_DDR_WORDS);
    $finish;
  end
endmodule
