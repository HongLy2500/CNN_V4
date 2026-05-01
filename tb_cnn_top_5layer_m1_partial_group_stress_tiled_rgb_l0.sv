`timescale 1ns/1ps
`include "cnn_ddr_defs.svh"

module tb_cnn_top_5layer_m1_partial_group_stress_tiled_rgb_l0;
  import cnn_layer_desc_pkg::*;

  localparam int DATA_W=8, PSUM_W=32;
  localparam int PTOTAL=8, PV_MAX=4, PF_MAX=4, PV_M1=2, PF_M1=4;
  localparam int PC_MODE2=4, PF_MODE2=2;
  localparam int C_MAX=16, F_MAX=16, W_MAX=49, H_MAX=69, HT=4, K_MAX=3;
  localparam int WGT_DEPTH=2048, OFM_BANK_DEPTH=H_MAX*W_MAX;
  localparam int OFM_LINEAR_DEPTH=C_MAX*OFM_BANK_DEPTH, CFG_DEPTH=8;
  localparam int DDR_ADDR_W=`CNN_DDR_ADDR_W, DDR_WORD_W=PV_MAX*DATA_W;
  localparam int MEM_DEPTH=(`DDR_RSVD_BASE + `DDR_RSVD_SIZE);
  localparam int CLK_PERIOD_NS=10, MAX_CYCLES=12000000;

  // 5-layer Mode-1 stress test with intentionally non-divisible/partial groups.
  // Stresses: H>HT, W>Pv and odd W, C/PF partial channel blocks (L0 is RGB C=3), F/PF partial filter groups,
  // DDR->IFM tiled refill for L0 and OFM->IFM tiled refill for L1..L4.
  // L0: 69x49x3 -> conv 67x47x10 K=3 -> pool 33x23x10
  // L1: 33x23x10 -> conv 31x21x13 K=3 -> pool 15x10x13
  // L2: 15x10x13 -> conv 13x8x9 K=3 -> pool 6x4x9
  // L3: 6x4x9 -> conv 6x4x15 K=1 -> pool 3x2x15
  // L4: 3x2x15 -> conv 3x2x11 K=1 -> pool 1x1x11

  localparam int L0_H_IN=69, L0_W_IN=49, L0_C_IN=3, L0_F_OUT=10, L0_K=3;
  localparam int L0_H_CONV_OUT=L0_H_IN-L0_K+1, L0_W_CONV_OUT=L0_W_IN-L0_K+1;
  localparam int L0_H_POOL_OUT=L0_H_CONV_OUT/2, L0_W_POOL_OUT=L0_W_CONV_OUT/2;

  localparam int L1_H_IN=L0_H_POOL_OUT, L1_W_IN=L0_W_POOL_OUT, L1_C_IN=L0_F_OUT;
  localparam int L1_F_OUT=13, L1_K=3;
  localparam int L1_H_CONV_OUT=L1_H_IN-L1_K+1, L1_W_CONV_OUT=L1_W_IN-L1_K+1;
  localparam int L1_H_POOL_OUT=L1_H_CONV_OUT/2, L1_W_POOL_OUT=L1_W_CONV_OUT/2;

  localparam int L2_H_IN=L1_H_POOL_OUT, L2_W_IN=L1_W_POOL_OUT, L2_C_IN=L1_F_OUT;
  localparam int L2_F_OUT=9, L2_K=3;
  localparam int L2_H_CONV_OUT=L2_H_IN-L2_K+1, L2_W_CONV_OUT=L2_W_IN-L2_K+1;
  localparam int L2_H_POOL_OUT=L2_H_CONV_OUT/2, L2_W_POOL_OUT=L2_W_CONV_OUT/2;

  localparam int L3_H_IN=L2_H_POOL_OUT, L3_W_IN=L2_W_POOL_OUT, L3_C_IN=L2_F_OUT;
  localparam int L3_F_OUT=15, L3_K=1;
  localparam int L3_H_CONV_OUT=L3_H_IN-L3_K+1, L3_W_CONV_OUT=L3_W_IN-L3_K+1;
  localparam int L3_H_POOL_OUT=L3_H_CONV_OUT/2, L3_W_POOL_OUT=L3_W_CONV_OUT/2;

  localparam int L4_H_IN=L3_H_POOL_OUT, L4_W_IN=L3_W_POOL_OUT, L4_C_IN=L3_F_OUT;
  localparam int L4_F_OUT=11, L4_K=1;
  localparam int L4_H_CONV_OUT=L4_H_IN-L4_K+1, L4_W_CONV_OUT=L4_W_IN-L4_K+1;
  localparam int L4_H_POOL_OUT=L4_H_CONV_OUT/2, L4_W_POOL_OUT=L4_W_CONV_OUT/2;

  localparam int L0_IFM_WORDS_PER_ROW=(L0_W_IN+PV_M1-1)/PV_M1;
  localparam int L0_EXPECT_IFM_DDR_WRITES=L0_C_IN*L0_H_IN*L0_IFM_WORDS_PER_ROW;
  localparam int L0_EXPECT_IFM_DDR_WRITES_AFTER_START=(L0_H_IN-HT)*L0_C_IN*L0_IFM_WORDS_PER_ROW;

  localparam int WGT_SUBWORDS=(PTOTAL+PV_MAX-1)/PV_MAX;
  localparam int L0_NUM_FGROUP=(L0_F_OUT+PF_M1-1)/PF_M1;
  localparam int L0_M1_LOGICAL_BUNDLES=L0_NUM_FGROUP*L0_C_IN*L0_K*L0_K;
  localparam int L0_M1_PHYS_WORDS=(L0_M1_LOGICAL_BUNDLES+PV_M1-1)/PV_M1;
  localparam int L0_M1_WGT_DDR_WORDS=L0_M1_PHYS_WORDS*WGT_SUBWORDS;
  localparam int L1_NUM_FGROUP=(L1_F_OUT+PF_M1-1)/PF_M1;
  localparam int L1_M1_LOGICAL_BUNDLES=L1_NUM_FGROUP*L1_C_IN*L1_K*L1_K;
  localparam int L1_M1_PHYS_WORDS=(L1_M1_LOGICAL_BUNDLES+PV_M1-1)/PV_M1;
  localparam int L1_M1_WGT_DDR_WORDS=L1_M1_PHYS_WORDS*WGT_SUBWORDS;
  localparam int L2_NUM_FGROUP=(L2_F_OUT+PF_M1-1)/PF_M1;
  localparam int L2_M1_LOGICAL_BUNDLES=L2_NUM_FGROUP*L2_C_IN*L2_K*L2_K;
  localparam int L2_M1_PHYS_WORDS=(L2_M1_LOGICAL_BUNDLES+PV_M1-1)/PV_M1;
  localparam int L2_M1_WGT_DDR_WORDS=L2_M1_PHYS_WORDS*WGT_SUBWORDS;
  localparam int L3_NUM_FGROUP=(L3_F_OUT+PF_M1-1)/PF_M1;
  localparam int L3_M1_LOGICAL_BUNDLES=L3_NUM_FGROUP*L3_C_IN*L3_K*L3_K;
  localparam int L3_M1_PHYS_WORDS=(L3_M1_LOGICAL_BUNDLES+PV_M1-1)/PV_M1;
  localparam int L3_M1_WGT_DDR_WORDS=L3_M1_PHYS_WORDS*WGT_SUBWORDS;
  localparam int L4_NUM_FGROUP=(L4_F_OUT+PF_M1-1)/PF_M1;
  localparam int L4_M1_LOGICAL_BUNDLES=L4_NUM_FGROUP*L4_C_IN*L4_K*L4_K;
  localparam int L4_M1_PHYS_WORDS=(L4_M1_LOGICAL_BUNDLES+PV_M1-1)/PV_M1;
  localparam int L4_M1_WGT_DDR_WORDS=L4_M1_PHYS_WORDS*WGT_SUBWORDS;
  localparam int L0_WGT_DDR_BASE=`DDR_WGT_BASE;
  localparam int L1_WGT_DDR_BASE=L0_WGT_DDR_BASE+L0_M1_WGT_DDR_WORDS;
  localparam int L2_WGT_DDR_BASE=L1_WGT_DDR_BASE+L1_M1_WGT_DDR_WORDS;
  localparam int L3_WGT_DDR_BASE=L2_WGT_DDR_BASE+L2_M1_WGT_DDR_WORDS;
  localparam int L4_WGT_DDR_BASE=L3_WGT_DDR_BASE+L3_M1_WGT_DDR_WORDS;

  localparam int FINAL_STORE_PACK=1;
  localparam int FINAL_GROUPS=(L4_W_POOL_OUT+FINAL_STORE_PACK-1)/FINAL_STORE_PACK;
  localparam int EXP_OFM_WORDS=L4_F_OUT*L4_H_POOL_OUT*FINAL_GROUPS;

  localparam int L1_OFM2IFM_COL_BLKS=(L1_W_IN+PV_M1-1)/PV_M1;
  localparam int L1_OFM2IFM_CH_BLKS=(L1_C_IN+PF_M1-1)/PF_M1;
  localparam int L2_OFM2IFM_COL_BLKS=(L2_W_IN+PV_M1-1)/PV_M1;
  localparam int L2_OFM2IFM_CH_BLKS=(L2_C_IN+PF_M1-1)/PF_M1;
  localparam int L3_OFM2IFM_COL_BLKS=(L3_W_IN+PV_M1-1)/PV_M1;
  localparam int L3_OFM2IFM_CH_BLKS=(L3_C_IN+PF_M1-1)/PF_M1;
  localparam int L4_OFM2IFM_COL_BLKS=(L4_W_IN+PV_M1-1)/PV_M1;
  localparam int L4_OFM2IFM_CH_BLKS=(L4_C_IN+PF_M1-1)/PF_M1;
  localparam int EXP_OFM2IFM_STREAMS=(L1_H_IN*L1_OFM2IFM_COL_BLKS*L1_OFM2IFM_CH_BLKS) +
                                     (L2_H_IN*L2_OFM2IFM_COL_BLKS*L2_OFM2IFM_CH_BLKS) +
                                     (L3_H_IN*L3_OFM2IFM_COL_BLKS*L3_OFM2IFM_CH_BLKS) +
                                     (L4_H_IN*L4_OFM2IFM_COL_BLKS*L4_OFM2IFM_CH_BLKS);


  logic clk,rst_n,start,abort;
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
  logic busy,done,error;
  logic [$clog2(CFG_DEPTH)-1:0] dbg_layer_idx;
  logic dbg_mode, dbg_weight_bank;
  logic [3:0] dbg_error_vec;

  logic [DDR_WORD_W-1:0] ddr_mem [0:MEM_DEPTH-1];
  logic rd_pending_q;
  logic [DDR_ADDR_W-1:0] rd_addr_q;
  integer ddr_ofm_write_count, cycle_count, b;
  integer ifm_cmd_start_count, ifm_dma_wr_count, ifm_dma_wr_after_m1_start_count;
  integer ifm_free_count, old_m1_sm_refill_req_count;
  integer ofm_ifm_stream_start_count, ofm_ifm_stream_done_count;
  logic seen_m1_start_q;

  logic signed [DATA_W-1:0] l0_ifm [0:L0_C_IN-1][0:L0_H_IN-1][0:L0_W_IN-1];
  logic signed [DATA_W-1:0] l0_wgt [0:L0_F_OUT-1][0:L0_C_IN-1][0:L0_K-1][0:L0_K-1];
  integer signed l0_relu [0:L0_F_OUT-1][0:L0_H_CONV_OUT-1][0:L0_W_CONV_OUT-1];
  logic signed [DATA_W-1:0] l0_pool [0:L0_F_OUT-1][0:L0_H_POOL_OUT-1][0:L0_W_POOL_OUT-1];

  logic signed [DATA_W-1:0] l1_wgt [0:L1_F_OUT-1][0:L1_C_IN-1][0:L1_K-1][0:L1_K-1];
  integer signed l1_relu [0:L1_F_OUT-1][0:L1_H_CONV_OUT-1][0:L1_W_CONV_OUT-1];
  logic signed [DATA_W-1:0] l1_pool [0:L1_F_OUT-1][0:L1_H_POOL_OUT-1][0:L1_W_POOL_OUT-1];

  logic signed [DATA_W-1:0] l2_wgt [0:L2_F_OUT-1][0:L2_C_IN-1][0:L2_K-1][0:L2_K-1];
  integer signed l2_relu [0:L2_F_OUT-1][0:L2_H_CONV_OUT-1][0:L2_W_CONV_OUT-1];
  logic signed [DATA_W-1:0] l2_pool [0:L2_F_OUT-1][0:L2_H_POOL_OUT-1][0:L2_W_POOL_OUT-1];

  logic signed [DATA_W-1:0] l3_wgt [0:L3_F_OUT-1][0:L3_C_IN-1][0:L3_K-1][0:L3_K-1];
  integer signed l3_relu [0:L3_F_OUT-1][0:L3_H_CONV_OUT-1][0:L3_W_CONV_OUT-1];
  logic signed [DATA_W-1:0] l3_pool [0:L3_F_OUT-1][0:L3_H_POOL_OUT-1][0:L3_W_POOL_OUT-1];

  logic signed [DATA_W-1:0] l4_wgt [0:L4_F_OUT-1][0:L4_C_IN-1][0:L4_K-1][0:L4_K-1];
  integer signed l4_relu [0:L4_F_OUT-1][0:L4_H_CONV_OUT-1][0:L4_W_CONV_OUT-1];
  logic signed [DATA_W-1:0] l4_pool [0:L4_F_OUT-1][0:L4_H_POOL_OUT-1][0:L4_W_POOL_OUT-1];


  cnn_top #(
    .DATA_W(DATA_W), .PSUM_W(PSUM_W), .PTOTAL(PTOTAL), .PV_MAX(PV_MAX), .PF_MAX(PF_MAX),
    .PC_MODE2(PC_MODE2), .PF_MODE2(PF_MODE2), .C_MAX(C_MAX), .F_MAX(F_MAX),
    .W_MAX(W_MAX), .H_MAX(H_MAX), .HT(HT), .K_MAX(K_MAX), .WGT_DEPTH(WGT_DEPTH),
    .OFM_BANK_DEPTH(OFM_BANK_DEPTH), .OFM_LINEAR_DEPTH(OFM_LINEAR_DEPTH),
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
    .ifm_m1_free_valid(ifm_m1_free_valid), .ifm_m1_free_row_slot_l(ifm_m1_free_row_slot_l),
    .ifm_m1_free_row_g(ifm_m1_free_row_g),
    .busy(busy), .done(done), .error(error),
    .dbg_layer_idx(dbg_layer_idx), .dbg_mode(dbg_mode), .dbg_weight_bank(dbg_weight_bank),
    .dbg_error_vec(dbg_error_vec)
  );

  initial clk=1'b0;
  always #(CLK_PERIOD_NS/2) clk=~clk;

// ============================================================
// DEBUG: Layer 3 stuck diagnosis
// Paste into tb_cnn_top_5layer_m1_partial_group_stress_tiled_rgb_l0.sv
// ============================================================

integer dbg_l3_cycle_count;
integer dbg_l3_stream_start_count;
integer dbg_l3_stream_done_count;
integer dbg_l3_ofm_write_pulse_count;
integer dbg_l3_ofm_valid_elem_count;
integer dbg_l3_free_count;
logic   dbg_l3_m1_done_seen;
logic   dbg_l3_ofm_layer_done_seen;

always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    dbg_l3_cycle_count           <= 0;
    dbg_l3_stream_start_count    <= 0;
    dbg_l3_stream_done_count     <= 0;
    dbg_l3_ofm_write_pulse_count <= 0;
    dbg_l3_ofm_valid_elem_count  <= 0;
    dbg_l3_free_count            <= 0;
    dbg_l3_m1_done_seen          <= 1'b0;
    dbg_l3_ofm_layer_done_seen   <= 1'b0;
  end else begin
    if (dut.dbg_layer_idx == 3) begin
      dbg_l3_cycle_count <= dbg_l3_cycle_count + 1;

      if (dut.ofm_ifm_stream_start_s) begin
        dbg_l3_stream_start_count <= dbg_l3_stream_start_count + 1;
        $display("DBG_L3_STREAM_START t=%0t row=%0d col_base=%0d slot=%0d ch_blk=%0d",
                 $time,
                 dut.ofm_ifm_stream_row_base_s,
                 dut.ofm_ifm_stream_col_base_s,
                 dut.ofm_ifm_stream_m1_row_slot_l_s,
                 dut.ofm_ifm_stream_m1_ch_blk_g_s);
      end

      if (dut.ofm_ifm_stream_done_s) begin
        dbg_l3_stream_done_count <= dbg_l3_stream_done_count + 1;
        $display("DBG_L3_STREAM_DONE t=%0t row=%0d col_base=%0d slot=%0d ch_blk=%0d",
                 $time,
                 dut.ofm_ifm_stream_row_base_s,
                 dut.ofm_ifm_stream_col_base_s,
                 dut.ofm_ifm_stream_m1_row_slot_l_s,
                 dut.ofm_ifm_stream_m1_ch_blk_g_s);
      end

      if (dut.ifm_m1_free_valid) begin
        dbg_l3_free_count <= dbg_l3_free_count + 1;
        $display("DBG_L3_FREE t=%0t row_base_l=%0d free_slot=%0d free_row_g=%0d m1_out_row=%0d",
                 $time,
                 dut.ifm_m1_row_base_l_s,
                 dut.ifm_m1_free_row_slot_l,
                 dut.ifm_m1_free_row_g,
                 dut.m1_out_row_s);
      end

      if (dut.m1_ofm_write_en_s) begin
        dbg_l3_ofm_write_pulse_count <= dbg_l3_ofm_write_pulse_count + 1;
        dbg_l3_ofm_valid_elem_count  <= dbg_l3_ofm_valid_elem_count + dut.m1_ofm_write_count_s;

        $display("DBG_L3_OFM_WRITE t=%0t row=%0d col_base=%0d fbase=%0d count=%0d data0=%0h data1=%0h data2=%0h data3=%0h",
                 $time,
                 dut.m1_ofm_write_row_s,
                 dut.m1_ofm_write_col_base_s,
                 dut.m1_ofm_write_filter_base_s,
                 dut.m1_ofm_write_count_s,
                 dut.m1_ofm_write_data_s[0],
                 dut.m1_ofm_write_data_s[1],
                 dut.m1_ofm_write_data_s[2],
                 dut.m1_ofm_write_data_s[3]);
      end

      if (dut.m1_done_s && !dbg_l3_m1_done_seen) begin
        dbg_l3_m1_done_seen <= 1'b1;
        $display("DBG_L3_M1_DONE t=%0t", $time);
      end

      if (dut.ofm_layer_write_done_s && !dbg_l3_ofm_layer_done_seen) begin
        dbg_l3_ofm_layer_done_seen <= 1'b1;
        $display("DBG_L3_OFM_LAYER_WRITE_DONE t=%0t", $time);
      end

      if ((dbg_l3_cycle_count % 100000) == 0) begin
        $display("DBG_L3_STATUS t=%0t cycle=%0d stream_start=%0d stream_done=%0d ofm_write_pulse=%0d ofm_valid_elem=%0d free=%0d m1_busy=%0b m1_done=%0b ofm_done=%0b",
                 $time,
                 dbg_l3_cycle_count,
                 dbg_l3_stream_start_count,
                 dbg_l3_stream_done_count,
                 dbg_l3_ofm_write_pulse_count,
                 dbg_l3_ofm_valid_elem_count,
                 dbg_l3_free_count,
                 dut.m1_busy_s,
                 dut.m1_done_s,
                 dut.ofm_layer_write_done_s);
      end
    end
  end
end

final begin
  $display("DBG_L3_SUMMARY stream_start=%0d expected_L2_to_L3_total=36",
           dbg_l3_stream_start_count);
  $display("DBG_L3_SUMMARY stream_done=%0d expected_L2_to_L3_total=36",
           dbg_l3_stream_done_count);
  $display("DBG_L3_SUMMARY ofm_write_pulse=%0d expected_L3_write_pulses=24",
           dbg_l3_ofm_write_pulse_count);
  $display("DBG_L3_SUMMARY ofm_valid_elem=%0d expected_L3_valid_elements=90",
           dbg_l3_ofm_valid_elem_count);
  $display("DBG_L3_SUMMARY free_count=%0d m1_done_seen=%0b ofm_layer_done_seen=%0b",
           dbg_l3_free_count,
           dbg_l3_m1_done_seen,
           dbg_l3_ofm_layer_done_seen);
end


  always_ff @(posedge clk) begin
    if (!rst_n) begin
      ddr_rd_valid <= 1'b0; ddr_rd_data <= '0; rd_pending_q <= 1'b0; rd_addr_q <= '0;
      ddr_ofm_write_count <= 0; cycle_count <= 0;
      ifm_cmd_start_count <= 0; ifm_dma_wr_count <= 0; ifm_dma_wr_after_m1_start_count <= 0;
      ifm_free_count <= 0; old_m1_sm_refill_req_count <= 0;
      ofm_ifm_stream_start_count <= 0; ofm_ifm_stream_done_count <= 0;
      seen_m1_start_q <= 1'b0;
    end else begin
      cycle_count <= cycle_count + 1;
      ddr_rd_valid <= 1'b0;
      if (rd_pending_q) begin
        ddr_rd_valid <= 1'b1;
        ddr_rd_data <= (rd_addr_q < MEM_DEPTH) ? ddr_mem[rd_addr_q] : '0;
      end
      if (ddr_wr_en) begin
        if (ddr_wr_addr < MEM_DEPTH) begin
          for (b=0; b<(DDR_WORD_W/8); b=b+1)
            if (ddr_wr_be[b]) ddr_mem[ddr_wr_addr][8*b +: 8] <= ddr_wr_data[8*b +: 8];
        end
        if ((ddr_wr_addr >= `DDR_OFM_BASE) && (ddr_wr_addr < (`DDR_OFM_BASE + `DDR_OFM_SIZE)))
          ddr_ofm_write_count <= ddr_ofm_write_count + 1;
      end

      if (dut.m1_start_s)
        seen_m1_start_q <= 1'b1;
      if (dut.ifm_cmd_start_s)
        ifm_cmd_start_count <= ifm_cmd_start_count + 1;
      if (dut.ifm_dma_wr_en_s) begin
        ifm_dma_wr_count <= ifm_dma_wr_count + 1;
        if (seen_m1_start_q)
          ifm_dma_wr_after_m1_start_count <= ifm_dma_wr_after_m1_start_count + 1;
      end
      if (ifm_m1_free_valid)
        ifm_free_count <= ifm_free_count + 1;
      if (m1_sm_refill_req_valid && m1_sm_refill_req_ready)
        old_m1_sm_refill_req_count <= old_m1_sm_refill_req_count + 1;
      if (dut.ofm_ifm_stream_start_s)
        ofm_ifm_stream_start_count <= ofm_ifm_stream_start_count + 1;
      if (dut.ofm_ifm_stream_done_s)
        ofm_ifm_stream_done_count <= ofm_ifm_stream_done_count + 1;

      if (ddr_rd_req) begin
        rd_pending_q <= 1'b1; rd_addr_q <= ddr_rd_addr;
      end else if (rd_pending_q) begin
        rd_pending_q <= 1'b0;
      end
    end
  end

  function automatic logic signed [DATA_W-1:0] sat_to_i8(input integer signed x);
    begin
      if (x > 127) sat_to_i8 = 8'sd127;
      else if (x < -128) sat_to_i8 = -8'sd128;
      else sat_to_i8 = $signed(x);
    end
  endfunction

  task automatic build_golden;
    integer c,r,x,f,ky,kx,pr,pc,dy,dx,sum,max_v;
    begin
      for (c=0;c<L0_C_IN;c=c+1)
        for (r=0;r<L0_H_IN;r=r+1)
          for (x=0;x<L0_W_IN;x=x+1)
            l0_ifm[c][r][x] = $signed(((c*7 + r*3 + x*5 + (r*x)%7) % 7) - 3);

      for (f=0;f<L0_F_OUT;f=f+1)
        for (c=0;c<L0_C_IN;c=c+1)
          for (ky=0;ky<L0_K;ky=ky+1)
            for (kx=0;kx<L0_K;kx=kx+1)
              l0_wgt[f][c][ky][kx] = $signed(((f*7 + c*3 + ky*5 + kx*11 + 1) % 3) - 1);

      for (f=0;f<L0_F_OUT;f=f+1)
        for (r=0;r<L0_H_CONV_OUT;r=r+1)
          for (x=0;x<L0_W_CONV_OUT;x=x+1) begin
            sum=0;
            for (c=0;c<L0_C_IN;c=c+1)
              for (ky=0;ky<L0_K;ky=ky+1)
                for (kx=0;kx<L0_K;kx=kx+1)
                  sum += $signed(l0_ifm[c][r+ky][x+kx]) * $signed(l0_wgt[f][c][ky][kx]);
            l0_relu[f][r][x] = (sum > 0) ? sum : 0;
          end

      for (f=0;f<L0_F_OUT;f=f+1)
        for (pr=0;pr<L0_H_POOL_OUT;pr=pr+1)
          for (pc=0;pc<L0_W_POOL_OUT;pc=pc+1) begin
            max_v = l0_relu[f][2*pr][2*pc];
            for (dy=0;dy<2;dy=dy+1)
              for (dx=0;dx<2;dx=dx+1)
                if (l0_relu[f][2*pr+dy][2*pc+dx] > max_v) max_v = l0_relu[f][2*pr+dy][2*pc+dx];
            l0_pool[f][pr][pc] = sat_to_i8(max_v);
          end

      for (f=0;f<L1_F_OUT;f=f+1)
        for (c=0;c<L1_C_IN;c=c+1)
          for (ky=0;ky<L1_K;ky=ky+1)
            for (kx=0;kx<L1_K;kx=kx+1)
              l1_wgt[f][c][ky][kx] = $signed(((f*9 + c*5 + ky*6 + kx*12 + 2) % 3) - 1);

      for (f=0;f<L1_F_OUT;f=f+1)
        for (r=0;r<L1_H_CONV_OUT;r=r+1)
          for (x=0;x<L1_W_CONV_OUT;x=x+1) begin
            sum=0;
            for (c=0;c<L1_C_IN;c=c+1)
              for (ky=0;ky<L1_K;ky=ky+1)
                for (kx=0;kx<L1_K;kx=kx+1)
                  sum += $signed(l0_pool[c][r+ky][x+kx]) * $signed(l1_wgt[f][c][ky][kx]);
            l1_relu[f][r][x] = (sum > 0) ? sum : 0;
          end

      for (f=0;f<L1_F_OUT;f=f+1)
        for (pr=0;pr<L1_H_POOL_OUT;pr=pr+1)
          for (pc=0;pc<L1_W_POOL_OUT;pc=pc+1) begin
            max_v = l1_relu[f][2*pr][2*pc];
            for (dy=0;dy<2;dy=dy+1)
              for (dx=0;dx<2;dx=dx+1)
                if (l1_relu[f][2*pr+dy][2*pc+dx] > max_v) max_v = l1_relu[f][2*pr+dy][2*pc+dx];
            l1_pool[f][pr][pc] = sat_to_i8(max_v);
          end

      for (f=0;f<L2_F_OUT;f=f+1)
        for (c=0;c<L2_C_IN;c=c+1)
          for (ky=0;ky<L2_K;ky=ky+1)
            for (kx=0;kx<L2_K;kx=kx+1)
              l2_wgt[f][c][ky][kx] = $signed(((f*11 + c*7 + ky*7 + kx*13 + 3) % 3) - 1);

      for (f=0;f<L2_F_OUT;f=f+1)
        for (r=0;r<L2_H_CONV_OUT;r=r+1)
          for (x=0;x<L2_W_CONV_OUT;x=x+1) begin
            sum=0;
            for (c=0;c<L2_C_IN;c=c+1)
              for (ky=0;ky<L2_K;ky=ky+1)
                for (kx=0;kx<L2_K;kx=kx+1)
                  sum += $signed(l1_pool[c][r+ky][x+kx]) * $signed(l2_wgt[f][c][ky][kx]);
            l2_relu[f][r][x] = (sum > 0) ? sum : 0;
          end

      for (f=0;f<L2_F_OUT;f=f+1)
        for (pr=0;pr<L2_H_POOL_OUT;pr=pr+1)
          for (pc=0;pc<L2_W_POOL_OUT;pc=pc+1) begin
            max_v = l2_relu[f][2*pr][2*pc];
            for (dy=0;dy<2;dy=dy+1)
              for (dx=0;dx<2;dx=dx+1)
                if (l2_relu[f][2*pr+dy][2*pc+dx] > max_v) max_v = l2_relu[f][2*pr+dy][2*pc+dx];
            l2_pool[f][pr][pc] = sat_to_i8(max_v);
          end

      for (f=0;f<L3_F_OUT;f=f+1)
        for (c=0;c<L3_C_IN;c=c+1)
          for (ky=0;ky<L3_K;ky=ky+1)
            for (kx=0;kx<L3_K;kx=kx+1)
              l3_wgt[f][c][ky][kx] = $signed(((f*13 + c*9 + ky*8 + kx*14 + 4) % 3) - 1);

      for (f=0;f<L3_F_OUT;f=f+1)
        for (r=0;r<L3_H_CONV_OUT;r=r+1)
          for (x=0;x<L3_W_CONV_OUT;x=x+1) begin
            sum=0;
            for (c=0;c<L3_C_IN;c=c+1)
              for (ky=0;ky<L3_K;ky=ky+1)
                for (kx=0;kx<L3_K;kx=kx+1)
                  sum += $signed(l2_pool[c][r+ky][x+kx]) * $signed(l3_wgt[f][c][ky][kx]);
            l3_relu[f][r][x] = (sum > 0) ? sum : 0;
          end

      for (f=0;f<L3_F_OUT;f=f+1)
        for (pr=0;pr<L3_H_POOL_OUT;pr=pr+1)
          for (pc=0;pc<L3_W_POOL_OUT;pc=pc+1) begin
            max_v = l3_relu[f][2*pr][2*pc];
            for (dy=0;dy<2;dy=dy+1)
              for (dx=0;dx<2;dx=dx+1)
                if (l3_relu[f][2*pr+dy][2*pc+dx] > max_v) max_v = l3_relu[f][2*pr+dy][2*pc+dx];
            l3_pool[f][pr][pc] = sat_to_i8(max_v);
          end

      for (f=0;f<L4_F_OUT;f=f+1)
        for (c=0;c<L4_C_IN;c=c+1)
          for (ky=0;ky<L4_K;ky=ky+1)
            for (kx=0;kx<L4_K;kx=kx+1)
              l4_wgt[f][c][ky][kx] = $signed(((f*15 + c*11 + ky*9 + kx*15 + 5) % 3) - 1);

      for (f=0;f<L4_F_OUT;f=f+1)
        for (r=0;r<L4_H_CONV_OUT;r=r+1)
          for (x=0;x<L4_W_CONV_OUT;x=x+1) begin
            sum=0;
            for (c=0;c<L4_C_IN;c=c+1)
              for (ky=0;ky<L4_K;ky=ky+1)
                for (kx=0;kx<L4_K;kx=kx+1)
                  sum += $signed(l3_pool[c][r+ky][x+kx]) * $signed(l4_wgt[f][c][ky][kx]);
            l4_relu[f][r][x] = (sum > 0) ? sum : 0;
          end

      for (f=0;f<L4_F_OUT;f=f+1)
        for (pr=0;pr<L4_H_POOL_OUT;pr=pr+1)
          for (pc=0;pc<L4_W_POOL_OUT;pc=pc+1) begin
            max_v = l4_relu[f][2*pr][2*pc];
            for (dy=0;dy<2;dy=dy+1)
              for (dx=0;dx<2;dx=dx+1)
                if (l4_relu[f][2*pr+dy][2*pc+dx] > max_v) max_v = l4_relu[f][2*pr+dy][2*pc+dx];
            l4_pool[f][pr][pc] = sat_to_i8(max_v);
          end
    end
  endtask

  task automatic clear_ddr;
    integer i;
    begin
      for (i=0;i<MEM_DEPTH;i=i+1) ddr_mem[i]='0;
    end
  endtask

  task automatic load_l0_ifm_to_ddr;
    integer c,r,colg,lane,abs_col,addr;
    begin
      for (c=0;c<L0_C_IN;c=c+1)
        for (r=0;r<L0_H_IN;r=r+1)
          for (colg=0;colg<L0_IFM_WORDS_PER_ROW;colg=colg+1) begin
            addr = `DDR_IFM_BASE + ((c*L0_H_IN + r)*L0_IFM_WORDS_PER_ROW) + colg;
            ddr_mem[addr]='0;
            for (lane=0;lane<PV_MAX;lane=lane+1) begin
              abs_col = colg*PV_M1 + lane;
              if ((lane<PV_M1) && (abs_col<L0_W_IN))
                ddr_mem[addr][lane*DATA_W +: DATA_W] = l0_ifm[c][r][abs_col];
            end
          end
    end
  endtask

  task automatic write_wgt_lane_to_ddr(input integer base, input integer phys_word, input integer lane_abs, input logic signed [DATA_W-1:0] value);
    integer ddr_subword, ddr_lane, addr;
    begin
      ddr_subword = lane_abs / PV_MAX;
      ddr_lane    = lane_abs % PV_MAX;
      addr        = base + phys_word*WGT_SUBWORDS + ddr_subword;
      ddr_mem[addr][ddr_lane*DATA_W +: DATA_W] = value;
    end
  endtask


  task automatic load_l0_weights_to_ddr;
    integer p, logical_idx, phys_word, subword_idx, base_lane, fg,c,ky,kx,pf,f;
    begin
      for (p=0;p<L0_M1_WGT_DDR_WORDS;p=p+1) ddr_mem[L0_WGT_DDR_BASE+p]='0;
      logical_idx=0;
      for (fg=0;fg<L0_NUM_FGROUP;fg=fg+1)
        for (c=0;c<L0_C_IN;c=c+1)
          for (ky=0;ky<L0_K;ky=ky+1)
            for (kx=0;kx<L0_K;kx=kx+1) begin
              phys_word=logical_idx/PV_M1; subword_idx=logical_idx%PV_M1; base_lane=subword_idx*PF_M1;
              for (pf=0;pf<PF_M1;pf=pf+1) begin
                f=fg*PF_M1+pf;
                write_wgt_lane_to_ddr(L0_WGT_DDR_BASE, phys_word, base_lane+pf, (f<L0_F_OUT)?l0_wgt[f][c][ky][kx]:'0);
              end
              logical_idx++;
            end
    end
  endtask


  task automatic load_l1_weights_to_ddr;
    integer p, logical_idx, phys_word, subword_idx, base_lane, fg,c,ky,kx,pf,f;
    begin
      for (p=0;p<L1_M1_WGT_DDR_WORDS;p=p+1) ddr_mem[L1_WGT_DDR_BASE+p]='0;
      logical_idx=0;
      for (fg=0;fg<L1_NUM_FGROUP;fg=fg+1)
        for (c=0;c<L1_C_IN;c=c+1)
          for (ky=0;ky<L1_K;ky=ky+1)
            for (kx=0;kx<L1_K;kx=kx+1) begin
              phys_word=logical_idx/PV_M1; subword_idx=logical_idx%PV_M1; base_lane=subword_idx*PF_M1;
              for (pf=0;pf<PF_M1;pf=pf+1) begin
                f=fg*PF_M1+pf;
                write_wgt_lane_to_ddr(L1_WGT_DDR_BASE, phys_word, base_lane+pf, (f<L1_F_OUT)?l1_wgt[f][c][ky][kx]:'0);
              end
              logical_idx++;
            end
    end
  endtask


  task automatic load_l2_weights_to_ddr;
    integer p, logical_idx, phys_word, subword_idx, base_lane, fg,c,ky,kx,pf,f;
    begin
      for (p=0;p<L2_M1_WGT_DDR_WORDS;p=p+1) ddr_mem[L2_WGT_DDR_BASE+p]='0;
      logical_idx=0;
      for (fg=0;fg<L2_NUM_FGROUP;fg=fg+1)
        for (c=0;c<L2_C_IN;c=c+1)
          for (ky=0;ky<L2_K;ky=ky+1)
            for (kx=0;kx<L2_K;kx=kx+1) begin
              phys_word=logical_idx/PV_M1; subword_idx=logical_idx%PV_M1; base_lane=subword_idx*PF_M1;
              for (pf=0;pf<PF_M1;pf=pf+1) begin
                f=fg*PF_M1+pf;
                write_wgt_lane_to_ddr(L2_WGT_DDR_BASE, phys_word, base_lane+pf, (f<L2_F_OUT)?l2_wgt[f][c][ky][kx]:'0);
              end
              logical_idx++;
            end
    end
  endtask


  task automatic load_l3_weights_to_ddr;
    integer p, logical_idx, phys_word, subword_idx, base_lane, fg,c,ky,kx,pf,f;
    begin
      for (p=0;p<L3_M1_WGT_DDR_WORDS;p=p+1) ddr_mem[L3_WGT_DDR_BASE+p]='0;
      logical_idx=0;
      for (fg=0;fg<L3_NUM_FGROUP;fg=fg+1)
        for (c=0;c<L3_C_IN;c=c+1)
          for (ky=0;ky<L3_K;ky=ky+1)
            for (kx=0;kx<L3_K;kx=kx+1) begin
              phys_word=logical_idx/PV_M1; subword_idx=logical_idx%PV_M1; base_lane=subword_idx*PF_M1;
              for (pf=0;pf<PF_M1;pf=pf+1) begin
                f=fg*PF_M1+pf;
                write_wgt_lane_to_ddr(L3_WGT_DDR_BASE, phys_word, base_lane+pf, (f<L3_F_OUT)?l3_wgt[f][c][ky][kx]:'0);
              end
              logical_idx++;
            end
    end
  endtask


  task automatic load_l4_weights_to_ddr;
    integer p, logical_idx, phys_word, subword_idx, base_lane, fg,c,ky,kx,pf,f;
    begin
      for (p=0;p<L4_M1_WGT_DDR_WORDS;p=p+1) ddr_mem[L4_WGT_DDR_BASE+p]='0;
      logical_idx=0;
      for (fg=0;fg<L4_NUM_FGROUP;fg=fg+1)
        for (c=0;c<L4_C_IN;c=c+1)
          for (ky=0;ky<L4_K;ky=ky+1)
            for (kx=0;kx<L4_K;kx=kx+1) begin
              phys_word=logical_idx/PV_M1; subword_idx=logical_idx%PV_M1; base_lane=subword_idx*PF_M1;
              for (pf=0;pf<PF_M1;pf=pf+1) begin
                f=fg*PF_M1+pf;
                write_wgt_lane_to_ddr(L4_WGT_DDR_BASE, phys_word, base_lane+pf, (f<L4_F_OUT)?l4_wgt[f][c][ky][kx]:'0);
              end
              logical_idx++;
            end
    end
  endtask

  task automatic init_mem;
    begin
      clear_ddr();
      build_golden();
      load_l0_ifm_to_ddr();
      load_l0_weights_to_ddr();
      load_l1_weights_to_ddr();
      load_l2_weights_to_ddr();
      load_l3_weights_to_ddr();
      load_l4_weights_to_ddr();
      $display("TB_INFO: 5-layer Mode1 partial/non-divisible stress tiled test");
      $display("TB_INFO: L0 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> pool %0dx%0dx%0d", L0_H_IN,L0_W_IN,L0_C_IN,L0_H_CONV_OUT,L0_W_CONV_OUT,L0_F_OUT,L0_K,L0_H_POOL_OUT,L0_W_POOL_OUT,L0_F_OUT);
      $display("TB_INFO: L1 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> pool %0dx%0dx%0d", L1_H_IN,L1_W_IN,L1_C_IN,L1_H_CONV_OUT,L1_W_CONV_OUT,L1_F_OUT,L1_K,L1_H_POOL_OUT,L1_W_POOL_OUT,L1_F_OUT);
      $display("TB_INFO: L2 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> pool %0dx%0dx%0d", L2_H_IN,L2_W_IN,L2_C_IN,L2_H_CONV_OUT,L2_W_CONV_OUT,L2_F_OUT,L2_K,L2_H_POOL_OUT,L2_W_POOL_OUT,L2_F_OUT);
      $display("TB_INFO: L3 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> pool %0dx%0dx%0d", L3_H_IN,L3_W_IN,L3_C_IN,L3_H_CONV_OUT,L3_W_CONV_OUT,L3_F_OUT,L3_K,L3_H_POOL_OUT,L3_W_POOL_OUT,L3_F_OUT);
      $display("TB_INFO: L4 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> pool %0dx%0dx%0d", L4_H_IN,L4_W_IN,L4_C_IN,L4_H_CONV_OUT,L4_W_CONV_OUT,L4_F_OUT,L4_K,L4_H_POOL_OUT,L4_W_POOL_OUT,L4_F_OUT);
      $display("TB_INFO: stress params: H/W/C/F include non-divisible partial groups; HT=%0d Pv=%0d Pf=%0d C_MAX=%0d F_MAX=%0d", HT, PV_M1, PF_M1, C_MAX, F_MAX);
      $display("TB_INFO: Pv=%0d Pf=%0d PTOTAL=%0d", PV_M1, PF_M1, PTOTAL);
      $display("TB_INFO: L0 weight bundles=%0d phys=%0d DDR words=%0d base=0x%05h", L0_M1_LOGICAL_BUNDLES, L0_M1_PHYS_WORDS, L0_M1_WGT_DDR_WORDS, L0_WGT_DDR_BASE);
      $display("TB_INFO: L1 weight bundles=%0d phys=%0d DDR words=%0d base=0x%05h", L1_M1_LOGICAL_BUNDLES, L1_M1_PHYS_WORDS, L1_M1_WGT_DDR_WORDS, L1_WGT_DDR_BASE);
      $display("TB_INFO: L2 weight bundles=%0d phys=%0d DDR words=%0d base=0x%05h", L2_M1_LOGICAL_BUNDLES, L2_M1_PHYS_WORDS, L2_M1_WGT_DDR_WORDS, L2_WGT_DDR_BASE);
      $display("TB_INFO: L3 weight bundles=%0d phys=%0d DDR words=%0d base=0x%05h", L3_M1_LOGICAL_BUNDLES, L3_M1_PHYS_WORDS, L3_M1_WGT_DDR_WORDS, L3_WGT_DDR_BASE);
      $display("TB_INFO: L4 weight bundles=%0d phys=%0d DDR words=%0d base=0x%05h", L4_M1_LOGICAL_BUNDLES, L4_M1_PHYS_WORDS, L4_M1_WGT_DDR_WORDS, L4_WGT_DDR_BASE);
      $display("TB_INFO: expected final OFM DDR words=%0d", EXP_OFM_WORDS);
      $display("TB_INFO: expected DDR->IFM writes=%0d after-start=%0d", L0_EXPECT_IFM_DDR_WRITES, L0_EXPECT_IFM_DDR_WRITES_AFTER_START);
      $display("TB_INFO: expected OFM->IFM stream commands >= %0d", EXP_OFM2IFM_STREAMS);
    end
  endtask

  task automatic write_cfg(input logic [$clog2(CFG_DEPTH)-1:0] addr, input layer_desc_t cfg);
    begin
      @(posedge clk); cfg_wr_en<=1'b1; cfg_wr_addr<=addr; cfg_wr_data<=cfg;
      @(posedge clk); cfg_wr_en<=1'b0; cfg_wr_addr<='0; cfg_wr_data<='0;
    end
  endtask

  task automatic program_layers;
    layer_desc_t cfg0,cfg1,cfg2,cfg3,cfg4;
    begin

      cfg0='0;
      cfg0.layer_id=0; cfg0.mode=MODE1; cfg0.h_in=L0_H_IN; cfg0.w_in=L0_W_IN; cfg0.c_in=L0_C_IN; cfg0.f_out=L0_F_OUT; cfg0.k=L0_K;
      cfg0.h_out=L0_H_CONV_OUT; cfg0.w_out=L0_W_CONV_OUT;
      cfg0.pv_m1=PV_M1; cfg0.pf_m1=PF_M1; cfg0.pc_m2=PC_MODE2; cfg0.pf_m2=PF_MODE2;
      cfg0.conv_stride=1; cfg0.relu_en=1'b1; cfg0.pool_en=1'b1; cfg0.pool_k=2; cfg0.pool_stride=2;
      cfg0.ifm_ddr_base=`DDR_IFM_BASE; cfg0.wgt_ddr_base=L0_WGT_DDR_BASE; cfg0.ofm_ddr_base=`DDR_OFM_BASE;
      cfg0.first_layer=1'b1; cfg0.last_layer=1'b0;

      cfg1='0;
      cfg1.layer_id=1; cfg1.mode=MODE1; cfg1.h_in=L1_H_IN; cfg1.w_in=L1_W_IN; cfg1.c_in=L1_C_IN; cfg1.f_out=L1_F_OUT; cfg1.k=L1_K;
      cfg1.h_out=L1_H_CONV_OUT; cfg1.w_out=L1_W_CONV_OUT;
      cfg1.pv_m1=PV_M1; cfg1.pf_m1=PF_M1; cfg1.pc_m2=PC_MODE2; cfg1.pf_m2=PF_MODE2;
      cfg1.conv_stride=1; cfg1.relu_en=1'b1; cfg1.pool_en=1'b1; cfg1.pool_k=2; cfg1.pool_stride=2;
      cfg1.ifm_ddr_base=`DDR_IFM_BASE; cfg1.wgt_ddr_base=L1_WGT_DDR_BASE; cfg1.ofm_ddr_base=`DDR_OFM_BASE;
      cfg1.first_layer=1'b0; cfg1.last_layer=1'b0;

      cfg2='0;
      cfg2.layer_id=2; cfg2.mode=MODE1; cfg2.h_in=L2_H_IN; cfg2.w_in=L2_W_IN; cfg2.c_in=L2_C_IN; cfg2.f_out=L2_F_OUT; cfg2.k=L2_K;
      cfg2.h_out=L2_H_CONV_OUT; cfg2.w_out=L2_W_CONV_OUT;
      cfg2.pv_m1=PV_M1; cfg2.pf_m1=PF_M1; cfg2.pc_m2=PC_MODE2; cfg2.pf_m2=PF_MODE2;
      cfg2.conv_stride=1; cfg2.relu_en=1'b1; cfg2.pool_en=1'b1; cfg2.pool_k=2; cfg2.pool_stride=2;
      cfg2.ifm_ddr_base=`DDR_IFM_BASE; cfg2.wgt_ddr_base=L2_WGT_DDR_BASE; cfg2.ofm_ddr_base=`DDR_OFM_BASE;
      cfg2.first_layer=1'b0; cfg2.last_layer=1'b0;

      cfg3='0;
      cfg3.layer_id=3; cfg3.mode=MODE1; cfg3.h_in=L3_H_IN; cfg3.w_in=L3_W_IN; cfg3.c_in=L3_C_IN; cfg3.f_out=L3_F_OUT; cfg3.k=L3_K;
      cfg3.h_out=L3_H_CONV_OUT; cfg3.w_out=L3_W_CONV_OUT;
      cfg3.pv_m1=PV_M1; cfg3.pf_m1=PF_M1; cfg3.pc_m2=PC_MODE2; cfg3.pf_m2=PF_MODE2;
      cfg3.conv_stride=1; cfg3.relu_en=1'b1; cfg3.pool_en=1'b1; cfg3.pool_k=2; cfg3.pool_stride=2;
      cfg3.ifm_ddr_base=`DDR_IFM_BASE; cfg3.wgt_ddr_base=L3_WGT_DDR_BASE; cfg3.ofm_ddr_base=`DDR_OFM_BASE;
      cfg3.first_layer=1'b0; cfg3.last_layer=1'b0;

      cfg4='0;
      cfg4.layer_id=4; cfg4.mode=MODE1; cfg4.h_in=L4_H_IN; cfg4.w_in=L4_W_IN; cfg4.c_in=L4_C_IN; cfg4.f_out=L4_F_OUT; cfg4.k=L4_K;
      cfg4.h_out=L4_H_CONV_OUT; cfg4.w_out=L4_W_CONV_OUT;
      cfg4.pv_m1=PV_M1; cfg4.pf_m1=PF_M1; cfg4.pc_m2=PC_MODE2; cfg4.pf_m2=PF_MODE2;
      cfg4.conv_stride=1; cfg4.relu_en=1'b1; cfg4.pool_en=1'b1; cfg4.pool_k=2; cfg4.pool_stride=2;
      cfg4.ifm_ddr_base=`DDR_IFM_BASE; cfg4.wgt_ddr_base=L4_WGT_DDR_BASE; cfg4.ofm_ddr_base=`DDR_OFM_BASE;
      cfg4.first_layer=1'b0; cfg4.last_layer=1'b1;

      write_cfg(0,cfg0);
      write_cfg(1,cfg1);
      write_cfg(2,cfg2);
      write_cfg(3,cfg3);
      write_cfg(4,cfg4);
    end
  endtask


  task automatic pulse_start;
    begin
      @(posedge clk); start<=1'b1;
      @(posedge clk); start<=1'b0;
    end
  endtask

  task automatic dump_ofm_region;
    integer j;
    begin
      $display("--- OFM DDR region dump, first %0d expected words ---", EXP_OFM_WORDS);
      for (j=0;j<EXP_OFM_WORDS;j=j+1)
        $display("OFM[%0d] @0x%05h = 0x%08h", j, (`DDR_OFM_BASE+j), ddr_mem[`DDR_OFM_BASE+j]);
    end
  endtask

  task automatic check_final_ofm;
    integer lin,ch,rem,row,grp,col,fail_count,ofm_mismatch_count;
    logic signed [DATA_W-1:0] got,exp;
    begin
      fail_count=0;
      ofm_mismatch_count=0;
      if (ddr_ofm_write_count != EXP_OFM_WORDS) begin
        $display("TB_FAIL: OFM DDR write count mismatch. got=%0d expected=%0d", ddr_ofm_write_count, EXP_OFM_WORDS);
        fail_count++;
      end
      if (L0_H_IN > HT && ifm_cmd_start_count < 2) begin
        $display("TB_FAIL: initial DDR->IFM tiling was not exercised. ifm_cmd_start_count=%0d expected>=2", ifm_cmd_start_count);
        fail_count++;
      end
      if (ifm_dma_wr_count != L0_EXPECT_IFM_DDR_WRITES) begin
        $display("TB_FAIL: DDR->IFM write count mismatch. got=%0d expected=%0d", ifm_dma_wr_count, L0_EXPECT_IFM_DDR_WRITES);
        fail_count++;
      end
      if (ifm_dma_wr_after_m1_start_count < L0_EXPECT_IFM_DDR_WRITES_AFTER_START) begin
        $display("TB_FAIL: DDR->IFM runtime refill may be incomplete. after_start=%0d expected>=%0d", ifm_dma_wr_after_m1_start_count, L0_EXPECT_IFM_DDR_WRITES_AFTER_START);
        fail_count++;
      end
      if (ofm_ifm_stream_start_count < EXP_OFM2IFM_STREAMS) begin
        $display("TB_FAIL: OFM->IFM tiled refill may not be fully exercised. stream_start_count=%0d expected>=%0d", ofm_ifm_stream_start_count, EXP_OFM2IFM_STREAMS);
        fail_count++;
      end
      if (ofm_ifm_stream_done_count < EXP_OFM2IFM_STREAMS) begin
        $display("TB_FAIL: OFM->IFM stream done count too small. stream_done_count=%0d expected>=%0d", ofm_ifm_stream_done_count, EXP_OFM2IFM_STREAMS);
        fail_count++;
      end
      for (lin=0;lin<EXP_OFM_WORDS;lin=lin+1) begin
        ch=lin/(L4_H_POOL_OUT*FINAL_GROUPS);
        rem=lin%(L4_H_POOL_OUT*FINAL_GROUPS);
        row=rem/FINAL_GROUPS;
        grp=rem%FINAL_GROUPS;
        col=grp*FINAL_STORE_PACK;
        got=ddr_mem[`DDR_OFM_BASE+lin][0 +: DATA_W];
        exp=l4_pool[ch][row][col];
        if (got !== exp) begin
          $display("TB_FAIL: final OFM mismatch lin=%0d ch=%0d row=%0d col=%0d got=%0d/0x%02h expected=%0d/0x%02h word=0x%08h",
                   lin,ch,row,col,got,got,exp,exp,ddr_mem[`DDR_OFM_BASE+lin]);
          ofm_mismatch_count++;
        end
      end
      if ((fail_count==0) && (ofm_mismatch_count==0)) begin
        $display("TB_PASS: exact final OFM matches golden for 5-layer partial/non-divisible stress tiled mode1 test");
      end else begin
        if (ofm_mismatch_count != 0)
          $display("TB_FAIL: total final OFM mismatches = %0d", ofm_mismatch_count);
        if (fail_count != 0)
          $display("TB_FAIL: total non-value checker failures = %0d", fail_count);
        dump_ofm_region();
        $finish;
      end
    end
  endtask

  initial begin
    rst_n=1'b0; start=1'b0; abort=1'b0;
    cfg_wr_en=1'b0; cfg_wr_addr='0; cfg_wr_data='0; cfg_num_layers=5;
    m1_free_col_blk_g='0; m1_free_ch_blk_g='0;
    m1_sm_refill_req_ready=1'b1; m2_sm_refill_req_ready=1'b1;

    init_mem();

    repeat (5) @(posedge clk);
    rst_n=1'b1;

    program_layers();
    repeat (5) @(posedge clk);
    pulse_start();

    wait (done || error || (cycle_count > MAX_CYCLES));
    repeat (5) @(posedge clk);

    if (error) begin
      $display("TB_FAIL: DUT asserted error. dbg_error_vec=0x%0h layer=%0d mode=%0d", dbg_error_vec, dbg_layer_idx, dbg_mode);
      dump_ofm_region();
      $finish;
    end
    if (cycle_count > MAX_CYCLES) begin
      $display("TB_FAIL: timeout after %0d cycles. busy=%0b done=%0b error=%0b layer=%0d mode=%0d err=0x%0h",
               cycle_count,busy,done,error,dbg_layer_idx,dbg_mode,dbg_error_vec);
      dump_ofm_region();
      $finish;
    end

    $display("TB_INFO: done observed after %0d cycles", cycle_count);
    $display("TB_INFO: OFM DDR writes counted = %0d", ddr_ofm_write_count);
    $display("TB_INFO: IFM command starts counted = %0d", ifm_cmd_start_count);
    $display("TB_INFO: IFM DMA writes counted = %0d, after m1_start = %0d", ifm_dma_wr_count, ifm_dma_wr_after_m1_start_count);
    $display("TB_INFO: IFM m1_free_valid pulses counted = %0d", ifm_free_count);
    $display("TB_INFO: old M1 same-mode refill-manager requests accepted = %0d", old_m1_sm_refill_req_count);
    $display("TB_INFO: OFM->IFM stream starts counted = %0d, stream done counted = %0d", ofm_ifm_stream_start_count, ofm_ifm_stream_done_count);
    check_final_ofm();
    $finish;
  end

endmodule
