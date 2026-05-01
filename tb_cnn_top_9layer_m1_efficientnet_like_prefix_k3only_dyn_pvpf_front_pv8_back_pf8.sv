`timescale 1ns/1ps
`include "cnn_ddr_defs.svh"

module tb_cnn_top_9layer_m1_efficientnet_like_prefix_k3only_dyn_pvpf_front_pv8_back_pf8;
  import cnn_layer_desc_pkg::*;

  localparam int DATA_W=8, PSUM_W=32;
  localparam int PTOTAL=32, PV_MAX=8, PF_MAX=8;
  localparam int PC_MODE2=4, PF_MODE2=2;
  localparam int C_MAX=16, F_MAX=16, W_MAX=96, H_MAX=96, HT=4, K_MAX=3;
  localparam int WGT_DEPTH=4096, OFM_BANK_DEPTH=H_MAX*W_MAX;
  localparam int OFM_LINEAR_DEPTH=C_MAX*OFM_BANK_DEPTH, CFG_DEPTH=16;
  localparam int DDR_ADDR_W=`CNN_DDR_ADDR_W, DDR_WORD_W=PV_MAX*DATA_W;
  localparam int MEM_DEPTH=(`DDR_RSVD_BASE + `DDR_RSVD_SIZE);
  localparam int CLK_PERIOD_NS=10, MAX_CYCLES=30000000;

  // EfficientNet-like mode-1 prefix benchmark inspired by DCP-CNN Table VI (EfficientNet transition L=10).
  // It is not exact EfficientNet-B0: depthwise/SE/skip ops are omitted because current RTL supports standard conv+ReLU+optional 2x2 pool.
  // The goal is to stress the first 9 mode-1 layers: RGB input, all standard conv kernels use K=3, many OFM->IFM handoffs, partial C/F groups, and H/W tiling.
  // L0: 96x96x3 -> conv 94x94x8 K=3 -> pool 47x47x8
  // L1: 47x47x8 -> conv 45x45x8 K=3 -> no pool
  // L2: 45x45x8 -> conv 43x43x10 K=3 -> pool 21x21x10
  // L3: 21x21x10 -> conv 19x19x12 K=3 -> no pool
  // L4: 19x19x12 -> conv 17x17x13 K=3 -> no pool
  // L5: 17x17x13 -> conv 15x15x15 K=3 -> pool 7x7x15
  // L6: 7x7x15 -> conv 5x5x16 K=3 -> no pool
  // L7: 5x5x16 -> conv 3x3x16 K=3 -> no pool
  // L8: 3x3x16 -> conv 1x1x11 K=3 -> no pool 1x1x11

  localparam int L0_H_IN=96, L0_W_IN=96, L0_C_IN=3, L0_F_OUT=8, L0_K=3, L0_POOL_EN=1;
  localparam int L0_H_CONV_OUT=L0_H_IN-L0_K+1, L0_W_CONV_OUT=L0_W_IN-L0_K+1;
  localparam int L0_H_OUT=(L0_POOL_EN ? (L0_H_CONV_OUT/2) : L0_H_CONV_OUT);
  localparam int L0_W_OUT=(L0_POOL_EN ? (L0_W_CONV_OUT/2) : L0_W_CONV_OUT);

  localparam int L1_H_IN=47, L1_W_IN=47, L1_C_IN=8, L1_F_OUT=8, L1_K=3, L1_POOL_EN=0;
  localparam int L1_H_CONV_OUT=L1_H_IN-L1_K+1, L1_W_CONV_OUT=L1_W_IN-L1_K+1;
  localparam int L1_H_OUT=(L1_POOL_EN ? (L1_H_CONV_OUT/2) : L1_H_CONV_OUT);
  localparam int L1_W_OUT=(L1_POOL_EN ? (L1_W_CONV_OUT/2) : L1_W_CONV_OUT);

  localparam int L2_H_IN=45, L2_W_IN=45, L2_C_IN=8, L2_F_OUT=10, L2_K=3, L2_POOL_EN=1;
  localparam int L2_H_CONV_OUT=L2_H_IN-L2_K+1, L2_W_CONV_OUT=L2_W_IN-L2_K+1;
  localparam int L2_H_OUT=(L2_POOL_EN ? (L2_H_CONV_OUT/2) : L2_H_CONV_OUT);
  localparam int L2_W_OUT=(L2_POOL_EN ? (L2_W_CONV_OUT/2) : L2_W_CONV_OUT);

  localparam int L3_H_IN=21, L3_W_IN=21, L3_C_IN=10, L3_F_OUT=12, L3_K=3, L3_POOL_EN=0;
  localparam int L3_H_CONV_OUT=L3_H_IN-L3_K+1, L3_W_CONV_OUT=L3_W_IN-L3_K+1;
  localparam int L3_H_OUT=(L3_POOL_EN ? (L3_H_CONV_OUT/2) : L3_H_CONV_OUT);
  localparam int L3_W_OUT=(L3_POOL_EN ? (L3_W_CONV_OUT/2) : L3_W_CONV_OUT);

  localparam int L4_H_IN=19, L4_W_IN=19, L4_C_IN=12, L4_F_OUT=13, L4_K=3, L4_POOL_EN=0;
  localparam int L4_H_CONV_OUT=L4_H_IN-L4_K+1, L4_W_CONV_OUT=L4_W_IN-L4_K+1;
  localparam int L4_H_OUT=(L4_POOL_EN ? (L4_H_CONV_OUT/2) : L4_H_CONV_OUT);
  localparam int L4_W_OUT=(L4_POOL_EN ? (L4_W_CONV_OUT/2) : L4_W_CONV_OUT);

  localparam int L5_H_IN=17, L5_W_IN=17, L5_C_IN=13, L5_F_OUT=15, L5_K=3, L5_POOL_EN=1;
  localparam int L5_H_CONV_OUT=L5_H_IN-L5_K+1, L5_W_CONV_OUT=L5_W_IN-L5_K+1;
  localparam int L5_H_OUT=(L5_POOL_EN ? (L5_H_CONV_OUT/2) : L5_H_CONV_OUT);
  localparam int L5_W_OUT=(L5_POOL_EN ? (L5_W_CONV_OUT/2) : L5_W_CONV_OUT);

  localparam int L6_H_IN=7, L6_W_IN=7, L6_C_IN=15, L6_F_OUT=16, L6_K=3, L6_POOL_EN=0;
  localparam int L6_H_CONV_OUT=L6_H_IN-L6_K+1, L6_W_CONV_OUT=L6_W_IN-L6_K+1;
  localparam int L6_H_OUT=(L6_POOL_EN ? (L6_H_CONV_OUT/2) : L6_H_CONV_OUT);
  localparam int L6_W_OUT=(L6_POOL_EN ? (L6_W_CONV_OUT/2) : L6_W_CONV_OUT);

  localparam int L7_H_IN=5, L7_W_IN=5, L7_C_IN=16, L7_F_OUT=16, L7_K=3, L7_POOL_EN=0;
  localparam int L7_H_CONV_OUT=L7_H_IN-L7_K+1, L7_W_CONV_OUT=L7_W_IN-L7_K+1;
  localparam int L7_H_OUT=(L7_POOL_EN ? (L7_H_CONV_OUT/2) : L7_H_CONV_OUT);
  localparam int L7_W_OUT=(L7_POOL_EN ? (L7_W_CONV_OUT/2) : L7_W_CONV_OUT);

  localparam int L8_H_IN=3, L8_W_IN=3, L8_C_IN=16, L8_F_OUT=11, L8_K=3, L8_POOL_EN=0;
  localparam int L8_H_CONV_OUT=L8_H_IN-L8_K+1, L8_W_CONV_OUT=L8_W_IN-L8_K+1;
  localparam int L8_H_OUT=(L8_POOL_EN ? (L8_H_CONV_OUT/2) : L8_H_CONV_OUT);
  localparam int L8_W_OUT=(L8_POOL_EN ? (L8_W_CONV_OUT/2) : L8_W_CONV_OUT);

  // Runtime mode-1 parallelism changes by network depth while keeping Pv*Pf = PTOTAL = 32.
  // Early layers use higher pixel parallelism; later layers use higher filter parallelism,
  // matching the expected trend that spatial size shrinks while filter/channel pressure grows.
  localparam int L0_PV=8, L0_PF=4;
  localparam int L1_PV=8, L1_PF=4;
  localparam int L2_PV=8, L2_PF=4;
  localparam int L3_PV=8, L3_PF=4;
  localparam int L4_PV=4, L4_PF=8;
  localparam int L5_PV=4, L5_PF=8;
  localparam int L6_PV=4, L6_PF=8;
  localparam int L7_PV=4, L7_PF=8;
  localparam int L8_PV=4, L8_PF=8;

  localparam int L0_IFM_WORDS_PER_ROW=(L0_W_IN+L0_PV-1)/L0_PV;
  localparam int L0_EXPECT_IFM_DDR_WRITES=L0_C_IN*L0_H_IN*L0_IFM_WORDS_PER_ROW;
  localparam int L0_EXPECT_IFM_DDR_WRITES_AFTER_START=(L0_H_IN-HT)*L0_C_IN*L0_IFM_WORDS_PER_ROW;

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

  localparam int FINAL_STORE_PACK=1;
  localparam int FINAL_GROUPS=(L8_W_OUT+FINAL_STORE_PACK-1)/FINAL_STORE_PACK;
  localparam int EXP_OFM_WORDS=L8_F_OUT*L8_H_OUT*FINAL_GROUPS;

  localparam int L1_OFM2IFM_COL_BLKS=(L1_W_IN+L1_PV-1)/L1_PV;
  localparam int L1_OFM2IFM_CH_BLKS=(L1_C_IN+L1_PF-1)/L1_PF;
  localparam int L2_OFM2IFM_COL_BLKS=(L2_W_IN+L2_PV-1)/L2_PV;
  localparam int L2_OFM2IFM_CH_BLKS=(L2_C_IN+L2_PF-1)/L2_PF;
  localparam int L3_OFM2IFM_COL_BLKS=(L3_W_IN+L3_PV-1)/L3_PV;
  localparam int L3_OFM2IFM_CH_BLKS=(L3_C_IN+L3_PF-1)/L3_PF;
  localparam int L4_OFM2IFM_COL_BLKS=(L4_W_IN+L4_PV-1)/L4_PV;
  localparam int L4_OFM2IFM_CH_BLKS=(L4_C_IN+L4_PF-1)/L4_PF;
  localparam int L5_OFM2IFM_COL_BLKS=(L5_W_IN+L5_PV-1)/L5_PV;
  localparam int L5_OFM2IFM_CH_BLKS=(L5_C_IN+L5_PF-1)/L5_PF;
  localparam int L6_OFM2IFM_COL_BLKS=(L6_W_IN+L6_PV-1)/L6_PV;
  localparam int L6_OFM2IFM_CH_BLKS=(L6_C_IN+L6_PF-1)/L6_PF;
  localparam int L7_OFM2IFM_COL_BLKS=(L7_W_IN+L7_PV-1)/L7_PV;
  localparam int L7_OFM2IFM_CH_BLKS=(L7_C_IN+L7_PF-1)/L7_PF;
  localparam int L8_OFM2IFM_COL_BLKS=(L8_W_IN+L8_PV-1)/L8_PV;
  localparam int L8_OFM2IFM_CH_BLKS=(L8_C_IN+L8_PF-1)/L8_PF;
  localparam int EXP_OFM2IFM_STREAMS=(L1_H_IN*L1_OFM2IFM_COL_BLKS*L1_OFM2IFM_CH_BLKS) +
                                     (L2_H_IN*L2_OFM2IFM_COL_BLKS*L2_OFM2IFM_CH_BLKS) +
                                     (L3_H_IN*L3_OFM2IFM_COL_BLKS*L3_OFM2IFM_CH_BLKS) +
                                     (L4_H_IN*L4_OFM2IFM_COL_BLKS*L4_OFM2IFM_CH_BLKS) +
                                     (L5_H_IN*L5_OFM2IFM_COL_BLKS*L5_OFM2IFM_CH_BLKS) +
                                     (L6_H_IN*L6_OFM2IFM_COL_BLKS*L6_OFM2IFM_CH_BLKS) +
                                     (L7_H_IN*L7_OFM2IFM_COL_BLKS*L7_OFM2IFM_CH_BLKS) +
                                     (L8_H_IN*L8_OFM2IFM_COL_BLKS*L8_OFM2IFM_CH_BLKS);


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
  logic signed [DATA_W-1:0] l0_out [0:L0_F_OUT-1][0:L0_H_OUT-1][0:L0_W_OUT-1];

  logic signed [DATA_W-1:0] l1_wgt [0:L1_F_OUT-1][0:L1_C_IN-1][0:L1_K-1][0:L1_K-1];
  integer signed l1_relu [0:L1_F_OUT-1][0:L1_H_CONV_OUT-1][0:L1_W_CONV_OUT-1];
  logic signed [DATA_W-1:0] l1_out [0:L1_F_OUT-1][0:L1_H_OUT-1][0:L1_W_OUT-1];

  logic signed [DATA_W-1:0] l2_wgt [0:L2_F_OUT-1][0:L2_C_IN-1][0:L2_K-1][0:L2_K-1];
  integer signed l2_relu [0:L2_F_OUT-1][0:L2_H_CONV_OUT-1][0:L2_W_CONV_OUT-1];
  logic signed [DATA_W-1:0] l2_out [0:L2_F_OUT-1][0:L2_H_OUT-1][0:L2_W_OUT-1];

  logic signed [DATA_W-1:0] l3_wgt [0:L3_F_OUT-1][0:L3_C_IN-1][0:L3_K-1][0:L3_K-1];
  integer signed l3_relu [0:L3_F_OUT-1][0:L3_H_CONV_OUT-1][0:L3_W_CONV_OUT-1];
  logic signed [DATA_W-1:0] l3_out [0:L3_F_OUT-1][0:L3_H_OUT-1][0:L3_W_OUT-1];

  logic signed [DATA_W-1:0] l4_wgt [0:L4_F_OUT-1][0:L4_C_IN-1][0:L4_K-1][0:L4_K-1];
  integer signed l4_relu [0:L4_F_OUT-1][0:L4_H_CONV_OUT-1][0:L4_W_CONV_OUT-1];
  logic signed [DATA_W-1:0] l4_out [0:L4_F_OUT-1][0:L4_H_OUT-1][0:L4_W_OUT-1];

  logic signed [DATA_W-1:0] l5_wgt [0:L5_F_OUT-1][0:L5_C_IN-1][0:L5_K-1][0:L5_K-1];
  integer signed l5_relu [0:L5_F_OUT-1][0:L5_H_CONV_OUT-1][0:L5_W_CONV_OUT-1];
  logic signed [DATA_W-1:0] l5_out [0:L5_F_OUT-1][0:L5_H_OUT-1][0:L5_W_OUT-1];

  logic signed [DATA_W-1:0] l6_wgt [0:L6_F_OUT-1][0:L6_C_IN-1][0:L6_K-1][0:L6_K-1];
  integer signed l6_relu [0:L6_F_OUT-1][0:L6_H_CONV_OUT-1][0:L6_W_CONV_OUT-1];
  logic signed [DATA_W-1:0] l6_out [0:L6_F_OUT-1][0:L6_H_OUT-1][0:L6_W_OUT-1];

  logic signed [DATA_W-1:0] l7_wgt [0:L7_F_OUT-1][0:L7_C_IN-1][0:L7_K-1][0:L7_K-1];
  integer signed l7_relu [0:L7_F_OUT-1][0:L7_H_CONV_OUT-1][0:L7_W_CONV_OUT-1];
  logic signed [DATA_W-1:0] l7_out [0:L7_F_OUT-1][0:L7_H_OUT-1][0:L7_W_OUT-1];

  logic signed [DATA_W-1:0] l8_wgt [0:L8_F_OUT-1][0:L8_C_IN-1][0:L8_K-1][0:L8_K-1];
  integer signed l8_relu [0:L8_F_OUT-1][0:L8_H_CONV_OUT-1][0:L8_W_CONV_OUT-1];
  logic signed [DATA_W-1:0] l8_out [0:L8_F_OUT-1][0:L8_H_OUT-1][0:L8_W_OUT-1];


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

always_ff @(posedge clk) begin
  if (rst_n && dut.dbg_layer_idx == 0 && dut.u_control_unit_top.ofm_layer_write_done) begin
    $display("DBG_CU_L0_DONE_GATE t=%0t next_valid=%0b cur_mode=%0d next_mode=%0d sm_m1_active=%0b sm_m1_tiled_active=%0b ofm2ifm_active=%0b ofm2ifm_initial_ready=%0b transition_busy=%0b ofm_ifm_busy=%0b",
             $time,
             dut.u_control_unit_top.next_valid_s,
             dut.u_control_unit_top.cur_cfg_s.mode,
             dut.u_control_unit_top.next_cfg_s.mode,
             dut.u_control_unit_top.sm_m1_active,
             dut.u_control_unit_top.sm_m1_tiled_active_s,
             dut.u_control_unit_top.ofm2ifm_active_q,
             dut.u_control_unit_top.ofm2ifm_initial_ready_q,
             dut.u_control_unit_top.transition_busy_s,
             dut.u_control_unit_top.ofm_ifm_stream_busy);
  end
end

// ============================================================
// DEBUG L0 -> L1 OFM->IFM handoff
// Expected initial stream before entering L1:
//   L1 IFM = 47x47x8
//   HT = 4
//   Pv_dest = 8
//   Pf_dest = 4
//   commands = 4 * ceil(47/8) * ceil(8/4) = 48
// ============================================================

localparam int DBG_L01_EXP_INITIAL_STREAM = 48;

integer dbg_l01_stream_start_count;
integer dbg_l01_stream_done_count;
logic   dbg_l0_compute_done_seen_q;
logic   dbg_layer1_seen_q;

always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    dbg_l01_stream_start_count <= 0;
    dbg_l01_stream_done_count  <= 0;
    dbg_l0_compute_done_seen_q <= 1'b0;
    dbg_layer1_seen_q          <= 1'b0;
  end else begin
    if (dut.dbg_layer_idx == 0 && dut.m1_done_s) begin
      dbg_l0_compute_done_seen_q <= 1'b1;
      $display("DBG_L01_L0_M1_DONE t=%0t", $time);
    end

    if (!dbg_layer1_seen_q && dut.dbg_layer_idx == 1) begin
      dbg_layer1_seen_q <= 1'b1;
      $display("DBG_L01_LAYER1_SEEN t=%0t stream_start=%0d stream_done=%0d",
               $time,
               dbg_l01_stream_start_count,
               dbg_l01_stream_done_count);
    end

    if (dbg_l0_compute_done_seen_q && !dbg_layer1_seen_q) begin
      if (dut.ofm_ifm_stream_start_s) begin
        dbg_l01_stream_start_count <= dbg_l01_stream_start_count + 1;

        $display("DBG_L01_STREAM_START t=%0t cnt=%0d row=%0d col_base=%0d slot=%0d ch_blk=%0d",
                 $time,
                 dbg_l01_stream_start_count + 1,
                 dut.ofm_ifm_stream_row_base_s,
                 dut.ofm_ifm_stream_col_base_s,
                 dut.ofm_ifm_stream_m1_row_slot_l_s,
                 dut.ofm_ifm_stream_m1_ch_blk_g_s);
      end

      if (dut.ofm_ifm_stream_done_s) begin
        dbg_l01_stream_done_count <= dbg_l01_stream_done_count + 1;

        $display("DBG_L01_STREAM_DONE t=%0t cnt=%0d",
                 $time,
                 dbg_l01_stream_done_count + 1);
      end
    end
  end
end

final begin
  $display("DBG_L01_SUMMARY stream_start=%0d expected_initial=%0d",
           dbg_l01_stream_start_count,
           DBG_L01_EXP_INITIAL_STREAM);

  $display("DBG_L01_SUMMARY stream_done=%0d expected_initial=%0d",
           dbg_l01_stream_done_count,
           DBG_L01_EXP_INITIAL_STREAM);

  $display("DBG_L01_SUMMARY l0_done_seen=%0b layer1_seen=%0b",
           dbg_l0_compute_done_seen_q,
           dbg_layer1_seen_q);
end

// ============================================================
// DEBUG L0: dynamic Pv/Pf layer-0 stuck diagnosis
// Target test:
//   L0: 96x96x3 -> conv 94x94x8 K=3 pool_en=1 -> pool 47x47x8
//   Pv=8, Pf=4, PTOTAL=32
//
// Expected L0 OFM write pulses:
//   pooled H = 47
//   pooled W = 47
//   write col pack = Pv/2 = 4
//   filter groups = ceil(8/4) = 2
//   pulses = 47 * ceil(47/4) * 2 = 47 * 12 * 2 = 1128
//
// Expected L0 valid output elements:
//   47 * 47 * 8 = 17672
// ============================================================

localparam int DBG_L0_EXP_OFM_WRITE_PULSES = 1128;
localparam int DBG_L0_EXP_VALID_ELEMS      = 17672;

integer dbg_l0_cycle_count;
integer dbg_l0_ifm_dma_write_count;
integer dbg_l0_ofm_write_pulse_count;
integer dbg_l0_ofm_valid_elem_count;
integer dbg_l0_free_count;
logic   dbg_l0_start_seen_q;
logic   dbg_l0_m1_done_seen_q;
logic   dbg_l0_ofm_layer_done_seen_q;

always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    dbg_l0_cycle_count           <= 0;
    dbg_l0_ifm_dma_write_count   <= 0;
    dbg_l0_ofm_write_pulse_count <= 0;
    dbg_l0_ofm_valid_elem_count  <= 0;
    dbg_l0_free_count            <= 0;
    dbg_l0_start_seen_q          <= 1'b0;
    dbg_l0_m1_done_seen_q        <= 1'b0;
    dbg_l0_ofm_layer_done_seen_q <= 1'b0;
  end else begin
    if (dut.dbg_layer_idx == 0) begin
      dbg_l0_cycle_count <= dbg_l0_cycle_count + 1;

      if (dut.ifm_dma_wr_en_s) begin
        dbg_l0_ifm_dma_write_count <= dbg_l0_ifm_dma_write_count + 1;

        if (dbg_l0_ifm_dma_write_count < 20 ||
            (dbg_l0_ifm_dma_write_count % 200) == 0) begin
          $display("DBG_L0_IFM_DMA_WR t=%0t cnt=%0d bank=%0d row_slot=%0d col=%0d data=0x%0h",
                   $time,
                   dbg_l0_ifm_dma_write_count + 1,
                   dut.ifm_dma_wr_bank_s,
                   dut.ifm_dma_wr_row_idx_s,
                   dut.ifm_dma_wr_col_idx_s,
                   dut.ifm_dma_wr_data_s);
        end
      end

      if (dut.m1_start_s && !dbg_l0_start_seen_q) begin
        dbg_l0_start_seen_q <= 1'b1;

        $display("DBG_L0_M1_START t=%0t ifm_dma_wr_count=%0d m1_busy=%0b m1_step=%0b out_row=%0d out_col=%0d f_group=%0d c_iter=%0d ky=%0d kx=%0d",
                 $time,
                 dbg_l0_ifm_dma_write_count,
                 dut.m1_busy_s,
                 dut.m1_step_en_s,
                 dut.m1_out_row_s,
                 dut.m1_out_col_s,
                 dut.m1_f_group_s,
                 dut.m1_c_iter_s,
                 dut.m1_ky_s,
                 dut.m1_kx_s);
      end

      if (dut.ifm_m1_free_valid) begin
        dbg_l0_free_count <= dbg_l0_free_count + 1;

        if (dbg_l0_free_count < 20 ||
            (dbg_l0_free_count % 20) == 0) begin
          $display("DBG_L0_FREE t=%0t cnt=%0d row_base_l=%0d free_slot=%0d free_row_g=%0d m1_out_row=%0d",
                   $time,
                   dbg_l0_free_count + 1,
                   dut.ifm_m1_row_base_l_s,
                   dut.ifm_m1_free_row_slot_l,
                   dut.ifm_m1_free_row_g,
                   dut.m1_out_row_s);
        end
      end

      if (dut.m1_ofm_write_en_s) begin
        dbg_l0_ofm_write_pulse_count <= dbg_l0_ofm_write_pulse_count + 1;
        dbg_l0_ofm_valid_elem_count  <= dbg_l0_ofm_valid_elem_count + dut.m1_ofm_write_count_s;

        if (dbg_l0_ofm_write_pulse_count < 40 ||
            (dbg_l0_ofm_write_pulse_count % 100) == 0) begin
          $display("DBG_L0_OFM_WRITE t=%0t cnt=%0d row=%0d col_base=%0d fbase=%0d count=%0d data0=%0h data1=%0h data2=%0h data3=%0h data4=%0h data5=%0h data6=%0h data7=%0h",
                   $time,
                   dbg_l0_ofm_write_pulse_count + 1,
                   dut.m1_ofm_write_row_s,
                   dut.m1_ofm_write_col_base_s,
                   dut.m1_ofm_write_filter_base_s,
                   dut.m1_ofm_write_count_s,
                   dut.m1_ofm_write_data_s[0],
                   dut.m1_ofm_write_data_s[1],
                   dut.m1_ofm_write_data_s[2],
                   dut.m1_ofm_write_data_s[3],
                   dut.m1_ofm_write_data_s[4],
                   dut.m1_ofm_write_data_s[5],
                   dut.m1_ofm_write_data_s[6],
                   dut.m1_ofm_write_data_s[7]);
        end
      end

      if (dut.m1_done_s && !dbg_l0_m1_done_seen_q) begin
        dbg_l0_m1_done_seen_q <= 1'b1;

        $display("DBG_L0_M1_DONE t=%0t ofm_write_pulse=%0d/%0d valid_elem=%0d/%0d free=%0d",
                 $time,
                 dbg_l0_ofm_write_pulse_count,
                 DBG_L0_EXP_OFM_WRITE_PULSES,
                 dbg_l0_ofm_valid_elem_count,
                 DBG_L0_EXP_VALID_ELEMS,
                 dbg_l0_free_count);
      end

      if (dut.ofm_layer_write_done_s && !dbg_l0_ofm_layer_done_seen_q) begin
        dbg_l0_ofm_layer_done_seen_q <= 1'b1;

        $display("DBG_L0_OFM_LAYER_WRITE_DONE t=%0t ofm_write_pulse=%0d/%0d valid_elem=%0d/%0d",
                 $time,
                 dbg_l0_ofm_write_pulse_count,
                 DBG_L0_EXP_OFM_WRITE_PULSES,
                 dbg_l0_ofm_valid_elem_count,
                 DBG_L0_EXP_VALID_ELEMS);
      end

      if ((dbg_l0_cycle_count % 100000) == 0) begin
        $display("DBG_L0_STATUS t=%0t cycle=%0d ifm_dma_wr=%0d ofm_write_pulse=%0d/%0d valid_elem=%0d/%0d free=%0d start_seen=%0b m1_busy=%0b m1_done=%0b ofm_done=%0b m1_step=%0b out_row=%0d out_col=%0d f_group=%0d c_iter=%0d ky=%0d kx=%0d",
                 $time,
                 dbg_l0_cycle_count,
                 dbg_l0_ifm_dma_write_count,
                 dbg_l0_ofm_write_pulse_count,
                 DBG_L0_EXP_OFM_WRITE_PULSES,
                 dbg_l0_ofm_valid_elem_count,
                 DBG_L0_EXP_VALID_ELEMS,
                 dbg_l0_free_count,
                 dbg_l0_start_seen_q,
                 dut.m1_busy_s,
                 dut.m1_done_s,
                 dut.ofm_layer_write_done_s,
                 dut.m1_step_en_s,
                 dut.m1_out_row_s,
                 dut.m1_out_col_s,
                 dut.m1_f_group_s,
                 dut.m1_c_iter_s,
                 dut.m1_ky_s,
                 dut.m1_kx_s);
      end
    end
  end
end

final begin
  $display("DBG_L0_SUMMARY ifm_dma_wr=%0d", dbg_l0_ifm_dma_write_count);

  $display("DBG_L0_SUMMARY ofm_write_pulse=%0d expected=%0d",
           dbg_l0_ofm_write_pulse_count,
           DBG_L0_EXP_OFM_WRITE_PULSES);

  $display("DBG_L0_SUMMARY ofm_valid_elem=%0d expected=%0d",
           dbg_l0_ofm_valid_elem_count,
           DBG_L0_EXP_VALID_ELEMS);

  $display("DBG_L0_SUMMARY free_count=%0d start_seen=%0b m1_done_seen=%0b ofm_layer_done_seen=%0b",
           dbg_l0_free_count,
           dbg_l0_start_seen_q,
           dbg_l0_m1_done_seen_q,
           dbg_l0_ofm_layer_done_seen_q);
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
            l0_ifm[c][r][x] = $signed(((c*7 + r*3 + x*5 + (r*x)%11) % 7) - 3);

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

      if (L0_POOL_EN) begin
        for (f=0;f<L0_F_OUT;f=f+1)
          for (pr=0;pr<L0_H_OUT;pr=pr+1)
            for (pc=0;pc<L0_W_OUT;pc=pc+1) begin
              max_v = l0_relu[f][2*pr][2*pc];
              for (dy=0;dy<2;dy=dy+1)
                for (dx=0;dx<2;dx=dx+1)
                  if (l0_relu[f][2*pr+dy][2*pc+dx] > max_v) max_v = l0_relu[f][2*pr+dy][2*pc+dx];
              l0_out[f][pr][pc] = sat_to_i8(max_v);
            end
      end else begin
        for (f=0;f<L0_F_OUT;f=f+1)
          for (r=0;r<L0_H_OUT;r=r+1)
            for (x=0;x<L0_W_OUT;x=x+1)
              l0_out[f][r][x] = sat_to_i8(l0_relu[f][r][x]);
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
                  sum += $signed(l0_out[c][r+ky][x+kx]) * $signed(l1_wgt[f][c][ky][kx]);
            l1_relu[f][r][x] = (sum > 0) ? sum : 0;
          end

      if (L1_POOL_EN) begin
        for (f=0;f<L1_F_OUT;f=f+1)
          for (pr=0;pr<L1_H_OUT;pr=pr+1)
            for (pc=0;pc<L1_W_OUT;pc=pc+1) begin
              max_v = l1_relu[f][2*pr][2*pc];
              for (dy=0;dy<2;dy=dy+1)
                for (dx=0;dx<2;dx=dx+1)
                  if (l1_relu[f][2*pr+dy][2*pc+dx] > max_v) max_v = l1_relu[f][2*pr+dy][2*pc+dx];
              l1_out[f][pr][pc] = sat_to_i8(max_v);
            end
      end else begin
        for (f=0;f<L1_F_OUT;f=f+1)
          for (r=0;r<L1_H_OUT;r=r+1)
            for (x=0;x<L1_W_OUT;x=x+1)
              l1_out[f][r][x] = sat_to_i8(l1_relu[f][r][x]);
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
                  sum += $signed(l1_out[c][r+ky][x+kx]) * $signed(l2_wgt[f][c][ky][kx]);
            l2_relu[f][r][x] = (sum > 0) ? sum : 0;
          end

      if (L2_POOL_EN) begin
        for (f=0;f<L2_F_OUT;f=f+1)
          for (pr=0;pr<L2_H_OUT;pr=pr+1)
            for (pc=0;pc<L2_W_OUT;pc=pc+1) begin
              max_v = l2_relu[f][2*pr][2*pc];
              for (dy=0;dy<2;dy=dy+1)
                for (dx=0;dx<2;dx=dx+1)
                  if (l2_relu[f][2*pr+dy][2*pc+dx] > max_v) max_v = l2_relu[f][2*pr+dy][2*pc+dx];
              l2_out[f][pr][pc] = sat_to_i8(max_v);
            end
      end else begin
        for (f=0;f<L2_F_OUT;f=f+1)
          for (r=0;r<L2_H_OUT;r=r+1)
            for (x=0;x<L2_W_OUT;x=x+1)
              l2_out[f][r][x] = sat_to_i8(l2_relu[f][r][x]);
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
                  sum += $signed(l2_out[c][r+ky][x+kx]) * $signed(l3_wgt[f][c][ky][kx]);
            l3_relu[f][r][x] = (sum > 0) ? sum : 0;
          end

      if (L3_POOL_EN) begin
        for (f=0;f<L3_F_OUT;f=f+1)
          for (pr=0;pr<L3_H_OUT;pr=pr+1)
            for (pc=0;pc<L3_W_OUT;pc=pc+1) begin
              max_v = l3_relu[f][2*pr][2*pc];
              for (dy=0;dy<2;dy=dy+1)
                for (dx=0;dx<2;dx=dx+1)
                  if (l3_relu[f][2*pr+dy][2*pc+dx] > max_v) max_v = l3_relu[f][2*pr+dy][2*pc+dx];
              l3_out[f][pr][pc] = sat_to_i8(max_v);
            end
      end else begin
        for (f=0;f<L3_F_OUT;f=f+1)
          for (r=0;r<L3_H_OUT;r=r+1)
            for (x=0;x<L3_W_OUT;x=x+1)
              l3_out[f][r][x] = sat_to_i8(l3_relu[f][r][x]);
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
                  sum += $signed(l3_out[c][r+ky][x+kx]) * $signed(l4_wgt[f][c][ky][kx]);
            l4_relu[f][r][x] = (sum > 0) ? sum : 0;
          end

      if (L4_POOL_EN) begin
        for (f=0;f<L4_F_OUT;f=f+1)
          for (pr=0;pr<L4_H_OUT;pr=pr+1)
            for (pc=0;pc<L4_W_OUT;pc=pc+1) begin
              max_v = l4_relu[f][2*pr][2*pc];
              for (dy=0;dy<2;dy=dy+1)
                for (dx=0;dx<2;dx=dx+1)
                  if (l4_relu[f][2*pr+dy][2*pc+dx] > max_v) max_v = l4_relu[f][2*pr+dy][2*pc+dx];
              l4_out[f][pr][pc] = sat_to_i8(max_v);
            end
      end else begin
        for (f=0;f<L4_F_OUT;f=f+1)
          for (r=0;r<L4_H_OUT;r=r+1)
            for (x=0;x<L4_W_OUT;x=x+1)
              l4_out[f][r][x] = sat_to_i8(l4_relu[f][r][x]);
      end

      for (f=0;f<L5_F_OUT;f=f+1)
        for (c=0;c<L5_C_IN;c=c+1)
          for (ky=0;ky<L5_K;ky=ky+1)
            for (kx=0;kx<L5_K;kx=kx+1)
              l5_wgt[f][c][ky][kx] = $signed(((f*17 + c*13 + ky*10 + kx*16 + 6) % 3) - 1);

      for (f=0;f<L5_F_OUT;f=f+1)
        for (r=0;r<L5_H_CONV_OUT;r=r+1)
          for (x=0;x<L5_W_CONV_OUT;x=x+1) begin
            sum=0;
            for (c=0;c<L5_C_IN;c=c+1)
              for (ky=0;ky<L5_K;ky=ky+1)
                for (kx=0;kx<L5_K;kx=kx+1)
                  sum += $signed(l4_out[c][r+ky][x+kx]) * $signed(l5_wgt[f][c][ky][kx]);
            l5_relu[f][r][x] = (sum > 0) ? sum : 0;
          end

      if (L5_POOL_EN) begin
        for (f=0;f<L5_F_OUT;f=f+1)
          for (pr=0;pr<L5_H_OUT;pr=pr+1)
            for (pc=0;pc<L5_W_OUT;pc=pc+1) begin
              max_v = l5_relu[f][2*pr][2*pc];
              for (dy=0;dy<2;dy=dy+1)
                for (dx=0;dx<2;dx=dx+1)
                  if (l5_relu[f][2*pr+dy][2*pc+dx] > max_v) max_v = l5_relu[f][2*pr+dy][2*pc+dx];
              l5_out[f][pr][pc] = sat_to_i8(max_v);
            end
      end else begin
        for (f=0;f<L5_F_OUT;f=f+1)
          for (r=0;r<L5_H_OUT;r=r+1)
            for (x=0;x<L5_W_OUT;x=x+1)
              l5_out[f][r][x] = sat_to_i8(l5_relu[f][r][x]);
      end

      for (f=0;f<L6_F_OUT;f=f+1)
        for (c=0;c<L6_C_IN;c=c+1)
          for (ky=0;ky<L6_K;ky=ky+1)
            for (kx=0;kx<L6_K;kx=kx+1)
              l6_wgt[f][c][ky][kx] = $signed(((f*19 + c*15 + ky*11 + kx*17 + 7) % 3) - 1);

      for (f=0;f<L6_F_OUT;f=f+1)
        for (r=0;r<L6_H_CONV_OUT;r=r+1)
          for (x=0;x<L6_W_CONV_OUT;x=x+1) begin
            sum=0;
            for (c=0;c<L6_C_IN;c=c+1)
              for (ky=0;ky<L6_K;ky=ky+1)
                for (kx=0;kx<L6_K;kx=kx+1)
                  sum += $signed(l5_out[c][r+ky][x+kx]) * $signed(l6_wgt[f][c][ky][kx]);
            l6_relu[f][r][x] = (sum > 0) ? sum : 0;
          end

      if (L6_POOL_EN) begin
        for (f=0;f<L6_F_OUT;f=f+1)
          for (pr=0;pr<L6_H_OUT;pr=pr+1)
            for (pc=0;pc<L6_W_OUT;pc=pc+1) begin
              max_v = l6_relu[f][2*pr][2*pc];
              for (dy=0;dy<2;dy=dy+1)
                for (dx=0;dx<2;dx=dx+1)
                  if (l6_relu[f][2*pr+dy][2*pc+dx] > max_v) max_v = l6_relu[f][2*pr+dy][2*pc+dx];
              l6_out[f][pr][pc] = sat_to_i8(max_v);
            end
      end else begin
        for (f=0;f<L6_F_OUT;f=f+1)
          for (r=0;r<L6_H_OUT;r=r+1)
            for (x=0;x<L6_W_OUT;x=x+1)
              l6_out[f][r][x] = sat_to_i8(l6_relu[f][r][x]);
      end

      for (f=0;f<L7_F_OUT;f=f+1)
        for (c=0;c<L7_C_IN;c=c+1)
          for (ky=0;ky<L7_K;ky=ky+1)
            for (kx=0;kx<L7_K;kx=kx+1)
              l7_wgt[f][c][ky][kx] = $signed(((f*21 + c*17 + ky*12 + kx*18 + 8) % 3) - 1);

      for (f=0;f<L7_F_OUT;f=f+1)
        for (r=0;r<L7_H_CONV_OUT;r=r+1)
          for (x=0;x<L7_W_CONV_OUT;x=x+1) begin
            sum=0;
            for (c=0;c<L7_C_IN;c=c+1)
              for (ky=0;ky<L7_K;ky=ky+1)
                for (kx=0;kx<L7_K;kx=kx+1)
                  sum += $signed(l6_out[c][r+ky][x+kx]) * $signed(l7_wgt[f][c][ky][kx]);
            l7_relu[f][r][x] = (sum > 0) ? sum : 0;
          end

      if (L7_POOL_EN) begin
        for (f=0;f<L7_F_OUT;f=f+1)
          for (pr=0;pr<L7_H_OUT;pr=pr+1)
            for (pc=0;pc<L7_W_OUT;pc=pc+1) begin
              max_v = l7_relu[f][2*pr][2*pc];
              for (dy=0;dy<2;dy=dy+1)
                for (dx=0;dx<2;dx=dx+1)
                  if (l7_relu[f][2*pr+dy][2*pc+dx] > max_v) max_v = l7_relu[f][2*pr+dy][2*pc+dx];
              l7_out[f][pr][pc] = sat_to_i8(max_v);
            end
      end else begin
        for (f=0;f<L7_F_OUT;f=f+1)
          for (r=0;r<L7_H_OUT;r=r+1)
            for (x=0;x<L7_W_OUT;x=x+1)
              l7_out[f][r][x] = sat_to_i8(l7_relu[f][r][x]);
      end

      for (f=0;f<L8_F_OUT;f=f+1)
        for (c=0;c<L8_C_IN;c=c+1)
          for (ky=0;ky<L8_K;ky=ky+1)
            for (kx=0;kx<L8_K;kx=kx+1)
              l8_wgt[f][c][ky][kx] = $signed(((f*23 + c*19 + ky*13 + kx*19 + 9) % 3) - 1);

      for (f=0;f<L8_F_OUT;f=f+1)
        for (r=0;r<L8_H_CONV_OUT;r=r+1)
          for (x=0;x<L8_W_CONV_OUT;x=x+1) begin
            sum=0;
            for (c=0;c<L8_C_IN;c=c+1)
              for (ky=0;ky<L8_K;ky=ky+1)
                for (kx=0;kx<L8_K;kx=kx+1)
                  sum += $signed(l7_out[c][r+ky][x+kx]) * $signed(l8_wgt[f][c][ky][kx]);
            l8_relu[f][r][x] = (sum > 0) ? sum : 0;
          end

      if (L8_POOL_EN) begin
        for (f=0;f<L8_F_OUT;f=f+1)
          for (pr=0;pr<L8_H_OUT;pr=pr+1)
            for (pc=0;pc<L8_W_OUT;pc=pc+1) begin
              max_v = l8_relu[f][2*pr][2*pc];
              for (dy=0;dy<2;dy=dy+1)
                for (dx=0;dx<2;dx=dx+1)
                  if (l8_relu[f][2*pr+dy][2*pc+dx] > max_v) max_v = l8_relu[f][2*pr+dy][2*pc+dx];
              l8_out[f][pr][pc] = sat_to_i8(max_v);
            end
      end else begin
        for (f=0;f<L8_F_OUT;f=f+1)
          for (r=0;r<L8_H_OUT;r=r+1)
            for (x=0;x<L8_W_OUT;x=x+1)
              l8_out[f][r][x] = sat_to_i8(l8_relu[f][r][x]);
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
              abs_col = colg*L0_PV + lane;
              if ((lane<L0_PV) && (abs_col<L0_W_IN))
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
              phys_word=logical_idx/L0_PV; subword_idx=logical_idx%L0_PV; base_lane=subword_idx*L0_PF;
              for (pf=0;pf<L0_PF;pf=pf+1) begin
                f=fg*L0_PF+pf;
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
              phys_word=logical_idx/L1_PV; subword_idx=logical_idx%L1_PV; base_lane=subword_idx*L1_PF;
              for (pf=0;pf<L1_PF;pf=pf+1) begin
                f=fg*L1_PF+pf;
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
              phys_word=logical_idx/L2_PV; subword_idx=logical_idx%L2_PV; base_lane=subword_idx*L2_PF;
              for (pf=0;pf<L2_PF;pf=pf+1) begin
                f=fg*L2_PF+pf;
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
              phys_word=logical_idx/L3_PV; subword_idx=logical_idx%L3_PV; base_lane=subword_idx*L3_PF;
              for (pf=0;pf<L3_PF;pf=pf+1) begin
                f=fg*L3_PF+pf;
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
              phys_word=logical_idx/L4_PV; subword_idx=logical_idx%L4_PV; base_lane=subword_idx*L4_PF;
              for (pf=0;pf<L4_PF;pf=pf+1) begin
                f=fg*L4_PF+pf;
                write_wgt_lane_to_ddr(L4_WGT_DDR_BASE, phys_word, base_lane+pf, (f<L4_F_OUT)?l4_wgt[f][c][ky][kx]:'0);
              end
              logical_idx++;
            end
    end
  endtask

  task automatic load_l5_weights_to_ddr;
    integer p, logical_idx, phys_word, subword_idx, base_lane, fg,c,ky,kx,pf,f;
    begin
      for (p=0;p<L5_M1_WGT_DDR_WORDS;p=p+1) ddr_mem[L5_WGT_DDR_BASE+p]='0;
      logical_idx=0;
      for (fg=0;fg<L5_NUM_FGROUP;fg=fg+1)
        for (c=0;c<L5_C_IN;c=c+1)
          for (ky=0;ky<L5_K;ky=ky+1)
            for (kx=0;kx<L5_K;kx=kx+1) begin
              phys_word=logical_idx/L5_PV; subword_idx=logical_idx%L5_PV; base_lane=subword_idx*L5_PF;
              for (pf=0;pf<L5_PF;pf=pf+1) begin
                f=fg*L5_PF+pf;
                write_wgt_lane_to_ddr(L5_WGT_DDR_BASE, phys_word, base_lane+pf, (f<L5_F_OUT)?l5_wgt[f][c][ky][kx]:'0);
              end
              logical_idx++;
            end
    end
  endtask

  task automatic load_l6_weights_to_ddr;
    integer p, logical_idx, phys_word, subword_idx, base_lane, fg,c,ky,kx,pf,f;
    begin
      for (p=0;p<L6_M1_WGT_DDR_WORDS;p=p+1) ddr_mem[L6_WGT_DDR_BASE+p]='0;
      logical_idx=0;
      for (fg=0;fg<L6_NUM_FGROUP;fg=fg+1)
        for (c=0;c<L6_C_IN;c=c+1)
          for (ky=0;ky<L6_K;ky=ky+1)
            for (kx=0;kx<L6_K;kx=kx+1) begin
              phys_word=logical_idx/L6_PV; subword_idx=logical_idx%L6_PV; base_lane=subword_idx*L6_PF;
              for (pf=0;pf<L6_PF;pf=pf+1) begin
                f=fg*L6_PF+pf;
                write_wgt_lane_to_ddr(L6_WGT_DDR_BASE, phys_word, base_lane+pf, (f<L6_F_OUT)?l6_wgt[f][c][ky][kx]:'0);
              end
              logical_idx++;
            end
    end
  endtask

  task automatic load_l7_weights_to_ddr;
    integer p, logical_idx, phys_word, subword_idx, base_lane, fg,c,ky,kx,pf,f;
    begin
      for (p=0;p<L7_M1_WGT_DDR_WORDS;p=p+1) ddr_mem[L7_WGT_DDR_BASE+p]='0;
      logical_idx=0;
      for (fg=0;fg<L7_NUM_FGROUP;fg=fg+1)
        for (c=0;c<L7_C_IN;c=c+1)
          for (ky=0;ky<L7_K;ky=ky+1)
            for (kx=0;kx<L7_K;kx=kx+1) begin
              phys_word=logical_idx/L7_PV; subword_idx=logical_idx%L7_PV; base_lane=subword_idx*L7_PF;
              for (pf=0;pf<L7_PF;pf=pf+1) begin
                f=fg*L7_PF+pf;
                write_wgt_lane_to_ddr(L7_WGT_DDR_BASE, phys_word, base_lane+pf, (f<L7_F_OUT)?l7_wgt[f][c][ky][kx]:'0);
              end
              logical_idx++;
            end
    end
  endtask

  task automatic load_l8_weights_to_ddr;
    integer p, logical_idx, phys_word, subword_idx, base_lane, fg,c,ky,kx,pf,f;
    begin
      for (p=0;p<L8_M1_WGT_DDR_WORDS;p=p+1) ddr_mem[L8_WGT_DDR_BASE+p]='0;
      logical_idx=0;
      for (fg=0;fg<L8_NUM_FGROUP;fg=fg+1)
        for (c=0;c<L8_C_IN;c=c+1)
          for (ky=0;ky<L8_K;ky=ky+1)
            for (kx=0;kx<L8_K;kx=kx+1) begin
              phys_word=logical_idx/L8_PV; subword_idx=logical_idx%L8_PV; base_lane=subword_idx*L8_PF;
              for (pf=0;pf<L8_PF;pf=pf+1) begin
                f=fg*L8_PF+pf;
                write_wgt_lane_to_ddr(L8_WGT_DDR_BASE, phys_word, base_lane+pf, (f<L8_F_OUT)?l8_wgt[f][c][ky][kx]:'0);
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
      load_l5_weights_to_ddr();
      load_l6_weights_to_ddr();
      load_l7_weights_to_ddr();
      load_l8_weights_to_ddr();
      $display("TB_INFO: 9-layer EfficientNet-like Mode1 prefix test, depth-trend dynamic Pv/Pf");
      $display("TB_INFO: L0 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d pool_en=%0d -> out %0dx%0dx%0d", L0_H_IN,L0_W_IN,L0_C_IN,L0_H_CONV_OUT,L0_W_CONV_OUT,L0_F_OUT,L0_K,L0_POOL_EN,L0_H_OUT,L0_W_OUT,L0_F_OUT);
      $display("TB_INFO: L1 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d pool_en=%0d -> out %0dx%0dx%0d", L1_H_IN,L1_W_IN,L1_C_IN,L1_H_CONV_OUT,L1_W_CONV_OUT,L1_F_OUT,L1_K,L1_POOL_EN,L1_H_OUT,L1_W_OUT,L1_F_OUT);
      $display("TB_INFO: L2 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d pool_en=%0d -> out %0dx%0dx%0d", L2_H_IN,L2_W_IN,L2_C_IN,L2_H_CONV_OUT,L2_W_CONV_OUT,L2_F_OUT,L2_K,L2_POOL_EN,L2_H_OUT,L2_W_OUT,L2_F_OUT);
      $display("TB_INFO: L3 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d pool_en=%0d -> out %0dx%0dx%0d", L3_H_IN,L3_W_IN,L3_C_IN,L3_H_CONV_OUT,L3_W_CONV_OUT,L3_F_OUT,L3_K,L3_POOL_EN,L3_H_OUT,L3_W_OUT,L3_F_OUT);
      $display("TB_INFO: L4 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d pool_en=%0d -> out %0dx%0dx%0d", L4_H_IN,L4_W_IN,L4_C_IN,L4_H_CONV_OUT,L4_W_CONV_OUT,L4_F_OUT,L4_K,L4_POOL_EN,L4_H_OUT,L4_W_OUT,L4_F_OUT);
      $display("TB_INFO: L5 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d pool_en=%0d -> out %0dx%0dx%0d", L5_H_IN,L5_W_IN,L5_C_IN,L5_H_CONV_OUT,L5_W_CONV_OUT,L5_F_OUT,L5_K,L5_POOL_EN,L5_H_OUT,L5_W_OUT,L5_F_OUT);
      $display("TB_INFO: L6 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d pool_en=%0d -> out %0dx%0dx%0d", L6_H_IN,L6_W_IN,L6_C_IN,L6_H_CONV_OUT,L6_W_CONV_OUT,L6_F_OUT,L6_K,L6_POOL_EN,L6_H_OUT,L6_W_OUT,L6_F_OUT);
      $display("TB_INFO: L7 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d pool_en=%0d -> out %0dx%0dx%0d", L7_H_IN,L7_W_IN,L7_C_IN,L7_H_CONV_OUT,L7_W_CONV_OUT,L7_F_OUT,L7_K,L7_POOL_EN,L7_H_OUT,L7_W_OUT,L7_F_OUT);
      $display("TB_INFO: L8 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d pool_en=%0d -> out %0dx%0dx%0d", L8_H_IN,L8_W_IN,L8_C_IN,L8_H_CONV_OUT,L8_W_CONV_OUT,L8_F_OUT,L8_K,L8_POOL_EN,L8_H_OUT,L8_W_OUT,L8_F_OUT);
      $display("TB_INFO: benchmark intent: EfficientNet-like mode-1 prefix; all K=3; early Pv high, later Pf high, PTOTAL=32; no depthwise/SE/skip ops");
      $display("TB_INFO: stress params: H/W/C/F include non-divisible partial groups; HT=%0d C_MAX=%0d F_MAX=%0d", HT, C_MAX, F_MAX);
      $display("TB_INFO: depth-trend Pv/Pf per layer: L0=%0d/%0d L1=%0d/%0d L2=%0d/%0d L3=%0d/%0d L4=%0d/%0d L5=%0d/%0d L6=%0d/%0d L7=%0d/%0d L8=%0d/%0d PTOTAL=%0d",
               L0_PV,L0_PF,L1_PV,L1_PF,L2_PV,L2_PF,L3_PV,L3_PF,L4_PV,L4_PF,L5_PV,L5_PF,L6_PV,L6_PF,L7_PV,L7_PF,L8_PV,L8_PF,PTOTAL);
      $display("TB_INFO: L0 weight bundles=%0d phys=%0d DDR words=%0d base=0x%05h", L0_M1_LOGICAL_BUNDLES, L0_M1_PHYS_WORDS, L0_M1_WGT_DDR_WORDS, L0_WGT_DDR_BASE);
      $display("TB_INFO: L1 weight bundles=%0d phys=%0d DDR words=%0d base=0x%05h", L1_M1_LOGICAL_BUNDLES, L1_M1_PHYS_WORDS, L1_M1_WGT_DDR_WORDS, L1_WGT_DDR_BASE);
      $display("TB_INFO: L2 weight bundles=%0d phys=%0d DDR words=%0d base=0x%05h", L2_M1_LOGICAL_BUNDLES, L2_M1_PHYS_WORDS, L2_M1_WGT_DDR_WORDS, L2_WGT_DDR_BASE);
      $display("TB_INFO: L3 weight bundles=%0d phys=%0d DDR words=%0d base=0x%05h", L3_M1_LOGICAL_BUNDLES, L3_M1_PHYS_WORDS, L3_M1_WGT_DDR_WORDS, L3_WGT_DDR_BASE);
      $display("TB_INFO: L4 weight bundles=%0d phys=%0d DDR words=%0d base=0x%05h", L4_M1_LOGICAL_BUNDLES, L4_M1_PHYS_WORDS, L4_M1_WGT_DDR_WORDS, L4_WGT_DDR_BASE);
      $display("TB_INFO: L5 weight bundles=%0d phys=%0d DDR words=%0d base=0x%05h", L5_M1_LOGICAL_BUNDLES, L5_M1_PHYS_WORDS, L5_M1_WGT_DDR_WORDS, L5_WGT_DDR_BASE);
      $display("TB_INFO: L6 weight bundles=%0d phys=%0d DDR words=%0d base=0x%05h", L6_M1_LOGICAL_BUNDLES, L6_M1_PHYS_WORDS, L6_M1_WGT_DDR_WORDS, L6_WGT_DDR_BASE);
      $display("TB_INFO: L7 weight bundles=%0d phys=%0d DDR words=%0d base=0x%05h", L7_M1_LOGICAL_BUNDLES, L7_M1_PHYS_WORDS, L7_M1_WGT_DDR_WORDS, L7_WGT_DDR_BASE);
      $display("TB_INFO: L8 weight bundles=%0d phys=%0d DDR words=%0d base=0x%05h", L8_M1_LOGICAL_BUNDLES, L8_M1_PHYS_WORDS, L8_M1_WGT_DDR_WORDS, L8_WGT_DDR_BASE);
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

    layer_desc_t cfg0,cfg1,cfg2,cfg3,cfg4,cfg5,cfg6,cfg7,cfg8;
    begin

      cfg0='0;
      cfg0.layer_id=0; cfg0.mode=MODE1; cfg0.h_in=L0_H_IN; cfg0.w_in=L0_W_IN; cfg0.c_in=L0_C_IN; cfg0.f_out=L0_F_OUT; cfg0.k=L0_K;
      cfg0.h_out=L0_H_CONV_OUT; cfg0.w_out=L0_W_CONV_OUT;
      cfg0.pv_m1=L0_PV; cfg0.pf_m1=L0_PF; cfg0.pc_m2=PC_MODE2; cfg0.pf_m2=PF_MODE2;
      cfg0.conv_stride=1; cfg0.relu_en=1'b1; cfg0.pool_en=(L0_POOL_EN != 0); cfg0.pool_k=2; cfg0.pool_stride=2;
      cfg0.ifm_ddr_base=`DDR_IFM_BASE; cfg0.wgt_ddr_base=L0_WGT_DDR_BASE; cfg0.ofm_ddr_base=`DDR_OFM_BASE;
      cfg0.first_layer=1'b1; cfg0.last_layer=1'b0;

      cfg1='0;
      cfg1.layer_id=1; cfg1.mode=MODE1; cfg1.h_in=L1_H_IN; cfg1.w_in=L1_W_IN; cfg1.c_in=L1_C_IN; cfg1.f_out=L1_F_OUT; cfg1.k=L1_K;
      cfg1.h_out=L1_H_CONV_OUT; cfg1.w_out=L1_W_CONV_OUT;
      cfg1.pv_m1=L1_PV; cfg1.pf_m1=L1_PF; cfg1.pc_m2=PC_MODE2; cfg1.pf_m2=PF_MODE2;
      cfg1.conv_stride=1; cfg1.relu_en=1'b1; cfg1.pool_en=(L1_POOL_EN != 0); cfg1.pool_k=2; cfg1.pool_stride=2;
      cfg1.ifm_ddr_base=`DDR_IFM_BASE; cfg1.wgt_ddr_base=L1_WGT_DDR_BASE; cfg1.ofm_ddr_base=`DDR_OFM_BASE;
      cfg1.first_layer=1'b0; cfg1.last_layer=1'b0;

      cfg2='0;
      cfg2.layer_id=2; cfg2.mode=MODE1; cfg2.h_in=L2_H_IN; cfg2.w_in=L2_W_IN; cfg2.c_in=L2_C_IN; cfg2.f_out=L2_F_OUT; cfg2.k=L2_K;
      cfg2.h_out=L2_H_CONV_OUT; cfg2.w_out=L2_W_CONV_OUT;
      cfg2.pv_m1=L2_PV; cfg2.pf_m1=L2_PF; cfg2.pc_m2=PC_MODE2; cfg2.pf_m2=PF_MODE2;
      cfg2.conv_stride=1; cfg2.relu_en=1'b1; cfg2.pool_en=(L2_POOL_EN != 0); cfg2.pool_k=2; cfg2.pool_stride=2;
      cfg2.ifm_ddr_base=`DDR_IFM_BASE; cfg2.wgt_ddr_base=L2_WGT_DDR_BASE; cfg2.ofm_ddr_base=`DDR_OFM_BASE;
      cfg2.first_layer=1'b0; cfg2.last_layer=1'b0;

      cfg3='0;
      cfg3.layer_id=3; cfg3.mode=MODE1; cfg3.h_in=L3_H_IN; cfg3.w_in=L3_W_IN; cfg3.c_in=L3_C_IN; cfg3.f_out=L3_F_OUT; cfg3.k=L3_K;
      cfg3.h_out=L3_H_CONV_OUT; cfg3.w_out=L3_W_CONV_OUT;
      cfg3.pv_m1=L3_PV; cfg3.pf_m1=L3_PF; cfg3.pc_m2=PC_MODE2; cfg3.pf_m2=PF_MODE2;
      cfg3.conv_stride=1; cfg3.relu_en=1'b1; cfg3.pool_en=(L3_POOL_EN != 0); cfg3.pool_k=2; cfg3.pool_stride=2;
      cfg3.ifm_ddr_base=`DDR_IFM_BASE; cfg3.wgt_ddr_base=L3_WGT_DDR_BASE; cfg3.ofm_ddr_base=`DDR_OFM_BASE;
      cfg3.first_layer=1'b0; cfg3.last_layer=1'b0;

      cfg4='0;
      cfg4.layer_id=4; cfg4.mode=MODE1; cfg4.h_in=L4_H_IN; cfg4.w_in=L4_W_IN; cfg4.c_in=L4_C_IN; cfg4.f_out=L4_F_OUT; cfg4.k=L4_K;
      cfg4.h_out=L4_H_CONV_OUT; cfg4.w_out=L4_W_CONV_OUT;
      cfg4.pv_m1=L4_PV; cfg4.pf_m1=L4_PF; cfg4.pc_m2=PC_MODE2; cfg4.pf_m2=PF_MODE2;
      cfg4.conv_stride=1; cfg4.relu_en=1'b1; cfg4.pool_en=(L4_POOL_EN != 0); cfg4.pool_k=2; cfg4.pool_stride=2;
      cfg4.ifm_ddr_base=`DDR_IFM_BASE; cfg4.wgt_ddr_base=L4_WGT_DDR_BASE; cfg4.ofm_ddr_base=`DDR_OFM_BASE;
      cfg4.first_layer=1'b0; cfg4.last_layer=1'b0;

      cfg5='0;
      cfg5.layer_id=5; cfg5.mode=MODE1; cfg5.h_in=L5_H_IN; cfg5.w_in=L5_W_IN; cfg5.c_in=L5_C_IN; cfg5.f_out=L5_F_OUT; cfg5.k=L5_K;
      cfg5.h_out=L5_H_CONV_OUT; cfg5.w_out=L5_W_CONV_OUT;
      cfg5.pv_m1=L5_PV; cfg5.pf_m1=L5_PF; cfg5.pc_m2=PC_MODE2; cfg5.pf_m2=PF_MODE2;
      cfg5.conv_stride=1; cfg5.relu_en=1'b1; cfg5.pool_en=(L5_POOL_EN != 0); cfg5.pool_k=2; cfg5.pool_stride=2;
      cfg5.ifm_ddr_base=`DDR_IFM_BASE; cfg5.wgt_ddr_base=L5_WGT_DDR_BASE; cfg5.ofm_ddr_base=`DDR_OFM_BASE;
      cfg5.first_layer=1'b0; cfg5.last_layer=1'b0;

      cfg6='0;
      cfg6.layer_id=6; cfg6.mode=MODE1; cfg6.h_in=L6_H_IN; cfg6.w_in=L6_W_IN; cfg6.c_in=L6_C_IN; cfg6.f_out=L6_F_OUT; cfg6.k=L6_K;
      cfg6.h_out=L6_H_CONV_OUT; cfg6.w_out=L6_W_CONV_OUT;
      cfg6.pv_m1=L6_PV; cfg6.pf_m1=L6_PF; cfg6.pc_m2=PC_MODE2; cfg6.pf_m2=PF_MODE2;
      cfg6.conv_stride=1; cfg6.relu_en=1'b1; cfg6.pool_en=(L6_POOL_EN != 0); cfg6.pool_k=2; cfg6.pool_stride=2;
      cfg6.ifm_ddr_base=`DDR_IFM_BASE; cfg6.wgt_ddr_base=L6_WGT_DDR_BASE; cfg6.ofm_ddr_base=`DDR_OFM_BASE;
      cfg6.first_layer=1'b0; cfg6.last_layer=1'b0;

      cfg7='0;
      cfg7.layer_id=7; cfg7.mode=MODE1; cfg7.h_in=L7_H_IN; cfg7.w_in=L7_W_IN; cfg7.c_in=L7_C_IN; cfg7.f_out=L7_F_OUT; cfg7.k=L7_K;
      cfg7.h_out=L7_H_CONV_OUT; cfg7.w_out=L7_W_CONV_OUT;
      cfg7.pv_m1=L7_PV; cfg7.pf_m1=L7_PF; cfg7.pc_m2=PC_MODE2; cfg7.pf_m2=PF_MODE2;
      cfg7.conv_stride=1; cfg7.relu_en=1'b1; cfg7.pool_en=(L7_POOL_EN != 0); cfg7.pool_k=2; cfg7.pool_stride=2;
      cfg7.ifm_ddr_base=`DDR_IFM_BASE; cfg7.wgt_ddr_base=L7_WGT_DDR_BASE; cfg7.ofm_ddr_base=`DDR_OFM_BASE;
      cfg7.first_layer=1'b0; cfg7.last_layer=1'b0;

      cfg8='0;
      cfg8.layer_id=8; cfg8.mode=MODE1; cfg8.h_in=L8_H_IN; cfg8.w_in=L8_W_IN; cfg8.c_in=L8_C_IN; cfg8.f_out=L8_F_OUT; cfg8.k=L8_K;
      cfg8.h_out=L8_H_CONV_OUT; cfg8.w_out=L8_W_CONV_OUT;
      cfg8.pv_m1=L8_PV; cfg8.pf_m1=L8_PF; cfg8.pc_m2=PC_MODE2; cfg8.pf_m2=PF_MODE2;
      cfg8.conv_stride=1; cfg8.relu_en=1'b1; cfg8.pool_en=(L8_POOL_EN != 0); cfg8.pool_k=2; cfg8.pool_stride=2;
      cfg8.ifm_ddr_base=`DDR_IFM_BASE; cfg8.wgt_ddr_base=L8_WGT_DDR_BASE; cfg8.ofm_ddr_base=`DDR_OFM_BASE;
      cfg8.first_layer=1'b0; cfg8.last_layer=1'b1;

      write_cfg(0,cfg0);
      write_cfg(1,cfg1);
      write_cfg(2,cfg2);
      write_cfg(3,cfg3);
      write_cfg(4,cfg4);
      write_cfg(5,cfg5);
      write_cfg(6,cfg6);
      write_cfg(7,cfg7);
      write_cfg(8,cfg8);
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
        ch=lin/(L8_H_OUT*FINAL_GROUPS);
        rem=lin%(L8_H_OUT*FINAL_GROUPS);
        row=rem/FINAL_GROUPS;
        grp=rem%FINAL_GROUPS;
        col=grp*FINAL_STORE_PACK;
        got=ddr_mem[`DDR_OFM_BASE+lin][0 +: DATA_W];
        exp=l8_out[ch][row][col];
        if (got !== exp) begin
          $display("TB_FAIL: final OFM mismatch lin=%0d ch=%0d row=%0d col=%0d got=%0d/0x%02h expected=%0d/0x%02h word=0x%08h",
                   lin,ch,row,col,got,got,exp,exp,ddr_mem[`DDR_OFM_BASE+lin]);
          ofm_mismatch_count++;
        end
      end
      if ((fail_count==0) && (ofm_mismatch_count==0)) begin
        $display("TB_PASS: exact final OFM matches golden for 9-layer EfficientNet-like mode1 prefix test");
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
    cfg_wr_en=1'b0; cfg_wr_addr='0; cfg_wr_data='0; cfg_num_layers=9;
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
