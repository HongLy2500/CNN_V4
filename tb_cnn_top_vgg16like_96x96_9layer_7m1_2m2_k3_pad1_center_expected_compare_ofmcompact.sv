`timescale 1ns/1ps
`include "cnn_ddr_defs.svh"

module tb_cnn_top_vgg16like_96x96_9layer_7m1_2m2_k3_pad1_center_expected_compare;
  import cnn_layer_desc_pkg::*;

  // --------------------------------------------------------------------------
  // VGG16-like 96x96 first-9-conv regression
  // Channel scale = 1/4 of VGG16:
  //   16,16,32,32,64,64,64,128,128
  // L0-L6 : Mode1
  // L7-L8 : Mode2
  // K=3, stride=1, padding=1, ReLU enabled
  // MaxPool 2x2/stride2 after L1, L3, L6
  //
  // Sparse-weight center-tap model:
  //   only center tap (ky=1,kx=1) is non-zero, weight=1;
  //   output filter f selects input channel f % C_in for that layer.
  //
  // Final golden is computed by a small recursive software reference.
  // It applies the same center-tap channel mapping (f % C_in) layer by layer
  // and evaluates each MaxPool window explicitly. This avoids shortcut errors
  // when the input pattern is periodic.
  // --------------------------------------------------------------------------

  localparam int DATA_W = 8;
  localparam int PSUM_W = 32;

  localparam int PC = 16;
  localparam int PF = 16;
  localparam int PTOTAL = PC * PF;
  localparam int PV_M1 = 16;
  localparam int PF_M1 = 16;
  localparam int PV_MAX = 16;
  localparam int PF_MAX = 16;

  localparam int C_MAX = 128;
  localparam int F_MAX = 128;
  localparam int H_MAX = 96;
  localparam int W_MAX = 96;
  localparam int HT = 4;
  localparam int K_MAX = 3;

  localparam int WGT_DEPTH = 768;
  localparam int CFG_DEPTH = 16;

  // --------------------------------------------------------------------------
  // Compact OFM storage for this 96x96 9-layer benchmark.
  // One OFM memory word stores PV_MAX (=16) scalar pixels/lanes. Therefore the
  // physical row stride is the maximum number of packed words needed by any
  // layer row, not W_MAX.
  //
  // For this benchmark:
  //   L0 no-pool  W=96, pack=PV=16   -> ceil(96/16)=6
  //   L1 pool     W=48, pack=PV/2=8  -> ceil(48/8)=6
  //   L2 no-pool  W=48, pack=PV=16   -> ceil(48/16)=3
  //   L3 pool     W=24, pack=PV/2=8  -> ceil(24/8)=3
  //   L4-L6 M1    W<=24              -> <=3
  //   L7-L8 M2    W=12, pack=PC=16   -> ceil(12/16)=1
  // So OFM_ROW_STRIDE=6 is sufficient; do not allocate H_MAX*W_MAX words.
  // --------------------------------------------------------------------------
  localparam int OFM_ROW_STRIDE = 6;
  localparam int OFM_BANK_DEPTH = H_MAX * OFM_ROW_STRIDE;   // 96*6 = 576 words/bank
  localparam int OFM_LINEAR_DEPTH = F_MAX * OFM_BANK_DEPTH; // 128*576 = 73728 words total logical span

  localparam int DDR_ADDR_W = `CNN_DDR_ADDR_W;
  localparam int DDR_WORD_W = PV_MAX * DATA_W;
  localparam int DDR_LANES = PV_MAX;
  localparam int WGT_SUBWORDS = (PTOTAL + DDR_LANES - 1) / DDR_LANES;
  localparam int MEM_DEPTH = (1 << DDR_ADDR_W);

  localparam int TB_IFM_BASE = `DDR_IFM_BASE;
  localparam int TB_WGT_BASE = `DDR_WGT_BASE;
  localparam int TB_OFM_BASE = `DDR_OFM_BASE;

  localparam int CLK_PERIOD_NS = 10;
  localparam int MAX_CYCLES = 200000000;
  localparam int NUM_LAYERS = 9;
  localparam int LK = 3;
  localparam int PAD = 1;

  // --------------------------------------------------------------------------
  // 9-layer VGG16-like shape table.
  // h_conv_out/w_conv_out are descriptor conv output dims before pooling.
  // h_out/w_out are logical output dims after optional pooling.
  // --------------------------------------------------------------------------
  localparam int L0_H_IN=96, L0_W_IN=96, L0_C_IN=3,   L0_F_OUT=16,  L0_POOL_EN=0;
  localparam int L0_H_CONV_OUT=96, L0_W_CONV_OUT=96, L0_H_OUT=96, L0_W_OUT=96;

  localparam int L1_H_IN=96, L1_W_IN=96, L1_C_IN=16,  L1_F_OUT=16,  L1_POOL_EN=1;
  localparam int L1_H_CONV_OUT=96, L1_W_CONV_OUT=96, L1_H_OUT=48, L1_W_OUT=48;

  localparam int L2_H_IN=48, L2_W_IN=48, L2_C_IN=16,  L2_F_OUT=32,  L2_POOL_EN=0;
  localparam int L2_H_CONV_OUT=48, L2_W_CONV_OUT=48, L2_H_OUT=48, L2_W_OUT=48;

  localparam int L3_H_IN=48, L3_W_IN=48, L3_C_IN=32,  L3_F_OUT=32,  L3_POOL_EN=1;
  localparam int L3_H_CONV_OUT=48, L3_W_CONV_OUT=48, L3_H_OUT=24, L3_W_OUT=24;

  localparam int L4_H_IN=24, L4_W_IN=24, L4_C_IN=32,  L4_F_OUT=64,  L4_POOL_EN=0;
  localparam int L4_H_CONV_OUT=24, L4_W_CONV_OUT=24, L4_H_OUT=24, L4_W_OUT=24;

  localparam int L5_H_IN=24, L5_W_IN=24, L5_C_IN=64,  L5_F_OUT=64,  L5_POOL_EN=0;
  localparam int L5_H_CONV_OUT=24, L5_W_CONV_OUT=24, L5_H_OUT=24, L5_W_OUT=24;

  localparam int L6_H_IN=24, L6_W_IN=24, L6_C_IN=64,  L6_F_OUT=64,  L6_POOL_EN=1;
  localparam int L6_H_CONV_OUT=24, L6_W_CONV_OUT=24, L6_H_OUT=12, L6_W_OUT=12;

  localparam int L7_H_IN=12, L7_W_IN=12, L7_C_IN=64,  L7_F_OUT=128, L7_POOL_EN=0;
  localparam int L7_H_CONV_OUT=12, L7_W_CONV_OUT=12, L7_H_OUT=12, L7_W_OUT=12;

  localparam int L8_H_IN=12, L8_W_IN=12, L8_C_IN=128, L8_F_OUT=128, L8_POOL_EN=0;
  localparam int L8_H_CONV_OUT=12, L8_W_CONV_OUT=12, L8_H_OUT=12, L8_W_OUT=12;

  // Weight word counts.
  localparam int L0_NUM_CGROUP = (L0_C_IN + PV_M1 - 1) / PV_M1;
  localparam int L1_NUM_CGROUP = (L1_C_IN + PV_M1 - 1) / PV_M1;
  localparam int L2_NUM_CGROUP = (L2_C_IN + PV_M1 - 1) / PV_M1;
  localparam int L3_NUM_CGROUP = (L3_C_IN + PV_M1 - 1) / PV_M1;
  localparam int L4_NUM_CGROUP = (L4_C_IN + PV_M1 - 1) / PV_M1;
  localparam int L5_NUM_CGROUP = (L5_C_IN + PV_M1 - 1) / PV_M1;
  localparam int L6_NUM_CGROUP = (L6_C_IN + PV_M1 - 1) / PV_M1;
  localparam int L7_NUM_CGROUP = (L7_C_IN + PC - 1) / PC;
  localparam int L8_NUM_CGROUP = (L8_C_IN + PC - 1) / PC;

  localparam int L0_NUM_FGROUP = (L0_F_OUT + PF_M1 - 1) / PF_M1;
  localparam int L1_NUM_FGROUP = (L1_F_OUT + PF_M1 - 1) / PF_M1;
  localparam int L2_NUM_FGROUP = (L2_F_OUT + PF_M1 - 1) / PF_M1;
  localparam int L3_NUM_FGROUP = (L3_F_OUT + PF_M1 - 1) / PF_M1;
  localparam int L4_NUM_FGROUP = (L4_F_OUT + PF_M1 - 1) / PF_M1;
  localparam int L5_NUM_FGROUP = (L5_F_OUT + PF_M1 - 1) / PF_M1;
  localparam int L6_NUM_FGROUP = (L6_F_OUT + PF_M1 - 1) / PF_M1;
  localparam int L7_NUM_FGROUP = (L7_F_OUT + PF - 1) / PF;
  localparam int L8_NUM_FGROUP = (L8_F_OUT + PF - 1) / PF;

  localparam int L0_WGT_WORDS = ((L0_NUM_FGROUP * L0_C_IN * LK * LK) + PV_M1 - 1) / PV_M1;
  localparam int L1_WGT_WORDS = ((L1_NUM_FGROUP * L1_C_IN * LK * LK) + PV_M1 - 1) / PV_M1;
  localparam int L2_WGT_WORDS = ((L2_NUM_FGROUP * L2_C_IN * LK * LK) + PV_M1 - 1) / PV_M1;
  localparam int L3_WGT_WORDS = ((L3_NUM_FGROUP * L3_C_IN * LK * LK) + PV_M1 - 1) / PV_M1;
  localparam int L4_WGT_WORDS = ((L4_NUM_FGROUP * L4_C_IN * LK * LK) + PV_M1 - 1) / PV_M1;
  localparam int L5_WGT_WORDS = ((L5_NUM_FGROUP * L5_C_IN * LK * LK) + PV_M1 - 1) / PV_M1;
  localparam int L6_WGT_WORDS = ((L6_NUM_FGROUP * L6_C_IN * LK * LK) + PV_M1 - 1) / PV_M1;
  localparam int L7_WGT_WORDS = L7_NUM_FGROUP * L7_NUM_CGROUP * LK * LK;
  localparam int L8_WGT_WORDS = L8_NUM_FGROUP * L8_NUM_CGROUP * LK * LK;

  localparam int L0_WGT_DDR_WORDS = L0_WGT_WORDS * WGT_SUBWORDS;
  localparam int L1_WGT_DDR_WORDS = L1_WGT_WORDS * WGT_SUBWORDS;
  localparam int L2_WGT_DDR_WORDS = L2_WGT_WORDS * WGT_SUBWORDS;
  localparam int L3_WGT_DDR_WORDS = L3_WGT_WORDS * WGT_SUBWORDS;
  localparam int L4_WGT_DDR_WORDS = L4_WGT_WORDS * WGT_SUBWORDS;
  localparam int L5_WGT_DDR_WORDS = L5_WGT_WORDS * WGT_SUBWORDS;
  localparam int L6_WGT_DDR_WORDS = L6_WGT_WORDS * WGT_SUBWORDS;
  localparam int L7_WGT_DDR_WORDS = L7_WGT_WORDS * WGT_SUBWORDS;
  localparam int L8_WGT_DDR_WORDS = L8_WGT_WORDS * WGT_SUBWORDS;

  localparam int L0_WGT_DDR_BASE = TB_WGT_BASE;
  localparam int L1_WGT_DDR_BASE = L0_WGT_DDR_BASE + L0_WGT_DDR_WORDS;
  localparam int L2_WGT_DDR_BASE = L1_WGT_DDR_BASE + L1_WGT_DDR_WORDS;
  localparam int L3_WGT_DDR_BASE = L2_WGT_DDR_BASE + L2_WGT_DDR_WORDS;
  localparam int L4_WGT_DDR_BASE = L3_WGT_DDR_BASE + L3_WGT_DDR_WORDS;
  localparam int L5_WGT_DDR_BASE = L4_WGT_DDR_BASE + L4_WGT_DDR_WORDS;
  localparam int L6_WGT_DDR_BASE = L5_WGT_DDR_BASE + L5_WGT_DDR_WORDS;
  localparam int L7_WGT_DDR_BASE = L6_WGT_DDR_BASE + L6_WGT_DDR_WORDS;
  localparam int L8_WGT_DDR_BASE = L7_WGT_DDR_BASE + L7_WGT_DDR_WORDS;
  localparam int WGT_DDR_END = L8_WGT_DDR_BASE + L8_WGT_DDR_WORDS;

  localparam int L0_COLBLKS = (L0_W_IN + PV_M1 - 1) / PV_M1;
  localparam int L8_STORED_GROUPS = (L8_W_OUT + PC - 1) / PC;
  localparam int EXPECTED_IFM_DDR_READS = L0_C_IN * L0_H_IN * L0_COLBLKS;
  localparam int EXPECTED_WGT_DDR_READS = L0_WGT_DDR_WORDS + L1_WGT_DDR_WORDS + L2_WGT_DDR_WORDS + L3_WGT_DDR_WORDS + L4_WGT_DDR_WORDS + L5_WGT_DDR_WORDS + L6_WGT_DDR_WORDS + L7_WGT_DDR_WORDS + L8_WGT_DDR_WORDS;
  localparam int EXPECTED_OFM_DDR_WORDS = L8_F_OUT * L8_H_OUT * L8_STORED_GROUPS;
  localparam int EXPECTED_FINAL_ELEMENTS = L8_F_OUT * L8_H_OUT * L8_W_OUT;

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

  integer i, b;
  integer cycle_count;
  integer ddr_ifm_read_count;
  integer ddr_wgt_read_count;
  integer ddr_ofm_write_count;
  integer ofm_ifm_stream_start_count;
  integer ofm_ifm_stream_done_count;
  integer legacy_m2_mgr_req_count;
  logic done_seen, error_seen;
  logic [3:0] first_error_vec;
  logic [$clog2(CFG_DEPTH)-1:0] first_error_layer;
  logic first_error_mode;
  integer first_error_cycle;

  cnn_top #(
    .DATA_W(DATA_W),
    .PSUM_W(PSUM_W),
    .PTOTAL(PTOTAL),
    .PV_MAX(PV_MAX),
    .PF_MAX(PF_MAX),
    .PC_MODE2(PC),
    .PF_MODE2(PF),
    .C_MAX(C_MAX),
    .F_MAX(F_MAX),
    .W_MAX(W_MAX),
    .H_MAX(H_MAX),
    .HT(HT),
    .K_MAX(K_MAX),
    .WGT_DEPTH(WGT_DEPTH),
    .OFM_BANK_DEPTH(OFM_BANK_DEPTH),
    .OFM_LINEAR_DEPTH(OFM_LINEAR_DEPTH),
    .CFG_DEPTH(CFG_DEPTH),
    .DDR_ADDR_W(DDR_ADDR_W),
    .DDR_WORD_W(DDR_WORD_W)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .abort(abort),
    .cfg_wr_en(cfg_wr_en),
    .cfg_wr_addr(cfg_wr_addr),
    .cfg_wr_data(cfg_wr_data),
    .cfg_num_layers(cfg_num_layers),
    .ddr_rd_req(ddr_rd_req),
    .ddr_rd_addr(ddr_rd_addr),
    .ddr_rd_valid(ddr_rd_valid),
    .ddr_rd_data(ddr_rd_data),
    .ddr_wr_en(ddr_wr_en),
    .ddr_wr_addr(ddr_wr_addr),
    .ddr_wr_data(ddr_wr_data),
    .ddr_wr_be(ddr_wr_be),
    .m1_free_col_blk_g(m1_free_col_blk_g),
    .m1_free_ch_blk_g(m1_free_ch_blk_g),
    .m1_sm_refill_req_ready(m1_sm_refill_req_ready),
    .m1_sm_refill_req_valid(m1_sm_refill_req_valid),
    .m1_sm_refill_row_slot_l(m1_sm_refill_row_slot_l),
    .m1_sm_refill_row_g(m1_sm_refill_row_g),
    .m1_sm_refill_col_blk_g(m1_sm_refill_col_blk_g),
    .m1_sm_refill_ch_blk_g(m1_sm_refill_ch_blk_g),
    .m2_sm_refill_req_ready(m2_sm_refill_req_ready),
    .m2_sm_refill_req_valid(m2_sm_refill_req_valid),
    .m2_sm_refill_row_g(m2_sm_refill_row_g),
    .m2_sm_refill_col_g(m2_sm_refill_col_g),
    .m2_sm_refill_col_l(m2_sm_refill_col_l),
    .m2_sm_refill_cgrp_g(m2_sm_refill_cgrp_g),
    .ifm_m1_free_valid(ifm_m1_free_valid),
    .ifm_m1_free_row_slot_l(ifm_m1_free_row_slot_l),
    .ifm_m1_free_row_g(ifm_m1_free_row_g),
    .busy(busy),
    .done(done),
    .error(error),
    .dbg_layer_idx(dbg_layer_idx),
    .dbg_mode(dbg_mode),
    .dbg_weight_bank(dbg_weight_bank),
    .dbg_error_vec(dbg_error_vec)
  );

  initial clk = 1'b0;
  always #(CLK_PERIOD_NS/2) clk = ~clk;

always_ff @(posedge clk) begin : DBG_M1_L6_OFM_ROW_MON
    if (rst_n &&
        dut.u_ofm_buffer.m1_wr_en &&
        (dbg_layer_idx == 6) &&
        (dut.u_ofm_buffer.m1_wr_filter_base == 0) &&
        (dut.u_ofm_buffer.m1_wr_col_base == 0) &&
        (dut.u_ofm_buffer.m1_wr_row < 12)) begin

        $display("DBG_M1_L6_OFM_ROW t=%0t row=%0d col=%0d fbase=%0d count=%0d d0=%0d d1=%0d d2=%0d d3=%0d d4=%0d d5=%0d d6=%0d d7=%0d",
            $time,
            dut.u_ofm_buffer.m1_wr_row,
            dut.u_ofm_buffer.m1_wr_col_base,
            dut.u_ofm_buffer.m1_wr_filter_base,
            dut.u_ofm_buffer.m1_wr_count,
            $signed(dut.u_ofm_buffer.m1_wr_data[0]),
            $signed(dut.u_ofm_buffer.m1_wr_data[1]),
            $signed(dut.u_ofm_buffer.m1_wr_data[2]),
            $signed(dut.u_ofm_buffer.m1_wr_data[3]),
            $signed(dut.u_ofm_buffer.m1_wr_data[4]),
            $signed(dut.u_ofm_buffer.m1_wr_data[5]),
            $signed(dut.u_ofm_buffer.m1_wr_data[6]),
            $signed(dut.u_ofm_buffer.m1_wr_data[7])
        );
    end
end

always_ff @(posedge clk) begin : DBG_M1_L5_OFM_ROW_MON
    if (rst_n &&
        dut.u_ofm_buffer.m1_wr_en &&
        (dbg_layer_idx == 5) &&
        (dut.u_ofm_buffer.m1_wr_filter_base == 0) &&
        (dut.u_ofm_buffer.m1_wr_col_base == 0) &&
        (dut.u_ofm_buffer.m1_wr_row < 16'd24)) begin

        $display("DBG_M1_L5_OFM_ROW t=%0t row=%0d col=%0d fbase=%0d count=%0d d0=%0d d1=%0d d2=%0d d3=%0d d4=%0d d5=%0d d6=%0d d7=%0d",
            $time,
            dut.u_ofm_buffer.m1_wr_row,
            dut.u_ofm_buffer.m1_wr_col_base,
            dut.u_ofm_buffer.m1_wr_filter_base,
            dut.u_ofm_buffer.m1_wr_count,
            $signed(dut.u_ofm_buffer.m1_wr_data[0]),
            $signed(dut.u_ofm_buffer.m1_wr_data[1]),
            $signed(dut.u_ofm_buffer.m1_wr_data[2]),
            $signed(dut.u_ofm_buffer.m1_wr_data[3]),
            $signed(dut.u_ofm_buffer.m1_wr_data[4]),
            $signed(dut.u_ofm_buffer.m1_wr_data[5]),
            $signed(dut.u_ofm_buffer.m1_wr_data[6]),
            $signed(dut.u_ofm_buffer.m1_wr_data[7])
        );
    end
end


always_ff @(posedge clk) begin : DBG_M1_NOPOOL_VS_REF_FIXED
    int ly;
    int row_i;
    int col_i;
    int fbase_i;

    int g_f0, g_f1, g_f2, g_f3, g_f4, g_f5, g_f6, g_f7;
    int e_f0, e_f1, e_f2, e_f3, e_f4, e_f5, e_f6, e_f7;

    int g_x0, g_x1, g_x2, g_x3, g_x4, g_x5, g_x6, g_x7;
    int e_x0, e_x1, e_x2, e_x3, e_x4, e_x5, e_x6, e_x7;

    int err_f;
    int err_x;

    if (rst_n &&
        dut.u_ofm_buffer.m1_wr_en &&
        (dbg_layer_idx <= 6) &&
        !tb_is_pool_layer_dbg(dbg_layer_idx) &&
        (dut.u_ofm_buffer.m1_wr_filter_base == 0) &&
        (dut.u_ofm_buffer.m1_wr_col_base == 0) &&
        (dut.u_ofm_buffer.m1_wr_row < 32)) begin

        ly      = dbg_layer_idx;
        row_i   = dut.u_ofm_buffer.m1_wr_row;
        col_i   = dut.u_ofm_buffer.m1_wr_col_base;
        fbase_i = dut.u_ofm_buffer.m1_wr_filter_base;

        // View A: same x=0, filters f0..f7. lane = pv*PF + pf, pv=0.
        g_f0 = $signed(dut.u_ofm_buffer.m1_wr_data[0]);
        g_f1 = $signed(dut.u_ofm_buffer.m1_wr_data[1]);
        g_f2 = $signed(dut.u_ofm_buffer.m1_wr_data[2]);
        g_f3 = $signed(dut.u_ofm_buffer.m1_wr_data[3]);
        g_f4 = $signed(dut.u_ofm_buffer.m1_wr_data[4]);
        g_f5 = $signed(dut.u_ofm_buffer.m1_wr_data[5]);
        g_f6 = $signed(dut.u_ofm_buffer.m1_wr_data[6]);
        g_f7 = $signed(dut.u_ofm_buffer.m1_wr_data[7]);

        e_f0 = tb_ref_byte_dbg(ly, row_i, col_i, fbase_i + 0);
        e_f1 = tb_ref_byte_dbg(ly, row_i, col_i, fbase_i + 1);
        e_f2 = tb_ref_byte_dbg(ly, row_i, col_i, fbase_i + 2);
        e_f3 = tb_ref_byte_dbg(ly, row_i, col_i, fbase_i + 3);
        e_f4 = tb_ref_byte_dbg(ly, row_i, col_i, fbase_i + 4);
        e_f5 = tb_ref_byte_dbg(ly, row_i, col_i, fbase_i + 5);
        e_f6 = tb_ref_byte_dbg(ly, row_i, col_i, fbase_i + 6);
        e_f7 = tb_ref_byte_dbg(ly, row_i, col_i, fbase_i + 7);

        // View B: same f0, x0..x7. lane = pv*PF + pf, pf=0, PF=16.
        g_x0 = $signed(dut.u_ofm_buffer.m1_wr_data[0]);
        g_x1 = $signed(dut.u_ofm_buffer.m1_wr_data[16]);
        g_x2 = $signed(dut.u_ofm_buffer.m1_wr_data[32]);
        g_x3 = $signed(dut.u_ofm_buffer.m1_wr_data[48]);
        g_x4 = $signed(dut.u_ofm_buffer.m1_wr_data[64]);
        g_x5 = $signed(dut.u_ofm_buffer.m1_wr_data[80]);
        g_x6 = $signed(dut.u_ofm_buffer.m1_wr_data[96]);
        g_x7 = $signed(dut.u_ofm_buffer.m1_wr_data[112]);

        e_x0 = tb_ref_byte_dbg(ly, row_i, col_i + 0, fbase_i);
        e_x1 = tb_ref_byte_dbg(ly, row_i, col_i + 1, fbase_i);
        e_x2 = tb_ref_byte_dbg(ly, row_i, col_i + 2, fbase_i);
        e_x3 = tb_ref_byte_dbg(ly, row_i, col_i + 3, fbase_i);
        e_x4 = tb_ref_byte_dbg(ly, row_i, col_i + 4, fbase_i);
        e_x5 = tb_ref_byte_dbg(ly, row_i, col_i + 5, fbase_i);
        e_x6 = tb_ref_byte_dbg(ly, row_i, col_i + 6, fbase_i);
        e_x7 = tb_ref_byte_dbg(ly, row_i, col_i + 7, fbase_i);

        err_f = 0;
        if (g_f0 != e_f0) err_f = err_f | 1;
        if (g_f1 != e_f1) err_f = err_f | 2;
        if (g_f2 != e_f2) err_f = err_f | 4;
        if (g_f3 != e_f3) err_f = err_f | 8;
        if (g_f4 != e_f4) err_f = err_f | 16;
        if (g_f5 != e_f5) err_f = err_f | 32;
        if (g_f6 != e_f6) err_f = err_f | 64;
        if (g_f7 != e_f7) err_f = err_f | 128;

        err_x = 0;
        if (g_x0 != e_x0) err_x = err_x | 1;
        if (g_x1 != e_x1) err_x = err_x | 2;
        if (g_x2 != e_x2) err_x = err_x | 4;
        if (g_x3 != e_x3) err_x = err_x | 8;
        if (g_x4 != e_x4) err_x = err_x | 16;
        if (g_x5 != e_x5) err_x = err_x | 32;
        if (g_x6 != e_x6) err_x = err_x | 64;
        if (g_x7 != e_x7) err_x = err_x | 128;

        $display("DBG_M1_NOPOOL_VS_REF t=%0t layer=%0d row=%0d col=%0d fbase=%0d err_f=0x%0h err_x=0x%0h filt_got=%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d filt_exp=%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d x_got=%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d x_exp=%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
            $time, ly, row_i, col_i, fbase_i, err_f, err_x,
            g_f0,g_f1,g_f2,g_f3,g_f4,g_f5,g_f6,g_f7,
            e_f0,e_f1,e_f2,e_f3,e_f4,e_f5,e_f6,e_f7,
            g_x0,g_x1,g_x2,g_x3,g_x4,g_x5,g_x6,g_x7,
            e_x0,e_x1,e_x2,e_x3,e_x4,e_x5,e_x6,e_x7);

        if ((err_f != 0) || (err_x != 0)) begin
            $display("DBG_M1_NOPOOL_ERR t=%0t FIRST_BAD_CANDIDATE layer=%0d row=%0d col=%0d fbase=%0d err_f=0x%0h err_x=0x%0h",
                $time, ly, row_i, col_i, fbase_i, err_f, err_x);
        end
    end
end

always_ff @(posedge clk) begin : DBG_M1_POOL_IN_VS_REF_FIXED
    int ly;
    int row_i;
    int col_i;
    int fbase_i;

    int g0,g1,g2,g3,g4,g5,g6,g7;
    int e0,e1,e2,e3,e4,e5,e6,e7;
    int err_x;

    if (rst_n &&
        tb_is_pool_layer_dbg(dbg_layer_idx) &&
        dut.u_mode1_compute_top.u_pooling_mode1.in_valid &&
        (dut.u_mode1_compute_top.u_pooling_mode1.f_group == 0) &&
        (dut.u_mode1_compute_top.u_pooling_mode1.in_col == 0) &&
        (dut.u_mode1_compute_top.u_pooling_mode1.in_row < 32)) begin

        ly      = dbg_layer_idx;
        row_i   = dut.u_mode1_compute_top.u_pooling_mode1.in_row;
        col_i   = dut.u_mode1_compute_top.u_pooling_mode1.in_col;
        fbase_i = 0;

        g0 = $signed(dut.u_mode1_compute_top.u_pooling_mode1.in_data[0]);
        g1 = $signed(dut.u_mode1_compute_top.u_pooling_mode1.in_data[16]);
        g2 = $signed(dut.u_mode1_compute_top.u_pooling_mode1.in_data[32]);
        g3 = $signed(dut.u_mode1_compute_top.u_pooling_mode1.in_data[48]);
        g4 = $signed(dut.u_mode1_compute_top.u_pooling_mode1.in_data[64]);
        g5 = $signed(dut.u_mode1_compute_top.u_pooling_mode1.in_data[80]);
        g6 = $signed(dut.u_mode1_compute_top.u_pooling_mode1.in_data[96]);
        g7 = $signed(dut.u_mode1_compute_top.u_pooling_mode1.in_data[112]);

        e0 = tb_ref_byte_dbg(ly - 1, row_i, col_i + 0, fbase_i);
        e1 = tb_ref_byte_dbg(ly - 1, row_i, col_i + 1, fbase_i);
        e2 = tb_ref_byte_dbg(ly - 1, row_i, col_i + 2, fbase_i);
        e3 = tb_ref_byte_dbg(ly - 1, row_i, col_i + 3, fbase_i);
        e4 = tb_ref_byte_dbg(ly - 1, row_i, col_i + 4, fbase_i);
        e5 = tb_ref_byte_dbg(ly - 1, row_i, col_i + 5, fbase_i);
        e6 = tb_ref_byte_dbg(ly - 1, row_i, col_i + 6, fbase_i);
        e7 = tb_ref_byte_dbg(ly - 1, row_i, col_i + 7, fbase_i);

        err_x = 0;
        if (g0 != e0) err_x = err_x | 1;
        if (g1 != e1) err_x = err_x | 2;
        if (g2 != e2) err_x = err_x | 4;
        if (g3 != e3) err_x = err_x | 8;
        if (g4 != e4) err_x = err_x | 16;
        if (g5 != e5) err_x = err_x | 32;
        if (g6 != e6) err_x = err_x | 64;
        if (g7 != e7) err_x = err_x | 128;

        $display("DBG_M1_POOL_IN_VS_REF t=%0t layer=%0d in_row=%0d in_col=%0d fbase=%0d err_x=0x%0h got=%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d exp_prev=%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
            $time, ly, row_i, col_i, fbase_i, err_x,
            g0,g1,g2,g3,g4,g5,g6,g7,
            e0,e1,e2,e3,e4,e5,e6,e7);

        if (err_x != 0) begin
            $display("DBG_M1_POOL_IN_ERR t=%0t FIRST_BAD_CANDIDATE layer=%0d in_row=%0d in_col=%0d fbase=%0d err_x=0x%0h",
                $time, ly, row_i, col_i, fbase_i, err_x);
        end
    end
end

always_ff @(posedge clk) begin : DBG_M1_POOL_OUT_VS_REF_FIXED
    int ly;
    int row_i;
    int col_i;
    int g0,g1,g2,g3,g4,g5,g6,g7;
    int e0,e1,e2,e3,e4,e5,e6,e7;
    int err_x;

    if (rst_n &&
        dut.u_ofm_buffer.m1_wr_en &&
        tb_is_pool_layer_dbg(dbg_layer_idx) &&
        (dut.u_ofm_buffer.m1_wr_filter_base == 0) &&
        (dut.u_ofm_buffer.m1_wr_col_base == 0) &&
        (dut.u_ofm_buffer.m1_wr_row < 32)) begin

        ly    = dbg_layer_idx;
        row_i = dut.u_ofm_buffer.m1_wr_row;
        col_i = dut.u_ofm_buffer.m1_wr_col_base;

        g0 = $signed(dut.u_ofm_buffer.m1_wr_data[0]);
        g1 = $signed(dut.u_ofm_buffer.m1_wr_data[1]);
        g2 = $signed(dut.u_ofm_buffer.m1_wr_data[2]);
        g3 = $signed(dut.u_ofm_buffer.m1_wr_data[3]);
        g4 = $signed(dut.u_ofm_buffer.m1_wr_data[4]);
        g5 = $signed(dut.u_ofm_buffer.m1_wr_data[5]);
        g6 = $signed(dut.u_ofm_buffer.m1_wr_data[6]);
        g7 = $signed(dut.u_ofm_buffer.m1_wr_data[7]);

        e0 = tb_ref_byte_dbg(ly, row_i, col_i + 0, 0);
        e1 = tb_ref_byte_dbg(ly, row_i, col_i + 1, 0);
        e2 = tb_ref_byte_dbg(ly, row_i, col_i + 2, 0);
        e3 = tb_ref_byte_dbg(ly, row_i, col_i + 3, 0);
        e4 = tb_ref_byte_dbg(ly, row_i, col_i + 4, 0);
        e5 = tb_ref_byte_dbg(ly, row_i, col_i + 5, 0);
        e6 = tb_ref_byte_dbg(ly, row_i, col_i + 6, 0);
        e7 = tb_ref_byte_dbg(ly, row_i, col_i + 7, 0);

        err_x = 0;
        if (g0 != e0) err_x = err_x | 1;
        if (g1 != e1) err_x = err_x | 2;
        if (g2 != e2) err_x = err_x | 4;
        if (g3 != e3) err_x = err_x | 8;
        if (g4 != e4) err_x = err_x | 16;
        if (g5 != e5) err_x = err_x | 32;
        if (g6 != e6) err_x = err_x | 64;
        if (g7 != e7) err_x = err_x | 128;

        $display("DBG_M1_POOL_OUT_VS_REF t=%0t layer=%0d row=%0d col=%0d err_x=0x%0h got=%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d exp=%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
            $time, ly, row_i, col_i, err_x,
            g0,g1,g2,g3,g4,g5,g6,g7,
            e0,e1,e2,e3,e4,e5,e6,e7);

        if (err_x != 0) begin
            $display("DBG_M1_POOL_OUT_ERR t=%0t FIRST_BAD_CANDIDATE layer=%0d row=%0d col=%0d err_x=0x%0h",
                $time, ly, row_i, col_i, err_x);
        end
    end
end


always_ff @(posedge clk) begin : DBG_M1_L6_POOL_IN_MON
    if (rst_n &&
        (dbg_layer_idx == 6) &&
        dut.u_mode1_compute_top.u_pooling_mode1.in_valid &&
        (dut.u_mode1_compute_top.u_pooling_mode1.f_group == 0) &&
        (dut.u_mode1_compute_top.u_pooling_mode1.in_col == 0) &&
        (dut.u_mode1_compute_top.u_pooling_mode1.in_row < 16'd24)) begin

        $display("DBG_M1_L6_POOL_IN t=%0t layer=%0d in_row=%0d in_col=%0d f_group=%0d fbase=%0d d0=%0d d1=%0d d2=%0d d3=%0d d4=%0d d5=%0d d6=%0d d7=%0d",
            $time,
            dbg_layer_idx,
            dut.u_mode1_compute_top.u_pooling_mode1.in_row,
            dut.u_mode1_compute_top.u_pooling_mode1.in_col,
            dut.u_mode1_compute_top.u_pooling_mode1.f_group,
            dut.u_mode1_compute_top.u_pooling_mode1.f_group * dut.u_mode1_compute_top.u_pooling_mode1.Pf_cur,
            $signed(dut.u_mode1_compute_top.u_pooling_mode1.in_data[0]),
            $signed(dut.u_mode1_compute_top.u_pooling_mode1.in_data[1]),
            $signed(dut.u_mode1_compute_top.u_pooling_mode1.in_data[2]),
            $signed(dut.u_mode1_compute_top.u_pooling_mode1.in_data[3]),
            $signed(dut.u_mode1_compute_top.u_pooling_mode1.in_data[4]),
            $signed(dut.u_mode1_compute_top.u_pooling_mode1.in_data[5]),
            $signed(dut.u_mode1_compute_top.u_pooling_mode1.in_data[6]),
            $signed(dut.u_mode1_compute_top.u_pooling_mode1.in_data[7])
        );
    end
end


  function automatic logic [DATA_W-1:0] sat8(input int val);
    begin
      if (val < 0) sat8 = '0;
      else if (val > 127) sat8 = 8'd127;
      else sat8 = val[DATA_W-1:0];
    end
  endfunction

  function automatic int input_pattern_val(input int row, input int col, input int ch);
    begin
      input_pattern_val = 1 + (row % 16) + 2 * (col % 16) + 3 * (ch % 3);
    end
  endfunction

  function automatic bit layer_is_mode2(input int layer_id);
    begin
      layer_is_mode2 = (layer_id >= 7);
    end
  endfunction

  function automatic int layer_c_in(input int layer_id);
    begin
      case (layer_id)
        0: layer_c_in = L0_C_IN;
        1: layer_c_in = L1_C_IN;
        2: layer_c_in = L2_C_IN;
        3: layer_c_in = L3_C_IN;
        4: layer_c_in = L4_C_IN;
        5: layer_c_in = L5_C_IN;
        6: layer_c_in = L6_C_IN;
        7: layer_c_in = L7_C_IN;
        default: layer_c_in = L8_C_IN;
      endcase
    end
  endfunction

  function automatic int layer_f_out(input int layer_id);
    begin
      case (layer_id)
        0: layer_f_out = L0_F_OUT;
        1: layer_f_out = L1_F_OUT;
        2: layer_f_out = L2_F_OUT;
        3: layer_f_out = L3_F_OUT;
        4: layer_f_out = L4_F_OUT;
        5: layer_f_out = L5_F_OUT;
        6: layer_f_out = L6_F_OUT;
        7: layer_f_out = L7_F_OUT;
        default: layer_f_out = L8_F_OUT;
      endcase
    end
  endfunction

// ============================================================
// Debug golden helper for VGG16-like 96x96 9-layer
// L0-L8, center-tap weights, pool at L1/L3/L6
// ============================================================

function automatic bit tb_is_pool_layer_dbg(input int layer_id);
begin
    case (layer_id)
        1, 3, 6: tb_is_pool_layer_dbg = 1'b1;
        default: tb_is_pool_layer_dbg = 1'b0;
    endcase
end
endfunction

function automatic int tb_layer_c_in_dbg(input int layer_id);
begin
    case (layer_id)
        0: tb_layer_c_in_dbg = 3;
        1: tb_layer_c_in_dbg = 16;
        2: tb_layer_c_in_dbg = 16;
        3: tb_layer_c_in_dbg = 32;
        4: tb_layer_c_in_dbg = 32;
        5: tb_layer_c_in_dbg = 64;
        6: tb_layer_c_in_dbg = 64;
        7: tb_layer_c_in_dbg = 64;
        8: tb_layer_c_in_dbg = 128;
        default: tb_layer_c_in_dbg = 1;
    endcase
end
endfunction

function automatic int tb_input_pattern_dbg(
    input int row,
    input int col,
    input int ch
);
begin
    tb_input_pattern_dbg = 1 + (row % 16) + 2 * (col % 16) + 3 * (ch % 3);
end
endfunction

function automatic int tb_sat8_dbg(input int v);
begin
    if (v < 0)
        tb_sat8_dbg = 0;
    else if (v > 255)
        tb_sat8_dbg = 255;
    else
        tb_sat8_dbg = v;
end
endfunction

function automatic int tb_ref_layer_value_dbg(
    input int layer_id,
    input int row,
    input int col,
    input int ch
);
    int cin_ch;
    int v00;
    int v01;
    int v10;
    int v11;
    int max0;
    int max1;
begin
    if (layer_id < 0) begin
        tb_ref_layer_value_dbg = tb_input_pattern_dbg(row, col, ch);
    end
    else begin
        cin_ch = ch % tb_layer_c_in_dbg(layer_id);

        if (tb_is_pool_layer_dbg(layer_id)) begin
            v00 = tb_ref_layer_value_dbg(layer_id - 1, row * 2 + 0, col * 2 + 0, cin_ch);
            v01 = tb_ref_layer_value_dbg(layer_id - 1, row * 2 + 0, col * 2 + 1, cin_ch);
            v10 = tb_ref_layer_value_dbg(layer_id - 1, row * 2 + 1, col * 2 + 0, cin_ch);
            v11 = tb_ref_layer_value_dbg(layer_id - 1, row * 2 + 1, col * 2 + 1, cin_ch);

            max0 = (v00 > v01) ? v00 : v01;
            max1 = (v10 > v11) ? v10 : v11;

            tb_ref_layer_value_dbg = (max0 > max1) ? max0 : max1;
        end
        else begin
            tb_ref_layer_value_dbg = tb_ref_layer_value_dbg(layer_id - 1, row, col, cin_ch);
        end
    end
end
endfunction

function automatic int tb_ref_byte_dbg(
    input int layer_id,
    input int row,
    input int col,
    input int ch
);
begin
    tb_ref_byte_dbg = tb_sat8_dbg(tb_ref_layer_value_dbg(layer_id, row, col, ch));
end
endfunction

  function automatic logic [DDR_WORD_W-1:0] pack_m1_ifm_word(input int ch, input int row, input int col_blk);
    logic [DDR_WORD_W-1:0] word;
    int global_col;
    begin
      word = '0;
      for (int lane = 0; lane < PV_M1; lane++) begin
        global_col = col_blk * PV_M1 + lane;
        if ((ch < L0_C_IN) && (row < L0_H_IN) && (global_col < L0_W_IN)) begin
          word[lane*DATA_W +: DATA_W] = sat8(input_pattern_val(row, global_col, ch));
        end
      end
      return word;
    end
  endfunction

  function automatic bit layer_pool_en(input int layer_id);
    begin
      case (layer_id)
        1, 3, 6: layer_pool_en = 1'b1;
        default: layer_pool_en = 1'b0;
      endcase
    end
  endfunction

  function automatic int ref_layer_value(input int layer_id, input int row, input int col, input int ch);
    int cin_ch;
    int v00;
    int v01;
    int v10;
    int v11;
    int max0;
    int max1;
    begin
      if (layer_id < 0) begin
        ref_layer_value = input_pattern_val(row, col, ch);
      end else begin
        cin_ch = ch % layer_c_in(layer_id);

        if (layer_pool_en(layer_id)) begin
          v00 = ref_layer_value(layer_id - 1, row*2 + 0, col*2 + 0, cin_ch);
          v01 = ref_layer_value(layer_id - 1, row*2 + 0, col*2 + 1, cin_ch);
          v10 = ref_layer_value(layer_id - 1, row*2 + 1, col*2 + 0, cin_ch);
          v11 = ref_layer_value(layer_id - 1, row*2 + 1, col*2 + 1, cin_ch);
          max0 = (v00 > v01) ? v00 : v01;
          max1 = (v10 > v11) ? v10 : v11;
          ref_layer_value = (max0 > max1) ? max0 : max1;
        end else begin
          ref_layer_value = ref_layer_value(layer_id - 1, row, col, cin_ch);
        end
      end
    end
  endfunction

  function automatic logic [DATA_W-1:0] expected_final_byte(input int row, input int col, input int out_ch);
    begin
      expected_final_byte = sat8(ref_layer_value(8, row, col, out_ch));
    end
  endfunction

  function automatic logic signed [DATA_W-1:0] sparse_weight_value(input int layer_id, input int phys_word, input int subword, input int lane_in_ddr);
    int C;
    int F;
    int lane;
    int num_cgrp;
    int bundle;
    int tmp;
    int fgroup;
    int cidx;
    int cgrp;
    int ky;
    int kx;
    int pf;
    int pc_l;
    int fout;
    int cin;
    begin
      C = layer_c_in(layer_id);
      F = layer_f_out(layer_id);
      lane = subword * DDR_LANES + lane_in_ddr;
      sparse_weight_value = '0;

      if (!layer_is_mode2(layer_id)) begin
        bundle = phys_word * PV_M1 + (lane / PF_M1);
        pf = lane % PF_M1;
        tmp = bundle;
        kx = tmp % LK; tmp = tmp / LK;
        ky = tmp % LK; tmp = tmp / LK;
        cidx = tmp % C; tmp = tmp / C;
        fgroup = tmp;
        fout = fgroup * PF_M1 + pf;
        if ((fout < F) && (ky == 1) && (kx == 1) && (cidx == (fout % C))) begin
          sparse_weight_value = 8'sd1;
        end
      end else begin
        num_cgrp = (C + PC - 1) / PC;
        tmp = phys_word;
        kx = tmp % LK; tmp = tmp / LK;
        ky = tmp % LK; tmp = tmp / LK;
        cgrp = tmp % num_cgrp; tmp = tmp / num_cgrp;
        fgroup = tmp;
        pf = lane / PC;
        pc_l = lane % PC;
        fout = fgroup * PF + pf;
        cin = cgrp * PC + pc_l;
        if ((fout < F) && (cin < C) && (ky == 1) && (kx == 1) && (cin == (fout % C))) begin
          sparse_weight_value = 8'sd1;
        end
      end
    end
  endfunction

  function automatic logic [DDR_WORD_W-1:0] pack_weight_ddr_word(input int layer_id, input int phys_word, input int subword);
    logic [DDR_WORD_W-1:0] word;
    begin
      word = '0;
      for (int lane = 0; lane < DDR_LANES; lane++) begin
        word[lane*DATA_W +: DATA_W] = sparse_weight_value(layer_id, phys_word, subword, lane);
      end
      return word;
    end
  endfunction

  function automatic logic [DDR_WORD_W-1:0] expected_final_word(input int word_idx);
    logic [DDR_WORD_W-1:0] word;
    int words_per_ch;
    int ch;
    int rem;
    int row;
    int grp;
    int col_g;
    begin
      word = '0;
      words_per_ch = L8_H_OUT * L8_STORED_GROUPS;
      ch = word_idx / words_per_ch;
      rem = word_idx % words_per_ch;
      row = rem / L8_STORED_GROUPS;
      grp = rem % L8_STORED_GROUPS;
      for (int lane = 0; lane < DDR_LANES; lane++) begin
        col_g = grp * PC + lane;
        if ((ch < L8_F_OUT) && (row < L8_H_OUT) && (lane < PC) && (col_g < L8_W_OUT)) begin
          word[lane*DATA_W +: DATA_W] = expected_final_byte(row, col_g, ch);
        end
      end
      return word;
    end
  endfunction

  task automatic init_mem;
    int row;
    int col_blk;
    int addr;
    begin
      for (i = 0; i < MEM_DEPTH; i = i + 1) ddr_mem[i] = '0;

      for (int ch = 0; ch < L0_C_IN; ch++) begin
        for (row = 0; row < L0_H_IN; row++) begin
          for (col_blk = 0; col_blk < L0_COLBLKS; col_blk++) begin
            addr = TB_IFM_BASE + ch * L0_H_IN * L0_COLBLKS + row * L0_COLBLKS + col_blk;
            ddr_mem[addr] = pack_m1_ifm_word(ch, row, col_blk);
          end
        end
      end

      for (i = 0; i < L0_WGT_DDR_WORDS; i++) ddr_mem[L0_WGT_DDR_BASE + i] = pack_weight_ddr_word(0, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (i = 0; i < L1_WGT_DDR_WORDS; i++) ddr_mem[L1_WGT_DDR_BASE + i] = pack_weight_ddr_word(1, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (i = 0; i < L2_WGT_DDR_WORDS; i++) ddr_mem[L2_WGT_DDR_BASE + i] = pack_weight_ddr_word(2, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (i = 0; i < L3_WGT_DDR_WORDS; i++) ddr_mem[L3_WGT_DDR_BASE + i] = pack_weight_ddr_word(3, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (i = 0; i < L4_WGT_DDR_WORDS; i++) ddr_mem[L4_WGT_DDR_BASE + i] = pack_weight_ddr_word(4, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (i = 0; i < L5_WGT_DDR_WORDS; i++) ddr_mem[L5_WGT_DDR_BASE + i] = pack_weight_ddr_word(5, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (i = 0; i < L6_WGT_DDR_WORDS; i++) ddr_mem[L6_WGT_DDR_BASE + i] = pack_weight_ddr_word(6, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (i = 0; i < L7_WGT_DDR_WORDS; i++) ddr_mem[L7_WGT_DDR_BASE + i] = pack_weight_ddr_word(7, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (i = 0; i < L8_WGT_DDR_WORDS; i++) ddr_mem[L8_WGT_DDR_BASE + i] = pack_weight_ddr_word(8, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
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
    input int h_conv_out,
    input int w_conv_out,
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
      cfg.k = LK;
      cfg.h_out = h_conv_out;
      cfg.w_out = w_conv_out;
      cfg.pv_m1 = PV_M1;
      cfg.pf_m1 = PF_M1;
      cfg.pc_m2 = PC;
      cfg.pf_m2 = PF;
      cfg.conv_stride = 1;
      cfg.pad_top = PAD;
      cfg.pad_bottom = PAD;
      cfg.pad_left = PAD;
      cfg.pad_right = PAD;
      cfg.relu_en = 1'b1;
      cfg.pool_en = pool_en;
      cfg.pool_k = 2;
      cfg.pool_stride = 2;
      cfg.ifm_ddr_base = TB_IFM_BASE;
      cfg.wgt_ddr_base = wgt_base;
      cfg.ofm_ddr_base = TB_OFM_BASE;
      cfg.first_layer = first_layer;
      cfg.last_layer = last_layer;
    end
  endtask

  task automatic program_layers;
    layer_desc_t cfg0, cfg1, cfg2, cfg3, cfg4, cfg5, cfg6, cfg7, cfg8;
    begin
      fill_layer_cfg(cfg0, 0, 1'b0, L0_H_IN,L0_W_IN,L0_C_IN,L0_F_OUT,L0_H_CONV_OUT,L0_W_CONV_OUT,L0_POOL_EN,L0_WGT_DDR_BASE, 1'b1, 1'b0);
      fill_layer_cfg(cfg1, 1, 1'b0, L1_H_IN,L1_W_IN,L1_C_IN,L1_F_OUT,L1_H_CONV_OUT,L1_W_CONV_OUT,L1_POOL_EN,L1_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg2, 2, 1'b0, L2_H_IN,L2_W_IN,L2_C_IN,L2_F_OUT,L2_H_CONV_OUT,L2_W_CONV_OUT,L2_POOL_EN,L2_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg3, 3, 1'b0, L3_H_IN,L3_W_IN,L3_C_IN,L3_F_OUT,L3_H_CONV_OUT,L3_W_CONV_OUT,L3_POOL_EN,L3_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg4, 4, 1'b0, L4_H_IN,L4_W_IN,L4_C_IN,L4_F_OUT,L4_H_CONV_OUT,L4_W_CONV_OUT,L4_POOL_EN,L4_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg5, 5, 1'b0, L5_H_IN,L5_W_IN,L5_C_IN,L5_F_OUT,L5_H_CONV_OUT,L5_W_CONV_OUT,L5_POOL_EN,L5_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg6, 6, 1'b0, L6_H_IN,L6_W_IN,L6_C_IN,L6_F_OUT,L6_H_CONV_OUT,L6_W_CONV_OUT,L6_POOL_EN,L6_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg7, 7, 1'b1, L7_H_IN,L7_W_IN,L7_C_IN,L7_F_OUT,L7_H_CONV_OUT,L7_W_CONV_OUT,L7_POOL_EN,L7_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg8, 8, 1'b1, L8_H_IN,L8_W_IN,L8_C_IN,L8_F_OUT,L8_H_CONV_OUT,L8_W_CONV_OUT,L8_POOL_EN,L8_WGT_DDR_BASE, 1'b0, 1'b1);

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
    begin
      $display("--- OFM DDR region dump ---");
      for (int j = 0; j < 16; j++) begin
        $display("OFM[%0d] @0x%0h = 0x%0h", j, TB_OFM_BASE + j, ddr_mem[TB_OFM_BASE + j]);
      end
    end
  endtask

  task automatic check_final_ofm;
    logic [DDR_WORD_W-1:0] got_word;
    logic [DDR_WORD_W-1:0] exp_word;
    int mismatch;
    int words_per_ch;
    int dec_ch;
    int dec_rem;
    int dec_row;
    int dec_grp;
    begin
      mismatch = 0;
      words_per_ch = L8_H_OUT * L8_STORED_GROUPS;
      for (int j = 0; j < EXPECTED_OFM_DDR_WORDS; j++) begin
        got_word = ddr_mem[TB_OFM_BASE + j];
        exp_word = expected_final_word(j);
        if (got_word !== exp_word) begin
          dec_ch  = j / words_per_ch;
          dec_rem = j % words_per_ch;
          dec_row = dec_rem / L8_STORED_GROUPS;
          dec_grp = dec_rem % L8_STORED_GROUPS;
          if (mismatch < 64) begin
            $display("TB_MISMATCH_FINAL word=%0d ch=%0d row=%0d grp=%0d got=0x%0h exp=0x%0h", j, dec_ch, dec_row, dec_grp, got_word, exp_word);
          end
          mismatch++;
        end
      end

      if (mismatch != 0) begin
        dump_ofm_region();
        $fatal(1, "TB_FAIL: VGG16-like 96x96 9-layer 7M1+2M2 K3/P1 final OFM mismatch count=%0d", mismatch);
      end
    end
  endtask

  task automatic print_banner;
    begin
      $display("TB_INFO: VGG16-like 96x96 first 9 conv layers, 7M1+2M2, K=3 stride=1 pad=1");
      $display("TB_INFO: GOLDEN=recursive_refgolden_v2 OFM_COMPACT row_stride=%0d", OFM_ROW_STRIDE);
      $display("TB_INFO: params PTOTAL=%0d PV_MAX=%0d PF_MAX=%0d PC=%0d PF=%0d C_MAX=%0d F_MAX=%0d HT=%0d", PTOTAL, PV_MAX, PF_MAX, PC, PF, C_MAX, F_MAX, HT);
      $display("TB_INFO: L0-L6 MODE1, L7-L8 MODE2; final expected shape %0dx%0dx%0d", L8_H_OUT, L8_W_OUT, L8_F_OUT);
      $display("TB_INFO: DDR bases IFM=0x%0h WGT=0x%0h OFM=0x%0h WGT_END=0x%0h", TB_IFM_BASE, TB_WGT_BASE, TB_OFM_BASE, WGT_DDR_END);
      $display("TB_INFO: OFM compact storage row_stride=%0d bank_depth=%0d linear_depth=%0d", OFM_ROW_STRIDE, OFM_BANK_DEPTH, OFM_LINEAR_DEPTH);
      $display("TB_INFO: expected reads IFM=%0d WGT=%0d final OFM words=%0d elements=%0d", EXPECTED_IFM_DDR_READS, EXPECTED_WGT_DDR_READS, EXPECTED_OFM_DDR_WORDS, EXPECTED_FINAL_ELEMENTS);
    end
  endtask

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

        if ((ddr_wr_addr >= TB_OFM_BASE) && (ddr_wr_addr < (TB_OFM_BASE + EXPECTED_OFM_DDR_WORDS + 64))) begin
          ddr_ofm_write_count <= ddr_ofm_write_count + 1;
        end
      end

      if (ddr_rd_req) begin
        rd_pending_q <= 1'b1;
        rd_addr_q <= ddr_rd_addr;
        if ((ddr_rd_addr >= TB_IFM_BASE) && (ddr_rd_addr < (TB_IFM_BASE + EXPECTED_IFM_DDR_READS + 64))) begin
          ddr_ifm_read_count <= ddr_ifm_read_count + 1;
        end
        if ((ddr_rd_addr >= TB_WGT_BASE) && (ddr_rd_addr < WGT_DDR_END)) begin
          ddr_wgt_read_count <= ddr_wgt_read_count + 1;
        end
      end else if (rd_pending_q) begin
        rd_pending_q <= 1'b0;
      end

      if (dut.ofm_ifm_stream_start_s) ofm_ifm_stream_start_count <= ofm_ifm_stream_start_count + 1;
      if (dut.ofm_ifm_stream_done_s) ofm_ifm_stream_done_count <= ofm_ifm_stream_done_count + 1;
      if (m2_sm_refill_req_valid && m2_sm_refill_req_ready) legacy_m2_mgr_req_count <= legacy_m2_mgr_req_count + 1;
    end
  end

  always_ff @(posedge clk) begin
    if (rst_n) begin
      if (done || error || ((cycle_count % 100000) == 0)) begin
        $display("DBG_TOP_STATUS t=%0t cycle=%0d busy=%0b done=%0b error=%0b vec=%04b layer=%0d mode=%0d ifm_rd=%0d wgt_rd=%0d ofm_wr=%0d stream_start=%0d stream_done=%0d legacy_m2_mgr_req=%0d", $time, cycle_count, busy, done, error, dbg_error_vec, dbg_layer_idx, dbg_mode, ddr_ifm_read_count, ddr_wgt_read_count, ddr_ofm_write_count, ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, legacy_m2_mgr_req_count);
      end
    end
  end

  initial begin
    rst_n = 1'b0;
    start = 1'b0;
    abort = 1'b0;
    cfg_wr_en = 1'b0;
    cfg_wr_addr = '0;
    cfg_wr_data = '0;
    cfg_num_layers = NUM_LAYERS;
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
      $display("TB_FAIL: DUT asserted error. first_cycle=%0d layer=%0d mode=%0d vec=%04b", first_error_cycle, first_error_layer, first_error_mode, first_error_vec);
      $fatal(1, "TB_FAIL: DUT error");
    end

    if (!done_seen) begin
      $display("TB_FAIL: timeout after %0d cycles. busy=%0b done=%0b error=%0b", cycle_count, busy, done, error);
      dump_ofm_region();
      $fatal(1, "TB_FAIL: timeout");
    end

    $display("TB_INFO: VGG16-like 96x96 9-layer test done after %0d cycles", cycle_count);
    $display("TB_INFO: DDR counts: ifm_reads=%0d est=%0d, wgt_reads=%0d est=%0d, ofm_writes=%0d expected=%0d", ddr_ifm_read_count, EXPECTED_IFM_DDR_READS, ddr_wgt_read_count, EXPECTED_WGT_DDR_READS, ddr_ofm_write_count, EXPECTED_OFM_DDR_WORDS);
    $display("TB_INFO: OFM->IFM stream starts=%0d done=%0d legacy_m2_mgr_req=%0d", ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, legacy_m2_mgr_req_count);

    check_final_ofm();
    $display("TB_PASS: VGG16-like 96x96 9-layer 7M1+2M2 K3/P1 center-tap expected-compare passed");
    $finish;
  end
  
always_ff @(posedge clk) begin : DBG_FAST_L0_M1_NOPOOL_FILTER_CHECK
    int row_i;
    int col_i;
    int fbase_i;

    int g0, g1, g2, g3, g4, g5, g6, g7;
    int e0, e1, e2, e3, e4, e5, e6, e7;
    int err_f;

    if (rst_n &&
        dut.u_ofm_buffer.m1_wr_en &&
        (dbg_layer_idx == 0) &&
        (dut.u_ofm_buffer.m1_wr_filter_base == 0) &&
        (dut.u_ofm_buffer.m1_wr_row < 16'd16) &&
        (dut.u_ofm_buffer.m1_wr_col_base < 16'd16)) begin

        row_i   = dut.u_ofm_buffer.m1_wr_row;
        col_i   = dut.u_ofm_buffer.m1_wr_col_base;
        fbase_i = dut.u_ofm_buffer.m1_wr_filter_base;

        // No-pool Mode1 OFM write:
        // lane 0..15 = filters f0..f15 at one spatial col_base.
        g0 = $signed(dut.u_ofm_buffer.m1_wr_data[0]);
        g1 = $signed(dut.u_ofm_buffer.m1_wr_data[1]);
        g2 = $signed(dut.u_ofm_buffer.m1_wr_data[2]);
        g3 = $signed(dut.u_ofm_buffer.m1_wr_data[3]);
        g4 = $signed(dut.u_ofm_buffer.m1_wr_data[4]);
        g5 = $signed(dut.u_ofm_buffer.m1_wr_data[5]);
        g6 = $signed(dut.u_ofm_buffer.m1_wr_data[6]);
        g7 = $signed(dut.u_ofm_buffer.m1_wr_data[7]);

        e0 = tb_fast_exp_l0_dbg(row_i, col_i, fbase_i + 0);
        e1 = tb_fast_exp_l0_dbg(row_i, col_i, fbase_i + 1);
        e2 = tb_fast_exp_l0_dbg(row_i, col_i, fbase_i + 2);
        e3 = tb_fast_exp_l0_dbg(row_i, col_i, fbase_i + 3);
        e4 = tb_fast_exp_l0_dbg(row_i, col_i, fbase_i + 4);
        e5 = tb_fast_exp_l0_dbg(row_i, col_i, fbase_i + 5);
        e6 = tb_fast_exp_l0_dbg(row_i, col_i, fbase_i + 6);
        e7 = tb_fast_exp_l0_dbg(row_i, col_i, fbase_i + 7);

        err_f = 0;
        if (g0 != e0) err_f = err_f | 1;
        if (g1 != e1) err_f = err_f | 2;
        if (g2 != e2) err_f = err_f | 4;
        if (g3 != e3) err_f = err_f | 8;
        if (g4 != e4) err_f = err_f | 16;
        if (g5 != e5) err_f = err_f | 32;
        if (g6 != e6) err_f = err_f | 64;
        if (g7 != e7) err_f = err_f | 128;

        $display("DBG_FAST_L0_M1_FILTER_CHECK t=%0t row=%0d col=%0d err_f=0x%0h got=%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d exp=%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
            $time, row_i, col_i, err_f,
            g0,g1,g2,g3,g4,g5,g6,g7,
            e0,e1,e2,e3,e4,e5,e6,e7);

        if (err_f != 0) begin
            $display("DBG_FAST_L0_M1_FIRST_REAL_ERROR t=%0t row=%0d col=%0d err_f=0x%0h HT=%0d",
                $time, row_i, col_i, err_f, HT);
            $fatal(1, "DBG_FAST_L0_M1_FIRST_REAL_ERROR: stop early");
        end
    end
end  
  
  function automatic int tb_fast_input_pattern_l0_dbg(
    input int row,
    input int col,
    input int ch
);
begin
    tb_fast_input_pattern_l0_dbg =
        1 + (row % 16) + 2 * (col % 16) + 3 * (ch % 3);
end
endfunction

function automatic int tb_fast_exp_l0_dbg(
    input int row,
    input int col,
    input int fout
);
begin
    // L0 center-tap: output filter fout selects input channel fout % 3
    tb_fast_exp_l0_dbg = tb_fast_input_pattern_l0_dbg(row, col, fout % 3);
end
endfunction
  
endmodule
