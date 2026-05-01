`timescale 1ns/1ps
`include "cnn_ddr_defs.svh"

module tb_cnn_top_3layer_m1_stress_tiled;
  import cnn_layer_desc_pkg::*;

  localparam int DATA_W=8, PSUM_W=32;
  localparam int PTOTAL=8, PV_MAX=4, PF_MAX=4, PV_M1=2, PF_M1=4;
  localparam int PC_MODE2=4, PF_MODE2=2;
  localparam int C_MAX=12, F_MAX=12, W_MAX=16, H_MAX=40, HT=4, K_MAX=3;
  localparam int WGT_DEPTH=512, OFM_BANK_DEPTH=H_MAX*W_MAX;
  localparam int OFM_LINEAR_DEPTH=C_MAX*OFM_BANK_DEPTH, CFG_DEPTH=8;
  localparam int DDR_ADDR_W=`CNN_DDR_ADDR_W, DDR_WORD_W=PV_MAX*DATA_W;
  localparam int MEM_DEPTH=(`DDR_RSVD_BASE + `DDR_RSVD_SIZE);
  localparam int CLK_PERIOD_NS=10, MAX_CYCLES=3000000;

  // Stress target:
  // - 3 consecutive mode-1 layers.
  // - H_in > HT for all layers, so IFM is larger than IFM buffer.
  // - L0 input comes from DDR and requires DDR->IFM tiling/refill.
  // - L1 and L2 inputs come from previous OFM and require OFM->IFM tiled refill.
  // - C_in > PC_MODE2, F_out > PF_M1, W_in > PV_M1, and F/PF has multiple groups.
  // L0: 40x16x6 -> conv3x3 38x14x8 -> pool 19x7x8
  localparam int L0_H_IN=40, L0_W_IN=16, L0_C_IN=6, L0_F_OUT=8, L0_K=3;
  localparam int L0_H_CONV_OUT=L0_H_IN-L0_K+1, L0_W_CONV_OUT=L0_W_IN-L0_K+1;
  localparam int L0_H_POOL_OUT=L0_H_CONV_OUT/2, L0_W_POOL_OUT=L0_W_CONV_OUT/2;

  // L1: 19x7x8 -> conv1x1 19x7x12 -> pool 9x3x12
  localparam int L1_H_IN=L0_H_POOL_OUT, L1_W_IN=L0_W_POOL_OUT, L1_C_IN=L0_F_OUT;
  localparam int L1_F_OUT=12, L1_K=1;
  localparam int L1_H_CONV_OUT=L1_H_IN-L1_K+1, L1_W_CONV_OUT=L1_W_IN-L1_K+1;
  localparam int L1_H_POOL_OUT=L1_H_CONV_OUT/2, L1_W_POOL_OUT=L1_W_CONV_OUT/2;

  // L2: 9x3x12 -> conv1x1 9x3x12 -> pool 4x1x12
  localparam int L2_H_IN=L1_H_POOL_OUT, L2_W_IN=L1_W_POOL_OUT, L2_C_IN=L1_F_OUT;
  localparam int L2_F_OUT=12, L2_K=1;
  localparam int L2_H_CONV_OUT=L2_H_IN-L2_K+1, L2_W_CONV_OUT=L2_W_IN-L2_K+1;
  localparam int L2_H_POOL_OUT=L2_H_CONV_OUT/2, L2_W_POOL_OUT=L2_W_CONV_OUT/2;

  localparam int L0_IFM_WORDS_PER_ROW=(L0_W_IN+PV_M1-1)/PV_M1;
  localparam int L0_EXPECT_IFM_DDR_WRITES=L0_C_IN*L0_H_IN*L0_IFM_WORDS_PER_ROW;
  localparam int L0_EXPECT_IFM_DDR_WRITES_AFTER_START=(L0_H_IN-HT)*L0_C_IN*L0_IFM_WORDS_PER_ROW;

  localparam int L0_NUM_FGROUP=(L0_F_OUT+PF_M1-1)/PF_M1;
  localparam int L1_NUM_FGROUP=(L1_F_OUT+PF_M1-1)/PF_M1;
  localparam int L2_NUM_FGROUP=(L2_F_OUT+PF_M1-1)/PF_M1;

  localparam int L0_M1_LOGICAL_BUNDLES=L0_NUM_FGROUP*L0_C_IN*L0_K*L0_K;
  localparam int L1_M1_LOGICAL_BUNDLES=L1_NUM_FGROUP*L1_C_IN*L1_K*L1_K;
  localparam int L2_M1_LOGICAL_BUNDLES=L2_NUM_FGROUP*L2_C_IN*L2_K*L2_K;
  localparam int L0_M1_PHYS_WORDS=(L0_M1_LOGICAL_BUNDLES+PV_M1-1)/PV_M1;
  localparam int L1_M1_PHYS_WORDS=(L1_M1_LOGICAL_BUNDLES+PV_M1-1)/PV_M1;
  localparam int L2_M1_PHYS_WORDS=(L2_M1_LOGICAL_BUNDLES+PV_M1-1)/PV_M1;
  localparam int WGT_SUBWORDS=(PTOTAL+PV_MAX-1)/PV_MAX;
  localparam int L0_M1_WGT_DDR_WORDS=L0_M1_PHYS_WORDS*WGT_SUBWORDS;
  localparam int L1_M1_WGT_DDR_WORDS=L1_M1_PHYS_WORDS*WGT_SUBWORDS;
  localparam int L2_M1_WGT_DDR_WORDS=L2_M1_PHYS_WORDS*WGT_SUBWORDS;
  localparam int L0_WGT_DDR_BASE=`DDR_WGT_BASE;
  localparam int L1_WGT_DDR_BASE=`DDR_WGT_BASE+L0_M1_WGT_DDR_WORDS;
  localparam int L2_WGT_DDR_BASE=L1_WGT_DDR_BASE+L1_M1_WGT_DDR_WORDS;

  localparam int FINAL_STORE_PACK=1;
  localparam int FINAL_GROUPS=(L2_W_POOL_OUT+FINAL_STORE_PACK-1)/FINAL_STORE_PACK;
  localparam int EXP_OFM_WORDS=L2_F_OUT*L2_H_POOL_OUT*FINAL_GROUPS;

  localparam int L1_OFM2IFM_COL_BLKS=(L1_W_IN+PV_M1-1)/PV_M1;
  localparam int L1_OFM2IFM_CH_BLKS=(L1_C_IN+PF_M1-1)/PF_M1;
  localparam int L2_OFM2IFM_COL_BLKS=(L2_W_IN+PV_M1-1)/PV_M1;
  localparam int L2_OFM2IFM_CH_BLKS=(L2_C_IN+PF_M1-1)/PF_M1;
  localparam int EXP_OFM2IFM_STREAMS=(L1_H_IN*L1_OFM2IFM_COL_BLKS*L1_OFM2IFM_CH_BLKS) +
                                     (L2_H_IN*L2_OFM2IFM_COL_BLKS*L2_OFM2IFM_CH_BLKS);

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

initial begin
  $display("DBG_WIDTH C_MAX=%0d", C_MAX);
  $display("DBG_WIDTH bits dut.ifm_ofm_wr_bank_s = %0d", $bits(dut.ifm_ofm_wr_bank_s));
  $display("DBG_WIDTH bits dut.ifm_rd_bank_base_s = %0d", $bits(dut.ifm_rd_bank_base_s));
  $display("DBG_WIDTH bits dut.u_ifm_buffer.ofm_wr_bank = %0d", $bits(dut.u_ifm_buffer.ofm_wr_bank));
  $display("DBG_WIDTH bits dut.u_ifm_buffer.rd_bank_base = %0d", $bits(dut.u_ifm_buffer.rd_bank_base));
end

always @(posedge clk) begin
  if (rst_n &&
      dut.dbg_layer_idx == 1 &&
      dut.ifm_ofm_wr_en_s &&
      dut.ifm_ofm_wr_bank_s >= 8) begin
    $strobe("DBG_IFM_BANK_GE8_WRITE t=%0t bank=%0d row_slot=%0d col=%0d wr_data=0x%0h",
            $time,
            dut.ifm_ofm_wr_bank_s,
            dut.ifm_ofm_wr_row_idx_s,
            dut.ifm_ofm_wr_col_idx_s,
            dut.ifm_ofm_wr_data_s);
  end

  if (rst_n &&
      dut.dbg_layer_idx == 2 &&
      dut.ifm_rd_en_s &&
      dut.ifm_rd_bank_base_s >= 8) begin
    $strobe("DBG_IFM_BANK_GE8_READ t=%0t bank=%0d row_idx=%0d col=%0d rd_data=0x%0h",
            $time,
            dut.ifm_rd_bank_base_s,
            dut.ifm_rd_row_idx_s,
            dut.ifm_rd_col_idx_s,
            dut.ifm_rd_data_s);
  end
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

// ============================================================
// DEBUG: L1->L2 initial OFM->IFM tile + L2 IFM read path
// Paste into tb_cnn_top_3layer_m1_stress_tiled.sv
// ============================================================

integer dbg_l2_initial_ofm2ifm_wr_count;
integer dbg_l2_ifm_rd_count;
integer dbg_l2_dr_wr_count;
logic   dbg_seen_layer2_start_q;

always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    dbg_l2_initial_ofm2ifm_wr_count <= 0;
    dbg_l2_ifm_rd_count             <= 0;
    dbg_l2_dr_wr_count              <= 0;
    dbg_seen_layer2_start_q         <= 1'b0;
  end else begin

    // --------------------------------------------------------
    // 1) Detect transition into layer 2 and print IFM ring state
    // --------------------------------------------------------
    if (!dbg_seen_layer2_start_q && dut.dbg_layer_idx == 2) begin
      dbg_seen_layer2_start_q <= 1'b1;

      $display("DBG_L2_START t=%0t ifm_m1_row_base_l=%0d m1_busy=%0b m1_start=%0b dbg_layer_idx=%0d",
                $time,
                dut.ifm_m1_row_base_l_s,
                dut.m1_busy_s,
                dut.m1_start_s,
                dut.dbg_layer_idx);

      $display("DBG_L2_START cfg: out_row=%0d out_col=%0d f_group=%0d c_iter=%0d ky=%0d kx=%0d",
               dut.m1_out_row_s,
               dut.m1_out_col_s,
               dut.m1_f_group_s,
               dut.m1_c_iter_s,
               dut.m1_ky_s,
               dut.m1_kx_s);
    end

    // --------------------------------------------------------
    // 2) Print L1->L2 initial tile writes from OFM buffer into IFM buffer
    //
    // At this moment dbg_layer_idx is still 1.
    // This is the initial tile for layer 2:
    //   row_base 0..3
    //   row_slot 0..3
    //   col_base 0 or 2
    //   ch_blk 0,1,2
    //
    // We care most about row 0..3, ch_blk 2, banks 8..11.
    // --------------------------------------------------------
    if (rst_n &&
        dut.dbg_layer_idx == 1 &&
        dut.ifm_ofm_wr_en_s &&
        dut.ofm_ifm_stream_row_base_s < 4) begin

      dbg_l2_initial_ofm2ifm_wr_count <= dbg_l2_initial_ofm2ifm_wr_count + 1;

      $display("DBG_L1_TO_L2_INIT_WR t=%0t src_row=%0d col_base=%0d dst_slot=%0d ch_blk=%0d | wr_bank=%0d wr_row_slot=%0d wr_col=%0d keep=%b data=0x%0h word_ready=%0b",
               $time,
               dut.ofm_ifm_stream_row_base_s,
               dut.ofm_ifm_stream_col_base_s,
               dut.ofm_ifm_stream_m1_row_slot_l_s,
               dut.ofm_ifm_stream_m1_ch_blk_g_s,
               dut.ifm_ofm_wr_bank_s,
               dut.ifm_ofm_wr_row_idx_s,
               dut.ifm_ofm_wr_col_idx_s,
               dut.ifm_ofm_wr_keep_s,
               dut.ifm_ofm_wr_data_s,
               dut.u_ofm_buffer.word_ready_v);

      if (dut.ofm_ifm_stream_m1_ch_blk_g_s == 2) begin
        $display("  CHECK_INIT_CHBLK2 src_row=%0d col_base=%0d bank=%0d slot=%0d col=%0d keep=%b data=0x%0h",
                 dut.ofm_ifm_stream_row_base_s,
                 dut.ofm_ifm_stream_col_base_s,
                 dut.ifm_ofm_wr_bank_s,
                 dut.ifm_ofm_wr_row_idx_s,
                 dut.ifm_ofm_wr_col_idx_s,
                 dut.ifm_ofm_wr_keep_s,
                 dut.ifm_ofm_wr_data_s);
      end
    end

    // --------------------------------------------------------
    // 3) Print L2 IFM reads for early rows.
    //
    // This tells us whether CE/data register is reading from the
    // initial tile correctly after layer 2 starts.
    // --------------------------------------------------------
    if (rst_n &&
        dut.dbg_layer_idx == 2 &&
        dut.ifm_rd_en_s &&
        dut.m1_out_row_s <= 1) begin

      dbg_l2_ifm_rd_count <= dbg_l2_ifm_rd_count + 1;

      $display("DBG_L2_IFM_RD_REQ t=%0t out_row=%0d out_col=%0d f_group=%0d c_iter=%0d ky=%0d kx=%0d | row_base_l=%0d rd_bank_base=%0d rd_row_idx=%0d rd_col_idx=%0d",
               $time,
               dut.m1_out_row_s,
               dut.m1_out_col_s,
               dut.m1_f_group_s,
               dut.m1_c_iter_s,
               dut.m1_ky_s,
               dut.m1_kx_s,
               dut.ifm_m1_row_base_l_s,
               dut.ifm_rd_bank_base_s,
               dut.ifm_rd_row_idx_s,
               dut.ifm_rd_col_idx_s);
    end

    if (rst_n &&
        dut.dbg_layer_idx == 2 &&
        dut.ifm_rd_valid_s &&
        dut.m1_out_row_s <= 1) begin

      $display("DBG_L2_IFM_RD_DATA t=%0t out_row=%0d c_iter=%0d ky=%0d kx=%0d | rd_data=0x%0h",
               $time,
               dut.m1_out_row_s,
               dut.m1_c_iter_s,
               dut.m1_ky_s,
               dut.m1_kx_s,
               dut.ifm_rd_data_s);
    end

    // --------------------------------------------------------
    // 4) Print data-register writes for layer 2 early rows.
    //
    // If initial tile was written correctly but dr_write_data is zero,
    // the issue is between IFM read and data register.
    // If dr_write_data is already wrong/zero for early rows, layer 2
    // compute will naturally output zero.
    // --------------------------------------------------------
    if (rst_n &&
        dut.dbg_layer_idx == 2 &&
        dut.m1_dr_write_en_s &&
        dut.m1_out_row_s <= 1) begin

      dbg_l2_dr_wr_count <= dbg_l2_dr_wr_count + 1;

      $display("DBG_L2_DR_WRITE t=%0t out_row=%0d out_col=%0d f_group=%0d c_iter=%0d ky=%0d kx=%0d | dr_row_idx=%0d x_base=%0d data=0x%0h",
               $time,
               dut.m1_out_row_s,
               dut.m1_out_col_s,
               dut.m1_f_group_s,
               dut.m1_c_iter_s,
               dut.m1_ky_s,
               dut.m1_kx_s,
               dut.m1_dr_write_row_idx_s,
               dut.m1_dr_write_x_base_s,
               dut.m1_dr_write_data_s);
    end

    // --------------------------------------------------------
    // 5) Print IFM free tokens around layer 2.
    // This helps confirm row-base/ring movement.
    // --------------------------------------------------------
    if (rst_n &&
        dut.dbg_layer_idx == 2 &&
        dut.ifm_m1_free_valid) begin
      $display("DBG_L2_IFM_FREE t=%0t row_base_l=%0d free_slot=%0d free_row_g=%0d out_row=%0d",
               $time,
               dut.ifm_m1_row_base_l_s,
               dut.ifm_m1_free_row_slot_l,
               dut.ifm_m1_free_row_g,
               dut.m1_out_row_s);
    end
  end
end

final begin
  $display("DBG_SUMMARY: L1->L2 initial OFM->IFM writes observed = %0d",
           dbg_l2_initial_ofm2ifm_wr_count);
  $display("DBG_SUMMARY: L2 IFM read requests observed for out_row<=1 = %0d",
           dbg_l2_ifm_rd_count);
  $display("DBG_SUMMARY: L2 DR writes observed for out_row<=1 = %0d",
           dbg_l2_dr_wr_count);
end

// Debug final layer write into OFM buffer.
// Adjust hierarchy if these signals are under dut.u_control_unit or dut directly.
always_ff @(posedge clk) begin
  if (rst_n && dut.dbg_layer_idx == 2 && dut.m1_ofm_write_en_s) begin
    $display(
      "DBG_L2_OFM_WRITE t=%0t row=%0d col_base=%0d fbase=%0d count=%0d data0=%0h data1=%0h data2=%0h data3=%0h",
      $time,
      dut.m1_ofm_write_row_s,
      dut.m1_ofm_write_col_base_s,
      dut.m1_ofm_write_filter_base_s,
      dut.m1_ofm_write_count_s,
      dut.m1_ofm_write_data_s[0],
      dut.m1_ofm_write_data_s[1],
      dut.m1_ofm_write_data_s[2],
      dut.m1_ofm_write_data_s[3]
    );

    if (dut.m1_ofm_write_row_s <= 1) begin
      if (dut.m1_ofm_write_filter_base_s == 0) begin
        $display("  CHECK ch3 row=%0d got lane3=%0h",
                 dut.m1_ofm_write_row_s,
                 dut.m1_ofm_write_data_s[3]);
      end

      if (dut.m1_ofm_write_filter_base_s == 4) begin
        $display("  CHECK ch4 row=%0d got lane0=%0h",
                 dut.m1_ofm_write_row_s,
                 dut.m1_ofm_write_data_s[0]);
      end

      if (dut.m1_ofm_write_filter_base_s == 8) begin
        $display("  CHECK ch8 row=%0d got lane0=%0h, ch9 lane1=%0h",
                 dut.m1_ofm_write_row_s,
                 dut.m1_ofm_write_data_s[0],
                 dut.m1_ofm_write_data_s[1]);
      end
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
            l0_ifm[c][r][x] = $signed(((c*17 + r*5 + x*3) % 9) - 4);

      for (f=0;f<L0_F_OUT;f=f+1)
        for (c=0;c<L0_C_IN;c=c+1)
          for (ky=0;ky<L0_K;ky=ky+1)
            for (kx=0;kx<L0_K;kx=kx+1)
              l0_wgt[f][c][ky][kx] = $signed(((f*13 + c*7 + ky*3 + kx*5) % 5) - 2);

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
              l1_wgt[f][c][ky][kx] = $signed(((f*11 + c*5 + ky*7 + kx*3 + 1) % 5) - 2);

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
              l2_wgt[f][c][ky][kx] = $signed(((f*7 + c*3 + ky*5 + kx*11 + 2) % 5) - 2);

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

  task automatic init_mem;
    begin
      clear_ddr();
      build_golden();
      load_l0_ifm_to_ddr();
      load_l0_weights_to_ddr();
      load_l1_weights_to_ddr();
      load_l2_weights_to_ddr();
      $display("TB_INFO: 3-layer Mode1 stress tiled test");
      $display("TB_INFO: L0 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> pool %0dx%0dx%0d", L0_H_IN,L0_W_IN,L0_C_IN,L0_H_CONV_OUT,L0_W_CONV_OUT,L0_F_OUT,L0_K,L0_H_POOL_OUT,L0_W_POOL_OUT,L0_F_OUT);
      $display("TB_INFO: L1 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> pool %0dx%0dx%0d", L1_H_IN,L1_W_IN,L1_C_IN,L1_H_CONV_OUT,L1_W_CONV_OUT,L1_F_OUT,L1_K,L1_H_POOL_OUT,L1_W_POOL_OUT,L1_F_OUT);
      $display("TB_INFO: L2 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> pool %0dx%0dx%0d", L2_H_IN,L2_W_IN,L2_C_IN,L2_H_CONV_OUT,L2_W_CONV_OUT,L2_F_OUT,L2_K,L2_H_POOL_OUT,L2_W_POOL_OUT,L2_F_OUT);
      $display("TB_INFO: stress params: H_in all > HT=%0d, W_in all > Pv=%0d, C_in all > PC=%0d, F_out all > Pf=%0d", HT, PV_M1, PC_MODE2, PF_M1);
      $display("TB_INFO: Pv=%0d Pf=%0d PTOTAL=%0d", PV_M1, PF_M1, PTOTAL);
      $display("TB_INFO: L0 weight bundles=%0d phys=%0d DDR words=%0d base=0x%05h", L0_M1_LOGICAL_BUNDLES, L0_M1_PHYS_WORDS, L0_M1_WGT_DDR_WORDS, L0_WGT_DDR_BASE);
      $display("TB_INFO: L1 weight bundles=%0d phys=%0d DDR words=%0d base=0x%05h", L1_M1_LOGICAL_BUNDLES, L1_M1_PHYS_WORDS, L1_M1_WGT_DDR_WORDS, L1_WGT_DDR_BASE);
      $display("TB_INFO: L2 weight bundles=%0d phys=%0d DDR words=%0d base=0x%05h", L2_M1_LOGICAL_BUNDLES, L2_M1_PHYS_WORDS, L2_M1_WGT_DDR_WORDS, L2_WGT_DDR_BASE);
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
    layer_desc_t cfg0,cfg1,cfg2;
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
      cfg2.first_layer=1'b0; cfg2.last_layer=1'b1;

      write_cfg('0,cfg0);
      write_cfg(1,cfg1);
      write_cfg(2,cfg2);
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
      for (lin=0;lin<EXP_OFM_WORDS;lin=lin+1) begin
        ch=lin/(L2_H_POOL_OUT*FINAL_GROUPS);
        rem=lin%(L2_H_POOL_OUT*FINAL_GROUPS);
        row=rem/FINAL_GROUPS;
        grp=rem%FINAL_GROUPS;
        col=grp*FINAL_STORE_PACK;
        got=ddr_mem[`DDR_OFM_BASE+lin][0 +: DATA_W];
        exp=l2_pool[ch][row][col];
        if (got !== exp) begin
          $display("TB_FAIL: final OFM mismatch lin=%0d ch=%0d row=%0d col=%0d got=%0d/0x%02h expected=%0d/0x%02h word=0x%08h",
                   lin,ch,row,col,got,got,exp,exp,ddr_mem[`DDR_OFM_BASE+lin]);
          ofm_mismatch_count++;
        end
      end
      if ((fail_count==0) && (ofm_mismatch_count==0)) begin
        $display("TB_PASS: exact final OFM matches golden for 3-layer stress tiled mode1 test");
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
    cfg_wr_en=1'b0; cfg_wr_addr='0; cfg_wr_data='0; cfg_num_layers=3;
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
