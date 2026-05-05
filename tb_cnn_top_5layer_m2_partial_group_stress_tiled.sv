`timescale 1ns/1ps
`include "cnn_ddr_defs.svh"

module tb_cnn_top_5layer_m2_partial_group_stress_tiled;
  import cnn_layer_desc_pkg::*;

  // --------------------------------------------------------------------------
  // 5-layer Mode-2 partial-group/tiled stress test.
  //
  // Intent: Mode-2 analogue of the Mode-1 partial-group tiled stress test.
  // It stresses:
  //   - first-layer DDR->IFM preload in Mode 2,
  //   - Mode2->Mode2 OFM->IFM same-mode refill for multiple layers,
  //   - partial C groups (C not equal to PC; only valid channels contribute),
  //   - partial F groups (F not divisible by PF),
  //   - large enough H/W to require many OFM->IFM stream commands,
  //   - exact final DDR check with simple all-one IFM/weight data.
  //
  // Current Mode-2 IFM storage keeps one PC-wide horizontal segment per
  // channel/row, so this test chooses W0=PC and relies on pooling to shrink W.
  // --------------------------------------------------------------------------

  localparam int DATA_W = 8;
  localparam int PSUM_W = 32;

  localparam int PC = 32;
  localparam int PF = 3;
  localparam int PTOTAL = PC * PF;      // 96 lanes
  localparam int PV_MAX = PTOTAL;       // DDR word = one physical PTOTAL word
  localparam int PF_MAX = 16;

  localparam int C_MAX = 16;
  localparam int F_MAX = 16;
  localparam int H_MAX = 64;
  localparam int W_MAX = 32;
  localparam int HT = 4;
  localparam int K_MAX = 3;
  localparam int WGT_DEPTH = 256;

  // Row-aligned OFM storage.  W_MAX is also the maximum Mode-2 output row
  // stride needed in this test.
  localparam int OFM_ROW_STRIDE = W_MAX;
  localparam int OFM_BANK_DEPTH = H_MAX * OFM_ROW_STRIDE;
  localparam int OFM_LINEAR_DEPTH = C_MAX * OFM_BANK_DEPTH;

  localparam int CFG_DEPTH = 8;
  localparam int DDR_ADDR_W = `CNN_DDR_ADDR_W;
  localparam int DDR_WORD_W = PV_MAX * DATA_W;
  localparam int MEM_DEPTH = (`DDR_RSVD_BASE + `DDR_RSVD_SIZE);
  localparam int CLK_PERIOD_NS = 10;
  localparam int MAX_CYCLES = 5000000;

  // All layers use K=1 and pool2x2.  With IFM=1 and WGT=1, each layer's
  // output value is previous_value * C_in, saturated to signed 8-bit by
  // pooling_mode2 before being stored to OFM.
  // L0: 64x32x3  -> conv 64x32x5  -> pool 32x16x5   value 3
  // L1: 32x16x5  -> conv 32x16x7  -> pool 16x8x7    value 15
  // L2: 16x8x7   -> conv 16x8x10  -> pool 8x4x10    value 105
  // L3: 8x4x10   -> conv 8x4x11   -> pool 4x2x11    value 127 (sat)
  // L4: 4x2x11   -> conv 4x2x5    -> pool 2x1x5     value 127 (sat)

  localparam int L0_H_IN=64, L0_W_IN=32, L0_C_IN=3,  L0_F_OUT=5,  L0_K=1;
  localparam int L0_H_CONV_OUT=L0_H_IN-L0_K+1, L0_W_CONV_OUT=L0_W_IN-L0_K+1;
  localparam int L0_H_POOL_OUT=L0_H_CONV_OUT/2, L0_W_POOL_OUT=L0_W_CONV_OUT/2;

  localparam int L1_H_IN=L0_H_POOL_OUT, L1_W_IN=L0_W_POOL_OUT, L1_C_IN=L0_F_OUT;
  localparam int L1_F_OUT=7, L1_K=1;
  localparam int L1_H_CONV_OUT=L1_H_IN-L1_K+1, L1_W_CONV_OUT=L1_W_IN-L1_K+1;
  localparam int L1_H_POOL_OUT=L1_H_CONV_OUT/2, L1_W_POOL_OUT=L1_W_CONV_OUT/2;

  localparam int L2_H_IN=L1_H_POOL_OUT, L2_W_IN=L1_W_POOL_OUT, L2_C_IN=L1_F_OUT;
  localparam int L2_F_OUT=10, L2_K=1;
  localparam int L2_H_CONV_OUT=L2_H_IN-L2_K+1, L2_W_CONV_OUT=L2_W_IN-L2_K+1;
  localparam int L2_H_POOL_OUT=L2_H_CONV_OUT/2, L2_W_POOL_OUT=L2_W_CONV_OUT/2;

  localparam int L3_H_IN=L2_H_POOL_OUT, L3_W_IN=L2_W_POOL_OUT, L3_C_IN=L2_F_OUT;
  localparam int L3_F_OUT=11, L3_K=1;
  localparam int L3_H_CONV_OUT=L3_H_IN-L3_K+1, L3_W_CONV_OUT=L3_W_IN-L3_K+1;
  localparam int L3_H_POOL_OUT=L3_H_CONV_OUT/2, L3_W_POOL_OUT=L3_W_CONV_OUT/2;

  localparam int L4_H_IN=L3_H_POOL_OUT, L4_W_IN=L3_W_POOL_OUT, L4_C_IN=L3_F_OUT;
  localparam int L4_F_OUT=5, L4_K=1;
  localparam int L4_H_CONV_OUT=L4_H_IN-L4_K+1, L4_W_CONV_OUT=L4_W_IN-L4_K+1;
  localparam int L4_H_POOL_OUT=L4_H_CONV_OUT/2, L4_W_POOL_OUT=L4_W_CONV_OUT/2;

  localparam int L0_NUM_CGROUP=(L0_C_IN+PC-1)/PC;
  localparam int L0_NUM_FGROUP=(L0_F_OUT+PF-1)/PF;
  localparam int L1_NUM_CGROUP=(L1_C_IN+PC-1)/PC;
  localparam int L1_NUM_FGROUP=(L1_F_OUT+PF-1)/PF;
  localparam int L2_NUM_CGROUP=(L2_C_IN+PC-1)/PC;
  localparam int L2_NUM_FGROUP=(L2_F_OUT+PF-1)/PF;
  localparam int L3_NUM_CGROUP=(L3_C_IN+PC-1)/PC;
  localparam int L3_NUM_FGROUP=(L3_F_OUT+PF-1)/PF;
  localparam int L4_NUM_CGROUP=(L4_C_IN+PC-1)/PC;
  localparam int L4_NUM_FGROUP=(L4_F_OUT+PF-1)/PF;

  localparam int L0_WGT_WORDS=L0_NUM_FGROUP*L0_NUM_CGROUP*L0_K*L0_K;
  localparam int L1_WGT_WORDS=L1_NUM_FGROUP*L1_NUM_CGROUP*L1_K*L1_K;
  localparam int L2_WGT_WORDS=L2_NUM_FGROUP*L2_NUM_CGROUP*L2_K*L2_K;
  localparam int L3_WGT_WORDS=L3_NUM_FGROUP*L3_NUM_CGROUP*L3_K*L3_K;
  localparam int L4_WGT_WORDS=L4_NUM_FGROUP*L4_NUM_CGROUP*L4_K*L4_K;

  localparam int L0_WGT_DDR_BASE=`DDR_WGT_BASE;
  localparam int L1_WGT_DDR_BASE=L0_WGT_DDR_BASE+L0_WGT_WORDS;
  localparam int L2_WGT_DDR_BASE=L1_WGT_DDR_BASE+L1_WGT_WORDS;
  localparam int L3_WGT_DDR_BASE=L2_WGT_DDR_BASE+L2_WGT_WORDS;
  localparam int L4_WGT_DDR_BASE=L3_WGT_DDR_BASE+L3_WGT_WORDS;

  localparam int L1_STREAMS=L1_H_IN*L1_W_IN*L1_NUM_CGROUP;
  localparam int L2_STREAMS=L2_H_IN*L2_W_IN*L2_NUM_CGROUP;
  localparam int L3_STREAMS=L3_H_IN*L3_W_IN*L3_NUM_CGROUP;
  localparam int L4_STREAMS=L4_H_IN*L4_W_IN*L4_NUM_CGROUP;
  localparam int EXP_OFM2IFM_STREAMS=L1_STREAMS+L2_STREAMS+L3_STREAMS+L4_STREAMS;

  localparam int EXPECTED_OFM_DDR_WORDS=L4_F_OUT*L4_H_POOL_OUT*L4_W_POOL_OUT;
  localparam int EXPECTED_FINAL_VALUE=127;

  // --------------------------------------------------------------------------
  // DUT I/O
  // --------------------------------------------------------------------------
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

  // --------------------------------------------------------------------------
  // DDR model and counters
  // --------------------------------------------------------------------------
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

  // --------------------------------------------------------------------------
  // DUT
  // --------------------------------------------------------------------------
  cnn_top #(
    .DATA_W(DATA_W), .PSUM_W(PSUM_W), .PTOTAL(PTOTAL),
    .PV_MAX(PV_MAX), .PF_MAX(PF_MAX),
    .PC_MODE2(PC), .PF_MODE2(PF),
    .C_MAX(C_MAX), .F_MAX(F_MAX),
    .W_MAX(W_MAX), .H_MAX(H_MAX), .HT(HT), .K_MAX(K_MAX),
    .WGT_DEPTH(WGT_DEPTH),
    .OFM_BANK_DEPTH(OFM_BANK_DEPTH),
    .OFM_LINEAR_DEPTH(OFM_LINEAR_DEPTH),
    .CFG_DEPTH(CFG_DEPTH),
    .DDR_ADDR_W(DDR_ADDR_W),
    .DDR_WORD_W(DDR_WORD_W)
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

  initial clk = 1'b0;
  always #(CLK_PERIOD_NS/2) clk = ~clk;

  // --------------------------------------------------------------------------
  // DDR model
  // --------------------------------------------------------------------------
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
        $display("DBG_FIRST_ERROR t=%0t cycle=%0d dbg_error_vec=%04b layer=%0d mode=%0d busy=%0b done=%0b error=%0b",
          $time, cycle_count, dbg_error_vec, dbg_layer_idx, dbg_mode, busy, done, error);
        $display("DBG_ERROR_MAP bit0=dma_error bit1=ofm_error bit2=local_error bit3=transition_error");
      end

      if (rd_pending_q) begin
        ddr_rd_valid <= 1'b1;
        if (rd_addr_q < MEM_DEPTH)
          ddr_rd_data <= ddr_mem[rd_addr_q];
        else
          ddr_rd_data <= '0;
      end

      if (ddr_wr_en) begin
        if (ddr_wr_addr < MEM_DEPTH) begin
          for (b = 0; b < (DDR_WORD_W/8); b = b + 1) begin
            if (ddr_wr_be[b]) begin
              ddr_mem[ddr_wr_addr][8*b +: 8] <= ddr_wr_data[8*b +: 8];
            end
          end
        end
        if ((ddr_wr_addr >= `DDR_OFM_BASE) &&
            (ddr_wr_addr < (`DDR_OFM_BASE + `DDR_OFM_SIZE))) begin
          ddr_ofm_write_count <= ddr_ofm_write_count + 1;
        end
      end

      if (ddr_rd_req) begin
        rd_pending_q <= 1'b1;
        rd_addr_q <= ddr_rd_addr;
        if ((ddr_rd_addr >= `DDR_IFM_BASE) &&
            (ddr_rd_addr < (`DDR_IFM_BASE + `DDR_IFM_SIZE))) begin
          ddr_ifm_read_count <= ddr_ifm_read_count + 1;
        end
        if ((ddr_rd_addr >= `DDR_WGT_BASE) &&
            (ddr_rd_addr < (`DDR_WGT_BASE + `DDR_WGT_SIZE))) begin
          ddr_wgt_read_count <= ddr_wgt_read_count + 1;
        end
      end else if (rd_pending_q) begin
        rd_pending_q <= 1'b0;
      end

      if (dut.ofm_ifm_stream_start_s)
        ofm_ifm_stream_start_count <= ofm_ifm_stream_start_count + 1;
      if (dut.ofm_ifm_stream_done_s)
        ofm_ifm_stream_done_count <= ofm_ifm_stream_done_count + 1;
      if (m2_sm_refill_req_valid && m2_sm_refill_req_ready)
        m2_sm_refill_req_count <= m2_sm_refill_req_count + 1;
    end
  end

  always_ff @(posedge clk) begin
    if (rst_n) begin
      if (dut.ofm_ifm_stream_start_s || dut.ofm_ifm_stream_done_s || done || error) begin
        $display("DBG_TOP_STATUS t=%0t cycle=%0d busy=%0b done=%0b error=%0b vec=%04b layer=%0d mode=%0d ifm_rd=%0d wgt_rd=%0d ofm_wr=%0d stream_start=%0d stream_done=%0d m2_req=%0d",
          $time, cycle_count, busy, done, error, dbg_error_vec, dbg_layer_idx, dbg_mode,
          ddr_ifm_read_count, ddr_wgt_read_count, ddr_ofm_write_count,
          ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, m2_sm_refill_req_count);
      end
    end
  end

  // --------------------------------------------------------------------------
  // Helpers
  // --------------------------------------------------------------------------
  function automatic logic [DDR_WORD_W-1:0] pack_pc_ones_word;
    logic [DDR_WORD_W-1:0] word;
    begin
      word = '0;
      for (int lane = 0; lane < PC; lane++) begin
        word[lane*DATA_W +: DATA_W] = 8'sd1;
      end
      return word;
    end
  endfunction

  function automatic logic [DDR_WORD_W-1:0] pack_ptotal_ones_word;
    logic [DDR_WORD_W-1:0] word;
    begin
      word = '0;
      for (int lane = 0; lane < PTOTAL; lane++) begin
        word[lane*DATA_W +: DATA_W] = 8'sd1;
      end
      return word;
    end
  endfunction

  function automatic logic [DDR_WORD_W-1:0] expected_final_word;
    input int val;
    logic [DDR_WORD_W-1:0] word;
    begin
      word = '0;
      word[0 +: DATA_W] = val[DATA_W-1:0];
      return word;
    end
  endfunction

  task automatic init_mem;
    int row;
    int ch;
    int word_idx;
    begin
      for (i = 0; i < MEM_DEPTH; i = i + 1) begin
        ddr_mem[i] = '0;
      end

      // Mode 2 IFM DDR layout: flat [channel][row]; one word holds PC pixels.
      word_idx = 0;
      for (ch = 0; ch < L0_C_IN; ch = ch + 1) begin
        for (row = 0; row < L0_H_IN; row = row + 1) begin
          ddr_mem[`DDR_IFM_BASE + word_idx] = pack_pc_ones_word();
          word_idx = word_idx + 1;
        end
      end

      for (i = 0; i < L0_WGT_WORDS; i = i + 1) ddr_mem[L0_WGT_DDR_BASE + i] = pack_ptotal_ones_word();
      for (i = 0; i < L1_WGT_WORDS; i = i + 1) ddr_mem[L1_WGT_DDR_BASE + i] = pack_ptotal_ones_word();
      for (i = 0; i < L2_WGT_WORDS; i = i + 1) ddr_mem[L2_WGT_DDR_BASE + i] = pack_ptotal_ones_word();
      for (i = 0; i < L3_WGT_WORDS; i = i + 1) ddr_mem[L3_WGT_DDR_BASE + i] = pack_ptotal_ones_word();
      for (i = 0; i < L4_WGT_WORDS; i = i + 1) ddr_mem[L4_WGT_DDR_BASE + i] = pack_ptotal_ones_word();
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
      cfg.pf_m1 = PF;
      cfg.pc_m2 = PC;
      cfg.pf_m2 = PF;
      cfg.conv_stride = 1;
      cfg.pad_top = 0;
      cfg.pad_bottom = 0;
      cfg.pad_left = 0;
      cfg.pad_right = 0;
      cfg.relu_en = 1'b1;
      cfg.pool_en = 1'b1;
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
    layer_desc_t cfg0, cfg1, cfg2, cfg3, cfg4;
    begin
      fill_m2_cfg(cfg0, 0, L0_H_IN, L0_W_IN, L0_C_IN, L0_F_OUT, L0_K,
                  L0_H_CONV_OUT, L0_W_CONV_OUT, L0_WGT_DDR_BASE, 1'b1, 1'b0);
      fill_m2_cfg(cfg1, 1, L1_H_IN, L1_W_IN, L1_C_IN, L1_F_OUT, L1_K,
                  L1_H_CONV_OUT, L1_W_CONV_OUT, L1_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_m2_cfg(cfg2, 2, L2_H_IN, L2_W_IN, L2_C_IN, L2_F_OUT, L2_K,
                  L2_H_CONV_OUT, L2_W_CONV_OUT, L2_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_m2_cfg(cfg3, 3, L3_H_IN, L3_W_IN, L3_C_IN, L3_F_OUT, L3_K,
                  L3_H_CONV_OUT, L3_W_CONV_OUT, L3_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_m2_cfg(cfg4, 4, L4_H_IN, L4_W_IN, L4_C_IN, L4_F_OUT, L4_K,
                  L4_H_CONV_OUT, L4_W_CONV_OUT, L4_WGT_DDR_BASE, 1'b0, 1'b1);

      write_cfg('0, cfg0);
      write_cfg(1, cfg1);
      write_cfg(2, cfg2);
      write_cfg(3, cfg3);
      write_cfg(4, cfg4);
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
      for (j = 0; j < 16; j = j + 1) begin
        $display("OFM[%0d] @0x%05h = 0x%0h", j, (`DDR_OFM_BASE + j), ddr_mem[`DDR_OFM_BASE + j]);
      end
    end
  endtask

  task automatic check_final_ofm;
    int mismatch;
    logic [DDR_WORD_W-1:0] exp_word;
    begin
      mismatch = 0;
      exp_word = expected_final_word(EXPECTED_FINAL_VALUE);
      for (int j = 0; j < EXPECTED_OFM_DDR_WORDS; j = j + 1) begin
        if (ddr_mem[`DDR_OFM_BASE + j] !== exp_word) begin
          if (mismatch < 20) begin
            $display("TB_MISMATCH_M2_5L word=%0d got=0x%0h exp=0x%0h",
              j, ddr_mem[`DDR_OFM_BASE + j], exp_word);
          end
          mismatch = mismatch + 1;
        end
      end
      if (mismatch != 0) begin
        dump_ofm_region();
        $fatal(1, "TB_FAIL: 5-layer Mode2 final OFM mismatch count=%0d", mismatch);
      end
    end
  endtask

  task automatic print_banner;
    begin
      $display("TB_INFO: 5-layer Mode2 partial/non-divisible stress tiled test");
      $display("TB_INFO: L0 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> pool %0dx%0dx%0d",
        L0_H_IN, L0_W_IN, L0_C_IN, L0_H_CONV_OUT, L0_W_CONV_OUT, L0_F_OUT, L0_K,
        L0_H_POOL_OUT, L0_W_POOL_OUT, L0_F_OUT);
      $display("TB_INFO: L1 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> pool %0dx%0dx%0d",
        L1_H_IN, L1_W_IN, L1_C_IN, L1_H_CONV_OUT, L1_W_CONV_OUT, L1_F_OUT, L1_K,
        L1_H_POOL_OUT, L1_W_POOL_OUT, L1_F_OUT);
      $display("TB_INFO: L2 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> pool %0dx%0dx%0d",
        L2_H_IN, L2_W_IN, L2_C_IN, L2_H_CONV_OUT, L2_W_CONV_OUT, L2_F_OUT, L2_K,
        L2_H_POOL_OUT, L2_W_POOL_OUT, L2_F_OUT);
      $display("TB_INFO: L3 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> pool %0dx%0dx%0d",
        L3_H_IN, L3_W_IN, L3_C_IN, L3_H_CONV_OUT, L3_W_CONV_OUT, L3_F_OUT, L3_K,
        L3_H_POOL_OUT, L3_W_POOL_OUT, L3_F_OUT);
      $display("TB_INFO: L4 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> pool %0dx%0dx%0d",
        L4_H_IN, L4_W_IN, L4_C_IN, L4_H_CONV_OUT, L4_W_CONV_OUT, L4_F_OUT, L4_K,
        L4_H_POOL_OUT, L4_W_POOL_OUT, L4_F_OUT);
      $display("TB_INFO: Mode2 PC=%0d PF=%0d PTOTAL=%0d; partial F groups per layer = %0d,%0d,%0d,%0d,%0d",
        PC, PF, PTOTAL, L0_NUM_FGROUP, L1_NUM_FGROUP, L2_NUM_FGROUP, L3_NUM_FGROUP, L4_NUM_FGROUP);
      $display("TB_INFO: expected DDR->IFM reads=%0d", L0_C_IN*L0_H_IN);
      $display("TB_INFO: expected WGT reads=%0d", L0_WGT_WORDS+L1_WGT_WORDS+L2_WGT_WORDS+L3_WGT_WORDS+L4_WGT_WORDS);
      $display("TB_INFO: expected OFM->IFM M2 stream commands >= %0d", EXP_OFM2IFM_STREAMS);
      $display("TB_INFO: expected final OFM DDR words=%0d, each lane0 value=%0d", EXPECTED_OFM_DDR_WORDS, EXPECTED_FINAL_VALUE);
    end
  endtask

  // --------------------------------------------------------------------------
  // Stimulus
  // --------------------------------------------------------------------------
  initial begin
    rst_n = 1'b0;
    start = 1'b0;
    abort = 1'b0;
    cfg_wr_en = 1'b0;
    cfg_wr_addr = '0;
    cfg_wr_data = '0;
    cfg_num_layers = 5;

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
      $display("TB_FAIL: DUT asserted error. first_error_vec=%04b layer=%0d mode=%0d first_error_cycle=%0d",
        first_error_vec, first_error_layer, first_error_mode, first_error_cycle);
      $display("TB_ERROR_DECODE: bit0=dma_error bit1=ofm_error bit2=local_error bit3=transition_error");
      $display("DDR counts at stop: ifm_reads=%0d wgt_reads=%0d ofm_writes=%0d done_seen=%0b busy=%0b",
        ddr_ifm_read_count, ddr_wgt_read_count, ddr_ofm_write_count, done_seen, busy);
      $display("Stream counts at stop: start=%0d done=%0d m2_req=%0d expected>=%0d",
        ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, m2_sm_refill_req_count, EXP_OFM2IFM_STREAMS);
      dump_ofm_region();
      $fatal(1, "TB_FAIL: DUT error before successful completion");
    end

    if (cycle_count > MAX_CYCLES) begin
      $display("TB_FAIL: timeout after %0d cycles. busy=%0b done=%0b error=%0b layer=%0d mode=%0d vec=%04b",
        cycle_count, busy, done, error, dbg_layer_idx, dbg_mode, dbg_error_vec);
      $display("DDR counts at timeout: ifm_reads=%0d wgt_reads=%0d ofm_writes=%0d",
        ddr_ifm_read_count, ddr_wgt_read_count, ddr_ofm_write_count);
      $display("Stream counts at timeout: start=%0d done=%0d m2_req=%0d expected>=%0d",
        ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, m2_sm_refill_req_count, EXP_OFM2IFM_STREAMS);
      dump_ofm_region();
      $fatal(1, "TB_FAIL: timeout");
    end

    if (!done_seen) begin
      $fatal(1, "TB_FAIL: stopped without done_seen");
    end

    $display("TB_INFO: 5-layer Mode2 done after %0d cycles", cycle_count);
    $display("TB_INFO: DDR counts: ifm_reads=%0d expected=%0d, wgt_reads=%0d expected=%0d, ofm_writes=%0d expected=%0d",
      ddr_ifm_read_count, L0_C_IN*L0_H_IN,
      ddr_wgt_read_count, L0_WGT_WORDS+L1_WGT_WORDS+L2_WGT_WORDS+L3_WGT_WORDS+L4_WGT_WORDS,
      ddr_ofm_write_count, EXPECTED_OFM_DDR_WORDS);
    $display("TB_INFO: OFM->IFM stream starts=%0d done=%0d expected>=%0d, m2_refill_req=%0d",
      ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, EXP_OFM2IFM_STREAMS, m2_sm_refill_req_count);

    if (ddr_ifm_read_count != (L0_C_IN*L0_H_IN)) begin
      $fatal(1, "TB_FAIL: unexpected IFM DDR read count");
    end
    if (ddr_wgt_read_count != (L0_WGT_WORDS+L1_WGT_WORDS+L2_WGT_WORDS+L3_WGT_WORDS+L4_WGT_WORDS)) begin
      $fatal(1, "TB_FAIL: unexpected WGT DDR read count");
    end
    if (ddr_ofm_write_count != EXPECTED_OFM_DDR_WORDS) begin
      dump_ofm_region();
      $fatal(1, "TB_FAIL: unexpected OFM DDR write count");
    end
    if (ofm_ifm_stream_done_count < EXP_OFM2IFM_STREAMS) begin
      $fatal(1, "TB_FAIL: insufficient OFM->IFM stream done count");
    end

    check_final_ofm();

    $display("TB_PASS: 5-layer Mode2 partial-group tiled stress passed. final=%0d words=%0d", EXPECTED_FINAL_VALUE, EXPECTED_OFM_DDR_WORDS);
    $finish;
  end

endmodule
