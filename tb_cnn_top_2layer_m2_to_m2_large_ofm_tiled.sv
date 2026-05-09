`timescale 1ns/1ps
`include "cnn_ddr_defs.svh"

module tb_cnn_top_2layer_m2_to_m2_large_ofm_tiled;
  import cnn_layer_desc_pkg::*;

  // --------------------------------------------------------------------------
  // Purpose
  // --------------------------------------------------------------------------
  // Simple 2-layer Mode2->Mode2 test, analogous in intent to
  // tb_cnn_top_2layer_m1_to_m1_large_ofm_tiled.sv:
  //   - L0 is first/current layer loaded from DDR.
  //   - L0 OFM is larger than a trivial 1x1 case and must be streamed into IFM
  //     for L1 through the same-mode M2 OFM->IFM refill path.
  //   - L1 is final and stores final OFM to DDR.
  //
  // The arithmetic is intentionally simple and exact:
  //   IFM = 1, all weights = 1.
  //   L0: C=4, K=1 => conv result = 4, pooled result = 4.
  //   L1: C=4, K=1 with input 4 => conv result = 16, pooled result = 16.
  //
  // This test stresses Mode2 same-mode refill without involving Mode1 logic.

  localparam int DATA_W   = 8;
  localparam int PSUM_W   = 32;
  localparam int PC       = 4;
  localparam int PF       = 2;
  localparam int PTOTAL   = PC * PF;      // 8 lanes
  localparam int PV_MAX   = PTOTAL;       // DDR word = 8 bytes in this test
  localparam int PF_MAX   = PF;
  localparam int C_MAX    = 4;
  localparam int F_MAX    = 4;
  localparam int H_MAX    = 12;
  localparam int W_MAX    = 4;
  localparam int HT       = 4;
  localparam int K_MAX    = 3;
  localparam int WGT_DEPTH = 64;

  // Row-stride OFM layout. W_MAX=4 and PC=4, so one physical word/row is
  // enough for Mode2 source/destination in this test. Use a conservative small
  // stride to keep the test light while still using full-FM OFM storage.
  localparam int OFM_ROW_STRIDE   = 4;
  localparam int OFM_BANK_DEPTH   = H_MAX * OFM_ROW_STRIDE;
  localparam int OFM_LINEAR_DEPTH = C_MAX * OFM_BANK_DEPTH;

  localparam int CFG_DEPTH = 4;
  localparam int DDR_ADDR_W = `CNN_DDR_ADDR_W;
  localparam int DDR_WORD_W = PV_MAX * DATA_W;
  localparam int MEM_DEPTH  = (`DDR_RSVD_BASE + `DDR_RSVD_SIZE);
  localparam int CLK_PERIOD_NS = 10;
  localparam int MAX_CYCLES = 200000;

  // L0: 12x4x4 -> conv1x1 12x4x4 -> pool2x2 6x2x4
  localparam int L0_H_IN  = 12;
  localparam int L0_W_IN  = 4;
  localparam int L0_C_IN  = 4;
  localparam int L0_F_OUT = 4;
  localparam int L0_K     = 1;
  localparam int L0_H_CONV_OUT = L0_H_IN - L0_K + 1;
  localparam int L0_W_CONV_OUT = L0_W_IN - L0_K + 1;
  localparam int L0_H_POOL_OUT = L0_H_CONV_OUT / 2;
  localparam int L0_W_POOL_OUT = L0_W_CONV_OUT / 2;

  // L1: 6x2x4 -> conv1x1 6x2x2 -> pool2x2 3x1x2
  localparam int L1_H_IN  = L0_H_POOL_OUT;
  localparam int L1_W_IN  = L0_W_POOL_OUT;
  localparam int L1_C_IN  = L0_F_OUT;
  localparam int L1_F_OUT = 2;
  localparam int L1_K     = 1;
  localparam int L1_H_CONV_OUT = L1_H_IN - L1_K + 1;
  localparam int L1_W_CONV_OUT = L1_W_IN - L1_K + 1;
  localparam int L1_H_POOL_OUT = L1_H_CONV_OUT / 2;
  localparam int L1_W_POOL_OUT = L1_W_CONV_OUT / 2;

  localparam int L0_NUM_CGROUP = (L0_C_IN  + PC - 1) / PC;
  localparam int L0_NUM_FGROUP = (L0_F_OUT + PF - 1) / PF;
  localparam int L1_NUM_CGROUP = (L1_C_IN  + PC - 1) / PC;
  localparam int L1_NUM_FGROUP = (L1_F_OUT + PF - 1) / PF;

  localparam int L0_WGT_WORDS = L0_NUM_FGROUP * L0_NUM_CGROUP * L0_K * L0_K;
  localparam int L1_WGT_WORDS = L1_NUM_FGROUP * L1_NUM_CGROUP * L1_K * L1_K;
  localparam int L0_WGT_DDR_BASE = `DDR_WGT_BASE;
  localparam int L1_WGT_DDR_BASE = `DDR_WGT_BASE + L0_WGT_WORDS;

  // L0 OFM becomes L1 IFM. One M2 stream command is expected per
  // (row, col, cgrp) with col in global pixel coordinates.
  localparam int L1_NUM_CGROUP = (L1_C_IN + PC - 1) / PC;
  localparam int EXP_OFM2IFM_STREAMS = L1_H_IN * L1_W_IN * L1_NUM_CGROUP;

  // Final OFM readback uses one DDR word per output channel/filter per pixel.
  localparam int EXPECTED_OFM_DDR_WORDS = L1_F_OUT * L1_H_POOL_OUT * L1_W_POOL_OUT;

  // --------------------------------------------------------------------------
  // DUT I/O
  // --------------------------------------------------------------------------
  logic clk;
  logic rst_n;
  logic start;
  logic abort;

  logic cfg_wr_en;
  logic [$clog2(CFG_DEPTH)-1:0] cfg_wr_addr;
  layer_desc_t cfg_wr_data;
  logic [$clog2(CFG_DEPTH+1)-1:0] cfg_num_layers;

  logic ddr_rd_req;
  logic [DDR_ADDR_W-1:0] ddr_rd_addr;
  logic ddr_rd_valid;
  logic [DDR_WORD_W-1:0] ddr_rd_data;
  logic ddr_wr_en;
  logic [DDR_ADDR_W-1:0] ddr_wr_addr;
  logic [DDR_WORD_W-1:0] ddr_wr_data;
  logic [(DDR_WORD_W/8)-1:0] ddr_wr_be;

  logic [15:0] m1_free_col_blk_g;
  logic [15:0] m1_free_ch_blk_g;
  logic m1_sm_refill_req_ready;
  logic m1_sm_refill_req_valid;
  logic [$clog2(HT)-1:0] m1_sm_refill_row_slot_l;
  logic [15:0] m1_sm_refill_row_g;
  logic [15:0] m1_sm_refill_col_blk_g;
  logic [15:0] m1_sm_refill_ch_blk_g;

  logic m2_sm_refill_req_ready;
  logic m2_sm_refill_req_valid;
  logic [15:0] m2_sm_refill_row_g;
  logic [15:0] m2_sm_refill_col_g;
  logic [15:0] m2_sm_refill_col_l;
  logic [15:0] m2_sm_refill_cgrp_g;

  logic ifm_m1_free_valid;
  logic [$clog2(HT)-1:0] ifm_m1_free_row_slot_l;
  logic [15:0] ifm_m1_free_row_g;

  logic busy;
  logic done;
  logic error;
  logic [$clog2(CFG_DEPTH)-1:0] dbg_layer_idx;
  logic dbg_mode;
  logic dbg_weight_bank;
  logic [3:0] dbg_error_vec;

  // --------------------------------------------------------------------------
  // Simple DDR model and counters
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
  integer i;
  integer b;

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

  // --------------------------------------------------------------------------
  // Clock
  // --------------------------------------------------------------------------
  initial clk = 1'b0;
  always #(CLK_PERIOD_NS/2) clk = ~clk;

  // --------------------------------------------------------------------------
  // Single-driver DDR model + visibility counters
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

      if (done)
        done_seen <= 1'b1;

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

  // --------------------------------------------------------------------------
  // Optional light status monitor. It prints only major events.
  // --------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (rst_n) begin
      if (dut.ofm_ifm_stream_start_s || dut.ofm_ifm_stream_done_s || done || error) begin
        $display("DBG_TOP_STATUS t=%0t cycle=%0d start=%0b busy=%0b done=%0b error=%0b vec=%04b layer=%0d mode=%0d ifm_rd=%0d wgt_rd=%0d ofm_wr=%0d stream_start=%0d stream_done=%0d m2_req=%0d",
          $time, cycle_count, start, busy, done, error, dbg_error_vec, dbg_layer_idx, dbg_mode,
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

      // Mode 2 IFM DDR layout used by cnn_dma_direct:
      // flat order over channel then row; one word contains W<=PC pixels.
      word_idx = 0;
      for (ch = 0; ch < L0_C_IN; ch = ch + 1) begin
        for (row = 0; row < L0_H_IN; row = row + 1) begin
          ddr_mem[`DDR_IFM_BASE + word_idx] = pack_pc_ones_word();
          word_idx = word_idx + 1;
        end
      end

      // Mode 2 weight DDR layout: one physical PTOTAL-lane word per
      // (filter group, channel group, ky, kx). All weights = 1.
      for (i = 0; i < L0_WGT_WORDS; i = i + 1) begin
        ddr_mem[L0_WGT_DDR_BASE + i] = pack_ptotal_ones_word();
      end
      for (i = 0; i < L1_WGT_WORDS; i = i + 1) begin
        ddr_mem[L1_WGT_DDR_BASE + i] = pack_ptotal_ones_word();
      end
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

  task automatic program_layers;
    layer_desc_t cfg0;
    layer_desc_t cfg1;
    begin
      cfg0 = '0;
      cfg0.layer_id = 0;
      cfg0.mode = MODE2;
      cfg0.h_in = L0_H_IN;
      cfg0.w_in = L0_W_IN;
      cfg0.c_in = L0_C_IN;
      cfg0.f_out = L0_F_OUT;
      cfg0.k = L0_K;
      cfg0.h_out = L0_H_CONV_OUT;
      cfg0.w_out = L0_W_CONV_OUT;
      cfg0.pv_m1 = PV_MAX;
      cfg0.pf_m1 = PF;
      cfg0.pc_m2 = PC;
      cfg0.pf_m2 = PF;
      cfg0.conv_stride = 1;
      cfg0.pad_top = 0;
      cfg0.pad_bottom = 0;
      cfg0.pad_left = 0;
      cfg0.pad_right = 0;
      cfg0.relu_en = 1'b1;
      cfg0.pool_en = 1'b1;
      cfg0.pool_k = 2;
      cfg0.pool_stride = 2;
      cfg0.ifm_ddr_base = `DDR_IFM_BASE;
      cfg0.wgt_ddr_base = L0_WGT_DDR_BASE;
      cfg0.ofm_ddr_base = `DDR_OFM_BASE;
      cfg0.first_layer = 1'b1;
      cfg0.last_layer = 1'b0;

      cfg1 = '0;
      cfg1.layer_id = 1;
      cfg1.mode = MODE2;
      cfg1.h_in = L1_H_IN;
      cfg1.w_in = L1_W_IN;
      cfg1.c_in = L1_C_IN;
      cfg1.f_out = L1_F_OUT;
      cfg1.k = L1_K;
      cfg1.h_out = L1_H_CONV_OUT;
      cfg1.w_out = L1_W_CONV_OUT;
      cfg1.pv_m1 = PV_MAX;
      cfg1.pf_m1 = PF;
      cfg1.pc_m2 = PC;
      cfg1.pf_m2 = PF;
      cfg1.conv_stride = 1;
      cfg1.pad_top = 0;
      cfg1.pad_bottom = 0;
      cfg1.pad_left = 0;
      cfg1.pad_right = 0;
      cfg1.relu_en = 1'b1;
      cfg1.pool_en = 1'b1;
      cfg1.pool_k = 2;
      cfg1.pool_stride = 2;
      cfg1.ifm_ddr_base = `DDR_IFM_BASE;
      cfg1.wgt_ddr_base = L1_WGT_DDR_BASE;
      cfg1.ofm_ddr_base = `DDR_OFM_BASE;
      cfg1.first_layer = 1'b0;
      cfg1.last_layer = 1'b1;

      write_cfg('0, cfg0);
      write_cfg(1, cfg1);
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
      $display("--- OFM DDR region dump, first %0d expected words ---", EXPECTED_OFM_DDR_WORDS);
      for (j = 0; j < 16; j = j + 1) begin
        $display("OFM[%0d] @0x%05h = 0x%016h", j, (`DDR_OFM_BASE + j), ddr_mem[`DDR_OFM_BASE + j]);
      end
    end
  endtask

  task automatic check_final_ofm;
    int mismatch;
    logic [DDR_WORD_W-1:0] exp_word;
    begin
      mismatch = 0;
      exp_word = expected_final_word(16);

      for (int j = 0; j < EXPECTED_OFM_DDR_WORDS; j = j + 1) begin
        if (ddr_mem[`DDR_OFM_BASE + j] !== exp_word) begin
          if (mismatch < 20) begin
            $display("TB_MISMATCH_M2_2L word=%0d got=0x%016h exp=0x%016h",
                     j, ddr_mem[`DDR_OFM_BASE + j], exp_word);
          end
          mismatch = mismatch + 1;
        end
      end

      if (mismatch != 0) begin
        dump_ofm_region();
        $fatal(1, "TB_FAIL: 2-layer Mode2 final OFM mismatch count=%0d", mismatch);
      end
    end
  endtask

  task automatic print_banner;
    begin
      $display("TB_INFO: 2-layer Mode2->Mode2 large OFM tiled/refill test");
      $display("TB_INFO: L0 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> pool %0dx%0dx%0d",
        L0_H_IN, L0_W_IN, L0_C_IN, L0_H_CONV_OUT, L0_W_CONV_OUT, L0_F_OUT, L0_K,
        L0_H_POOL_OUT, L0_W_POOL_OUT, L0_F_OUT);
      $display("TB_INFO: L1 %0dx%0dx%0d -> conv %0dx%0dx%0d K=%0d -> pool %0dx%0dx%0d",
        L1_H_IN, L1_W_IN, L1_C_IN, L1_H_CONV_OUT, L1_W_CONV_OUT, L1_F_OUT, L1_K,
        L1_H_POOL_OUT, L1_W_POOL_OUT, L1_F_OUT);
      $display("TB_INFO: Mode2 PC=%0d PF=%0d PTOTAL=%0d", PC, PF, PTOTAL);
      $display("TB_INFO: L0 weight words=%0d base=0x%05h; L1 weight words=%0d base=0x%05h",
        L0_WGT_WORDS, L0_WGT_DDR_BASE, L1_WGT_WORDS, L1_WGT_DDR_BASE);
      $display("TB_INFO: expected OFM->IFM M2 stream commands >= %0d", EXP_OFM2IFM_STREAMS);
      $display("TB_INFO: expected final OFM DDR words=%0d, each lane0 value=16", EXPECTED_OFM_DDR_WORDS);
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
    cfg_num_layers = 2;

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
      $display("Stream counts at stop: start=%0d done=%0d m2_req=%0d",
               ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, m2_sm_refill_req_count);
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

    $display("TB_INFO: 2-layer Mode2 done after %0d cycles", cycle_count);
    $display("TB_INFO: DDR counts: ifm_reads=%0d expected=%0d, wgt_reads=%0d expected=%0d, ofm_writes=%0d expected=%0d",
             ddr_ifm_read_count, L0_C_IN*L0_H_IN,
             ddr_wgt_read_count, L0_WGT_WORDS + L1_WGT_WORDS,
             ddr_ofm_write_count, EXPECTED_OFM_DDR_WORDS);
    $display("TB_INFO: OFM->IFM stream starts=%0d done=%0d expected>=%0d, m2_refill_req=%0d",
             ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, EXP_OFM2IFM_STREAMS, m2_sm_refill_req_count);

    if (ddr_ifm_read_count != (L0_C_IN*L0_H_IN)) begin
      $fatal(1, "TB_FAIL: unexpected IFM DDR read count");
    end
    if (ddr_wgt_read_count != (L0_WGT_WORDS + L1_WGT_WORDS)) begin
      $fatal(1, "TB_FAIL: unexpected WGT DDR read count");
    end
    if (ddr_ofm_write_count != EXPECTED_OFM_DDR_WORDS) begin
      dump_ofm_region();
      $fatal(1, "TB_FAIL: unexpected OFM DDR write count");
    end
    if (ofm_ifm_stream_start_count < EXP_OFM2IFM_STREAMS ||
        ofm_ifm_stream_done_count  < EXP_OFM2IFM_STREAMS) begin
      $fatal(1, "TB_FAIL: Mode2 OFM->IFM refill path was not fully exercised");
    end

    check_final_ofm();
    dump_ofm_region();
    $display("TB_PASS: 2-layer Mode2->Mode2 tiled/refill test passed with exact final OFM check");
    $finish;
  end
endmodule
