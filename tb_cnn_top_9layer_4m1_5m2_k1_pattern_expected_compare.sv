`timescale 1ns/1ps
`include "cnn_ddr_defs.svh"

module tb_cnn_top_9layer_4m1_5m2_k1_pattern_expected_compare;
  import cnn_layer_desc_pkg::*;

  // --------------------------------------------------------------------------
  // 9-layer mixed regression: first 4 layers Mode1, last 5 layers Mode2.
  //
  // Purpose:
  //   - Exercise M1->M1 same-mode streaming for several consecutive layers.
  //   - Exercise one M1->M2 transition after L3 completes and OFM is fully written.
  //   - Exercise M2->M2 rolling refill for several consecutive layers.
  //   - Use non-uniform input and sparse K=3 weights so final OFM values are
  //     different across row/column/channel instead of saturating to 127.
  //
  // Shape/data model:
  //   L0 uses RGB-like C=3.  Later layers use practical partial channel/filter
  //   counts below PC/PF to stress tail masks and partial groups without forcing
  //   saturation.  K=3 is still used in every layer.  Each layer has exactly one
  //   non-zero tap per output filter: source channel = filter_id % C_in, and
  //   a layer-specific (ky,kx) tap.  This keeps the golden result simple while
  //   still exercising K=3 address generation.
  //
  //   final(row,col,f) = input(row + SUM_KY, col + SUM_KX, traced_source_ch(f))
  // --------------------------------------------------------------------------

  localparam int DATA_W = 8;
  localparam int PSUM_W = 32;

  localparam int PC = 32;
  localparam int PF = 64;
  localparam int PTOTAL = PC * PF;

  localparam int PV_M1 = 32;
  localparam int PF_M1 = 64;

  localparam int PV_MAX = 128;
  localparam int PF_MAX = 128;

  localparam int C_MAX = 64;
  localparam int F_MAX = 64;
  localparam int H_MAX = 64;
  localparam int W_MAX = 96;
  localparam int HT = 4;
  localparam int K_MAX = 3;

  localparam int WGT_DEPTH = 512;
  // M1->M2 no-pool transition stores Mode1 OFM with store_pack=1,
  // so each row may need one OFM group per spatial column.
  localparam int OFM_ROW_STRIDE = W_MAX;
  localparam int OFM_BANK_DEPTH = H_MAX * OFM_ROW_STRIDE;
  localparam int OFM_LINEAR_DEPTH = C_MAX * OFM_BANK_DEPTH;
  localparam int CFG_DEPTH = 16;

  localparam int DDR_ADDR_W = `CNN_DDR_ADDR_W;
  localparam int DDR_WORD_W = PV_MAX * DATA_W;
  localparam int DDR_LANES  = PV_MAX;
  localparam int WGT_SUBWORDS = (PTOTAL + PV_MAX - 1) / PV_MAX;
  localparam int MEM_DEPTH = (`DDR_RSVD_BASE + `DDR_RSVD_SIZE);
  localparam int CLK_PERIOD_NS = 10;
  localparam int MAX_CYCLES = 20000000;

  localparam int LK = 3;
  localparam int LPOOL = 0;

  // Layer spatial trend for K=3, stride=1, no padding, no pooling.
  // Channel/filter trend: 3 -> 8 -> 8 -> 16 -> 16 -> 24 -> 24 -> 32 -> 32 -> 16.
  localparam int L0_H_IN=64, L0_W_IN=96, L0_C_IN=3,  L0_F_OUT=8,  L0_K=LK, L0_POOL_EN=LPOOL;
  localparam int L0_H_CONV_OUT=62, L0_W_CONV_OUT=94, L0_H_OUT=62, L0_W_OUT=94;
  localparam int L1_H_IN=62, L1_W_IN=94, L1_C_IN=L0_F_OUT, L1_F_OUT=8,  L1_K=LK, L1_POOL_EN=LPOOL;
  localparam int L1_H_CONV_OUT=60, L1_W_CONV_OUT=92, L1_H_OUT=60, L1_W_OUT=92;
  localparam int L2_H_IN=60, L2_W_IN=92, L2_C_IN=L1_F_OUT, L2_F_OUT=16, L2_K=LK, L2_POOL_EN=LPOOL;
  localparam int L2_H_CONV_OUT=58, L2_W_CONV_OUT=90, L2_H_OUT=58, L2_W_OUT=90;
  localparam int L3_H_IN=58, L3_W_IN=90, L3_C_IN=L2_F_OUT, L3_F_OUT=16, L3_K=LK, L3_POOL_EN=LPOOL;
  localparam int L3_H_CONV_OUT=56, L3_W_CONV_OUT=88, L3_H_OUT=56, L3_W_OUT=88;
  localparam int L4_H_IN=56, L4_W_IN=88, L4_C_IN=L3_F_OUT, L4_F_OUT=24, L4_K=LK, L4_POOL_EN=LPOOL;
  localparam int L4_H_CONV_OUT=54, L4_W_CONV_OUT=86, L4_H_OUT=54, L4_W_OUT=86;
  localparam int L5_H_IN=54, L5_W_IN=86, L5_C_IN=L4_F_OUT, L5_F_OUT=24, L5_K=LK, L5_POOL_EN=LPOOL;
  localparam int L5_H_CONV_OUT=52, L5_W_CONV_OUT=84, L5_H_OUT=52, L5_W_OUT=84;
  localparam int L6_H_IN=52, L6_W_IN=84, L6_C_IN=L5_F_OUT, L6_F_OUT=32, L6_K=LK, L6_POOL_EN=LPOOL;
  localparam int L6_H_CONV_OUT=50, L6_W_CONV_OUT=82, L6_H_OUT=50, L6_W_OUT=82;
  localparam int L7_H_IN=50, L7_W_IN=82, L7_C_IN=L6_F_OUT, L7_F_OUT=32, L7_K=LK, L7_POOL_EN=LPOOL;
  localparam int L7_H_CONV_OUT=48, L7_W_CONV_OUT=80, L7_H_OUT=48, L7_W_OUT=80;
  localparam int L8_H_IN=48, L8_W_IN=80, L8_C_IN=L7_F_OUT, L8_F_OUT=16, L8_K=LK, L8_POOL_EN=LPOOL;
  localparam int L8_H_CONV_OUT=46, L8_W_CONV_OUT=78, L8_H_OUT=46, L8_W_OUT=78;

  localparam int SUM_KY = 9;
  localparam int SUM_KX = 8;

  localparam int L0_NUM_CGROUP=(L0_C_IN+PV_M1-1)/PV_M1, L0_NUM_FGROUP=(L0_F_OUT+PF_M1-1)/PF_M1;
  localparam int L1_NUM_CGROUP=(L1_C_IN+PV_M1-1)/PV_M1, L1_NUM_FGROUP=(L1_F_OUT+PF_M1-1)/PF_M1;
  localparam int L2_NUM_CGROUP=(L2_C_IN+PV_M1-1)/PV_M1, L2_NUM_FGROUP=(L2_F_OUT+PF_M1-1)/PF_M1;
  localparam int L3_NUM_CGROUP=(L3_C_IN+PV_M1-1)/PV_M1, L3_NUM_FGROUP=(L3_F_OUT+PF_M1-1)/PF_M1;
  localparam int L4_NUM_CGROUP=(L4_C_IN+PC-1)/PC,     L4_NUM_FGROUP=(L4_F_OUT+PF-1)/PF;
  localparam int L5_NUM_CGROUP=(L5_C_IN+PC-1)/PC,     L5_NUM_FGROUP=(L5_F_OUT+PF-1)/PF;
  localparam int L6_NUM_CGROUP=(L6_C_IN+PC-1)/PC,     L6_NUM_FGROUP=(L6_F_OUT+PF-1)/PF;
  localparam int L7_NUM_CGROUP=(L7_C_IN+PC-1)/PC,     L7_NUM_FGROUP=(L7_F_OUT+PF-1)/PF;
  localparam int L8_NUM_CGROUP=(L8_C_IN+PC-1)/PC,     L8_NUM_FGROUP=(L8_F_OUT+PF-1)/PF;

  localparam int L0_WGT_WORDS=((L0_NUM_FGROUP*L0_C_IN*LK*LK)+PV_M1-1)/PV_M1;
  localparam int L1_WGT_WORDS=((L1_NUM_FGROUP*L1_C_IN*LK*LK)+PV_M1-1)/PV_M1;
  localparam int L2_WGT_WORDS=((L2_NUM_FGROUP*L2_C_IN*LK*LK)+PV_M1-1)/PV_M1;
  localparam int L3_WGT_WORDS=((L3_NUM_FGROUP*L3_C_IN*LK*LK)+PV_M1-1)/PV_M1;
  localparam int L4_WGT_WORDS=L4_NUM_FGROUP*L4_NUM_CGROUP*LK*LK;
  localparam int L5_WGT_WORDS=L5_NUM_FGROUP*L5_NUM_CGROUP*LK*LK;
  localparam int L6_WGT_WORDS=L6_NUM_FGROUP*L6_NUM_CGROUP*LK*LK;
  localparam int L7_WGT_WORDS=L7_NUM_FGROUP*L7_NUM_CGROUP*LK*LK;
  localparam int L8_WGT_WORDS=L8_NUM_FGROUP*L8_NUM_CGROUP*LK*LK;

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
  localparam int L8_STORED_GROUPS=(L8_W_OUT+PC-1)/PC;
  localparam int EXPECTED_IFM_DDR_READS = L0_COLBLKS * L0_H_IN * L0_C_IN;
  localparam int EXPECTED_WGT_DDR_READS = L0_WGT_DDR_WORDS+L1_WGT_DDR_WORDS+L2_WGT_DDR_WORDS+L3_WGT_DDR_WORDS+L4_WGT_DDR_WORDS+L5_WGT_DDR_WORDS+L6_WGT_DDR_WORDS+L7_WGT_DDR_WORDS+L8_WGT_DDR_WORDS;
  localparam int EXPECTED_OFM_DDR_WORDS = L8_F_OUT * L8_H_OUT * L8_STORED_GROUPS;
  localparam int EXPECTED_FINAL_ELEMENTS = L8_F_OUT * L8_H_OUT * L8_W_OUT;
  localparam int EXPECTED_WEIGHT_PRODUCT = 0; // final checker expects saturation to 127

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

`ifndef SYNTHESIS
always_ff @(posedge clk) begin
  if (rst_n) begin
    if ((dbg_layer_idx >= 4) && (m2_sm_refill_req_valid || m2_sm_refill_req_ready)) begin
      $display("DBG_TB_M2_REFILL_REQ t=%0t cycle=%0d layer=%0d mode=%0b valid=%0b ready=%0b take=%0b row=%0d col_g=%0d col_l=%0d cgrp=%0d stream_start_cnt=%0d stream_done_cnt=%0d busy=%0b done=%0b err=%0b vec=%b", $time, cycle_count, dbg_layer_idx, dbg_mode, m2_sm_refill_req_valid, m2_sm_refill_req_ready, m2_sm_refill_req_valid && m2_sm_refill_req_ready, m2_sm_refill_row_g, m2_sm_refill_col_g, m2_sm_refill_col_l, m2_sm_refill_cgrp_g, ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, busy, done, error, dbg_error_vec);
    end
  end
end
`endif


task automatic dbg_tb_decode_final_word(input int word_idx, input logic [DDR_WORD_W-1:0] got_word, input logic [DDR_WORD_W-1:0] exp_word);
  int words_per_ch;
  int ch;
  int rem;
  int row;
  int grp;
  int col_g;
  int src_row;
  int src_col;
  int src_ch;
  logic [DATA_W-1:0] got_b;
  logic [DATA_W-1:0] exp_b;
begin
  words_per_ch = L8_H_OUT * L8_STORED_GROUPS;
  ch = word_idx / words_per_ch;
  rem = word_idx % words_per_ch;
  row = rem / L8_STORED_GROUPS;
  grp = rem % L8_STORED_GROUPS;

  $display("DBG_TB_FINAL_WORD_DECODE word=%0d ch=%0d row=%0d grp=%0d got=%h exp=%h words_per_ch=%0d", word_idx, ch, row, grp, got_word, exp_word, words_per_ch);

  for (int lane = 0; lane < DDR_LANES; lane++) begin
    col_g = grp * PC + lane;
    got_b = got_word[lane*DATA_W +: DATA_W];
    exp_b = exp_word[lane*DATA_W +: DATA_W];
    src_row = row + SUM_KY;
    src_col = col_g + SUM_KX;
    src_ch = trace_final_to_input_ch(ch);

    if ((lane < PC) && (col_g < L8_W_OUT)) begin
      $display("DBG_TB_FINAL_LANE word=%0d lane=%0d ch=%0d row=%0d col=%0d got=%0d exp=%0d src_row=%0d src_col=%0d src_ch=%0d match=%0b", word_idx, lane, ch, row, col_g, got_b, exp_b, src_row, src_col, src_ch, (got_b === exp_b));
    end
  end
end
endtask

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

  function automatic bit layer_is_mode2(input int layer_id);
    begin
      layer_is_mode2 = (layer_id >= 4);
    end
  endfunction

  // One active K=3 tap per layer.  These offsets intentionally vary so the
  // expected output checks true K=3 address selection, not only center taps.
  function automatic int layer_active_ky(input int layer_id);
    begin
      case (layer_id)
        0: layer_active_ky = 1;
        1: layer_active_ky = 0;
        2: layer_active_ky = 2;
        3: layer_active_ky = 1;
        4: layer_active_ky = 0;
        5: layer_active_ky = 2;
        6: layer_active_ky = 1;
        7: layer_active_ky = 0;
        8: layer_active_ky = 2;
        default: layer_active_ky = 1;
      endcase
    end
  endfunction

  function automatic int layer_active_kx(input int layer_id);
    begin
      case (layer_id)
        0: layer_active_kx = 1;
        1: layer_active_kx = 2;
        2: layer_active_kx = 0;
        3: layer_active_kx = 1;
        4: layer_active_kx = 1;
        5: layer_active_kx = 2;
        6: layer_active_kx = 0;
        7: layer_active_kx = 0;
        default: layer_active_kx = 1;
      endcase
    end
  endfunction

  function automatic int input_pattern_val(input int row, input int col, input int ch);
    begin
      input_pattern_val = 1 + ((3*row + 5*col + 7*ch) % 11);
    end
  endfunction


  function automatic logic [DDR_WORD_W-1:0] pack_m1_ifm_word(
    input int ch,
    input int row,
    input int col_blk
  );
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

  function automatic logic [DATA_W-1:0] sat8(input int val);
    begin
      if (val < 0) sat8 = '0;
      else if (val > 127) sat8 = 8'd127;
      else sat8 = val[DATA_W-1:0];
    end
  endfunction

  function automatic int trace_final_to_input_ch(input int final_ch);
    int ch;
    begin
      ch = final_ch;
      for (int layer_id = 8; layer_id >= 0; layer_id--) begin
        ch = ch % layer_c_in(layer_id);
      end
      trace_final_to_input_ch = ch;
    end
  endfunction

  function automatic logic [DATA_W-1:0] expected_final_byte(input int row, input int col, input int out_ch);
    int src_row;
    int src_col;
    int src_ch;
    begin
      src_row = row + SUM_KY;
      src_col = col + SUM_KX;
      src_ch  = trace_final_to_input_ch(out_ch);
      expected_final_byte = sat8(input_pattern_val(src_row, src_col, src_ch));
    end
  endfunction

  function automatic logic signed [DATA_W-1:0] sparse_weight_value(
    input int layer_id,
    input int phys_word,
    input int subword,
    input int lane_in_ddr
  );
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
        pf     = lane % PF_M1;
        tmp    = bundle;
        kx     = tmp % LK; tmp = tmp / LK;
        ky     = tmp % LK; tmp = tmp / LK;
        cidx   = tmp % C;  tmp = tmp / C;
        fgroup = tmp;
        fout   = fgroup * PF_M1 + pf;
        if ((fout < F) && (ky == layer_active_ky(layer_id)) &&
            (kx == layer_active_kx(layer_id)) && (cidx == (fout % C))) begin
          sparse_weight_value = 8'sd1;
        end
      end
      else begin
        num_cgrp = (C + PC - 1) / PC;
        tmp      = phys_word;
        kx       = tmp % LK; tmp = tmp / LK;
        ky       = tmp % LK; tmp = tmp / LK;
        cgrp     = tmp % num_cgrp; tmp = tmp / num_cgrp;
        fgroup   = tmp;
        pf       = lane / PC;
        pc_l     = lane % PC;
        fout     = fgroup * PF + pf;
        cin      = cgrp * PC + pc_l;
        if ((fout < F) && (cin < C) && (ky == layer_active_ky(layer_id)) &&
            (kx == layer_active_kx(layer_id)) && (cin == (fout % C))) begin
          sparse_weight_value = 8'sd1;
        end
      end
    end
  endfunction

  function automatic logic [DDR_WORD_W-1:0] pack_weight_ddr_word(
    input int layer_id,
    input int phys_word,
    input int subword
  );
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

      // First-layer Mode1 IFM DDR layout: C-major, then row, then PV-wide column block.
      for (int ch = 0; ch < L0_C_IN; ch = ch + 1) begin
        for (row = 0; row < L0_H_IN; row = row + 1) begin
          for (col_blk = 0; col_blk < L0_COLBLKS; col_blk = col_blk + 1) begin
            addr = `DDR_IFM_BASE + ch * L0_H_IN * L0_COLBLKS + row * L0_COLBLKS + col_blk;
            ddr_mem[addr] = pack_m1_ifm_word(ch, row, col_blk);
          end
        end
      end

      for (i = 0; i < L0_WGT_DDR_WORDS; i = i + 1) ddr_mem[L0_WGT_DDR_BASE + i] = pack_weight_ddr_word(0, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (i = 0; i < L1_WGT_DDR_WORDS; i = i + 1) ddr_mem[L1_WGT_DDR_BASE + i] = pack_weight_ddr_word(1, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (i = 0; i < L2_WGT_DDR_WORDS; i = i + 1) ddr_mem[L2_WGT_DDR_BASE + i] = pack_weight_ddr_word(2, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (i = 0; i < L3_WGT_DDR_WORDS; i = i + 1) ddr_mem[L3_WGT_DDR_BASE + i] = pack_weight_ddr_word(3, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (i = 0; i < L4_WGT_DDR_WORDS; i = i + 1) ddr_mem[L4_WGT_DDR_BASE + i] = pack_weight_ddr_word(4, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (i = 0; i < L5_WGT_DDR_WORDS; i = i + 1) ddr_mem[L5_WGT_DDR_BASE + i] = pack_weight_ddr_word(5, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (i = 0; i < L6_WGT_DDR_WORDS; i = i + 1) ddr_mem[L6_WGT_DDR_BASE + i] = pack_weight_ddr_word(6, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (i = 0; i < L7_WGT_DDR_WORDS; i = i + 1) ddr_mem[L7_WGT_DDR_BASE + i] = pack_weight_ddr_word(7, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (i = 0; i < L8_WGT_DDR_WORDS; i = i + 1) ddr_mem[L8_WGT_DDR_BASE + i] = pack_weight_ddr_word(8, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
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
      fill_layer_cfg(cfg0, 0, 1'b0, L0_H_IN,L0_W_IN,L0_C_IN,L0_F_OUT,L0_K,L0_H_CONV_OUT,L0_W_CONV_OUT,L0_POOL_EN,L0_WGT_DDR_BASE, 1'b1, 1'b0);
      fill_layer_cfg(cfg1, 1, 1'b0, L1_H_IN,L1_W_IN,L1_C_IN,L1_F_OUT,L1_K,L1_H_CONV_OUT,L1_W_CONV_OUT,L1_POOL_EN,L1_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg2, 2, 1'b0, L2_H_IN,L2_W_IN,L2_C_IN,L2_F_OUT,L2_K,L2_H_CONV_OUT,L2_W_CONV_OUT,L2_POOL_EN,L2_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg3, 3, 1'b0, L3_H_IN,L3_W_IN,L3_C_IN,L3_F_OUT,L3_K,L3_H_CONV_OUT,L3_W_CONV_OUT,L3_POOL_EN,L3_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg4, 4, 1'b1, L4_H_IN,L4_W_IN,L4_C_IN,L4_F_OUT,L4_K,L4_H_CONV_OUT,L4_W_CONV_OUT,L4_POOL_EN,L4_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg5, 5, 1'b1, L5_H_IN,L5_W_IN,L5_C_IN,L5_F_OUT,L5_K,L5_H_CONV_OUT,L5_W_CONV_OUT,L5_POOL_EN,L5_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg6, 6, 1'b1, L6_H_IN,L6_W_IN,L6_C_IN,L6_F_OUT,L6_K,L6_H_CONV_OUT,L6_W_CONV_OUT,L6_POOL_EN,L6_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg7, 7, 1'b1, L7_H_IN,L7_W_IN,L7_C_IN,L7_F_OUT,L7_K,L7_H_CONV_OUT,L7_W_CONV_OUT,L7_POOL_EN,L7_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg8, 8, 1'b1, L8_H_IN,L8_W_IN,L8_C_IN,L8_F_OUT,L8_K,L8_H_CONV_OUT,L8_W_CONV_OUT,L8_POOL_EN,L8_WGT_DDR_BASE, 1'b0, 1'b1);
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
        $display("OFM[%0d] @0x%0h = 0x%0h", j, `DDR_OFM_BASE + j, ddr_mem[`DDR_OFM_BASE + j]);
      end
    end
  endtask

  task automatic check_final_ofm;
    logic [DDR_WORD_W-1:0] got_word;
    logic [DDR_WORD_W-1:0] exp_word;
    int mismatch;
    begin
      mismatch = 0;
      for (int j = 0; j < EXPECTED_OFM_DDR_WORDS; j++) begin
        got_word = ddr_mem[`DDR_OFM_BASE + j];
        exp_word = expected_final_word(j);
        if (got_word !== exp_word) begin
          if (mismatch < 16) begin
            $display("TB_MISMATCH_FINAL word=%0d got=0x%0h exp=0x%0h", j, got_word, exp_word);
            if (j < 16) begin
                dbg_tb_decode_final_word(j, got_word, exp_word);
            end
          end
          mismatch++;
        end
      end
      if (mismatch != 0) begin
        dump_ofm_region();
        $fatal(1, "TB_FAIL: 4M1+5M2 variable-C/F K3 final OFM mismatch count=%0d", mismatch);
      end
    end
  endtask

  task automatic print_banner;
    begin
      $display("TB_INFO: 9-layer 4xMode1 + 5xMode2 variable-C/F K=3 sparse-weight expected-compare test");
      $display("TB_INFO: L0-L3 MODE1, L4-L8 MODE2, L0=%0dx%0dx%0d -> L8=%0dx%0dx%0d", L0_H_IN, L0_W_IN, L0_C_IN, L8_H_OUT, L8_W_OUT, L8_F_OUT);
      $display("TB_INFO: input(row,col,ch)=1+((3*row+5*col+7*ch)%%11), sparse K=3 weights, final expected is non-saturating and spatial/channel-varying");
      $display("TB_INFO: expected final OFM words=%0d logical elements=%0d", EXPECTED_OFM_DDR_WORDS, EXPECTED_FINAL_ELEMENTS);
      $display("TB_INFO: Mode2 begins at L4 with IFM %0dx%0d; W > PC=%0d, so rolling horizontal refill is required", L4_H_IN, L4_W_IN, PC);
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
      if (dut.ofm_ifm_stream_done_s)  ofm_ifm_stream_done_count  <= ofm_ifm_stream_done_count + 1;
      if (m2_sm_refill_req_valid && m2_sm_refill_req_ready) legacy_m2_mgr_req_count <= legacy_m2_mgr_req_count + 1;
    end
  end

  always_ff @(posedge clk) begin
    if (rst_n) begin
      if (done || error || ((cycle_count % 50000) == 0)) begin
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
      $display("DDR counts at stop: ifm_reads=%0d wgt_reads=%0d ofm_writes=%0d done_seen=%0b busy=%0b", ddr_ifm_read_count, ddr_wgt_read_count, ddr_ofm_write_count, done_seen, busy);
      $display("Stream counts at stop: start=%0d done=%0d legacy_m2_mgr_req=%0d", ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, legacy_m2_mgr_req_count);
      dump_ofm_region();
      $fatal(1, "TB_FAIL: DUT error before completion");
    end

    if (cycle_count > MAX_CYCLES) begin
      $display("TB_FAIL: timeout after %0d cycles. busy=%0b done=%0b error=%0b layer=%0d mode=%0d vec=%04b", cycle_count, busy, done, error, dbg_layer_idx, dbg_mode, dbg_error_vec);
      dump_ofm_region();
      $fatal(1, "TB_FAIL: timeout");
    end

    if (!done_seen) $fatal(1, "TB_FAIL: stopped without done_seen");

    $display("TB_INFO: 9-layer 4M1+5M2 patterned test done after %0d cycles", cycle_count);
    $display("TB_INFO: DDR counts: ifm_reads=%0d est=%0d, wgt_reads=%0d est=%0d, ofm_writes=%0d expected=%0d", ddr_ifm_read_count, EXPECTED_IFM_DDR_READS, ddr_wgt_read_count, EXPECTED_WGT_DDR_READS, ddr_ofm_write_count, EXPECTED_OFM_DDR_WORDS);
    $display("TB_INFO: OFM->IFM stream starts=%0d done=%0d legacy_m2_mgr_req=%0d", ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, legacy_m2_mgr_req_count);

    if (ddr_ofm_write_count != EXPECTED_OFM_DDR_WORDS) begin
      dump_ofm_region();
      $fatal(1, "TB_FAIL: unexpected OFM DDR write count got=%0d expected=%0d", ddr_ofm_write_count, EXPECTED_OFM_DDR_WORDS);
    end

    if (ofm_ifm_stream_start_count != ofm_ifm_stream_done_count) begin
      $fatal(1, "TB_FAIL: OFM->IFM stream start/done imbalance start=%0d done=%0d", ofm_ifm_stream_start_count, ofm_ifm_stream_done_count);
    end

    if (legacy_m2_mgr_req_count != 0) begin
      $fatal(1, "TB_FAIL: legacy Mode2 refill-manager request fired count=%0d", legacy_m2_mgr_req_count);
    end

    check_final_ofm();
    $display("TB_PASS: 9-layer 4M1+5M2 variable-C/F K3 expected-compare passed. final_words=%0d", EXPECTED_OFM_DDR_WORDS);
    $finish;
  end

endmodule
