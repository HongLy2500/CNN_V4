`timescale 1ns/1ps
`include "cnn_ddr_defs.svh"

module tb_cnn_top_vgg7_large_80x80_4m1_3m2_k3_pad1_tapsweep_expected_compare;
  import cnn_layer_desc_pkg::*;

  // --------------------------------------------------------------------------
  // Large-shape VGG7-style regression for the current CNN accelerator.
  //
  // Purpose:
  //   Stress dimensions larger than PV/PF/PC=16 in both modes:
  //     - spatial H/W > 16
  //     - Mode1 C/F groups > 1
  //     - Mode2 C/F groups > 1
  //     - Mode2 input width > PC after the M1->M2 transition
  //
  // Layer modes match the agreed benchmark split:
  //   L0-L3 : Mode1
  //   L4-L6 : Mode2
  //
  // All layers use K=3, stride=1, pad=1.  Weights are sparse but not only
  // center-tap: each output filter selects one input channel and one K3 tap;
  // the selected tap sweeps over all 9 ky/kx positions across filters/layers.
  // This stresses K3/P1 padding and row/column alignment without quickly
  // saturating the 8-bit output as all-ones 3x3 kernels would.
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
  localparam int H_MAX = 80;
  localparam int W_MAX = 80;
  localparam int HT = 4;
  localparam int K_MAX = 3;
  localparam int WGT_DEPTH = 1024;
  localparam int CFG_DEPTH = 16;

  localparam int OFM_ROW_STRIDE  = (W_MAX + PV_MAX - 1) / PV_MAX;
  localparam int OFM_BANK_DEPTH  = H_MAX * OFM_ROW_STRIDE;
  localparam int OFM_LINEAR_DEPTH = F_MAX * OFM_BANK_DEPTH;

  localparam int DDR_ADDR_W = `CNN_DDR_ADDR_W;
  localparam int DDR_WORD_W = PV_MAX * DATA_W;
  localparam int DDR_LANES  = PV_MAX;
  localparam int WGT_SUBWORDS = (PTOTAL + DDR_LANES - 1) / DDR_LANES;
  localparam int MEM_DEPTH = (1 << DDR_ADDR_W);

  localparam int TB_IFM_BASE = `DDR_IFM_BASE;
  localparam int TB_WGT_BASE = `DDR_WGT_BASE;
  localparam int TB_OFM_BASE = `DDR_OFM_BASE;

  localparam int CLK_PERIOD_NS = 10;
  localparam int MAX_CYCLES = 100000000;
  localparam int LK  = 3;
  localparam int PAD = 1;

  // --------------------------------------------------------------------------
  // Large 7-layer shape set.
  // All input/output C/F are > 16; mode2 H/W remain > 16 until final pooling.
  // h_conv_out/w_conv_out are post-convolution, pre-pooling dimensions.
  // h_out/w_out are logical layer outputs after optional pooling.
  // --------------------------------------------------------------------------
  localparam int L0_H_IN=80, L0_W_IN=80, L0_C_IN=32, L0_F_OUT=32, L0_POOL_EN=0;
  localparam int L0_H_CONV_OUT=80, L0_W_CONV_OUT=80, L0_H_OUT=80, L0_W_OUT=80;

  localparam int L1_H_IN=80, L1_W_IN=80, L1_C_IN=32, L1_F_OUT=48, L1_POOL_EN=1;
  localparam int L1_H_CONV_OUT=80, L1_W_CONV_OUT=80, L1_H_OUT=40, L1_W_OUT=40;

  localparam int L2_H_IN=40, L2_W_IN=40, L2_C_IN=48, L2_F_OUT=64, L2_POOL_EN=0;
  localparam int L2_H_CONV_OUT=40, L2_W_CONV_OUT=40, L2_H_OUT=40, L2_W_OUT=40;

  localparam int L3_H_IN=40, L3_W_IN=40, L3_C_IN=64, L3_F_OUT=64, L3_POOL_EN=1;
  localparam int L3_H_CONV_OUT=40, L3_W_CONV_OUT=40, L3_H_OUT=20, L3_W_OUT=20;

  localparam int L4_H_IN=20, L4_W_IN=20, L4_C_IN=64, L4_F_OUT=80, L4_POOL_EN=0;
  localparam int L4_H_CONV_OUT=20, L4_W_CONV_OUT=20, L4_H_OUT=20, L4_W_OUT=20;

  localparam int L5_H_IN=20, L5_W_IN=20, L5_C_IN=80, L5_F_OUT=96, L5_POOL_EN=0;
  localparam int L5_H_CONV_OUT=20, L5_W_CONV_OUT=20, L5_H_OUT=20, L5_W_OUT=20;

  localparam int L6_H_IN=20, L6_W_IN=20, L6_C_IN=96, L6_F_OUT=96, L6_POOL_EN=1;
  localparam int L6_H_CONV_OUT=20, L6_W_CONV_OUT=20, L6_H_OUT=10, L6_W_OUT=10;

  localparam int L0_NUM_CGROUP = (L0_C_IN + PV_M1 - 1) / PV_M1;
  localparam int L1_NUM_CGROUP = (L1_C_IN + PV_M1 - 1) / PV_M1;
  localparam int L2_NUM_CGROUP = (L2_C_IN + PV_M1 - 1) / PV_M1;
  localparam int L3_NUM_CGROUP = (L3_C_IN + PV_M1 - 1) / PV_M1;
  localparam int L4_NUM_CGROUP = (L4_C_IN + PC - 1) / PC;
  localparam int L5_NUM_CGROUP = (L5_C_IN + PC - 1) / PC;
  localparam int L6_NUM_CGROUP = (L6_C_IN + PC - 1) / PC;

  localparam int L0_NUM_FGROUP = (L0_F_OUT + PF_M1 - 1) / PF_M1;
  localparam int L1_NUM_FGROUP = (L1_F_OUT + PF_M1 - 1) / PF_M1;
  localparam int L2_NUM_FGROUP = (L2_F_OUT + PF_M1 - 1) / PF_M1;
  localparam int L3_NUM_FGROUP = (L3_F_OUT + PF_M1 - 1) / PF_M1;
  localparam int L4_NUM_FGROUP = (L4_F_OUT + PF - 1) / PF;
  localparam int L5_NUM_FGROUP = (L5_F_OUT + PF - 1) / PF;
  localparam int L6_NUM_FGROUP = (L6_F_OUT + PF - 1) / PF;

  // Weight-buffer physical words, before DDR subword expansion.
  localparam int L0_WGT_WORDS = ((L0_NUM_FGROUP * L0_C_IN * LK * LK) + PV_M1 - 1) / PV_M1;
  localparam int L1_WGT_WORDS = ((L1_NUM_FGROUP * L1_C_IN * LK * LK) + PV_M1 - 1) / PV_M1;
  localparam int L2_WGT_WORDS = ((L2_NUM_FGROUP * L2_C_IN * LK * LK) + PV_M1 - 1) / PV_M1;
  localparam int L3_WGT_WORDS = ((L3_NUM_FGROUP * L3_C_IN * LK * LK) + PV_M1 - 1) / PV_M1;
  localparam int L4_WGT_WORDS = L4_NUM_FGROUP * L4_NUM_CGROUP * LK * LK;
  localparam int L5_WGT_WORDS = L5_NUM_FGROUP * L5_NUM_CGROUP * LK * LK;
  localparam int L6_WGT_WORDS = L6_NUM_FGROUP * L6_NUM_CGROUP * LK * LK;

  localparam int L0_WGT_DDR_WORDS = L0_WGT_WORDS * WGT_SUBWORDS;
  localparam int L1_WGT_DDR_WORDS = L1_WGT_WORDS * WGT_SUBWORDS;
  localparam int L2_WGT_DDR_WORDS = L2_WGT_WORDS * WGT_SUBWORDS;
  localparam int L3_WGT_DDR_WORDS = L3_WGT_WORDS * WGT_SUBWORDS;
  localparam int L4_WGT_DDR_WORDS = L4_WGT_WORDS * WGT_SUBWORDS;
  localparam int L5_WGT_DDR_WORDS = L5_WGT_WORDS * WGT_SUBWORDS;
  localparam int L6_WGT_DDR_WORDS = L6_WGT_WORDS * WGT_SUBWORDS;

  localparam int L0_WGT_DDR_BASE = TB_WGT_BASE;
  localparam int L1_WGT_DDR_BASE = L0_WGT_DDR_BASE + L0_WGT_DDR_WORDS;
  localparam int L2_WGT_DDR_BASE = L1_WGT_DDR_BASE + L1_WGT_DDR_WORDS;
  localparam int L3_WGT_DDR_BASE = L2_WGT_DDR_BASE + L2_WGT_DDR_WORDS;
  localparam int L4_WGT_DDR_BASE = L3_WGT_DDR_BASE + L3_WGT_DDR_WORDS;
  localparam int L5_WGT_DDR_BASE = L4_WGT_DDR_BASE + L4_WGT_DDR_WORDS;
  localparam int L6_WGT_DDR_BASE = L5_WGT_DDR_BASE + L5_WGT_DDR_WORDS;
  localparam int WGT_DDR_END = L6_WGT_DDR_BASE + L6_WGT_DDR_WORDS;

  localparam int L0_COLBLKS = (L0_W_IN + PV_M1 - 1) / PV_M1;
  localparam int L6_STORED_GROUPS = (L6_W_OUT + PC - 1) / PC;

  localparam int EXPECTED_IFM_DDR_READS = L0_C_IN * L0_H_IN * L0_COLBLKS;
  localparam int EXPECTED_WGT_DDR_READS = L0_WGT_DDR_WORDS + L1_WGT_DDR_WORDS + L2_WGT_DDR_WORDS + L3_WGT_DDR_WORDS + L4_WGT_DDR_WORDS + L5_WGT_DDR_WORDS + L6_WGT_DDR_WORDS;
  localparam int EXPECTED_OFM_DDR_WORDS = L6_F_OUT * L6_H_OUT * L6_STORED_GROUPS;
  localparam int EXPECTED_FINAL_ELEMENTS = L6_F_OUT * L6_H_OUT * L6_W_OUT;

  // --------------------------------------------------------------------------
  // DUT wiring
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
  logic done_seen, error_seen;
  logic [3:0] first_error_vec;
  logic [$clog2(CFG_DEPTH)-1:0] first_error_layer;
  logic first_error_mode;
  integer first_error_cycle;

  // The current top-level keeps these Mode1 free-token metadata inputs
  // visible; the tested RTL path uses the row token and internal stream state.
  assign m1_free_col_blk_g = 16'd0;
  assign m1_free_ch_blk_g  = 16'd0;
  assign m1_sm_refill_req_ready = 1'b1;
  assign m2_sm_refill_req_ready = 1'b1;

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

  // --------------------------------------------------------------------------
  // Software reference model storage.
  // ref_cur_sel tells expected_final_word which reference bank contains L6 output.
  // --------------------------------------------------------------------------
  int ref_a [0:H_MAX-1][0:W_MAX-1][0:F_MAX-1];
  int ref_b [0:H_MAX-1][0:W_MAX-1][0:F_MAX-1];
  bit ref_cur_sel;

  function automatic logic [DATA_W-1:0] sat8(input int val);
    begin
      if (val < 0) sat8 = '0;
      else if (val > 127) sat8 = 8'd127;
      else sat8 = val[DATA_W-1:0];
    end
  endfunction

  function automatic int input_pattern_val(input int row, input int col, input int ch);
    begin
      // Bounded, non-negative, spatial/channel-distinct pattern.
      input_pattern_val = 1 + (row % 11) + 2 * (col % 13) + 3 * (ch % 7);
    end
  endfunction

  function automatic int ref_get(input bit sel, input int row, input int col, input int ch);
    begin
      if (sel == 1'b0) ref_get = ref_a[row][col][ch];
      else             ref_get = ref_b[row][col][ch];
    end
  endfunction

  task automatic ref_set(input bit sel, input int row, input int col, input int ch, input int val);
    begin
      if (sel == 1'b0) ref_a[row][col][ch] = val;
      else             ref_b[row][col][ch] = val;
    end
  endtask

  function automatic bit layer_is_mode2(input int layer_id);
    begin
      layer_is_mode2 = (layer_id >= 4);
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
        default: layer_c_in = L6_C_IN;
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
        default: layer_f_out = L6_F_OUT;
      endcase
    end
  endfunction

  function automatic int layer_h_in(input int layer_id);
    begin
      case (layer_id)
        0: layer_h_in = L0_H_IN;
        1: layer_h_in = L1_H_IN;
        2: layer_h_in = L2_H_IN;
        3: layer_h_in = L3_H_IN;
        4: layer_h_in = L4_H_IN;
        5: layer_h_in = L5_H_IN;
        default: layer_h_in = L6_H_IN;
      endcase
    end
  endfunction

  function automatic int layer_w_in(input int layer_id);
    begin
      case (layer_id)
        0: layer_w_in = L0_W_IN;
        1: layer_w_in = L1_W_IN;
        2: layer_w_in = L2_W_IN;
        3: layer_w_in = L3_W_IN;
        4: layer_w_in = L4_W_IN;
        5: layer_w_in = L5_W_IN;
        default: layer_w_in = L6_W_IN;
      endcase
    end
  endfunction

  function automatic int layer_h_out(input int layer_id);
    begin
      case (layer_id)
        0: layer_h_out = L0_H_OUT;
        1: layer_h_out = L1_H_OUT;
        2: layer_h_out = L2_H_OUT;
        3: layer_h_out = L3_H_OUT;
        4: layer_h_out = L4_H_OUT;
        5: layer_h_out = L5_H_OUT;
        default: layer_h_out = L6_H_OUT;
      endcase
    end
  endfunction

  function automatic int layer_w_out(input int layer_id);
    begin
      case (layer_id)
        0: layer_w_out = L0_W_OUT;
        1: layer_w_out = L1_W_OUT;
        2: layer_w_out = L2_W_OUT;
        3: layer_w_out = L3_W_OUT;
        4: layer_w_out = L4_W_OUT;
        5: layer_w_out = L5_W_OUT;
        default: layer_w_out = L6_W_OUT;
      endcase
    end
  endfunction

  function automatic bit layer_pool_en(input int layer_id);
    begin
      case (layer_id)
        0: layer_pool_en = L0_POOL_EN;
        1: layer_pool_en = L1_POOL_EN;
        2: layer_pool_en = L2_POOL_EN;
        3: layer_pool_en = L3_POOL_EN;
        4: layer_pool_en = L4_POOL_EN;
        5: layer_pool_en = L5_POOL_EN;
        default: layer_pool_en = L6_POOL_EN;
      endcase
    end
  endfunction

  function automatic int model_tap_ky(input int layer_id, input int fout);
    begin
      model_tap_ky = (fout + layer_id) % 3;
    end
  endfunction

  function automatic int model_tap_kx(input int layer_id, input int fout);
    begin
      model_tap_kx = ((fout / 3) + layer_id) % 3;
    end
  endfunction

  function automatic logic signed [DATA_W-1:0] model_weight_value(
    input int layer_id,
    input int fout,
    input int cin,
    input int ky,
    input int kx
  );
    int C;
    begin
      C = layer_c_in(layer_id);
      if ((cin == (fout % C)) &&
          (ky == model_tap_ky(layer_id, fout)) &&
          (kx == model_tap_kx(layer_id, fout))) begin
        model_weight_value = 8'sd1;
      end else begin
        model_weight_value = 8'sd0;
      end
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
        // Mode1: physical weight word is organized by bundle groups; PF lanes
        // inside the Ptotal word correspond to filters within f_group.
        bundle = phys_word * PV_M1 + (lane / PF_M1);
        tmp = bundle;
        kx = tmp % LK; tmp = tmp / LK;
        ky = tmp % LK; tmp = tmp / LK;
        cidx = tmp % C; tmp = tmp / C;
        fgroup = tmp;
        pf = lane % PF_M1;
        fout = fgroup * PF_M1 + pf;
        if ((fout < F) && (cidx < C)) begin
          sparse_weight_value = model_weight_value(layer_id, fout, cidx, ky, kx);
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
        if ((fout < F) && (cin < C)) begin
          sparse_weight_value = model_weight_value(layer_id, fout, cin, ky, kx);
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

  task automatic golden_init_input;
    begin
      for (int r = 0; r < H_MAX; r++) begin
        for (int c = 0; c < W_MAX; c++) begin
          for (int ch = 0; ch < F_MAX; ch++) begin
            ref_a[r][c][ch] = 0;
            ref_b[r][c][ch] = 0;
          end
        end
      end
      for (int r = 0; r < L0_H_IN; r++) begin
        for (int c = 0; c < L0_W_IN; c++) begin
          for (int ch = 0; ch < L0_C_IN; ch++) begin
            ref_a[r][c][ch] = sat8(input_pattern_val(r, c, ch));
          end
        end
      end
      ref_cur_sel = 1'b0;
    end
  endtask

  task automatic golden_compute_layer(input int layer_id);
    int h_in, w_in, c_in, f_out, h_out, w_out;
    bit pool_en;
    bit conv_sel;
    int sum;
    int rr, cc;
    int maxv;
    int v;
    begin
      h_in = layer_h_in(layer_id);
      w_in = layer_w_in(layer_id);
      c_in = layer_c_in(layer_id);
      f_out = layer_f_out(layer_id);
      h_out = layer_h_out(layer_id);
      w_out = layer_w_out(layer_id);
      pool_en = layer_pool_en(layer_id);
      conv_sel = !ref_cur_sel;

      for (int r = 0; r < h_in; r++) begin
        for (int c = 0; c < w_in; c++) begin
          for (int f = 0; f < f_out; f++) begin
            sum = 0;
            for (int ky = 0; ky < LK; ky++) begin
              for (int kx = 0; kx < LK; kx++) begin
                int src_r;
                int src_c;
                int src_ch;
                int wgt;
                src_r = r + ky - PAD;
                src_c = c + kx - PAD;
                src_ch = f % c_in;
                wgt = model_weight_value(layer_id, f, src_ch, ky, kx);
                if ((src_r >= 0) && (src_r < h_in) && (src_c >= 0) && (src_c < w_in)) begin
                  sum += ref_get(ref_cur_sel, src_r, src_c, src_ch) * wgt;
                end
              end
            end
            // ReLU + 8-bit saturation, matching the existing regression model.
            ref_set(conv_sel, r, c, f, sat8(sum));
          end
        end
      end

      if (pool_en) begin
        for (int r = 0; r < h_out; r++) begin
          for (int c = 0; c < w_out; c++) begin
            for (int f = 0; f < f_out; f++) begin
              maxv = 0;
              for (int py = 0; py < 2; py++) begin
                for (int px = 0; px < 2; px++) begin
                  rr = r * 2 + py;
                  cc = c * 2 + px;
                  v = ref_get(conv_sel, rr, cc, f);
                  if ((py == 0) && (px == 0)) maxv = v;
                  else if (v > maxv) maxv = v;
                end
              end
              ref_set(ref_cur_sel, r, c, f, maxv);
            end
          end
        end
      end else begin
        ref_cur_sel = conv_sel;
      end
    end
  endtask

  task automatic golden_compute_all;
    begin
      golden_init_input();
      for (int lid = 0; lid < 7; lid++) begin
        golden_compute_layer(lid);
      end
    end
  endtask

  function automatic logic [DATA_W-1:0] expected_final_byte(input int row, input int col, input int out_ch);
    begin
      expected_final_byte = sat8(ref_get(ref_cur_sel, row, col, out_ch));
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
      words_per_ch = L6_H_OUT * L6_STORED_GROUPS;
      ch = word_idx / words_per_ch;
      rem = word_idx % words_per_ch;
      row = rem / L6_STORED_GROUPS;
      grp = rem % L6_STORED_GROUPS;
      for (int lane = 0; lane < DDR_LANES; lane++) begin
        col_g = grp * PC + lane;
        if ((ch < L6_F_OUT) && (row < L6_H_OUT) && (lane < PC) && (col_g < L6_W_OUT)) begin
          word[lane*DATA_W +: DATA_W] = expected_final_byte(row, col_g, ch);
        end
      end
      return word;
    end
  endfunction

  task automatic init_mem;
    int addr;
    begin
      for (int i = 0; i < MEM_DEPTH; i++) ddr_mem[i] = '0;
      for (int ch = 0; ch < L0_C_IN; ch++) begin
        for (int row = 0; row < L0_H_IN; row++) begin
          for (int col_blk = 0; col_blk < L0_COLBLKS; col_blk++) begin
            addr = TB_IFM_BASE + ch * L0_H_IN * L0_COLBLKS + row * L0_COLBLKS + col_blk;
            ddr_mem[addr] = pack_m1_ifm_word(ch, row, col_blk);
          end
        end
      end

      for (int i = 0; i < L0_WGT_DDR_WORDS; i++) ddr_mem[L0_WGT_DDR_BASE + i] = pack_weight_ddr_word(0, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (int i = 0; i < L1_WGT_DDR_WORDS; i++) ddr_mem[L1_WGT_DDR_BASE + i] = pack_weight_ddr_word(1, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (int i = 0; i < L2_WGT_DDR_WORDS; i++) ddr_mem[L2_WGT_DDR_BASE + i] = pack_weight_ddr_word(2, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (int i = 0; i < L3_WGT_DDR_WORDS; i++) ddr_mem[L3_WGT_DDR_BASE + i] = pack_weight_ddr_word(3, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (int i = 0; i < L4_WGT_DDR_WORDS; i++) ddr_mem[L4_WGT_DDR_BASE + i] = pack_weight_ddr_word(4, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (int i = 0; i < L5_WGT_DDR_WORDS; i++) ddr_mem[L5_WGT_DDR_BASE + i] = pack_weight_ddr_word(5, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
      for (int i = 0; i < L6_WGT_DDR_WORDS; i++) ddr_mem[L6_WGT_DDR_BASE + i] = pack_weight_ddr_word(6, i / WGT_SUBWORDS, i % WGT_SUBWORDS);
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
    layer_desc_t cfg0, cfg1, cfg2, cfg3, cfg4, cfg5, cfg6;
    begin
      fill_layer_cfg(cfg0, 0, 1'b0, L0_H_IN,L0_W_IN,L0_C_IN,L0_F_OUT,L0_H_CONV_OUT,L0_W_CONV_OUT,L0_POOL_EN,L0_WGT_DDR_BASE, 1'b1, 1'b0);
      fill_layer_cfg(cfg1, 1, 1'b0, L1_H_IN,L1_W_IN,L1_C_IN,L1_F_OUT,L1_H_CONV_OUT,L1_W_CONV_OUT,L1_POOL_EN,L1_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg2, 2, 1'b0, L2_H_IN,L2_W_IN,L2_C_IN,L2_F_OUT,L2_H_CONV_OUT,L2_W_CONV_OUT,L2_POOL_EN,L2_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg3, 3, 1'b0, L3_H_IN,L3_W_IN,L3_C_IN,L3_F_OUT,L3_H_CONV_OUT,L3_W_CONV_OUT,L3_POOL_EN,L3_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg4, 4, 1'b1, L4_H_IN,L4_W_IN,L4_C_IN,L4_F_OUT,L4_H_CONV_OUT,L4_W_CONV_OUT,L4_POOL_EN,L4_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg5, 5, 1'b1, L5_H_IN,L5_W_IN,L5_C_IN,L5_F_OUT,L5_H_CONV_OUT,L5_W_CONV_OUT,L5_POOL_EN,L5_WGT_DDR_BASE, 1'b0, 1'b0);
      fill_layer_cfg(cfg6, 6, 1'b1, L6_H_IN,L6_W_IN,L6_C_IN,L6_F_OUT,L6_H_CONV_OUT,L6_W_CONV_OUT,L6_POOL_EN,L6_WGT_DDR_BASE, 1'b0, 1'b1);

      write_cfg(0, cfg0);
      write_cfg(1, cfg1);
      write_cfg(2, cfg2);
      write_cfg(3, cfg3);
      write_cfg(4, cfg4);
      write_cfg(5, cfg5);
      write_cfg(6, cfg6);
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
    begin
      mismatch = 0;
      for (int j = 0; j < EXPECTED_OFM_DDR_WORDS; j++) begin
        got_word = ddr_mem[TB_OFM_BASE + j];
        exp_word = expected_final_word(j);
        if (got_word !== exp_word) begin
          if (mismatch < 64) begin
            $display("TB_MISMATCH_FINAL word=%0d got=0x%0h exp=0x%0h", j, got_word, exp_word);
          end
          mismatch++;
        end
      end

      if (mismatch != 0) begin
        dump_ofm_region();
        $fatal(1, "TB_FAIL: large 80x80 4M1+3M2 K3/P1 tap-sweep final OFM mismatch count=%0d", mismatch);
      end else begin
        $display("TB_PASS: large 80x80 4M1+3M2 K3/P1 tap-sweep final OFM matched %0d DDR words", EXPECTED_OFM_DDR_WORDS);
      end
    end
  endtask

  task automatic print_banner;
    begin
      $display("TB_INFO: VGG7-large 80x80 stress, 4M1+3M2, K=3 stride=1 pad=1, tap-sweep weights");
      $display("TB_INFO: params PTOTAL=%0d PV_MAX=%0d PF_MAX=%0d PC=%0d PF=%0d C_MAX=%0d F_MAX=%0d", PTOTAL, PV_MAX, PF_MAX, PC, PF, C_MAX, F_MAX);
      $display("TB_INFO: L0-L3 MODE1, L4-L6 MODE2; final expected shape %0dx%0dx%0d", L6_H_OUT, L6_W_OUT, L6_F_OUT);
      $display("TB_INFO: dimensions exceed 16: L0 %0dx%0dx%0d -> F%0d; L6 input %0dx%0dx%0d -> F%0d", L0_H_IN, L0_W_IN, L0_C_IN, L0_F_OUT, L6_H_IN, L6_W_IN, L6_C_IN, L6_F_OUT);
      $display("TB_INFO: DDR bases IFM=0x%0h WGT=0x%0h OFM=0x%0h WGT_END=0x%0h", TB_IFM_BASE, TB_WGT_BASE, TB_OFM_BASE, WGT_DDR_END);
      $display("TB_INFO: expected reads IFM=%0d WGT=%0d final OFM words=%0d elements=%0d", EXPECTED_IFM_DDR_READS, EXPECTED_WGT_DDR_READS, EXPECTED_OFM_DDR_WORDS, EXPECTED_FINAL_ELEMENTS);
    end
  endtask

  // --------------------------------------------------------------------------
  // DDR model and counters
  // --------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_pending_q <= 1'b0;
      rd_addr_q <= '0;
      ddr_rd_valid <= 1'b0;
      ddr_rd_data <= '0;
      ddr_ifm_read_count <= 0;
      ddr_wgt_read_count <= 0;
      ddr_ofm_write_count <= 0;
    end else begin
      ddr_rd_valid <= rd_pending_q;
      ddr_rd_data <= rd_pending_q ? ddr_mem[rd_addr_q] : '0;

      rd_pending_q <= ddr_rd_req;
      if (ddr_rd_req) begin
        rd_addr_q <= ddr_rd_addr;
        if ((ddr_rd_addr >= TB_IFM_BASE) && (ddr_rd_addr < TB_WGT_BASE)) begin
          ddr_ifm_read_count <= ddr_ifm_read_count + 1;
        end else if ((ddr_rd_addr >= TB_WGT_BASE) && (ddr_rd_addr < TB_OFM_BASE)) begin
          ddr_wgt_read_count <= ddr_wgt_read_count + 1;
        end
      end

      if (ddr_wr_en) begin
        for (int by = 0; by < DDR_WORD_W/8; by++) begin
          if (ddr_wr_be[by]) begin
            ddr_mem[ddr_wr_addr][by*8 +: 8] <= ddr_wr_data[by*8 +: 8];
          end
        end
        if (ddr_wr_addr >= TB_OFM_BASE) begin
          ddr_ofm_write_count <= ddr_ofm_write_count + 1;
        end
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cycle_count <= 0;
      ofm_ifm_stream_start_count <= 0;
      ofm_ifm_stream_done_count <= 0;
      legacy_m2_mgr_req_count <= 0;
      done_seen <= 1'b0;
      error_seen <= 1'b0;
      first_error_vec <= '0;
      first_error_layer <= '0;
      first_error_mode <= 1'b0;
      first_error_cycle <= -1;
    end else begin
      cycle_count <= cycle_count + 1;

      if (dut.u_control_unit_top.ofm_ifm_stream_start) ofm_ifm_stream_start_count <= ofm_ifm_stream_start_count + 1;
      if (dut.u_control_unit_top.ofm_ifm_stream_done) ofm_ifm_stream_done_count <= ofm_ifm_stream_done_count + 1;
      if (m2_sm_refill_req_valid) legacy_m2_mgr_req_count <= legacy_m2_mgr_req_count + 1;

      if (done) done_seen <= 1'b1;
      if (error && !error_seen) begin
        error_seen <= 1'b1;
        first_error_vec <= dbg_error_vec;
        first_error_layer <= dbg_layer_idx;
        first_error_mode <= dbg_mode;
        first_error_cycle <= cycle_count;
        $display("DBG_FIRST_ERROR t=%0t cycle=%0d dbg_error_vec=%b layer=%0d mode=%0d", $time, cycle_count, dbg_error_vec, dbg_layer_idx, dbg_mode);
        $display("DBG_ERROR_MAP bit0=dma_error bit1=ofm_error bit2=local_error bit3=transition_error");
      end

      if ((cycle_count % 200000) == 0) begin
        $display("DBG_TOP_STATUS t=%0t cycle=%0d busy=%0d done=%0d error=%0d vec=%b layer=%0d mode=%0d ifm_rd=%0d wgt_rd=%0d ofm_wr=%0d stream_start=%0d stream_done=%0d",
          $time, cycle_count, busy, done, error, dbg_error_vec, dbg_layer_idx, dbg_mode,
          ddr_ifm_read_count, ddr_wgt_read_count, ddr_ofm_write_count,
          ofm_ifm_stream_start_count, ofm_ifm_stream_done_count);
      end
    end
  end

  // --------------------------------------------------------------------------
  // Main sequence
  // --------------------------------------------------------------------------
  initial begin
    rst_n = 1'b0;
    start = 1'b0;
    abort = 1'b0;
    cfg_wr_en = 1'b0;
    cfg_wr_addr = '0;
    cfg_wr_data = '0;
    cfg_num_layers = 7;

    print_banner();
    init_mem();
    golden_compute_all();

    repeat (10) @(posedge clk);
    rst_n = 1'b1;
    repeat (5) @(posedge clk);

    program_layers();
    repeat (5) @(posedge clk);
    pulse_start();

    while (!done && !error && (cycle_count < MAX_CYCLES)) begin
      @(posedge clk);
    end

    if (error) begin
      $fatal(1, "TB_FAIL: DUT asserted error. first_cycle=%0d layer=%0d mode=%0d vec=%b", first_error_cycle, first_error_layer, first_error_mode, first_error_vec);
    end

    if (!done) begin
      $fatal(1, "TB_FAIL: timeout after %0d cycles. busy=%0d done=%0d error=%0d", cycle_count, busy, done, error);
    end

    $display("TB_INFO: large 80x80 stress test done after %0d cycles", cycle_count);
    $display("TB_INFO: DDR counts: ifm_reads=%0d est=%0d, wgt_reads=%0d est=%0d, ofm_writes=%0d expected=%0d",
      ddr_ifm_read_count, EXPECTED_IFM_DDR_READS,
      ddr_wgt_read_count, EXPECTED_WGT_DDR_READS,
      ddr_ofm_write_count, EXPECTED_OFM_DDR_WORDS);
    $display("TB_INFO: OFM->IFM stream starts=%0d done=%0d legacy_m2_mgr_req=%0d",
      ofm_ifm_stream_start_count, ofm_ifm_stream_done_count, legacy_m2_mgr_req_count);

    check_final_ofm();
    $finish;
  end

endmodule
