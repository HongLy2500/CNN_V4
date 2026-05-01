`timescale 1ns/1ps
`include "cnn_ddr_defs.svh"

// ============================================================================
// Benchmark-like top-level testbench for cnn_top, MODE 1 only
//
// Purpose:
// - Keep the same structure as the existing tb_cnn_top.sv.
// - Exercise a non-trivial mode-1 layer, closer to a real benchmark layer:
//     * multiple input channels
//     * multiple output filter groups
//     * K=3
//     * multiple output rows/columns
//     * Pv/Pf runtime grouping
//     * weight DMA packing where one physical PTOTAL word spans multiple DDR words
//     * final OFM DDR exact checking against a golden reference
//
// Important current-RTL behavior:
// - mode1_compute_top always routes CE output through ReLU + pooling_mode1.
// - control_unit_top drives OFM h/w for mode 1 as cur_cfg.h_out/2 and w_out/2
//   when h_out/w_out > 1.
// - Therefore this TB treats cfg.h_out/w_out as convolution output dimensions,
//   and checks the final pooled 2x2 output in DDR.
// ============================================================================

module tb_cnn_top_m1_benchmark_like;
  import cnn_layer_desc_pkg::*;

  // --------------------------------------------------------------------------
  // DUT configuration: larger than smoke, still simulation-friendly
  // --------------------------------------------------------------------------
  localparam int DATA_W           = 8;
  localparam int PSUM_W           = 32;

  // Mode 1 runtime: PTOTAL = Pv * Pf = 2 * 4 = 8
  localparam int PTOTAL           = 8;
  localparam int PV_MAX           = 4;
  localparam int PF_MAX           = 4;
  localparam int PV_M1            = 2;
  localparam int PF_M1            = 4;

  // Mode 2 is not used in this TB, but keep parameters consistent with PTOTAL.
  localparam int PC_MODE2         = 4;
  localparam int PF_MODE2         = 2;

  localparam int C_MAX            = 8;
  localparam int F_MAX            = 8;
  localparam int W_MAX            = 8;
  localparam int H_MAX            = 8;
  localparam int HT               = 8;
  localparam int K_MAX            = 3;
  localparam int WGT_DEPTH        = 128;
  localparam int OFM_BANK_DEPTH   = H_MAX * W_MAX;
  localparam int OFM_LINEAR_DEPTH = C_MAX * OFM_BANK_DEPTH;
  localparam int CFG_DEPTH        = 4;
  localparam int DDR_ADDR_W       = `CNN_DDR_ADDR_W;
  localparam int DDR_WORD_W       = PV_MAX * DATA_W;
  localparam int MEM_DEPTH        = (`DDR_RSVD_BASE + `DDR_RSVD_SIZE);
  localparam int CLK_PERIOD_NS    = 10;
  localparam int MAX_CYCLES       = 200000;

  // --------------------------------------------------------------------------
  // Benchmark-like layer shape
  // --------------------------------------------------------------------------
  localparam int L_H_IN           = 6;
  localparam int L_W_IN           = 6;
  localparam int L_C_IN           = 3;
  localparam int L_F_OUT          = 8;
  localparam int L_K              = 3;
  localparam int L_H_CONV_OUT     = L_H_IN - L_K + 1;   // 4
  localparam int L_W_CONV_OUT     = L_W_IN - L_K + 1;   // 4
  localparam int L_H_POOL_OUT     = L_H_CONV_OUT / 2;   // 2
  localparam int L_W_POOL_OUT     = L_W_CONV_OUT / 2;   // 2

  localparam int IFM_WORDS_PER_ROW = (L_W_IN + PV_M1 - 1) / PV_M1;
  localparam int NUM_FGROUP        = (L_F_OUT + PF_M1 - 1) / PF_M1;

  // Mode-1 weight preload:
  // logical bundle order = (f_group, c, ky, kx), each bundle has PF_M1 weights.
  // one PTOTAL physical word packs PV_M1 logical bundles.
  localparam int M1_LOGICAL_BUNDLES = NUM_FGROUP * L_C_IN * L_K * L_K;
  localparam int M1_PHYS_WORDS      = (M1_LOGICAL_BUNDLES + PV_M1 - 1) / PV_M1;
  localparam int WGT_SUBWORDS       = (PTOTAL + PV_MAX - 1) / PV_MAX;
  localparam int M1_WGT_DDR_WORDS   = M1_PHYS_WORDS * WGT_SUBWORDS;

  // Current control_unit_top uses cfg_pv_next=1 on final mode-1 layer.
  // So OFM buffer stores one final pixel per DDR word for final-layer readback.
  localparam int FINAL_STORE_PACK   = 1;
  localparam int FINAL_GROUPS       = (L_W_POOL_OUT + FINAL_STORE_PACK - 1) / FINAL_STORE_PACK;
  localparam int EXP_OFM_WORDS      = L_F_OUT * L_H_POOL_OUT * FINAL_GROUPS;

  // --------------------------------------------------------------------------
  // DUT I/O
  // --------------------------------------------------------------------------
  logic clk;
  logic rst_n;

  logic start;
  logic abort;

  logic                           cfg_wr_en;
  logic [$clog2(CFG_DEPTH)-1:0]   cfg_wr_addr;
  layer_desc_t                    cfg_wr_data;
  logic [$clog2(CFG_DEPTH+1)-1:0] cfg_num_layers;

  logic                      ddr_rd_req;
  logic [DDR_ADDR_W-1:0]     ddr_rd_addr;
  logic                      ddr_rd_valid;
  logic [DDR_WORD_W-1:0]     ddr_rd_data;

  logic                      ddr_wr_en;
  logic [DDR_ADDR_W-1:0]     ddr_wr_addr;
  logic [DDR_WORD_W-1:0]     ddr_wr_data;
  logic [(DDR_WORD_W/8)-1:0] ddr_wr_be;

  logic [15:0] m1_free_col_blk_g;
  logic [15:0] m1_free_ch_blk_g;

  logic        m1_sm_refill_req_ready;
  logic        m1_sm_refill_req_valid;
  logic [$clog2(HT)-1:0] m1_sm_refill_row_slot_l;
  logic [15:0] m1_sm_refill_row_g;
  logic [15:0] m1_sm_refill_col_blk_g;
  logic [15:0] m1_sm_refill_ch_blk_g;

  logic        m2_sm_refill_req_ready;
  logic        m2_sm_refill_req_valid;
  logic [15:0] m2_sm_refill_row_g;
  logic [15:0] m2_sm_refill_col_g;
  logic [15:0] m2_sm_refill_col_l;
  logic [15:0] m2_sm_refill_cgrp_g;

  logic                     ifm_m1_free_valid;
  logic [$clog2(HT)-1:0]    ifm_m1_free_row_slot_l;
  logic [15:0]              ifm_m1_free_row_g;

  logic busy;
  logic done;
  logic error;

  logic [$clog2(CFG_DEPTH)-1:0] dbg_layer_idx;
  logic                         dbg_mode;
  logic                         dbg_weight_bank;
  logic [3:0]                   dbg_error_vec;

  // --------------------------------------------------------------------------
  // Simple DDR model
  // --------------------------------------------------------------------------
  logic [DDR_WORD_W-1:0] ddr_mem [0:MEM_DEPTH-1];
  logic                  rd_pending_q;
  logic [DDR_ADDR_W-1:0] rd_addr_q;
  integer                ddr_ofm_write_count;
  integer                cycle_count;
  integer                b;

  // --------------------------------------------------------------------------
  // Golden reference storage
  // --------------------------------------------------------------------------
  logic signed [DATA_W-1:0] ifm_ref  [0:L_C_IN-1][0:L_H_IN-1][0:L_W_IN-1];
  logic signed [DATA_W-1:0] wgt_ref  [0:L_F_OUT-1][0:L_C_IN-1][0:L_K-1][0:L_K-1];
  integer signed            conv_ref [0:L_F_OUT-1][0:L_H_CONV_OUT-1][0:L_W_CONV_OUT-1];
  integer signed            relu_ref [0:L_F_OUT-1][0:L_H_CONV_OUT-1][0:L_W_CONV_OUT-1];
  logic signed [DATA_W-1:0] pool_ref [0:L_F_OUT-1][0:L_H_POOL_OUT-1][0:L_W_POOL_OUT-1];

  // --------------------------------------------------------------------------
  // DUT
  // --------------------------------------------------------------------------
  cnn_top #(
    .DATA_W          (DATA_W),
    .PSUM_W          (PSUM_W),
    .PTOTAL          (PTOTAL),
    .PV_MAX          (PV_MAX),
    .PF_MAX          (PF_MAX),
    .PC_MODE2        (PC_MODE2),
    .PF_MODE2        (PF_MODE2),
    .C_MAX           (C_MAX),
    .F_MAX           (F_MAX),
    .W_MAX           (W_MAX),
    .H_MAX           (H_MAX),
    .HT              (HT),
    .K_MAX           (K_MAX),
    .WGT_DEPTH       (WGT_DEPTH),
    .OFM_BANK_DEPTH  (OFM_BANK_DEPTH),
    .OFM_LINEAR_DEPTH(OFM_LINEAR_DEPTH),
    .CFG_DEPTH       (CFG_DEPTH),
    .DDR_ADDR_W      (DDR_ADDR_W),
    .DDR_WORD_W      (DDR_WORD_W)
  ) dut (
    .clk                    (clk),
    .rst_n                  (rst_n),
    .start                  (start),
    .abort                  (abort),
    .cfg_wr_en              (cfg_wr_en),
    .cfg_wr_addr            (cfg_wr_addr),
    .cfg_wr_data            (cfg_wr_data),
    .cfg_num_layers         (cfg_num_layers),
    .ddr_rd_req             (ddr_rd_req),
    .ddr_rd_addr            (ddr_rd_addr),
    .ddr_rd_valid           (ddr_rd_valid),
    .ddr_rd_data            (ddr_rd_data),
    .ddr_wr_en              (ddr_wr_en),
    .ddr_wr_addr            (ddr_wr_addr),
    .ddr_wr_data            (ddr_wr_data),
    .ddr_wr_be              (ddr_wr_be),
    .m1_free_col_blk_g      (m1_free_col_blk_g),
    .m1_free_ch_blk_g       (m1_free_ch_blk_g),
    .m1_sm_refill_req_ready (m1_sm_refill_req_ready),
    .m1_sm_refill_req_valid (m1_sm_refill_req_valid),
    .m1_sm_refill_row_slot_l(m1_sm_refill_row_slot_l),
    .m1_sm_refill_row_g     (m1_sm_refill_row_g),
    .m1_sm_refill_col_blk_g (m1_sm_refill_col_blk_g),
    .m1_sm_refill_ch_blk_g  (m1_sm_refill_ch_blk_g),
    .m2_sm_refill_req_ready (m2_sm_refill_req_ready),
    .m2_sm_refill_req_valid (m2_sm_refill_req_valid),
    .m2_sm_refill_row_g     (m2_sm_refill_row_g),
    .m2_sm_refill_col_g     (m2_sm_refill_col_g),
    .m2_sm_refill_col_l     (m2_sm_refill_col_l),
    .m2_sm_refill_cgrp_g    (m2_sm_refill_cgrp_g),
    .ifm_m1_free_valid      (ifm_m1_free_valid),
    .ifm_m1_free_row_slot_l (ifm_m1_free_row_slot_l),
    .ifm_m1_free_row_g      (ifm_m1_free_row_g),
    .busy                   (busy),
    .done                   (done),
    .error                  (error),
    .dbg_layer_idx          (dbg_layer_idx),
    .dbg_mode               (dbg_mode),
    .dbg_weight_bank        (dbg_weight_bank),
    .dbg_error_vec          (dbg_error_vec)
  );

  // --------------------------------------------------------------------------
  // Clock
  // --------------------------------------------------------------------------
  initial clk = 1'b0;
  always #(CLK_PERIOD_NS/2) clk = ~clk;

  // --------------------------------------------------------------------------
  // Single-driver DDR model
  // --------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      ddr_rd_valid        <= 1'b0;
      ddr_rd_data         <= '0;
      rd_pending_q        <= 1'b0;
      rd_addr_q           <= '0;
      ddr_ofm_write_count <= 0;
      cycle_count         <= 0;
    end else begin
      cycle_count  <= cycle_count + 1;
      ddr_rd_valid <= 1'b0;

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
            if (ddr_wr_be[b])
              ddr_mem[ddr_wr_addr][8*b +: 8] <= ddr_wr_data[8*b +: 8];
          end
        end

        if ((ddr_wr_addr >= `DDR_OFM_BASE) &&
            (ddr_wr_addr < (`DDR_OFM_BASE + `DDR_OFM_SIZE))) begin
          ddr_ofm_write_count <= ddr_ofm_write_count + 1;
        end
      end

      if (ddr_rd_req) begin
        rd_pending_q <= 1'b1;
        rd_addr_q    <= ddr_rd_addr;
      end else if (rd_pending_q) begin
        rd_pending_q <= 1'b0;
      end
    end
  end

  // --------------------------------------------------------------------------
  // Reference data generation
  // --------------------------------------------------------------------------
  function automatic logic signed [DATA_W-1:0] gen_ifm_value(input integer c, input integer r, input integer x);
    integer tmp;
    begin
      // Deterministic small signed values in [-4, +4]
      tmp = ((c*17 + r*5 + x*3) % 9) - 4;
      gen_ifm_value = $signed(tmp);
    end
  endfunction

  function automatic logic signed [DATA_W-1:0] gen_wgt_value(input integer f, input integer c, input integer ky, input integer kx);
    integer tmp;
    begin
      // Deterministic small signed values in [-2, +2]
      tmp = ((f*13 + c*7 + ky*3 + kx*5) % 5) - 2;
      gen_wgt_value = $signed(tmp);
    end
  endfunction

  function automatic logic signed [DATA_W-1:0] sat_to_i8(input integer signed x);
    begin
      if (x > 127)
        sat_to_i8 = 8'sd127;
      else if (x < -128)
        sat_to_i8 = -8'sd128;
      else
        sat_to_i8 = $signed(x);
    end
  endfunction

  task automatic build_reference_tensors;
    integer c, r, x;
    integer f, ky, kx;
    integer sum;
    integer pr, pc;
    integer dy, dx;
    integer max_v;
    begin
      for (c = 0; c < L_C_IN; c = c + 1) begin
        for (r = 0; r < L_H_IN; r = r + 1) begin
          for (x = 0; x < L_W_IN; x = x + 1) begin
            ifm_ref[c][r][x] = gen_ifm_value(c, r, x);
          end
        end
      end

      for (f = 0; f < L_F_OUT; f = f + 1) begin
        for (c = 0; c < L_C_IN; c = c + 1) begin
          for (ky = 0; ky < L_K; ky = ky + 1) begin
            for (kx = 0; kx < L_K; kx = kx + 1) begin
              wgt_ref[f][c][ky][kx] = gen_wgt_value(f, c, ky, kx);
            end
          end
        end
      end

      for (f = 0; f < L_F_OUT; f = f + 1) begin
        for (r = 0; r < L_H_CONV_OUT; r = r + 1) begin
          for (x = 0; x < L_W_CONV_OUT; x = x + 1) begin
            sum = 0;
            for (c = 0; c < L_C_IN; c = c + 1) begin
              for (ky = 0; ky < L_K; ky = ky + 1) begin
                for (kx = 0; kx < L_K; kx = kx + 1) begin
                  sum = sum + ($signed(ifm_ref[c][r+ky][x+kx]) * $signed(wgt_ref[f][c][ky][kx]));
                end
              end
            end
            conv_ref[f][r][x] = sum;
            relu_ref[f][r][x] = (sum > 0) ? sum : 0;
          end
        end
      end

      // pooling_mode1 is 2x2 max-pool with stride 2 after ReLU, with int8 saturation.
      for (f = 0; f < L_F_OUT; f = f + 1) begin
        for (pr = 0; pr < L_H_POOL_OUT; pr = pr + 1) begin
          for (pc = 0; pc < L_W_POOL_OUT; pc = pc + 1) begin
            max_v = relu_ref[f][2*pr][2*pc];
            for (dy = 0; dy < 2; dy = dy + 1) begin
              for (dx = 0; dx < 2; dx = dx + 1) begin
                if (relu_ref[f][2*pr + dy][2*pc + dx] > max_v)
                  max_v = relu_ref[f][2*pr + dy][2*pc + dx];
              end
            end
            pool_ref[f][pr][pc] = sat_to_i8(max_v);
          end
        end
      end
    end
  endtask

  // --------------------------------------------------------------------------
  // DDR initialization helpers
  // --------------------------------------------------------------------------
  task automatic clear_ddr;
    integer i;
    begin
      for (i = 0; i < MEM_DEPTH; i = i + 1)
        ddr_mem[i] = '0;
    end
  endtask

  task automatic load_ifm_to_ddr_mode1;
    integer c, r, colg, lane;
    integer abs_col;
    integer addr;
    begin
      for (c = 0; c < L_C_IN; c = c + 1) begin
        for (r = 0; r < L_H_IN; r = r + 1) begin
          for (colg = 0; colg < IFM_WORDS_PER_ROW; colg = colg + 1) begin
            addr = `DDR_IFM_BASE + ((c * L_H_IN + r) * IFM_WORDS_PER_ROW) + colg;
            ddr_mem[addr] = '0;
            for (lane = 0; lane < PV_MAX; lane = lane + 1) begin
              abs_col = colg * PV_M1 + lane;
              if ((lane < PV_M1) && (abs_col < L_W_IN)) begin
                ddr_mem[addr][lane*DATA_W +: DATA_W] = ifm_ref[c][r][abs_col];
              end
            end
          end
        end
      end
    end
  endtask

  task automatic write_wgt_lane_to_ddr(
    input integer phys_word,
    input integer lane_abs,
    input logic signed [DATA_W-1:0] value
  );
    integer ddr_subword;
    integer ddr_lane;
    integer addr;
    begin
      ddr_subword = lane_abs / PV_MAX;
      ddr_lane    = lane_abs % PV_MAX;
      addr        = `DDR_WGT_BASE + phys_word * WGT_SUBWORDS + ddr_subword;
      ddr_mem[addr][ddr_lane*DATA_W +: DATA_W] = value;
    end
  endtask

  task automatic load_weights_to_ddr_mode1;
    integer p;
    integer logical_idx;
    integer phys_word;
    integer subword_idx;
    integer base_lane;
    integer fg, c, ky, kx, pf, f;
    begin
      // Clear only the needed DDR weight stream words.
      for (p = 0; p < M1_WGT_DDR_WORDS; p = p + 1)
        ddr_mem[`DDR_WGT_BASE + p] = '0;

      logical_idx = 0;
      for (fg = 0; fg < NUM_FGROUP; fg = fg + 1) begin
        for (c = 0; c < L_C_IN; c = c + 1) begin
          for (ky = 0; ky < L_K; ky = ky + 1) begin
            for (kx = 0; kx < L_K; kx = kx + 1) begin
              phys_word   = logical_idx / PV_M1;
              subword_idx = logical_idx % PV_M1;
              base_lane   = subword_idx * PF_M1;

              for (pf = 0; pf < PF_M1; pf = pf + 1) begin
                f = fg * PF_M1 + pf;
                if (f < L_F_OUT)
                  write_wgt_lane_to_ddr(phys_word, base_lane + pf, wgt_ref[f][c][ky][kx]);
                else
                  write_wgt_lane_to_ddr(phys_word, base_lane + pf, '0);
              end

              logical_idx = logical_idx + 1;
            end
          end
        end
      end
    end
  endtask

  task automatic init_mem;
    begin
      clear_ddr();
      build_reference_tensors();
      load_ifm_to_ddr_mode1();
      load_weights_to_ddr_mode1();

      $display("TB_INFO: Mode1 benchmark-like layer");
      $display("TB_INFO: IFM HxWxC = %0dx%0dx%0d", L_H_IN, L_W_IN, L_C_IN);
      $display("TB_INFO: CONV HxW/F/K = %0dx%0d / %0d / %0d", L_H_CONV_OUT, L_W_CONV_OUT, L_F_OUT, L_K);
      $display("TB_INFO: POOL HxW/F = %0dx%0d / %0d", L_H_POOL_OUT, L_W_POOL_OUT, L_F_OUT);
      $display("TB_INFO: Pv=%0d Pf=%0d PTOTAL=%0d", PV_M1, PF_M1, PTOTAL);
      $display("TB_INFO: weight logical bundles=%0d physical words=%0d DDR words=%0d",
               M1_LOGICAL_BUNDLES, M1_PHYS_WORDS, M1_WGT_DDR_WORDS);
      $display("TB_INFO: expected final OFM DDR words=%0d", EXP_OFM_WORDS);
    end
  endtask

  // --------------------------------------------------------------------------
  // Layer programming
  // --------------------------------------------------------------------------
  task automatic program_mode1_benchmark_like_layer;
    layer_desc_t cfg;
    begin
      cfg = '0;
      cfg.layer_id      = 0;
      cfg.mode          = MODE1;
      cfg.h_in          = L_H_IN;
      cfg.w_in          = L_W_IN;
      cfg.c_in          = L_C_IN;
      cfg.f_out         = L_F_OUT;
      cfg.k             = L_K;

      // These are convolution output dimensions.
      // control_unit_top will configure the OFM buffer to h_out/2, w_out/2 for mode 1.
      cfg.h_out         = L_H_CONV_OUT;
      cfg.w_out         = L_W_CONV_OUT;

      cfg.pv_m1         = PV_M1;
      cfg.pf_m1         = PF_M1;
      cfg.pc_m2         = PC_MODE2;
      cfg.pf_m2         = PF_MODE2;
      cfg.conv_stride   = 1;
      cfg.pad_top       = 0;
      cfg.pad_bottom    = 0;
      cfg.pad_left      = 0;
      cfg.pad_right     = 0;

      // These flags are kept for descriptor completeness.
      // Current mode1_compute_top/pooling path does not consume pool_en directly.
      cfg.relu_en       = 1'b1;
      cfg.pool_en       = 1'b1;
      cfg.pool_k        = 2;
      cfg.pool_stride   = 2;

      cfg.ifm_ddr_base  = `DDR_IFM_BASE;
      cfg.wgt_ddr_base  = `DDR_WGT_BASE;
      cfg.ofm_ddr_base  = `DDR_OFM_BASE;
      cfg.first_layer   = 1'b1;
      cfg.last_layer    = 1'b1;

      @(posedge clk);
      cfg_wr_en   <= 1'b1;
      cfg_wr_addr <= '0;
      cfg_wr_data <= cfg;
      @(posedge clk);
      cfg_wr_en   <= 1'b0;
      cfg_wr_addr <= '0;
      cfg_wr_data <= '0;
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

  // --------------------------------------------------------------------------
  // Debug dumps / exact checker
  // --------------------------------------------------------------------------
  task automatic dump_ofm_region;
    integer j;
    begin
      $display("--- OFM DDR region dump, first %0d expected words ---", EXP_OFM_WORDS);
      for (j = 0; j < EXP_OFM_WORDS; j = j + 1) begin
        $display("OFM[%0d] @0x%05h = 0x%08h", j, (`DDR_OFM_BASE + j), ddr_mem[`DDR_OFM_BASE + j]);
      end
    end
  endtask

  task automatic check_ofm_against_golden;
    integer lin;
    integer ch;
    integer rem;
    integer row;
    integer grp;
    integer col;
    integer mismatch;
    logic signed [DATA_W-1:0] got;
    logic signed [DATA_W-1:0] exp;
    begin
      mismatch = 0;

      if (ddr_ofm_write_count != EXP_OFM_WORDS) begin
        $display("TB_FAIL: OFM DDR write count mismatch. got=%0d expected=%0d",
                 ddr_ofm_write_count, EXP_OFM_WORDS);
        mismatch = mismatch + 1;
      end

      for (lin = 0; lin < EXP_OFM_WORDS; lin = lin + 1) begin
        ch  = lin / (L_H_POOL_OUT * FINAL_GROUPS);
        rem = lin % (L_H_POOL_OUT * FINAL_GROUPS);
        row = rem / FINAL_GROUPS;
        grp = rem % FINAL_GROUPS;
        col = grp * FINAL_STORE_PACK;

        got = ddr_mem[`DDR_OFM_BASE + lin][0 +: DATA_W];
        exp = pool_ref[ch][row][col];

        if (got !== exp) begin
          $display("TB_FAIL: OFM mismatch lin=%0d ch=%0d row=%0d col=%0d got=%0d/0x%02h expected=%0d/0x%02h word=0x%08h",
                   lin, ch, row, col, got, got, exp, exp, ddr_mem[`DDR_OFM_BASE + lin]);
          mismatch = mismatch + 1;
        end
      end

      if (mismatch == 0) begin
        $display("TB_PASS: exact final OFM matches golden for benchmark-like mode-1 layer");
      end else begin
        $display("TB_FAIL: total OFM mismatches = %0d", mismatch);
        dump_ofm_region();
        $finish;
      end
    end
  endtask

  // --------------------------------------------------------------------------
  // Stimulus
  // --------------------------------------------------------------------------
  initial begin
    rst_n                  = 1'b0;
    start                  = 1'b0;
    abort                  = 1'b0;
    cfg_wr_en              = 1'b0;
    cfg_wr_addr            = '0;
    cfg_wr_data            = '0;
    cfg_num_layers         = 1;
    m1_free_col_blk_g      = '0;
    m1_free_ch_blk_g       = '0;
    m1_sm_refill_req_ready = 1'b1;
    m2_sm_refill_req_ready = 1'b1;

    init_mem();

    repeat (5) @(posedge clk);
    rst_n = 1'b1;

    program_mode1_benchmark_like_layer();
    repeat (5) @(posedge clk);
    pulse_start();

    wait (done || error || (cycle_count > MAX_CYCLES));
    repeat (5) @(posedge clk);

    if (error) begin
      $display("TB_FAIL: DUT asserted error. dbg_error_vec=0x%0h layer=%0d mode=%0d",
               dbg_error_vec, dbg_layer_idx, dbg_mode);
      dump_ofm_region();
      $finish;
    end

    if (cycle_count > MAX_CYCLES) begin
      $display("TB_FAIL: timeout after %0d cycles. busy=%0b done=%0b error=%0b layer=%0d mode=%0d err=0x%0h",
               cycle_count, busy, done, error, dbg_layer_idx, dbg_mode, dbg_error_vec);
      dump_ofm_region();
      $finish;
    end

    $display("TB_INFO: done observed after %0d cycles", cycle_count);
    $display("TB_INFO: OFM DDR writes counted = %0d", ddr_ofm_write_count);

    check_ofm_against_golden();
    $finish;
  end

endmodule
