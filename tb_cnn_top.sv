`timescale 1ns/1ps
`include "cnn_ddr_defs.svh"

module tb_cnn_top;
  import cnn_layer_desc_pkg::*;

  // --------------------------------------------------------------------------
  // Small simulation configuration
  // --------------------------------------------------------------------------
  localparam int DATA_W           = 8;
  localparam int PSUM_W           = 32;
  localparam int PTOTAL           = 4;
  localparam int PV_MAX           = 4;
  localparam int PF_MAX           = 4;
  localparam int PC_MODE2         = 2;
  localparam int PF_MODE2         = 2;
  localparam int C_MAX            = 4;
  localparam int F_MAX            = 4;
  localparam int W_MAX            = 4;
  localparam int H_MAX            = 4;
  localparam int HT               = 4;
  localparam int K_MAX            = 3;
  localparam int WGT_DEPTH        = 16;
  localparam int OFM_BANK_DEPTH   = H_MAX * W_MAX;
  localparam int OFM_LINEAR_DEPTH = C_MAX * OFM_BANK_DEPTH;
  localparam int CFG_DEPTH        = 4;
  localparam int DDR_ADDR_W       = `CNN_DDR_ADDR_W;
  localparam int DDR_WORD_W       = PV_MAX * DATA_W;
  localparam int MEM_DEPTH        = (`DDR_RSVD_BASE + `DDR_RSVD_SIZE);
  localparam int CLK_PERIOD_NS    = 10;
  localparam int MAX_CYCLES       = 50000;

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
  integer                i;
  integer                b;

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

      // Return one pending read.
      if (rd_pending_q) begin
        ddr_rd_valid <= 1'b1;
        if (rd_addr_q < MEM_DEPTH)
          ddr_rd_data <= ddr_mem[rd_addr_q];
        else
          ddr_rd_data <= '0;
      end

      // Byte-enable write handling.
      if (ddr_wr_en) begin
        if (ddr_wr_addr < MEM_DEPTH) begin
          for (b = 0; b < (DDR_WORD_W/8); b = b + 1) begin
            if (ddr_wr_be[b])
              ddr_mem[ddr_wr_addr][8*b +: 8] <= ddr_wr_data[8*b +: 8];
          end
        end
        if ((ddr_wr_addr >= `DDR_OFM_BASE) && (ddr_wr_addr < (`DDR_OFM_BASE + `DDR_OFM_SIZE)))
          ddr_ofm_write_count <= ddr_ofm_write_count + 1;
      end

      // Accept at most one outstanding read, but allow back-to-back requests.
      if (ddr_rd_req) begin
        rd_pending_q <= 1'b1;
        rd_addr_q    <= ddr_rd_addr;
      end else if (rd_pending_q) begin
        rd_pending_q <= 1'b0;
      end
    end
  end

  // --------------------------------------------------------------------------
  // Helpers
  // --------------------------------------------------------------------------
  task automatic init_mem;
    begin
      for (i = 0; i < MEM_DEPTH; i = i + 1)
        ddr_mem[i] = '0;

      // Simple IFM pattern in DDR IFM region
      for (i = 0; i < 64; i = i + 1)
        ddr_mem[`DDR_IFM_BASE + i] = 32'h1000_0000 + i;

      // Simple weight pattern in DDR WGT region
      for (i = 0; i < 64; i = i + 1)
        ddr_mem[`DDR_WGT_BASE + i] = 32'h2000_0000 + i;
    end
  endtask

  task automatic program_single_mode1_layer;
    layer_desc_t cfg;
    begin
      cfg = '0;
      cfg.layer_id      = 0;
      cfg.mode          = MODE1;
      cfg.h_in          = 4;
      cfg.w_in          = 4;
      cfg.c_in          = 1;
      cfg.f_out         = 2;
      cfg.k             = 3;
      cfg.h_out         = 2;
      cfg.w_out         = 2;
      cfg.pv_m1         = 2;
      cfg.pf_m1         = 2;
      cfg.pc_m2         = 2;
      cfg.pf_m2         = 2;
      cfg.conv_stride   = 1;
      cfg.pad_top       = 0;
      cfg.pad_bottom    = 0;
      cfg.pad_left      = 0;
      cfg.pad_right     = 0;
      cfg.relu_en       = 0;
      cfg.pool_en       = 0;
      cfg.pool_k        = 0;
      cfg.pool_stride   = 0;
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

  task automatic dump_ofm_region;
    integer j;
    begin
      $display("--- OFM DDR region dump ---");
      for (j = 0; j < 16; j = j + 1)
        $display("OFM[%0d] @0x%05h = 0x%08h", j, (`DDR_OFM_BASE + j), ddr_mem[`DDR_OFM_BASE + j]);
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

    program_single_mode1_layer();
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
      $display("TB_FAIL: timeout after %0d cycles. busy=%0b done=%0b error=%0b", cycle_count, busy, done, error);
      dump_ofm_region();
      $finish;
    end

    $display("TB_INFO: done observed after %0d cycles", cycle_count);
    $display("TB_INFO: OFM DDR writes counted = %0d", ddr_ofm_write_count);
    dump_ofm_region();

    if (ddr_ofm_write_count == 0) begin
      $display("TB_WARN: DUT finished but no OFM write was observed into DDR OFM region");
    end else begin
      $display("TB_PASS: smoke test completed with OFM DDR write activity");
    end

    $finish;
  end

endmodule
