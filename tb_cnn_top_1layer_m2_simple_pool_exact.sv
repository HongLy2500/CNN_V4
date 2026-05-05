`timescale 1ns/1ps
`include "cnn_ddr_defs.svh"

module tb_cnn_top_1layer_m2_simple_pool_exact;
  import cnn_layer_desc_pkg::*;

  // --------------------------------------------------------------------------
  // Simple Mode-2 configuration
  // --------------------------------------------------------------------------
  localparam int DATA_W   = 8;
  localparam int PSUM_W   = 32;
  localparam int PC       = 4;
  localparam int PF       = 2;
  localparam int PTOTAL   = PC * PF;      // 8 lanes
  localparam int PV_MAX   = PTOTAL;       // DDR word = 8 bytes in this test
  localparam int PF_MAX   = PF;
  localparam int C_MAX    = 4;
  localparam int F_MAX    = 4;
  localparam int H_MAX    = 4;
  localparam int W_MAX    = 4;
  localparam int HT       = 4;
  localparam int K_MAX    = 3;
  localparam int WGT_DEPTH = 32;

  // Row-stride OFM layout: 1 word/row is enough for final pooled 1x1.
  // Use a small but safe depth for this simple test.
  localparam int OFM_ROW_STRIDE  = 4;
  localparam int OFM_BANK_DEPTH  = H_MAX * OFM_ROW_STRIDE;
  localparam int OFM_LINEAR_DEPTH = C_MAX * OFM_BANK_DEPTH;

  localparam int CFG_DEPTH = 4;
  localparam int DDR_ADDR_W = `CNN_DDR_ADDR_W;
  localparam int DDR_WORD_W = PV_MAX * DATA_W;
  localparam int MEM_DEPTH  = (`DDR_RSVD_BASE + `DDR_RSVD_SIZE);
  localparam int CLK_PERIOD_NS = 10;
  localparam int MAX_CYCLES = 20000;

  // Layer under test:
  // IFM 4x4x4, K=3, F=2, Mode2 PC=4/PF=2.
  // Conv output is 2x2x2, pooling_mode2 performs 2x2 maxpool -> final 1x1x2.
  // IFM = 1, weights = 1, so each conv result = 4*3*3 = 36.
  // ReLU keeps 36, maxpool keeps 36. Expected final DDR:
  //   word 0: channel/filter 0, lane0 = 0x24
  //   word 1: channel/filter 1, lane0 = 0x24
  localparam int IFM_H = 4;
  localparam int IFM_W = 4;
  localparam int IFM_C = 4;
  localparam int F_OUT = 2;
  localparam int K_CUR = 3;
  localparam int CONV_H = IFM_H - K_CUR + 1; // 2
  localparam int CONV_W = IFM_W - K_CUR + 1; // 2
  localparam int FINAL_H = CONV_H / 2;       // 1
  localparam int FINAL_W = CONV_W / 2;       // 1
  localparam int EXPECTED_OFM_DDR_WORDS = FINAL_H * FINAL_W * F_OUT; // 2

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

  // Latch terminal pulses. Some DUT status/error signals may pulse for a single
  // cycle; the original wait(done || error) could miss the root cause after the
  // following repeat cycles.
  logic done_seen;
  logic error_seen;
  logic [3:0] first_error_vec;
  logic [$clog2(CFG_DEPTH)-1:0] first_error_layer;
  logic first_error_mode;
  integer first_error_cycle;

  // --------------------------------------------------------------------------
  // Simple DDR model
  // --------------------------------------------------------------------------
  logic [DDR_WORD_W-1:0] ddr_mem [0:MEM_DEPTH-1];
  logic rd_pending_q;
  logic [DDR_ADDR_W-1:0] rd_addr_q;

  integer cycle_count;
  integer ddr_ifm_read_count;
  integer ddr_wgt_read_count;
  integer ddr_ofm_write_count;
  integer i;
  integer b;

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


always_ff @(posedge clk) begin
  if (rst_n && ddr_wr_en &&
      (ddr_wr_addr >= `DDR_OFM_BASE) &&
      (ddr_wr_addr < (`DDR_OFM_BASE + `DDR_OFM_SIZE))) begin
    $display("DBG_DDR_OFM_WRITE t=%0t cycle=%0d addr=0x%0h data=0x%0h be=0x%0h",
      $time,
      cycle_count,
      ddr_wr_addr,
      ddr_wr_data,
      ddr_wr_be
    );
  end
end

always_ff @(posedge clk) begin
  if (rst_n && dut.u_mode2_compute_top.ofm_wr_en) begin
    $display("DBG_M2_OFM_WR t=%0t cycle=%0d row=%0d col=%0d fbase=%0d data=0x%0h lane0=%0d lane1=%0d",
      $time,
      cycle_count,
      dut.u_mode2_compute_top.ofm_wr_row,
      dut.u_mode2_compute_top.ofm_wr_col,
      dut.u_mode2_compute_top.ofm_wr_f_base,
      dut.u_mode2_compute_top.ofm_wr_data,
      $signed(dut.u_mode2_compute_top.ofm_wr_data[0*DATA_W +: DATA_W]),
      $signed(dut.u_mode2_compute_top.ofm_wr_data[1*DATA_W +: DATA_W])
    );
  end
end

always_ff @(posedge clk) begin
  if (rst_n && dut.u_mode2_compute_top.ce_mac_data_out_valid) begin
    $display("DBG_M2_MAC_OUT t=%0t cycle=%0d out_row=%0d out_col=%0d fgrp=%0d mac0=%0d mac1=%0d",
      $time,
      cycle_count,
      dut.u_mode2_compute_top.out_row,
      dut.u_mode2_compute_top.out_col,
      dut.u_mode2_compute_top.f_group,
      $signed(dut.u_mode2_compute_top.ce_mac_data_out[0*PSUM_W +: PSUM_W]),
      $signed(dut.u_mode2_compute_top.ce_mac_data_out[1*PSUM_W +: PSUM_W])
    );
  end
end

always_ff @(posedge clk) begin
  if (rst_n && dut.u_mode2_compute_top.relu_data_out_valid) begin
    $display("DBG_M2_RELU_OUT t=%0t cycle=%0d relu0=%0d relu1=%0d group_start=%0b fbase=%0d",
      $time,
      cycle_count,
      $signed(dut.u_mode2_compute_top.relu_data_out[0*PSUM_W +: PSUM_W]),
      $signed(dut.u_mode2_compute_top.relu_data_out[1*PSUM_W +: PSUM_W]),
      dut.u_mode2_compute_top.relu_group_start,
      dut.u_mode2_compute_top.relu_f_base
    );
  end
end

always_ff @(posedge clk) begin
  if (rst_n && dut.u_ofm_buffer.ofm_dma_rd_valid) begin
    $display("DBG_OFM_DMA_RD t=%0t cycle=%0d rd_en=%0b rd_addr=%0d data=0x%0h keep=0x%0h layer_words=%0d done=%0b",
      $time,
      cycle_count,
      dut.u_ofm_buffer.ofm_dma_rd_en,
      dut.u_ofm_buffer.ofm_dma_rd_addr,
      dut.u_ofm_buffer.ofm_dma_rd_data,
      dut.u_ofm_buffer.ofm_dma_rd_keep,
      dut.u_ofm_buffer.layer_num_words,
      dut.u_ofm_buffer.layer_write_done
    );
  end
end
  // --------------------------------------------------------------------------
  // Single-driver DDR model
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
    end else begin
      cycle_count <= cycle_count + 1;
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
    end
  end

  // --------------------------------------------------------------------------
  // Terminal-status latch / first-error monitor
  // --------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      done_seen         <= 1'b0;
      error_seen        <= 1'b0;
      first_error_vec   <= '0;
      first_error_layer <= '0;
      first_error_mode  <= 1'b0;
      first_error_cycle <= 0;
    end else begin
      if (done) begin
        done_seen <= 1'b1;
      end

      if (error && !error_seen) begin
        error_seen        <= 1'b1;
        first_error_vec   <= dbg_error_vec;
        first_error_layer <= dbg_layer_idx;
        first_error_mode  <= dbg_mode;
        first_error_cycle <= cycle_count;

        $display("DBG_FIRST_ERROR t=%0t cycle=%0d dbg_error_vec=%04b layer=%0d mode=%0d busy=%0b done=%0b error=%0b",
                 $time, cycle_count, dbg_error_vec, dbg_layer_idx, dbg_mode, busy, done, error);
        $display("DBG_ERROR_MAP bit0=dma_error bit1=ofm_error bit2=local_error bit3=transition_error");
      end
    end
  end

  // Lightweight progress monitor. This uses only top-level DUT signals, so it
  // is safe even if internal instance names change.
  always_ff @(posedge clk) begin
    if (rst_n && (start || done || error || ddr_wr_en || ddr_rd_req)) begin
      if (error || done) begin
        $display("DBG_TOP_STATUS t=%0t cycle=%0d start=%0b busy=%0b done=%0b error=%0b vec=%04b layer=%0d mode=%0d ifm_rd=%0d wgt_rd=%0d ofm_wr=%0d rd_req=%0b rd_addr=0x%0h wr_en=%0b wr_addr=0x%0h",
                 $time, cycle_count, start, busy, done, error, dbg_error_vec, dbg_layer_idx, dbg_mode,
                 ddr_ifm_read_count, ddr_wgt_read_count, ddr_ofm_write_count,
                 ddr_rd_req, ddr_rd_addr, ddr_wr_en, ddr_wr_addr);
      end
    end
  end



  // Deep local-dataflow monitor for the current Mode-2 failure.
  // This catches the exact addr_gen_ifm_m2 state when local_error is asserted.
  // If your internal instance names differ, disable this block and capture the same signals in waveform.
`ifndef TB_DISABLE_DEEP_M2_LOCAL_MONITOR
  always_ff @(posedge clk) begin
    if (rst_n && dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.error) begin
      $display("DBG_M2_LOCAL_ERROR t=%0t cycle=%0d K=%0d C=%0d F=%0d H_in=%0d W_in=%0d Hout=%0d Wout=%0d num_cgrp=%0d num_fgrp=%0d block_row=%0d block_col=%0d issue_cgrp=%0d ky=%0d kx=%0d issue_any=%0b issue_first=%0b issue_succ=%0b addr_valid=%0b bank_base=%0d abs_row=%0d abs_col=%0d tile_base=%0d col_l=%0d out_row=%0d out_col=%0d f_group=%0d pass_start=%0b mac_en=%0b out_valid=%0b stream_active=%0b last_issue=%0b final_out_valid=%0b",
        $time,
        cycle_count,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.K_cur,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.C_cur,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.F_cur,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.H_in,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.W_in,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.Hout_cur,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.Wout_cur,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.num_cgroup,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.num_fgroup,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.block_row_q,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.block_col_q,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.issue_cgroup_q,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.issue_ky_q,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.issue_kx_q,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.issue_any,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.issue_first,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.issue_succ,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.issue_addr_valid,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.issue_bank_base16,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.issue_abs_row16,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.issue_abs_col_g16,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.issue_tile_base_g16,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.issue_col_sel_l16,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.out_row,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.out_col,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.f_group,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.pass_start_pulse,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.mac_en,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.out_valid,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.stream_active_q,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.last_issue,
        dut.u_control_unit_top.u_local_dataflow_manager.u_addr_gen_ifm_m2.final_out_valid
      );
    end
  end
`endif

  // Optional deep monitor. Enable only if these hierarchy names match your
  // current cnn_top. It is disabled by default to keep the testbench portable.
`ifdef TB_DEEP_M2_MONITOR
  always_ff @(posedge clk) begin
    if (rst_n && dut.u_ofm_buffer.error) begin
      $display("DBG_OFM_ERROR t=%0t cycle=%0d src_mode=%0b next_mode=%0b h=%0d w=%0d f=%0d pv_cur=%0d pf_cur=%0d pv_next=%0d pf_next=%0d store_pack=%0d groups=%0d pixels=%0d/%0d layer_done=%0b m2_wr_en=%0b m2_row=%0d m2_col=%0d m2_fbase=%0d",
        $time, cycle_count,
        dut.u_ofm_buffer.src_mode_q,
        dut.u_ofm_buffer.next_mode_q,
        dut.u_ofm_buffer.h_out_q,
        dut.u_ofm_buffer.w_out_q,
        dut.u_ofm_buffer.f_out_q,
        dut.u_ofm_buffer.pv_cur_q,
        dut.u_ofm_buffer.pf_cur_q,
        dut.u_ofm_buffer.pv_next_q,
        dut.u_ofm_buffer.pf_next_q,
        dut.u_ofm_buffer.store_pack_q,
        dut.u_ofm_buffer.stored_groups_q,
        dut.u_ofm_buffer.layer_pixels_written,
        dut.u_ofm_buffer.layer_num_pixels,
        dut.u_ofm_buffer.layer_write_done,
        dut.u_ofm_buffer.m2_wr_en,
        dut.u_ofm_buffer.m2_wr_row,
        dut.u_ofm_buffer.m2_wr_col,
        dut.u_ofm_buffer.m2_wr_f_base);
    end
  end
`endif

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
      for (ch = 0; ch < IFM_C; ch = ch + 1) begin
        for (row = 0; row < IFM_H; row = row + 1) begin
          ddr_mem[`DDR_IFM_BASE + word_idx] = pack_pc_ones_word();
          word_idx = word_idx + 1;
        end
      end

      // Mode 2 weight DDR layout: one physical PTOTAL-lane word per
      // (filter group, channel group, ky, kx). All weights = 1.
      for (i = 0; i < (K_CUR*K_CUR); i = i + 1) begin
        ddr_mem[`DDR_WGT_BASE + i] = pack_ptotal_ones_word();
      end
    end
  endtask

  task automatic program_single_mode2_layer;
    layer_desc_t cfg;
    begin
      cfg = '0;
      cfg.layer_id = 0;
      cfg.mode = MODE2;
      cfg.h_in = IFM_H;
      cfg.w_in = IFM_W;
      cfg.c_in = IFM_C;
      cfg.f_out = F_OUT;
      cfg.k = K_CUR;
      // For mode2_compute_top this is the convolution output size.
      // pooling_mode2 then emits final H/2 x W/2 output.
      cfg.h_out = CONV_H;
      cfg.w_out = CONV_W;
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
      cfg.wgt_ddr_base = `DDR_WGT_BASE;
      cfg.ofm_ddr_base = `DDR_OFM_BASE;
      cfg.first_layer = 1'b1;
      cfg.last_layer = 1'b1;

      @(posedge clk);
      cfg_wr_en <= 1'b1;
      cfg_wr_addr <= '0;
      cfg_wr_data <= cfg;
      @(posedge clk);
      cfg_wr_en <= 1'b0;
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
    int j;
    begin
      $display("--- OFM DDR region dump ---");
      for (j = 0; j < 8; j = j + 1) begin
        $display("OFM[%0d] @0x%05h = 0x%016h", j, (`DDR_OFM_BASE + j), ddr_mem[`DDR_OFM_BASE + j]);
      end
    end
  endtask

  task automatic check_final_ofm;
    int mismatch;
    logic [DDR_WORD_W-1:0] exp0;
    begin
      mismatch = 0;
      exp0 = expected_final_word(36);

      for (int j = 0; j < EXPECTED_OFM_DDR_WORDS; j = j + 1) begin
        if (ddr_mem[`DDR_OFM_BASE + j] !== exp0) begin
          $display("TB_MISMATCH_M2 word=%0d got=0x%016h exp=0x%016h",
                   j, ddr_mem[`DDR_OFM_BASE + j], exp0);
          mismatch = mismatch + 1;
        end
      end

      if (mismatch != 0) begin
        dump_ofm_region();
        $fatal(1, "TB_FAIL: Mode2 simple final OFM mismatch count=%0d", mismatch);
      end
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
    cfg_num_layers = 1;

    m1_free_col_blk_g = '0;
    m1_free_ch_blk_g = '0;
    m1_sm_refill_req_ready = 1'b1;
    m2_sm_refill_req_ready = 1'b1;

    init_mem();

    repeat (5) @(posedge clk);
    rst_n = 1'b1;

    program_single_mode2_layer();
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
      dump_ofm_region();
      $fatal(1, "TB_FAIL: DUT error before successful completion");
    end

    if (cycle_count > MAX_CYCLES) begin
      $display("TB_FAIL: timeout after %0d cycles. busy=%0b done=%0b error=%0b layer=%0d mode=%0d",
               cycle_count, busy, done, error, dbg_layer_idx, dbg_mode);
      $display("DDR counts: ifm_reads=%0d wgt_reads=%0d ofm_writes=%0d",
               ddr_ifm_read_count, ddr_wgt_read_count, ddr_ofm_write_count);
      dump_ofm_region();
      $finish;
    end

    if (!done_seen) begin
      $fatal(1, "TB_FAIL: stopped without done_seen");
    end

    $display("TB_INFO: Mode2 done after %0d cycles", cycle_count);
    $display("TB_INFO: DDR counts: ifm_reads=%0d expected=%0d, wgt_reads=%0d expected=%0d, ofm_writes=%0d expected=%0d",
             ddr_ifm_read_count, IFM_C*IFM_H,
             ddr_wgt_read_count, K_CUR*K_CUR,
             ddr_ofm_write_count, EXPECTED_OFM_DDR_WORDS);

    if (ddr_ifm_read_count != (IFM_C*IFM_H)) begin
      $fatal(1, "TB_FAIL: unexpected IFM DDR read count");
    end
    if (ddr_wgt_read_count != (K_CUR*K_CUR)) begin
      $fatal(1, "TB_FAIL: unexpected WGT DDR read count");
    end
    if (ddr_ofm_write_count != EXPECTED_OFM_DDR_WORDS) begin
      dump_ofm_region();
      $fatal(1, "TB_FAIL: unexpected OFM DDR write count");
    end

    check_final_ofm();
    dump_ofm_region();
    $display("TB_PASS: simple single-layer Mode2 pool test passed with exact final OFM check");
    $finish;
  end
endmodule
