`include "cnn_ddr_defs.svh"

// ============================================================================
// CNN top-level integration wrapper
//
// Compile this wrapper with the FIXED RTL variants already produced in this
// conversation:
//   - control_unit_top_fixed2.sv
//   - ifm_buffer_fixed.sv
//   - ofm_buffer_fixed.sv
//   - local_dataflow_manager_fixed2.sv
//   - transition_manager_fixed.sv
//   - addr_gen_ifm_m2_fixed.sv
//
// Notes:
// - This file instantiates and wires together the modules that exist today.
// - The current RTL set still keeps the same-mode refill request/ready path as
//   exported hooks from control_unit_top; there is no standalone refill-exec
//   module in the provided files. To avoid inventing a new datapath component,
//   those hooks are preserved at this top level.
// - The mode-1 free-token metadata (col-block / ch-block) is also preserved as
//   top-level inputs because the current RTL set does not synthesize their
//   source internally.
// ============================================================================
module cnn_top
  import cnn_layer_desc_pkg::*;
#(
  parameter int DATA_W           = 8,
  parameter int PSUM_W           = 32,
  parameter int PTOTAL           = 16,
  parameter int PV_MAX           = 8,
  parameter int PF_MAX           = 8,
  parameter int PC_MODE2         = 8,
  parameter int PF_MODE2         = 4,
  parameter int C_MAX            = 64,
  parameter int F_MAX            = 512,
  parameter int W_MAX            = 224,
  parameter int H_MAX            = 224,
  parameter int HT               = 8,
  parameter int K_MAX            = 7,
  parameter int WGT_DEPTH        = 4096,
  parameter int OFM_BANK_DEPTH   = H_MAX * W_MAX,
  parameter int OFM_LINEAR_DEPTH = C_MAX * OFM_BANK_DEPTH,
  parameter int CFG_DEPTH        = 64,
  parameter int DDR_ADDR_W       = `CNN_DDR_ADDR_W,
  parameter int DDR_WORD_W       = PV_MAX * DATA_W,
  parameter int WB_ADDR_W        = ((WGT_DEPTH <= 1) ? 1 : $clog2(WGT_DEPTH))
)(
  input  logic clk,
  input  logic rst_n,

  input  logic start,
  input  logic abort,

  input  logic                           cfg_wr_en,
  input  logic [$clog2(CFG_DEPTH)-1:0]   cfg_wr_addr,
  input  layer_desc_t                    cfg_wr_data,
  input  logic [$clog2(CFG_DEPTH+1)-1:0] cfg_num_layers,

  // --------------------------------------------------------------------------
  // DDR direct interface exposed by cnn_dma_direct
  // --------------------------------------------------------------------------
  output logic                      ddr_rd_req,
  output logic [DDR_ADDR_W-1:0]     ddr_rd_addr,
  input  logic                      ddr_rd_valid,
  input  logic [DDR_WORD_W-1:0]     ddr_rd_data,

  output logic                      ddr_wr_en,
  output logic [DDR_ADDR_W-1:0]     ddr_wr_addr,
  output logic [DDR_WORD_W-1:0]     ddr_wr_data,
  output logic [(DDR_WORD_W/8)-1:0] ddr_wr_be,

  // --------------------------------------------------------------------------
  // Existing same-mode refill hooks kept visible at top level
  // --------------------------------------------------------------------------
  // Mode-1 free-token metadata source that is not synthesized by the current
  // RTL set. The row-slot / row-g pulse itself comes from ifm_buffer and is
  // also exported below for synchronization.
  input  logic [15:0] m1_free_col_blk_g,
  input  logic [15:0] m1_free_ch_blk_g,

  // Same-mode refill request handshake exported by control_unit_top.
  input  logic        m1_sm_refill_req_ready,
  output logic        m1_sm_refill_req_valid,
  output logic [$clog2(HT)-1:0] m1_sm_refill_row_slot_l,
  output logic [15:0] m1_sm_refill_row_g,
  output logic [15:0] m1_sm_refill_col_blk_g,
  output logic [15:0] m1_sm_refill_ch_blk_g,

  input  logic        m2_sm_refill_req_ready,
  output logic        m2_sm_refill_req_valid,
  output logic [15:0] m2_sm_refill_row_g,
  output logic [15:0] m2_sm_refill_col_g,
  output logic [15:0] m2_sm_refill_col_l,
  output logic [15:0] m2_sm_refill_cgrp_g,

  // --------------------------------------------------------------------------
  // Useful visibility for testbench / adapter glue
  // --------------------------------------------------------------------------
  output logic                     ifm_m1_free_valid,
  output logic [$clog2(HT)-1:0]    ifm_m1_free_row_slot_l,
  output logic [15:0]              ifm_m1_free_row_g,

  output logic busy,
  output logic done,
  output logic error,

  output logic [$clog2(CFG_DEPTH)-1:0] dbg_layer_idx,
  output logic                         dbg_mode,
  output logic                         dbg_weight_bank,
  output logic [3:0]                   dbg_error_vec
);

  // Mode-1 IFM column group index is based on runtime cfg_pv_cur.
  // Worst case cfg_pv_cur = 1 => number of groups = W_MAX.
  // Must match cnn_dma_direct.ifm_dma_wr_col_idx width.
  localparam int IFM_DMA_COL_W = ((W_MAX <= 1) ? 1 : $clog2(W_MAX));
  localparam int OFM_DMA_AW    = ((OFM_LINEAR_DEPTH <= 1) ? 1 : $clog2(OFM_LINEAR_DEPTH));
  localparam int IFM_ROW_W     = ((H_MAX <= 1) ? 1 : $clog2(H_MAX));
  localparam int IFM_COL_W     = ((W_MAX <= 1) ? 1 : $clog2(W_MAX));
  localparam int IFM_M1_FREE_ROW_W = ((H_MAX+1 <= 1) ? 1 : $clog2(H_MAX+1));

  // --------------------------------------------------------------------------
  // DMA <-> buffer wires
  // --------------------------------------------------------------------------
  logic dma_busy_s, dma_done_s, dma_done_ifm_s, dma_done_wgt_s, dma_done_ofm_s, dma_error_s;
  logic dma_error_fb_q;

  logic                        dma_cfg_mode_s;
  logic [$clog2(W_MAX+1)-1:0]  dma_cfg_w_in_s;
  logic [$clog2(H_MAX+1)-1:0]  dma_cfg_h_in_s;
  logic [$clog2(C_MAX+1)-1:0]  dma_cfg_c_in_s;
  logic [$clog2(PV_MAX+1)-1:0] dma_cfg_pv_cur_s;

  logic                        ifm_cmd_start_s;
  logic [DDR_ADDR_W-1:0]       ifm_cmd_ddr_base_s;
  logic [$clog2(H_MAX+1)-1:0]  ifm_cmd_num_rows_s;
  logic [IFM_ROW_W-1:0]        ifm_cmd_buf_row_base_s;

  logic                        wgt_cmd_start_s;
  logic                        wgt_cmd_buf_sel_s;
  logic [DDR_ADDR_W-1:0]       wgt_cmd_ddr_base_s;
  logic [$clog2(WGT_DEPTH+1)-1:0] wgt_cmd_num_words_s;

  logic                        ofm_cmd_start_s;
  logic [DDR_ADDR_W-1:0]       ofm_cmd_ddr_base_s;
  logic [$clog2(OFM_LINEAR_DEPTH+1)-1:0] ofm_cmd_num_words_s;
  logic [OFM_DMA_AW-1:0]       ofm_cmd_buf_base_s;

  logic                        ifm_dma_wr_en_s;
  logic [$clog2(C_MAX)-1:0]    ifm_dma_wr_bank_s;
  logic [IFM_ROW_W-1:0]        ifm_dma_wr_row_idx_s;
  logic [IFM_DMA_COL_W-1:0]    ifm_dma_wr_col_idx_narrow_s;
  logic [PV_MAX*DATA_W-1:0]    ifm_dma_wr_data_s;
  logic [PV_MAX-1:0]           ifm_dma_wr_keep_s;
  logic                        ifm_dma_wr_ready_s;
  logic [IFM_COL_W-1:0]        ifm_dma_wr_col_idx_s;

  logic                        wgt_dma_wr_en_s;
  logic                        wgt_dma_wr_buf_sel_s;
  logic [WB_ADDR_W-1:0]        wgt_dma_wr_addr_s;
  logic [PTOTAL*DATA_W-1:0]    wgt_dma_wr_data_s;
  logic [PTOTAL-1:0]           wgt_dma_wr_keep_s;
  logic                        wgt_dma_wr_ready_s;
  logic                        wgt_dma_load_done_s;
  logic                        wgt_dma_load_buf_sel_s;

  logic                        ofm_dma_rd_en_s;
  logic [OFM_DMA_AW-1:0]       ofm_dma_rd_addr_s;
  logic                        ofm_dma_rd_valid_s;
  logic [PV_MAX*DATA_W-1:0]    ofm_dma_rd_data_s;
  logic [PV_MAX-1:0]           ofm_dma_rd_keep_s;

  // --------------------------------------------------------------------------
  // IFM buffer / local-dataflow / compute wires
  // --------------------------------------------------------------------------
  logic                        ifm_cfg_load_s;
  logic                        ifm_cfg_mode_s;
  logic [$clog2(W_MAX+1)-1:0]  ifm_cfg_w_in_s;
  logic [$clog2(H_MAX+1)-1:0]  ifm_cfg_h_in_s;
  logic [$clog2(C_MAX+1)-1:0]  ifm_cfg_c_in_s;
  logic [$clog2(PV_MAX+1)-1:0] ifm_cfg_pv_cur_s;
  logic                        ifm_m1_advance_row_s;

  logic                        ifm_rd_en_s;
  logic [$clog2(C_MAX)-1:0]    ifm_rd_bank_base_s;
  logic [IFM_ROW_W-1:0]        ifm_rd_row_idx_s;
  logic [IFM_COL_W-1:0]        ifm_rd_col_idx_s;
  logic                        ifm_rd_valid_s;
  logic [PV_MAX*DATA_W-1:0]    ifm_rd_data_s;

  logic                        ifm_ofm_wr_en_s;
  logic [$clog2(C_MAX)-1:0]    ifm_ofm_wr_bank_s;
  logic [IFM_ROW_W-1:0]        ifm_ofm_wr_row_idx_s;
  logic [IFM_COL_W-1:0]        ifm_ofm_wr_col_idx_s;
  logic [PV_MAX*DATA_W-1:0]    ifm_ofm_wr_data_s;
  logic [PV_MAX-1:0]           ifm_ofm_wr_keep_s;
  logic                        ifm_ofm_wr_ready_s;

  logic [$clog2(HT)-1:0]       ifm_m1_row_base_l_s;
  logic                        ifm_m1_free_valid_s;
  logic [$clog2(HT)-1:0]       ifm_m1_free_row_slot_l_s;
  logic [$clog2(H_MAX+1)-1:0]  ifm_m1_free_row_g_narrow_s;
  logic [15:0]                 ifm_m1_free_row_g_s;

  // --------------------------------------------------------------------------
  // Weight-buffer / compute wires
  // --------------------------------------------------------------------------
  logic                        bank0_ready_s, bank1_ready_s;
  logic                        bank0_release_s, bank1_release_s;

  logic                        m1_wb_rd_en_s;
  logic                        m1_wb_rd_buf_sel_s;
  logic [WB_ADDR_W-1:0]        m1_wb_rd_addr_s;
  logic [($clog2(PTOTAL)>0?$clog2(PTOTAL):1)-1:0] m1_wb_rd_base_lane_s;
  logic [PF_MAX*DATA_W-1:0]    m1_wb_rd_data_s;
  logic [PF_MAX-1:0]           m1_wb_rd_keep_s;
  logic                        m1_wb_rd_valid_s;

  logic                        m2_wb_rd_en_s;
  logic                        m2_wb_rd_buf_sel_s;
  logic [WB_ADDR_W-1:0]        m2_wb_rd_addr_s;
  logic [PTOTAL*DATA_W-1:0]    m2_wb_rd_data_s;
  logic [PTOTAL-1:0]           m2_wb_rd_keep_s;
  logic                        m2_wb_rd_valid_s;

  // --------------------------------------------------------------------------
  // Control <-> compute wires
  // --------------------------------------------------------------------------
  logic                        m1_start_s, m1_step_en_s;
  logic                        m1_pool_en_s;
  logic [15:0]                 m1_k_cur_s, m1_c_cur_s, m1_f_cur_s, m1_hout_cur_s, m1_wout_cur_s, m1_w_cur_s;
  logic [15:0]                 m1_pv_cur_s, m1_pf_cur_s;
  logic                        m1_weight_bank_sel_s, m1_weight_bank_ready_s;
  logic                        m1_dr_write_en_s;
  logic [$clog2(K_MAX)-1:0]    m1_dr_write_row_idx_s;
  logic [15:0]                 m1_dr_write_x_base_s;
  logic [PV_MAX*DATA_W-1:0]    m1_dr_write_data_s;

  logic [15:0]                 m1_out_row_s, m1_out_col_s, m1_f_group_s, m1_c_iter_s;
  logic [$clog2(K_MAX)-1:0]    m1_ky_s, m1_kx_s;
  logic                        m1_mac_en_s, m1_clear_psum_s, m1_ce_out_valid_s;
  logic                        m1_pass_start_pulse_s, m1_row_done_pulse_s, m1_chan_done_pulse_s;
  logic [$clog2(K_MAX)-1:0]    m1_row_done_ky_s;
  logic                        m1_f_group_done_pulse_s, m1_out_row_done_pulse_s, m1_done_s, m1_busy_s;
  logic [PV_MAX*DATA_W-1:0]    m1_ce_data_out_logic_s;
  logic signed [DATA_W-1:0]    m1_ce_weight_out_lane_s [0:PTOTAL-1];
  logic signed [PSUM_W-1:0]    m1_ce_psum_out_lane_s   [0:PTOTAL-1];
  logic signed [PSUM_W-1:0]    m1_relu_out_lane_s      [0:PTOTAL-1];
  logic                        m1_ofm_write_en_s;
  logic [15:0]                 m1_ofm_write_filter_base_s, m1_ofm_write_row_s, m1_ofm_write_col_base_s, m1_ofm_write_count_s;
  logic signed [DATA_W-1:0]    m1_ofm_write_data_s [0:PTOTAL-1];

  logic                        m2_start_s, m2_step_en_s;
  logic [15:0]                 m2_k_cur_s, m2_c_cur_s, m2_f_cur_s, m2_hout_cur_s, m2_wout_cur_s;
  logic                        m2_weight_bank_sel_s, m2_weight_bank_ready_s;
  logic                        m2_dr_write_en_s;
  logic [$clog2(K_MAX)-1:0]    m2_dr_write_row_idx_s;
  logic [PC_MODE2*DATA_W-1:0]  m2_dr_write_data_s;

  logic [15:0]                 m2_out_row_s, m2_out_col_s, m2_f_group_s, m2_c_group_s;
  logic [$clog2(K_MAX)-1:0]    m2_ky_s, m2_kx_s;
  logic                        m2_mac_en_s, m2_clear_psum_s, m2_ce_out_valid_s, m2_pass_start_pulse_s;
  logic                        m2_group_start_pulse_s, m2_row_done_pulse_s, m2_c_group_done_pulse_s;
  logic [$clog2(K_MAX)-1:0]    m2_row_done_ky_s;
  logic                        m2_pixel_done_pulse_s, m2_f_group_done_pulse_s, m2_done_s, m2_busy_s;
  logic [PC_MODE2*DATA_W-1:0]  m2_ce_data_out_logic_s;
  logic [PF_MODE2*PC_MODE2*DATA_W-1:0] m2_ce_weight_out_s;
  logic [PF_MODE2*PSUM_W-1:0]  m2_ce_mac_data_out_s;
  logic                        m2_ce_mac_data_out_valid_s;
  logic [PF_MODE2*PSUM_W-1:0]  m2_relu_data_out_s;
  logic                        m2_relu_data_out_valid_s, m2_relu_group_start_s;
  logic [15:0]                 m2_relu_f_base_s;
  logic                        m2_ofm_wr_en_s;
  logic [15:0]                 m2_ofm_wr_row_s, m2_ofm_wr_col_s, m2_ofm_wr_f_base_s;
  logic [PF_MODE2*DATA_W-1:0]  m2_ofm_wr_data_s;
  logic [15:0]                 m2_ofm_wr_row_g_s, m2_ofm_wr_col_g_s;

  // --------------------------------------------------------------------------
  // OFM buffer / control wires
  // --------------------------------------------------------------------------
  logic                        ofm_layer_start_s;
  logic                        ofm_cfg_src_mode_s, ofm_cfg_next_mode_s;
  logic                        ofm_cfg_pool_en_s;
  logic [$clog2(H_MAX+1)-1:0]  ofm_cfg_h_out_s;
  logic [$clog2(W_MAX+1)-1:0]  ofm_cfg_w_out_s;
  logic [7:0]                  ofm_cfg_f_out_s;
  logic [7:0]                  ofm_cfg_pv_cur_s;
  logic [7:0]                  ofm_cfg_pf_cur_s;
  logic [7:0]                  ofm_cfg_pv_next_s;
  logic [7:0]                  ofm_cfg_pf_next_s;

  logic                        ofm_ifm_stream_start_s;
  logic [$clog2(H_MAX+1)-1:0]  ofm_ifm_stream_row_base_s;
  logic [$clog2(H_MAX+1)-1:0]  ofm_ifm_stream_num_rows_s;
  logic [$clog2(W_MAX+1)-1:0]  ofm_ifm_stream_col_base_s;
  logic [$clog2(H_MAX)-1:0]     ofm_ifm_stream_m1_row_slot_l_s;
  logic [15:0]                  ofm_ifm_stream_m1_ch_blk_g_s;
  logic [15:0]                  ofm_ifm_stream_m2_cgrp_g_s;
  logic                        ofm_ifm_stream_busy_s, ofm_ifm_stream_done_s;

  logic [31:0]                 ofm_layer_num_words_s;
  logic                        ofm_layer_write_done_s;
  logic                        ofm_error_s;

  logic [PTOTAL-1:0]           m1_sm_ready_valid_s;
  logic [15:0]                 m1_sm_ready_bank_s      [0:PTOTAL-1];
  logic [15:0]                 m1_sm_ready_row_g_s     [0:PTOTAL-1];
  logic [15:0]                 m1_sm_ready_colgrp_g_s  [0:PTOTAL-1];

  logic [PF_MODE2-1:0]         m2_sm_ready_valid_s;
  logic [15:0]                 m2_sm_ready_bank_s      [0:PF_MODE2-1];
  logic [15:0]                 m2_sm_ready_row_g_s     [0:PF_MODE2-1];
  logic [15:0]                 m2_sm_ready_colbase_g_s [0:PF_MODE2-1];

  // --------------------------------------------------------------------------
  // Simple width adaptation
  // --------------------------------------------------------------------------
  always_comb begin
    ifm_dma_wr_col_idx_s = '0;
    ifm_dma_wr_col_idx_s[IFM_DMA_COL_W-1:0] = ifm_dma_wr_col_idx_narrow_s;
  end

  // IFM buffer exposes mode-1 free row with the minimum required width.
  // control_unit_top and the top-level debug port use 16-bit row metadata.
  // Zero-extend explicitly here to avoid high-Z upper bits in simulation.
  always_comb begin
    ifm_m1_free_row_g_s = '0;
    ifm_m1_free_row_g_s[IFM_M1_FREE_ROW_W-1:0] = ifm_m1_free_row_g_narrow_s;
  end

  assign ifm_m1_free_valid     = ifm_m1_free_valid_s;
  assign ifm_m1_free_row_slot_l= ifm_m1_free_row_slot_l_s;
  assign ifm_m1_free_row_g     = ifm_m1_free_row_g_s;

  // --------------------------------------------------------------------------
  // Registered DMA error feedback
  // --------------------------------------------------------------------------
  // Break the combinational feedback loop reported by Vivado DRC LUTLP-1:
  //   cnn_dma_direct.error -> control_unit_top/dma_phase_manager/addr_gen
  //   -> DMA command/error logic -> cnn_dma_direct.
  // DMA error is a status/control signal, so a one-cycle feedback latency is
  // acceptable and avoids an unsafe combinational loop before bitstream write.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      dma_error_fb_q <= 1'b0;
    end else begin
      dma_error_fb_q <= dma_error_s;
    end
  end

  // --------------------------------------------------------------------------
  // Control unit (integrated scheduler / local dataflow / same-mode managers)
  // --------------------------------------------------------------------------
  control_unit_top #(
    .PTOTAL          (PTOTAL),
    .DATA_W          (DATA_W),
    .PF_MAX          (PF_MAX),
    .PV_MAX          (PV_MAX),
    .PC_MODE2        (PC_MODE2),
    .PF_MODE2        (PF_MODE2),
    .C_MAX           (C_MAX),
    .W_MAX           (W_MAX),
    .H_MAX           (H_MAX),
    .HT              (HT),
    .K_MAX           (K_MAX),
    .F_MAX           (F_MAX),
    .WGT_DEPTH       (WGT_DEPTH),
    .OFM_LINEAR_DEPTH(OFM_LINEAR_DEPTH),
    .CFG_DEPTH       (CFG_DEPTH),
    .DDR_ADDR_W      (DDR_ADDR_W)
  ) u_control_unit_top (
    .clk                    (clk),
    .rst_n                  (rst_n),
    .start                  (start),
    .abort                  (abort),
    .cfg_wr_en              (cfg_wr_en),
    .cfg_wr_addr            (cfg_wr_addr),
    .cfg_wr_data            (cfg_wr_data),
    .cfg_num_layers         (cfg_num_layers),
    .dma_busy               (dma_busy_s),
    .dma_done               (dma_done_s),
    .dma_done_ifm           (dma_done_ifm_s),
    .dma_done_wgt           (dma_done_wgt_s),
    .dma_done_ofm           (dma_done_ofm_s),
    .dma_error              (dma_error_fb_q),
    .dma_cfg_mode           (dma_cfg_mode_s),
    .dma_cfg_w_in           (dma_cfg_w_in_s),
    .dma_cfg_h_in           (dma_cfg_h_in_s),
    .dma_cfg_c_in           (dma_cfg_c_in_s),
    .dma_cfg_pv_cur         (dma_cfg_pv_cur_s),
    .ifm_cmd_start          (ifm_cmd_start_s),
    .ifm_cmd_ddr_base       (ifm_cmd_ddr_base_s),
    .ifm_cmd_num_rows       (ifm_cmd_num_rows_s),
    .ifm_cmd_buf_row_base   (ifm_cmd_buf_row_base_s),
    .wgt_cmd_start          (wgt_cmd_start_s),
    .wgt_cmd_buf_sel        (wgt_cmd_buf_sel_s),
    .wgt_cmd_ddr_base       (wgt_cmd_ddr_base_s),
    .wgt_cmd_num_words      (wgt_cmd_num_words_s),
    .ofm_cmd_start          (ofm_cmd_start_s),
    .ofm_cmd_ddr_base       (ofm_cmd_ddr_base_s),
    .ofm_cmd_num_words      (ofm_cmd_num_words_s),
    .ofm_cmd_buf_base       (ofm_cmd_buf_base_s),
    .ifm_cfg_load           (ifm_cfg_load_s),
    .ifm_cfg_mode           (ifm_cfg_mode_s),
    .ifm_cfg_w_in           (ifm_cfg_w_in_s),
    .ifm_cfg_h_in           (ifm_cfg_h_in_s),
    .ifm_cfg_c_in           (ifm_cfg_c_in_s),
    .ifm_cfg_pv_cur         (ifm_cfg_pv_cur_s),
    .ifm_m1_advance_row     (ifm_m1_advance_row_s),
    .ifm_rd_en              (ifm_rd_en_s),
    .ifm_rd_bank_base       (ifm_rd_bank_base_s),
    .ifm_rd_row_idx         (ifm_rd_row_idx_s),
    .ifm_rd_col_idx         (ifm_rd_col_idx_s),
    .ifm_rd_valid           (ifm_rd_valid_s),
    .ifm_rd_data            (ifm_rd_data_s),
    .m1_free_valid          (ifm_m1_free_valid_s),
    .m1_free_row_slot_l     (ifm_m1_free_row_slot_l_s),
    .m1_free_row_g          (ifm_m1_free_row_g_s),
    .m1_free_col_blk_g      (m1_free_col_blk_g),
    .m1_free_ch_blk_g       (m1_free_ch_blk_g),
    .m2_free_valid          ('0),
    .m2_free_row_g          ('0),
    .m2_free_col_g          ('0),
    .m2_free_col_l          ('0),
    .m2_free_cgrp_g         ('0),
    .bank0_ready            (bank0_ready_s),
    .bank1_ready            (bank1_ready_s),
    .bank0_release          (bank0_release_s),
    .bank1_release          (bank1_release_s),
    .m1_start               (m1_start_s),
    .m1_step_en             (m1_step_en_s),
    .m1_pool_en             (m1_pool_en_s),
    .m1_k_cur               (m1_k_cur_s),
    .m1_c_cur               (m1_c_cur_s),
    .m1_f_cur               (m1_f_cur_s),
    .m1_hout_cur            (m1_hout_cur_s),
    .m1_wout_cur            (m1_wout_cur_s),
    .m1_w_cur               (m1_w_cur_s),
    .m1_pv_cur              (m1_pv_cur_s),
    .m1_pf_cur              (m1_pf_cur_s),
    .m1_weight_bank_sel     (m1_weight_bank_sel_s),
    .m1_weight_bank_ready   (m1_weight_bank_ready_s),
    .m1_dr_write_en         (m1_dr_write_en_s),
    .m1_dr_write_row_idx    (m1_dr_write_row_idx_s),
    .m1_dr_write_x_base     (m1_dr_write_x_base_s),
    .m1_dr_write_data       (m1_dr_write_data_s),
    .m1_out_row             (m1_out_row_s),
    .m1_out_col             (m1_out_col_s),
    .m1_f_group             (m1_f_group_s),
    .m1_c_iter              (m1_c_iter_s),
    .m1_ky                  (m1_ky_s),
    .m1_kx                  (m1_kx_s),
    .m1_mac_en              (m1_mac_en_s),
    .m1_clear_psum          (m1_clear_psum_s),
    .m1_ce_out_valid        (m1_ce_out_valid_s),
    .m1_pass_start_pulse    (m1_pass_start_pulse_s),
    .m1_row_done_pulse      (m1_row_done_pulse_s),
    .m1_row_done_ky         (m1_row_done_ky_s),
    .m1_chan_done_pulse     (m1_chan_done_pulse_s),
    .m1_f_group_done_pulse  (m1_f_group_done_pulse_s),
    .m1_out_row_done_pulse  (m1_out_row_done_pulse_s),
    .m1_done                (m1_done_s),
    .m1_busy                (m1_busy_s),
    .m2_start               (m2_start_s),
    .m2_step_en             (m2_step_en_s),
    .m2_k_cur               (m2_k_cur_s),
    .m2_c_cur               (m2_c_cur_s),
    .m2_f_cur               (m2_f_cur_s),
    .m2_hout_cur            (m2_hout_cur_s),
    .m2_wout_cur            (m2_wout_cur_s),
    .m2_weight_bank_sel     (m2_weight_bank_sel_s),
    .m2_weight_bank_ready   (m2_weight_bank_ready_s),
    .m2_dr_write_en         (m2_dr_write_en_s),
    .m2_dr_write_row_idx    (m2_dr_write_row_idx_s),
    .m2_dr_write_data       (m2_dr_write_data_s),
    .m2_out_row             (m2_out_row_s),
    .m2_out_col             (m2_out_col_s),
    .m2_f_group             (m2_f_group_s),
    .m2_c_group             (m2_c_group_s),
    .m2_ky                  (m2_ky_s),
    .m2_kx                  (m2_kx_s),
    .m2_mac_en              (m2_mac_en_s),
    .m2_clear_psum          (m2_clear_psum_s),
    .m2_ce_out_valid        (m2_ce_out_valid_s),
    .m2_pass_start_pulse    (m2_pass_start_pulse_s),
    .m2_group_start_pulse   (m2_group_start_pulse_s),
    .m2_row_done_pulse      (m2_row_done_pulse_s),
    .m2_row_done_ky         (m2_row_done_ky_s),
    .m2_c_group_done_pulse  (m2_c_group_done_pulse_s),
    .m2_pixel_done_pulse    (m2_pixel_done_pulse_s),
    .m2_f_group_done_pulse  (m2_f_group_done_pulse_s),
    .m2_done                (m2_done_s),
    .m2_busy                (m2_busy_s),
    .ofm_layer_start        (ofm_layer_start_s),
    .ofm_cfg_src_mode       (ofm_cfg_src_mode_s),
    .ofm_cfg_next_mode      (ofm_cfg_next_mode_s),
    .ofm_cfg_pool_en        (ofm_cfg_pool_en_s),
    .ofm_cfg_h_out          (ofm_cfg_h_out_s),
    .ofm_cfg_w_out          (ofm_cfg_w_out_s),
    .ofm_cfg_f_out          (ofm_cfg_f_out_s),
    .ofm_cfg_pv_cur         (ofm_cfg_pv_cur_s),
    .ofm_cfg_pf_cur         (ofm_cfg_pf_cur_s),
    .ofm_cfg_pv_next        (ofm_cfg_pv_next_s),
    .ofm_cfg_pf_next        (ofm_cfg_pf_next_s),
    .ofm_ifm_stream_start   (ofm_ifm_stream_start_s),
    .ofm_ifm_stream_row_base(ofm_ifm_stream_row_base_s),
    .ofm_ifm_stream_num_rows(ofm_ifm_stream_num_rows_s),
    .ofm_ifm_stream_col_base(ofm_ifm_stream_col_base_s),
    .ofm_ifm_stream_m1_row_slot_l(ofm_ifm_stream_m1_row_slot_l_s),
    .ofm_ifm_stream_m1_ch_blk_g(ofm_ifm_stream_m1_ch_blk_g_s),
    .ofm_ifm_stream_m2_cgrp_g(ofm_ifm_stream_m2_cgrp_g_s),
    .ofm_ifm_stream_busy    (ofm_ifm_stream_busy_s),
    .ofm_ifm_stream_done    (ofm_ifm_stream_done_s),
    .ofm_layer_num_words    (ofm_layer_num_words_s),
    .ofm_layer_write_done   (ofm_layer_write_done_s),
    .ofm_error              (ofm_error_s),
    .m1_sm_ready_valid      (m1_sm_ready_valid_s),
    .m1_sm_ready_bank       (m1_sm_ready_bank_s),
    .m1_sm_ready_row_g      (m1_sm_ready_row_g_s),
    .m1_sm_ready_colgrp_g   (m1_sm_ready_colgrp_g_s),
    .m2_sm_ready_valid      (m2_sm_ready_valid_s),
    .m2_sm_ready_bank       (m2_sm_ready_bank_s),
    .m2_sm_ready_row_g      (m2_sm_ready_row_g_s),
    .m2_sm_ready_colbase_g  (m2_sm_ready_colbase_g_s),
    .m1_sm_refill_req_valid (m1_sm_refill_req_valid),
    .m1_sm_refill_req_ready (m1_sm_refill_req_ready),
    .m1_sm_refill_row_slot_l(m1_sm_refill_row_slot_l),
    .m1_sm_refill_row_g     (m1_sm_refill_row_g),
    .m1_sm_refill_col_blk_g (m1_sm_refill_col_blk_g),
    .m1_sm_refill_ch_blk_g  (m1_sm_refill_ch_blk_g),
    .m2_sm_refill_req_valid (m2_sm_refill_req_valid),
    .m2_sm_refill_req_ready (m2_sm_refill_req_ready),
    .m2_sm_refill_row_g     (m2_sm_refill_row_g),
    .m2_sm_refill_col_g     (m2_sm_refill_col_g),
    .m2_sm_refill_col_l     (m2_sm_refill_col_l),
    .m2_sm_refill_cgrp_g    (m2_sm_refill_cgrp_g),
    .busy                   (busy),
    .done                   (done),
    .error                  (error),
    .dbg_layer_idx          (dbg_layer_idx),
    .dbg_mode               (dbg_mode),
    .dbg_weight_bank        (dbg_weight_bank),
    .dbg_error_vec          (dbg_error_vec)
  );

  // --------------------------------------------------------------------------
  // DMA block
  // --------------------------------------------------------------------------
  cnn_dma_direct #(
    .DATA_W          (DATA_W),
    .PV_MAX          (PV_MAX),
    .PC              (PC_MODE2),
    .C_MAX           (C_MAX),
    .W_MAX           (W_MAX),
    .H_MAX           (H_MAX),
    .HT              (HT),
    .PTOTAL          (PTOTAL),
    .DDR_ADDR_W      (DDR_ADDR_W),
    .DDR_WORD_W      (DDR_WORD_W),
    .WGT_WORD_LANES  (PTOTAL),
    .WGT_WORD_W      (PTOTAL*DATA_W),
    .WGT_DEPTH       (WGT_DEPTH),
    .OFM_BANK_DEPTH  (OFM_BANK_DEPTH),
    .OFM_LINEAR_DEPTH(OFM_LINEAR_DEPTH)
  ) u_cnn_dma_direct (
    .clk                (clk),
    .rst_n              (rst_n),
    .cfg_mode           (dma_cfg_mode_s),
    .cfg_w_in           (dma_cfg_w_in_s),
    .cfg_h_in           (dma_cfg_h_in_s),
    .cfg_c_in           (dma_cfg_c_in_s),
    .cfg_pv_cur         (dma_cfg_pv_cur_s),
    .ifm_cmd_start      (ifm_cmd_start_s),
    .ifm_cmd_ddr_base   (ifm_cmd_ddr_base_s),
    .ifm_cmd_num_rows   (ifm_cmd_num_rows_s),
    .ifm_cmd_buf_row_base(ifm_cmd_buf_row_base_s),
    .wgt_cmd_start      (wgt_cmd_start_s),
    .wgt_cmd_buf_sel    (wgt_cmd_buf_sel_s),
    .wgt_cmd_ddr_base   (wgt_cmd_ddr_base_s),
    .wgt_cmd_num_words  (wgt_cmd_num_words_s),
    .ofm_cmd_start      (ofm_cmd_start_s),
    .ofm_cmd_ddr_base   (ofm_cmd_ddr_base_s),
    .ofm_cmd_num_words  (ofm_cmd_num_words_s),
    .ofm_cmd_buf_base   (ofm_cmd_buf_base_s),
    .busy               (dma_busy_s),
    .done               (dma_done_s),
    .done_ifm           (dma_done_ifm_s),
    .done_wgt           (dma_done_wgt_s),
    .done_ofm           (dma_done_ofm_s),
    .error              (dma_error_s),
    .ddr_rd_req         (ddr_rd_req),
    .ddr_rd_addr        (ddr_rd_addr),
    .ddr_rd_valid       (ddr_rd_valid),
    .ddr_rd_data        (ddr_rd_data),
    .ddr_wr_en          (ddr_wr_en),
    .ddr_wr_addr        (ddr_wr_addr),
    .ddr_wr_data        (ddr_wr_data),
    .ddr_wr_be          (ddr_wr_be),
    .ifm_dma_wr_en      (ifm_dma_wr_en_s),
    .ifm_dma_wr_bank    (ifm_dma_wr_bank_s),
    .ifm_dma_wr_row_idx (ifm_dma_wr_row_idx_s),
    .ifm_dma_wr_col_idx (ifm_dma_wr_col_idx_narrow_s),
    .ifm_dma_wr_data    (ifm_dma_wr_data_s),
    .ifm_dma_wr_keep    (ifm_dma_wr_keep_s),
    .ifm_dma_wr_ready   (ifm_dma_wr_ready_s),
    .wgt_dma_wr_en      (wgt_dma_wr_en_s),
    .wgt_dma_wr_buf_sel (wgt_dma_wr_buf_sel_s),
    .wgt_dma_wr_addr    (wgt_dma_wr_addr_s),
    .wgt_dma_wr_data    (wgt_dma_wr_data_s),
    .wgt_dma_wr_keep    (wgt_dma_wr_keep_s),
    .wgt_dma_wr_ready   (wgt_dma_wr_ready_s),
    .wgt_dma_load_done  (wgt_dma_load_done_s),
    .wgt_dma_load_buf_sel(wgt_dma_load_buf_sel_s),
    .ofm_dma_rd_en      (ofm_dma_rd_en_s),
    .ofm_dma_rd_addr    (ofm_dma_rd_addr_s),
    .ofm_dma_rd_valid   (ofm_dma_rd_valid_s),
    .ofm_dma_rd_data    (ofm_dma_rd_data_s),
    .ofm_dma_rd_keep    (ofm_dma_rd_keep_s)
  );

  // --------------------------------------------------------------------------
  // IFM buffer
  // --------------------------------------------------------------------------
  ifm_buffer #(
    .DATA_W (DATA_W),
    .PV_MAX (PV_MAX),
    .PC     (PC_MODE2),
    .C_MAX  (C_MAX),
    .W_MAX  (W_MAX),
    .H_MAX  (H_MAX),
    .HT     (HT)
  ) u_ifm_buffer (
    .clk          (clk),
    .rst_n        (rst_n),
    .cfg_load     (ifm_cfg_load_s),
    .cfg_mode     (ifm_cfg_mode_s),
    .cfg_w_in     (ifm_cfg_w_in_s),
    .cfg_h_in     (ifm_cfg_h_in_s),
    .cfg_c_in     (ifm_cfg_c_in_s),
    .cfg_pv_cur   (ifm_cfg_pv_cur_s),
    .m1_advance_row(ifm_m1_advance_row_s),
    .dma_wr_en    (ifm_dma_wr_en_s),
    .dma_wr_bank  (ifm_dma_wr_bank_s),
    .dma_wr_row_idx(ifm_dma_wr_row_idx_s),
    .dma_wr_col_idx(ifm_dma_wr_col_idx_s),
    .dma_wr_data  (ifm_dma_wr_data_s),
    .dma_wr_keep  (ifm_dma_wr_keep_s),
    .ofm_wr_en    (ifm_ofm_wr_en_s),
    .ofm_wr_bank  (ifm_ofm_wr_bank_s),
    .ofm_wr_row_idx(ifm_ofm_wr_row_idx_s),
    .ofm_wr_col_idx(ifm_ofm_wr_col_idx_s),
    .ofm_wr_data  (ifm_ofm_wr_data_s),
    .ofm_wr_keep  (ifm_ofm_wr_keep_s),
    .rd_en        (ifm_rd_en_s),
    .rd_bank_base (ifm_rd_bank_base_s),
    .rd_row_idx   (ifm_rd_row_idx_s),
    .rd_col_idx   (ifm_rd_col_idx_s),
    .rd_valid     (ifm_rd_valid_s),
    .rd_data      (ifm_rd_data_s),
    .dma_wr_ready (ifm_dma_wr_ready_s),
    .ofm_wr_ready (ifm_ofm_wr_ready_s),
    .dbg_m1_row_base(),
    .m1_row_base_l(ifm_m1_row_base_l_s),
    .m1_free_valid(ifm_m1_free_valid_s),
    .m1_free_row_slot_l(ifm_m1_free_row_slot_l_s),
    .m1_free_row_g(ifm_m1_free_row_g_narrow_s)
  );

  // --------------------------------------------------------------------------
  // Weight buffer
  // --------------------------------------------------------------------------
  weight_buffer #(
    .DATA_W    (DATA_W),
    .WORD_LANES(PTOTAL),
    .PF_MAX    (PF_MAX),
    .ADDR_W    (WB_ADDR_W),
    .DEPTH     (WGT_DEPTH)
  ) u_weight_buffer (
    .clk            (clk),
    .rst_n          (rst_n),
    .dma_wr_en      (wgt_dma_wr_en_s),
    .dma_wr_buf_sel (wgt_dma_wr_buf_sel_s),
    .dma_wr_addr    (wgt_dma_wr_addr_s),
    .dma_wr_data    (wgt_dma_wr_data_s),
    .dma_wr_keep    (wgt_dma_wr_keep_s),
    .dma_wr_ready   (wgt_dma_wr_ready_s),
    .dma_load_done  (wgt_dma_load_done_s),
    .dma_load_buf_sel(wgt_dma_load_buf_sel_s),
    .bank0_release  (bank0_release_s),
    .bank1_release  (bank1_release_s),
    .m1_rd_en       (m1_wb_rd_en_s),
    .m1_rd_buf_sel  (m1_wb_rd_buf_sel_s),
    .m1_rd_addr     (m1_wb_rd_addr_s),
    .m1_rd_base_lane(m1_wb_rd_base_lane_s),
    .m1_rd_data     (m1_wb_rd_data_s),
    .m1_rd_keep     (m1_wb_rd_keep_s),
    .m1_rd_valid    (m1_wb_rd_valid_s),
    .rd_en          (m2_wb_rd_en_s),
    .rd_buf_sel     (m2_wb_rd_buf_sel_s),
    .rd_addr        (m2_wb_rd_addr_s),
    .rd_data        (m2_wb_rd_data_s),
    .rd_keep        (m2_wb_rd_keep_s),
    .rd_valid       (m2_wb_rd_valid_s),
    .bank0_ready    (bank0_ready_s),
    .bank1_ready    (bank1_ready_s)
  );

  // --------------------------------------------------------------------------
  // Compute path: mode 1
  // --------------------------------------------------------------------------
  mode1_compute_top #(
    .DATA_W    (DATA_W),
    .PSUM_W    (PSUM_W),
    .K_MAX     (K_MAX),
    .W_MAX     (W_MAX),
    .PV_MAX    (PV_MAX),
    .PF_MAX    (PF_MAX),
    .PTOTAL    (PTOTAL),
    .F_MAX     (F_MAX),
    .HOUT_MAX  (H_MAX),
    .WOUT_MAX  (W_MAX),
    .WB_ADDR_W (WB_ADDR_W)
  ) u_mode1_compute_top (
    .clk               (clk),
    .rst_n             (rst_n),
    .start             (m1_start_s),
    .step_en           (m1_step_en_s),
    .pool_en           (m1_pool_en_s),
    .K_cur             (m1_k_cur_s),
    .C_cur             (m1_c_cur_s),
    .F_cur             (m1_f_cur_s),
    .Hout_cur          (m1_hout_cur_s),
    .Wout_cur          (m1_wout_cur_s),
    .W_cur             (m1_w_cur_s),
    .Pv_cur            (m1_pv_cur_s),
    .Pf_cur            (m1_pf_cur_s),
    .dr_write_en       (m1_dr_write_en_s),
    .dr_write_row_idx  (m1_dr_write_row_idx_s),
    .dr_write_x_base   (m1_dr_write_x_base_s),
    .dr_write_data     (m1_dr_write_data_s),
    .weight_bank_sel   (m1_weight_bank_sel_s),
    .weight_bank_ready (m1_weight_bank_ready_s),
    .wb_rd_en          (m1_wb_rd_en_s),
    .wb_rd_buf_sel     (m1_wb_rd_buf_sel_s),
    .wb_rd_addr        (m1_wb_rd_addr_s),
    .wb_rd_base_lane   (m1_wb_rd_base_lane_s),
    .wb_rd_data        (m1_wb_rd_data_s),
    .wb_rd_valid       (m1_wb_rd_valid_s),
    .out_row           (m1_out_row_s),
    .out_col           (m1_out_col_s),
    .f_group           (m1_f_group_s),
    .c_iter            (m1_c_iter_s),
    .ky                (m1_ky_s),
    .kx                (m1_kx_s),
    .mac_en            (m1_mac_en_s),
    .clear_psum        (m1_clear_psum_s),
    .ce_out_valid      (m1_ce_out_valid_s),
    .pass_start_pulse  (m1_pass_start_pulse_s),
    .row_done_pulse    (m1_row_done_pulse_s),
    .row_done_ky       (m1_row_done_ky_s),
    .chan_done_pulse   (m1_chan_done_pulse_s),
    .f_group_done_pulse(m1_f_group_done_pulse_s),
    .out_row_done_pulse(m1_out_row_done_pulse_s),
    .done              (m1_done_s),
    .busy              (m1_busy_s),
    .ce_data_out_logic (m1_ce_data_out_logic_s),
    .ce_weight_out_lane(m1_ce_weight_out_lane_s),
    .ce_psum_out_lane  (m1_ce_psum_out_lane_s),
    .relu_out_lane     (m1_relu_out_lane_s),
    .ofm_write_en      (m1_ofm_write_en_s),
    .ofm_write_filter_base(m1_ofm_write_filter_base_s),
    .ofm_write_row     (m1_ofm_write_row_s),
    .ofm_write_col_base(m1_ofm_write_col_base_s),
    .ofm_write_count   (m1_ofm_write_count_s),
    .ofm_write_data    (m1_ofm_write_data_s)
  );

  // --------------------------------------------------------------------------
  // Compute path: mode 2
  // --------------------------------------------------------------------------
  mode2_compute_top #(
    .DATA_W    (DATA_W),
    .PSUM_W    (PSUM_W),
    .K_MAX     (K_MAX),
    .PC        (PC_MODE2),
    .PF        (PF_MODE2),
    .HOUT_MAX  (H_MAX),
    .WOUT_MAX  (W_MAX),
    .WB_LANES  (PTOTAL),
    .WB_ADDR_W (WB_ADDR_W)
  ) u_mode2_compute_top (
    .clk               (clk),
    .rst_n             (rst_n),
    .start             (m2_start_s),
    .step_en           (m2_step_en_s),
    .K_cur             (m2_k_cur_s),
    .C_cur             (m2_c_cur_s),
    .F_cur             (m2_f_cur_s),
    .Hout_cur          (m2_hout_cur_s),
    .Wout_cur          (m2_wout_cur_s),
    .dr_write_en       (m2_dr_write_en_s),
    .dr_write_row_idx  (m2_dr_write_row_idx_s),
    .dr_write_data     (m2_dr_write_data_s),
    .weight_bank_sel   (m2_weight_bank_sel_s),
    .weight_bank_ready (m2_weight_bank_ready_s),
    .wb_rd_en          (m2_wb_rd_en_s),
    .wb_rd_buf_sel     (m2_wb_rd_buf_sel_s),
    .wb_rd_addr        (m2_wb_rd_addr_s),
    .wb_rd_data        (m2_wb_rd_data_s),
    .wb_rd_valid       (m2_wb_rd_valid_s),
    .out_row           (m2_out_row_s),
    .out_col           (m2_out_col_s),
    .f_group           (m2_f_group_s),
    .c_group           (m2_c_group_s),
    .ky                (m2_ky_s),
    .kx                (m2_kx_s),
    .out_row_g         (),
    .out_col_g         (),
    .mac_en            (m2_mac_en_s),
    .clear_psum        (m2_clear_psum_s),
    .ce_out_valid      (m2_ce_out_valid_s),
    .pass_start_pulse  (m2_pass_start_pulse_s),
    .group_start_pulse (m2_group_start_pulse_s),
    .row_done_pulse    (m2_row_done_pulse_s),
    .row_done_ky       (m2_row_done_ky_s),
    .c_group_done_pulse(m2_c_group_done_pulse_s),
    .pixel_done_pulse  (m2_pixel_done_pulse_s),
    .f_group_done_pulse(m2_f_group_done_pulse_s),
    .done              (m2_done_s),
    .busy              (m2_busy_s),
    .ce_data_out_logic (m2_ce_data_out_logic_s),
    .ce_weight_out     (m2_ce_weight_out_s),
    .ce_mac_data_out   (m2_ce_mac_data_out_s),
    .ce_mac_data_out_valid(m2_ce_mac_data_out_valid_s),
    .relu_data_out     (m2_relu_data_out_s),
    .relu_data_out_valid(m2_relu_data_out_valid_s),
    .relu_group_start  (m2_relu_group_start_s),
    .relu_f_base       (m2_relu_f_base_s),
    .ofm_wr_en         (m2_ofm_wr_en_s),
    .ofm_wr_row        (m2_ofm_wr_row_s),
    .ofm_wr_col        (m2_ofm_wr_col_s),
    .ofm_wr_f_base     (m2_ofm_wr_f_base_s),
    .ofm_wr_data       (m2_ofm_wr_data_s),
    .ofm_wr_row_g      (m2_ofm_wr_row_g_s),
    .ofm_wr_col_g      (m2_ofm_wr_col_g_s)
  );

  // --------------------------------------------------------------------------
  // OFM buffer
  // --------------------------------------------------------------------------
  ofm_buffer #(
    .DATA_W  (DATA_W),
    .M1_IN_W (DATA_W),
    .M2_IN_W (DATA_W),
    .PV_MAX  (PV_MAX),
    .PC      (PC_MODE2),
    .PF      (PF_MODE2),
    .PTOTAL  (PTOTAL),
    .C_MAX   (C_MAX),
    .H_MAX   (H_MAX),
    .W_MAX   (W_MAX),
    .DEPTH   (OFM_BANK_DEPTH)
  ) u_ofm_buffer (
    .clk                (clk),
    .rst_n              (rst_n),
    .layer_start        (ofm_layer_start_s),
    .cfg_src_mode       (ofm_cfg_src_mode_s),
    .cfg_next_mode      (ofm_cfg_next_mode_s),
    .cfg_pool_en        (ofm_cfg_pool_en_s),
    .cfg_h_out          (ofm_cfg_h_out_s),
    .cfg_w_out          (ofm_cfg_w_out_s),
    .cfg_f_out          (ofm_cfg_f_out_s),
    .cfg_pv_cur         (ofm_cfg_pv_cur_s),
    .cfg_pf_cur         (ofm_cfg_pf_cur_s),
    .cfg_pv_next        (ofm_cfg_pv_next_s),
    .cfg_pf_next        (ofm_cfg_pf_next_s),
    .m1_wr_en           (m1_ofm_write_en_s),
    .m1_wr_filter_base  (m1_ofm_write_filter_base_s),
    .m1_wr_row          (m1_ofm_write_row_s),
    .m1_wr_col_base     (m1_ofm_write_col_base_s),
    .m1_wr_count        (m1_ofm_write_count_s),
    .m1_wr_data         (m1_ofm_write_data_s),
    .m2_wr_en           (m2_ofm_wr_en_s),
    .m2_wr_row          (m2_ofm_wr_row_g_s),
    .m2_wr_col          (m2_ofm_wr_col_g_s),
    .m2_wr_f_base       (m2_ofm_wr_f_base_s),
    .m2_wr_data         (m2_ofm_wr_data_s),
    .ifm_stream_start   (ofm_ifm_stream_start_s),
    .ifm_stream_row_base(ofm_ifm_stream_row_base_s),
    .ifm_stream_num_rows(ofm_ifm_stream_num_rows_s),
    .ifm_stream_col_base(ofm_ifm_stream_col_base_s),
    .ifm_stream_m1_row_slot_l(ofm_ifm_stream_m1_row_slot_l_s),
    .ifm_stream_m1_ch_blk_g(ofm_ifm_stream_m1_ch_blk_g_s),
    .ifm_stream_m2_cgrp_g(ofm_ifm_stream_m2_cgrp_g_s),
    .ifm_stream_busy    (ofm_ifm_stream_busy_s),
    .ifm_stream_done    (ofm_ifm_stream_done_s),
    .ifm_ofm_wr_en      (ifm_ofm_wr_en_s),
    .ifm_ofm_wr_bank    (ifm_ofm_wr_bank_s),
    .ifm_ofm_wr_row_idx (ifm_ofm_wr_row_idx_s),
    .ifm_ofm_wr_col_idx (ifm_ofm_wr_col_idx_s),
    .ifm_ofm_wr_data    (ifm_ofm_wr_data_s),
    .ifm_ofm_wr_keep    (ifm_ofm_wr_keep_s),
    .ifm_ofm_wr_ready   (ifm_ofm_wr_ready_s),
    .m1_sm_ready_valid  (m1_sm_ready_valid_s),
    .m1_sm_ready_bank   (m1_sm_ready_bank_s),
    .m1_sm_ready_row_g  (m1_sm_ready_row_g_s),
    .m1_sm_ready_colgrp_g(m1_sm_ready_colgrp_g_s),
    .m2_sm_ready_valid  (m2_sm_ready_valid_s),
    .m2_sm_ready_bank   (m2_sm_ready_bank_s),
    .m2_sm_ready_row_g  (m2_sm_ready_row_g_s),
    .m2_sm_ready_colbase_g(m2_sm_ready_colbase_g_s),
    .ofm_dma_rd_en      (ofm_dma_rd_en_s),
    .ofm_dma_rd_addr    (ofm_dma_rd_addr_s),
    .ofm_dma_rd_valid   (ofm_dma_rd_valid_s),
    .ofm_dma_rd_data    (ofm_dma_rd_data_s),
    .ofm_dma_rd_keep    (ofm_dma_rd_keep_s),
    .layer_num_words    (ofm_layer_num_words_s),
    .layer_num_pixels   (),
    .layer_pixels_written(),
    .layer_write_done   (ofm_layer_write_done_s),
    .error              (ofm_error_s)
  );

endmodule
