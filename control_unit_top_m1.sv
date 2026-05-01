`include "cnn_ddr_defs.svh"

module control_unit_top
  import cnn_layer_desc_pkg::*;
#(
  parameter int PTOTAL           = 16,
  parameter int DATA_W           = 8,
  parameter int PF_MAX           = 8,
  parameter int PV_MAX           = 8,
  parameter int PC_MODE2         = 8,
  parameter int PF_MODE2         = 4,
  parameter int C_MAX            = 64,
  parameter int W_MAX            = 224,
  parameter int H_MAX            = 224,
  parameter int HT               = 8,
  parameter int K_MAX            = 7,
  parameter int F_MAX            = 512,
  parameter int WGT_DEPTH        = 4096,
  parameter int OFM_LINEAR_DEPTH = 4096,
  parameter int CFG_DEPTH        = 64,
  parameter int DDR_ADDR_W       = `CNN_DDR_ADDR_W,
  parameter int SM_M1_RDY_Q_DEPTH = 64,
  parameter int SM_M2_RDY_Q_DEPTH = 16
)(
  input  logic clk,
  input  logic rst_n,

  input  logic start,
  input  logic abort,

  input  logic                           cfg_wr_en,
  input  logic [$clog2(CFG_DEPTH)-1:0]   cfg_wr_addr,
  input  layer_desc_t                    cfg_wr_data,
  input  logic [$clog2(CFG_DEPTH+1)-1:0] cfg_num_layers,

  input  logic dma_busy,
  input  logic dma_done,
  input  logic dma_done_ifm,
  input  logic dma_done_wgt,
  input  logic dma_done_ofm,
  input  logic dma_error,

  // Runtime config to DMA (current layer only)
  output logic                          dma_cfg_mode,
  output logic [$clog2(W_MAX+1)-1:0]    dma_cfg_w_in,
  output logic [$clog2(H_MAX+1)-1:0]    dma_cfg_h_in,
  output logic [$clog2(C_MAX+1)-1:0]    dma_cfg_c_in,
  output logic [$clog2(PV_MAX+1)-1:0]   dma_cfg_pv_cur,

  output logic                          ifm_cmd_start,
  output logic [DDR_ADDR_W-1:0]         ifm_cmd_ddr_base,
  output logic [$clog2(H_MAX+1)-1:0]    ifm_cmd_num_rows,
  output logic [((H_MAX <= 1) ? 1 : $clog2(H_MAX))-1:0] ifm_cmd_buf_row_base,

  output logic                          wgt_cmd_start,
  output logic                          wgt_cmd_buf_sel,
  output logic [DDR_ADDR_W-1:0]         wgt_cmd_ddr_base,
  output logic [$clog2(WGT_DEPTH+1)-1:0] wgt_cmd_num_words,

  output logic                          ofm_cmd_start,
  output logic [DDR_ADDR_W-1:0]         ofm_cmd_ddr_base,
  output logic [$clog2(OFM_LINEAR_DEPTH+1)-1:0] ofm_cmd_num_words,
  output logic [((OFM_LINEAR_DEPTH <= 1) ? 1 : $clog2(OFM_LINEAR_DEPTH))-1:0] ofm_cmd_buf_base,

  output logic                          ifm_cfg_load,
  output logic                          ifm_cfg_mode,
  output logic [$clog2(W_MAX+1)-1:0]    ifm_cfg_w_in,
  output logic [$clog2(H_MAX+1)-1:0]    ifm_cfg_h_in,
  output logic [$clog2(C_MAX+1)-1:0]    ifm_cfg_c_in,
  output logic [$clog2(PV_MAX+1)-1:0]   ifm_cfg_pv_cur,
  output logic                          ifm_m1_advance_row,

  output logic                          ifm_rd_en,
  output logic [$clog2(C_MAX)-1:0]      ifm_rd_bank_base,
  output logic [$clog2(H_MAX)-1:0]      ifm_rd_row_idx,
  output logic [$clog2(W_MAX)-1:0]      ifm_rd_col_idx,
  input  logic                          ifm_rd_valid,
  input  logic [PV_MAX*DATA_W-1:0]      ifm_rd_data,

  // Minimal functional visibility from ifm_buffer / IFM-side free-token path
  input  logic                          m1_free_valid,
  input  logic [$clog2(HT)-1:0]         m1_free_row_slot_l,
  input  logic [15:0]                   m1_free_row_g,
  input  logic [15:0]                   m1_free_col_blk_g,
  input  logic [15:0]                   m1_free_ch_blk_g,

  // Mode-2 IFM-side free-token source (must come from local dataflow / compute side)
  input  logic                          m2_free_valid,
  input  logic [15:0]                   m2_free_row_g,
  input  logic [15:0]                   m2_free_col_g,
  input  logic [15:0]                   m2_free_col_l,
  input  logic [15:0]                   m2_free_cgrp_g,

  input  logic bank0_ready,
  input  logic bank1_ready,
  output logic bank0_release,
  output logic bank1_release,

  output logic                          m1_start,
  output logic                          m1_step_en,
  output logic [3:0]                    m1_k_cur,
  output logic [7:0]                    m1_c_cur,
  output logic [7:0]                    m1_f_cur,
  output logic [15:0]                   m1_hout_cur,
  output logic [15:0]                   m1_wout_cur,
  output logic [15:0]                   m1_w_cur,
  output logic [7:0]                    m1_pv_cur,
  output logic [7:0]                    m1_pf_cur,
  output logic                          m1_weight_bank_sel,
  output logic                          m1_weight_bank_ready,

  output logic                          m1_dr_write_en,
  output logic [$clog2(K_MAX)-1:0]      m1_dr_write_row_idx,
  output logic [15:0]                   m1_dr_write_x_base,
  output logic [PV_MAX*DATA_W-1:0]      m1_dr_write_data,

  input  logic [15:0]                   m1_out_row,
  input  logic [15:0]                   m1_out_col,
  input  logic [15:0]                   m1_f_group,
  input  logic [15:0]                   m1_c_iter,
  input  logic [7:0]                    m1_ky,
  input  logic [7:0]                    m1_kx,
  input  logic                          m1_mac_en,
  input  logic                          m1_clear_psum,
  input  logic                          m1_ce_out_valid,
  input  logic                          m1_pass_start_pulse,
  input  logic                          m1_row_done_pulse,
  input  logic [7:0]                    m1_row_done_ky,
  input  logic                          m1_chan_done_pulse,
  input  logic                          m1_f_group_done_pulse,
  input  logic                          m1_out_row_done_pulse,
  input  logic                          m1_done,
  input  logic                          m1_busy,

  output logic                          m2_start,
  output logic                          m2_step_en,
  output logic [3:0]                    m2_k_cur,
  output logic [7:0]                    m2_c_cur,
  output logic [7:0]                    m2_f_cur,
  output logic [15:0]                   m2_hout_cur,
  output logic [15:0]                   m2_wout_cur,
  output logic                          m2_weight_bank_sel,
  output logic                          m2_weight_bank_ready,

  output logic                          m2_dr_write_en,
  output logic [$clog2(K_MAX)-1:0]      m2_dr_write_row_idx,
  output logic [PC_MODE2*DATA_W-1:0]    m2_dr_write_data,

  input  logic [15:0]                   m2_out_row,
  input  logic [15:0]                   m2_out_col,
  input  logic [15:0]                   m2_f_group,
  input  logic [15:0]                   m2_c_group,
  input  logic [7:0]                    m2_ky,
  input  logic [7:0]                    m2_kx,
  input  logic                          m2_mac_en,
  input  logic                          m2_clear_psum,
  input  logic                          m2_ce_out_valid,
  input  logic                          m2_pass_start_pulse,
  input  logic                          m2_group_start_pulse,
  input  logic                          m2_row_done_pulse,
  input  logic [7:0]                    m2_row_done_ky,
  input  logic                          m2_c_group_done_pulse,
  input  logic                          m2_pixel_done_pulse,
  input  logic                          m2_f_group_done_pulse,
  input  logic                          m2_done,
  input  logic                          m2_busy,

  output logic                          ofm_layer_start,
  output logic                          ofm_cfg_src_mode,
  output logic                          ofm_cfg_next_mode,
  output logic [$clog2(H_MAX+1)-1:0]    ofm_cfg_h_out,
  output logic [$clog2(W_MAX+1)-1:0]    ofm_cfg_w_out,
  output logic [7:0]   ofm_cfg_f_out,
  output logic [7:0]   ofm_cfg_pv_cur,
  output logic [7:0]   ofm_cfg_pf_cur,
  output logic [7:0]   ofm_cfg_pv_next,
  output logic [7:0]   ofm_cfg_pf_next,

  output logic                          ofm_ifm_stream_start,
  output logic [$clog2(H_MAX+1)-1:0]    ofm_ifm_stream_row_base,
  output logic [$clog2(H_MAX+1)-1:0]    ofm_ifm_stream_num_rows,
  output logic [$clog2(W_MAX+1)-1:0]    ofm_ifm_stream_col_base,
  output logic [$clog2(H_MAX)-1:0]      ofm_ifm_stream_m1_row_slot_l,
  output logic [15:0]                   ofm_ifm_stream_m1_ch_blk_g,
  output logic [15:0]                   ofm_ifm_stream_m2_cgrp_g,

  input  logic                          ofm_ifm_stream_busy,
  input  logic                          ofm_ifm_stream_done,
  input  logic [31:0]                   ofm_layer_num_words,
  input  logic                          ofm_layer_write_done,
  input  logic                          ofm_error,

  // Same-mode ready-token visibility from ofm_buffer
  input  logic [PTOTAL-1:0]             m1_sm_ready_valid,
  input  logic [15:0]                   m1_sm_ready_bank [0:PTOTAL-1],
  input  logic [15:0]                   m1_sm_ready_row_g [0:PTOTAL-1],
  input  logic [15:0]                   m1_sm_ready_colgrp_g [0:PTOTAL-1],

  input  logic [PF_MODE2-1:0]           m2_sm_ready_valid,
  input  logic [15:0]                   m2_sm_ready_bank [0:PF_MODE2-1],
  input  logic [15:0]                   m2_sm_ready_row_g [0:PF_MODE2-1],
  input  logic [15:0]                   m2_sm_ready_colbase_g [0:PF_MODE2-1],

  // Same-mode refill requests exported to datapath/refill adapter
  output logic                          m1_sm_refill_req_valid,
  input  logic                          m1_sm_refill_req_ready,
  output logic [$clog2(HT)-1:0]         m1_sm_refill_row_slot_l,
  output logic [15:0]                   m1_sm_refill_row_g,
  output logic [15:0]                   m1_sm_refill_col_blk_g,
  output logic [15:0]                   m1_sm_refill_ch_blk_g,

  output logic                          m2_sm_refill_req_valid,
  input  logic                          m2_sm_refill_req_ready,
  output logic [15:0]                   m2_sm_refill_row_g,
  output logic [15:0]                   m2_sm_refill_col_g,
  output logic [15:0]                   m2_sm_refill_col_l,
  output logic [15:0]                   m2_sm_refill_cgrp_g,

  output logic                          busy,
  output logic                          done,
  output logic                          error,

  output logic [$clog2(CFG_DEPTH)-1:0]  dbg_layer_idx,
  output logic                          dbg_mode,
  output logic                          dbg_weight_bank,
  output logic [3:0]                    dbg_error_vec
);

  localparam int CFG_AW    = (CFG_DEPTH > 1) ? $clog2(CFG_DEPTH) : 1;
  localparam int ROW_W     = (H_MAX <= 1) ? 1 : $clog2(H_MAX+1);
  localparam int COL_W     = (W_MAX <= 1) ? 1 : $clog2(W_MAX+1);
  localparam int BUF_ROW_W = (H_MAX <= 1) ? 1 : $clog2(H_MAX);
  localparam int TILE_W    = (((W_MAX + PC_MODE2 - 1) / PC_MODE2) <= 1) ? 1 : $clog2(((W_MAX + PC_MODE2 - 1) / PC_MODE2) + 1);
  localparam int M1Q_AW    = (SM_M1_RDY_Q_DEPTH <= 1) ? 1 : $clog2(SM_M1_RDY_Q_DEPTH+1);
  localparam int M2Q_AW    = (SM_M2_RDY_Q_DEPTH <= 1) ? 1 : $clog2(SM_M2_RDY_Q_DEPTH+1);
  localparam int M1FQ_AW   = (HT <= 1) ? 1 : $clog2(HT+1);

  typedef struct packed {
    logic [15:0] row_g;
    logic [15:0] col_blk_g;
    logic [15:0] ch_blk_g;
  } m1_rdy_tok_t;

  typedef struct packed {
    logic [15:0] row_g;
    logic [15:0] col_g;
    logic [15:0] cgrp_g;
  } m2_rdy_tok_t;

  logic start_q, start_pulse;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      start_q <= 1'b0;
    else
      start_q <= start;
  end
  assign start_pulse = start && !start_q;

  logic              cur_valid_s, next_valid_s;
  logic [CFG_AW-1:0] cur_layer_idx_s;
  layer_desc_t       cur_cfg_s, next_cfg_s;
  logic              cur_first_layer_s, cur_last_layer_s;

  logic compute_bank_sel_s, preload_bank_sel_s, compute_bank_ready_s;

  logic kick_ifm_load_s, kick_wgt_preload_s, kick_compute_s;
  logic kick_same_mode_stream_s, kick_transition_stream_s, kick_ofm_store_s;
  logic sched_hold_compute_s;
  logic advance_layer_s, swap_weight_bank_s;
  logic sched_busy_s, sched_done_s, sched_error_s;

  logic ifm_load_done_s, wgt_load_done_s, ofm_store_done_s, phase_error_s;
  logic local_hold_compute_s, local_busy_s, local_done_s, local_error_s;
  logic transition_done_s, transition_busy_s, transition_error_s;
  logic same_mode_drain_done_s, sched_next_path_done_s;
  logic compute_done_s, compute_busy_s;
  logic any_error_s;
  logic control_error_s;

  logic [ROW_W-1:0] ifm_req_abs_row_base_s;
  logic [ROW_W-1:0] ifm_req_num_rows_s;
  logic [BUF_ROW_W-1:0] ifm_req_buf_row_base_s;
  logic [TILE_W-1:0] ifm_req_m2_tile_idx_s;

  logic [ROW_W-1:0] stream_req_row_base_s;
  logic [ROW_W-1:0] stream_req_num_rows_s;
  logic [COL_W-1:0] stream_req_col_base_s;

  logic use_next_ifm_cfg;

  // Mode-2 global-coordinate aliases
  logic [15:0] m2_out_row_g_s, m2_out_col_g_s;
  assign m2_out_row_g_s = m2_out_row;
  assign m2_out_col_g_s = m2_out_col;

  // Internal mode-2 free-token source from local_dataflow_manager.
  // Keep top-level m2_free_* ports unchanged for compatibility, but the
  // same-mode M2 refill path in this integrated control unit must consume the
  // local-dataflow token stream generated when data_register_mode2 captures a
  // tuple successfully.
  logic        ldm_m2_free_valid_s;
  logic [15:0] ldm_m2_free_row_g_s;
  logic [15:0] ldm_m2_free_col_g_s;
  logic [15:0] ldm_m2_free_col_l_s;
  logic [15:0] ldm_m2_free_cgrp_g_s;

  // Internal mode-1 free-token metadata expander.
  // ifm_buffer only reports that one physical row slot became free; control_unit
  // expands that row event into the (row_slot,row_g,col_blk_g,ch_blk_g) tuples
  // expected by same_mode_refill_manager_m1.
  logic [$clog2(HT)-1:0] m1f_row_slot_fifo [0:HT-1];
  logic [15:0]           m1f_row_g_fifo    [0:HT-1];
  logic [M1FQ_AW-1:0]    m1f_head_q, m1f_tail_q, m1f_count_q;
  logic                  m1f_scan_active_q;
  logic [$clog2(HT)-1:0] m1f_scan_row_slot_q;
  logic [15:0]           m1f_scan_row_g_q, m1f_scan_col_blk_q, m1f_scan_ch_blk_q;
  logic                  m1f_overflow_q;
  logic                  m1f_emit_valid_s;
  logic [$clog2(HT)-1:0] m1f_emit_row_slot_l_s;
  logic [15:0]           m1f_emit_row_g_s, m1f_emit_col_blk_g_s, m1f_emit_ch_blk_g_s;
  logic [15:0]           m1_next_num_col_blks_s, m1_next_num_ch_blks_s;

  // True current-layer OFM size as stored in ofm_buffer for mode 1.
  // Keep these expressions aligned with ofm_cfg_h_out/ofm_cfg_w_out so the
  // same-mode M1 refill manager does not wait for raw pre-pool dimensions.
  logic [15:0]           cur_m1_ofm_h_for_sm_s;
  logic [15:0]           cur_m1_ofm_w_for_sm_s;

  // Internal same-mode refill requests from managers; top-level ports remain as
  // visibility hooks only.
  logic                  m1_sm_req_valid_i, m1_sm_req_ready_i;
  logic [$clog2(HT)-1:0] m1_sm_row_slot_l_i;
  logic [15:0]           m1_sm_row_g_i, m1_sm_col_blk_g_i, m1_sm_ch_blk_g_i;
  logic                  m2_sm_req_valid_i, m2_sm_req_ready_i;
  logic [15:0]           m2_sm_row_g_i, m2_sm_col_g_i, m2_sm_col_l_i, m2_sm_cgrp_g_i;

  // Transition-manager command is muxed with internal same-mode stream command.
  logic                  trans_ifm_stream_start_s;
  logic [ROW_W-1:0]      trans_ifm_stream_row_base_s;
  logic [ROW_W-1:0]      trans_ifm_stream_num_rows_s;
  logic [COL_W-1:0]      trans_ifm_stream_col_base_s;

  logic                  sm_stream_start_s;
  logic [ROW_W-1:0]      sm_stream_row_base_s;
  logic [ROW_W-1:0]      sm_stream_num_rows_s;
  logic [COL_W-1:0]      sm_stream_col_base_s;
  logic [BUF_ROW_W-1:0]  sm_stream_m1_row_slot_l_s;
  logic [15:0]           sm_stream_m1_ch_blk_g_s;
  logic [15:0]           sm_stream_m2_cgrp_g_s;
  logic                  sm_exec_active_q, sm_exec_mode_q;
  logic [$clog2(HT)-1:0] sm_exec_m1_row_slot_l_q;
  logic [15:0]           sm_exec_row_g_q, sm_exec_col_base_g_q;

  // Same-mode activity windows
  logic sm_m1_active, sm_m2_active;
  logic rst_n_sm_m1, rst_n_sm_m2;
  assign sm_m1_active = next_valid_s && (cur_cfg_s.mode == MODE1) && (next_cfg_s.mode == MODE1);
  assign sm_m2_active = next_valid_s && (cur_cfg_s.mode == MODE2) && (next_cfg_s.mode == MODE2);
  assign rst_n_sm_m1  = rst_n && sm_m1_active;
  assign rst_n_sm_m2  = rst_n && sm_m2_active;

  // --------------------------------------------------------------------------
  // Ready-token serializer queues for OFM-side vector tokens
  // --------------------------------------------------------------------------
  m1_rdy_tok_t m1q_mem [0:SM_M1_RDY_Q_DEPTH-1];
  logic [M1Q_AW-1:0] m1q_count_q;
  logic              m1q_overflow_q;

  m2_rdy_tok_t m2q_mem [0:SM_M2_RDY_Q_DEPTH-1];
  logic [M2Q_AW-1:0] m2q_count_q;
  logic              m2q_overflow_q;

  // Scalarized ready tokens into managers
  logic        m1_ready_tok_valid_s;
  logic [15:0] m1_ready_tok_row_g_s, m1_ready_tok_col_blk_g_s, m1_ready_tok_ch_blk_g_s;
  logic        m2_ready_tok_valid_s;
  logic [15:0] m2_ready_tok_row_g_s, m2_ready_tok_col_g_s, m2_ready_tok_cgrp_g_s;

  // Manager status
  logic m1_sm_busy_s, m1_sm_error_s, m1_sm_free_full_s, m1_sm_ready_full_s;
  logic m2_sm_busy_s, m2_sm_error_s, m2_sm_free_full_s, m2_sm_ready_full_s;

  assign dma_cfg_mode   = (cur_cfg_s.mode == MODE2);
  assign dma_cfg_w_in   = cur_cfg_s.w_in[$clog2(W_MAX+1)-1:0];
  assign dma_cfg_h_in   = cur_cfg_s.h_in[$clog2(H_MAX+1)-1:0];
  assign dma_cfg_c_in   = cur_cfg_s.c_in[$clog2(C_MAX+1)-1:0];
  assign dma_cfg_pv_cur = cur_cfg_s.pv_m1[$clog2(PV_MAX+1)-1:0];

  always_comb begin
    ifm_req_abs_row_base_s = '0;
    ifm_req_buf_row_base_s = '0;
    ifm_req_m2_tile_idx_s  = '0;

    if (cur_cfg_s.mode == MODE1) begin
      if (cur_cfg_s.h_in < HT[ROW_W-1:0])
        ifm_req_num_rows_s = cur_cfg_s.h_in[ROW_W-1:0];
      else
        ifm_req_num_rows_s = HT[ROW_W-1:0];
    end
    else begin
      ifm_req_num_rows_s = cur_cfg_s.h_in[ROW_W-1:0];
    end

    stream_req_row_base_s = '0;
    stream_req_num_rows_s = cur_cfg_s.h_out[ROW_W-1:0];
    stream_req_col_base_s = '0;
  end

  // Same-mode refill no longer switches IFM to the next-layer config mid-pass.
  // Instead, while the current layer is running we keep the current mode/read
  // view and widen only the write-side bounds that must also accommodate the
  // next layer. The true config handoff still happens only on layer advance or
  // on the special mode1->mode2 transition stream.
  assign use_next_ifm_cfg = next_valid_s && (advance_layer_s || kick_transition_stream_s);

  assign m1_next_num_col_blks_s = (next_cfg_s.pv_m1 == 0) ? 16'd1 : ((next_cfg_s.w_in + next_cfg_s.pv_m1 - 1) / next_cfg_s.pv_m1);
  assign m1_next_num_ch_blks_s  = (next_cfg_s.pf_m1 == 0) ? 16'd1 : ((next_cfg_s.c_in + next_cfg_s.pf_m1 - 1) / next_cfg_s.pf_m1);

  always_comb begin
    logic [$clog2(C_MAX+1)-1:0] ifm_cur_c_in_v;
    logic [$clog2(H_MAX+1)-1:0] ifm_cur_h_in_v;

    ifm_cfg_load = kick_ifm_load_s || use_next_ifm_cfg;

    ifm_cur_c_in_v = cur_cfg_s.c_in[$clog2(C_MAX+1)-1:0];
    ifm_cur_h_in_v = cur_cfg_s.h_in[$clog2(H_MAX+1)-1:0];

    if (sm_m1_active || sm_m2_active) begin
      if (next_cfg_s.c_in[$clog2(C_MAX+1)-1:0] > ifm_cur_c_in_v)
        ifm_cur_c_in_v = next_cfg_s.c_in[$clog2(C_MAX+1)-1:0];
      if (sm_m2_active && (next_cfg_s.h_in[$clog2(H_MAX+1)-1:0] > ifm_cur_h_in_v))
        ifm_cur_h_in_v = next_cfg_s.h_in[$clog2(H_MAX+1)-1:0];
    end

    if (use_next_ifm_cfg) begin
      ifm_cfg_mode   = (next_cfg_s.mode == MODE2);
      ifm_cfg_w_in   = next_cfg_s.w_in[$clog2(W_MAX+1)-1:0];
      ifm_cfg_h_in   = next_cfg_s.h_in[$clog2(H_MAX+1)-1:0];
      ifm_cfg_c_in   = next_cfg_s.c_in[$clog2(C_MAX+1)-1:0];
      ifm_cfg_pv_cur = next_cfg_s.pv_m1[$clog2(PV_MAX+1)-1:0];
    end
    else begin
      ifm_cfg_mode   = (cur_cfg_s.mode == MODE2);
      ifm_cfg_w_in   = cur_cfg_s.w_in[$clog2(W_MAX+1)-1:0];
      ifm_cfg_h_in   = ifm_cur_h_in_v;
      ifm_cfg_c_in   = ifm_cur_c_in_v;
      ifm_cfg_pv_cur = cur_cfg_s.pv_m1[$clog2(PV_MAX+1)-1:0];
    end
  end

  assign ifm_m1_advance_row = (cur_cfg_s.mode == MODE1) ? m1_out_row_done_pulse : 1'b0;

  assign ofm_layer_start  = kick_compute_s;
  assign ofm_cfg_src_mode = (cur_cfg_s.mode == MODE2);
  assign ofm_cfg_h_out    = (cur_cfg_s.mode == MODE1)
                            ? ((cur_cfg_s.h_out > 1) ? (cur_cfg_s.h_out >> 1) : 16'd1)
                            : cur_cfg_s.h_out[$clog2(H_MAX+1)-1:0];
  assign ofm_cfg_w_out    = (cur_cfg_s.mode == MODE1)
                            ? ((cur_cfg_s.w_out > 1) ? (cur_cfg_s.w_out >> 1) : 16'd1)
                            : cur_cfg_s.w_out[$clog2(W_MAX+1)-1:0];
  assign cur_m1_ofm_h_for_sm_s = (cur_cfg_s.h_out > 16'd1) ? (cur_cfg_s.h_out >> 1) : 16'd1;
  assign cur_m1_ofm_w_for_sm_s = (cur_cfg_s.w_out > 16'd1) ? (cur_cfg_s.w_out >> 1) : 16'd1;

  assign ofm_cfg_f_out    = cur_cfg_s.f_out[7:0];
  assign ofm_cfg_pv_cur   = cur_cfg_s.pv_m1[7:0];
  assign ofm_cfg_pf_cur   = (cur_cfg_s.mode == MODE1) ? cur_cfg_s.pf_m1[7:0] : cur_cfg_s.pf_m2[7:0];

  // For the final layer there is no real "next" IFM layout, but ofm_buffer still
  // requires a non-zero same-mode-compatible fallback to avoid flagging a false
  // configuration error on cfg_pv_next/cfg_pf_next.
  assign ofm_cfg_next_mode = next_valid_s ? (next_cfg_s.mode == MODE2) : (cur_cfg_s.mode == MODE2);
  assign ofm_cfg_pv_next   = next_valid_s ? next_cfg_s.pv_m1[7:0] : 8'd1;
  assign ofm_cfg_pf_next   = next_valid_s ?
                             ((next_cfg_s.mode == MODE1) ? next_cfg_s.pf_m1[7:0] : next_cfg_s.pf_m2[7:0]) :
                             ((cur_cfg_s.mode == MODE1) ? ((cur_cfg_s.pf_m1[7:0] != 8'd0) ? cur_cfg_s.pf_m1[7:0] : 8'd1) :
                                                         ((cur_cfg_s.pf_m2[7:0] != 8'd0) ? cur_cfg_s.pf_m2[7:0] : 8'd1));

  // --------------------------------------------------------------------------
  // M1 ready-token queue: captures all vector tokens from OFM side
  // and emits one scalar token/cycle to same_mode_refill_manager_m1.
  // --------------------------------------------------------------------------
  integer j, wr_idx;
  always_ff @(posedge clk or negedge rst_n_sm_m1) begin
    if (!rst_n_sm_m1) begin
      m1q_count_q    <= '0;
      m1q_overflow_q <= 1'b0;
      for (int i = 0; i < SM_M1_RDY_Q_DEPTH; i++) begin
        m1q_mem[i] <= '0;
      end
    end
    else begin
      // pop one token if we are issuing to manager this cycle
      if (m1_ready_tok_valid_s) begin
        for (j = 0; j < SM_M1_RDY_Q_DEPTH-1; j++) begin
          m1q_mem[j] <= m1q_mem[j+1];
        end
        m1q_mem[SM_M1_RDY_Q_DEPTH-1] <= '0;
        if (m1q_count_q != 0)
          m1q_count_q <= m1q_count_q - 1'b1;
      end

      // append incoming valid vector tokens
      wr_idx = m1q_count_q - (m1_ready_tok_valid_s ? 1 : 0);
      for (int i = 0; i < PTOTAL; i++) begin
        if (sm_m1_active && m1_sm_ready_valid[i]) begin
          if (wr_idx < SM_M1_RDY_Q_DEPTH) begin
            m1q_mem[wr_idx].row_g     <= m1_sm_ready_row_g[i];
            m1q_mem[wr_idx].col_blk_g <= m1_sm_ready_colgrp_g[i];
            m1q_mem[wr_idx].ch_blk_g  <= m1_sm_ready_bank[i];
            wr_idx = wr_idx + 1;
          end
          else begin
            m1q_overflow_q <= 1'b1;
          end
        end
      end
      m1q_count_q <= wr_idx[M1Q_AW-1:0];
    end
  end

  assign m1_ready_tok_valid_s     = sm_m1_active && (m1q_count_q != 0) && !m1_sm_ready_full_s;
  assign m1_ready_tok_row_g_s     = m1q_mem[0].row_g;
  assign m1_ready_tok_col_blk_g_s = m1q_mem[0].col_blk_g;
  assign m1_ready_tok_ch_blk_g_s  = m1q_mem[0].ch_blk_g;

  // --------------------------------------------------------------------------
  // M2 ready-token queue
  // --------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n_sm_m2) begin
    if (!rst_n_sm_m2) begin
      m2q_count_q    <= '0;
      m2q_overflow_q <= 1'b0;
      for (int i = 0; i < SM_M2_RDY_Q_DEPTH; i++) begin
        m2q_mem[i] <= '0;
      end
    end
    else begin
      if (m2_ready_tok_valid_s) begin
        for (j = 0; j < SM_M2_RDY_Q_DEPTH-1; j++) begin
          m2q_mem[j] <= m2q_mem[j+1];
        end
        m2q_mem[SM_M2_RDY_Q_DEPTH-1] <= '0;
        if (m2q_count_q != 0)
          m2q_count_q <= m2q_count_q - 1'b1;
      end

      wr_idx = m2q_count_q - (m2_ready_tok_valid_s ? 1 : 0);
      for (int i = 0; i < PF_MODE2; i++) begin
        if (sm_m2_active && m2_sm_ready_valid[i]) begin
          if (wr_idx < SM_M2_RDY_Q_DEPTH) begin
            m2q_mem[wr_idx].row_g  <= m2_sm_ready_row_g[i];
            // ofm_buffer exports the GLOBAL column base of the ready PC-wide
            // segment on the legacy-named *_colbase_g port; manager_m2 matches
            // that value as ready_col_g.
            m2q_mem[wr_idx].col_g  <= m2_sm_ready_colbase_g[i];
            m2q_mem[wr_idx].cgrp_g <= m2_sm_ready_bank[i];
            wr_idx = wr_idx + 1;
          end
          else begin
            m2q_overflow_q <= 1'b1;
          end
        end
      end
      m2q_count_q <= wr_idx[M2Q_AW-1:0];
    end
  end

  assign m2_ready_tok_valid_s = sm_m2_active && (m2q_count_q != 0) && !m2_sm_ready_full_s;
  assign m2_ready_tok_row_g_s = m2q_mem[0].row_g;
  assign m2_ready_tok_col_g_s = m2q_mem[0].col_g;
  assign m2_ready_tok_cgrp_g_s= m2q_mem[0].cgrp_g;

  // --------------------------------------------------------------------------
  // Mode-1 free-token metadata expander
  // --------------------------------------------------------------------------
  assign m1f_emit_valid_s      = sm_m1_active && m1f_scan_active_q && !m1_sm_free_full_s;
  assign m1f_emit_row_slot_l_s = m1f_scan_row_slot_q;
  assign m1f_emit_row_g_s      = m1f_scan_row_g_q;
  assign m1f_emit_col_blk_g_s  = m1f_scan_col_blk_q;
  assign m1f_emit_ch_blk_g_s   = m1f_scan_ch_blk_q;

  always_ff @(posedge clk or negedge rst_n_sm_m1) begin : PROC_M1_FREE_EXPAND
    integer cnt_tmp, head_tmp, tail_tmp;
    integer num_col_tmp, num_ch_tmp;
    logic   take_free_now;

    if (!rst_n_sm_m1) begin
      m1f_head_q         <= '0;
      m1f_tail_q         <= '0;
      m1f_count_q        <= '0;
      m1f_scan_active_q  <= 1'b0;
      m1f_scan_row_slot_q<= '0;
      m1f_scan_row_g_q   <= '0;
      m1f_scan_col_blk_q <= '0;
      m1f_scan_ch_blk_q  <= '0;
      m1f_overflow_q     <= 1'b0;
    end
    else begin
      cnt_tmp  = m1f_count_q;
      head_tmp = m1f_head_q;
      tail_tmp = m1f_tail_q;

      // Accept only free rows that belong to the next layer IFM.
      // In M1->M1 with pooling, the current layer can free more conv rows than
      // the next layer needs after pooling. Passing those extra rows to the
      // same-mode manager leaves unmatched free tokens and can keep it busy.
      take_free_now = m1_free_valid &&
                      sm_m1_active &&
                      (m1_free_row_g < next_cfg_s.h_in);

      // If the pending-free FIFO is empty and the scanner is idle, consume the
      // incoming free token directly. Do not enqueue then dequeue in the same
      // clock, because the FIFO write uses nonblocking assignment and the read
      // would see the old/X value.
      if (!m1f_scan_active_q && (cnt_tmp == 0) && take_free_now) begin
        m1f_scan_active_q   <= 1'b1;
        m1f_scan_row_slot_q <= m1_free_row_slot_l;
        m1f_scan_row_g_q    <= m1_free_row_g;
        m1f_scan_col_blk_q  <= '0;
        m1f_scan_ch_blk_q   <= '0;
      end
      else begin
        if (take_free_now) begin
          if (cnt_tmp < HT) begin
            m1f_row_slot_fifo[tail_tmp] <= m1_free_row_slot_l;
            m1f_row_g_fifo[tail_tmp]    <= m1_free_row_g;
            tail_tmp = (tail_tmp + 1) % HT;
            cnt_tmp  = cnt_tmp + 1;
          end
          else begin
            m1f_overflow_q <= 1'b1;
          end
        end

        if (!m1f_scan_active_q && (cnt_tmp > 0)) begin
          m1f_scan_active_q   <= 1'b1;
          m1f_scan_row_slot_q <= m1f_row_slot_fifo[head_tmp];
          m1f_scan_row_g_q    <= m1f_row_g_fifo[head_tmp];
          m1f_scan_col_blk_q  <= '0;
          m1f_scan_ch_blk_q   <= '0;
          head_tmp = (head_tmp + 1) % HT;
          cnt_tmp  = cnt_tmp - 1;
        end
        else if (m1f_emit_valid_s) begin
          num_col_tmp = (m1_next_num_col_blks_s == 0) ? 1 : m1_next_num_col_blks_s;
          num_ch_tmp  = (m1_next_num_ch_blks_s  == 0) ? 1 : m1_next_num_ch_blks_s;

          if ((m1f_scan_ch_blk_q + 1) < num_ch_tmp[15:0]) begin
            m1f_scan_ch_blk_q <= m1f_scan_ch_blk_q + 1'b1;
          end
          else begin
            m1f_scan_ch_blk_q <= '0;
            if ((m1f_scan_col_blk_q + 1) < num_col_tmp[15:0]) begin
              m1f_scan_col_blk_q <= m1f_scan_col_blk_q + 1'b1;
            end
            else begin
              m1f_scan_col_blk_q <= '0;
              m1f_scan_active_q  <= 1'b0;
            end
          end
        end
      end

      m1f_head_q  <= head_tmp[M1FQ_AW-1:0];
      m1f_tail_q  <= tail_tmp[M1FQ_AW-1:0];
      m1f_count_q <= cnt_tmp[M1FQ_AW-1:0];
    end
  end

  // --------------------------------------------------------------------------
  // Internal same-mode refill command controller
  // --------------------------------------------------------------------------
  always_comb begin
    sm_stream_start_s         = 1'b0;
    sm_stream_row_base_s      = '0;
    sm_stream_num_rows_s      = '0;
    sm_stream_col_base_s      = '0;
    sm_stream_m1_row_slot_l_s = '0;
    sm_stream_m1_ch_blk_g_s   = '0;
    sm_stream_m2_cgrp_g_s     = '0;

    if (!sm_exec_active_q && !transition_busy_s && !trans_ifm_stream_start_s && !ofm_ifm_stream_busy) begin
      if (sm_m1_active && m1_sm_req_valid_i) begin
        sm_stream_start_s         = 1'b1;
        sm_stream_row_base_s      = m1_sm_row_g_i[ROW_W-1:0];
        sm_stream_num_rows_s      = ROW_W'(1);
        sm_stream_col_base_s = m1_sm_col_blk_g_i * ((next_cfg_s.pv_m1 == 0) ? 16'd1 : next_cfg_s.pv_m1);
        sm_stream_m1_row_slot_l_s = m1_sm_row_slot_l_i[BUF_ROW_W-1:0];
        sm_stream_m1_ch_blk_g_s   = m1_sm_ch_blk_g_i;
      end
      else if (sm_m2_active && m2_sm_req_valid_i) begin
        sm_stream_start_s         = 1'b1;
        sm_stream_row_base_s      = m2_sm_row_g_i[ROW_W-1:0];
        sm_stream_num_rows_s      = ROW_W'(1);
        sm_stream_col_base_s      = m2_sm_col_g_i[COL_W-1:0];
        sm_stream_m2_cgrp_g_s     = m2_sm_cgrp_g_i;
      end
    end
  end

  assign m1_sm_req_ready_i = sm_m1_active && (
                             (sm_stream_start_s && m1_sm_req_valid_i) ||
                             (sm_exec_active_q && !sm_exec_mode_q &&
                              (m1_sm_row_slot_l_i == sm_exec_m1_row_slot_l_q) &&
                              (m1_sm_row_g_i      == sm_exec_row_g_q)));

  assign m2_sm_req_ready_i = sm_m2_active && (
                             (sm_stream_start_s && !m1_sm_req_valid_i && m2_sm_req_valid_i) ||
                             (sm_exec_active_q && sm_exec_mode_q &&
                              (m2_sm_row_g_i == sm_exec_row_g_q) &&
                              (m2_sm_col_g_i == sm_exec_col_base_g_q)));

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      sm_exec_active_q       <= 1'b0;
      sm_exec_mode_q         <= 1'b0;
      sm_exec_m1_row_slot_l_q<= '0;
      sm_exec_row_g_q        <= '0;
      sm_exec_col_base_g_q   <= '0;
    end
    else begin
      if (!sm_exec_active_q) begin
        if (sm_stream_start_s && m1_sm_req_valid_i && sm_m1_active) begin
          sm_exec_active_q        <= 1'b1;
          sm_exec_mode_q          <= 1'b0;
          sm_exec_m1_row_slot_l_q <= m1_sm_row_slot_l_i;
          sm_exec_row_g_q         <= m1_sm_row_g_i;
          sm_exec_col_base_g_q    <= (m1_sm_col_blk_g_i * ((next_cfg_s.pv_m1 == 0) ? 16'd1 : next_cfg_s.pv_m1));
        end
        else if (sm_stream_start_s && !m1_sm_req_valid_i && m2_sm_req_valid_i && sm_m2_active) begin
          sm_exec_active_q        <= 1'b1;
          sm_exec_mode_q          <= 1'b1;
          sm_exec_m1_row_slot_l_q <= '0;
          sm_exec_row_g_q         <= m2_sm_row_g_i;
          sm_exec_col_base_g_q    <= m2_sm_col_g_i;
        end
      end
      else if (ofm_ifm_stream_done) begin
        sm_exec_active_q <= 1'b0;
      end
    end
  end

  // External visibility mirrors for same-mode requests. These are no longer
  // integration hooks; the control unit consumes the requests internally.
  assign m1_sm_refill_req_valid   = m1_sm_req_valid_i;
  assign m1_sm_refill_row_slot_l  = m1_sm_row_slot_l_i;
  assign m1_sm_refill_row_g       = m1_sm_row_g_i;
  assign m1_sm_refill_col_blk_g   = m1_sm_col_blk_g_i;
  assign m1_sm_refill_ch_blk_g    = m1_sm_ch_blk_g_i;

  assign m2_sm_refill_req_valid   = m2_sm_req_valid_i;
  assign m2_sm_refill_row_g       = m2_sm_row_g_i;
  assign m2_sm_refill_col_g       = m2_sm_col_g_i;
  assign m2_sm_refill_col_l       = m2_sm_col_l_i;
  assign m2_sm_refill_cgrp_g      = m2_sm_cgrp_g_i;

  // --------------------------------------------------------------------------
  // Sub-block instantiation
  // --------------------------------------------------------------------------
  layer_cfg_manager #(
    .CFG_DEPTH(CFG_DEPTH)
  ) u_layer_cfg_manager (
    .clk(clk), .rst_n(rst_n),
    .cfg_wr_en(cfg_wr_en), .cfg_wr_addr(cfg_wr_addr), .cfg_wr_data(cfg_wr_data), .cfg_num_layers(cfg_num_layers),
    .load_first(start_pulse), .advance_layer(advance_layer_s),
    .cur_valid(cur_valid_s), .next_valid(next_valid_s), .cur_layer_idx(cur_layer_idx_s),
    .cur_cfg(cur_cfg_s), .next_cfg(next_cfg_s), .cur_first_layer(cur_first_layer_s), .cur_last_layer(cur_last_layer_s)
  );

  weight_bank_manager u_weight_bank_manager (
    .clk(clk), .rst_n(rst_n),
    .start_first_preload(start_pulse),
    .preload_done(wgt_load_done_s),
    .layer_done(compute_done_s),
    .swap_req(swap_weight_bank_s),
    .bank0_ready(bank0_ready), .bank1_ready(bank1_ready),
    .compute_bank_sel(compute_bank_sel_s),
    .preload_bank_sel(preload_bank_sel_s),
    .compute_bank_ready(compute_bank_ready_s),
    .bank0_release(bank0_release), .bank1_release(bank1_release)
  );

  compute_dispatcher #(
    .PC_MODE2(PC_MODE2)
  ) u_compute_dispatcher (
    .clk(clk), .rst_n(rst_n),
    .cur_cfg(cur_cfg_s),
    .compute_bank_sel(compute_bank_sel_s),
    .compute_bank_ready(compute_bank_ready_s),
    .kick_compute(kick_compute_s),
    .hold_compute(sched_hold_compute_s | local_hold_compute_s),
    .cur_mode(cur_cfg_s.mode == MODE2),
    .m1_done(m1_done), .m1_busy(m1_busy), .m2_done(m2_done), .m2_busy(m2_busy),
    .m1_start(m1_start), .m1_step_en(m1_step_en),
    .m1_k_cur(m1_k_cur), .m1_c_cur(m1_c_cur), .m1_f_cur(m1_f_cur),
    .m1_hout_cur(m1_hout_cur), .m1_wout_cur(m1_wout_cur), .m1_w_cur(m1_w_cur),
    .m1_pv_cur(m1_pv_cur), .m1_pf_cur(m1_pf_cur),
    .m1_weight_bank_sel(m1_weight_bank_sel), .m1_weight_bank_ready(m1_weight_bank_ready),
    .m2_start(m2_start), .m2_step_en(m2_step_en),
    .m2_k_cur(m2_k_cur), .m2_c_cur(m2_c_cur), .m2_f_cur(m2_f_cur),
    .m2_hout_cur(m2_hout_cur), .m2_wout_cur(m2_wout_cur),
    .m2_weight_bank_sel(m2_weight_bank_sel), .m2_weight_bank_ready(m2_weight_bank_ready),
    .compute_done(compute_done_s), .compute_busy(compute_busy_s)
  );

  dma_phase_manager #(
    .PTOTAL(PTOTAL), .PV_MAX(PV_MAX), .PC_MODE2(PC_MODE2),
    .C_MAX(C_MAX), .W_MAX(W_MAX), .H_MAX(H_MAX), .HT(HT),
    .WGT_DEPTH(WGT_DEPTH), .OFM_LINEAR_DEPTH(OFM_LINEAR_DEPTH), .DDR_ADDR_W(DDR_ADDR_W)
  ) u_dma_phase_manager (
    .clk(clk), .rst_n(rst_n),
    .req_ifm_load(kick_ifm_load_s), .req_wgt_load(kick_wgt_preload_s), .req_ofm_store(kick_ofm_store_s),
    .cur_cfg(cur_cfg_s), .preload_bank_sel(preload_bank_sel_s),
    .req_ifm_abs_row_base(ifm_req_abs_row_base_s), .req_ifm_num_rows(ifm_req_num_rows_s),
    .req_ifm_buf_row_base(ifm_req_buf_row_base_s), .req_m2_tile_idx(ifm_req_m2_tile_idx_s),
    .ofm_layer_num_words(ofm_layer_num_words), .ofm_buf_base('0),
    .ofm_layer_write_done(ofm_layer_write_done), .ofm_error(ofm_error),
    .dma_busy(dma_busy), .dma_done_ifm(dma_done_ifm), .dma_done_wgt(dma_done_wgt),
    .dma_done_ofm(dma_done_ofm), .dma_error(dma_error),
    .ifm_cmd_start(ifm_cmd_start), .ifm_cmd_ddr_base(ifm_cmd_ddr_base),
    .ifm_cmd_num_rows(ifm_cmd_num_rows), .ifm_cmd_buf_row_base(ifm_cmd_buf_row_base),
    .wgt_cmd_start(wgt_cmd_start), .wgt_cmd_buf_sel(wgt_cmd_buf_sel),
    .wgt_cmd_ddr_base(wgt_cmd_ddr_base), .wgt_cmd_num_words(wgt_cmd_num_words),
    .ofm_cmd_start(ofm_cmd_start), .ofm_cmd_ddr_base(ofm_cmd_ddr_base),
    .ofm_cmd_num_words(ofm_cmd_num_words), .ofm_cmd_buf_base(ofm_cmd_buf_base),
    .ifm_load_done(ifm_load_done_s), .wgt_load_done(wgt_load_done_s), .ofm_store_done(ofm_store_done_s),
    .phase_error(phase_error_s), .phase_busy(), .dbg_active_phase(), .dbg_pending_mask()
  );

  transition_manager #(
    .PV_MAX(PV_MAX), .PC(PC_MODE2), .F_MAX(F_MAX), .H_MAX(H_MAX), .W_MAX(W_MAX)
  ) u_transition_manager (
    .clk(clk), .rst_n(rst_n),
    .cur_cfg(cur_cfg_s), .next_cfg(next_cfg_s), .next_valid(next_valid_s),
    // Same-mode refill no longer goes through transition_manager.
    .kick_same_mode_stream(1'b0),
    .kick_transition_stream(kick_transition_stream_s),
    .req_row_base(stream_req_row_base_s), .req_num_rows(stream_req_num_rows_s), .req_col_base(stream_req_col_base_s),
    .ofm_layer_write_done(ofm_layer_write_done),
    .ofm_ifm_stream_busy(ofm_ifm_stream_busy), .ofm_ifm_stream_done(ofm_ifm_stream_done), .ofm_error(ofm_error),
    .ofm_ifm_stream_start(trans_ifm_stream_start_s),
    .ofm_ifm_stream_row_base(trans_ifm_stream_row_base_s),
    .ofm_ifm_stream_num_rows(trans_ifm_stream_num_rows_s),
    .ofm_ifm_stream_col_base(trans_ifm_stream_col_base_s),
    .transition_done(transition_done_s), .transition_busy(transition_busy_s), .transition_error(transition_error_s),
    .dbg_req_kind(), .dbg_need_full_store(), .dbg_waiting_for_layer(), .dbg_waiting_for_stream()
  );

  local_dataflow_manager #(
    .DATA_W(DATA_W), .PTOTAL(PTOTAL), .PF_MAX(PF_MAX), .PV_MAX(PV_MAX),
    .PC_MODE2(PC_MODE2), .PF_MODE2(PF_MODE2), .C_MAX(C_MAX), .W_MAX(W_MAX), .H_MAX(H_MAX), .K_MAX(K_MAX)
  ) u_local_dataflow_manager (
    .clk(clk), .rst_n(rst_n),
    .cur_cfg(cur_cfg_s), .cur_mode(cur_cfg_s.mode),
    .ifm_rd_en(ifm_rd_en), .ifm_rd_bank_base(ifm_rd_bank_base), .ifm_rd_row_idx(ifm_rd_row_idx),
    .ifm_rd_col_idx(ifm_rd_col_idx), .ifm_rd_valid(ifm_rd_valid), .ifm_rd_data(ifm_rd_data),
    .m1_pass_start_pulse(m1_pass_start_pulse), .m1_chan_done_pulse(m1_chan_done_pulse), .m1_c_iter(m1_c_iter),
    .m1_dr_write_en(m1_dr_write_en), .m1_dr_write_row_idx(m1_dr_write_row_idx),
    .m1_dr_write_x_base(m1_dr_write_x_base), .m1_dr_write_data(m1_dr_write_data),
    .m2_start(m2_start), .m2_pass_start_pulse(m2_pass_start_pulse), .m2_mac_en(m2_mac_en),
    .m2_ce_out_valid(m2_ce_out_valid), .m2_out_row(m2_out_row_g_s), .m2_out_col(m2_out_col_g_s), .m2_f_group(m2_f_group),
    .m2_dr_write_en(m2_dr_write_en), .m2_dr_write_row_idx(m2_dr_write_row_idx), .m2_dr_write_data(m2_dr_write_data),
    .hold_compute(local_hold_compute_s), .local_busy(local_busy_s), .local_done(local_done_s), .local_error(local_error_s),
    .m1_local_busy(), .m2_local_busy(),
    .m2_free_valid(ldm_m2_free_valid_s),
    .m2_free_row_g(ldm_m2_free_row_g_s),
    .m2_free_col_g(ldm_m2_free_col_g_s),
    .m2_free_col_l(ldm_m2_free_col_l_s),
    .m2_free_cgrp_g(ldm_m2_free_cgrp_g_s)
  );

  // Same-mode refill managers
  same_mode_refill_manager_m1 #(
    .HT(HT), .H_W(16), .COLG_W(16), .CHG_W(16)
  ) u_same_mode_refill_manager_m1 (
    .clk(clk), .rst_n(rst_n_sm_m1),
    .cfg_cur_h_out(cur_m1_ofm_h_for_sm_s), .cfg_cur_w_out(cur_m1_ofm_w_for_sm_s), .cfg_cur_f_out(cur_cfg_s.f_out),
    .cfg_next_h_in(next_cfg_s.h_in), .cfg_next_w_in(next_cfg_s.w_in), .cfg_next_c_in(next_cfg_s.c_in),
    .cfg_next_pv(next_cfg_s.pv_m1), .cfg_next_pf(next_cfg_s.pf_m1),
    .free_valid(m1f_emit_valid_s),
    .free_row_slot_l(m1f_emit_row_slot_l_s),
    .free_row_g(m1f_emit_row_g_s),
    .free_col_blk_g(m1f_emit_col_blk_g_s),
    .free_ch_blk_g(m1f_emit_ch_blk_g_s),
    .ready_valid(m1_ready_tok_valid_s),
    .ready_row_g(m1_ready_tok_row_g_s),
    .ready_col_blk_g(m1_ready_tok_col_blk_g_s),
    .ready_ch_blk_g(m1_ready_tok_ch_blk_g_s),
    .refill_req_valid(m1_sm_req_valid_i),
    .refill_req_ready(m1_sm_req_ready_i),
    .refill_row_slot_l(m1_sm_row_slot_l_i),
    .refill_row_g(m1_sm_row_g_i),
    .refill_col_blk_g(m1_sm_col_blk_g_i),
    .refill_ch_blk_g(m1_sm_ch_blk_g_i),
    .busy(m1_sm_busy_s),
    .error(m1_sm_error_s),
    .free_fifo_full(m1_sm_free_full_s),
    .ready_fifo_full(m1_sm_ready_full_s)
  );

  same_mode_refill_manager_m2 #(
    .H_W(16), .COLG_W(16), .COLL_W(16), .CGRP_W(16)
  ) u_same_mode_refill_manager_m2 (
    .clk(clk), .rst_n(rst_n_sm_m2),
    .cfg_cur_h_out(cur_cfg_s.h_out), .cfg_cur_w_out(cur_cfg_s.w_out), .cfg_cur_f_out(cur_cfg_s.f_out),
    .cfg_next_h_in(next_cfg_s.h_in), .cfg_next_w_in(next_cfg_s.w_in), .cfg_next_c_in(next_cfg_s.c_in),
    .cfg_next_pc(next_cfg_s.pc_m2), .cfg_next_pf(next_cfg_s.pf_m2),
    .free_valid(ldm_m2_free_valid_s && sm_m2_active),
    .free_row_g(ldm_m2_free_row_g_s),
    .free_col_g(ldm_m2_free_col_g_s),
    .free_col_l(ldm_m2_free_col_l_s),
    .free_cgrp_g(ldm_m2_free_cgrp_g_s),
    .ready_valid(m2_ready_tok_valid_s),
    .ready_row_g(m2_ready_tok_row_g_s),
    .ready_col_g(m2_ready_tok_col_g_s),
    .ready_cgrp_g(m2_ready_tok_cgrp_g_s),
    .refill_req_valid(m2_sm_req_valid_i),
    .refill_req_ready(m2_sm_req_ready_i),
    .refill_row_g(m2_sm_row_g_i),
    .refill_col_g(m2_sm_col_g_i),
    .refill_col_l(m2_sm_col_l_i),
    .refill_cgrp_g(m2_sm_cgrp_g_i),
    .busy(m2_sm_busy_s),
    .error(m2_sm_error_s),
    .free_fifo_full(m2_sm_free_full_s),
    .ready_fifo_full(m2_sm_ready_full_s)
  );

  assign same_mode_drain_done_s = (sm_m1_active || sm_m2_active) &&
                                  ofm_layer_write_done &&
                                  !sm_exec_active_q &&
                                  !sm_stream_start_s &&
                                  !ofm_ifm_stream_busy &&
                                  !m1_sm_busy_s && !m2_sm_busy_s &&
                                  (m1q_count_q == '0) &&
                                  (m2q_count_q == '0) &&
                                  (m1f_count_q == '0) &&
                                  !m1f_scan_active_q;

  assign sched_next_path_done_s = ((next_valid_s && (cur_cfg_s.mode == MODE1) && (next_cfg_s.mode == MODE2)) ? transition_done_s :
                                   ((sm_m1_active || sm_m2_active) ? same_mode_drain_done_s : 1'b0));

  global_scheduler_fsm u_global_scheduler_fsm (
    .clk(clk), .rst_n(rst_n),
    .start(start_pulse), .abort(abort),
    .cur_valid(cur_valid_s), .next_valid(next_valid_s),
    .cur_first_layer(cur_first_layer_s), .cur_last_layer(cur_last_layer_s),
    .cur_mode(cur_cfg_s.mode == MODE2), .next_mode(next_cfg_s.mode == MODE2),
    .bank_compute_ready(compute_bank_ready_s),
    .ifm_load_done(ifm_load_done_s), .wgt_load_done(wgt_load_done_s),
    .compute_done(compute_done_s), .compute_busy(compute_busy_s),
    .ofm_layer_write_done(ofm_layer_write_done), .ofm_ifm_stream_done(sched_next_path_done_s), .ofm_store_done(ofm_store_done_s),
    .any_error(any_error_s),
    .kick_ifm_load(kick_ifm_load_s), .kick_wgt_preload(kick_wgt_preload_s),
    .kick_compute(kick_compute_s), .kick_same_mode_stream(kick_same_mode_stream_s),
    .kick_transition_stream(kick_transition_stream_s), .kick_ofm_store(kick_ofm_store_s),
    .hold_compute(sched_hold_compute_s),
    .advance_layer(advance_layer_s), .swap_weight_bank(swap_weight_bank_s),
    .sched_busy(sched_busy_s), .sched_done(sched_done_s), .sched_error(sched_error_s)
  );

  // OFM->IFM stream command mux: transition_manager has priority over
  // internally controlled same-mode refill transactions.
  assign ofm_ifm_stream_start          = trans_ifm_stream_start_s | sm_stream_start_s;
  assign ofm_ifm_stream_row_base       = trans_ifm_stream_start_s ? trans_ifm_stream_row_base_s : sm_stream_row_base_s;
  assign ofm_ifm_stream_num_rows       = trans_ifm_stream_start_s ? trans_ifm_stream_num_rows_s : sm_stream_num_rows_s;
  assign ofm_ifm_stream_col_base       = trans_ifm_stream_start_s ? trans_ifm_stream_col_base_s : sm_stream_col_base_s;
  assign ofm_ifm_stream_m1_row_slot_l  = trans_ifm_stream_start_s ? '0 : sm_stream_m1_row_slot_l_s;
  assign ofm_ifm_stream_m1_ch_blk_g    = trans_ifm_stream_start_s ? '0 : sm_stream_m1_ch_blk_g_s;
  assign ofm_ifm_stream_m2_cgrp_g      = trans_ifm_stream_start_s ? '0 : sm_stream_m2_cgrp_g_s;

  assign control_error_s = m1_sm_error_s | m2_sm_error_s |
                           m1q_overflow_q | m2q_overflow_q | m1f_overflow_q |
                           m1_sm_free_full_s | m1_sm_ready_full_s |
                           m2_sm_free_full_s | m2_sm_ready_full_s;

  assign any_error_s = phase_error_s | local_error_s | transition_error_s | ofm_error | control_error_s;

  status_manager #(
    .LAYER_IDX_W(CFG_AW)
  ) u_status_manager (
    .clk(clk), .rst_n(rst_n),
    .sched_busy(sched_busy_s), .sched_done(sched_done_s), .sched_error(sched_error_s),
    .dma_error(dma_error | phase_error_s), .ofm_error(ofm_error),
    .local_error(local_error_s | control_error_s), .transition_error(transition_error_s),
    .cur_layer_idx(cur_layer_idx_s), .cur_mode(cur_cfg_s.mode == MODE2), .compute_bank_sel(compute_bank_sel_s),
    .busy(busy), .done(done), .error(error),
    .dbg_layer_idx(dbg_layer_idx), .dbg_mode(dbg_mode),
    .dbg_weight_bank(dbg_weight_bank), .dbg_error_vec(dbg_error_vec)
  );

endmodule
