`include "cnn_ddr_defs.svh"

module control_unit_top
  import cnn_layer_desc_pkg::*;
#(
  parameter int PTOTAL           = 256,
  parameter int DATA_W           = 8,
  parameter int PF_MAX           = 16,
  parameter int PV_MAX           = 16,
  parameter int PC_MODE2         = 16,
  parameter int PF_MODE2         = 16,
  parameter int C_MAX            = 512,
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
  output logic [$clog2(W_MAX)-1:0]      ifm_rd_col_g,
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
  output logic [9:0]                    m1_c_cur,
  output logic [9:0]                    m1_f_cur,
  output logic [15:0]                   m1_hout_cur,
  output logic [15:0]                   m1_wout_cur,
  output logic [15:0]                   m1_w_cur,
  output logic [7:0]                    m1_pv_cur,
  output logic [7:0]                    m1_pf_cur,
  output logic                          m1_weight_bank_sel,
  output logic                          m1_weight_bank_ready,
  output logic                          m1_pool_en,

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
  output logic [9:0]                    m2_c_cur,
  output logic [9:0]                    m2_f_cur,
  output logic [15:0]                   m2_hout_cur,
  output logic [15:0]                   m2_wout_cur,
  // Mode-2 tile window ports kept for cnn_top/mode2_compute_top interface compatibility.
  // This control version performs whole-layer Mode-2 compute; a zero count means
  // 'use all remaining columns' in the tile-aware Mode-2 CE/controller.
  output logic [15:0]                   m2_tile_col_base_g,
  output logic [15:0]                   m2_tile_col_count,
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
  output logic                          ofm_cfg_pool_en,
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
  output logic [1:0]                    ofm_ifm_stream_kind,

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

  // OFM->IFM stream command kind. Metadata only; payload/stream priority
  // remains unchanged. This prevents ofm_buffer from inferring stream type
  // from the current layer's next-mode context.
  localparam logic [1:0] OFM_STRM_IDLE      = 2'd0;
  localparam logic [1:0] OFM_STRM_M1_DIRECT = 2'd1;
  localparam logic [1:0] OFM_STRM_M2_DIRECT = 2'd2;
  localparam logic [1:0] OFM_STRM_M1_TO_M2  = 2'd3;

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

  // Use the OFM layer-done rising edge only for the new Mode-2
  // deterministic handoff arming.  ofm_layer_write_done is a level
  // signal and can remain high for the just-finished previous layer
  // after scheduler advance; using the level directly can incorrectly
  // arm the next M2->M2 handoff before the destination layer has
  // produced any OFM.  Keep Mode 1 logic unchanged.
  logic ofm_layer_write_done_q;
  logic ofm_layer_write_done_pulse_s;

  assign ofm_layer_write_done_pulse_s = ofm_layer_write_done && !ofm_layer_write_done_q;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      ofm_layer_write_done_q <= 1'b0;
    end else begin
      ofm_layer_write_done_q <= ofm_layer_write_done;
    end
  end

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
  // K3/P1 Mode1 keeps one previous input row alive.  Suppress the first
  // row-advance/free pulse of each Mode1 layer so output row 1 can still
  // use input row 0 as its ky=0 tap.
  logic m1_pady1_first_row_pending_q;
  logic transition_done_s, transition_busy_s, transition_error_s;
  logic same_mode_drain_done_s, sched_next_path_done_s;
  logic compute_done_s, compute_busy_s;
  // Raw compute_dispatcher status.  Mode1 and Mode2 now use the same
  // layer-level compute scheduler model: one kick_compute_s starts one layer,
  // and dispatch_compute_done_s is the layer done indication.  Mode2 no longer
  // wraps compute in a resident horizontal-tile scheduler.
  logic dispatch_compute_done_s, dispatch_compute_busy_s;
  logic kick_compute_dispatch_s;
  logic dispatch_m2_start_s, dispatch_m2_step_en_s;

  typedef enum logic [2:0] {
    M2TS_IDLE,
    M2TS_START_TILE,
    M2TS_WAIT_TILE,
    M2TS_DDR_REQ,
    M2TS_DDR_WAIT,
    M2TS_OFM_REQ,
    M2TS_OFM_WAIT
  } m2_tile_state_t;

  m2_tile_state_t m2_tile_state_q;
  logic [TILE_W-1:0] m2_tile_idx_q;
  logic [15:0]       m2_tile_base_s;
  logic [15:0]       m2_tile_count_s;
  logic [15:0]       m2_num_tiles_s;
  logic              m2_tile_last_s;
  logic              m2_tile_layer_done_pulse_s;
  logic              m2_ifm_tile_load_req_s;
  logic              m2_runtime_stream_req_s;
  logic              m2_runtime_stream_done_pulse_s;

  logic weight_bank_layer_done_s;
  layer_desc_t wgt_dma_cfg_s;
  logic any_error_s;
  logic control_error_s;

  logic [ROW_W-1:0] ifm_req_abs_row_base_s;
  logic [ROW_W-1:0] ifm_req_num_rows_s;
  logic [BUF_ROW_W-1:0] ifm_req_buf_row_base_s;
  logic [TILE_W-1:0] ifm_req_m2_tile_idx_s;

  // Initial/current-layer mode-1 DDR->IFM tiling refill.
  // This is distinct from same-mode OFM->IFM refill between layers.
  logic             init_ifm_tile_active_q;
  logic             init_ifm_refill_busy_q;
  logic             init_ifm_refill_req_s;
  logic             init_ifm_refill_hold_s;
  logic             init_ifm_refill_claim_s;
  logic [ROW_W-1:0] init_ifm_next_row_q;
  logic [ROW_W-1:0] init_ifm_refill_row_q;

  // Same-mode M1 OFM->IFM refill for the next layer.
  // The initial tile rows are always streamed before layer advance.
  // If next IFM height is greater than HT, remaining rows are streamed after
  // advance as IFM free-row tokens arrive.  If next IFM height is <= HT, this
  // block completes the whole handoff in the initial phase and must not fall
  // back to the legacy free-token manager.
  logic             sm_m1_tiled_active_s;
  logic             sm_m1_mgr_active_s;
  logic             same_mode_legacy_drain_done_s;
  logic             same_mode_initial_tile_ready_s;
  logic             sm_m1_drain_idle_s;
  logic             sm_m2_drain_idle_s;

  logic             ofm2ifm_active_q;
  logic             ofm2ifm_initial_ready_q;
  logic             ofm2ifm_runtime_pending_q;
  logic             ofm2ifm_stream_busy_q;
  logic             ofm2ifm_stream_start_s;
  logic             ofm2ifm_runtime_hold_s;
  logic             ofm2ifm_free_claim_s;

  logic [15:0]      ofm2ifm_h_q;
  logic [15:0]      ofm2ifm_w_q;
  logic [15:0]      ofm2ifm_c_q;
  logic [15:0]      ofm2ifm_pv_q;
  logic [15:0]      ofm2ifm_pf_q;
  logic [15:0]      ofm2ifm_num_col_blks_q;
  logic [15:0]      ofm2ifm_num_ch_blks_q;
  logic [ROW_W-1:0] ofm2ifm_initial_rows_q;
  logic [ROW_W-1:0] ofm2ifm_next_row_q;
  logic [ROW_W-1:0] ofm2ifm_row_q;
  logic [15:0]      ofm2ifm_col_blk_q;
  logic [15:0]      ofm2ifm_ch_blk_q;
  logic [BUF_ROW_W-1:0] ofm2ifm_row_slot_q;
  // Latched destination height for OFM->IFM tiled handoff.  Keep an
  // effective signal so runtime refill after layer advance never depends on
  // next_cfg_s, which is invalid for the last layer.
  logic [ROW_W-1:0] ofm2ifm_dst_h_eff_s;
  logic [7:0]       ofm2ifm_dst_layer_id_q;
  logic             ofm2ifm_in_dst_layer_s;

  // Same-mode M2 OFM->IFM handoff.
  // Follow the Mode-1 lifecycle: stream only the initial IFM window before
  // layer advance; after advance, runtime refill is driven by Mode-2
  // free-entry tokens.  The old tile scheduler is not used as the scheduler
  // for this path.
  logic             sm_m2_mgr_active_s;
  logic             m2_ofm2ifm_active_q;
  logic             m2_ofm2ifm_ready_q;
  logic             m2_ofm2ifm_stream_busy_q;
  logic             m2_ofm2ifm_stream_start_s;
  logic             m2_ofm2ifm_runtime_q;
  logic [ROW_W-1:0] m2_ofm2ifm_row_q;
  logic [15:0]      m2_ofm2ifm_col_blk_q;
  logic [15:0]      m2_ofm2ifm_cgrp_q;
  logic [ROW_W-1:0] m2_ofm2ifm_num_rows_q;
  logic [15:0]      m2_ofm2ifm_num_col_blks_q;
  logic [15:0]      m2_ofm2ifm_num_cgrps_q;
  logic [15:0]      m2_ofm2ifm_w_q;
  logic [15:0]      m2_ofm2ifm_initial_cols_q;
  logic [15:0]      m2_ofm2ifm_col_end_q;
  logic [15:0]      m2_ofm2ifm_runtime_col_base_q;
  logic [7:0]       m2_ofm2ifm_src_layer_id_q;
  logic [7:0]       m2_ofm2ifm_dst_layer_id_q;
  logic             m2_ofm2ifm_seen_dst_q;
  logic             m2_ofm2ifm_in_dst_layer_s;
  logic             m2_ofm2ifm_ctx_cur_to_next_s;
  logic             m2_ofm2ifm_need_arm_s;
  logic             m2_ofm2ifm_arm_s;
  // Mode2 stream lifecycle signals. These mirror the Mode1 pattern:
  // request/start is generated only when the shared OFM->IFM stream path can
  // accept the transaction; busy is set only for an accepted transaction and
  // is cleared only by the real ofm_ifm_stream_done.
  logic             m2_ofm2ifm_init_phase_s;
  logic             m2_ofm2ifm_runtime_phase_s;
  logic             m2_ofm2ifm_stream_req_s;
  logic             m2_ofm2ifm_stream_grant_s;
  logic             m2_ofm2ifm_stream_accepted_s;
  logic             m2_ofm2ifm_tx_runtime_q;
  logic             m2_producer_done_seen_q;
  logic [7:0]       m2_producer_done_layer_id_q;
  logic             m2_producer_done_for_cur_s;
  logic             m2_free_entry_valid_s;
  logic             m2_free_entry_take_s;
  logic             m2_ofm2ifm_runtime_hold_s;
  logic             m2_ofm2ifm_src_mode2_q;
  logic             m2_ofm2ifm_runtime_src_mode2_q;
  logic             m2_ofm2ifm_stream_src_mode2_s;

  localparam int M2FQ_DEPTH = (SM_M2_RDY_Q_DEPTH < 2) ? 2 : SM_M2_RDY_Q_DEPTH;
  localparam int M2FQ_PTR_W = (M2FQ_DEPTH <= 1) ? 1 : $clog2(M2FQ_DEPTH);
  localparam int M2FQ_CNT_W = (M2FQ_DEPTH <= 1) ? 1 : $clog2(M2FQ_DEPTH+1);
  localparam logic [M2FQ_CNT_W-1:0] M2FQ_DEPTH_C = M2FQ_CNT_W'(M2FQ_DEPTH);
  logic [15:0] m2_free_fifo_row_g_q  [0:M2FQ_DEPTH-1];
  logic [15:0] m2_free_fifo_col_g_q  [0:M2FQ_DEPTH-1];
  logic [15:0] m2_free_fifo_col_l_q  [0:M2FQ_DEPTH-1];
  logic [15:0] m2_free_fifo_cgrp_g_q [0:M2FQ_DEPTH-1];
  logic        m2_free_fifo_src_mode2_q [0:M2FQ_DEPTH-1];
  logic [M2FQ_PTR_W-1:0] m2_free_fifo_rptr_q;
  logic [M2FQ_PTR_W-1:0] m2_free_fifo_wptr_q;
  logic [M2FQ_CNT_W-1:0] m2_free_fifo_count_q;
  logic m2_free_fifo_full_s;
  logic m2_free_fifo_empty_s;
  logic m2_free_fifo_push_s;
  logic m2_free_fifo_overflow_q;
  logic [15:0] m2_free_head_row_g_s;
  logic [15:0] m2_free_head_col_g_s;
  logic [15:0] m2_free_head_col_l_s;
  logic [15:0] m2_free_head_cgrp_g_s;
  logic        m2_free_head_src_mode2_s;
  logic m1_to_m2_transition_ready_s;

  logic [ROW_W-1:0] stream_req_row_base_s;
  logic [ROW_W-1:0] stream_req_num_rows_s;
  logic [COL_W-1:0] stream_req_col_base_s;

  logic use_next_ifm_cfg;

  // Mode-2 global-coordinate aliases
  logic [15:0] m2_out_row_g_s, m2_out_col_g_s;
  assign m2_out_row_g_s = m2_out_row;
  assign m2_out_col_g_s = m2_out_col;

  // Mode-2 tile-window outputs.
  // Mode1 has no control-level compute tile scheduler.  To make Mode2 follow
  // the same scheduler model, control_unit_top no longer slices Mode2 compute
  // into resident horizontal tiles.  The interface ports are kept for
  // compatibility; a zero count asks the tile-aware Mode2 datapath to use the
  // full/remaining output width.
  always_comb begin
    if (PC_MODE2 != 0)
      m2_num_tiles_s = (cur_cfg_s.w_out + 16'(PC_MODE2) - 16'd1) / 16'(PC_MODE2);
    else
      m2_num_tiles_s = 16'd0;

    m2_tile_base_s = 16'(m2_tile_idx_q) * 16'(PC_MODE2);

    if ((cur_cfg_s.w_out == 0) || (m2_tile_base_s >= cur_cfg_s.w_out)) begin
      m2_tile_count_s = 16'd0;
    end
    else if ((m2_tile_base_s + 16'(PC_MODE2)) <= cur_cfg_s.w_out) begin
      m2_tile_count_s = 16'(PC_MODE2);
    end
    else begin
      m2_tile_count_s = cur_cfg_s.w_out - m2_tile_base_s;
    end

    m2_tile_last_s = (m2_num_tiles_s == 16'd0) ? 1'b1 :
                     ((16'(m2_tile_idx_q) + 16'd1) >= m2_num_tiles_s);
  end

  assign m2_tile_col_base_g = 16'd0;
  assign m2_tile_col_count  = 16'd0;

  // Internal mode-2 free-token source from local_dataflow_manager.
  // Keep top-level m2_free_* ports unchanged for compatibility, but the
  // same-mode M2 refill path in this integrated control unit must consume the
  // local-dataflow token stream generated when data_register_mode2 captures a
  // tuple successfully.
  logic        ldm_m2_free_valid_s;
  logic        ldm_m2_free_ready_s;
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
  // same-mode M1 refill manager does not wait for the wrong dimensions.
  // pool_en=1 keeps the legacy pooled geometry; pool_en=0 uses raw conv H/W.
  logic                  cur_m1_pool_active_s;
  logic [15:0]           cur_m1_final_h_out_s;
  logic [15:0]           cur_m1_final_w_out_s;
  logic [15:0]           cur_m1_ofm_h_for_sm_s;
  logic [15:0]           cur_m1_ofm_w_for_sm_s;

  // True current-layer OFM size as stored in ofm_buffer for mode 2.
  // Mode 2 compute emits convolution H/W, then pooling_mode2 may halve them
  // before writing OFM, so ofm_buffer/refill must see the final pooled size.
  logic                  cur_m2_pool_active_s;
  logic [15:0]           cur_m2_final_h_out_s;
  logic [15:0]           cur_m2_final_w_out_s;

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
  logic [1:0]            sm_stream_kind_s;
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
  assign sm_m1_active       = next_valid_s && (cur_cfg_s.mode == MODE1) && (next_cfg_s.mode == MODE1);
  assign sm_m2_active       = next_valid_s && (cur_cfg_s.mode == MODE2) && (next_cfg_s.mode == MODE2);
  // All M1->M1 handoffs use the internal OFM->IFM initial-tile manager.
  // The legacy M1 refill manager is left reset so it cannot consume stale/free
  // tokens for small next layers (next_h_in <= HT), which previously caused
  // row_slot X and incomplete handoff.
  assign sm_m1_tiled_active_s = sm_m1_active;
  assign sm_m1_mgr_active_s   = 1'b0;
  // Disable the legacy M2 free/ready-token manager on the main path.  Mode2
  // uses the custom Mode1-style path below: initial preload before layer
  // advance, then runtime free-token capture -> pending -> stream.
  assign sm_m2_mgr_active_s   = 1'b0;
  assign rst_n_sm_m1        = rst_n && sm_m1_mgr_active_s;
  assign rst_n_sm_m2        = rst_n && sm_m2_mgr_active_s;

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

    // Mode2 no longer uses a control-level resident tile scheduler.  Keep the
    // DDR request tile index at zero so the normal initial IFM load path remains
    // interface-compatible; runtime rolling refill is handled by OFM->IFM
    // streams, not by m2_tile_state_q.
    ifm_req_m2_tile_idx_s = '0;

    // Initial preload command loads rows [0 .. min(H,HT)-1].
    // Later initial-IFM tiling refill commands load one absolute DDR row at a
    // time into the logical tail row of the active IFM window.
    if (init_ifm_refill_req_s || init_ifm_refill_busy_q) begin
      ifm_req_abs_row_base_s = init_ifm_refill_row_q;
      ifm_req_num_rows_s     = ROW_W'(1);
      ifm_req_buf_row_base_s = BUF_ROW_W'(HT-1);
    end
    else begin
      if (cur_cfg_s.mode == MODE1) begin
        if (cur_cfg_s.h_in < ROW_W'(HT))
          ifm_req_num_rows_s = cur_cfg_s.h_in[ROW_W-1:0];
        else
          ifm_req_num_rows_s = ROW_W'(HT);
      end
      else begin
        ifm_req_num_rows_s = cur_cfg_s.h_in[ROW_W-1:0];
      end
    end

    stream_req_row_base_s = '0;
    stream_req_num_rows_s = (cur_cfg_s.mode == MODE1)
                            ? cur_m1_final_h_out_s[ROW_W-1:0]
                            : cur_m2_final_h_out_s[ROW_W-1:0];
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
      if ((sm_m1_active || sm_m2_active) && (next_cfg_s.h_in[$clog2(H_MAX+1)-1:0] > ifm_cur_h_in_v))
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

  always_ff @(posedge clk or negedge rst_n) begin : PROC_M1_PADY1_ROW_ADVANCE_GUARD
    if (!rst_n) begin
      m1_pady1_first_row_pending_q <= 1'b1;
    end
    else if (abort || start_pulse || kick_compute_s || advance_layer_s) begin
      m1_pady1_first_row_pending_q <= 1'b1;
    end
    else if ((cur_cfg_s.mode == MODE1) && m1_out_row_done_pulse && (cur_cfg_s.k == 4'd3)) begin
      m1_pady1_first_row_pending_q <= 1'b0;
    end
  end

  // For full K3/P1 Mode1, output row r still needs input row r-1.
  // Therefore the first completed output row must not free/advance the IFM
  // sliding window.  For non-K3 layers, preserve the original behavior.
  assign ifm_m1_advance_row = (cur_cfg_s.mode == MODE1) &&
                              m1_out_row_done_pulse &&
                              ((cur_cfg_s.k != 4'd3) || !m1_pady1_first_row_pending_q);

  // DDR->IFM refill for a first/current mode-1 layer whose input height is
  // larger than the HT rows resident in ifm_buffer.  Initial preload fills
  // rows 0..HT-1.  Each IFM free-row event then claims one new absolute DDR
  // row and refills it into the logical tail row (HT-1) of the sliding window.
  assign init_ifm_refill_hold_s = init_ifm_refill_req_s | init_ifm_refill_busy_q;
  assign init_ifm_refill_claim_s = init_ifm_tile_active_q &&
                                   (init_ifm_next_row_q < cur_cfg_s.h_in[ROW_W-1:0]);

  always_ff @(posedge clk or negedge rst_n) begin : PROC_INIT_IFM_M1_DDR_REFILL
    if (!rst_n) begin
      init_ifm_tile_active_q <= 1'b0;
      init_ifm_refill_busy_q <= 1'b0;
      init_ifm_refill_req_s  <= 1'b0;
      init_ifm_next_row_q    <= '0;
      init_ifm_refill_row_q  <= '0;
    end
    else begin
      init_ifm_refill_req_s <= 1'b0;

      if (abort || advance_layer_s) begin
        init_ifm_tile_active_q <= 1'b0;
        init_ifm_refill_busy_q <= 1'b0;
        init_ifm_next_row_q    <= '0;
        init_ifm_refill_row_q  <= '0;
      end
      else begin
        // Scheduler initial IFM preload for the first layer.
        if (kick_ifm_load_s && cur_first_layer_s && (cur_cfg_s.mode == MODE1)) begin
          init_ifm_tile_active_q <= (cur_cfg_s.h_in > ROW_W'(HT));
          init_ifm_next_row_q    <= (cur_cfg_s.h_in < ROW_W'(HT)) ?
                                    cur_cfg_s.h_in[ROW_W-1:0] : ROW_W'(HT);
          init_ifm_refill_busy_q <= 1'b0;
          init_ifm_refill_row_q  <= '0;
        end

        // A freed row slot in the current IFM window can be reused for the
        // next absolute input row still missing from DDR.  Hold compute until
        // the DMA command completes so CE never reads an unfilled tail row.
        if (init_ifm_tile_active_q &&
            m1_free_valid &&
            !init_ifm_refill_busy_q &&
            (init_ifm_next_row_q < cur_cfg_s.h_in[ROW_W-1:0])) begin
          init_ifm_refill_row_q  <= init_ifm_next_row_q;
          init_ifm_refill_busy_q <= 1'b1;
          init_ifm_refill_req_s  <= 1'b1;
        end

        if (init_ifm_refill_busy_q && ifm_load_done_s) begin
          init_ifm_refill_busy_q <= 1'b0;
          init_ifm_next_row_q    <= init_ifm_next_row_q + ROW_W'(1);
          if ((init_ifm_next_row_q + ROW_W'(1)) >= cur_cfg_s.h_in[ROW_W-1:0]) begin
            init_ifm_tile_active_q <= 1'b0;
          end
        end
      end
    end
  end

  // --------------------------------------------------------------------------
  // Same-mode M1 OFM->IFM refill
  // --------------------------------------------------------------------------
  // For any M1->M1 handoff, stream the initial IFM tile before layer advance.
  // This covers both next_h_in > HT and next_h_in <= HT.  When next_h_in > HT,
  // remaining rows are streamed after the next layer starts using IFM free-row
  // tokens as row-slot availability.  When next_h_in <= HT, the initial phase
  // streams the whole next IFM and no runtime refill is needed.
  assign ofm2ifm_dst_h_eff_s = (|ofm2ifm_h_q[ROW_W-1:0]) ?
                                  ofm2ifm_h_q[ROW_W-1:0] :
                                  cur_cfg_s.h_in[ROW_W-1:0];

  // Runtime OFM->IFM tiled refill must consume free-row tokens only from the
  // destination layer. During a multi-layer chain, source-layer free tokens can
  // still appear after the initial tile for the next layer is ready but before
  // scheduler advances. Claiming those stale tokens launches row refill too
  // early and can hold the destination compute forever.
  assign ofm2ifm_in_dst_layer_s = ofm2ifm_active_q &&
                                  ofm2ifm_initial_ready_q &&
                                  (cur_cfg_s.layer_id[7:0] == ofm2ifm_dst_layer_id_q);

  assign same_mode_initial_tile_ready_s = ofm2ifm_active_q && ofm2ifm_initial_ready_q;
  assign ofm2ifm_free_claim_s = ofm2ifm_in_dst_layer_s &&
                                (ofm2ifm_next_row_q < ofm2ifm_dst_h_eff_s);
  assign ofm2ifm_runtime_hold_s = ofm2ifm_in_dst_layer_s &&
                                  ((ofm2ifm_runtime_pending_q || ofm2ifm_stream_busy_q) ||
                                   ((ofm2ifm_next_row_q < ofm2ifm_dst_h_eff_s) && m1_free_valid));

  always_comb begin
    ofm2ifm_stream_start_s = 1'b0;

    if (ofm2ifm_active_q && !ofm2ifm_stream_busy_q &&
        !transition_busy_s && !trans_ifm_stream_start_s && !ofm_ifm_stream_busy) begin
      if (!ofm2ifm_initial_ready_q && (ofm2ifm_row_q < ofm2ifm_initial_rows_q)) begin
        ofm2ifm_stream_start_s = 1'b1;
      end
      else if (ofm2ifm_initial_ready_q && ofm2ifm_runtime_pending_q) begin
        ofm2ifm_stream_start_s = 1'b1;
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin : PROC_OFM2IFM_M1_TILED_REFILL
    logic [15:0] next_col_blks_v;
    logic [15:0] next_ch_blks_v;
    logic [ROW_W-1:0] initial_rows_v;

    if (!rst_n) begin
      ofm2ifm_active_q          <= 1'b0;
      ofm2ifm_initial_ready_q   <= 1'b0;
      ofm2ifm_runtime_pending_q <= 1'b0;
      ofm2ifm_stream_busy_q     <= 1'b0;
      ofm2ifm_h_q               <= '0;
      ofm2ifm_w_q               <= '0;
      ofm2ifm_c_q               <= '0;
      ofm2ifm_pv_q              <= '0;
      ofm2ifm_pf_q              <= '0;
      ofm2ifm_num_col_blks_q    <= '0;
      ofm2ifm_num_ch_blks_q     <= '0;
      ofm2ifm_initial_rows_q    <= '0;
      ofm2ifm_next_row_q        <= '0;
      ofm2ifm_row_q             <= '0;
      ofm2ifm_col_blk_q         <= '0;
      ofm2ifm_ch_blk_q          <= '0;
      ofm2ifm_row_slot_q        <= '0;
      ofm2ifm_dst_layer_id_q    <= '0;
    end
    else begin
      if (abort) begin
        ofm2ifm_active_q          <= 1'b0;
        ofm2ifm_initial_ready_q   <= 1'b0;
        ofm2ifm_runtime_pending_q <= 1'b0;
        ofm2ifm_stream_busy_q     <= 1'b0;
        ofm2ifm_dst_layer_id_q    <= '0;
      end
      else begin
        next_col_blks_v = (next_cfg_s.pv_m1 == 0) ? 16'd1 :
                          ((next_cfg_s.w_in + next_cfg_s.pv_m1 - 1) / next_cfg_s.pv_m1);
        next_ch_blks_v  = (next_cfg_s.pf_m1 == 0) ? 16'd1 :
                          ((next_cfg_s.c_in + next_cfg_s.pf_m1 - 1) / next_cfg_s.pf_m1);
        initial_rows_v  = (next_cfg_s.h_in < 16'(HT)) ? next_cfg_s.h_in[ROW_W-1:0] : ROW_W'(HT);

        // Start an M1->M1 handoff after the current OFM is fully written.
        // Save next-layer dimensions because cur/next_cfg will change on
        // advance.  Always restart initial row/column/channel counters at 0;
        // the initial phase maps row r to physical row slot r.
        if (!ofm2ifm_active_q && sm_m1_tiled_active_s && ofm_layer_write_done) begin
          ofm2ifm_active_q          <= 1'b1;
          ofm2ifm_initial_ready_q   <= 1'b0;
          ofm2ifm_runtime_pending_q <= 1'b0;
          ofm2ifm_stream_busy_q     <= 1'b0;
          ofm2ifm_h_q               <= next_cfg_s.h_in;
          ofm2ifm_w_q               <= next_cfg_s.w_in;
          ofm2ifm_c_q               <= next_cfg_s.c_in;
          ofm2ifm_pv_q              <= (next_cfg_s.pv_m1 == 0) ? 16'd1 : next_cfg_s.pv_m1;
          ofm2ifm_pf_q              <= (next_cfg_s.pf_m1 == 0) ? 16'd1 : next_cfg_s.pf_m1;
          ofm2ifm_num_col_blks_q    <= (next_col_blks_v == 0) ? 16'd1 : next_col_blks_v;
          ofm2ifm_num_ch_blks_q     <= (next_ch_blks_v  == 0) ? 16'd1 : next_ch_blks_v;
          ofm2ifm_initial_rows_q    <= initial_rows_v;
          ofm2ifm_next_row_q        <= initial_rows_v;
          ofm2ifm_row_q             <= '0;
          ofm2ifm_col_blk_q         <= '0;
          ofm2ifm_ch_blk_q          <= '0;
          ofm2ifm_row_slot_q        <= '0;
          ofm2ifm_dst_layer_id_q    <= next_cfg_s.layer_id[7:0];
        end

        // If the whole next IFM fit into the initial tile, retire the handoff
        // once the scheduler has consumed same_mode_initial_tile_ready_s and
        // advanced to the destination layer.  For taller IFMs, keep the handoff
        // active so runtime free-row tokens can pull the remaining rows.
        if (advance_layer_s && ofm2ifm_active_q && ofm2ifm_initial_ready_q &&
            (ofm2ifm_next_row_q >= ofm2ifm_dst_h_eff_s)) begin
          ofm2ifm_active_q          <= 1'b0;
          ofm2ifm_initial_ready_q   <= 1'b0;
          ofm2ifm_runtime_pending_q <= 1'b0;
          ofm2ifm_stream_busy_q     <= 1'b0;
          ofm2ifm_row_q             <= '0;
          ofm2ifm_col_blk_q         <= '0;
          ofm2ifm_ch_blk_q          <= '0;
          ofm2ifm_row_slot_q        <= '0;
          ofm2ifm_dst_layer_id_q    <= '0;
        end

        // After the initial tile is ready and the next layer is computing, use
        // free row slots to pull the remaining OFM rows into IFM.
        // Capture the IFM free-row token only after scheduler has advanced to the destination layer. Do not
        // require stream_busy=0 here; the pending request can wait until the
        // stream datapath is idle, while the one-cycle free token would be lost.
        if (ofm2ifm_in_dst_layer_s && m1_busy &&
            !ofm2ifm_runtime_pending_q &&
            (ofm2ifm_next_row_q < ofm2ifm_dst_h_eff_s) && m1_free_valid) begin
          ofm2ifm_runtime_pending_q <= 1'b1;
          ofm2ifm_row_q             <= ofm2ifm_next_row_q;
          ofm2ifm_col_blk_q         <= '0;
          ofm2ifm_ch_blk_q          <= '0;
          ofm2ifm_row_slot_q        <= m1_free_row_slot_l;
        end

        if (ofm2ifm_stream_start_s) begin
          ofm2ifm_stream_busy_q <= 1'b1;
        end

        if (ofm2ifm_stream_busy_q && ofm_ifm_stream_done) begin
          ofm2ifm_stream_busy_q <= 1'b0;

          if ((ofm2ifm_ch_blk_q + 16'd1) < ofm2ifm_num_ch_blks_q) begin
            ofm2ifm_ch_blk_q <= ofm2ifm_ch_blk_q + 16'd1;
          end
          else begin
            ofm2ifm_ch_blk_q <= '0;
            if ((ofm2ifm_col_blk_q + 16'd1) < ofm2ifm_num_col_blks_q) begin
              ofm2ifm_col_blk_q <= ofm2ifm_col_blk_q + 16'd1;
            end
            else begin
              ofm2ifm_col_blk_q <= '0;

              if (!ofm2ifm_initial_ready_q) begin
                if ((ofm2ifm_row_q + ROW_W'(1)) < ofm2ifm_initial_rows_q) begin
                  ofm2ifm_row_q      <= ofm2ifm_row_q + ROW_W'(1);
                  ofm2ifm_row_slot_q <= ofm2ifm_row_slot_q + 1'b1;
                end
                else begin
                  ofm2ifm_initial_ready_q <= 1'b1;
                  ofm2ifm_row_q           <= ofm2ifm_next_row_q;
                  ofm2ifm_row_slot_q      <= '0;
                end
              end
              else begin
                ofm2ifm_runtime_pending_q <= 1'b0;
                ofm2ifm_next_row_q        <= ofm2ifm_next_row_q + ROW_W'(1);
                if ((ofm2ifm_next_row_q + ROW_W'(1)) >= ofm2ifm_dst_h_eff_s) begin
                  ofm2ifm_active_q        <= 1'b0;
                  ofm2ifm_initial_ready_q <= 1'b0;
                end
              end
            end
          end
        end
      end
    end
  end


  // --------------------------------------------------------------------------
  // Mode-2 compute tile scheduler compatibility
  // --------------------------------------------------------------------------
  // Mode2 is intentionally aligned with Mode1's scheduler model in this
  // control unit: kick_compute_s starts one whole layer through
  // compute_dispatcher, and dispatch_compute_done_s is the layer-done pulse.
  // The old resident horizontal-tile FSM is kept reset/inert only to preserve
  // existing signal names used elsewhere in this file.
  assign m2_ifm_tile_load_req_s = 1'b0;
  assign m2_runtime_stream_req_s = 1'b0;

  always_ff @(posedge clk or negedge rst_n) begin : PROC_M2_COMPUTE_TILE_SCHED
    if (!rst_n) begin
      m2_tile_state_q <= M2TS_IDLE;
      m2_tile_idx_q   <= '0;
      m2_tile_layer_done_pulse_s <= 1'b0;
    end
    else begin
      m2_tile_state_q <= M2TS_IDLE;
      m2_tile_idx_q   <= '0;
      m2_tile_layer_done_pulse_s <= 1'b0;
    end
  end


  // --------------------------------------------------------------------------
  // Same-mode M2 OFM->IFM refill (Mode2-only, Mode1-style lifecycle)
  // --------------------------------------------------------------------------
  // Mode1 reference behavior:
  //   initial preload before layer advance;
  //   after the destination layer starts, capture a one-cycle free token into
  //   a pending request, stream that refill unit, then clear pending on done.
  // Mode2 follows the same lifecycle.  Its refill unit is not a row slot; it is
  // one exact rolling-buffer entry {row_g, global_col_g, cgrp_g}.  The physical
  // IFM slot is derived downstream as global_col_g % PC_MODE2.
  assign m2_ofm2ifm_in_dst_layer_s = m2_ofm2ifm_active_q &&
                                      m2_ofm2ifm_ready_q &&
                                      (cur_cfg_s.mode == MODE2) &&
                                      (cur_cfg_s.layer_id[7:0] == m2_ofm2ifm_dst_layer_id_q);

  // Mode2 next-path must follow the same level-based lifecycle as Mode1:
  // producer OFM complete -> arm current->next context -> stream the initial
  // IFM window -> raise ready so the global scheduler may advance.  Do not
  // depend on dispatch_compute_done_s being high in the same cycle as
  // ofm_layer_write_done; latch the producer-done event by layer id so a pulse
  // cannot be missed while an old destination context is being retired.
  assign m2_ofm2ifm_ctx_cur_to_next_s =
      m2_ofm2ifm_active_q &&
      (m2_ofm2ifm_src_layer_id_q == cur_cfg_s.layer_id[7:0]) &&
      (m2_ofm2ifm_dst_layer_id_q == next_cfg_s.layer_id[7:0]);

  // Same as Mode1's producer-done event, but protected from stale
  // layer_write_done levels after layer advance: only a fresh write-done edge
  // for the current producer layer, or a latched copy of that edge, may arm
  // the current->next Mode2 handoff.
  assign m2_producer_done_for_cur_s =
      ofm_layer_write_done_pulse_s ||
      (m2_producer_done_seen_q &&
       (m2_producer_done_layer_id_q == cur_cfg_s.layer_id[7:0]));

  assign m2_ofm2ifm_need_arm_s =
      sm_m2_active &&
      m2_producer_done_for_cur_s &&
      !m2_ofm2ifm_ctx_cur_to_next_s &&
      !m2_ofm2ifm_runtime_q &&
      !m2_ofm2ifm_stream_busy_q;

  assign m2_ofm2ifm_arm_s = m2_ofm2ifm_need_arm_s;

  // Capture the one-cycle Mode2 free token exactly like Mode1 captures
  // m1_free_valid.  The token already names the next global IFM column that
  // must be refilled into the freed rolling slot, so control only latches the
  // entry and issues one OFM->IFM stream command for it.
  assign ldm_m2_free_ready_s = m2_ofm2ifm_in_dst_layer_s &&
                                 m2_busy &&
                                 (ldm_m2_free_row_g_s < 16'(m2_ofm2ifm_num_rows_q)) &&
                                 (ldm_m2_free_col_g_s < m2_ofm2ifm_w_q) &&
                                 (ldm_m2_free_cgrp_g_s < m2_ofm2ifm_num_cgrps_q) &&
                                 (!m2_free_fifo_full_s || m2_free_entry_take_s);

  assign m2_free_entry_valid_s = ldm_m2_free_valid_s && ldm_m2_free_ready_s;

  assign m2_free_fifo_full_s  = (m2_free_fifo_count_q == M2FQ_DEPTH_C);
  assign m2_free_fifo_empty_s = (m2_free_fifo_count_q == '0);
  assign m2_free_fifo_push_s  = m2_free_entry_valid_s && (!m2_free_fifo_full_s || m2_free_entry_take_s);

  assign m2_free_head_row_g_s       = m2_free_fifo_row_g_q[m2_free_fifo_rptr_q];
  assign m2_free_head_col_g_s       = m2_free_fifo_col_g_q[m2_free_fifo_rptr_q];
  assign m2_free_head_col_l_s       = m2_free_fifo_col_l_q[m2_free_fifo_rptr_q];
  assign m2_free_head_cgrp_g_s      = m2_free_fifo_cgrp_g_q[m2_free_fifo_rptr_q];
  assign m2_free_head_src_mode2_s   = m2_free_fifo_src_mode2_q[m2_free_fifo_rptr_q];

  assign m2_free_entry_take_s = !m2_free_fifo_empty_s &&
                                !m2_ofm2ifm_runtime_q &&
                                !m2_ofm2ifm_stream_busy_q;

  assign m2_ofm2ifm_runtime_hold_s = m2_ofm2ifm_in_dst_layer_s &&
                                     (!m2_free_fifo_empty_s ||
                                      m2_free_entry_valid_s ||
                                      m2_ofm2ifm_runtime_q ||
                                      m2_ofm2ifm_stream_busy_q);

  assign m1_to_m2_transition_ready_s = next_valid_s &&
                                       (cur_cfg_s.mode == MODE1) &&
                                       (next_cfg_s.mode == MODE2) &&
                                       transition_done_s;

  assign m2_ofm2ifm_init_phase_s =
      m2_ofm2ifm_active_q &&
      !m2_ofm2ifm_ready_q &&
      (m2_ofm2ifm_row_q < m2_ofm2ifm_num_rows_q) &&
      (m2_ofm2ifm_col_blk_q < m2_ofm2ifm_col_end_q) &&
      (m2_ofm2ifm_cgrp_q < m2_ofm2ifm_num_cgrps_q);

  assign m2_ofm2ifm_runtime_phase_s =
      m2_ofm2ifm_active_q &&
      m2_ofm2ifm_ready_q &&
      m2_ofm2ifm_runtime_q;

  // Mode2 equivalent of the Mode1 start condition: generate a start only when
  // the shared OFM->IFM datapath can actually accept the command.  Do not start
  // in the same cycle as a new context is armed because the cursor/geometry is
  // being reinitialized in that cycle.
  assign m2_ofm2ifm_stream_req_s =
      (cur_cfg_s.mode == MODE2) &&
      !m2_ofm2ifm_arm_s &&
      !m2_ofm2ifm_stream_busy_q &&
      (m2_ofm2ifm_init_phase_s || m2_ofm2ifm_runtime_phase_s);

  assign m2_ofm2ifm_stream_grant_s =
      !transition_busy_s &&
      !trans_ifm_stream_start_s &&
      !ofm2ifm_stream_start_s &&
      !ofm_ifm_stream_busy;

  assign m2_ofm2ifm_stream_start_s =
      m2_ofm2ifm_stream_req_s &&
      m2_ofm2ifm_stream_grant_s;

  assign m2_ofm2ifm_stream_accepted_s = m2_ofm2ifm_stream_start_s;

  // Source layout for the command currently being issued.
  // Initial/context streams use the context source kind; runtime exact-entry
  // refills use the source kind latched with the free/demand token.
  assign m2_ofm2ifm_stream_src_mode2_s = m2_ofm2ifm_runtime_q
                                       ? m2_ofm2ifm_runtime_src_mode2_q
                                       : m2_ofm2ifm_src_mode2_q;

  always_ff @(posedge clk or negedge rst_n) begin : PROC_OFM2IFM_M2_TILED_REFILL
    logic [15:0] init_cols_v;
    logic [15:0] init_cgrps_v;

    if (!rst_n) begin
      m2_ofm2ifm_active_q          <= 1'b0;
      m2_ofm2ifm_ready_q           <= 1'b0;
      m2_ofm2ifm_stream_busy_q     <= 1'b0;
      m2_ofm2ifm_tx_runtime_q      <= 1'b0;
      m2_ofm2ifm_row_q             <= '0;
      m2_ofm2ifm_col_blk_q         <= '0;
      m2_ofm2ifm_cgrp_q            <= '0;
      m2_ofm2ifm_num_rows_q        <= '0;
      m2_ofm2ifm_num_col_blks_q    <= '0;
      m2_ofm2ifm_num_cgrps_q       <= '0;
      m2_ofm2ifm_w_q               <= '0;
      m2_ofm2ifm_initial_cols_q    <= '0;
      m2_ofm2ifm_col_end_q         <= '0;
      m2_ofm2ifm_runtime_col_base_q<= '0;
      m2_ofm2ifm_runtime_q         <= 1'b0;
      m2_runtime_stream_done_pulse_s <= 1'b0;
      m2_ofm2ifm_src_layer_id_q    <= '0;
      m2_ofm2ifm_dst_layer_id_q    <= '0;
      m2_ofm2ifm_seen_dst_q        <= 1'b0;
      m2_ofm2ifm_src_mode2_q       <= 1'b1;
      m2_ofm2ifm_runtime_src_mode2_q <= 1'b1;
      m2_free_fifo_rptr_q          <= '0;
      m2_free_fifo_wptr_q          <= '0;
      m2_free_fifo_count_q         <= '0;
      m2_free_fifo_overflow_q      <= 1'b0;
      m2_producer_done_seen_q      <= 1'b0;
      m2_producer_done_layer_id_q  <= '0;
    end
    else begin
      m2_runtime_stream_done_pulse_s <= 1'b0;

      if (abort) begin
        m2_ofm2ifm_active_q          <= 1'b0;
        m2_ofm2ifm_ready_q           <= 1'b0;
        m2_ofm2ifm_stream_busy_q     <= 1'b0;
        m2_ofm2ifm_tx_runtime_q      <= 1'b0;
        m2_ofm2ifm_row_q             <= '0;
        m2_ofm2ifm_col_blk_q         <= '0;
        m2_ofm2ifm_cgrp_q            <= '0;
        m2_ofm2ifm_num_rows_q        <= '0;
        m2_ofm2ifm_num_col_blks_q    <= '0;
        m2_ofm2ifm_num_cgrps_q       <= '0;
        m2_ofm2ifm_w_q               <= '0;
        m2_ofm2ifm_initial_cols_q    <= '0;
        m2_ofm2ifm_col_end_q         <= '0;
        m2_ofm2ifm_runtime_col_base_q<= '0;
        m2_ofm2ifm_runtime_q         <= 1'b0;
        m2_ofm2ifm_src_layer_id_q    <= '0;
        m2_ofm2ifm_dst_layer_id_q    <= '0;
        m2_ofm2ifm_seen_dst_q        <= 1'b0;
        m2_ofm2ifm_src_mode2_q       <= 1'b1;
        m2_ofm2ifm_runtime_src_mode2_q <= 1'b1;
        m2_free_fifo_rptr_q          <= '0;
        m2_free_fifo_wptr_q          <= '0;
        m2_free_fifo_count_q         <= '0;
        m2_free_fifo_overflow_q      <= 1'b0;
        m2_producer_done_seen_q      <= 1'b0;
        m2_producer_done_layer_id_q  <= '0;
      end
      else begin
        // Keep a fresh producer OFM-done edge as a level by layer id until the
        // matching current->next context is armed.  This is the Mode2 equivalent
        // of Mode1's pending/initial-ready lifecycle and prevents a true done
        // pulse from being lost while also rejecting stale done levels from the
        // previous layer after advance.
        if (!sm_m2_active) begin
          m2_producer_done_seen_q     <= 1'b0;
          m2_producer_done_layer_id_q <= '0;
        end
        else if (m2_ofm2ifm_arm_s) begin
          m2_producer_done_seen_q     <= 1'b0;
          m2_producer_done_layer_id_q <= '0;
        end
        else if (ofm_layer_write_done_pulse_s) begin
          m2_producer_done_seen_q     <= 1'b1;
          m2_producer_done_layer_id_q <= cur_cfg_s.layer_id[7:0];
        end
        else if (m2_producer_done_seen_q &&
                 (m2_producer_done_layer_id_q != cur_cfg_s.layer_id[7:0])) begin
          m2_producer_done_seen_q     <= 1'b0;
          m2_producer_done_layer_id_q <= '0;
        end

        // Arm a new same-mode M2 handoff after the producer layer's final OFM
        // is complete, mirroring Mode1.  Geometry is the consumer IFM geometry.
        // This is based on producer OFM done + current->next context mismatch,
        // not on the dispatch_compute_done_s pulse.
        if (m2_ofm2ifm_arm_s) begin
          init_cols_v  = (next_cfg_s.w_in == 0) ? 16'd1 :
                         ((next_cfg_s.w_in < 16'(PC_MODE2)) ? next_cfg_s.w_in : 16'(PC_MODE2));
          init_cgrps_v = (next_cfg_s.c_in == 0) ? 16'd1 :
                         ((next_cfg_s.c_in + 16'(PC_MODE2) - 16'd1) / 16'(PC_MODE2));

          m2_ofm2ifm_active_q       <= 1'b1;
          m2_ofm2ifm_ready_q        <= 1'b0;
          m2_ofm2ifm_runtime_q      <= 1'b0;
          m2_ofm2ifm_stream_busy_q  <= 1'b0;
          m2_ofm2ifm_tx_runtime_q   <= 1'b0;
          m2_ofm2ifm_row_q          <= '0;
          m2_ofm2ifm_col_blk_q      <= '0;
          m2_ofm2ifm_cgrp_q         <= '0;
          m2_ofm2ifm_num_rows_q     <= next_cfg_s.h_in[ROW_W-1:0];
          m2_ofm2ifm_num_col_blks_q <= init_cols_v;
          m2_ofm2ifm_num_cgrps_q    <= (init_cgrps_v == 0) ? 16'd1 : init_cgrps_v;
          m2_ofm2ifm_initial_cols_q <= init_cols_v;
          m2_ofm2ifm_col_end_q      <= init_cols_v;
          m2_ofm2ifm_runtime_col_base_q <= 16'd0;
          m2_ofm2ifm_w_q            <= next_cfg_s.w_in;
          m2_ofm2ifm_src_layer_id_q <= cur_cfg_s.layer_id[7:0];
          m2_ofm2ifm_dst_layer_id_q <= next_cfg_s.layer_id[7:0];
          m2_ofm2ifm_seen_dst_q     <= 1'b0;
          // New current->next Mode2 context: source OFM layout is Mode2.
          // Drop any stale runtime tokens that belonged to the previous
          // destination context; they are no longer needed once the producer
          // layer has completed and this new context is armed.
          m2_ofm2ifm_src_mode2_q    <= 1'b1;
          m2_ofm2ifm_runtime_src_mode2_q <= 1'b1;
          m2_free_fifo_rptr_q       <= '0;
          m2_free_fifo_wptr_q       <= '0;
          m2_free_fifo_count_q      <= '0;
        end

        if (!m2_ofm2ifm_arm_s && m1_to_m2_transition_ready_s) begin
          init_cols_v  = (next_cfg_s.w_in == 0) ? 16'd1 :
                         ((next_cfg_s.w_in < 16'(PC_MODE2)) ? next_cfg_s.w_in : 16'(PC_MODE2));
          init_cgrps_v = (next_cfg_s.c_in == 0) ? 16'd1 :
                         ((next_cfg_s.c_in + 16'(PC_MODE2) - 16'd1) / 16'(PC_MODE2));
          m2_ofm2ifm_active_q       <= 1'b1;
          m2_ofm2ifm_ready_q        <= 1'b1;
          m2_ofm2ifm_runtime_q      <= 1'b0;
          m2_ofm2ifm_stream_busy_q  <= 1'b0;
          m2_ofm2ifm_tx_runtime_q   <= 1'b0;
          m2_ofm2ifm_row_q          <= '0;
          m2_ofm2ifm_col_blk_q      <= '0;
          m2_ofm2ifm_cgrp_q         <= '0;
          m2_ofm2ifm_num_rows_q     <= next_cfg_s.h_in[ROW_W-1:0];
          m2_ofm2ifm_num_col_blks_q <= init_cols_v;
          m2_ofm2ifm_num_cgrps_q    <= (init_cgrps_v == 0) ? 16'd1 : init_cgrps_v;
          m2_ofm2ifm_initial_cols_q <= init_cols_v;
          m2_ofm2ifm_col_end_q      <= init_cols_v;
          m2_ofm2ifm_runtime_col_base_q <= 16'd0;
          m2_ofm2ifm_w_q            <= next_cfg_s.w_in;
          m2_ofm2ifm_src_layer_id_q <= cur_cfg_s.layer_id[7:0];
          m2_ofm2ifm_dst_layer_id_q <= next_cfg_s.layer_id[7:0];
          m2_ofm2ifm_seen_dst_q     <= 1'b0;
          // M1->M2 transition context: source OFM layout is Mode1.
          // Clear stale runtime tokens from any older context.
          m2_ofm2ifm_src_mode2_q    <= 1'b0;
          m2_ofm2ifm_runtime_src_mode2_q <= 1'b0;
          m2_free_fifo_rptr_q       <= '0;
          m2_free_fifo_wptr_q       <= '0;
          m2_free_fifo_count_q      <= '0;
        end

        // Runtime refill after the destination Mode2 layer has started.
        // This mirrors the Mode1 runtime path: capture the one-cycle free token
        // into a pending request so it cannot be lost while the stream datapath
        // arbitrates with transition/Mode1 traffic.
        if (!m2_ofm2ifm_arm_s && m2_free_entry_take_s) begin
          m2_ofm2ifm_runtime_q          <= 1'b1;
          m2_ofm2ifm_row_q              <= ROW_W'(m2_free_head_row_g_s);
          m2_ofm2ifm_col_blk_q          <= m2_free_head_col_g_s;
          m2_ofm2ifm_runtime_col_base_q <= m2_free_head_col_g_s;
          m2_ofm2ifm_col_end_q          <= m2_free_head_col_g_s + 16'd1;
          m2_ofm2ifm_cgrp_q             <= m2_free_head_cgrp_g_s;
          m2_ofm2ifm_runtime_src_mode2_q <= m2_free_head_src_mode2_s;
        end

        if (m2_free_fifo_push_s) begin
          m2_free_fifo_row_g_q[m2_free_fifo_wptr_q]  <= ldm_m2_free_row_g_s;
          m2_free_fifo_col_g_q[m2_free_fifo_wptr_q]  <= ldm_m2_free_col_g_s;
          m2_free_fifo_col_l_q[m2_free_fifo_wptr_q]  <= ldm_m2_free_col_l_s;
          m2_free_fifo_cgrp_g_q[m2_free_fifo_wptr_q] <= ldm_m2_free_cgrp_g_s;
          // Capture the source layout with the token.  This prevents a stale
          // M1->M2 transition context from being reused after the next layer
          // becomes an M2->M2 producer/consumer pair.
          m2_free_fifo_src_mode2_q[m2_free_fifo_wptr_q] <= m2_ofm2ifm_src_mode2_q;
          if (m2_free_fifo_wptr_q == M2FQ_PTR_W'(M2FQ_DEPTH-1))
            m2_free_fifo_wptr_q <= '0;
          else
            m2_free_fifo_wptr_q <= m2_free_fifo_wptr_q + 1'b1;
        end
        else if (m2_free_entry_valid_s && m2_free_fifo_full_s && !m2_free_entry_take_s) begin
          m2_free_fifo_overflow_q <= 1'b1;
        end

        if (m2_free_entry_take_s) begin
          if (m2_free_fifo_rptr_q == M2FQ_PTR_W'(M2FQ_DEPTH-1))
            m2_free_fifo_rptr_q <= '0;
          else
            m2_free_fifo_rptr_q <= m2_free_fifo_rptr_q + 1'b1;
        end

        case ({m2_free_fifo_push_s, m2_free_entry_take_s})
          2'b10: m2_free_fifo_count_q <= m2_free_fifo_count_q + 1'b1;
          2'b01: m2_free_fifo_count_q <= m2_free_fifo_count_q - 1'b1;
          default: begin end
        endcase

        // Match Mode1's stream lifecycle: set internal busy only after a
        // command has actually been accepted onto the shared OFM->IFM stream
        // path.  This prevents a Mode2-only request from self-locking busy
        // without a real stream transaction.
        if (m2_ofm2ifm_stream_accepted_s) begin
          m2_ofm2ifm_stream_busy_q <= 1'b1;
          m2_ofm2ifm_tx_runtime_q  <= m2_ofm2ifm_runtime_q;
        end

        if (m2_ofm2ifm_stream_busy_q && ofm_ifm_stream_done) begin
          m2_ofm2ifm_stream_busy_q <= 1'b0;
          m2_ofm2ifm_tx_runtime_q  <= 1'b0;

          if (!m2_ofm2ifm_tx_runtime_q) begin
            if ((m2_ofm2ifm_cgrp_q + 16'd1) < m2_ofm2ifm_num_cgrps_q) begin
              m2_ofm2ifm_cgrp_q <= m2_ofm2ifm_cgrp_q + 16'd1;
            end
            else begin
              m2_ofm2ifm_cgrp_q <= '0;
              if ((m2_ofm2ifm_col_blk_q + 16'd1) < m2_ofm2ifm_initial_cols_q) begin
                m2_ofm2ifm_col_blk_q <= m2_ofm2ifm_col_blk_q + 16'd1;
              end
              else begin
                m2_ofm2ifm_col_blk_q <= '0;
                if ((m2_ofm2ifm_row_q + ROW_W'(1)) < m2_ofm2ifm_num_rows_q) begin
                  m2_ofm2ifm_row_q <= m2_ofm2ifm_row_q + ROW_W'(1);
                end
                else begin
                  m2_ofm2ifm_ready_q   <= 1'b1;
                  m2_ofm2ifm_row_q     <= '0;
                  m2_ofm2ifm_col_blk_q <= '0;
                  m2_ofm2ifm_cgrp_q    <= '0;
                end
              end
            end
          end
          else begin
            // Runtime Mode2 refill is one exact entry per free token.
            // The next token will provide the next {row,col,cgrp}; do not walk
            // a resident-tile cursor here.
            m2_ofm2ifm_runtime_q          <= 1'b0;
            m2_ofm2ifm_row_q              <= '0;
            m2_ofm2ifm_col_blk_q          <= '0;
            m2_ofm2ifm_cgrp_q             <= '0;
            m2_ofm2ifm_runtime_col_base_q <= '0;
            m2_ofm2ifm_col_end_q          <= m2_ofm2ifm_initial_cols_q;
            m2_runtime_stream_done_pulse_s <= 1'b1;
          end
        end

        if (!m2_ofm2ifm_arm_s &&
            m2_ofm2ifm_active_q && m2_ofm2ifm_ready_q &&
            (cur_cfg_s.layer_id[7:0] == m2_ofm2ifm_dst_layer_id_q)) begin
          m2_ofm2ifm_seen_dst_q <= 1'b1;
        end

        // Retire the previous source->destination refill context only after
        // the destination layer itself has completed and no new current->next
        // producer context needs to be armed.  Arm has priority so the next-path
        // ready event cannot be lost at a layer boundary.
        if (!m2_ofm2ifm_arm_s &&
            m2_ofm2ifm_in_dst_layer_s && dispatch_compute_done_s &&
            !m2_ofm2ifm_runtime_q && !m2_ofm2ifm_stream_busy_q) begin
          m2_ofm2ifm_active_q   <= 1'b0;
          m2_ofm2ifm_ready_q    <= 1'b0;
          m2_ofm2ifm_seen_dst_q <= 1'b0;
        end
      end
    end
  end

  assign ofm_layer_start  = kick_compute_s;
  assign ofm_cfg_src_mode = (cur_cfg_s.mode == MODE2);

  // Mode-1 final OFM geometry depends on pool_en.
  // Keep legacy behavior for pool_en=1 or X/Z; only explicit 0 selects no-pool bypass geometry.
  assign cur_m1_pool_active_s = (cur_cfg_s.mode == MODE1) && (cur_cfg_s.pool_en !== 1'b0);
  assign cur_m1_final_h_out_s = cur_m1_pool_active_s
                              ? ((cur_cfg_s.h_out > 16'd1) ? (cur_cfg_s.h_out >> 1) : 16'd1)
                              : cur_cfg_s.h_out;
  assign cur_m1_final_w_out_s = cur_m1_pool_active_s
                              ? ((cur_cfg_s.w_out > 16'd1) ? (cur_cfg_s.w_out >> 1) : 16'd1)
                              : cur_cfg_s.w_out;

  assign cur_m2_pool_active_s = (cur_cfg_s.mode == MODE2) && (cur_cfg_s.pool_en !== 1'b0);
  assign cur_m2_final_h_out_s = cur_m2_pool_active_s
                              ? ((cur_cfg_s.h_out > 16'd1) ? (cur_cfg_s.h_out >> 1) : 16'd1)
                              : cur_cfg_s.h_out;
  assign cur_m2_final_w_out_s = cur_m2_pool_active_s
                              ? ((cur_cfg_s.w_out > 16'd1) ? (cur_cfg_s.w_out >> 1) : 16'd1)
                              : cur_cfg_s.w_out;

  assign m1_pool_en       = cur_m1_pool_active_s;
  // Keep Mode1 behavior unchanged, but Mode2 pooling also consumes this cfg.
  assign ofm_cfg_pool_en  = (cur_cfg_s.mode == MODE2) ? cur_m2_pool_active_s : cur_m1_pool_active_s;

  assign ofm_cfg_h_out    = (cur_cfg_s.mode == MODE1)
                            ? cur_m1_final_h_out_s[$clog2(H_MAX+1)-1:0]
                            : cur_m2_final_h_out_s[$clog2(H_MAX+1)-1:0];
  assign ofm_cfg_w_out    = (cur_cfg_s.mode == MODE1)
                            ? cur_m1_final_w_out_s[$clog2(W_MAX+1)-1:0]
                            : cur_m2_final_w_out_s[$clog2(W_MAX+1)-1:0];
  assign cur_m1_ofm_h_for_sm_s = cur_m1_final_h_out_s;
  assign cur_m1_ofm_w_for_sm_s = cur_m1_final_w_out_s;

  // Weight DMA may target either the current layer or the next layer.
  // Initial preload in PREP uses cur_cfg_s; preload issued while current
  // compute is busy is the next-layer preload, so it must use next_cfg_s.
  always_comb begin
    wgt_dma_cfg_s = cur_cfg_s;
    if (compute_busy_s && next_valid_s) begin
      wgt_dma_cfg_s = next_cfg_s;
    end
  end

  // weight_bank_manager requires layer_done to overlap swap_req.
  // Scheduler asserts swap_weight_bank_s after compute_done_s, during the
  // wait/advance path, so include swap_weight_bank_s in the done qualifier.
  assign weight_bank_layer_done_s = compute_done_s | swap_weight_bank_s;

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
        if (sm_m1_mgr_active_s && m1_sm_ready_valid[i]) begin
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

  assign m1_ready_tok_valid_s     = sm_m1_mgr_active_s && (m1q_count_q != 0) && !m1_sm_ready_full_s;
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
        if (sm_m2_mgr_active_s && sm_m2_active && m2_sm_ready_valid[i]) begin
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

  assign m2_ready_tok_valid_s = sm_m2_mgr_active_s && sm_m2_active && (m2q_count_q != 0) && !m2_sm_ready_full_s;
  assign m2_ready_tok_row_g_s = m2q_mem[0].row_g;
  assign m2_ready_tok_col_g_s = m2q_mem[0].col_g;
  assign m2_ready_tok_cgrp_g_s= m2q_mem[0].cgrp_g;

  // --------------------------------------------------------------------------
  // Mode-1 free-token metadata expander
  // --------------------------------------------------------------------------
  assign m1f_emit_valid_s      = sm_m1_mgr_active_s && m1f_scan_active_q && !m1_sm_free_full_s;
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
                      sm_m1_mgr_active_s &&
                      !init_ifm_refill_claim_s &&
                      !ofm2ifm_free_claim_s &&
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
    sm_stream_kind_s          = OFM_STRM_IDLE;
    sm_stream_row_base_s      = '0;
    sm_stream_num_rows_s      = '0;
    sm_stream_col_base_s      = '0;
    sm_stream_m1_row_slot_l_s = '0;
    sm_stream_m1_ch_blk_g_s   = '0;
    sm_stream_m2_cgrp_g_s     = '0;

    if (!sm_exec_active_q && !transition_busy_s && !trans_ifm_stream_start_s && !ofm_ifm_stream_busy) begin
      if (sm_m1_mgr_active_s && m1_sm_req_valid_i) begin
        sm_stream_start_s         = 1'b1;
        sm_stream_kind_s          = OFM_STRM_M1_DIRECT;
        sm_stream_row_base_s      = m1_sm_row_g_i[ROW_W-1:0];
        sm_stream_num_rows_s      = ROW_W'(1);
        sm_stream_col_base_s = m1_sm_col_blk_g_i * ((next_cfg_s.pv_m1 == 0) ? 16'd1 : next_cfg_s.pv_m1);
        sm_stream_m1_row_slot_l_s = m1_sm_row_slot_l_i[BUF_ROW_W-1:0];
        sm_stream_m1_ch_blk_g_s   = m1_sm_ch_blk_g_i;
      end
      else if (sm_m2_mgr_active_s && sm_m2_active && m2_sm_req_valid_i) begin
        sm_stream_start_s         = 1'b1;
        sm_stream_kind_s          = OFM_STRM_M2_DIRECT;
        sm_stream_row_base_s      = m2_sm_row_g_i[ROW_W-1:0];
        sm_stream_num_rows_s      = ROW_W'(1);
        sm_stream_col_base_s      = m2_sm_col_g_i[COL_W-1:0];
        sm_stream_m2_cgrp_g_s     = m2_sm_cgrp_g_i;
      end
    end
  end

  assign m1_sm_req_ready_i = sm_m1_mgr_active_s && (
                             (sm_stream_start_s && m1_sm_req_valid_i) ||
                             (sm_exec_active_q && !sm_exec_mode_q &&
                              (m1_sm_row_slot_l_i == sm_exec_m1_row_slot_l_q) &&
                              (m1_sm_row_g_i      == sm_exec_row_g_q)));

  assign m2_sm_req_ready_i = sm_m2_mgr_active_s && sm_m2_active && (
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
        if (sm_stream_start_s && m1_sm_req_valid_i && sm_m1_mgr_active_s) begin
          sm_exec_active_q        <= 1'b1;
          sm_exec_mode_q          <= 1'b0;
          sm_exec_m1_row_slot_l_q <= m1_sm_row_slot_l_i;
          sm_exec_row_g_q         <= m1_sm_row_g_i;
          sm_exec_col_base_g_q    <= (m1_sm_col_blk_g_i * ((next_cfg_s.pv_m1 == 0) ? 16'd1 : next_cfg_s.pv_m1));
        end
        else if (sm_stream_start_s && !m1_sm_req_valid_i && m2_sm_req_valid_i && sm_m2_mgr_active_s && sm_m2_active) begin
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

  assign m2_sm_refill_req_valid   = sm_m2_mgr_active_s && m2_sm_req_valid_i;
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
    .layer_done(weight_bank_layer_done_s),
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
    .kick_compute(kick_compute_dispatch_s),
    .hold_compute(sched_hold_compute_s | local_hold_compute_s | init_ifm_refill_hold_s | ofm2ifm_runtime_hold_s | m2_ofm2ifm_runtime_hold_s),
    .cur_mode(cur_cfg_s.mode == MODE2),
    .m1_done(m1_done), .m1_busy(m1_busy), .m2_done(m2_done), .m2_busy(m2_busy),
    .m1_start(m1_start), .m1_step_en(m1_step_en),
    .m1_k_cur(m1_k_cur), .m1_c_cur(m1_c_cur), .m1_f_cur(m1_f_cur),
    .m1_hout_cur(m1_hout_cur), .m1_wout_cur(m1_wout_cur), .m1_w_cur(m1_w_cur),
    .m1_pv_cur(m1_pv_cur), .m1_pf_cur(m1_pf_cur),
    .m1_weight_bank_sel(m1_weight_bank_sel), .m1_weight_bank_ready(m1_weight_bank_ready),
    .m2_start(dispatch_m2_start_s), .m2_step_en(dispatch_m2_step_en_s),
    .m2_k_cur(m2_k_cur), .m2_c_cur(m2_c_cur), .m2_f_cur(m2_f_cur),
    .m2_hout_cur(m2_hout_cur), .m2_wout_cur(m2_wout_cur),
    .m2_weight_bank_sel(m2_weight_bank_sel), .m2_weight_bank_ready(m2_weight_bank_ready),
    .compute_done(dispatch_compute_done_s), .compute_busy(dispatch_compute_busy_s)
  );

  assign m2_start   = dispatch_m2_start_s;
  assign m2_step_en = dispatch_m2_step_en_s;

  // Match Mode1's compute scheduler for both modes: one global scheduler
  // compute kick maps to one compute_dispatcher layer run, and dispatcher done
  // is the layer-done indication.  Mode2 resident-tile scheduling is not used
  // as a control-level scheduler.
  assign kick_compute_dispatch_s = kick_compute_s;

  assign compute_done_s = dispatch_compute_done_s;

  assign compute_busy_s = dispatch_compute_busy_s;

  dma_phase_manager #(
    .PTOTAL(PTOTAL), .PV_MAX(PV_MAX), .PC_MODE2(PC_MODE2),
    .C_MAX(C_MAX), .W_MAX(W_MAX), .H_MAX(H_MAX), .HT(HT),
    .WGT_DEPTH(WGT_DEPTH), .OFM_LINEAR_DEPTH(OFM_LINEAR_DEPTH), .DDR_ADDR_W(DDR_ADDR_W)
  ) u_dma_phase_manager (
    .clk(clk), .rst_n(rst_n),
    .req_ifm_load(kick_ifm_load_s | init_ifm_refill_req_s), .req_wgt_load(kick_wgt_preload_s), .req_ofm_store(kick_ofm_store_s),
    .cur_cfg(cur_cfg_s), .wgt_cfg(wgt_dma_cfg_s), .preload_bank_sel(preload_bank_sel_s),
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
    .m2_tile_col_base_g(m2_tile_col_base_g),
    .ifm_rd_en(ifm_rd_en), .ifm_rd_bank_base(ifm_rd_bank_base), .ifm_rd_row_idx(ifm_rd_row_idx),
    .ifm_rd_col_idx(ifm_rd_col_idx), .ifm_rd_col_g(ifm_rd_col_g), .ifm_rd_valid(ifm_rd_valid), .ifm_rd_data(ifm_rd_data),
    .m1_pass_start_pulse(m1_pass_start_pulse), .m1_chan_done_pulse(m1_chan_done_pulse), .m1_c_iter(m1_c_iter),
    .m1_out_row(m1_out_row),
    .m1_dr_write_en(m1_dr_write_en), .m1_dr_write_row_idx(m1_dr_write_row_idx),
    .m1_dr_write_x_base(m1_dr_write_x_base), .m1_dr_write_data(m1_dr_write_data),
    .m2_start(m2_start), .m2_pass_start_pulse(m2_pass_start_pulse), .m2_mac_en(m2_mac_en),
    .m2_ce_out_valid(m2_ce_out_valid), .m2_out_row(m2_out_row_g_s), .m2_out_col(m2_out_col_g_s), .m2_f_group(m2_f_group),
    .m2_dr_write_en(m2_dr_write_en), .m2_dr_write_row_idx(m2_dr_write_row_idx), .m2_dr_write_data(m2_dr_write_data),
    .hold_compute(local_hold_compute_s), .local_busy(local_busy_s), .local_done(local_done_s), .local_error(local_error_s),
    .m1_local_busy(), .m2_local_busy(),
    .m2_free_ready(ldm_m2_free_ready_s),
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
    .cfg_cur_h_out(cur_m2_final_h_out_s), .cfg_cur_w_out(cur_m2_final_w_out_s), .cfg_cur_f_out(cur_cfg_s.f_out),
    .cfg_next_h_in(next_cfg_s.h_in), .cfg_next_w_in(next_cfg_s.w_in), .cfg_next_c_in(next_cfg_s.c_in),
    .cfg_next_pc(next_cfg_s.pc_m2), .cfg_next_pf(next_cfg_s.pf_m2),
    .free_valid(sm_m2_mgr_active_s && ldm_m2_free_valid_s && sm_m2_active),
    .free_row_g(ldm_m2_free_row_g_s),
    .free_col_g(ldm_m2_free_col_g_s),
    .free_col_l(ldm_m2_free_col_l_s),
    .free_cgrp_g(ldm_m2_free_cgrp_g_s),
    .ready_valid(sm_m2_mgr_active_s && m2_ready_tok_valid_s),
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

  assign sm_m1_drain_idle_s = !m1_sm_busy_s &&
                              (m1q_count_q == '0) &&
                              (m1f_count_q == '0) &&
                              !m1f_scan_active_q;

  assign sm_m2_drain_idle_s = !m2_sm_busy_s &&
                              (m2q_count_q == '0);

  assign same_mode_legacy_drain_done_s = (sm_m1_mgr_active_s || (sm_m2_mgr_active_s && sm_m2_active)) &&
                                  ofm_layer_write_done &&
                                  !sm_exec_active_q &&
                                  !sm_stream_start_s &&
                                  !ofm_ifm_stream_busy &&
                                  ((sm_m1_mgr_active_s && sm_m1_drain_idle_s) ||
                                   (sm_m2_mgr_active_s && sm_m2_active && sm_m2_drain_idle_s));

  assign same_mode_drain_done_s = sm_m1_active ?
                                  same_mode_initial_tile_ready_s :
                                  (sm_m2_active ? ((cur_cfg_s.mode == MODE2) &&
                                                   (next_cfg_s.mode == MODE2) &&
                                                   m2_ofm2ifm_active_q &&
                                                   m2_ofm2ifm_ready_q &&
                                                   (m2_ofm2ifm_src_layer_id_q == cur_cfg_s.layer_id[7:0]) &&
                                                   (m2_ofm2ifm_dst_layer_id_q == next_cfg_s.layer_id[7:0]))
                                                : same_mode_legacy_drain_done_s);

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
  assign ofm_ifm_stream_start = trans_ifm_stream_start_s |
                                  ofm2ifm_stream_start_s |
                                  (((cur_cfg_s.mode == MODE2) && m2_ofm2ifm_stream_start_s) ? 1'b1 : 1'b0) |
                                  sm_stream_start_s;

  // Stream-kind mux uses the exact same priority as the stream payload mux.
  assign ofm_ifm_stream_kind = trans_ifm_stream_start_s ? OFM_STRM_M1_TO_M2 :
                               (ofm2ifm_stream_start_s ? OFM_STRM_M1_DIRECT :
                               (((cur_cfg_s.mode == MODE2) && m2_ofm2ifm_stream_start_s) ?
                                  (m2_ofm2ifm_stream_src_mode2_s ? OFM_STRM_M2_DIRECT : OFM_STRM_M1_TO_M2) :
                                  (sm_stream_start_s ? sm_stream_kind_s : OFM_STRM_IDLE)));

  assign ofm_ifm_stream_row_base = trans_ifm_stream_start_s ? trans_ifm_stream_row_base_s :
                                   (ofm2ifm_stream_start_s ? ofm2ifm_row_q :
                                   (((cur_cfg_s.mode == MODE2) && m2_ofm2ifm_stream_start_s) ? m2_ofm2ifm_row_q : sm_stream_row_base_s));
  assign ofm_ifm_stream_num_rows = trans_ifm_stream_start_s ? trans_ifm_stream_num_rows_s :
                                   ((ofm2ifm_stream_start_s || ((cur_cfg_s.mode == MODE2) && m2_ofm2ifm_stream_start_s)) ? ROW_W'(1) : sm_stream_num_rows_s);
  assign ofm_ifm_stream_col_base = trans_ifm_stream_start_s ? trans_ifm_stream_col_base_s :
                                   (ofm2ifm_stream_start_s ?
                                    (ofm2ifm_col_blk_q[COL_W-1:0] * ofm2ifm_pv_q[COL_W-1:0]) :
                                   (((cur_cfg_s.mode == MODE2) && m2_ofm2ifm_stream_start_s) ?
                                    m2_ofm2ifm_col_blk_q[COL_W-1:0] :
                                    sm_stream_col_base_s));
  assign ofm_ifm_stream_m1_row_slot_l = trans_ifm_stream_start_s ? '0 :
                                        (ofm2ifm_stream_start_s ? ofm2ifm_row_slot_q : '0);
  assign ofm_ifm_stream_m1_ch_blk_g   = trans_ifm_stream_start_s ? '0 :
                                        (ofm2ifm_stream_start_s ? ofm2ifm_ch_blk_q : sm_stream_m1_ch_blk_g_s);
  assign ofm_ifm_stream_m2_cgrp_g     = trans_ifm_stream_start_s ? '0 :
                                        (((cur_cfg_s.mode == MODE2) && m2_ofm2ifm_stream_start_s) ? m2_ofm2ifm_cgrp_q : sm_stream_m2_cgrp_g_s);

  assign control_error_s = m1_sm_error_s |
                           m1q_overflow_q | m1f_overflow_q |
                           m2_free_fifo_overflow_q |
                           m1_sm_free_full_s | m1_sm_ready_full_s |
                           (sm_m2_mgr_active_s ?
                            (m2_sm_error_s | m2q_overflow_q |
                             m2_sm_free_full_s | m2_sm_ready_full_s) : 1'b0);

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

// -----------------------------------------------------------------------------
// DEBUG ONLY: Mode2 next-path lifecycle monitor
// Define DBG_M2_NEXTPATH_MON to enable. Does not change datapath behavior.
// -----------------------------------------------------------------------------

logic [7:0] dbg_np_layer_q;
logic       dbg_np_active_q;
logic       dbg_np_ready_q;
logic       dbg_np_req_q;
logic       dbg_np_grant_q;
logic       dbg_np_acc_q;
logic       dbg_np_m2_start_q;
logic       dbg_np_m2_busy_q;
logic       dbg_np_ofm_start_q;
logic       dbg_np_ofm_busy_q;
logic       dbg_np_ofm_done_q;
logic       dbg_np_ofm_layer_done_q;
logic       dbg_np_next_path_done_q;
logic       dbg_np_advance_q;
logic [31:0] dbg_np_stuck_cnt_q;

always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    dbg_np_layer_q          <= '0;
    dbg_np_active_q         <= 1'b0;
    dbg_np_ready_q          <= 1'b0;
    dbg_np_req_q            <= 1'b0;
    dbg_np_grant_q          <= 1'b0;
    dbg_np_acc_q            <= 1'b0;
    dbg_np_m2_start_q       <= 1'b0;
    dbg_np_m2_busy_q        <= 1'b0;
    dbg_np_ofm_start_q      <= 1'b0;
    dbg_np_ofm_busy_q       <= 1'b0;
    dbg_np_ofm_done_q       <= 1'b0;
    dbg_np_ofm_layer_done_q <= 1'b0;
    dbg_np_next_path_done_q <= 1'b0;
    dbg_np_advance_q        <= 1'b0;
    dbg_np_stuck_cnt_q      <= 32'd0;
  end
  else begin
    dbg_np_layer_q          <= cur_cfg_s.layer_id[7:0];
    dbg_np_active_q         <= m2_ofm2ifm_active_q;
    dbg_np_ready_q          <= m2_ofm2ifm_ready_q;
    dbg_np_req_q            <= m2_ofm2ifm_stream_req_s;
    dbg_np_grant_q          <= m2_ofm2ifm_stream_grant_s;
    dbg_np_acc_q            <= m2_ofm2ifm_stream_accepted_s;
    dbg_np_m2_start_q       <= m2_ofm2ifm_stream_start_s;
    dbg_np_m2_busy_q        <= m2_ofm2ifm_stream_busy_q;
    dbg_np_ofm_start_q      <= ofm_ifm_stream_start;
    dbg_np_ofm_busy_q       <= ofm_ifm_stream_busy;
    dbg_np_ofm_done_q       <= ofm_ifm_stream_done;
    dbg_np_ofm_layer_done_q <= ofm_layer_write_done;
    dbg_np_next_path_done_q <= sched_next_path_done_s;
    dbg_np_advance_q        <= advance_layer_s;

    if ((cur_cfg_s.layer_id[7:0] != dbg_np_layer_q) ||
        (m2_ofm2ifm_active_q != dbg_np_active_q) ||
        (m2_ofm2ifm_ready_q != dbg_np_ready_q) ||
        (m2_ofm2ifm_stream_req_s != dbg_np_req_q) ||
        (m2_ofm2ifm_stream_grant_s != dbg_np_grant_q) ||
        (m2_ofm2ifm_stream_accepted_s != dbg_np_acc_q) ||
        (m2_ofm2ifm_stream_start_s != dbg_np_m2_start_q) ||
        (m2_ofm2ifm_stream_busy_q != dbg_np_m2_busy_q) ||
        (ofm_ifm_stream_start != dbg_np_ofm_start_q) ||
        (ofm_ifm_stream_busy != dbg_np_ofm_busy_q) ||
        (ofm_ifm_stream_done != dbg_np_ofm_done_q) ||
        (ofm_layer_write_done != dbg_np_ofm_layer_done_q) ||
        (sched_next_path_done_s != dbg_np_next_path_done_q) ||
        (advance_layer_s != dbg_np_advance_q)) begin

      $display("DBG_M2_EDGE t=%0t layer=%0d next=%0d mode=%0d next_mode=%0d active=%0b ready=%0b src=%0d dst=%0d ctx_cur_next=%0b row=%0d col=%0d cgrp=%0d req=%0b grant=%0b acc=%0b m2_start=%0b m2_busy=%0b ofm_start=%0b ofm_busy=%0b ofm_done=%0b ofm_layer_done=%0b next_done=%0b advance=%0b runtime=%0b free_v=%0b free_take=%0b",
               $time,
               cur_cfg_s.layer_id[7:0],
               next_cfg_s.layer_id[7:0],
               cur_cfg_s.mode,
               next_cfg_s.mode,
               m2_ofm2ifm_active_q,
               m2_ofm2ifm_ready_q,
               m2_ofm2ifm_src_layer_id_q,
               m2_ofm2ifm_dst_layer_id_q,
               m2_ofm2ifm_ctx_cur_to_next_s,
               m2_ofm2ifm_row_q,
               m2_ofm2ifm_col_blk_q,
               m2_ofm2ifm_cgrp_q,
               m2_ofm2ifm_stream_req_s,
               m2_ofm2ifm_stream_grant_s,
               m2_ofm2ifm_stream_accepted_s,
               m2_ofm2ifm_stream_start_s,
               m2_ofm2ifm_stream_busy_q,
               ofm_ifm_stream_start,
               ofm_ifm_stream_busy,
               ofm_ifm_stream_done,
               ofm_layer_write_done,
               sched_next_path_done_s,
               advance_layer_s,
               m2_ofm2ifm_runtime_q,
               m2_free_entry_valid_s,
               m2_free_entry_take_s);
    end

    if ((cur_cfg_s.layer_id[7:0] == 8'd1) && busy && !done) begin
      dbg_np_stuck_cnt_q <= dbg_np_stuck_cnt_q + 32'd1;

      if (dbg_np_stuck_cnt_q[15:0] == 16'hffff) begin
        $display("DBG_M2_STUCK t=%0t layer=%0d next=%0d active=%0b ready=%0b src=%0d dst=%0d row=%0d col=%0d cgrp=%0d req=%0b grant=%0b acc=%0b m2_start=%0b m2_busy=%0b ofm_start=%0b ofm_busy=%0b ofm_done=%0b ofm_layer_done=%0b next_done=%0b advance=%0b",
                 $time,
                 cur_cfg_s.layer_id[7:0],
                 next_cfg_s.layer_id[7:0],
                 m2_ofm2ifm_active_q,
                 m2_ofm2ifm_ready_q,
                 m2_ofm2ifm_src_layer_id_q,
                 m2_ofm2ifm_dst_layer_id_q,
                 m2_ofm2ifm_row_q,
                 m2_ofm2ifm_col_blk_q,
                 m2_ofm2ifm_cgrp_q,
                 m2_ofm2ifm_stream_req_s,
                 m2_ofm2ifm_stream_grant_s,
                 m2_ofm2ifm_stream_accepted_s,
                 m2_ofm2ifm_stream_start_s,
                 m2_ofm2ifm_stream_busy_q,
                 ofm_ifm_stream_start,
                 ofm_ifm_stream_busy,
                 ofm_ifm_stream_done,
                 ofm_layer_write_done,
                 sched_next_path_done_s,
                 advance_layer_s);
      end
    end
    else begin
      dbg_np_stuck_cnt_q <= 32'd0;
    end
  end
end

always_ff @(posedge clk) begin : DBG_M1_ROW_ADV_GUARD_MON
    if (rst_n && (cur_cfg_s.mode == MODE1) && m1_out_row_done_pulse) begin
        $display("DBG_M1_ROW_ADV_GUARD t=%0t layer=%0d out_row=%0d row_done=1 first_pending=%0d adv=%0d",
            $time,
            dbg_layer_idx,
            m1_out_row,
            m1_pady1_first_row_pending_q,
            ifm_m1_advance_row
        );
    end
end


endmodule
