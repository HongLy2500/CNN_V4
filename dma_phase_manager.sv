`include "cnn_ddr_defs.svh"

module dma_phase_manager
  import cnn_layer_desc_pkg::*;
#(
  parameter int PTOTAL           = 16,
  parameter int PV_MAX           = 8,
  parameter int PC_MODE2         = 8,
  parameter int C_MAX            = 64,
  parameter int W_MAX            = 224,
  parameter int H_MAX            = 224,
  parameter int HT               = 8,
  parameter int WGT_DEPTH        = 4096,
  parameter int OFM_LINEAR_DEPTH = 4096,
  parameter int DDR_ADDR_W       = `CNN_DDR_ADDR_W
)(
  input  logic clk,
  input  logic rst_n,

  // Requests from scheduler. These may be 1-cycle pulses; the manager
  // latches them as pending until the corresponding phase completes.
  input  logic req_ifm_load,
  input  logic req_wgt_load,
  input  logic req_ofm_store,

  // Current-layer configuration / context
  input  layer_desc_t cur_cfg,

  // Weight-DMA configuration. This may be either cur_cfg or next_cfg,
  // selected by control_unit_top according to the preload context.
  input  layer_desc_t wgt_cfg,

  input  logic        preload_bank_sel,

  // IFM request context
  input  logic [$clog2(H_MAX+1)-1:0] req_ifm_abs_row_base,
  input  logic [$clog2(H_MAX+1)-1:0] req_ifm_num_rows,
  input  logic [((H_MAX <= 1) ? 1 : $clog2(H_MAX))-1:0] req_ifm_buf_row_base,
  input  logic [(((W_MAX + PC_MODE2 - 1) / PC_MODE2) <= 1 ? 1 : $clog2(((W_MAX + PC_MODE2 - 1) / PC_MODE2) + 1))-1:0] req_m2_tile_idx,

  // OFM request context
  input  logic [31:0] ofm_layer_num_words,
  input  logic [((OFM_LINEAR_DEPTH <= 1) ? 1 : $clog2(OFM_LINEAR_DEPTH))-1:0] ofm_buf_base,
  input  logic ofm_layer_write_done,
  input  logic ofm_error,

  // Shared DMA status
  input  logic dma_busy,
  input  logic dma_done_ifm,
  input  logic dma_done_wgt,
  input  logic dma_done_ofm,
  input  logic dma_error,

  // Shared command outputs to cnn_dma_direct
  output logic ifm_cmd_start,
  output logic [DDR_ADDR_W-1:0] ifm_cmd_ddr_base,
  output logic [$clog2(H_MAX+1)-1:0] ifm_cmd_num_rows,
  output logic [((H_MAX <= 1) ? 1 : $clog2(H_MAX))-1:0] ifm_cmd_buf_row_base,

  output logic wgt_cmd_start,
  output logic wgt_cmd_buf_sel,
  output logic [DDR_ADDR_W-1:0] wgt_cmd_ddr_base,
  output logic [$clog2(WGT_DEPTH+1)-1:0] wgt_cmd_num_words,

  output logic ofm_cmd_start,
  output logic [DDR_ADDR_W-1:0] ofm_cmd_ddr_base,
  output logic [$clog2(OFM_LINEAR_DEPTH+1)-1:0] ofm_cmd_num_words,
  output logic [((OFM_LINEAR_DEPTH <= 1) ? 1 : $clog2(OFM_LINEAR_DEPTH))-1:0] ofm_cmd_buf_base,

  // Phase completion back to scheduler
  output logic ifm_load_done,
  output logic wgt_load_done,
  output logic ofm_store_done,
  output logic phase_error,

  // Optional phase visibility
  output logic phase_busy,
  output logic [1:0] dbg_active_phase,   // 0:none, 1:ifm, 2:wgt, 3:ofm
  output logic [2:0] dbg_pending_mask    // {ofm, wgt, ifm}
);

  localparam int BUF_ROW_W = (H_MAX <= 1) ? 1 : $clog2(H_MAX);
  localparam int OFM_AW    = (OFM_LINEAR_DEPTH <= 1) ? 1 : $clog2(OFM_LINEAR_DEPTH);
  localparam int WGT_NW    = (WGT_DEPTH <= 1) ? 1 : $clog2(WGT_DEPTH+1);
  localparam int M2_TILE_W = ((((W_MAX + PC_MODE2 - 1) / PC_MODE2) <= 1) ? 1 :
                              $clog2(((W_MAX + PC_MODE2 - 1) / PC_MODE2) + 1));

  typedef enum logic [1:0] {
    PH_NONE = 2'd0,
    PH_IFM  = 2'd1,
    PH_WGT  = 2'd2,
    PH_OFM  = 2'd3
  } phase_t;

  logic pending_ifm_q, pending_ifm_d;
  logic pending_wgt_q, pending_wgt_d;
  logic pending_ofm_q, pending_ofm_d;

  // Latched IFM request context. IFM refill requests can be one-cycle
  // pulses and may wait behind a weight/ofm phase. Keep the row/tile
  // context stable until addr_gen_ifm_ddr is granted.
  logic [$clog2(H_MAX+1)-1:0] ifm_req_abs_row_base_q;
  logic [$clog2(H_MAX+1)-1:0] ifm_req_num_rows_q;
  logic [BUF_ROW_W-1:0]       ifm_req_buf_row_base_q;
  logic [M2_TILE_W-1:0]       ifm_req_m2_tile_idx_q;

  phase_t active_phase_q, active_phase_d;
  logic   subgen_active_q, subgen_active_d;

  logic start_ifm_gen, start_wgt_gen, start_ofm_gen;
  phase_t grant_phase;
  phase_t selected_phase_for_mux;

  // --------------------------------------------------------------------------
  // Sub-generator wires
  // --------------------------------------------------------------------------
  logic ifm_gen_done, ifm_gen_error;
  logic ifm_gen_cmd_start;
  logic [DDR_ADDR_W-1:0] ifm_gen_cmd_ddr_base;
  logic [$clog2(H_MAX+1)-1:0] ifm_gen_cmd_num_rows;
  logic [BUF_ROW_W-1:0] ifm_gen_cmd_buf_row_base;

  logic wgt_gen_done, wgt_gen_error;
  logic wgt_gen_cmd_start, wgt_gen_cmd_buf_sel;
  logic [DDR_ADDR_W-1:0] wgt_gen_cmd_ddr_base;
  logic [WGT_NW-1:0]     wgt_gen_cmd_num_words;

  logic ofm_gen_done, ofm_gen_error;
  logic ofm_gen_cmd_start;
  logic [DDR_ADDR_W-1:0] ofm_gen_cmd_ddr_base;
  logic [$clog2(OFM_LINEAR_DEPTH+1)-1:0] ofm_gen_cmd_num_words;
  logic [OFM_AW-1:0] ofm_gen_cmd_buf_base;

  // --------------------------------------------------------------------------
  // Request latching
  // --------------------------------------------------------------------------
  always_comb begin
    pending_ifm_d = pending_ifm_q || req_ifm_load;
    pending_wgt_d = pending_wgt_q || req_wgt_load;
    pending_ofm_d = pending_ofm_q || req_ofm_store;

    if (ifm_gen_done || ifm_gen_error) pending_ifm_d = 1'b0;
    if (wgt_gen_done || wgt_gen_error) pending_wgt_d = 1'b0;
    if (ofm_gen_done || ofm_gen_error) pending_ofm_d = 1'b0;
  end

  // --------------------------------------------------------------------------
  // Arbitration
  // Static priority while idle: WGT > IFM > OFM.
  // The scheduler can still avoid simultaneous requests if a different
  // system-level policy is desired.
  // --------------------------------------------------------------------------
  always_comb begin
    grant_phase   = PH_NONE;
    start_ifm_gen = 1'b0;
    start_wgt_gen = 1'b0;
    start_ofm_gen = 1'b0;

    if (!subgen_active_q) begin
      if (pending_wgt_q) begin
        grant_phase   = PH_WGT;
        start_wgt_gen = 1'b1;
      end
      else if (pending_ifm_q) begin
        grant_phase   = PH_IFM;
        start_ifm_gen = 1'b1;
      end
      else if (pending_ofm_q) begin
        grant_phase   = PH_OFM;
        start_ofm_gen = 1'b1;
      end
    end
  end

  // --------------------------------------------------------------------------
  // Active phase bookkeeping
  // --------------------------------------------------------------------------
  always_comb begin
    active_phase_d  = active_phase_q;
    subgen_active_d = subgen_active_q;

    if (!subgen_active_q) begin
      if (grant_phase != PH_NONE) begin
        active_phase_d  = grant_phase;
        subgen_active_d = 1'b1;
      end
      else begin
        active_phase_d  = PH_NONE;
        subgen_active_d = 1'b0;
      end
    end
    else begin
      case (active_phase_q)
        PH_IFM: begin
          if (ifm_gen_done || ifm_gen_error) begin
            active_phase_d  = PH_NONE;
            subgen_active_d = 1'b0;
          end
        end

        PH_WGT: begin
          if (wgt_gen_done || wgt_gen_error) begin
            active_phase_d  = PH_NONE;
            subgen_active_d = 1'b0;
          end
        end

        PH_OFM: begin
          if (ofm_gen_done || ofm_gen_error) begin
            active_phase_d  = PH_NONE;
            subgen_active_d = 1'b0;
          end
        end

        default: begin
          active_phase_d  = PH_NONE;
          subgen_active_d = 1'b0;
        end
      endcase
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pending_ifm_q            <= 1'b0;
      pending_wgt_q            <= 1'b0;
      pending_ofm_q            <= 1'b0;
      active_phase_q           <= PH_NONE;
      subgen_active_q          <= 1'b0;
      ifm_req_abs_row_base_q   <= '0;
      ifm_req_num_rows_q       <= '0;
      ifm_req_buf_row_base_q   <= '0;
      ifm_req_m2_tile_idx_q    <= '0;
    end
    else begin
      pending_ifm_q   <= pending_ifm_d;
      pending_wgt_q   <= pending_wgt_d;
      pending_ofm_q   <= pending_ofm_d;
      active_phase_q  <= active_phase_d;
      subgen_active_q <= subgen_active_d;

      // Capture IFM request context exactly when a new IFM request is
      // accepted into the pending slot. The control unit keeps requests
      // one-at-a-time, so a request while pending_ifm_q is already set is
      // intentionally ignored here rather than overwriting the active one.
      if (req_ifm_load && !pending_ifm_q) begin
        ifm_req_abs_row_base_q <= req_ifm_abs_row_base;
        ifm_req_num_rows_q     <= req_ifm_num_rows;
        ifm_req_buf_row_base_q <= req_ifm_buf_row_base;
        ifm_req_m2_tile_idx_q  <= req_m2_tile_idx;
      end
    end
  end

  // Forward command of the currently active phase. On the first grant cycle,
  // use grant_phase so the initial *_cmd_start pulse is not lost.
  assign selected_phase_for_mux =
    subgen_active_q ? active_phase_q : grant_phase;

  // --------------------------------------------------------------------------
  // Sub-generator instantiation
  // --------------------------------------------------------------------------
  addr_gen_ifm_ddr #(
    .PV_MAX    (PV_MAX),
    .PC        (PC_MODE2),
    .C_MAX     (C_MAX),
    .W_MAX     (W_MAX),
    .H_MAX     (H_MAX),
    .HT        (HT),
    .DDR_ADDR_W(DDR_ADDR_W)
  ) u_addr_gen_ifm_ddr (
    .clk                 (clk),
    .rst_n               (rst_n),
    .start               (start_ifm_gen),
    .cfg_mode            (cur_cfg.mode == MODE2),
    .cfg_ifm_ddr_base    (cur_cfg.ifm_ddr_base),
    .cfg_w_layer         (cur_cfg.w_in[$clog2(W_MAX+1)-1:0]),
    .cfg_h_in            (cur_cfg.h_in[$clog2(H_MAX+1)-1:0]),
    .cfg_c_in            (cur_cfg.c_in[$clog2(C_MAX+1)-1:0]),
    .cfg_pv_cur          (cur_cfg.pv_m1[$clog2(PV_MAX+1)-1:0]),
    .req_abs_row_base    (ifm_req_abs_row_base_q),
    .req_num_rows        (ifm_req_num_rows_q),
    .req_buf_row_base    (ifm_req_buf_row_base_q),
    .req_m2_tile_idx     (ifm_req_m2_tile_idx_q),
    .dma_busy            (dma_busy),
    .dma_done_ifm        (dma_done_ifm),
    .dma_error           (dma_error),
    .ifm_cmd_start       (ifm_gen_cmd_start),
    .ifm_cmd_ddr_base    (ifm_gen_cmd_ddr_base),
    .ifm_cmd_num_rows    (ifm_gen_cmd_num_rows),
    .ifm_cmd_buf_row_base(ifm_gen_cmd_buf_row_base),
    .busy                (),
    .done                (ifm_gen_done),
    .error               (ifm_gen_error),
    .dbg_mode1_words_per_row(),
    .dbg_mode2_num_tiles (),
    .dbg_total_words     (),
    .dbg_ddr_base_word_addr(),
    .dbg_ddr_last_word_addr(),
    .dbg_waiting_for_dma ()
  );

  addr_gen_wgt_ddr #(
    .PTOTAL     (PTOTAL),
    .PC_MODE2   (PC_MODE2),
    .WGT_DEPTH  (WGT_DEPTH),
    .DDR_ADDR_W (DDR_ADDR_W)
  ) u_addr_gen_wgt_ddr (
    .clk                 (clk),
    .rst_n               (rst_n),
    .start               (start_wgt_gen),
    .cfg_mode            (wgt_cfg.mode == MODE2),
    .cfg_buf_sel         (preload_bank_sel),
    .cfg_k_cur           (wgt_cfg.k),
    .cfg_c_cur           (wgt_cfg.c_in),
    .cfg_f_cur           (wgt_cfg.f_out),
    .cfg_pv_mode1_cur    (wgt_cfg.pv_m1),
    .cfg_pf_mode1_cur    (wgt_cfg.pf_m1),
    .cfg_pf_mode2        (wgt_cfg.pf_m2),
    .cfg_wgt_ddr_base    (wgt_cfg.wgt_ddr_base),
    .dma_busy            (dma_busy),
    .dma_done_wgt        (dma_done_wgt),
    .dma_error           (dma_error),
    .wgt_cmd_start       (wgt_gen_cmd_start),
    .wgt_cmd_buf_sel     (wgt_gen_cmd_buf_sel),
    .wgt_cmd_ddr_base    (wgt_gen_cmd_ddr_base),
    .wgt_cmd_num_words   (wgt_gen_cmd_num_words),
    .busy                (),
    .done                (wgt_gen_done),
    .error               (wgt_gen_error),
    .dbg_num_fgroup      (),
    .dbg_num_cgroup      (),
    .dbg_num_logical_bundles(),
    .dbg_num_physical_words()
  );

  addr_gen_ofm_ddr #(
    .DDR_ADDR_W       (DDR_ADDR_W),
    .OFM_LINEAR_DEPTH (OFM_LINEAR_DEPTH)
  ) u_addr_gen_ofm_ddr (
    .clk                 (clk),
    .rst_n               (rst_n),
    .start               (start_ofm_gen),
    .cfg_ofm_ddr_base    (cur_cfg.ofm_ddr_base),
    .cfg_ofm_buf_base    (ofm_buf_base),
    .ofm_layer_num_words (ofm_layer_num_words),
    .ofm_layer_write_done(ofm_layer_write_done),
    .ofm_error           (ofm_error),
    .dma_busy            (dma_busy),
    .dma_done_ofm        (dma_done_ofm),
    .dma_error           (dma_error),
    .ofm_cmd_start       (ofm_gen_cmd_start),
    .ofm_cmd_ddr_base    (ofm_gen_cmd_ddr_base),
    .ofm_cmd_num_words   (ofm_gen_cmd_num_words),
    .ofm_cmd_buf_base    (ofm_gen_cmd_buf_base),
    .busy                (),
    .done                (ofm_gen_done),
    .error               (ofm_gen_error),
    .dbg_num_words       (),
    .dbg_ddr_last_addr   (),
    .dbg_buf_last_addr   (),
    .dbg_waiting_for_layer(),
    .dbg_waiting_for_dma ()
  );

  // --------------------------------------------------------------------------
  // Shared command mux
  // --------------------------------------------------------------------------
  always_comb begin
    ifm_cmd_start        = 1'b0;
    ifm_cmd_ddr_base     = '0;
    ifm_cmd_num_rows     = '0;
    ifm_cmd_buf_row_base = '0;

    wgt_cmd_start        = 1'b0;
    wgt_cmd_buf_sel      = 1'b0;
    wgt_cmd_ddr_base     = '0;
    wgt_cmd_num_words    = '0;

    ofm_cmd_start        = 1'b0;
    ofm_cmd_ddr_base     = '0;
    ofm_cmd_num_words    = '0;
    ofm_cmd_buf_base     = '0;

    case (selected_phase_for_mux)
      PH_IFM: begin
        ifm_cmd_start        = ifm_gen_cmd_start;
        ifm_cmd_ddr_base     = ifm_gen_cmd_ddr_base;
        ifm_cmd_num_rows     = ifm_gen_cmd_num_rows;
        ifm_cmd_buf_row_base = ifm_gen_cmd_buf_row_base;
      end

      PH_WGT: begin
        wgt_cmd_start        = wgt_gen_cmd_start;
        wgt_cmd_buf_sel      = wgt_gen_cmd_buf_sel;
        wgt_cmd_ddr_base     = wgt_gen_cmd_ddr_base;
        wgt_cmd_num_words    = wgt_gen_cmd_num_words;
      end

      PH_OFM: begin
        ofm_cmd_start        = ofm_gen_cmd_start;
        ofm_cmd_ddr_base     = ofm_gen_cmd_ddr_base;
        ofm_cmd_num_words    = ofm_gen_cmd_num_words;
        ofm_cmd_buf_base     = ofm_gen_cmd_buf_base;
      end

      default: begin end
    endcase
  end

  // --------------------------------------------------------------------------
  // Completion / error summary to scheduler
  // --------------------------------------------------------------------------
  assign ifm_load_done  = ifm_gen_done;
  assign wgt_load_done  = wgt_gen_done;
  assign ofm_store_done = ofm_gen_done;
  assign phase_error    = ifm_gen_error | wgt_gen_error | ofm_gen_error;

  assign phase_busy       = subgen_active_q || (grant_phase != PH_NONE);
  assign dbg_active_phase = selected_phase_for_mux;
  assign dbg_pending_mask = {pending_ofm_q, pending_wgt_q, pending_ifm_q};

endmodule
