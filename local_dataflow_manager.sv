module local_dataflow_manager
  import cnn_layer_desc_pkg::*;
#(
  parameter int DATA_W   = 8,
  parameter int PTOTAL   = 256,
  parameter int PF_MAX   = 16,
  parameter int PV_MAX   = 16,
  parameter int PC_MODE2 = 16,
  parameter int PF_MODE2 = 16,
  parameter int C_MAX    = 512,
  parameter int W_MAX    = 224,
  parameter int H_MAX    = 224,
  parameter int K_MAX    = 7
)(
  input  logic clk,
  input  logic rst_n,

  // Current layer configuration / selected compute mode
  input  layer_desc_t cur_cfg,
  input  logic        cur_mode,   // 0: mode1, 1: mode2

  // --------------------------------------------------------------------------
  // Shared IFM buffer read port (one physical port, selected by mode)
  // --------------------------------------------------------------------------
  output logic                     ifm_rd_en,
  output logic [$clog2(C_MAX)-1:0] ifm_rd_bank_base,
  output logic [$clog2(H_MAX)-1:0] ifm_rd_row_idx,
  output logic [$clog2(W_MAX)-1:0] ifm_rd_col_idx,
  output logic [$clog2(W_MAX)-1:0] ifm_rd_col_g,
  input  logic                     ifm_rd_valid,
  input  logic [PV_MAX*DATA_W-1:0] ifm_rd_data,

  // --------------------------------------------------------------------------
  // Inputs from mode1_compute_top
  // --------------------------------------------------------------------------
  input  logic        m1_pass_start_pulse,
  input  logic        m1_chan_done_pulse,
  input  logic [15:0] m1_c_iter,
  input  logic [15:0] m1_out_row,
  input  logic [15:0] m1_out_col,

  output logic                     m1_dr_write_en,
  output logic [$clog2(K_MAX)-1:0] m1_dr_write_row_idx,
  output logic [15:0]              m1_dr_write_x_base,
  output logic [PV_MAX*DATA_W-1:0] m1_dr_write_data,

  // --------------------------------------------------------------------------
  // Inputs from mode2_compute_top
  //
  // FINAL ROLLING CONTRACT:
  // - m2_out_row / m2_out_col are GLOBAL output coordinates.
  // - Mode2 follows Mode1's scheduler model: layer-level compute scheduling
  //   plus free-token-driven runtime refill.
  // - m2_tile_col_base_g is retained only for interface compatibility/debug;
  //   local dataflow no longer converts output columns into a resident-tile
  //   local frame.
  // - The physical IFM Mode2 column slot is rolling:
  //     col_l = global_col % PC_MODE2.
  // - Mode1 is intentionally untouched.
  // --------------------------------------------------------------------------
  input  logic [15:0] m2_tile_col_base_g,

  input  logic        m2_start,
  input  logic        m2_pass_start_pulse,
  input  logic        m2_mac_en,
  input  logic        m2_ce_out_valid,
  input  logic [15:0] m2_out_row,  // GLOBAL
  input  logic [15:0] m2_out_col,  // GLOBAL
  input  logic [15:0] m2_f_group,

  output logic                        m2_dr_write_en,
  output logic [$clog2(K_MAX)-1:0]    m2_dr_write_row_idx,
  output logic [PC_MODE2*DATA_W-1:0]  m2_dr_write_data,

  // --------------------------------------------------------------------------
  // Backpressure / status to higher-level control
  // --------------------------------------------------------------------------
  output logic hold_compute,
  output logic local_busy,
  output logic local_done,
  output logic local_error,

  // Optional visibility
  output logic m1_local_busy,
  output logic m2_local_busy,

  // Optional same-mode free-token visibility for mode 2.
  // MODE2 FOLLOW-MODE1 CONTRACT:
  // - This token must mean the IFM entry/word is truly free for refill, not
  //   merely that the entry was captured into data_register_mode2.
  // - Therefore it is emitted after the corresponding output pixel has been
  //   compute-consumed for the last PF/f-group, and it requests the next
  //   global column that can reuse the same local col_l slot.
  // - Mode1 is intentionally untouched; this token is Mode2-only.
  input  logic        m2_free_ready,
  output logic        m2_free_valid,
  output logic [15:0] m2_free_row_g,
  output logic [15:0] m2_free_col_g,
  output logic [15:0] m2_free_col_l,
  output logic [15:0] m2_free_cgrp_g
);

  // --------------------------------------------------------------------------
  // addr_gen_ifm_m1 wires
  // --------------------------------------------------------------------------
  logic                     m1_ifm_rd_en_s;
  logic [$clog2(C_MAX)-1:0] m1_ifm_rd_bank_base_s;
  logic [$clog2(H_MAX)-1:0] m1_ifm_rd_row_idx_s;
  logic [$clog2(W_MAX)-1:0] m1_ifm_rd_col_idx_s;

  logic                     m1_dr_write_en_s;
  logic [$clog2(K_MAX)-1:0] m1_dr_write_row_idx_s;
  logic [15:0]              m1_dr_write_x_base_s;
  logic [PV_MAX*DATA_W-1:0] m1_dr_write_data_s;

  logic                     m1_busy_s;
  logic                     m1_done_s;
  logic                     m1_error_s;

  // --------------------------------------------------------------------------
  // addr_gen_ifm_m2 wires
  // --------------------------------------------------------------------------
  logic                     m2_ifm_rd_en_s;
  logic [$clog2(C_MAX)-1:0] m2_ifm_rd_bank_base_s;
  logic [$clog2(H_MAX)-1:0] m2_ifm_rd_row_idx_s;
  logic [$clog2(W_MAX)-1:0] m2_ifm_rd_col_idx_s;
  logic [$clog2(W_MAX)-1:0] m2_ifm_rd_col_g_s;
  logic                     m2_miss_refill_valid_s;
  logic [15:0]              m2_miss_refill_row_g_s;
  logic [15:0]              m2_miss_refill_col_g_s;
  logic [15:0]              m2_miss_refill_col_l_s;
  logic [15:0]              m2_miss_refill_cgrp_g_s;

  logic                     m2_dr_write_en_s;
  logic [$clog2(K_MAX)-1:0] m2_dr_write_row_idx_s;
  logic [PC_MODE2*DATA_W-1:0] m2_dr_write_data_s;

  logic                     m2_busy_s;
  logic                     m2_done_s;
  logic                     m2_error_s;

  // Explicit internal aliases to make GLOBAL coordinate intent obvious.
  logic [15:0] m2_out_row_g_s;
  logic [15:0] m2_out_col_g_s;
  logic [15:0] m2_tile_col_base_g_s;
  logic        m2_issue_block_in_tile_s;

  // --------------------------------------------------------------------------
  // Mode-2 addr_gen forwarding.
  //
  // Legacy tile-boundary hold / synthetic prefetch state has been removed from
  // synthesis. Under the final rolling Mode2 contract, control_unit_top owns
  // tile/refill scheduling and this local manager only forwards CE events to
  // addr_gen_ifm_m2.
  // --------------------------------------------------------------------------
  logic        m2_start_to_addrgen_s;
  logic        m2_pass_start_to_addrgen_s;
  logic        m2_mac_en_to_addrgen_s;
  logic        m2_out_valid_to_addrgen_s;


  // --------------------------------------------------------------------------
  // Mode2 follow-Mode1 free-slot bookkeeping
  //
  // This block mirrors the meaning of Mode1's free-slot notification, but
  // with the Mode2 refill unit.  It must NOT declare a slot free merely
  // because IFM data was captured into data_register_mode2.  Instead, it
  // emits a free event after compute has consumed the entry for the last
  // PF/f-group.
  //
  // Mode2 refill unit:
  //   row_g  : global IFM row of the slot being reused
  //   col_g  : next global IFM column to refill into the same physical col_l
  //   col_l  : local IFM bank/column slot inside the resident WT=PC tile
  //   cgrp_g : channel group stored at IFM col_idx
  //
  // For the current K=1 Mode2 regression, output pixel {row,col} consumes
  // exactly IFM entry {row,col}.  Therefore the slot for col_l can be reused
  // by global column col+PC after the last f-group completes.
  // --------------------------------------------------------------------------
  logic [15:0] m2_num_fgroup_s;
  logic [15:0] m2_num_cgrp_s;
  logic        m2_last_fgroup_s;
  logic        m2_pixel_consumed_last_fgroup_s;

  // Local-view signals passed into addr_gen_ifm_m2.
  // addr_gen_ifm_m2 sees the same global output column frame under the rolling
  // Mode2 contract; the physical IFM slot is derived downstream as col % PC.
  logic [15:0] m2_out_col_l_s;
  logic [15:0] m2_ag_wout_cur_s;
  logic [15:0] m2_ag_tile_col_base_s;


  // Mode2 follow-Mode1 free-slot emitter.
  // A free event is generated from compute-consume completion, not from the
  // IFM return path.  The stored event describes the NEXT global IFM column
  // that may reuse the just-consumed local col_l slot.
  logic        m2_free_emit_active_q;
  logic [15:0] m2_free_emit_row_g_q;
  logic [15:0] m2_free_emit_col_g_q;
  logic [15:0] m2_free_emit_col_l_q;
  logic [15:0] m2_free_emit_cgrp_q;
  logic [15:0] m2_free_emit_num_cgrp_q;
  logic        m2_miss_pending_q;
  logic [15:0] m2_miss_row_g_q;
  logic [15:0] m2_miss_col_g_q;
  logic [15:0] m2_miss_col_l_q;
  logic [15:0] m2_miss_cgrp_g_q;


  logic        m2_free_refill_needed_s;
  logic [15:0] m2_free_refill_col_g_s;
  logic [15:0] m2_free_consumed_col_l_s;
  logic        m2_free_consumed_col_in_tile_s;
  logic        m2_free_emit_hold_s;

  assign m2_out_row_g_s       = m2_out_row;
  assign m2_out_col_g_s       = m2_out_col;
  assign m2_tile_col_base_g_s = m2_tile_col_base_g;

  // --------------------------------------------------------------------------
  // Mode2 follow-Mode1 free-slot metadata
  //
  // FINAL ROLLING CONTRACT:
  // - Mode2 follows Mode1's scheduler model: layer-level compute scheduling and
  //   free-token-driven runtime refill.
  // - This manager therefore keeps GLOBAL output coordinates for addr_gen_ifm_m2
  //   and does not translate them into a resident-tile-local frame.
  // - The physical IFM Mode2 slot is derived by the rolling mapping
  //     col_l = global_col % PC_MODE2
  //   while all bounds are expressed in the consumer layer's IFM geometry
  //     H_in / W_in / C_in.
  // - Mode1 logic is intentionally untouched.
  // --------------------------------------------------------------------------
  always_comb begin
    if (PF_MODE2 != 0)
      m2_num_fgroup_s = (cur_cfg.f_out + PF_MODE2 - 1) / PF_MODE2;
    else
      m2_num_fgroup_s = 16'd0;

    if (PC_MODE2 != 0)
      m2_num_cgrp_s = (cur_cfg.c_in + PC_MODE2 - 1) / PC_MODE2;
    else
      m2_num_cgrp_s = 16'd0;

    // Rolling/full-layer view for Mode2.  No resident tile-local conversion is
    // performed here; addr_gen_ifm_m2 receives global output columns.
    m2_out_col_l_s        = m2_out_col_g_s;
    m2_ag_wout_cur_s      = cur_cfg.w_out;
    m2_ag_tile_col_base_s = 16'd0;

    // Free-token timing: emit only after the last PF/f-group for the output
    // pixel has been consumed, matching Mode1's lifecycle.
    m2_last_fgroup_s = (m2_num_fgroup_s == 0) ? 1'b1 :
                       (m2_f_group == (m2_num_fgroup_s - 1));

    m2_pixel_consumed_last_fgroup_s = (cur_mode == MODE2) &&
                                      m2_ce_out_valid &&
                                      m2_last_fgroup_s;

    m2_free_refill_col_g_s = m2_out_col_g_s + PC_MODE2;
    if (PC_MODE2 != 0) begin
      m2_free_consumed_col_l_s = m2_out_col_g_s % PC_MODE2;
      m2_free_consumed_col_in_tile_s = 1'b1;
    end
    else begin
      m2_free_consumed_col_l_s = 16'd0;
      m2_free_consumed_col_in_tile_s = 1'b0;
    end

    // Refill bounds use consumer IFM geometry, not output geometry and not a
    // resident tile window.
    m2_free_refill_needed_s = m2_pixel_consumed_last_fgroup_s &&
                              m2_free_consumed_col_in_tile_s &&
                              (m2_num_cgrp_s != 16'd0) &&
                              (m2_out_row_g_s < cur_cfg.h_in) &&
                              (m2_free_refill_col_g_s < cur_cfg.w_in);
  end

  // --------------------------------------------------------------------------
  // Mode-2 addr_gen forwarding
  // --------------------------------------------------------------------------
  always_comb begin
    m2_start_to_addrgen_s      = m2_start;
    m2_pass_start_to_addrgen_s = m2_pass_start_pulse;
    m2_mac_en_to_addrgen_s     = m2_mac_en;
    m2_out_valid_to_addrgen_s  = m2_ce_out_valid;
  end

  // --------------------------------------------------------------------------
  // Mode-1 local IFM feeder
  // --------------------------------------------------------------------------
  addr_gen_ifm_m1 #(
    .DATA_W (DATA_W),
    .PV_MAX (PV_MAX),
    .C_MAX  (C_MAX),
    .W_MAX  (W_MAX),
    .H_MAX  (H_MAX),
    .K_MAX  (K_MAX)
  ) u_addr_gen_ifm_m1 (
    .clk               (clk),
    .rst_n             (rst_n),

    .K_cur             (cur_cfg.k),
    .C_cur             (cur_cfg.c_in),
    .H_cur             (cur_cfg.h_in),
    .W_cur             (cur_cfg.w_in),
    .Pv_cur            (cur_cfg.pv_m1),

    .pass_start_pulse  (m1_pass_start_pulse),
    .chan_done_pulse   (m1_chan_done_pulse),
    .c_iter            (m1_c_iter),
    .out_row           (m1_out_row),
    .out_col           (m1_out_col),

    .ifm_rd_en         (m1_ifm_rd_en_s),
    .ifm_rd_bank_base  (m1_ifm_rd_bank_base_s),
    .ifm_rd_row_idx    (m1_ifm_rd_row_idx_s),
    .ifm_rd_col_idx    (m1_ifm_rd_col_idx_s),
    .ifm_rd_valid      (ifm_rd_valid),
    .ifm_rd_data       (ifm_rd_data),

    .dr_write_en       (m1_dr_write_en_s),
    .dr_write_row_idx  (m1_dr_write_row_idx_s),
    .dr_write_x_base   (m1_dr_write_x_base_s),
    .dr_write_data     (m1_dr_write_data_s),

    .busy              (m1_busy_s),
    .done              (m1_done_s),
    .error             (m1_error_s),

    .dbg_target_channel(),
    .dbg_words_per_row (),
    .dbg_issue_row     (),
    .dbg_issue_col     (),
    .dbg_waiting_for_return()
  );

  // --------------------------------------------------------------------------
  // Mode-2 local IFM feeder
  //
  // By contract, m2_out_row_g_s / m2_out_col_g_s are GLOBAL coordinates.
  // Mode2 now follows Mode1's scheduler model, so addr_gen_ifm_m2 is fed the
  // full-layer/global output coordinate frame.  The rolling physical IFM slot
  // is selected downstream as global_col % PC_MODE2.
  // --------------------------------------------------------------------------
  addr_gen_ifm_m2 #(
    .DATA_W (DATA_W),
    .PV_MAX (PV_MAX),
    .PC     (PC_MODE2),
    .PF     (PF_MODE2),
    .C_MAX  (C_MAX),
    .W_MAX  (W_MAX),
    .H_MAX  (H_MAX),
    .K_MAX  (K_MAX)
  ) u_addr_gen_ifm_m2 (
    .clk               (clk),
    .rst_n             (rst_n),

    .K_cur             (cur_cfg.k),
    .C_cur             (cur_cfg.c_in),
    .F_cur             (cur_cfg.f_out),
    .H_in              (cur_cfg.h_in),
    .W_in              (cur_cfg.w_in),
    .Hout_cur          (cur_cfg.h_out),
    .Wout_cur          (m2_ag_wout_cur_s),
    .tile_col_base_g   (m2_ag_tile_col_base_s),

    .start             (m2_start_to_addrgen_s),
    .pass_start_pulse  (m2_pass_start_to_addrgen_s),
    .mac_en            (m2_mac_en_to_addrgen_s),
    .out_valid         (m2_out_valid_to_addrgen_s),
    .out_row           (m2_out_row_g_s),
    .out_col           (m2_out_col_l_s),
    .f_group           (m2_f_group),

    .ifm_rd_en         (m2_ifm_rd_en_s),
    .ifm_rd_bank_base  (m2_ifm_rd_bank_base_s),
    .ifm_rd_row_idx    (m2_ifm_rd_row_idx_s),
    .ifm_rd_col_idx    (m2_ifm_rd_col_idx_s),
    .ifm_rd_col_g      (m2_ifm_rd_col_g_s),
    .ifm_rd_valid      (ifm_rd_valid),
    .ifm_rd_data       (ifm_rd_data),

    .m2_miss_refill_valid  (m2_miss_refill_valid_s),
    .m2_miss_refill_row_g  (m2_miss_refill_row_g_s),
    .m2_miss_refill_col_g  (m2_miss_refill_col_g_s),
    .m2_miss_refill_col_l  (m2_miss_refill_col_l_s),
    .m2_miss_refill_cgrp_g (m2_miss_refill_cgrp_g_s),

    .dr_write_en       (m2_dr_write_en_s),
    .dr_write_row_idx  (m2_dr_write_row_idx_s),
    .dr_write_data     (m2_dr_write_data_s),

    .busy              (m2_busy_s),
    .done              (m2_done_s),
    .error             (m2_error_s),

    .dbg_num_fgroup    (),
    .dbg_num_cgroup    (),
    .dbg_block_row     (),
    .dbg_block_col     (),
    .dbg_issue_cgroup  (),
    .dbg_issue_ky      (),
    .dbg_issue_kx      (),
    .dbg_waiting_for_return()
  );

  // --------------------------------------------------------------------------
  // Shared IFM read-port mux
  // Only the selected mode drives the physical ifm_buffer read port.
  // --------------------------------------------------------------------------
  always_comb begin
    ifm_rd_en        = 1'b0;
    ifm_rd_bank_base = '0;
    ifm_rd_row_idx   = '0;
    ifm_rd_col_idx   = '0;
    ifm_rd_col_g     = '0;

    case (cur_mode)
      MODE1: begin
        ifm_rd_en        = m1_ifm_rd_en_s;
        ifm_rd_bank_base = m1_ifm_rd_bank_base_s;
        ifm_rd_row_idx   = m1_ifm_rd_row_idx_s;
        ifm_rd_col_idx   = m1_ifm_rd_col_idx_s;
      end

      MODE2: begin
        ifm_rd_en        = m2_ifm_rd_en_s;
        ifm_rd_bank_base = m2_ifm_rd_bank_base_s;
        ifm_rd_row_idx   = m2_ifm_rd_row_idx_s;
        ifm_rd_col_idx   = m2_ifm_rd_col_idx_s;
        ifm_rd_col_g     = m2_ifm_rd_col_g_s;
      end

      default: begin end
    endcase
  end

  // --------------------------------------------------------------------------
  // data_register write-port mux / gating
  // Only the selected mode is allowed to write its data_register.
  // --------------------------------------------------------------------------
  always_comb begin
    m1_dr_write_en      = 1'b0;
    m1_dr_write_row_idx = '0;
    m1_dr_write_x_base  = '0;
    m1_dr_write_data    = '0;

    m2_dr_write_en      = 1'b0;
    m2_dr_write_row_idx = '0;
    m2_dr_write_data    = '0;

    case (cur_mode)
      MODE1: begin
        m1_dr_write_en      = m1_dr_write_en_s;
        m1_dr_write_row_idx = m1_dr_write_row_idx_s;
        m1_dr_write_x_base  = m1_dr_write_x_base_s;
        m1_dr_write_data    = m1_dr_write_data_s;
      end

      MODE2: begin
        m2_dr_write_en      = m2_dr_write_en_s;
        m2_dr_write_row_idx = m2_dr_write_row_idx_s;
        m2_dr_write_data    = m2_dr_write_data_s;
      end

      default: begin end
    endcase
  end

  // --------------------------------------------------------------------------
  // Mode-2 free-token bookkeeping state
  // --------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      m2_free_emit_active_q     <= 1'b0;
      m2_free_emit_row_g_q      <= 16'd0;
      m2_free_emit_col_g_q      <= 16'd0;
      m2_free_emit_col_l_q      <= 16'd0;
      m2_free_emit_cgrp_q       <= 16'd0;
      m2_free_emit_num_cgrp_q   <= 16'd0;
      m2_miss_pending_q         <= 1'b0;
      m2_miss_row_g_q           <= 16'd0;
      m2_miss_col_g_q           <= 16'd0;
      m2_miss_col_l_q           <= 16'd0;
      m2_miss_cgrp_g_q          <= 16'd0;
    end
    else begin
      // Mode2 follow-Mode1 free-slot emitter plus demand refill fallback.
      // Normal free events are prefetch tokens.  If a tagged IFM read misses,
      // addr_gen_ifm_m2 emits a demand token naming the exact missing entry;
      // that token has priority and is held until it can be exposed.
      if (cur_mode != MODE2) begin
        m2_free_emit_active_q   <= 1'b0;
        m2_free_emit_cgrp_q     <= 16'd0;
        m2_miss_pending_q       <= 1'b0;
      end
      else begin
        if (m2_miss_refill_valid_s && !m2_miss_pending_q && !m2_free_emit_active_q) begin
          m2_free_emit_active_q   <= 1'b1;
          m2_free_emit_row_g_q    <= m2_miss_refill_row_g_s;
          m2_free_emit_col_g_q    <= m2_miss_refill_col_g_s;
          m2_free_emit_col_l_q    <= m2_miss_refill_col_l_s;
          m2_free_emit_cgrp_q     <= m2_miss_refill_cgrp_g_s;
          m2_free_emit_num_cgrp_q <= m2_miss_refill_cgrp_g_s + 16'd1;
        end
        else if (m2_miss_refill_valid_s && !m2_miss_pending_q) begin
          m2_miss_pending_q <= 1'b1;
          m2_miss_row_g_q   <= m2_miss_refill_row_g_s;
          m2_miss_col_g_q   <= m2_miss_refill_col_g_s;
          m2_miss_col_l_q   <= m2_miss_refill_col_l_s;
          m2_miss_cgrp_g_q  <= m2_miss_refill_cgrp_g_s;
        end

        if (m2_free_emit_active_q) begin
          // Hold the token until control_unit_top confirms it has been queued.
          if (m2_free_ready) begin
            if ((m2_free_emit_cgrp_q + 16'd1) < m2_free_emit_num_cgrp_q) begin
              m2_free_emit_cgrp_q <= m2_free_emit_cgrp_q + 16'd1;
            end
            else begin
              m2_free_emit_active_q <= 1'b0;
              m2_free_emit_cgrp_q   <= 16'd0;
            end
          end
        end
        else if (m2_miss_pending_q) begin
          m2_free_emit_active_q   <= 1'b1;
          m2_free_emit_row_g_q    <= m2_miss_row_g_q;
          m2_free_emit_col_g_q    <= m2_miss_col_g_q;
          m2_free_emit_col_l_q    <= m2_miss_col_l_q;
          m2_free_emit_cgrp_q     <= m2_miss_cgrp_g_q;
          m2_free_emit_num_cgrp_q <= m2_miss_cgrp_g_q + 16'd1;
          m2_miss_pending_q       <= 1'b0;
        end
        else if (m2_free_refill_needed_s) begin
          m2_free_emit_active_q   <= 1'b1;
          m2_free_emit_row_g_q    <= m2_out_row_g_s;
          m2_free_emit_col_g_q    <= m2_free_refill_col_g_s;
          m2_free_emit_col_l_q    <= m2_free_consumed_col_l_s;
          m2_free_emit_cgrp_q     <= 16'd0;
          m2_free_emit_num_cgrp_q <= m2_num_cgrp_s;
        end
      end
    end
  end

  always_comb begin
    m2_free_valid  = 1'b0;
    m2_free_row_g  = m2_free_emit_row_g_q;
    m2_free_col_g  = m2_free_emit_col_g_q;
    m2_free_col_l  = m2_free_emit_col_l_q;
    m2_free_cgrp_g = m2_free_emit_cgrp_q;

    if (cur_mode == MODE2)
      m2_free_valid = m2_free_emit_active_q;
  end

  // --------------------------------------------------------------------------
  // Status / backpressure
  //
  // Mode1 behavior is unchanged.  Mode2 follows the same lifecycle principle:
  // hold only while a free-token event is being serialized so control_unit_top
  // can observe every cgrp refill request.  There is no tile-boundary hold.
  // --------------------------------------------------------------------------
  // Mode1 behavior is unchanged.  Mode2 asserts a narrow hold only while
  // serializing cgrp free events so no compute-consumed slot is dropped.
  assign m2_free_emit_hold_s = m2_free_emit_active_q || m2_free_refill_needed_s || m2_miss_pending_q || m2_miss_refill_valid_s;

  assign hold_compute = (cur_mode == MODE1) ? m1_busy_s :
                        (cur_mode == MODE2) ? m2_free_emit_hold_s : 1'b0;

  assign local_busy  = (cur_mode == MODE1) ? m1_busy_s  :
                       (cur_mode == MODE2) ? m2_busy_s  : 1'b0;

  assign local_done  = (cur_mode == MODE1) ? m1_done_s  :
                       (cur_mode == MODE2) ? m2_done_s  : 1'b0;

  assign local_error = (cur_mode == MODE1) ? m1_error_s :
                       (cur_mode == MODE2) ? m2_error_s : 1'b0;

  assign m1_local_busy = m1_busy_s;
  assign m2_local_busy = m2_busy_s;

`ifndef SYNTHESIS
  // Keep the fixed mode-2 parallelism assumption visible in simulation.
  // Also make the GLOBAL-column contract explicit at the manager boundary.
  always_ff @(posedge clk) begin
    if (rst_n && (cur_mode == MODE2) && m2_start) begin
      if (cur_cfg.pc_m2 != PC_MODE2) begin
        $error("local_dataflow_manager: cur_cfg.pc_m2 (%0d) != PC_MODE2 parameter (%0d).",
               cur_cfg.pc_m2, PC_MODE2);
      end
      if (cur_cfg.pf_m2 != PF_MODE2) begin
        $error("local_dataflow_manager: cur_cfg.pf_m2 (%0d) != PF_MODE2 parameter (%0d).",
               cur_cfg.pf_m2, PF_MODE2);
      end
      if ((cur_cfg.w_out != 0) && (m2_out_col_g_s >= cur_cfg.w_out)) begin
        $error("local_dataflow_manager: mode-2 output col (%0d) is outside Wout_cur (%0d).",
               m2_out_col_g_s, cur_cfg.w_out);
      end
      if ((cur_cfg.h_out != 0) && (m2_out_row_g_s >= cur_cfg.h_out)) begin
        $error("local_dataflow_manager: mode-2 output row (%0d) is outside Hout_cur (%0d).",
               m2_out_row_g_s, cur_cfg.h_out);
      end
      if ((PC_MODE2 != 0) && (m2_free_consumed_col_l_s != (m2_out_col_g_s % PC_MODE2))) begin
        $error("local_dataflow_manager: rolling col_l mismatch for out_col=%0d PC=%0d col_l=%0d.",
               m2_out_col_g_s, PC_MODE2, m2_free_consumed_col_l_s);
      end
    end
  end
`endif

`ifndef SYNTHESIS

always_ff @(posedge clk or negedge rst_n) begin : DBG_LDM_IFM_RD_MUX_MON
  if (!rst_n) begin
  end else begin
    if (ifm_rd_en || ifm_rd_valid) begin
      $display("DBG_LDM_IFM_RD_MUX t=%0t cur_mode=%0b ifm_rd_en=%0b bank_base=%0d row=%0d col_idx=%0d col_g=%0d ifm_rd_valid=%0b data0=%0d data1=%0d data2=%0d data3=%0d",
        $time,
        cur_mode,
        ifm_rd_en,
        ifm_rd_bank_base,
        ifm_rd_row_idx,
        ifm_rd_col_idx,
        ifm_rd_col_g,
        ifm_rd_valid,
        $signed(ifm_rd_data[0*DATA_W +: DATA_W]),
        $signed(ifm_rd_data[1*DATA_W +: DATA_W]),
        $signed(ifm_rd_data[2*DATA_W +: DATA_W]),
        $signed(ifm_rd_data[3*DATA_W +: DATA_W])
      );
    end
  end
end

`endif

endmodule
