module local_dataflow_manager
  import cnn_layer_desc_pkg::*;
#(
  parameter int DATA_W   = 8,
  parameter int PTOTAL   = 16,
  parameter int PF_MAX   = 8,
  parameter int PV_MAX   = 8,
  parameter int PC_MODE2 = 8,
  parameter int PF_MODE2 = 4,
  parameter int C_MAX    = 64,
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
  input  logic                     ifm_rd_valid,
  input  logic [PV_MAX*DATA_W-1:0] ifm_rd_data,

  // --------------------------------------------------------------------------
  // Inputs from mode1_compute_top
  // --------------------------------------------------------------------------
  input  logic        m1_pass_start_pulse,
  input  logic        m1_chan_done_pulse,
  input  logic [15:0] m1_c_iter,

  output logic                     m1_dr_write_en,
  output logic [$clog2(K_MAX)-1:0] m1_dr_write_row_idx,
  output logic [15:0]              m1_dr_write_x_base,
  output logic [PV_MAX*DATA_W-1:0] m1_dr_write_data,

  // --------------------------------------------------------------------------
  // Inputs from mode2_compute_top
  //
  // UPDATED CONTRACT:
  // - m2_out_row / m2_out_col are GLOBAL output coordinates.
  // - local tile coordinate must NOT be inferred here by policy guesswork.
  // - addr_gen_ifm_m2 is the single place that derives local tile column
  //   from:
  //       out_col_global + kx
  //       tile_base_global (currently inferred there from PC)
  //
  // This keeps local_dataflow_manager as a clean mux/handshake wrapper
  // between compute and the mode-specific IFM generators.
  // --------------------------------------------------------------------------
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
  // This does NOT change existing behavior of the manager itself; it only
  // exposes the IFM tuple that has actually been captured into
  // data_register_mode2 so higher-level control can use it as a free token.
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

  logic                     m2_dr_write_en_s;
  logic [$clog2(K_MAX)-1:0] m2_dr_write_row_idx_s;
  logic [PC_MODE2*DATA_W-1:0] m2_dr_write_data_s;

  logic                     m2_busy_s;
  logic                     m2_done_s;
  logic                     m2_error_s;

  // Explicit internal aliases to make GLOBAL coordinate intent obvious.
  logic [15:0] m2_out_row_g_s;
  logic [15:0] m2_out_col_g_s;

  // --------------------------------------------------------------------------
  // Optional mode-2 free-token bookkeeping
  //
  // The current system contract needs a free token when one IFM tuple has
  // been successfully captured into data_register_mode2. We derive that token
  // here from the already-issued ifm_buffer read metadata, without changing
  // the existing read/write behavior.
  //
  // IMPORTANT:
  // - row_g is taken directly from the absolute ifm_buffer read row.
  // - col_l is the local tile column already used to index ifm_buffer.
  // - cgrp_g comes from bank_base / PC_MODE2.
  // - col_g is the GLOBAL base column of the resident PC-wide segment.
  //   This is the identifier that same_mode_refill_manager_m2 matches against
  //   ofm_buffer ready-token colbase_g. It is intentionally NOT the absolute
  //   input-pixel column; that per-word local position is carried separately
  //   by col_l.
  // --------------------------------------------------------------------------
  logic [15:0] m2_num_fgroup_s;
  logic        m2_last_col_s;
  logic        m2_last_row_s;
  logic        m2_last_fgroup_s;
  logic        m2_have_next_block_s;
  logic [15:0] m2_next_block_col_s;
  logic [15:0] m2_issue_block_col_g_s;
  logic [15:0] m2_issue_block_tile_base_g_s;
  logic [15:0] m2_issue_block_col_mod_s;

  logic        m2_meta_stream_active_q;
  logic [15:0] m2_meta_block_col_q;

  logic        m2_ret_meta_valid_q;
  logic [15:0] m2_ret_row_g_q;
  logic [15:0] m2_ret_col_l_q;
  logic [15:0] m2_ret_block_tile_base_g_q;
  logic [15:0] m2_ret_block_col_mod_q;
  logic [15:0] m2_ret_seg_base_g_s;
  logic [15:0] m2_ret_cgrp_g_q;

  assign m2_out_row_g_s = m2_out_row;
  assign m2_out_col_g_s = m2_out_col;

  // --------------------------------------------------------------------------
  // Optional mode-2 free-token reconstruction
  // --------------------------------------------------------------------------
  always_comb begin
    if (PF_MODE2 != 0)
      m2_num_fgroup_s = (cur_cfg.f_out + PF_MODE2 - 1) / PF_MODE2;
    else
      m2_num_fgroup_s = 16'd0;

    m2_last_col_s    = (cur_cfg.w_out == 0) ? 1'b1 : (m2_out_col_g_s == (cur_cfg.w_out - 1));
    m2_last_row_s    = (cur_cfg.h_out == 0) ? 1'b1 : (m2_out_row_g_s == (cur_cfg.h_out - 1));
    m2_last_fgroup_s = (m2_num_fgroup_s == 0) ? 1'b1 : (m2_f_group == (m2_num_fgroup_s - 1));
    m2_have_next_block_s = !(m2_last_col_s && m2_last_row_s && m2_last_fgroup_s);

    if (!m2_last_col_s)
      m2_next_block_col_s = m2_out_col_g_s + 16'd1;
    else
      m2_next_block_col_s = 16'd0;

    // Match the block-column used by addr_gen_ifm_m2 for the read issued
    // in the current cycle:
    // - first issue after start uses block_col = 0
    // - first issue after out_valid uses next_block_col
    // - all other issues stay in the current block
    if (m2_start) begin
      m2_issue_block_col_g_s = 16'd0;
    end
    else if (m2_meta_stream_active_q && m2_ce_out_valid && m2_have_next_block_s) begin
      m2_issue_block_col_g_s = m2_next_block_col_s;
    end
    else begin
      m2_issue_block_col_g_s = m2_meta_block_col_q;
    end

    if (PC_MODE2 != 0) begin
      m2_issue_block_tile_base_g_s = (m2_issue_block_col_g_s / PC_MODE2) * PC_MODE2;
      m2_issue_block_col_mod_s     = m2_issue_block_col_g_s - m2_issue_block_tile_base_g_s;
    end
    else begin
      m2_issue_block_tile_base_g_s = 16'd0;
      m2_issue_block_col_mod_s     = 16'd0;
    end
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
    .W_cur             (cur_cfg.w_in),
    .Pv_cur            (cur_cfg.pv_m1),

    .pass_start_pulse  (m1_pass_start_pulse),
    .chan_done_pulse   (m1_chan_done_pulse),
    .c_iter            (m1_c_iter),

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
  // addr_gen_ifm_m2 is responsible for deriving LOCAL tile coordinate.
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
    .Wout_cur          (cur_cfg.w_out),

    .start             (m2_start),
    .pass_start_pulse  (m2_pass_start_pulse),
    .mac_en            (m2_mac_en),
    .out_valid         (m2_ce_out_valid),
    .out_row           (m2_out_row_g_s),
    .out_col           (m2_out_col_g_s),
    .f_group           (m2_f_group),

    .ifm_rd_en         (m2_ifm_rd_en_s),
    .ifm_rd_bank_base  (m2_ifm_rd_bank_base_s),
    .ifm_rd_row_idx    (m2_ifm_rd_row_idx_s),
    .ifm_rd_col_idx    (m2_ifm_rd_col_idx_s),
    .ifm_rd_valid      (ifm_rd_valid),
    .ifm_rd_data       (ifm_rd_data),

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
      m2_meta_stream_active_q <= 1'b0;
      m2_meta_block_col_q     <= 16'd0;
      m2_ret_meta_valid_q     <= 1'b0;
      m2_ret_row_g_q          <= 16'd0;
      m2_ret_col_l_q          <= 16'd0;
      m2_ret_block_tile_base_g_q <= 16'd0;
      m2_ret_block_col_mod_q    <= 16'd0;
      m2_ret_cgrp_g_q           <= 16'd0;
    end
    else begin
      // Track the GLOBAL block-column currently being prefetched by
      // addr_gen_ifm_m2 so that the returned local column can be lifted back
      // to a GLOBAL IFM column.
      if (m2_start) begin
        m2_meta_stream_active_q <= 1'b1;
        m2_meta_block_col_q     <= 16'd0;
      end
      else if (m2_meta_stream_active_q && m2_ce_out_valid) begin
        if (m2_have_next_block_s)
          m2_meta_block_col_q <= m2_next_block_col_s;
        else
          m2_meta_stream_active_q <= 1'b0;
      end

      // One-cycle return metadata aligned to the current addr_gen_ifm_m2
      // contract and to m2_dr_write_en_s.
      m2_ret_meta_valid_q <= m2_ifm_rd_en_s;
      if (m2_ifm_rd_en_s) begin
        m2_ret_row_g_q            <= m2_ifm_rd_row_idx_s;
        m2_ret_col_l_q            <= m2_ifm_rd_col_idx_s;
        m2_ret_block_tile_base_g_q<= m2_issue_block_tile_base_g_s;
        m2_ret_block_col_mod_q    <= m2_issue_block_col_mod_s;
        if (PC_MODE2 != 0)
          m2_ret_cgrp_g_q         <= m2_ifm_rd_bank_base_s / PC_MODE2;
        else
          m2_ret_cgrp_g_q         <= 16'd0;
      end
    end
  end

  always_comb begin
    m2_ret_seg_base_g_s = m2_ret_block_tile_base_g_q;
    if ((PC_MODE2 != 0) && (m2_ret_col_l_q < m2_ret_block_col_mod_q))
      m2_ret_seg_base_g_s = m2_ret_block_tile_base_g_q + PC_MODE2;

    m2_free_valid  = 1'b0;
    m2_free_row_g  = m2_ret_row_g_q;
    m2_free_col_g  = m2_ret_seg_base_g_s;
    m2_free_col_l  = m2_ret_col_l_q;
    m2_free_cgrp_g = m2_ret_cgrp_g_q;

    if (cur_mode == MODE2)
      m2_free_valid = m2_dr_write_en_s && m2_ret_meta_valid_q;
  end

  // --------------------------------------------------------------------------
  // Status / backpressure
  //
  // In the current design, only mode 1 requires coarse hold of the compute
  // path while addr_gen_ifm_m1 is busy refilling data_register for the next
  // channel. mode 2 prefetches IFM tuples in lock-step with controller pulses,
  // so we do not assert a global hold from m2_busy here.
  // --------------------------------------------------------------------------
  assign hold_compute = (cur_mode == MODE1) ? m1_busy_s : 1'b0;

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
      // Free-token segment-base reconstruction uses the current fixed mode-2
      // contract where one Kx sweep can cross at most one PC-wide boundary.
      if (cur_cfg.k > PC_MODE2) begin
        $error("local_dataflow_manager: cur_cfg.k (%0d) > PC_MODE2 (%0d); mode-2 free-token segment-base reconstruction is ambiguous.",
               cur_cfg.k, PC_MODE2);
      end
    end
  end
`endif

endmodule
