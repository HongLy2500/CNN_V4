module ce_mode2_top #(
  parameter int DATA_W    = 8,
  parameter int PSUM_W    = 32,
  parameter int K_MAX     = 3,
  parameter int PC        = 16,
  parameter int PF        = 8,
  parameter int HOUT_MAX  = 32,
  parameter int WOUT_MAX  = 32,
  // Physical word width shared with weight_buffer.WORD_LANES.
  // Mode 2 consumes the low PF*PC logical lanes from this physical word.
  parameter int WB_LANES  = 32,
  parameter int WB_ADDR_W = 12
)(
  input  logic clk,
  input  logic rst_n,

  input  logic start,
  input  logic step_en,

  input  logic [3:0] K_cur,
  input  logic [9:0] C_cur,
  input  logic [9:0] F_cur,
  input  logic [15:0] Hout_cur,
  input  logic [15:0] Wout_cur,

  // Mode-2 resident-tile window. One start computes one horizontal tile.
  // out_col exported by the controller remains GLOBAL:
  //   out_col = tile_col_base_g + local_col
  input  logic [15:0] tile_col_base_g,
  input  logic [15:0] tile_col_count,

  input  logic                     dr_write_en,
  input  logic [$clog2(K_MAX)-1:0] dr_write_row_idx,
  input  logic [PC*DATA_W-1:0]     dr_write_data,

  input  logic                     weight_bank_sel,
  input  logic                     weight_bank_ready,
  output logic                     wb_rd_en,
  output logic                     wb_rd_buf_sel,
  output logic [WB_ADDR_W-1:0]     wb_rd_addr,
  input  logic [WB_LANES*DATA_W-1:0] wb_rd_data,
  input  logic                     wb_rd_valid,

  output logic [15:0]              out_row,
  output logic [15:0]              out_col,
  output logic [15:0]              f_group,
  output logic [15:0]              c_group,
  output logic [$clog2(K_MAX)-1:0] ky,
  output logic [$clog2(K_MAX)-1:0] kx,

  output logic                     mac_en,
  output logic                     clear_psum,
  output logic                     out_valid,
  output logic                     pass_start_pulse,
  output logic                     group_start_pulse,
  output logic                     row_done_pulse,
  output logic [$clog2(K_MAX)-1:0] row_done_ky,
  output logic                     c_group_done_pulse,
  output logic                     pixel_done_pulse,
  output logic                     f_group_done_pulse,
  output logic                     done,
  output logic                     busy,

  output logic [PC*DATA_W-1:0]     data_out_logic,
  output logic [PF*PC*DATA_W-1:0]  weight_out,
  output logic [PF*PSUM_W-1:0]     mac_data_out,
  output logic                     mac_data_out_valid,
  output logic                     mac_group_start,
  output logic [15:0]              mac_f_base
);

  logic [15:0] f_base_cur;
  logic [15:0] c_base_cur;
  logic        weight_write_en;
  logic [PF*PC*DATA_W-1:0] weight_write_data;

  // Raw outputs from the internal registers. These may contain X/rubbish
  // on inactive lanes in partial C/F groups, so they are masked before MAC.
  logic [PC*DATA_W-1:0]       data_out_raw;
  logic [PF*PC*DATA_W-1:0]    weight_out_raw;

  // Consume-qualified operand readiness for Mode 2.
  //
  // Mode 2 now follows the same essential contract as Mode 1:
  //   - exactly one current IFM tuple is held;
  //   - exactly one current weight tuple is held;
  //   - ce_controller_mode2 may advance only when both current tuples
  //     are valid and the MAC really consumes them.
  //
  // This intentionally removes the previous multi-entry data skid FIFO,
  // because the FIFO allowed the controller metadata (f_group/c_group/ky/kx)
  // to advance ahead of the data tuple at Mode2 same-mode refill boundaries.
  logic [PC*DATA_W-1:0] data_tuple_q;
  logic [PC*DATA_W-1:0] data_tuple_raw;
  logic                 data_valid_q;
  logic                 data_tuple_valid_q;
  logic                 data_accept_s;
  logic                 data_overflow_s;

  logic wgt_tuple_ready_q;
  logic tuple_ready_s;
  logic ctrl_step_en;

  // One outstanding Mode-2 weight request at a time.  This prevents the
  // weight register from being overwritten by a prefetched bundle before
  // the current registered bundle is consumed by the MAC.
  logic weight_req_inflight_q;
  logic weight_bank_ready_m2_s;

  // Bypass/availability signals for one-cycle consume-on-return.
  //
  // Existing Mode2 path held exactly one IFM tuple and one weight tuple.
  // That was functionally safe, but produced a strict load/consume alternation:
  //   return data+weight -> next cycle consume -> next cycle return ...
  // These signals let the controller consume a tuple in the same cycle it
  // arrives when the active slot is empty, while still retaining the old
  // Mode1-like one-current-tuple contract when an active tuple is already held.
  logic data_available_s;
  logic wgt_available_s;
  logic [PF*PC*DATA_W-1:0] weight_tuple_raw;
  logic weight_req_slot_free_s;
  logic weight_tuple_slot_free_s;

  always_comb begin
    f_base_cur = f_group * PF;
    c_base_cur = c_group * PC;
  end

  assign data_tuple_valid_q = data_valid_q;

  // Data/weight availability includes a registered active tuple OR a tuple
  // arriving in the current cycle.  This removes the previous half-rate
  // behavior where Mode2 had to wait one extra cycle after every load.
  assign data_available_s = data_valid_q || dr_write_en;
  assign wgt_available_s  = wgt_tuple_ready_q || weight_write_en;

  // Use the registered tuple when one is already active.  Otherwise, if a new
  // tuple arrives in this cycle and the controller fires, the MAC sees the
  // incoming tuple directly through this bypass path.
  assign data_tuple_raw   = data_valid_q ? data_tuple_q : dr_write_data;
  assign weight_tuple_raw = wgt_tuple_ready_q ? weight_out_raw : weight_write_data;

  // A write is accepted if the active data slot is empty, or if the active
  // tuple is consumed in the same cycle.  If the slot is full and there is no
  // consume, the write is flagged as overflow and ignored by the valid tracker.
  assign data_accept_s   = dr_write_en && (!data_valid_q || mac_en || start);
  assign data_overflow_s = dr_write_en && data_valid_q && !mac_en && !start;

  assign tuple_ready_s = data_available_s && wgt_available_s;
  assign ctrl_step_en  = step_en && tuple_ready_s;

  // Allow a new weight read in the same cycle a previous read returns, but
  // only when the weight tuple slot is/will be free.  This prevents prefetching
  // over an unconsumed weight tuple, while enabling the desired sequence:
  //   return current tuple + consume current tuple + issue next tuple.
  assign weight_req_slot_free_s   = !weight_req_inflight_q || weight_write_en;
  assign weight_tuple_slot_free_s = start || mac_en || (!wgt_tuple_ready_q && !weight_write_en);

  assign weight_bank_ready_m2_s = weight_bank_ready &&
                                  weight_req_slot_free_s &&
                                  weight_tuple_slot_free_s;

  ce_controller_mode2 #(
    .K_MAX    (K_MAX),
    .HOUT_MAX (HOUT_MAX),
    .WOUT_MAX (WOUT_MAX),
    .PC       (PC),
    .PF       (PF)
  ) u_ce_controller_mode2 (
    .clk               (clk),
    .rst_n             (rst_n),
    .start             (start),
    .step_en           (ctrl_step_en),
    .tuple_ready       (tuple_ready_s),
    .K_cur             (K_cur),
    .C_cur             (C_cur),
    .F_cur             (F_cur),
    .Hout_cur          (Hout_cur),
    .Wout_cur          (Wout_cur),
    .tile_col_base_g   (tile_col_base_g),
    .tile_col_count    (tile_col_count),
    .out_row           (out_row),
    .out_col           (out_col),
    .f_group           (f_group),
    .c_group           (c_group),
    .ky                (ky),
    .kx                (kx),
    .mac_en            (mac_en),
    .clear_psum        (clear_psum),
    .out_valid         (out_valid),
    .pass_start_pulse  (pass_start_pulse),
    .group_start_pulse (group_start_pulse),
    .row_done_pulse    (row_done_pulse),
    .row_done_ky       (row_done_ky),
    .c_group_done_pulse(c_group_done_pulse),
    .pixel_done_pulse  (pixel_done_pulse),
    .f_group_done_pulse(f_group_done_pulse),
    .done              (done),
    .busy              (busy)
  );

  data_register_mode2 #(
    .K_MAX  (K_MAX),
    .DATA_W (DATA_W),
    .PC     (PC)
  ) u_data_register_mode2 (
    .clk          (clk),
    .rst_n        (rst_n),
    .K_cur        (K_cur),
    .C_cur        (C_cur),
    .c_group      (c_group),
    .write_en     (dr_write_en),
    .write_row_idx(dr_write_row_idx),
    .write_data   (dr_write_data),
    .read_row_idx (ky),
    .data_out     (data_out_raw)
  );

  weight_read_ctrl_mode2 #(
    .DATA_W   (DATA_W),
    .PC       (PC),
    .PF       (PF),
    .WB_LANES (WB_LANES),
    .WB_ADDR_W(WB_ADDR_W)
  ) u_weight_read_ctrl_mode2 (
    .clk             (clk),
    .rst_n           (rst_n),
    .K_cur           (K_cur),
    .C_cur           (C_cur),
    .F_cur           (F_cur),
    .Hout_cur        (Hout_cur),
    .Wout_cur        (Wout_cur),
    .start           (start),
    .pass_start_pulse(pass_start_pulse),
    .mac_en          (mac_en),
    .out_valid       (out_valid),
    .f_group         (f_group),
    .out_row         (out_row),
    .out_col         (out_col),
    .wb_bank_sel     (weight_bank_sel),
    .wb_bank_ready   (weight_bank_ready_m2_s),
    .wb_rd_en        (wb_rd_en),
    .wb_rd_buf_sel   (wb_rd_buf_sel),
    .wb_rd_addr      (wb_rd_addr),
    .wb_rd_data      (wb_rd_data),
    .wb_rd_valid     (wb_rd_valid),
    .weight_write_en (weight_write_en),
    .weight_write_data(weight_write_data)
  );

  weight_register_mode2 #(
    .DATA_W (DATA_W),
    .PC     (PC),
    .PF     (PF)
  ) u_weight_register_mode2 (
    .clk       (clk),
    .rst_n     (rst_n),
    .write_en  (weight_write_en),
    .write_data(weight_write_data),
    .weight_out(weight_out_raw)
  );

  // ------------------------------------------------------------
  // Consume-qualified tuple tracking for Mode 2.
  //
  // This is the Mode-2 counterpart of Mode1's data_ready_q /
  // weight_valid_q / ctrl_step_en contract.  A returned IFM tuple is
  // queued and held until mac_en consumes it.  A returned weight bundle
  // first loads weight_register_mode2.  Only on the following cycle does
  // wgt_tuple_ready_q allow ce_controller_mode2 to assert mac_en and
  // advance the loop counters.
  //
  // Important: ctrl_step_en can also be used by ce_controller_mode2 to
  // leave S_CLEAR via pass_start_pulse.  That is not a MAC consume yet,
  // so the data/weight current tuple is cleared only on mac_en.
  // ------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin : M2_CONSUME_TRACKING
    if (!rst_n) begin
      data_tuple_q          <= '0;
      data_valid_q          <= 1'b0;
      wgt_tuple_ready_q     <= 1'b0;
      weight_req_inflight_q <= 1'b0;
    end
    else begin
      if (start) begin
        data_tuple_q          <= '0;
        data_valid_q          <= 1'b0;
        wgt_tuple_ready_q     <= 1'b0;
        weight_req_inflight_q <= wb_rd_en;
      end
      else begin
        // ----------------------------------------------------------
        // Current IFM tuple validity.
        //
        // Cases:
        //   data_valid=1, mac=1, dr_write=1:
        //     consume old registered tuple and queue arriving tuple.
        //   data_valid=0, mac=1, dr_write=1:
        //     consume arriving tuple through bypass; do NOT mark it valid
        //     for the next cycle, otherwise it would be reused.
        //   data_valid=0, mac=0, dr_write=1:
        //     queue arriving tuple for a later consume.
        //   data_valid=1, mac=0, dr_write=1:
        //     overflow/ignored; active tuple must not be overwritten.
        // ----------------------------------------------------------
        if (dr_write_en && data_valid_q && !mac_en) begin
          data_valid_q <= data_valid_q;
        end
        else if (mac_en && dr_write_en) begin
          data_tuple_q <= dr_write_data;
          data_valid_q <= data_valid_q;  // old valid -> refill, bypass -> consumed
        end
        else if (dr_write_en) begin
          data_tuple_q <= dr_write_data;
          data_valid_q <= 1'b1;
        end
        else if (mac_en) begin
          data_valid_q <= 1'b0;
        end

        // ----------------------------------------------------------
        // Current weight tuple validity.
        //
        // weight_write_en loads weight_register_mode2.  If there was no
        // previously valid weight and mac_en fires in the same cycle, the
        // MAC consumed the arriving weight through bypass, so the registered
        // copy must NOT be considered valid on the next cycle.
        // ----------------------------------------------------------
        if (mac_en && weight_write_en) begin
          wgt_tuple_ready_q <= wgt_tuple_ready_q; // refill old-valid, consume bypass if old invalid
        end
        else if (weight_write_en) begin
          wgt_tuple_ready_q <= 1'b1;
        end
        else if (mac_en) begin
          wgt_tuple_ready_q <= 1'b0;
        end

        // Track one outstanding weight request.  If a return and a new
        // request happen in the same cycle, keep the request in-flight.
        if (weight_write_en) begin
          weight_req_inflight_q <= 1'b0;
        end
        if (wb_rd_en) begin
          weight_req_inflight_q <= 1'b1;
        end
      end
    end
  end

  // ------------------------------------------------------------
  // Final Mode-2 lane mask before the MAC array.
  //
  // Mode 2 reduces across all PC lanes. For partial C groups
  // (for example C_cur=3, PC=32), inactive channel lanes must be
  // zero, otherwise a single X on pc>=C_cur poisons the whole sum.
  // Also mask partial F groups so inactive filters cannot consume
  // stale/rubbish weights.
  // ------------------------------------------------------------
  always_comb begin
    data_out_logic = '0;
    weight_out     = '0;

    for (int pc_i = 0; pc_i < PC; pc_i++) begin
      if ((c_base_cur + pc_i) < C_cur) begin
        data_out_logic[pc_i*DATA_W +: DATA_W]
          = data_tuple_raw[pc_i*DATA_W +: DATA_W];
      end
    end

    for (int pf_i = 0; pf_i < PF; pf_i++) begin
      for (int pc_i = 0; pc_i < PC; pc_i++) begin
        if (((c_base_cur + pc_i) < C_cur) &&
            ((f_base_cur + pf_i) < F_cur)) begin
          weight_out[(pf_i*PC + pc_i)*DATA_W +: DATA_W]
            = weight_tuple_raw[(pf_i*PC + pc_i)*DATA_W +: DATA_W];
        end
      end
    end
  end

  mac_array_mode2 #(
    .DATA_W (DATA_W),
    .PSUM_W (PSUM_W),
    .PC     (PC),
    .PF     (PF)
  ) u_mac_array_mode2 (
    .clk           (clk),
    .rst_n         (rst_n),
    .mac_en        (mac_en),
    .out_valid     (out_valid),
    .data_in       (data_out_logic),
    .weight_in     (weight_out),
    .data_out      (mac_data_out),
    .data_out_valid(mac_data_out_valid)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      mac_group_start <= 1'b0;
      mac_f_base      <= '0;
    end
    else begin
      mac_group_start <= group_start_pulse;
      if (out_valid)
        mac_f_base <= f_base_cur;
    end
  end
  
  
endmodule
