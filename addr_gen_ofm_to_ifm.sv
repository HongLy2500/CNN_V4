module addr_gen_ofm_to_ifm #(
  parameter int PV_MAX = 8,
  parameter int PC     = 8,
  parameter int F_MAX  = 512,
  parameter int H_MAX  = 224,
  parameter int W_MAX  = 224
)(
  input  logic clk,
  input  logic rst_n,

  // --------------------------------------------------
  // Request from control unit / scheduler
  //
  // One request generates exactly one OFM->IFM stream command.
  //
  // Supported cases:
  //   src=0, next=0 : mode1 -> mode1 (continuous direct refill)
  //   src=1, next=1 : mode2 -> mode2 (continuous direct refill)
  //   src=0, next=1 : mode1 -> mode2 (special full-store then refill)
  //
  // Unsupported:
  //   src=1, next=0 : mode2 -> mode1
  //
  // Request fields:
  //   req_row_base  : absolute OFM row base to stream
  //   req_num_rows  : number of rows to stream
  //   req_col_base  : used only when next_mode=1, selects horizontal tile base
  //                   in pixel coordinates; must be aligned to PC
  //                   for mode2-style IFM refill.
  // --------------------------------------------------
  input  logic                        start,
  input  logic                        cfg_src_mode,
  input  logic                        cfg_next_mode,
  input  logic [$clog2(H_MAX+1)-1:0]  cfg_h_out,
  input  logic [$clog2(W_MAX+1)-1:0]  cfg_w_out,
  input  logic [$clog2(F_MAX+1)-1:0]  cfg_f_out,
  input  logic [$clog2(PV_MAX+1)-1:0] cfg_pv_next,

  input  logic [$clog2(H_MAX+1)-1:0]  req_row_base,
  input  logic [$clog2(H_MAX+1)-1:0]  req_num_rows,
  input  logic [$clog2(W_MAX+1)-1:0]  req_col_base,

  // --------------------------------------------------
  // Status from ofm_buffer stream interface
  // --------------------------------------------------
  input  logic                        ofm_stream_busy,
  input  logic                        ofm_stream_done,
  input  logic                        ofm_layer_write_done,
  input  logic                        ofm_error,

  // --------------------------------------------------
  // Command to ofm_buffer
  // --------------------------------------------------
  output logic                        ifm_stream_start,
  output logic [$clog2(H_MAX+1)-1:0]  ifm_stream_row_base,
  output logic [$clog2(H_MAX+1)-1:0]  ifm_stream_num_rows,
  output logic [$clog2(W_MAX+1)-1:0]  ifm_stream_col_base,

  // --------------------------------------------------
  // Status back to control unit
  // --------------------------------------------------
  output logic                        busy,
  output logic                        done,
  output logic                        error,

  // --------------------------------------------------
  // Optional debug / visibility
  // --------------------------------------------------
  output logic [31:0]                 dbg_num_mode1_groups,
  output logic [31:0]                 dbg_num_mode2_tiles,
  output logic [31:0]                 dbg_last_row,
  output logic [31:0]                 dbg_last_col,
  output logic                        dbg_need_full_store,
  output logic                        dbg_waiting_for_layer,
  output logic                        dbg_waiting_for_stream
);

  localparam int ROW_W = (H_MAX <= 1) ? 1 : $clog2(H_MAX+1);
  localparam int COL_W = (W_MAX <= 1) ? 1 : $clog2(W_MAX+1);

  typedef enum logic [2:0] {
    ST_IDLE,
    ST_WAIT_LAYER,
    ST_WAIT_READY,
    ST_WAIT_STREAM,
    ST_DONE,
    ST_ERROR
  } state_t;

  state_t state_q, state_d;

  logic        src_mode_q, src_mode_d;
  logic        next_mode_q, next_mode_d;
  logic [ROW_W-1:0] row_base_q, row_base_d;
  logic [ROW_W-1:0] num_rows_q, num_rows_d;
  logic [COL_W-1:0] col_base_q, col_base_d;

  logic [31:0] num_mode1_groups32;
  logic [31:0] num_mode2_tiles32;
  logic [31:0] req_last_row32;
  logic [31:0] req_last_col32;
  logic        supported_cfg;
  logic        need_full_store;
  logic        cfg_valid;

  // --------------------------------------------------
  // Derived counts / layout checks
  // --------------------------------------------------
  always_comb begin
    if (cfg_pv_next == 0)
      num_mode1_groups32 = 32'd0;
    else
      num_mode1_groups32 = (cfg_w_out + cfg_pv_next - 1) / cfg_pv_next;
  end

  always_comb begin
    num_mode2_tiles32 = (cfg_w_out + PC - 1) / PC;
    req_last_row32    = (req_num_rows == 0) ? req_row_base : (req_row_base + req_num_rows - 1);
    req_last_col32    = (cfg_next_mode == 1'b0)
                      ? ((cfg_w_out == 0) ? 32'd0 : (cfg_w_out - 1))
                      : ((req_col_base >= cfg_w_out) ? req_col_base : (((req_col_base + PC) > cfg_w_out) ? (cfg_w_out - 1) : (req_col_base + PC - 1)));
  end

  always_comb begin
    supported_cfg  = 1'b0;
    need_full_store= 1'b0;
    cfg_valid      = 1'b0;

    // Supported mode transitions only.
    if ((!cfg_src_mode && !cfg_next_mode) ||
        ( cfg_src_mode &&  cfg_next_mode) ||
        (!cfg_src_mode &&  cfg_next_mode)) begin
      supported_cfg = 1'b1;
    end

    need_full_store = (!cfg_src_mode && cfg_next_mode);

    if (supported_cfg &&
        (cfg_h_out != 0) &&
        (cfg_w_out != 0) &&
        (cfg_f_out != 0) &&
        (req_num_rows != 0) &&
        (req_row_base < cfg_h_out) &&
        ((req_row_base + req_num_rows) <= cfg_h_out)) begin

      if (!cfg_next_mode) begin
        // next mode 1: row range only, col base ignored by ofm_buffer.
        if ((cfg_pv_next != 0) && (cfg_pv_next <= PV_MAX) && (req_col_base == 0))
          cfg_valid = 1'b1;
      end
      else begin
        // next mode 2: req_col_base selects one PC-wide horizontal tile.
        if ((req_col_base < cfg_w_out) && ((req_col_base % PC) == 0))
          cfg_valid = 1'b1;
      end
    end
  end

  assign dbg_num_mode1_groups = num_mode1_groups32;
  assign dbg_num_mode2_tiles  = num_mode2_tiles32;
  assign dbg_last_row         = req_last_row32;
  assign dbg_last_col         = req_last_col32;
  assign dbg_need_full_store  = need_full_store;
  assign dbg_waiting_for_layer  = (state_q == ST_WAIT_LAYER);
  assign dbg_waiting_for_stream = (state_q == ST_WAIT_READY) || (state_q == ST_WAIT_STREAM);

  // --------------------------------------------------
  // Next-state / command generation
  // --------------------------------------------------
  always_comb begin
    state_d     = state_q;
    src_mode_d  = src_mode_q;
    next_mode_d = next_mode_q;
    row_base_d  = row_base_q;
    num_rows_d  = num_rows_q;
    col_base_d  = col_base_q;

    ifm_stream_start    = 1'b0;
    ifm_stream_row_base = row_base_q;
    ifm_stream_num_rows = num_rows_q;
    ifm_stream_col_base = col_base_q;

    done  = 1'b0;
    error = 1'b0;

    case (state_q)
      ST_IDLE: begin
        if (start) begin
          if (!cfg_valid) begin
            state_d = ST_ERROR;
          end
          else begin
            src_mode_d  = cfg_src_mode;
            next_mode_d = cfg_next_mode;
            row_base_d  = req_row_base;
            num_rows_d  = req_num_rows;
            col_base_d  = req_col_base;

            if (need_full_store && !ofm_layer_write_done) begin
              state_d = ST_WAIT_LAYER;
            end
            else if (!ofm_stream_busy) begin
              ifm_stream_start    = 1'b1;
              ifm_stream_row_base = req_row_base;
              ifm_stream_num_rows = req_num_rows;
              ifm_stream_col_base = req_col_base;
              state_d             = ST_WAIT_STREAM;
            end
            else begin
              state_d = ST_WAIT_READY;
            end
          end
        end
      end

      ST_WAIT_LAYER: begin
        ifm_stream_row_base = row_base_q;
        ifm_stream_num_rows = num_rows_q;
        ifm_stream_col_base = col_base_q;

        if (!supported_cfg) begin
          state_d = ST_ERROR;
        end
        else if (ofm_error) begin
          state_d = ST_ERROR;
        end
        else if (ofm_layer_write_done) begin
          if (!ofm_stream_busy) begin
            ifm_stream_start = 1'b1;
            state_d          = ST_WAIT_STREAM;
          end
          else begin
            state_d = ST_WAIT_READY;
          end
        end
      end

      ST_WAIT_READY: begin
        ifm_stream_row_base = row_base_q;
        ifm_stream_num_rows = num_rows_q;
        ifm_stream_col_base = col_base_q;

        if (ofm_error) begin
          state_d = ST_ERROR;
        end
        else if ((!need_full_store || ofm_layer_write_done) && !ofm_stream_busy) begin
          ifm_stream_start = 1'b1;
          state_d          = ST_WAIT_STREAM;
        end
        else if (need_full_store && !ofm_layer_write_done) begin
          state_d = ST_WAIT_LAYER;
        end
      end

      ST_WAIT_STREAM: begin
        ifm_stream_row_base = row_base_q;
        ifm_stream_num_rows = num_rows_q;
        ifm_stream_col_base = col_base_q;

        if (ofm_error) begin
          state_d = ST_ERROR;
        end
        else if (ofm_stream_done) begin
          state_d = ST_DONE;
        end
      end

      ST_DONE: begin
        done    = 1'b1;
        state_d = ST_IDLE;
      end

      ST_ERROR: begin
        error   = 1'b1;
        state_d = ST_IDLE;
      end

      default: begin
        state_d = ST_IDLE;
      end
    endcase
  end

  assign busy = (state_q != ST_IDLE);

  // --------------------------------------------------
  // Sequential state / latched request fields
  // --------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state_q     <= ST_IDLE;
      src_mode_q  <= 1'b0;
      next_mode_q <= 1'b0;
      row_base_q  <= '0;
      num_rows_q  <= '0;
      col_base_q  <= '0;
    end
    else begin
      state_q     <= state_d;
      src_mode_q  <= src_mode_d;
      next_mode_q <= next_mode_d;
      row_base_q  <= row_base_d;
      num_rows_q  <= num_rows_d;
      col_base_q  <= col_base_d;
    end
  end

endmodule
