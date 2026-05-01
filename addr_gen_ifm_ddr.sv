`include "cnn_ddr_defs.svh"

module addr_gen_ifm_ddr #(
  parameter int PV_MAX     = 8,
  parameter int PC         = 8,
  parameter int C_MAX      = 64,
  parameter int W_MAX      = 224,
  parameter int H_MAX      = 224,
  parameter int HT         = 8,
  parameter int DDR_ADDR_W = `CNN_DDR_ADDR_W
)(
  input  logic clk,
  input  logic rst_n,

  // --------------------------------------------------
  // Request from control unit / scheduler
  //
  // One request generates exactly one IFM DMA command.
  //
  // Mode 1 DDR layout assumption:
  //   [channel][row][col_group]
  //   col_group count per row = ceil(W_layer / Pv_mode1)
  //   A command loads num_rows consecutive rows across all channels,
  //   starting from abs_row_base, into ifm_buffer rows beginning at
  //   buf_row_base inside the current HT window.
  //
  // Mode 2 DDR layout assumption:
  //   [tile_x][channel][row]
  //   one word = one PC-wide horizontal segment for a given (channel,row)
  //   req_m2_tile_idx selects which horizontal tile is loaded.
  //   A command loads num_rows consecutive absolute rows across all channels,
  //   starting from abs_row_base, into ifm_buffer rows beginning at
  //   buf_row_base (normally equal to abs_row_base for mode 2).
  // --------------------------------------------------
  input  logic                        start,
  input  logic                        cfg_mode,          // 0: mode1, 1: mode2
  input  logic [DDR_ADDR_W-1:0]       cfg_ifm_ddr_base,
  input  logic [$clog2(W_MAX+1)-1:0]  cfg_w_layer,
  input  logic [$clog2(H_MAX+1)-1:0]  cfg_h_in,
  input  logic [$clog2(C_MAX+1)-1:0]  cfg_c_in,
  input  logic [$clog2(PV_MAX+1)-1:0] cfg_pv_cur,

  // Request-local shape / placement
  input  logic [$clog2(H_MAX+1)-1:0]  req_abs_row_base,
  input  logic [$clog2(H_MAX+1)-1:0]  req_num_rows,
  input  logic [((H_MAX <= 1) ? 1 : $clog2(H_MAX))-1:0] req_buf_row_base,
  input  logic [(((W_MAX + PC - 1) / PC) <= 1 ? 1 : $clog2(((W_MAX + PC - 1) / PC) + 1))-1:0] req_m2_tile_idx,

  // --------------------------------------------------
  // Handshake/status from DMA
  // --------------------------------------------------
  input  logic                        dma_busy,
  input  logic                        dma_done_ifm,
  input  logic                        dma_error,

  // --------------------------------------------------
  // Command to cnn_dma_direct
  // --------------------------------------------------
  output logic                        ifm_cmd_start,
  output logic [DDR_ADDR_W-1:0]       ifm_cmd_ddr_base,
  output logic [$clog2(H_MAX+1)-1:0]  ifm_cmd_num_rows,
  output logic [((H_MAX <= 1) ? 1 : $clog2(H_MAX))-1:0] ifm_cmd_buf_row_base,

  // --------------------------------------------------
  // Status back to control unit
  // --------------------------------------------------
  output logic                        busy,
  output logic                        done,
  output logic                        error,

  // Optional debug / visibility
  output logic [31:0]                 dbg_mode1_words_per_row,
  output logic [31:0]                 dbg_mode2_num_tiles,
  output logic [31:0]                 dbg_total_words,
  output logic [31:0]                 dbg_ddr_base_word_addr,
  output logic [31:0]                 dbg_ddr_last_word_addr,
  output logic                        dbg_waiting_for_dma
);

  localparam int CMD_ROW_W  = (H_MAX <= 1) ? 1 : $clog2(H_MAX+1);
  localparam int BUF_ROW_W  = (H_MAX <= 1) ? 1 : $clog2(H_MAX);
  localparam int TILES_MAX  = (W_MAX + PC - 1) / PC;
  localparam int TILE_W     = (TILES_MAX <= 1) ? 1 : $clog2(TILES_MAX + 1);

  localparam logic [DDR_ADDR_W-1:0] DDR_IFM_END = `DDR_IFM_BASE + `DDR_IFM_SIZE - 1;

  typedef enum logic [2:0] {
    ST_IDLE,
    ST_WAIT_READY,
    ST_WAIT_DMA,
    ST_DONE,
    ST_ERROR
  } state_t;

  state_t state_q, state_d;

  logic [DDR_ADDR_W-1:0] ddr_base_q, ddr_base_d;
  logic [CMD_ROW_W-1:0]  num_rows_q, num_rows_d;
  logic [BUF_ROW_W-1:0]  buf_row_base_q, buf_row_base_d;

  logic [31:0] mode1_words_per_row32;
  logic [31:0] mode2_num_tiles32;
  logic [31:0] tile_words32;
  logic [31:0] req_num_rows32;
  logic [31:0] req_abs_row_base32;
  logic [31:0] req_buf_row_base32;
  logic [31:0] req_tile_idx32;
  logic [31:0] total_words32;
  logic [31:0] ddr_base_word_addr32;
  logic [31:0] ddr_last_word_addr32;
  logic        cfg_valid;

  assign req_num_rows32      = req_num_rows;
  assign req_abs_row_base32  = req_abs_row_base;
  assign req_buf_row_base32  = req_buf_row_base;
  assign req_tile_idx32      = req_m2_tile_idx;

  // --------------------------------------------------
  // Derived counts / layout parameters
  // --------------------------------------------------
  always_comb begin
    if (cfg_pv_cur == 0)
      mode1_words_per_row32 = 32'd0;
    else
      mode1_words_per_row32 = (cfg_w_layer + cfg_pv_cur - 1) / cfg_pv_cur;
  end

  always_comb begin
    mode2_num_tiles32 = (cfg_w_layer + PC - 1) / PC;
    tile_words32      = cfg_c_in * cfg_h_in;
  end

  // --------------------------------------------------
  // Command legality checks + computed command body
  // --------------------------------------------------
  always_comb begin
    total_words32        = 32'd0;
    ddr_base_word_addr32 = cfg_ifm_ddr_base;
    ddr_last_word_addr32 = cfg_ifm_ddr_base;
    cfg_valid            = 1'b0;

    if (!cfg_mode) begin
      // --------------------------
      // Mode 1: [channel][row][col_group]
      // --------------------------
      total_words32        = cfg_c_in * req_num_rows32 * mode1_words_per_row32;
      ddr_base_word_addr32 = cfg_ifm_ddr_base + (req_abs_row_base32 * mode1_words_per_row32);
      ddr_last_word_addr32 = ddr_base_word_addr32 + ((total_words32 == 0) ? 0 : (total_words32 - 1));

      if ((cfg_pv_cur != 0) &&
          (cfg_pv_cur <= PV_MAX) &&
          (cfg_c_in != 0) &&
          (cfg_h_in != 0) &&
          (cfg_w_layer != 0) &&
          (req_num_rows32 != 0) &&
          (req_abs_row_base32 < cfg_h_in) &&
          ((req_abs_row_base32 + req_num_rows32) <= cfg_h_in) &&
          (req_buf_row_base32 < HT) &&
          ((req_buf_row_base32 + req_num_rows32) <= HT) &&
          (cfg_ifm_ddr_base >= `DDR_IFM_BASE) &&
          (ddr_last_word_addr32 <= DDR_IFM_END)) begin
        cfg_valid = 1'b1;
      end
    end
    else begin
      // --------------------------
      // Mode 2: [tile_x][channel][row]
      // one tile = C * H words
      // --------------------------
      total_words32        = cfg_c_in * req_num_rows32;
      ddr_base_word_addr32 = cfg_ifm_ddr_base + (req_tile_idx32 * tile_words32) + req_abs_row_base32;
      ddr_last_word_addr32 = ddr_base_word_addr32 + ((total_words32 == 0) ? 0 : (total_words32 - 1));

      if ((cfg_c_in != 0) &&
          (cfg_h_in != 0) &&
          (cfg_w_layer != 0) &&
          (req_num_rows32 != 0) &&
          (req_abs_row_base32 < cfg_h_in) &&
          ((req_abs_row_base32 + req_num_rows32) <= cfg_h_in) &&
          (req_buf_row_base32 < H_MAX) &&
          ((req_buf_row_base32 + req_num_rows32) <= H_MAX) &&
          (req_tile_idx32 < mode2_num_tiles32) &&
          (cfg_ifm_ddr_base >= `DDR_IFM_BASE) &&
          (ddr_last_word_addr32 <= DDR_IFM_END)) begin
        cfg_valid = 1'b1;
      end
    end
  end

  assign dbg_mode1_words_per_row = mode1_words_per_row32;
  assign dbg_mode2_num_tiles     = mode2_num_tiles32;
  assign dbg_total_words         = total_words32;
  assign dbg_ddr_base_word_addr  = ddr_base_word_addr32;
  assign dbg_ddr_last_word_addr  = ddr_last_word_addr32;
  assign dbg_waiting_for_dma     = (state_q == ST_WAIT_READY) && dma_busy;

  // --------------------------------------------------
  // Next-state / command pulse generation
  // --------------------------------------------------
  always_comb begin
    state_d        = state_q;
    ddr_base_d     = ddr_base_q;
    num_rows_d     = num_rows_q;
    buf_row_base_d = buf_row_base_q;

    ifm_cmd_start        = 1'b0;
    ifm_cmd_ddr_base     = ddr_base_q;
    ifm_cmd_num_rows     = num_rows_q;
    ifm_cmd_buf_row_base = buf_row_base_q;

    done  = 1'b0;
    error = 1'b0;

    case (state_q)
      ST_IDLE: begin
        if (start) begin
          if (!cfg_valid) begin
            state_d = ST_ERROR;
          end
          else begin
            ddr_base_d     = ddr_base_word_addr32[DDR_ADDR_W-1:0];
            num_rows_d     = req_num_rows;
            buf_row_base_d = req_buf_row_base;

            if (!dma_busy) begin
              ifm_cmd_start        = 1'b1;
              ifm_cmd_ddr_base     = ddr_base_word_addr32[DDR_ADDR_W-1:0];
              ifm_cmd_num_rows     = req_num_rows;
              ifm_cmd_buf_row_base = req_buf_row_base;
              state_d              = ST_WAIT_DMA;
            end
            else begin
              state_d = ST_WAIT_READY;
            end
          end
        end
      end

      ST_WAIT_READY: begin
        ifm_cmd_ddr_base     = ddr_base_q;
        ifm_cmd_num_rows     = num_rows_q;
        ifm_cmd_buf_row_base = buf_row_base_q;

        if (!cfg_valid) begin
          state_d = ST_ERROR;
        end
        else if (!dma_busy) begin
          ifm_cmd_start = 1'b1;
          state_d       = ST_WAIT_DMA;
        end
      end

      ST_WAIT_DMA: begin
        ifm_cmd_ddr_base     = ddr_base_q;
        ifm_cmd_num_rows     = num_rows_q;
        ifm_cmd_buf_row_base = buf_row_base_q;

        if (dma_error) begin
          state_d = ST_ERROR;
        end
        else if (dma_done_ifm) begin
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
  // Sequential state / command register storage
  // --------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state_q        <= ST_IDLE;
      ddr_base_q     <= '0;
      num_rows_q     <= '0;
      buf_row_base_q <= '0;
    end
    else begin
      state_q        <= state_d;
      ddr_base_q     <= ddr_base_d;
      num_rows_q     <= num_rows_d;
      buf_row_base_q <= buf_row_base_d;
    end
  end

endmodule
