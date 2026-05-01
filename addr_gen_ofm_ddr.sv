`include "cnn_ddr_defs.svh"

module addr_gen_ofm_ddr #(
  parameter int DDR_ADDR_W       = `CNN_DDR_ADDR_W,
  parameter int OFM_LINEAR_DEPTH = 4096
)(
  input  logic clk,
  input  logic rst_n,

  // --------------------------------------------------
  // Request from control unit / scheduler
  //
  // This generator is intended for final-layer writeback (or optional dump)
  // from ofm_buffer to DDR. The OFM local layout is already owned by
  // ofm_buffer, so this block does NOT recompute the number of words.
  // Instead it consumes the exact linear word count reported by ofm_buffer.
  // --------------------------------------------------
  input  logic                          start,
  input  logic [DDR_ADDR_W-1:0]         cfg_ofm_ddr_base,
  input  logic [$clog2(OFM_LINEAR_DEPTH)-1:0] cfg_ofm_buf_base,

  // --------------------------------------------------
  // Status from ofm_buffer
  // --------------------------------------------------
  input  logic [31:0]                   ofm_layer_num_words,
  input  logic                          ofm_layer_write_done,
  input  logic                          ofm_error,

  // --------------------------------------------------
  // Handshake/status from DMA
  // --------------------------------------------------
  input  logic                          dma_busy,
  input  logic                          dma_done_ofm,
  input  logic                          dma_error,

  // --------------------------------------------------
  // Command to cnn_dma_direct
  // --------------------------------------------------
  output logic                          ofm_cmd_start,
  output logic [DDR_ADDR_W-1:0]         ofm_cmd_ddr_base,
  output logic [$clog2(OFM_LINEAR_DEPTH+1)-1:0] ofm_cmd_num_words,
  output logic [$clog2(OFM_LINEAR_DEPTH)-1:0]   ofm_cmd_buf_base,

  // --------------------------------------------------
  // Status back to control unit
  // --------------------------------------------------
  output logic                          busy,
  output logic                          done,
  output logic                          error,

  // Optional debug / visibility
  output logic [31:0]                   dbg_num_words,
  output logic [31:0]                   dbg_ddr_last_addr,
  output logic [31:0]                   dbg_buf_last_addr,
  output logic                          dbg_waiting_for_layer,
  output logic                          dbg_waiting_for_dma
);

  localparam int CMD_W    = (OFM_LINEAR_DEPTH <= 1) ? 1 : $clog2(OFM_LINEAR_DEPTH+1);
  localparam int BUF_AW   = (OFM_LINEAR_DEPTH <= 1) ? 1 : $clog2(OFM_LINEAR_DEPTH);

  localparam logic [DDR_ADDR_W-1:0] DDR_OFM_END = `DDR_OFM_BASE + `DDR_OFM_SIZE - 1;

  typedef enum logic [2:0] {
    ST_IDLE,
    ST_WAIT_READY,
    ST_WAIT_DMA,
    ST_DONE,
    ST_ERROR
  } state_t;

  state_t state_q, state_d;

  logic [DDR_ADDR_W-1:0] ddr_base_q, ddr_base_d;
  logic [CMD_W-1:0]      num_words_q, num_words_d;
  logic [BUF_AW-1:0]     buf_base_q,  buf_base_d;

  logic [31:0]           num_words32;
  logic [31:0]           buf_base32;
  logic [31:0]           buf_last_addr32;
  logic [31:0]           ddr_last_addr32;
  logic                  cfg_valid;
  logic                  wait_for_layer;
  logic                  wait_for_dma;

  // --------------------------------------------------
  // Command/count validity
  // --------------------------------------------------
  always_comb begin
    num_words32    = ofm_layer_num_words;
    buf_base32     = cfg_ofm_buf_base;
    buf_last_addr32= 32'd0;
    ddr_last_addr32= 32'd0;
    cfg_valid      = 1'b0;

    if (num_words32 != 0) begin
      buf_last_addr32 = buf_base32 + num_words32 - 1;
      ddr_last_addr32 = cfg_ofm_ddr_base + num_words32 - 1;

      // Valid only when:
      // 1) local OFM read range stays within the linear OFM storage space
      // 2) DDR write range stays within the OFM DDR region
      if ((buf_last_addr32 < OFM_LINEAR_DEPTH) &&
          (cfg_ofm_ddr_base >= `DDR_OFM_BASE) &&
          (ddr_last_addr32 <= DDR_OFM_END)) begin
        cfg_valid = 1'b1;
      end
    end
  end

  assign dbg_num_words     = num_words32;
  assign dbg_ddr_last_addr = ddr_last_addr32;
  assign dbg_buf_last_addr = buf_last_addr32;

  assign wait_for_layer = !ofm_layer_write_done;
  assign wait_for_dma   = dma_busy;

  assign dbg_waiting_for_layer = (state_q == ST_WAIT_READY) && wait_for_layer;
  assign dbg_waiting_for_dma   = (state_q == ST_WAIT_READY) && !wait_for_layer && wait_for_dma;

  // --------------------------------------------------
  // Next-state / command pulse generation
  // --------------------------------------------------
  always_comb begin
    state_d     = state_q;
    ddr_base_d  = ddr_base_q;
    num_words_d = num_words_q;
    buf_base_d  = buf_base_q;

    ofm_cmd_start     = 1'b0;
    ofm_cmd_ddr_base  = ddr_base_q;
    ofm_cmd_num_words = num_words_q;
    ofm_cmd_buf_base  = buf_base_q;

    done  = 1'b0;
    error = 1'b0;

    case (state_q)
      ST_IDLE: begin
        if (start) begin
          if (ofm_error || !cfg_valid) begin
            state_d = ST_ERROR;
          end
          else begin
            ddr_base_d  = cfg_ofm_ddr_base;
            num_words_d = num_words32[CMD_W-1:0];
            buf_base_d  = cfg_ofm_buf_base;

            if (ofm_layer_write_done && !dma_busy) begin
              ofm_cmd_start     = 1'b1;
              ofm_cmd_ddr_base  = cfg_ofm_ddr_base;
              ofm_cmd_num_words = num_words32[CMD_W-1:0];
              ofm_cmd_buf_base  = cfg_ofm_buf_base;
              state_d           = ST_WAIT_DMA;
            end
            else begin
              state_d = ST_WAIT_READY;
            end
          end
        end
      end

      ST_WAIT_READY: begin
        ofm_cmd_ddr_base  = ddr_base_q;
        ofm_cmd_num_words = num_words_q;
        ofm_cmd_buf_base  = buf_base_q;

        if (ofm_error || !cfg_valid) begin
          state_d = ST_ERROR;
        end
        else if (ofm_layer_write_done && !dma_busy) begin
          ofm_cmd_start = 1'b1;
          state_d       = ST_WAIT_DMA;
        end
      end

      ST_WAIT_DMA: begin
        ofm_cmd_ddr_base  = ddr_base_q;
        ofm_cmd_num_words = num_words_q;
        ofm_cmd_buf_base  = buf_base_q;

        if (dma_error) begin
          state_d = ST_ERROR;
        end
        else if (dma_done_ofm) begin
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
      state_q     <= ST_IDLE;
      ddr_base_q  <= '0;
      num_words_q <= '0;
      buf_base_q  <= '0;
    end
    else begin
      state_q     <= state_d;
      ddr_base_q  <= ddr_base_d;
      num_words_q <= num_words_d;
      buf_base_q  <= buf_base_d;
    end
  end

endmodule
