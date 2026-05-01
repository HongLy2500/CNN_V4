module weight_buffer #(
  parameter int DATA_W     = 8,
  // Physical storage width of one weight-buffer word.
  // In the integrated design this should be PTOTAL.
  parameter int WORD_LANES = 16,
  // Logical read width for mode 1.
  parameter int PF_MAX     = 8,
  parameter int ADDR_W     = 12,
  parameter int DEPTH      = 4096
)(
  input  logic clk,
  input  logic rst_n,

  // =====================================================
  // DMA write side (physical-word write)
  // =====================================================
  input  logic                         dma_wr_en,
  input  logic                         dma_wr_buf_sel,
  input  logic [ADDR_W-1:0]            dma_wr_addr,
  input  logic [WORD_LANES*DATA_W-1:0] dma_wr_data,
  input  logic [WORD_LANES-1:0]        dma_wr_keep,
  output logic                         dma_wr_ready,

  input  logic                         dma_load_done,
  input  logic                         dma_load_buf_sel,
  input  logic                         bank0_release,
  input  logic                         bank1_release,

  // =====================================================
  // Mode-1 logical read side
  // One physical address stores several mode-1 logical bundles.
  // m1_rd_addr selects the physical word.
  // m1_rd_base_lane selects which Pf-wide chunk inside that word
  // is returned to mode 1 in this cycle.
  // =====================================================
  input  logic                         m1_rd_en,
  input  logic                         m1_rd_buf_sel,
  input  logic [ADDR_W-1:0]            m1_rd_addr,
  input  logic [$clog2(WORD_LANES)-1:0] m1_rd_base_lane,
  output logic [PF_MAX*DATA_W-1:0]     m1_rd_data,
  output logic [PF_MAX-1:0]            m1_rd_keep,
  output logic                         m1_rd_valid,

  // =====================================================
  // Full physical-word read side (mode 2 / debug / raw access)
  // One synchronous read, 1-cycle latency.
  // =====================================================
  input  logic                         rd_en,
  input  logic                         rd_buf_sel,
  input  logic [ADDR_W-1:0]            rd_addr,
  output logic [WORD_LANES*DATA_W-1:0] rd_data,
  output logic [WORD_LANES-1:0]        rd_keep,
  output logic                         rd_valid,

  output logic                         bank0_ready,
  output logic                         bank1_ready
);

  localparam int WORD_W    = WORD_LANES * DATA_W;
  localparam int BASE_W    = (WORD_LANES > 1) ? $clog2(WORD_LANES) : 1;

  logic [WORD_W-1:0]       mem0 [0:DEPTH-1];
  logic [WORD_W-1:0]       mem1 [0:DEPTH-1];
  logic [WORD_LANES-1:0]   keep0[0:DEPTH-1];
  logic [WORD_LANES-1:0]   keep1[0:DEPTH-1];

  // Registered raw/full-word read request
  logic                    rd_en_q;
  logic                    rd_buf_sel_q;
  logic [ADDR_W-1:0]       rd_addr_q;

  // Registered mode-1 logical read request
  logic                    m1_rd_en_q;
  logic                    m1_rd_buf_sel_q;
  logic [ADDR_W-1:0]       m1_rd_addr_q;
  logic [BASE_W-1:0]       m1_rd_base_lane_q;

  logic [WORD_W-1:0]       m1_raw_word;
  logic [WORD_LANES-1:0]   m1_raw_keep;

  integer lane;
  integer i;
  integer src_lane;

  assign dma_wr_ready = 1'b1;

  // -----------------------------------------------------
  // Write / ready-state management
  // -----------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      bank0_ready <= 1'b0;
      bank1_ready <= 1'b0;
    end
    else begin
      if (bank0_release)
        bank0_ready <= 1'b0;
      if (bank1_release)
        bank1_ready <= 1'b0;

      if (dma_wr_en) begin
        if (!dma_wr_buf_sel)
          bank0_ready <= 1'b0;
        else
          bank1_ready <= 1'b0;

        for (lane = 0; lane < WORD_LANES; lane++) begin
          if (dma_wr_keep[lane]) begin
            if (!dma_wr_buf_sel)
              mem0[dma_wr_addr][lane*DATA_W +: DATA_W] <= dma_wr_data[lane*DATA_W +: DATA_W];
            else
              mem1[dma_wr_addr][lane*DATA_W +: DATA_W] <= dma_wr_data[lane*DATA_W +: DATA_W];
          end
        end

        if (!dma_wr_buf_sel)
          keep0[dma_wr_addr] <= dma_wr_keep;
        else
          keep1[dma_wr_addr] <= dma_wr_keep;
      end

      if (dma_load_done) begin
        if (!dma_load_buf_sel)
          bank0_ready <= 1'b1;
        else
          bank1_ready <= 1'b1;
      end
    end
  end

  // -----------------------------------------------------
  // Registered read requests
  // -----------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_en_q            <= 1'b0;
      rd_buf_sel_q       <= 1'b0;
      rd_addr_q          <= '0;
      m1_rd_en_q         <= 1'b0;
      m1_rd_buf_sel_q    <= 1'b0;
      m1_rd_addr_q       <= '0;
      m1_rd_base_lane_q  <= '0;
    end
    else begin
      rd_en_q            <= rd_en;
      rd_buf_sel_q       <= rd_buf_sel;
      rd_addr_q          <= rd_addr;
      m1_rd_en_q         <= m1_rd_en;
      m1_rd_buf_sel_q    <= m1_rd_buf_sel;
      m1_rd_addr_q       <= m1_rd_addr;
      m1_rd_base_lane_q  <= m1_rd_base_lane;
    end
  end

  // -----------------------------------------------------
  // Registered full-word read data (mode 2 / raw)
  // -----------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_data  <= '0;
      rd_keep  <= '0;
      rd_valid <= 1'b0;
    end
    else begin
      rd_valid <= rd_en_q;

      if (rd_en_q) begin
        if (!rd_buf_sel_q) begin
          rd_data <= mem0[rd_addr_q];
          rd_keep <= keep0[rd_addr_q];
        end
        else begin
          rd_data <= mem1[rd_addr_q];
          rd_keep <= keep1[rd_addr_q];
        end
      end
    end
  end

  // -----------------------------------------------------
  // Registered mode-1 logical read data
  // -----------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      m1_raw_word <= '0;
      m1_raw_keep <= '0;
      m1_rd_valid <= 1'b0;
      m1_rd_data  <= '0;
      m1_rd_keep  <= '0;
    end
    else begin
      m1_rd_valid <= m1_rd_en_q;

      if (m1_rd_en_q) begin
        if (!m1_rd_buf_sel_q) begin
          m1_raw_word <= mem0[m1_rd_addr_q];
          m1_raw_keep <= keep0[m1_rd_addr_q];
        end
        else begin
          m1_raw_word <= mem1[m1_rd_addr_q];
          m1_raw_keep <= keep1[m1_rd_addr_q];
        end

        for (i = 0; i < PF_MAX; i++) begin
          src_lane = m1_rd_base_lane_q + i;
          if (src_lane < WORD_LANES) begin
            m1_rd_data[i*DATA_W +: DATA_W] <= ( !m1_rd_buf_sel_q ? mem0[m1_rd_addr_q][src_lane*DATA_W +: DATA_W]
                                                              : mem1[m1_rd_addr_q][src_lane*DATA_W +: DATA_W] );
            m1_rd_keep[i]                  <= ( !m1_rd_buf_sel_q ? keep0[m1_rd_addr_q][src_lane]
                                                              : keep1[m1_rd_addr_q][src_lane] );
          end
          else begin
            m1_rd_data[i*DATA_W +: DATA_W] <= '0;
            m1_rd_keep[i]                  <= 1'b0;
          end
        end
      end
    end
  end

endmodule
