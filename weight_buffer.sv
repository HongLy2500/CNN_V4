// weight_buffer.sv
// BRAM-friendly rewrite of stable weight_buffer.
//
// Key synthesis fixes:
//   - Remove large data/keep memories from the main control always block.
//   - Use small RAM wrapper modules with clean synchronous read/write ports.
//   - Duplicate the storage per read interface (raw/mode2 and mode1) so the
//     existing interface can support independent raw and mode1 read requests
//     without requiring a multi-read-port RAM.
//   - Do not reset RAM contents. Only ready/status/request registers are reset.

module weight_data_bram_sdp #(
  parameter int DATA_W     = 8,
  parameter int WORD_LANES = 16,
  parameter int ADDR_W     = 12,
  parameter int DEPTH      = 512
)(
  input  logic clk,

  input  logic                         wr_en,
  input  logic [ADDR_W-1:0]            wr_addr,
  input  logic [WORD_LANES*DATA_W-1:0] wr_data,
  input  logic [WORD_LANES-1:0]        wr_keep,

  input  logic                         rd_en,
  input  logic [ADDR_W-1:0]            rd_addr,
  output logic [WORD_LANES*DATA_W-1:0] rd_data
);
  localparam int WORD_W = WORD_LANES * DATA_W;

  (* ram_style = "block" *)
  logic [WORD_W-1:0] ram [0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (wr_en) begin
      for (int lane = 0; lane < WORD_LANES; lane++) begin
        if (wr_keep[lane]) begin
          ram[wr_addr][lane*DATA_W +: DATA_W] <= wr_data[lane*DATA_W +: DATA_W];
        end
      end
    end

    if (rd_en) begin
      rd_data <= ram[rd_addr];
    end
  end
endmodule


module weight_keep_bram_sdp #(
  parameter int WORD_LANES = 16,
  parameter int ADDR_W     = 12,
  parameter int DEPTH      = 512
)(
  input  logic clk,

  input  logic                  wr_en,
  input  logic [ADDR_W-1:0]     wr_addr,
  input  logic [WORD_LANES-1:0] wr_data,

  input  logic                  rd_en,
  input  logic [ADDR_W-1:0]     rd_addr,
  output logic [WORD_LANES-1:0] rd_data
);
  (* ram_style = "block" *)
  logic [WORD_LANES-1:0] ram [0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (wr_en) begin
      ram[wr_addr] <= wr_data;
    end

    if (rd_en) begin
      rd_data <= ram[rd_addr];
    end
  end
endmodule


module weight_buffer #(
  parameter int DATA_W     = 8,
  // Physical storage width of one weight-buffer word.
  // In the integrated design this should be PTOTAL.
  parameter int WORD_LANES = 16,
  // Logical read width for mode 1.
  parameter int PF_MAX     = 64,
  parameter int ADDR_W     = 12,
  parameter int DEPTH      = 512
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

  assign dma_wr_ready = 1'b1;

  // -----------------------------------------------------
  // RAM write command decode
  // -----------------------------------------------------
  logic wr_bank0;
  logic wr_bank1;

  assign wr_bank0 = dma_wr_en && !dma_wr_buf_sel;
  assign wr_bank1 = dma_wr_en &&  dma_wr_buf_sel;

  // -----------------------------------------------------
  // Registered read requests.
  // These align valid and selected buffer/base-lane with the 1-cycle
  // synchronous RAM outputs.
  // -----------------------------------------------------
  logic                    rd_en_q;
  logic                    rd_buf_sel_q;
  logic [ADDR_W-1:0]       rd_addr_q;

  logic                    m1_rd_en_q;
  logic                    m1_rd_buf_sel_q;
  logic [ADDR_W-1:0]       m1_rd_addr_q;
  logic [BASE_W-1:0]       m1_rd_base_lane_q;

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

  assign rd_valid    = rd_en_q;
  assign m1_rd_valid = m1_rd_en_q;

  // -----------------------------------------------------
  // BRAM storage
  //
  // The stable version allowed raw/mode2 and mode1 read ports to be active
  // independently. A single simple-dual-port BRAM has one read port, so this
  // version duplicates the payload and keep memories for the two read
  // interfaces. DMA writes update both copies, preserving behavior while
  // keeping each RAM instance template-like for Vivado.
  // -----------------------------------------------------
  logic [WORD_W-1:0]     raw_data0, raw_data1;
  logic [WORD_W-1:0]     m1_data0,  m1_data1;
  logic [WORD_LANES-1:0] raw_keep0, raw_keep1;
  logic [WORD_LANES-1:0] m1_keep0,  m1_keep1;

  weight_data_bram_sdp #(
    .DATA_W(DATA_W),
    .WORD_LANES(WORD_LANES),
    .ADDR_W(ADDR_W),
    .DEPTH(DEPTH)
  ) u_raw_data0 (
    .clk(clk),
    .wr_en(wr_bank0),
    .wr_addr(dma_wr_addr),
    .wr_data(dma_wr_data),
    .wr_keep(dma_wr_keep),
    .rd_en(rd_en && !rd_buf_sel),
    .rd_addr(rd_addr),
    .rd_data(raw_data0)
  );

  weight_data_bram_sdp #(
    .DATA_W(DATA_W),
    .WORD_LANES(WORD_LANES),
    .ADDR_W(ADDR_W),
    .DEPTH(DEPTH)
  ) u_raw_data1 (
    .clk(clk),
    .wr_en(wr_bank1),
    .wr_addr(dma_wr_addr),
    .wr_data(dma_wr_data),
    .wr_keep(dma_wr_keep),
    .rd_en(rd_en && rd_buf_sel),
    .rd_addr(rd_addr),
    .rd_data(raw_data1)
  );

  weight_data_bram_sdp #(
    .DATA_W(DATA_W),
    .WORD_LANES(WORD_LANES),
    .ADDR_W(ADDR_W),
    .DEPTH(DEPTH)
  ) u_m1_data0 (
    .clk(clk),
    .wr_en(wr_bank0),
    .wr_addr(dma_wr_addr),
    .wr_data(dma_wr_data),
    .wr_keep(dma_wr_keep),
    .rd_en(m1_rd_en && !m1_rd_buf_sel),
    .rd_addr(m1_rd_addr),
    .rd_data(m1_data0)
  );

  weight_data_bram_sdp #(
    .DATA_W(DATA_W),
    .WORD_LANES(WORD_LANES),
    .ADDR_W(ADDR_W),
    .DEPTH(DEPTH)
  ) u_m1_data1 (
    .clk(clk),
    .wr_en(wr_bank1),
    .wr_addr(dma_wr_addr),
    .wr_data(dma_wr_data),
    .wr_keep(dma_wr_keep),
    .rd_en(m1_rd_en && m1_rd_buf_sel),
    .rd_addr(m1_rd_addr),
    .rd_data(m1_data1)
  );

  weight_keep_bram_sdp #(
    .WORD_LANES(WORD_LANES),
    .ADDR_W(ADDR_W),
    .DEPTH(DEPTH)
  ) u_raw_keep0 (
    .clk(clk),
    .wr_en(wr_bank0),
    .wr_addr(dma_wr_addr),
    .wr_data(dma_wr_keep),
    .rd_en(rd_en && !rd_buf_sel),
    .rd_addr(rd_addr),
    .rd_data(raw_keep0)
  );

  weight_keep_bram_sdp #(
    .WORD_LANES(WORD_LANES),
    .ADDR_W(ADDR_W),
    .DEPTH(DEPTH)
  ) u_raw_keep1 (
    .clk(clk),
    .wr_en(wr_bank1),
    .wr_addr(dma_wr_addr),
    .wr_data(dma_wr_keep),
    .rd_en(rd_en && rd_buf_sel),
    .rd_addr(rd_addr),
    .rd_data(raw_keep1)
  );

  weight_keep_bram_sdp #(
    .WORD_LANES(WORD_LANES),
    .ADDR_W(ADDR_W),
    .DEPTH(DEPTH)
  ) u_m1_keep0 (
    .clk(clk),
    .wr_en(wr_bank0),
    .wr_addr(dma_wr_addr),
    .wr_data(dma_wr_keep),
    .rd_en(m1_rd_en && !m1_rd_buf_sel),
    .rd_addr(m1_rd_addr),
    .rd_data(m1_keep0)
  );

  weight_keep_bram_sdp #(
    .WORD_LANES(WORD_LANES),
    .ADDR_W(ADDR_W),
    .DEPTH(DEPTH)
  ) u_m1_keep1 (
    .clk(clk),
    .wr_en(wr_bank1),
    .wr_addr(dma_wr_addr),
    .wr_data(dma_wr_keep),
    .rd_en(m1_rd_en && m1_rd_buf_sel),
    .rd_addr(m1_rd_addr),
    .rd_data(m1_keep1)
  );

  // -----------------------------------------------------
  // Ready-state management.
  // RAM contents are intentionally not reset.
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
  // Raw/full-word read outputs
  // -----------------------------------------------------
  assign rd_data = rd_buf_sel_q ? raw_data1 : raw_data0;
  assign rd_keep = rd_buf_sel_q ? raw_keep1 : raw_keep0;

  // -----------------------------------------------------
  // Mode-1 logical read packing
  // -----------------------------------------------------
  logic [WORD_W-1:0]     m1_sel_word;
  logic [WORD_LANES-1:0] m1_sel_keep;

  assign m1_sel_word = m1_rd_buf_sel_q ? m1_data1 : m1_data0;
  assign m1_sel_keep = m1_rd_buf_sel_q ? m1_keep1 : m1_keep0;

  always_comb begin
    m1_rd_data = '0;
    m1_rd_keep = '0;

    for (int out_lane = 0; out_lane < PF_MAX; out_lane++) begin
      int src_lane;
      src_lane = int'(m1_rd_base_lane_q) + out_lane;

      if (src_lane < WORD_LANES) begin
        m1_rd_data[out_lane*DATA_W +: DATA_W] = m1_sel_word[src_lane*DATA_W +: DATA_W];
        m1_rd_keep[out_lane]                  = m1_sel_keep[src_lane];
      end
    end
  end

endmodule
