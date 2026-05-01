module cnn_dma_to_axi_bridge_kv260 #(
    // cnn_dma_direct side
    parameter int DDR_ADDR_W          = 20,
    parameter int DDR_WORD_W          = 64,

    // AXI side
    parameter int AXI_ADDR_W          = 40,
    parameter int AXI_DATA_W          = DDR_WORD_W,
    parameter int AXI_ID_W            = 1,

    // Base BYTE address in PS DDR that corresponds to cnn_dma_direct word address 0.
    // Example: reserve a contiguous DDR window in software/device-tree and point this base to it.
    parameter logic [AXI_ADDR_W-1:0] AXI_DDR_BASE_ADDR = '0,

    // Write queue depth. Needed because cnn_dma_direct has no write-side backpressure.
    parameter int WR_FIFO_DEPTH       = 16
) (
    input  logic clk,
    input  logic rst_n,

    //==================================================
    // cnn_dma_direct "DDR direct" side
    // Address unit here is 1 DDR word, not byte.
    //==================================================
    input  logic                      ddr_rd_req,
    input  logic [DDR_ADDR_W-1:0]     ddr_rd_addr,
    output logic                      ddr_rd_valid,
    output logic [DDR_WORD_W-1:0]     ddr_rd_data,

    input  logic                      ddr_wr_en,
    input  logic [DDR_ADDR_W-1:0]     ddr_wr_addr,
    input  logic [DDR_WORD_W-1:0]     ddr_wr_data,
    input  logic [(DDR_WORD_W/8)-1:0] ddr_wr_be,

    //==================================================
    // Status
    //==================================================
    output logic                      busy,
    output logic                      error,
    output logic                      wr_fifo_overflow,

    //==================================================
    // AXI4 master side
    // Single-beat INCR accesses, suitable for PS HP/HPC ports.
    //==================================================
    output logic [AXI_ID_W-1:0]       m_axi_awid,
    output logic [AXI_ADDR_W-1:0]     m_axi_awaddr,
    output logic [7:0]                m_axi_awlen,
    output logic [2:0]                m_axi_awsize,
    output logic [1:0]                m_axi_awburst,
    output logic                      m_axi_awlock,
    output logic [3:0]                m_axi_awcache,
    output logic [2:0]                m_axi_awprot,
    output logic [3:0]                m_axi_awqos,
    output logic [3:0]                m_axi_awregion,
    output logic                      m_axi_awvalid,
    input  logic                      m_axi_awready,

    output logic [AXI_DATA_W-1:0]     m_axi_wdata,
    output logic [(AXI_DATA_W/8)-1:0] m_axi_wstrb,
    output logic                      m_axi_wlast,
    output logic                      m_axi_wvalid,
    input  logic                      m_axi_wready,

    input  logic [AXI_ID_W-1:0]       m_axi_bid,
    input  logic [1:0]                m_axi_bresp,
    input  logic                      m_axi_bvalid,
    output logic                      m_axi_bready,

    output logic [AXI_ID_W-1:0]       m_axi_arid,
    output logic [AXI_ADDR_W-1:0]     m_axi_araddr,
    output logic [7:0]                m_axi_arlen,
    output logic [2:0]                m_axi_arsize,
    output logic [1:0]                m_axi_arburst,
    output logic                      m_axi_arlock,
    output logic [3:0]                m_axi_arcache,
    output logic [2:0]                m_axi_arprot,
    output logic [3:0]                m_axi_arqos,
    output logic [3:0]                m_axi_arregion,
    output logic                      m_axi_arvalid,
    input  logic                      m_axi_arready,

    input  logic [AXI_ID_W-1:0]       m_axi_rid,
    input  logic [AXI_DATA_W-1:0]     m_axi_rdata,
    input  logic [1:0]                m_axi_rresp,
    input  logic                      m_axi_rlast,
    input  logic                      m_axi_rvalid,
    output logic                      m_axi_rready
);

    localparam int DDR_WORD_BYTES = DDR_WORD_W / 8;
    localparam int AXI_DATA_BYTES = AXI_DATA_W / 8;
    localparam int AXI_SIZE_W     = (AXI_DATA_BYTES <= 1) ? 1 : $clog2(AXI_DATA_BYTES);
    localparam int WR_FIFO_W      = (WR_FIFO_DEPTH <= 1) ? 1 : $clog2(WR_FIFO_DEPTH);
    localparam int WR_CNT_W       = $clog2(WR_FIFO_DEPTH + 1);
    localparam int OUTST_W        = $clog2(WR_FIFO_DEPTH + 1);

    typedef logic [AXI_ADDR_W-1:0] axi_addr_t;

    // Synthesis-time assumptions for the simple bridge.
    initial begin
        if (AXI_DATA_W != DDR_WORD_W)
            $error("cnn_dma_to_axi_bridge_kv260: AXI_DATA_W must equal DDR_WORD_W for this simple bridge. Use SmartConnect width conversion or re-parameterize the DMA word width.");
        if ((DDR_WORD_W % 8) != 0)
            $error("cnn_dma_to_axi_bridge_kv260: DDR_WORD_W must be byte-aligned.");
    end

    function automatic axi_addr_t word_to_byte_addr(input logic [DDR_ADDR_W-1:0] word_addr);
        axi_addr_t tmp;
        begin
            tmp = AXI_DDR_BASE_ADDR + (axi_addr_t'(word_addr) << AXI_SIZE_W);
            return tmp;
        end
    endfunction

    //==================================================
    // Read path: one outstanding read at a time.
    // This matches cnn_dma_direct, which waits for ddr_rd_valid.
    //==================================================
    logic                  rd_cmd_active_q, rd_cmd_active_n;
    logic                  rd_wait_data_q,  rd_wait_data_n;
    axi_addr_t             rd_addr_q,       rd_addr_n;

    //==================================================
    // Write path: queue requests because cnn_dma_direct has no write backpressure.
    //==================================================
    axi_addr_t             wr_fifo_addr_q [0:WR_FIFO_DEPTH-1];
    logic [AXI_DATA_W-1:0] wr_fifo_data_q [0:WR_FIFO_DEPTH-1];
    logic [(AXI_DATA_W/8)-1:0] wr_fifo_strb_q [0:WR_FIFO_DEPTH-1];

    logic [WR_FIFO_W-1:0]  wr_head_q, wr_head_n;
    logic [WR_FIFO_W-1:0]  wr_tail_q, wr_tail_n;
    logic [WR_CNT_W-1:0]   wr_count_q, wr_count_n;

    logic                  wr_issue_active_q, wr_issue_active_n;
    logic                  wr_aw_sent_q, wr_aw_sent_n;
    logic                  wr_w_sent_q,  wr_w_sent_n;
    axi_addr_t             wr_issue_addr_q, wr_issue_addr_n;
    logic [AXI_DATA_W-1:0] wr_issue_data_q, wr_issue_data_n;
    logic [(AXI_DATA_W/8)-1:0] wr_issue_strb_q, wr_issue_strb_n;

    logic [OUTST_W-1:0]    wr_outstanding_q, wr_outstanding_n;

    logic                  error_q, error_n;
    logic                  wr_fifo_overflow_q, wr_fifo_overflow_n;

    integer i;

    //==================================================
    // AXI defaults and combinational control
    //==================================================
    always_comb begin
        // defaults
        ddr_rd_valid        = 1'b0;
        ddr_rd_data         = m_axi_rdata;

        rd_cmd_active_n     = rd_cmd_active_q;
        rd_wait_data_n      = rd_wait_data_q;
        rd_addr_n           = rd_addr_q;

        wr_head_n           = wr_head_q;
        wr_tail_n           = wr_tail_q;
        wr_count_n          = wr_count_q;
        wr_issue_active_n   = wr_issue_active_q;
        wr_aw_sent_n        = wr_aw_sent_q;
        wr_w_sent_n         = wr_w_sent_q;
        wr_issue_addr_n     = wr_issue_addr_q;
        wr_issue_data_n     = wr_issue_data_q;
        wr_issue_strb_n     = wr_issue_strb_q;
        wr_outstanding_n    = wr_outstanding_q;

        error_n             = error_q;
        wr_fifo_overflow_n  = wr_fifo_overflow_q;

        // ---------------- AXI static fields ----------------
        m_axi_awid          = '0;
        m_axi_awlen         = 8'd0;
        m_axi_awsize        = AXI_SIZE_W[2:0];
        m_axi_awburst       = 2'b01; // INCR
        m_axi_awlock        = 1'b0;
        m_axi_awcache       = 4'b0000;
        m_axi_awprot        = 3'b000;
        m_axi_awqos         = 4'b0000;
        m_axi_awregion      = 4'b0000;

        m_axi_arid          = '0;
        m_axi_arlen         = 8'd0;
        m_axi_arsize        = AXI_SIZE_W[2:0];
        m_axi_arburst       = 2'b01; // INCR
        m_axi_arlock        = 1'b0;
        m_axi_arcache       = 4'b0000;
        m_axi_arprot        = 3'b000;
        m_axi_arqos         = 4'b0000;
        m_axi_arregion      = 4'b0000;

        // ---------------- Read channel ----------------
        m_axi_araddr        = rd_addr_q;
        m_axi_arvalid       = rd_cmd_active_q && !rd_wait_data_q;
        m_axi_rready        = rd_wait_data_q;

        if (!rd_cmd_active_q && ddr_rd_req) begin
            rd_addr_n       = word_to_byte_addr(ddr_rd_addr);
            rd_cmd_active_n = 1'b1;
            rd_wait_data_n  = 1'b0;
        end

        if (rd_cmd_active_q && !rd_wait_data_q && m_axi_arvalid && m_axi_arready) begin
            rd_wait_data_n = 1'b1;
        end

        if (rd_wait_data_q && m_axi_rvalid) begin
            ddr_rd_valid    = 1'b1;
            if (!m_axi_rlast)
                error_n     = 1'b1;
            if (m_axi_rresp != 2'b00)
                error_n     = 1'b1;
            rd_cmd_active_n = 1'b0;
            rd_wait_data_n  = 1'b0;
        end

        // ---------------- Write queue ingest ----------------
        if (ddr_wr_en) begin
            if (wr_count_q < WR_FIFO_DEPTH) begin
                // The array write itself is done in the sequential block.
                wr_tail_n  = wr_tail_q + 1'b1;
                wr_count_n = wr_count_q + 1'b1;
            end
            else begin
                wr_fifo_overflow_n = 1'b1;
                error_n            = 1'b1;
            end
        end

        // ---------------- Write issue engine ----------------
        if (!wr_issue_active_q && (wr_count_q != 0)) begin
            wr_issue_active_n = 1'b1;
            wr_aw_sent_n      = 1'b0;
            wr_w_sent_n       = 1'b0;
            wr_issue_addr_n   = wr_fifo_addr_q[wr_head_q];
            wr_issue_data_n   = wr_fifo_data_q[wr_head_q];
            wr_issue_strb_n   = wr_fifo_strb_q[wr_head_q];
        end

        m_axi_awaddr  = wr_issue_addr_q;
        m_axi_wdata   = wr_issue_data_q;
        m_axi_wstrb   = wr_issue_strb_q;
        m_axi_wlast   = 1'b1;
        m_axi_awvalid = wr_issue_active_q && !wr_aw_sent_q;
        m_axi_wvalid  = wr_issue_active_q && !wr_w_sent_q;
        m_axi_bready  = 1'b1;

        if (wr_issue_active_q && !wr_aw_sent_q && m_axi_awready)
            wr_aw_sent_n = 1'b1;

        if (wr_issue_active_q && !wr_w_sent_q && m_axi_wready)
            wr_w_sent_n  = 1'b1;

        if (wr_issue_active_q &&
            ((wr_aw_sent_q || m_axi_awready) && (wr_w_sent_q || m_axi_wready))) begin
            wr_issue_active_n = 1'b0;
            wr_aw_sent_n      = 1'b0;
            wr_w_sent_n       = 1'b0;
            wr_head_n         = wr_head_q + 1'b1;
            wr_count_n        = wr_count_n - 1'b1;
            wr_outstanding_n  = wr_outstanding_q + 1'b1;
        end

        if (m_axi_bvalid) begin
            if (m_axi_bresp != 2'b00)
                error_n = 1'b1;
            if (wr_outstanding_q != 0)
                wr_outstanding_n = wr_outstanding_q - 1'b1;
        end
    end

    //==================================================
    // Registers and write queue memory updates
    //==================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_cmd_active_q     <= 1'b0;
            rd_wait_data_q      <= 1'b0;
            rd_addr_q           <= '0;

            wr_head_q           <= '0;
            wr_tail_q           <= '0;
            wr_count_q          <= '0;
            wr_issue_active_q   <= 1'b0;
            wr_aw_sent_q        <= 1'b0;
            wr_w_sent_q         <= 1'b0;
            wr_issue_addr_q     <= '0;
            wr_issue_data_q     <= '0;
            wr_issue_strb_q     <= '0;
            wr_outstanding_q    <= '0;

            error_q             <= 1'b0;
            wr_fifo_overflow_q  <= 1'b0;

            for (i = 0; i < WR_FIFO_DEPTH; i++) begin
                wr_fifo_addr_q[i] <= '0;
                wr_fifo_data_q[i] <= '0;
                wr_fifo_strb_q[i] <= '0;
            end
        end
        else begin
            rd_cmd_active_q    <= rd_cmd_active_n;
            rd_wait_data_q     <= rd_wait_data_n;
            rd_addr_q          <= rd_addr_n;

            // enqueue write request into FIFO memory
            if (ddr_wr_en && (wr_count_q < WR_FIFO_DEPTH)) begin
                wr_fifo_addr_q[wr_tail_q] <= word_to_byte_addr(ddr_wr_addr);
                wr_fifo_data_q[wr_tail_q] <= ddr_wr_data;
                wr_fifo_strb_q[wr_tail_q] <= ddr_wr_be;
            end

            wr_head_q          <= wr_head_n;
            wr_tail_q          <= wr_tail_n;
            wr_count_q         <= wr_count_n;
            wr_issue_active_q  <= wr_issue_active_n;
            wr_aw_sent_q       <= wr_aw_sent_n;
            wr_w_sent_q        <= wr_w_sent_n;
            wr_issue_addr_q    <= wr_issue_addr_n;
            wr_issue_data_q    <= wr_issue_data_n;
            wr_issue_strb_q    <= wr_issue_strb_n;
            wr_outstanding_q   <= wr_outstanding_n;

            error_q            <= error_n;
            wr_fifo_overflow_q <= wr_fifo_overflow_n;
        end
    end

    assign busy             = rd_cmd_active_q || rd_wait_data_q ||
                              (wr_count_q != 0) || wr_issue_active_q || (wr_outstanding_q != 0);
    assign error            = error_q;
    assign wr_fifo_overflow = wr_fifo_overflow_q;

endmodule
