module ifm_buffer #(
    parameter int DATA_W = 8,
    parameter int PV_MAX = 8,
    parameter int PC     = 8,    // fixed for mode 2, also WT = PC
    parameter int C_MAX  = 64,
    parameter int W_MAX  = 224,
    parameter int H_MAX  = 224,
    parameter int HT     = 8,    // fixed tile height for mode 1
    parameter int DEPTH  = (((HT * W_MAX) > H_MAX) ? (HT * W_MAX) : H_MAX)
)(
    input  logic clk,
    input  logic rst_n,

    //==================================================
    // Configuration
    // cfg_mode = 0: mode 1
    // cfg_mode = 1: mode 2
    //==================================================
    input  logic                        cfg_load,
    input  logic                        cfg_mode,
    input  logic [$clog2(W_MAX+1)-1:0]  cfg_w_in,
    input  logic [$clog2(H_MAX+1)-1:0]  cfg_h_in,
    input  logic [$clog2(C_MAX+1)-1:0]  cfg_c_in,
    input  logic [$clog2(PV_MAX+1)-1:0] cfg_pv_cur,

    //==================================================
    // Mode 1 row sliding control
    //==================================================
    input  logic                        m1_advance_row,

    //==================================================
    // DMA write port
    // mode 1:
    //   dma_wr_bank    = channel index
    //   dma_wr_row_idx = logical row in tile [0..HT-1]
    //   dma_wr_col_idx = horizontal group index based on cfg_pv_cur
    //   dma_wr_data    = low cfg_pv_cur lanes hold valid pixels
    //
    // mode 2:
    //   dma_wr_bank    = channel index
    //   dma_wr_row_idx = absolute row index [0..H-1]
    //   dma_wr_col_idx = unused
    //   dma_wr_data    = low PC lanes hold one horizontal segment of that row
    //==================================================
    input  logic                        dma_wr_en,
    input  logic [$clog2(C_MAX)-1:0]    dma_wr_bank,
    input  logic [$clog2(H_MAX)-1:0]    dma_wr_row_idx,
    input  logic [$clog2(W_MAX)-1:0]    dma_wr_col_idx,
    input  logic [PV_MAX*DATA_W-1:0]    dma_wr_data,
    input  logic [PV_MAX-1:0]           dma_wr_keep,

    //==================================================
    // OFM write port
    //
    // Mode 1 contract used by the same-mode refill path:
    //   ofm_wr_row_idx = physical/free row slot inside the HT ring
    //   ofm_wr_col_idx = horizontal group index in the stored IFM layout
    //
    // Mode 2 contract stays aligned with DMA write semantics:
    //   ofm_wr_bank    = absolute channel index
    //   ofm_wr_row_idx = absolute row index [0..H-1]
    //   ofm_wr_col_idx = unused (single PC-wide segment per active tile)
    //==================================================
    input  logic                        ofm_wr_en,
    input  logic [$clog2(C_MAX)-1:0]    ofm_wr_bank,
    input  logic [$clog2(H_MAX)-1:0]    ofm_wr_row_idx,
    input  logic [$clog2(W_MAX)-1:0]    ofm_wr_col_idx,
    input  logic [PV_MAX*DATA_W-1:0]    ofm_wr_data,
    input  logic [PV_MAX-1:0]           ofm_wr_keep,

    //==================================================
    // Read port to data_register
    // mode 1:
    //   rd_bank_base = channel index
    //   rd_row_idx   = logical row in current HT window
    //   rd_col_idx   = horizontal group index based on cfg_pv_cur
    //
    // mode 2:
    //   rd_bank_base = first channel index of current Pc group
    //   rd_row_idx   = absolute row index
    //   rd_col_idx   = pixel select inside stored PC-wide segment
    //==================================================
    input  logic                        rd_en,
    input  logic [$clog2(C_MAX)-1:0]    rd_bank_base,
    input  logic [$clog2(H_MAX)-1:0]    rd_row_idx,
    input  logic [$clog2(W_MAX)-1:0]    rd_col_idx,

    output logic                        rd_valid,
    output logic [PV_MAX*DATA_W-1:0]    rd_data,

    //==================================================
    // Status / functional visibility
    //
    // UPDATED CONTRACT:
    // - dbg_m1_row_base is kept for backward compatibility.
    // - m1_row_base_l is the functional alias of the current mode-1 ring-row base.
    // - m1_free_valid/m1_free_row_* expose the row-slot that becomes free when
    //   m1_advance_row is accepted. This is the minimal functional contract that
    //   control/refill logic can safely use from ifm_buffer today.
    //
    // NOTE:
    // - mode 2 free-token generation is NOT synthesized here, because this
    //   buffer alone does not know when a spatial/channel block has truly been
    //   consumed by the mode-2 compute path. That contract must come from the
    //   local dataflow / compute side.
    //==================================================
    output logic                        dma_wr_ready,
    output logic                        ofm_wr_ready,
    output logic [$clog2(HT)-1:0]       dbg_m1_row_base,

    output logic [$clog2(HT)-1:0]       m1_row_base_l,
    output logic                        m1_free_valid,
    output logic [$clog2(HT)-1:0]       m1_free_row_slot_l,
    output logic [$clog2(H_MAX+1)-1:0]  m1_free_row_g
);

    localparam int WORD_W    = PV_MAX * DATA_W;
    localparam int DEPTH_W   = (DEPTH <= 1) ? 1 : $clog2(DEPTH);
    localparam int COL_W     = (W_MAX <= 1) ? 1 : $clog2(W_MAX);
    localparam int ROWBASE_W = (HT <= 1) ? 1 : $clog2(HT);
    localparam int HCFG_W    = $clog2(H_MAX+1);
    localparam int M1_STRIDE = W_MAX;

    //==================================================
    // Physical storage
    // C_MAX banks, each address width = PV_MAX * DATA_W
    //==================================================
    (* ram_style = "block" *)
    logic [WORD_W-1:0] mem [0:C_MAX-1][0:DEPTH-1];

    //==================================================
    // Latched configuration
    //==================================================
    logic                        cfg_mode_q;
    logic [$clog2(W_MAX+1)-1:0]  cfg_w_in_q;
    logic [$clog2(H_MAX+1)-1:0]  cfg_h_in_q;
    logic [$clog2(C_MAX+1)-1:0]  cfg_c_in_q;
    logic [$clog2(PV_MAX+1)-1:0] cfg_pv_cur_q;
    logic [$clog2(W_MAX+1)-1:0]  m1_words_per_row_q;
    logic [ROWBASE_W-1:0]        m1_row_base_q;

    // Global row index of the mode-1 row currently at logical slot 0.
    // This lets us expose which physical slot becomes free on each advance.
    logic [HCFG_W-1:0]           m1_row_base_g_q;

    // Free-row event registers (1-cycle pulse)
    logic                        m1_free_valid_q;
    logic [ROWBASE_W-1:0]        m1_free_row_slot_l_q;
    logic [HCFG_W-1:0]           m1_free_row_g_q;

    //==================================================
    // Unified selected write request
    //==================================================
    logic                     wr_en_sel;
    logic [$clog2(C_MAX)-1:0] wr_bank_sel;
    logic [$clog2(H_MAX)-1:0] wr_row_idx_sel;
    logic [COL_W-1:0]         wr_col_idx_sel;
    logic [WORD_W-1:0]        wr_data_sel;
    logic [PV_MAX-1:0]        wr_keep_sel;

    //==================================================
    // Derived addresses
    //==================================================
    logic [ROWBASE_W-1:0] wr_m1_phys_row_dma;
    logic [ROWBASE_W-1:0] wr_m1_phys_row_ofm;
    logic [ROWBASE_W-1:0] wr_m1_phys_row;
    logic [DEPTH_W-1:0]   wr_addr;
    logic [DEPTH_W-1:0]   rd_addr_m1;
    logic [DEPTH_W-1:0]   rd_addr_m2;
    logic [DEPTH_W-1:0]   wr_row_idx_ext;
    logic [DEPTH_W-1:0]   rd_row_idx_ext;
    logic                 wr_addr_valid;
    logic                 wr_src_is_ofm;

    //==================================================
    // Read registers
    //==================================================
    logic              rd_valid_q;
    logic [WORD_W-1:0] rd_data_q;

    assign rd_valid          = rd_valid_q;
    assign rd_data           = rd_data_q;
    assign dma_wr_ready      = ~ofm_wr_en;
    assign ofm_wr_ready      = ~dma_wr_en;
    assign dbg_m1_row_base   = m1_row_base_q;

    assign m1_row_base_l     = m1_row_base_q;
    assign m1_free_valid     = m1_free_valid_q;
    assign m1_free_row_slot_l= m1_free_row_slot_l_q;
    assign m1_free_row_g     = m1_free_row_g_q;

    //==================================================
    // Width-safe row index extension
    //==================================================
    // Vivado does not allow part-selects wider than the source signal.
    // With smoke-test parameters, e.g. H_MAX=4 and DEPTH=16,
    // wr_row_idx_sel/rd_row_idx are 2-bit signals while DEPTH_W is 4.
    // Assigning into a DEPTH_W-wide temporary zero-extends safely.
    always_comb begin
        wr_row_idx_ext = '0;
        rd_row_idx_ext = '0;
        wr_row_idx_ext = wr_row_idx_sel;
        rd_row_idx_ext = rd_row_idx;
    end

    //==================================================
    // Compile-time assumptions
    //==================================================
    initial begin
        if (PC > PV_MAX) begin
            $error("ifm_buffer: PC must be <= PV_MAX");
        end
        if (DEPTH < (HT * W_MAX)) begin
            $error("ifm_buffer: DEPTH too small for mode 1 worst-case Pv=1 mapping");
        end
        if (DEPTH < H_MAX) begin
            $error("ifm_buffer: DEPTH too small for mode 2 max row storage");
        end
    end

    function automatic [$clog2(W_MAX+1)-1:0] ceil_div_w(
        input [$clog2(W_MAX+1)-1:0] a,
        input [$clog2(PV_MAX+1)-1:0] b
    );
        logic [$clog2(W_MAX+PV_MAX+1)-1:0] tmp;
        begin
            if (b == 0)
                ceil_div_w = a;
            else begin
                tmp = a + b - 1'b1;
                ceil_div_w = tmp / b;
            end
        end
    endfunction

    //==================================================
    // Config registers + mode 1 ring-row base
    //==================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cfg_mode_q         <= 1'b0;
            cfg_w_in_q         <= '0;
            cfg_h_in_q         <= '0;
            cfg_c_in_q         <= '0;
            cfg_pv_cur_q       <= '0;
            m1_words_per_row_q <= '0;
            m1_row_base_q      <= '0;
            m1_row_base_g_q    <= '0;

            m1_free_valid_q       <= 1'b0;
            m1_free_row_slot_l_q  <= '0;
            m1_free_row_g_q       <= '0;
        end
        else begin
            // default: free event is a 1-cycle pulse
            m1_free_valid_q <= 1'b0;

            if (cfg_load) begin
                cfg_mode_q         <= cfg_mode;
                cfg_w_in_q         <= cfg_w_in;
                cfg_h_in_q         <= cfg_h_in;
                cfg_c_in_q         <= cfg_c_in;
                cfg_pv_cur_q       <= cfg_pv_cur;
                m1_words_per_row_q <= ceil_div_w(cfg_w_in, (cfg_pv_cur == 0) ? 'd1 : cfg_pv_cur);
                m1_row_base_q      <= '0;
                m1_row_base_g_q    <= '0;

                m1_free_row_slot_l_q <= '0;
                m1_free_row_g_q      <= '0;
            end
            else if (!cfg_mode_q && m1_advance_row) begin
                // The row currently at logical slot 0 becomes free.
                m1_free_valid_q      <= 1'b1;
                m1_free_row_slot_l_q <= m1_row_base_q;
                m1_free_row_g_q      <= m1_row_base_g_q;

                if (m1_row_base_q == HT-1)
                    m1_row_base_q <= '0;
                else
                    m1_row_base_q <= m1_row_base_q + 1'b1;

                // Advance the GLOBAL row attached to logical slot 0.
                // Saturate at cfg_h_in_q-1 if control keeps toggling beyond the
                // legal window progression.
                if (cfg_h_in_q != '0) begin
                    if (m1_row_base_g_q < (cfg_h_in_q - 1'b1))
                        m1_row_base_g_q <= m1_row_base_g_q + 1'b1;
                    else
                        m1_row_base_g_q <= m1_row_base_g_q;
                end
                else begin
                    m1_row_base_g_q <= '0;
                end
            end
        end
    end

    //==================================================
    // Write-source arbitration
    //==================================================
    always_comb begin
        wr_en_sel      = 1'b0;
        wr_src_is_ofm  = 1'b0;
        wr_bank_sel    = '0;
        wr_row_idx_sel = '0;
        wr_col_idx_sel = '0;
        wr_data_sel    = '0;
        wr_keep_sel    = '0;

        if (dma_wr_en) begin
            wr_en_sel      = 1'b1;
            wr_bank_sel    = dma_wr_bank;
            wr_row_idx_sel = dma_wr_row_idx;
            wr_col_idx_sel = dma_wr_col_idx;
            wr_data_sel    = dma_wr_data;
            wr_keep_sel    = dma_wr_keep;
        end
        else if (ofm_wr_en) begin
            wr_en_sel      = 1'b1;
            wr_src_is_ofm  = 1'b1;
            wr_bank_sel    = ofm_wr_bank;
            wr_row_idx_sel = ofm_wr_row_idx;
            wr_col_idx_sel = ofm_wr_col_idx;
            wr_data_sel    = ofm_wr_data;
            wr_keep_sel    = ofm_wr_keep;
        end
    end

    //==================================================
    // Helper: mode 1 write-row resolution
    // - DMA path still presents a logical row within the active HT window.
    // - OFM same-mode refill path presents the already-resolved physical/free
    //   row slot from control/refill logic.
    //
    // To keep same-mode OFM->IFM refill independent of the currently running
    // layer's Pv, mode-1 storage uses the fixed worst-case per-row stride W_MAX
    // that this RAM was already sized for.
    //==================================================
    always_comb begin
        logic [$clog2(HT+1)-1:0] tmp;

        tmp = m1_row_base_q + wr_row_idx_sel[ROWBASE_W-1:0];
        if (tmp >= HT)
            wr_m1_phys_row_dma = tmp - HT;
        else
            wr_m1_phys_row_dma = tmp[ROWBASE_W-1:0];

        wr_m1_phys_row_ofm = wr_row_idx_sel[ROWBASE_W-1:0];

        if (wr_src_is_ofm)
            wr_m1_phys_row = wr_m1_phys_row_ofm;
        else
            wr_m1_phys_row = wr_m1_phys_row_dma;
    end

    //==================================================
    // Write address mapping
    //==================================================
    always_comb begin
        logic wr_bank_valid_v;
        logic wr_row_valid_v;

        wr_addr          = '0;
        wr_addr_valid    = 1'b0;
        wr_bank_valid_v  = 1'b0;
        wr_row_valid_v   = 1'b0;

        // DMA writes are for the currently configured layer, so the
        // current cfg_c_in_q guard is still correct.
        //
        // OFM->IFM writes may be filling the NEXT layer before cfg_load
        // advances this buffer to that next layer. In that phase cfg_c_in_q
        // can still describe the source/current layer, so using cfg_c_in_q
        // would incorrectly drop valid next-layer banks, for example bank
        // 8..11 when current C=8 and next C=12.
        wr_bank_valid_v = wr_src_is_ofm
                        ? (wr_bank_sel < C_MAX)
                        : (wr_bank_sel < cfg_c_in_q);

        if (!cfg_mode_q) begin
            // Mode 1 OFM refill provides a physical/free row slot inside HT.
            // DMA preload/refill also presents a row inside the active HT
            // window, so the same row guard is valid for both sources.
            wr_row_valid_v = (wr_row_idx_sel < HT);

            if (wr_bank_valid_v &&
                wr_row_valid_v &&
                (wr_col_idx_sel < W_MAX)) begin
                wr_addr       = (wr_m1_phys_row * M1_STRIDE) + wr_col_idx_sel;
                wr_addr_valid = (wr_addr < DEPTH);
            end
        end
        else begin
            // Mode 2 DMA writes are for the current layer and remain guarded
            // by cfg_h_in_q. OFM writes may target the next layer before the
            // config latch advances, so only require the physical row to be
            // representable by this buffer.
            wr_row_valid_v = wr_src_is_ofm
                           ? (wr_row_idx_sel < H_MAX)
                           : (wr_row_idx_sel < cfg_h_in_q);

            if (wr_bank_valid_v && wr_row_valid_v) begin
                wr_addr       = wr_row_idx_ext;
                wr_addr_valid = (wr_addr < DEPTH);
            end
        end
    end

    //==================================================
    // Write path
    //==================================================
    integer wlane;
    always_ff @(posedge clk) begin
        if (wr_en_sel && wr_addr_valid) begin
            for (wlane = 0; wlane < PV_MAX; wlane++) begin
                if (wr_keep_sel[wlane]) begin
                    mem[wr_bank_sel][wr_addr][wlane*DATA_W +: DATA_W]
                        <= wr_data_sel[wlane*DATA_W +: DATA_W];
                end
            end
        end
    end

    //==================================================
    // Read address mapping
    //==================================================
    always_comb begin : GEN_RD_ADDR
        logic [ROWBASE_W-1:0] rd_m1_phys_row;
        logic [$clog2(HT+1)-1:0] tmp;

        tmp = m1_row_base_q + rd_row_idx[ROWBASE_W-1:0];
        if (tmp >= HT)
            rd_m1_phys_row = tmp - HT;
        else
            rd_m1_phys_row = tmp[ROWBASE_W-1:0];

        rd_addr_m1 = (rd_m1_phys_row * M1_STRIDE) + rd_col_idx;
        rd_addr_m2 = rd_row_idx_ext;
    end

    //==================================================
    // Read path: 1-cycle registered output
    //==================================================
    integer rlane;
    integer bank_idx_i;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_valid_q <= 1'b0;
            rd_data_q  <= '0;
        end
        else begin
            rd_valid_q <= rd_en;
            rd_data_q  <= '0;

            if (rd_en) begin
                if (!cfg_mode_q) begin
                    for (rlane = 0; rlane < PV_MAX; rlane++) begin
                        if (rlane < cfg_pv_cur_q) begin
                            rd_data_q[rlane*DATA_W +: DATA_W]
                                <= mem[rd_bank_base][rd_addr_m1][rlane*DATA_W +: DATA_W];
                        end
                        else begin
                            rd_data_q[rlane*DATA_W +: DATA_W] <= '0;
                        end
                    end
                end
                else begin
                    for (rlane = 0; rlane < PV_MAX; rlane++) begin
                        if ((rlane < PC) && ((rd_bank_base + rlane) < cfg_c_in_q) && (rd_col_idx < PC)) begin
                            bank_idx_i = rd_bank_base + rlane;
                            rd_data_q[rlane*DATA_W +: DATA_W]
                                <= mem[bank_idx_i[$clog2(C_MAX)-1:0]][rd_addr_m2][rd_col_idx*DATA_W +: DATA_W];
                        end
                        else begin
                            rd_data_q[rlane*DATA_W +: DATA_W] <= '0;
                        end
                    end
                end
            end
        end
    end

endmodule
