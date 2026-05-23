module ifm_data_bram_sdp #(
    parameter int DATA_W = 8,
    parameter int LANES  = 8,
    parameter int DEPTH  = 128,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
)(
    input  logic clk,

    input  logic                  wr_en,
    input  logic [ADDR_W-1:0]     wr_addr,
    input  logic [LANES-1:0]      wr_keep,
    input  logic [LANES*DATA_W-1:0] wr_data,

    input  logic                  rd_en,
    input  logic [ADDR_W-1:0]     rd_addr,
    output logic [LANES*DATA_W-1:0] rd_data
);

    (* ram_style = "block" *)
    logic [LANES*DATA_W-1:0] ram [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (wr_en) begin
            for (int lane = 0; lane < LANES; lane++) begin
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

module ifm_buffer #(
    parameter int DATA_W = 8,
    parameter int PV_MAX = 8,
    parameter int PC     = 16,    // fixed for mode 2, also WT = PC
    parameter int C_MAX  = 64,
    parameter int W_MAX  = 32,
    parameter int H_MAX  = 32,
    parameter int HT     = 4,    // fixed tile height for mode 1
    // DEPTH must cover both layouts:
    //   mode 1: HT rows * W_MAX words/row
    //   mode 2: H_MAX rows, because mode-2 folds cgrp into the bank index:
    //           bank = cgrp*PC + col_l, addr = row
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
    //   WT = PC. The resident IFM tile is a horizontal tile of width PC.
    //   dma_wr_bank    = local column col_l inside the resident W tile [0..PC-1]
    //   dma_wr_row_idx = absolute row index [0..H-1]
    //   dma_wr_col_idx = channel-group index cgrp = floor(channel/PC)
    //   dma_wr_data    = low PC lanes hold PC channel values at (row, col_l, cgrp)
    //   Internal Mode2 physical layout maps these to:
    //     bank = cgrp*PC + col_l, addr = row
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
    //   WT = PC. The resident IFM tile is a horizontal tile of width PC.
    //   ofm_wr_bank    = local column col_l inside the resident W tile [0..PC-1]
    //   ofm_wr_row_idx = absolute row index [0..H-1]
    //   ofm_wr_col_idx = channel-group index cgrp = floor(channel/PC)
    //   ofm_wr_data    = low PC lanes hold PC channel values at (row, col_l, cgrp)
    //   Internal Mode2 physical layout maps these to:
    //     bank = cgrp*PC + col_l, addr = row
    //==================================================
    input  logic                        ofm_wr_en,
    input  logic [$clog2(C_MAX)-1:0]    ofm_wr_bank,
    input  logic [$clog2(H_MAX)-1:0]    ofm_wr_row_idx,
    input  logic [$clog2(W_MAX)-1:0]    ofm_wr_col_idx,
    input  logic [PV_MAX*DATA_W-1:0]    ofm_wr_data,
    input  logic [PV_MAX-1:0]           ofm_wr_keep,
    // Mode2 OFM->IFM target metadata.  Only asserted/used for OFM stream
    // beats whose destination layout is IFM Mode2.  Mode1/DMA paths ignore it.
    input  logic [$clog2(W_MAX)-1:0]    ofm_wr_col_g,
    input  logic                        ofm_wr_mode2,

    //==================================================
    // Read port to data_register
    // mode 1:
    //   rd_bank_base = channel index
    //   rd_row_idx   = logical row in current HT window
    //   rd_col_idx   = horizontal group index based on cfg_pv_cur
    //
    // mode 2:
    //   rd_bank_base = first channel index of current PC group (cgrp*PC).
    //   rd_row_idx   = absolute row index
    //   rd_col_idx   = local column col_l inside the resident W tile [0..PC-1]
    //   rd_data      = PC channel lanes for the requested (row, col_l, cgrp)
    //   Internal Mode2 physical layout maps these to:
    //     bank = rd_bank_base + rd_col_idx, addr = row
    //==================================================
    input  logic                        rd_en,
    input  logic [$clog2(C_MAX)-1:0]    rd_bank_base,
    input  logic [$clog2(H_MAX)-1:0]    rd_row_idx,
    input  logic [$clog2(W_MAX)-1:0]    rd_col_idx,
    // Global IFM column requested by Mode2 addr_gen.  The physical read
    // column above remains the rolling slot (global_col % PC).
    input  logic [$clog2(W_MAX)-1:0]    rd_col_g,

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
    localparam int BANK_W    = (C_MAX <= 1) ? 1 : $clog2(C_MAX);
    localparam int COL_W     = (W_MAX <= 1) ? 1 : $clog2(W_MAX);
    localparam int ROWBASE_W = (HT <= 1) ? 1 : $clog2(HT);
    localparam int HCFG_W    = $clog2(H_MAX+1);
    localparam int M1_STRIDE = W_MAX;

    // Mode 2 storage contract:
    //   interface still presents:
    //     wr_bank/rd_col_idx = local column col_l inside resident Wt=PC
    //     wr_col_idx/rd_bank_base = channel group cgrp information
    //   physical storage is the finalized Mode2 layout:
    //     bank = cgrp * PC + col_l
    //     addr = row
    //     word lanes = PC channel values of that cgrp
    localparam int M2_CGRP_MAX = (C_MAX + PC - 1) / PC;
    localparam int M2_BANKS    = M2_CGRP_MAX * PC;

    //==================================================
    // Physical data storage
    // One synchronous BRAM-style RAM per logical bank.  The logical layout is
    // unchanged; only the storage implementation is made inference-friendly.
    //
    // Mode1:
    //   bank = channel, addr = row_slot*W_MAX + col_group, lane = Pv pixel lane
    // Mode2:
    //   bank = cgrp*PC + col_l, addr = row, lane = PC channel lane
    //==================================================
    logic                  data_wr_en   [0:C_MAX-1];
    logic [DEPTH_W-1:0]    data_wr_addr [0:C_MAX-1];
    logic [PV_MAX-1:0]     data_wr_keep [0:C_MAX-1];
    logic [WORD_W-1:0]     data_wr_data [0:C_MAX-1];

    logic                  data_rd_en   [0:C_MAX-1];
    logic [DEPTH_W-1:0]    data_rd_addr [0:C_MAX-1];
    logic [WORD_W-1:0]     data_rd_data [0:C_MAX-1];

    genvar ifm_bank_gen;
    generate
        for (ifm_bank_gen = 0; ifm_bank_gen < C_MAX; ifm_bank_gen++) begin : G_IFM_DATA_BRAM
            ifm_data_bram_sdp #(
                .DATA_W(DATA_W),
                .LANES (PV_MAX),
                .DEPTH (DEPTH),
                .ADDR_W(DEPTH_W)
            ) u_data_bram (
                .clk    (clk),
                .wr_en  (data_wr_en  [ifm_bank_gen]),
                .wr_addr(data_wr_addr[ifm_bank_gen]),
                .wr_keep(data_wr_keep[ifm_bank_gen]),
                .wr_data(data_wr_data[ifm_bank_gen]),
                .rd_en  (data_rd_en  [ifm_bank_gen]),
                .rd_addr(data_rd_addr[ifm_bank_gen]),
                .rd_data(data_rd_data[ifm_bank_gen])
            );
        end
    endgenerate

    // Mode2 rolling-slot content tag.  Mode1 never reads these tags.
    // Flattened 1D metadata avoids Vivado 3D/record-RAM runtime warnings.
    // Logical mapping is unchanged:
    //   meta_idx = bank * DEPTH + addr
    localparam int M2_SLOT_COUNT = C_MAX * DEPTH;
    localparam int M2_SLOT_AW    = (M2_SLOT_COUNT <= 1) ? 1 : $clog2(M2_SLOT_COUNT);

    logic                 m2_slot_valid_flat   [0:M2_SLOT_COUNT-1];
    logic [COL_W-1:0]     m2_slot_col_tag_flat [0:M2_SLOT_COUNT-1];

    logic [M2_SLOT_AW-1:0] m2_wr_slot_idx;
    logic [M2_SLOT_AW-1:0] m2_rd_slot_idx;
    logic                 m2_rd_slot_valid;

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
    logic [$clog2(C_MAX)-1:0] wr_bank_phys_sel;
    logic [DEPTH_W-1:0]   rd_addr_m1;
    logic [DEPTH_W-1:0]   rd_addr_m2;
    logic [DEPTH_W-1:0]   wr_row_idx_ext;
    logic [DEPTH_W-1:0]   rd_row_idx_ext;
    logic                 wr_addr_valid;
    logic                 wr_src_is_ofm;
    logic                 wr_use_mode2_layout;
    logic                 rd_m2_tag_hit;

    assign wr_use_mode2_layout = wr_src_is_ofm ? ofm_wr_mode2 : cfg_mode_q;

    // Mode 2 read decode helpers. Kept separate from mode 1 so the mode 1
    // address/read behavior remains unchanged.
    logic [31:0]          rd_m2_cgrp_u32;
    logic [31:0]          rd_m2_col_l_u32;
    logic [31:0]          rd_m2_bank_u32;
    logic [31:0]          rd_addr_m2_u32;

    //==================================================
    // Read pipeline registers
    //==================================================
    logic                 rd_req_valid;
    logic [BANK_W-1:0]    rd_req_bank;
    logic [DEPTH_W-1:0]   rd_req_addr;

    logic                 rd_pipe_valid_q;
    logic                 rd_pipe_mode_q;
    logic [BANK_W-1:0]    rd_pipe_bank_q;
    logic [$clog2(PV_MAX+1)-1:0] rd_pipe_pv_cur_q;
    logic [$clog2(C_MAX+1)-1:0]  rd_pipe_c_in_q;
    logic [BANK_W-1:0]    rd_pipe_bank_base_q;
    logic [31:0]          rd_pipe_m2_cgrp_u32_q;
    logic [31:0]          rd_pipe_m2_col_l_u32_q;
    logic [31:0]          rd_pipe_m2_bank_u32_q;
    logic [31:0]          rd_pipe_m2_addr_u32_q;

    logic [WORD_W-1:0]    rd_bram_word;
    logic [WORD_W-1:0]    rd_data_masked;

    assign rd_valid          = rd_pipe_valid_q;
    assign rd_data           = rd_data_masked;
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
        if (M2_BANKS > C_MAX) begin
            $error("ifm_buffer: C_MAX must cover all Mode2 banks cgrp*PC+col_l; require ceil(C_MAX/PC)*PC <= C_MAX, i.e. C_MAX divisible by PC for this shared bank array");
        end
        if (DEPTH < (HT * W_MAX)) begin
            $error("ifm_buffer: DEPTH too small for mode 1 worst-case Pv=1 mapping");
        end
        if (DEPTH < H_MAX) begin
            $error("ifm_buffer: DEPTH too small for mode 2 H_MAX mapping");
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
            for (int ti = 0; ti < M2_SLOT_COUNT; ti++) begin
                m2_slot_valid_flat[ti]   <= 1'b0;
                m2_slot_col_tag_flat[ti] <= '0;
            end
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

            // Mode2 metadata must be driven by the same sequential process that
            // initializes it.  Driving m2_slot_* from a second always_ff creates
            // multi-driven registers in Vivado.
            if (wr_en_sel && wr_addr_valid) begin
                if (wr_src_is_ofm && ofm_wr_mode2 && (wr_bank_phys_sel < C_MAX)) begin
                    m2_slot_valid_flat[m2_wr_slot_idx]   <= 1'b1;
                    m2_slot_col_tag_flat[m2_wr_slot_idx] <= ofm_wr_col_g[COL_W-1:0];
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
        logic wr_cgrp_valid_v;
        logic [31:0] cfg_m2_cgroups_v;
        logic [31:0] wr_m2_bank_u32;
        logic [31:0] wr_m2_addr_u32;

        wr_addr          = '0;
        wr_bank_phys_sel = '0;
        wr_addr_valid    = 1'b0;
        wr_bank_valid_v  = 1'b0;
        wr_row_valid_v   = 1'b0;
        wr_cgrp_valid_v  = 1'b0;
        cfg_m2_cgroups_v = '0;
        wr_m2_bank_u32   = '0;
        wr_m2_addr_u32   = '0;

        if (!wr_use_mode2_layout) begin
            // MODE 1 UNCHANGED:
            // bank = channel, addr = physical_ring_row * W_MAX + col_group,
            // lane = Pv pixel lane.
            //
            // DMA writes are for the currently configured layer, so the
            // current cfg_c_in_q guard is still correct. OFM->IFM writes may
            // be filling the NEXT layer before cfg_load advances this buffer,
            // so they are guarded only by physical C_MAX.
            wr_bank_valid_v = wr_src_is_ofm
                            ? (wr_bank_sel < C_MAX)
                            : (wr_bank_sel < cfg_c_in_q);

            // Mode 1 OFM refill provides a physical/free row slot inside HT.
            // DMA preload/refill also presents a row inside the active HT
            // window, so the same row guard is valid for both sources.
            wr_row_valid_v = (wr_row_idx_sel < HT);

            if (wr_bank_valid_v &&
                wr_row_valid_v &&
                (wr_col_idx_sel < W_MAX)) begin
                wr_bank_phys_sel = wr_bank_sel;
                wr_addr          = (wr_m1_phys_row * M1_STRIDE) + wr_col_idx_sel;
                wr_addr_valid    = (wr_addr < DEPTH);
            end
        end
        else begin
            // MODE 2 FIXED CONTRACT:
            //   interface:
            //     wr_bank_sel    = col_l inside resident W tile, 0..PC-1
            //     wr_row_idx_sel = absolute row
            //     wr_col_idx_sel = cgrp
            //     wr_data lanes  = PC channel lanes
            //   physical storage:
            //     bank = cgrp*PC + col_l
            //     addr = row
            //
            // OFM->IFM writes may target the NEXT layer before cfg_load
            // advances this buffer, so they must not be limited by the
            // current cfg_c_in_q. DMA writes are for the current layer and
            // can use cfg_c_in_q to reject impossible cgrp values.
            cfg_m2_cgroups_v = (cfg_c_in_q + PC - 1) / PC;

            wr_bank_valid_v = (wr_bank_sel < PC) && (wr_bank_sel < C_MAX);
            wr_row_valid_v  = wr_src_is_ofm
                            ? (wr_row_idx_sel < H_MAX)
                            : (wr_row_idx_sel < cfg_h_in_q);
            wr_cgrp_valid_v = wr_src_is_ofm
                            ? (wr_col_idx_sel < M2_CGRP_MAX)
                            : (wr_col_idx_sel < cfg_m2_cgroups_v);

            if (wr_bank_valid_v && wr_row_valid_v && wr_cgrp_valid_v) begin
                wr_m2_bank_u32 = (wr_col_idx_sel * PC) + wr_bank_sel;
                wr_m2_addr_u32 = wr_row_idx_sel;
                wr_bank_phys_sel = wr_m2_bank_u32[$clog2(C_MAX)-1:0];
                wr_addr          = wr_m2_addr_u32;
                wr_addr_valid    = (wr_m2_bank_u32 < C_MAX) && (wr_m2_addr_u32 < DEPTH);
            end
        end
    end

    //==================================================
    // Mode2 metadata flat indices
    //==================================================
    always_comb begin
        m2_wr_slot_idx = '0;
        if (wr_addr_valid && (wr_bank_phys_sel < C_MAX) && (wr_addr < DEPTH)) begin
            m2_wr_slot_idx = (wr_bank_phys_sel * DEPTH) + wr_addr;
        end

        m2_rd_slot_idx   = '0;
        m2_rd_slot_valid = 1'b0;
        if ((rd_m2_bank_u32 < C_MAX) && (rd_addr_m2_u32 < DEPTH)) begin
            m2_rd_slot_idx   = (rd_m2_bank_u32 * DEPTH) + rd_addr_m2_u32;
            m2_rd_slot_valid = 1'b1;
        end
    end

    //==================================================
    // Write path
    // Data payload goes through per-bank BRAM wrapper ports.  Metadata remains
    // unchanged and is updated only for OFM->IFM Mode2 writes as in the stable
    // version.
    //==================================================
    always_comb begin
        for (int wb = 0; wb < C_MAX; wb++) begin
            data_wr_en  [wb] = 1'b0;
            data_wr_addr[wb] = '0;
            data_wr_keep[wb] = '0;
            data_wr_data[wb] = '0;
        end

        if (wr_en_sel && wr_addr_valid && (wr_bank_phys_sel < C_MAX)) begin
            data_wr_en  [wr_bank_phys_sel] = 1'b1;
            data_wr_addr[wr_bank_phys_sel] = wr_addr;
            data_wr_keep[wr_bank_phys_sel] = wr_keep_sel;
            data_wr_data[wr_bank_phys_sel] = wr_data_sel;
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

        // MODE 1 UNCHANGED.
        rd_addr_m1 = (rd_m1_phys_row * M1_STRIDE) + rd_col_idx;

        // MODE 2 FIXED CONTRACT:
        //   rd_bank_base carries cgrp*PC.
        //   rd_col_idx carries local column col_l.
        //   IFM physical bank is cgrp*PC + col_l = rd_bank_base + rd_col_idx.
        //   IFM physical addr is row.
        rd_m2_cgrp_u32  = rd_bank_base / PC;
        rd_m2_col_l_u32 = rd_col_idx;
        rd_m2_bank_u32  = rd_bank_base + rd_col_idx;
        rd_addr_m2_u32  = rd_row_idx_ext;
        rd_addr_m2      = rd_addr_m2_u32;
    end

    always_comb begin
        rd_m2_tag_hit = 1'b0;
        if (m2_rd_slot_valid) begin
            rd_m2_tag_hit = m2_slot_valid_flat[m2_rd_slot_idx] &&
                            (m2_slot_col_tag_flat[m2_rd_slot_idx] == rd_col_g[COL_W-1:0]);
        end
    end

    //==================================================
    // Read path
    // The BRAM wrapper has a synchronous read port.  The request bank/address
    // are driven before the clock edge; the selected bank output and the
    // pipelined request metadata are used together after that edge.
    //==================================================
    always_comb begin
        rd_req_valid = 1'b0;
        rd_req_bank  = '0;
        rd_req_addr  = '0;

        if (rd_en) begin
            if (!cfg_mode_q) begin
                rd_req_bank  = rd_bank_base[BANK_W-1:0];
                rd_req_addr  = rd_addr_m1;
                rd_req_valid = (rd_bank_base < C_MAX) && (rd_addr_m1 < DEPTH);
            end else if (rd_m2_tag_hit) begin
                rd_req_bank  = rd_m2_bank_u32[BANK_W-1:0];
                rd_req_addr  = rd_addr_m2;
                rd_req_valid = (rd_m2_bank_u32 < C_MAX) && (rd_addr_m2_u32 < DEPTH);
            end
        end
    end

    always_comb begin
        for (int rb = 0; rb < C_MAX; rb++) begin
            data_rd_en  [rb] = 1'b0;
            data_rd_addr[rb] = '0;
        end

        if (rd_req_valid) begin
            data_rd_en  [rd_req_bank] = 1'b1;
            data_rd_addr[rd_req_bank] = rd_req_addr;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_pipe_valid_q        <= 1'b0;
            rd_pipe_mode_q         <= 1'b0;
            rd_pipe_bank_q         <= '0;
            rd_pipe_pv_cur_q       <= '0;
            rd_pipe_c_in_q         <= '0;
            rd_pipe_bank_base_q    <= '0;
            rd_pipe_m2_cgrp_u32_q  <= 32'd0;
            rd_pipe_m2_col_l_u32_q <= 32'd0;
            rd_pipe_m2_bank_u32_q  <= 32'd0;
            rd_pipe_m2_addr_u32_q  <= 32'd0;
        end else begin
            rd_pipe_valid_q        <= rd_req_valid;
            rd_pipe_mode_q         <= cfg_mode_q;
            rd_pipe_bank_q         <= rd_req_bank;
            rd_pipe_pv_cur_q       <= cfg_pv_cur_q;
            rd_pipe_c_in_q         <= cfg_c_in_q;
            rd_pipe_bank_base_q    <= rd_bank_base[BANK_W-1:0];
            rd_pipe_m2_cgrp_u32_q  <= rd_m2_cgrp_u32;
            rd_pipe_m2_col_l_u32_q <= rd_m2_col_l_u32;
            rd_pipe_m2_bank_u32_q  <= rd_m2_bank_u32;
            rd_pipe_m2_addr_u32_q  <= rd_addr_m2_u32;
        end
    end

    always_comb begin
        rd_bram_word   = '0;
        rd_data_masked = '0;

        if (rd_pipe_bank_q < C_MAX) begin
            rd_bram_word = data_rd_data[rd_pipe_bank_q];
        end

        if (rd_pipe_valid_q) begin
            if (!rd_pipe_mode_q) begin
                for (int lane = 0; lane < PV_MAX; lane++) begin
                    if (lane < rd_pipe_pv_cur_q) begin
                        rd_data_masked[lane*DATA_W +: DATA_W] = rd_bram_word[lane*DATA_W +: DATA_W];
                    end
                end
            end else begin
                for (int lane = 0; lane < PV_MAX; lane++) begin
                    if ((lane < PC) &&
                        ((rd_pipe_bank_base_q + lane) < rd_pipe_c_in_q) &&
                        (rd_pipe_m2_col_l_u32_q < PC) &&
                        (rd_pipe_m2_cgrp_u32_q < M2_CGRP_MAX) &&
                        (rd_pipe_m2_bank_u32_q < C_MAX) &&
                        (rd_pipe_m2_addr_u32_q < DEPTH)) begin
                        rd_data_masked[lane*DATA_W +: DATA_W] = rd_bram_word[lane*DATA_W +: DATA_W];
                    end
                end
            end
        end
    end


endmodule
