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
    // Physical storage
    // C_MAX banks, each address width = PV_MAX * DATA_W
    //==================================================
    (* ram_style = "block" *)
    logic [WORD_W-1:0] mem [0:C_MAX-1][0:DEPTH-1];

    // Mode2 rolling-slot content tag.  Mode1 never reads these tags.
    logic                 m2_slot_valid   [0:C_MAX-1][0:DEPTH-1];
    logic [COL_W-1:0]     m2_slot_col_tag [0:C_MAX-1][0:DEPTH-1];

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
            for (int ti = 0; ti < C_MAX; ti++) begin
                for (int tj = 0; tj < DEPTH; tj++) begin
                    m2_slot_valid[ti][tj]   <= 1'b0;
                    m2_slot_col_tag[ti][tj] <= '0;
                end
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
    // Write path
    //==================================================
    integer wlane;
    always_ff @(posedge clk) begin
        if (wr_en_sel && wr_addr_valid) begin
            for (wlane = 0; wlane < PV_MAX; wlane++) begin
                if (wr_keep_sel[wlane]) begin
                    mem[wr_bank_phys_sel][wr_addr][wlane*DATA_W +: DATA_W]
                        <= wr_data_sel[wlane*DATA_W +: DATA_W];
                end
            end

            if (wr_src_is_ofm && ofm_wr_mode2 && (wr_bank_phys_sel < C_MAX)) begin
                m2_slot_valid[wr_bank_phys_sel][wr_addr]   <= 1'b1;
                m2_slot_col_tag[wr_bank_phys_sel][wr_addr] <= ofm_wr_col_g[COL_W-1:0];
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
        if ((rd_m2_bank_u32 < C_MAX) && (rd_addr_m2_u32 < DEPTH)) begin
            rd_m2_tag_hit = m2_slot_valid[rd_m2_bank_u32][rd_addr_m2] &&
                            (m2_slot_col_tag[rd_m2_bank_u32][rd_addr_m2] == rd_col_g[COL_W-1:0]);
        end
    end

    //==================================================
    // Read path: 1-cycle registered output
    //==================================================
    integer rlane;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_valid_q <= 1'b0;
            rd_data_q  <= '0;
        end
        else begin
            rd_valid_q <= rd_en && (!cfg_mode_q || rd_m2_tag_hit);
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
                else if (rd_m2_tag_hit) begin
                    // MODE 2 FIXED CONTRACT:
                    //   bank = cgrp*PC + col_l = rd_bank_base + rd_col_idx
                    //   addr = row
                    //   lane = pc_l = rlane
                    //
                    // rd_bank_base carries cgrp*PC and rd_col_idx carries col_l,
                    // so existing upstream mode-2 address generation remains
                    // source-compatible while the physical layout is corrected.
                    for (rlane = 0; rlane < PV_MAX; rlane++) begin
                        if ((rlane < PC) &&
                            ((rd_bank_base + rlane) < cfg_c_in_q) &&
                            (rd_m2_col_l_u32 < PC) &&
                            (rd_m2_cgrp_u32 < M2_CGRP_MAX) &&
                            (rd_m2_bank_u32 < C_MAX) &&
                            (rd_addr_m2_u32 < DEPTH)) begin

                            rd_data_q[rlane*DATA_W +: DATA_W]
                                <= mem[rd_m2_bank_u32][rd_addr_m2][rlane*DATA_W +: DATA_W];
                        end
                        else begin
                            rd_data_q[rlane*DATA_W +: DATA_W] <= '0;
                        end
                    end
                end
            end
        end
    end
    

always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    // no-op
  end else begin
    if (cfg_mode) begin

      if (dma_wr_en && dma_wr_ready) begin
        $display("DBG_IFM_M2_DMA_WR t=%0t bank_col_l=%0d row=%0d cgrp=%0d keep=%h data0=%0d data1=%0d",
                 $time,
                 dma_wr_bank,
                 dma_wr_row_idx,
                 dma_wr_col_idx,
                 dma_wr_keep,
                 $signed(dma_wr_data[0*DATA_W +: DATA_W]),
                 $signed(dma_wr_data[1*DATA_W +: DATA_W]));
      end

      if (ofm_wr_en && ofm_wr_ready) begin
        $display("DBG_IFM_M2_OFM_WR t=%0t bank_col_l=%0d row=%0d cgrp=%0d keep=%h data0=%0d data1=%0d",
                 $time,
                 ofm_wr_bank,
                 ofm_wr_row_idx,
                 ofm_wr_col_idx,
                 ofm_wr_keep,
                 $signed(ofm_wr_data[0*DATA_W +: DATA_W]),
                 $signed(ofm_wr_data[1*DATA_W +: DATA_W]));
      end

      // Focus on reads near the problematic right edge of the resident PC window.
      if (rd_en && ((rd_col_idx <= 2) || (rd_col_idx >= PC-2))) begin
        $display("DBG_IFM_M2_RD t=%0t bank_base=%0d row=%0d col_l=%0d cgrp=%0d rd_valid=%0b data0=%0d data1=%0d",
                 $time,
                 rd_bank_base,
                 rd_row_idx,
                 rd_col_idx,
                 (PC == 0) ? 0 : (rd_bank_base / PC),
                 rd_valid,
                 $signed(rd_data[0*DATA_W +: DATA_W]),
                 $signed(rd_data[1*DATA_W +: DATA_W]));
      end
    end
  end
end

`ifndef SYNTHESIS

logic        dbg_ifm_m2_rd_q;
logic        dbg_ifm_m2_rd_focus_q;
logic [$clog2(C_MAX)-1:0] dbg_ifm_m2_bank_base_q;
logic [$clog2(H_MAX)-1:0] dbg_ifm_m2_row_q;
logic [$clog2(W_MAX)-1:0] dbg_ifm_m2_col_l_q;
logic [31:0] dbg_ifm_m2_cgrp_q;
logic [31:0] dbg_ifm_m2_addr_q;

function automatic logic dbg_ifm_m2_col_focus(input logic [31:0] col_l);
begin
  dbg_ifm_m2_col_focus = (col_l <= 32'd2) || ((PC > 2) && ((col_l + 32'd2) >= PC));
end
endfunction

always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    dbg_ifm_m2_rd_q         <= 1'b0;
    dbg_ifm_m2_rd_focus_q   <= 1'b0;
    dbg_ifm_m2_bank_base_q  <= '0;
    dbg_ifm_m2_row_q        <= '0;
    dbg_ifm_m2_col_l_q      <= '0;
    dbg_ifm_m2_cgrp_q       <= 32'd0;
    dbg_ifm_m2_addr_q       <= 32'd0;
  end else begin

    if (cfg_mode_q) begin

      if (ofm_wr_en && ofm_wr_ready && ((ofm_wr_bank <= 2) || ((PC > 2) && ((ofm_wr_bank + 16'd2) >= PC)) || (ofm_wr_row_idx < 4))) begin
        $display("DBG_IFM_M2_OFM_WR_X t=%0t col_l=%0d row=%0d cgrp=%0d phys_bank=%0d keep=%h wr_addr_valid=%0b wr_addr=%0d data0=%0d data1=%0d", $time, ofm_wr_bank, ofm_wr_row_idx, ofm_wr_col_idx, wr_bank_phys_sel, ofm_wr_keep, wr_addr_valid, wr_addr, $signed(ofm_wr_data[0*DATA_W +: DATA_W]), $signed(ofm_wr_data[1*DATA_W +: DATA_W]));
      end

      if (rd_en && dbg_ifm_m2_col_focus(rd_col_idx)) begin
        $display("DBG_IFM_M2_RD_REQ t=%0t bank_base=%0d row=%0d col_l=%0d cgrp=%0d phys_bank=%0d rd_addr_u32=%0d rd_addr=%0d cfg_C=%0d cfg_H=%0d cfg_W=%0d", $time, rd_bank_base, rd_row_idx, rd_col_idx, rd_m2_cgrp_u32, rd_m2_bank_u32, rd_addr_m2_u32, rd_addr_m2, cfg_c_in_q, cfg_h_in_q, cfg_w_in_q);
      end

      if (dbg_ifm_m2_rd_q && dbg_ifm_m2_rd_focus_q) begin
        $display("DBG_IFM_M2_RD_RET t=%0t bank_base=%0d row=%0d col_l=%0d cgrp=%0d addr=%0d rd_valid=%0b data0=%0d data1=%0d cfg_C=%0d", $time, dbg_ifm_m2_bank_base_q, dbg_ifm_m2_row_q, dbg_ifm_m2_col_l_q, dbg_ifm_m2_cgrp_q, dbg_ifm_m2_addr_q, rd_valid_q, $signed(rd_data_q[0*DATA_W +: DATA_W]), $signed(rd_data_q[1*DATA_W +: DATA_W]), cfg_c_in_q);
      end

    end

    dbg_ifm_m2_rd_q <= cfg_mode_q && rd_en;

    if (cfg_mode_q && rd_en) begin
      dbg_ifm_m2_bank_base_q <= rd_bank_base;
      dbg_ifm_m2_row_q       <= rd_row_idx;
      dbg_ifm_m2_col_l_q     <= rd_col_idx;
      dbg_ifm_m2_cgrp_q      <= rd_m2_cgrp_u32;
      dbg_ifm_m2_addr_q      <= rd_addr_m2_u32;
      dbg_ifm_m2_rd_focus_q  <= dbg_ifm_m2_col_focus(rd_col_idx) || (rd_row_idx < 4);
    end else begin
      dbg_ifm_m2_rd_focus_q <= 1'b0;
    end

  end
end

`endif

`ifndef SYNTHESIS

always_ff @(posedge clk or negedge rst_n) begin : DBG_IFM_M2_PAYLOAD_PATH_MON
  if (!rst_n) begin
    // no-op
  end else begin
    if (cfg_mode_q) begin
      // OFM->IFM write request as seen by IFM buffer
      if (ofm_wr_en && ofm_wr_ready && ((ofm_wr_row_idx < 4) || (ofm_wr_col_idx < 2) || (ofm_wr_data[0*DATA_W +: DATA_W] == '0))) begin
        $display("DBG_IFM_M2_OFM_WR_IN t=%0t bank_col_l=%0d row=%0d cgrp=%0d keep=%h data0=%0d data1=%0d data2=%0d data3=%0d", $time, ofm_wr_bank, ofm_wr_row_idx, ofm_wr_col_idx, ofm_wr_keep, $signed(ofm_wr_data[0*DATA_W +: DATA_W]), $signed(ofm_wr_data[1*DATA_W +: DATA_W]), $signed(ofm_wr_data[2*DATA_W +: DATA_W]), $signed(ofm_wr_data[3*DATA_W +: DATA_W]));
      end

      // Actual selected write mapping inside IFM buffer
      if (wr_en_sel && wr_src_is_ofm && wr_addr_valid && ((wr_row_idx_sel < 4) || (wr_col_idx_sel < 2) || (wr_data_sel[0*DATA_W +: DATA_W] == '0))) begin
        $display("DBG_IFM_M2_OFM_WR_MAP t=%0t wr_bank=%0d wr_row=%0d wr_colidx_cgrp=%0d wr_addr=%0d keep=%h data0=%0d data1=%0d data2=%0d data3=%0d", $time, wr_bank_sel, wr_row_idx_sel, wr_col_idx_sel, wr_addr, wr_keep_sel, $signed(wr_data_sel[0*DATA_W +: DATA_W]), $signed(wr_data_sel[1*DATA_W +: DATA_W]), $signed(wr_data_sel[2*DATA_W +: DATA_W]), $signed(wr_data_sel[3*DATA_W +: DATA_W]));
      end

      // Read request from addr_gen/CE
      if (rd_en && ((rd_row_idx < 4) || (rd_col_idx < 4) || (rd_valid_q == 1'b0))) begin
        $display("DBG_IFM_M2_RD_PATH t=%0t rd_en=%0b bank_base=%0d row=%0d col_l=%0d cgrp=%0d rd_addr_u32=%0d rd_addr=%0d rd_valid_q=%0b data0=%0d data1=%0d data2=%0d data3=%0d", $time, rd_en, rd_bank_base, rd_row_idx, rd_col_idx, rd_m2_cgrp_u32, rd_addr_m2_u32, rd_addr_m2, rd_valid_q, $signed(rd_data_q[0*DATA_W +: DATA_W]), $signed(rd_data_q[1*DATA_W +: DATA_W]), $signed(rd_data_q[2*DATA_W +: DATA_W]), $signed(rd_data_q[3*DATA_W +: DATA_W]));
      end
    end
  end
end

`endif

`ifndef SYNTHESIS

logic        dbg_m2_wr_q;
integer      dbg_m2_wr_bank_q;
integer      dbg_m2_wr_addr_q;
integer      dbg_m2_wr_row_q;
integer      dbg_m2_wr_cgrp_q;
integer      dbg_m2_wr_col_l_q;
integer      dbg_m2_wr_col_g_q;

always_ff @(posedge clk or negedge rst_n) begin : DBG_IFM_M2_PHYS_MAP_MON
  if (!rst_n) begin
    dbg_m2_wr_q       <= 1'b0;
    dbg_m2_wr_bank_q  <= 0;
    dbg_m2_wr_addr_q  <= 0;
    dbg_m2_wr_row_q   <= 0;
    dbg_m2_wr_cgrp_q  <= 0;
    dbg_m2_wr_col_l_q <= 0;
    dbg_m2_wr_col_g_q <= 0;
  end else begin
    if (dbg_m2_wr_q) begin
      if ((dbg_m2_wr_bank_q >= 0) && (dbg_m2_wr_bank_q < C_MAX) &&
          (dbg_m2_wr_addr_q >= 0) && (dbg_m2_wr_addr_q < DEPTH)) begin
        $display("DBG_IFM_M2_WR_COMMIT_PHYS t=%0t bank_phys=%0d addr=%0d row=%0d cgrp=%0d col_l=%0d col_g=%0d mem0=%0d mem1=%0d mem2=%0d mem3=%0d tag_valid=%0b tag=%0d",
          $time,
          dbg_m2_wr_bank_q,
          dbg_m2_wr_addr_q,
          dbg_m2_wr_row_q,
          dbg_m2_wr_cgrp_q,
          dbg_m2_wr_col_l_q,
          dbg_m2_wr_col_g_q,
          $signed(mem[dbg_m2_wr_bank_q][dbg_m2_wr_addr_q][0*DATA_W +: DATA_W]),
          $signed(mem[dbg_m2_wr_bank_q][dbg_m2_wr_addr_q][1*DATA_W +: DATA_W]),
          $signed(mem[dbg_m2_wr_bank_q][dbg_m2_wr_addr_q][2*DATA_W +: DATA_W]),
          $signed(mem[dbg_m2_wr_bank_q][dbg_m2_wr_addr_q][3*DATA_W +: DATA_W]),
          m2_slot_valid[dbg_m2_wr_bank_q][dbg_m2_wr_addr_q],
          m2_slot_col_tag[dbg_m2_wr_bank_q][dbg_m2_wr_addr_q]
        );
      end
    end

    dbg_m2_wr_q <= 1'b0;

    if (wr_en_sel && wr_src_is_ofm && ofm_wr_mode2 && wr_addr_valid &&
        (wr_row_idx_sel < 4) && (wr_col_idx_sel < 4) && (wr_bank_sel < 4)) begin
      $display("DBG_IFM_M2_WR_REQ_PHYS t=%0t logical_col_l=%0d cgrp=%0d row=%0d col_g=%0d bank_phys_calc=%0d bank_phys_used=%0d addr_calc=%0d addr_used=%0d data0=%0d data1=%0d data2=%0d data3=%0d",
        $time,
        wr_bank_sel,
        wr_col_idx_sel,
        wr_row_idx_sel,
        ofm_wr_col_g,
        (wr_col_idx_sel * PC) + wr_bank_sel,
        wr_bank_phys_sel,
        wr_row_idx_sel,
        wr_addr,
        $signed(wr_data_sel[0*DATA_W +: DATA_W]),
        $signed(wr_data_sel[1*DATA_W +: DATA_W]),
        $signed(wr_data_sel[2*DATA_W +: DATA_W]),
        $signed(wr_data_sel[3*DATA_W +: DATA_W])
      );

      dbg_m2_wr_q       <= 1'b1;
      dbg_m2_wr_bank_q  <= wr_bank_phys_sel;
      dbg_m2_wr_addr_q  <= wr_addr;
      dbg_m2_wr_row_q   <= wr_row_idx_sel;
      dbg_m2_wr_cgrp_q  <= wr_col_idx_sel;
      dbg_m2_wr_col_l_q <= wr_bank_sel;
      dbg_m2_wr_col_g_q <= ofm_wr_col_g;
    end

    if (rd_en && cfg_mode_q && (rd_row_idx < 4) && (rd_bank_base < 64) && (rd_col_idx < 4)) begin
      $display("DBG_IFM_M2_RD_REQ_PHYS t=%0t rd_bank_base=%0d cgrp=%0d col_l=%0d row=%0d rd_col_g=%0d bank_phys_calc=%0d addr_calc=%0d rd_valid_q=%0b data0=%0d data1=%0d data2=%0d data3=%0d tag_valid=%0b tag=%0d",
        $time,
        rd_bank_base,
        (rd_bank_base / PC),
        rd_col_idx,
        rd_row_idx,
        rd_col_g,
        rd_bank_base + rd_col_idx,
        rd_row_idx,
        rd_valid_q,
        $signed(rd_data_q[0*DATA_W +: DATA_W]),
        $signed(rd_data_q[1*DATA_W +: DATA_W]),
        $signed(rd_data_q[2*DATA_W +: DATA_W]),
        $signed(rd_data_q[3*DATA_W +: DATA_W]),
        (((rd_bank_base + rd_col_idx) < C_MAX) && (rd_row_idx < DEPTH)) ? m2_slot_valid[rd_bank_base + rd_col_idx][rd_row_idx] : 1'b0,
        (((rd_bank_base + rd_col_idx) < C_MAX) && (rd_row_idx < DEPTH)) ? m2_slot_col_tag[rd_bank_base + rd_col_idx][rd_row_idx] : '0
      );
    end
  end
end

`endif

`ifndef SYNTHESIS

always_ff @(posedge clk or negedge rst_n) begin : DBG_IFM_ANY_RD_MON
  integer dbg_bank_phys;
  integer dbg_addr_phys;
begin
  if (!rst_n) begin
  end else begin
    if (rd_en) begin
      dbg_bank_phys = rd_bank_base + rd_col_idx;
      dbg_addr_phys = rd_row_idx;

      $display("DBG_IFM_ANY_RD t=%0t cfg_mode=%0b rd_en=%0b rd_bank_base=%0d rd_row=%0d rd_col_idx=%0d rd_col_g=%0d calc_bank=%0d calc_addr=%0d rd_valid_q=%0b rd_data0=%0d rd_data1=%0d rd_data2=%0d rd_data3=%0d",
        $time,
        cfg_mode_q,
        rd_en,
        rd_bank_base,
        rd_row_idx,
        rd_col_idx,
        rd_col_g,
        dbg_bank_phys,
        dbg_addr_phys,
        rd_valid_q,
        $signed(rd_data_q[0*DATA_W +: DATA_W]),
        $signed(rd_data_q[1*DATA_W +: DATA_W]),
        $signed(rd_data_q[2*DATA_W +: DATA_W]),
        $signed(rd_data_q[3*DATA_W +: DATA_W])
      );

      if ((dbg_bank_phys >= 0) && (dbg_bank_phys < C_MAX) &&
          (dbg_addr_phys >= 0) && (dbg_addr_phys < DEPTH)) begin
        $display("DBG_IFM_ANY_RD_MEM t=%0t calc_bank=%0d calc_addr=%0d mem0=%0d mem1=%0d mem2=%0d mem3=%0d tag_valid=%0b tag=%0d",
          $time,
          dbg_bank_phys,
          dbg_addr_phys,
          $signed(mem[dbg_bank_phys][dbg_addr_phys][0*DATA_W +: DATA_W]),
          $signed(mem[dbg_bank_phys][dbg_addr_phys][1*DATA_W +: DATA_W]),
          $signed(mem[dbg_bank_phys][dbg_addr_phys][2*DATA_W +: DATA_W]),
          $signed(mem[dbg_bank_phys][dbg_addr_phys][3*DATA_W +: DATA_W]),
          m2_slot_valid[dbg_bank_phys][dbg_addr_phys],
          m2_slot_col_tag[dbg_bank_phys][dbg_addr_phys]
        );
      end
    end
  end
end
end

`endif


`ifndef SYNTHESIS
always_ff @(posedge clk or negedge rst_n) begin : DBG_IFM_RD_RAW_ALWAYS
  integer b;
  integer a;
  if (!rst_n) begin
  end else begin
    if (rd_en) begin
      b = rd_bank_base + rd_col_idx;
      a = rd_row_idx;

      $display("DBG_IFM_RD_RAW t=%0t cfg_mode=%0b rd_en=%0b rd_bank_base=%0d rd_col_idx=%0d rd_row=%0d rd_col_g=%0d calc_bank=%0d calc_addr=%0d rd_valid_q=%0b rd_data0=%0d rd_data1=%0d rd_data2=%0d rd_data3=%0d",
        $time, cfg_mode_q, rd_en,
        rd_bank_base, rd_col_idx, rd_row_idx, rd_col_g,
        b, a, rd_valid_q,
        $signed(rd_data_q[0*DATA_W +: DATA_W]),
        $signed(rd_data_q[1*DATA_W +: DATA_W]),
        $signed(rd_data_q[2*DATA_W +: DATA_W]),
        $signed(rd_data_q[3*DATA_W +: DATA_W])
      );

      if ((b >= 0) && (b < C_MAX) && (a >= 0) && (a < DEPTH)) begin
        $display("DBG_IFM_RD_RAW_MEM t=%0t calc_bank=%0d calc_addr=%0d mem0=%0d mem1=%0d mem2=%0d mem3=%0d tag_valid=%0b tag=%0d",
          $time, b, a,
          $signed(mem[b][a][0*DATA_W +: DATA_W]),
          $signed(mem[b][a][1*DATA_W +: DATA_W]),
          $signed(mem[b][a][2*DATA_W +: DATA_W]),
          $signed(mem[b][a][3*DATA_W +: DATA_W]),
          m2_slot_valid[b][a],
          m2_slot_col_tag[b][a]
        );
      end
    end
  end
end
`endif

endmodule
