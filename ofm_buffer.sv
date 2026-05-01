module ofm_buffer #(
    parameter int DATA_W   = 8,    // stored OFM width
    parameter int M1_IN_W  = DATA_W, // input width from pooling_mode1/compute top
    parameter int M2_IN_W  = DATA_W, // input width from pooling_mode2/compute top
    parameter int PV_MAX   = 8,
    parameter int PC       = 8,
    parameter int PF       = 4,
    parameter int PTOTAL   = 16,
    parameter int C_MAX    = 128,
    parameter int H_MAX    = 224,
    parameter int W_MAX    = 224,
    // Full-FM storage: one bank per channel, one word holds up to PV_MAX pixels.
    // To support arbitrary dynamic Pv_next in mode 1, worst case is 1 pixel/word.
    parameter int DEPTH    = H_MAX * W_MAX,
    parameter int TAG_W    = 8
)(
    input  logic clk,
    input  logic rst_n,

    // ============================================================
    // Layer configuration
    // cfg_src_mode  = 0: write stream comes from mode 1 pooling
    // cfg_src_mode  = 1: write stream comes from mode 2 pooling
    // cfg_next_mode = 0: next layer IFM uses mode 1 layout
    // cfg_next_mode = 1: next layer IFM uses mode 2 layout
    //
    // Behavior:
    // - same-mode refill is continuous:
    //     * mode1 -> mode1 : write directly in next Pv layout
    //     * mode2 -> mode2 : write directly in mode2 row-segment layout
    // - mode1 -> mode2 boundary:
    //     * write full OFM in source mode1 layout first
    //     * then use special read path to refill IFM mode2
    // ============================================================
    input  logic                        layer_start,
    input  logic                        cfg_src_mode,
    input  logic                        cfg_next_mode,
    input  logic [$clog2(H_MAX+1)-1:0]  cfg_h_out,
    input  logic [$clog2(W_MAX+1)-1:0]  cfg_w_out,
    input  logic [7:0]                  cfg_f_out,
    input  logic [7:0]                  cfg_pv_cur,   // current layer Pv before pooling (mode 1 only)
    input  logic [7:0]                  cfg_pf_cur,   // current layer Pf (mode 1 only)
    input  logic [7:0]                  cfg_pv_next,  // next layer Pv (mode 1 only)
    input  logic [7:0]                  cfg_pf_next,  // next layer Pf/grouping for same-mode mode1

    // ============================================================
    // Write input from mode1 pooling / compute path
    // Lane mapping assumed from pooling_mode1:
    //   lane = pf * (Pv_cur/2) + x
    // ============================================================
    input  logic                        m1_wr_en,
    input  logic [15:0]                 m1_wr_filter_base,
    input  logic [15:0]                 m1_wr_row,
    input  logic [15:0]                 m1_wr_col_base,
    input  logic [15:0]                 m1_wr_count,
    input  logic signed [M1_IN_W-1:0]   m1_wr_data [0:PTOTAL-1],

    // ============================================================
    // Write input from mode2 pooling / compute path
    // Current interface carries one pooled spatial location across PF channels.
    // ============================================================
    input  logic                        m2_wr_en,
    input  logic [15:0]                 m2_wr_row,
    input  logic [15:0]                 m2_wr_col,
    input  logic [15:0]                 m2_wr_f_base,
    input  logic [PF*M2_IN_W-1:0]       m2_wr_data,

    // ============================================================
    // Stream output to IFM buffer write port
    //
    // Control provides which portion of the full OFM should be refilled now.
    // - next_mode = 0:
    //     row_base / num_rows define the mode1 tile height to load
    //     col_base is ignored
    // - next_mode = 1:
    //     row_base / num_rows define the rows to load
    //     col_base selects the horizontal tile base (pixel index)
    // ============================================================
    input logic [$clog2(H_MAX+1)-1:0]   ifm_stream_row_base,
    input logic [$clog2(H_MAX+1)-1:0]   ifm_stream_num_rows,
    input logic [$clog2(W_MAX+1)-1:0]   ifm_stream_col_base,
    input  logic                        ifm_stream_start,
    input  logic [$clog2(H_MAX)-1:0]    ifm_stream_m1_row_slot_l,
    input  logic [15:0]                 ifm_stream_m1_ch_blk_g,
    input  logic [15:0]                 ifm_stream_m2_cgrp_g,
    output logic                        ifm_stream_busy,
    output logic                        ifm_stream_done,

    output logic                        ifm_ofm_wr_en,
    output logic [$clog2(C_MAX)-1:0]    ifm_ofm_wr_bank,
    output logic [$clog2(H_MAX)-1:0]    ifm_ofm_wr_row_idx,
    // Width is sized to W_MAX so mode 1 dynamic Pv can use the full group index range.
    output logic [$clog2(W_MAX)-1:0]    ifm_ofm_wr_col_idx,
    output logic [PV_MAX*DATA_W-1:0]    ifm_ofm_wr_data,
    output logic [PV_MAX-1:0]           ifm_ofm_wr_keep,
    input  logic                        ifm_ofm_wr_ready,

    // ============================================================
    // Same-mode ready-token visibility
    //
    // UPDATED CONTRACT:
    // - These outputs expose which stored OFM words became complete after the
    //   previous cycle's writes.
    // - They are only meaningful in same-mode direct storage:
    //     * mode1 -> mode1 : m1_sm_ready_*
    //     * mode2 -> mode2 : m2_sm_ready_*
    // - mode1 -> mode2 still uses the coarse stream / special read path.
    //
    // Each valid bit is a 1-cycle pulse. A token identifies a same-mode refill
    // unit that is now fully ready:
    // - mode1->mode1:
    //     * row_g / colgrp_g identify the stored spatial word in next-Pv layout
    //     * bank carries the channel-block id used by the direct same-mode path
    //       in this module (grouped by cfg_pf_cur for mode-1 source writes)
    // - mode2->mode2:
    //     * row_g / colbase_g identify the spatial location
    //     * bank carries cgrp_g = floor(channel / PC)
    // ============================================================
    output logic [PTOTAL-1:0]           m1_sm_ready_valid,
    output logic [15:0]                 m1_sm_ready_bank [0:PTOTAL-1],
    output logic [15:0]                 m1_sm_ready_row_g [0:PTOTAL-1],
    output logic [15:0]                 m1_sm_ready_colgrp_g [0:PTOTAL-1],

    output logic [PF-1:0]               m2_sm_ready_valid,
    output logic [15:0]                 m2_sm_ready_bank [0:PF-1],
    output logic [15:0]                 m2_sm_ready_row_g [0:PF-1],
    output logic [15:0]                 m2_sm_ready_colbase_g [0:PF-1],

    // ============================================================
    // DMA linear read port
    // Address is relative to the beginning of the currently active layer.
    // ============================================================
    input  logic                        ofm_dma_rd_en,
    input  logic [$clog2(C_MAX*DEPTH)-1:0] ofm_dma_rd_addr,
    output logic                        ofm_dma_rd_valid,
    output logic [PV_MAX*DATA_W-1:0]    ofm_dma_rd_data,
    output logic [PV_MAX-1:0]           ofm_dma_rd_keep,

    // ============================================================
    // Status
    // ============================================================
    output logic [31:0]                 layer_num_words,
    output logic [31:0]                 layer_num_pixels,
    output logic [31:0]                 layer_pixels_written,
    output logic                        layer_write_done,
    output logic                        error
);

    localparam int WORD_W    = PV_MAX * DATA_W;
    localparam int DEPTH_W   = (DEPTH <= 1) ? 1 : $clog2(DEPTH);
    localparam int BANK_W    = (C_MAX <= 1) ? 1 : $clog2(C_MAX);
    localparam int ROW_W     = (H_MAX <= 1) ? 1 : $clog2(H_MAX);
    localparam int COLIDX_W  = (W_MAX <= 1) ? 1 : $clog2(W_MAX);
    localparam int DMA_AW    = (C_MAX * DEPTH <= 1) ? 1 : $clog2(C_MAX * DEPTH);

    typedef enum logic [1:0] {
        STRM_IDLE,
        STRM_M1_DIRECT,
        STRM_M2_DIRECT,
        STRM_M1_TO_M2
    } strm_mode_t;

    // ============================================================
    // Physical storage
    // ============================================================
    (* ram_style = "block" *)
    logic [WORD_W-1:0]   mem_data [0:C_MAX-1][0:DEPTH-1];
    (* ram_style = "distributed" *)
    logic [PV_MAX-1:0]   mem_fill [0:C_MAX-1][0:DEPTH-1];
    (* ram_style = "distributed" *)
    logic [TAG_W-1:0]    mem_tag  [0:C_MAX-1][0:DEPTH-1];

    // ============================================================
    // Latched layer configuration
    // ============================================================
    logic         src_mode_q;
    logic         next_mode_q;
    logic [$clog2(H_MAX+1)-1:0] h_out_q;
    logic [$clog2(W_MAX+1)-1:0] w_out_q;
    logic [7:0]   f_out_q;
    logic [7:0]   pv_cur_q, pf_cur_q, pv_next_q, pf_next_q;
    logic [15:0]  src_pack_q;      // pooled Pv from current mode1 layer
    logic [15:0]  store_pack_q;    // pack used by stored words for this layer
    logic [15:0]  stored_groups_q; // groups per row in storage
    logic [31:0]  total_pixels_q;
    logic [31:0]  pixels_written_q;
    logic         layer_write_done_q;
    logic [TAG_W-1:0] layer_tag_q;
    logic         error_q;

    assign layer_num_words      = f_out_q * h_out_q * stored_groups_q;
    assign layer_num_pixels     = total_pixels_q;
    assign layer_pixels_written = pixels_written_q;
    assign layer_write_done     = layer_write_done_q;
    assign error                = error_q;

    // ============================================================
    // Stream state
    // ============================================================
    strm_mode_t strm_mode_q;
    logic       strm_active_q;
    logic [15:0] strm_row_base_q, strm_num_rows_q, strm_col_base_q;
    logic [15:0] strm_row_q;      // local row index for mode1 direct, absolute row for mode2 paths
    logic [15:0] strm_ch_q;       // channel offset within selected slice
    logic [15:0] strm_colgrp_q;   // kept for transition-path compatibility
    logic [$clog2(H_MAX)-1:0] strm_m1_row_slot_q;
    logic [15:0] strm_m1_ch_blk_q;
    logic [15:0] strm_m2_cgrp_q;

    // ============================================================
    // DMA read registers
    // ============================================================
    logic                     dma_valid_q;
    logic [WORD_W-1:0]        dma_data_q;
    logic [PV_MAX-1:0]        dma_keep_q;

    // ============================================================
    // Same-mode ready-token trackers
    //
    // We track which words were touched in the previous cycle's write path,
    // then on the next cycle emit a ready token if that word is now complete.
    // This avoids missing readiness when multiple writes in one cycle finish
    // the same stored word.
    // ============================================================
    logic [PTOTAL-1:0]        m1_touch_v_q;
    logic [15:0]              m1_touch_bank_q   [0:PTOTAL-1];
    logic [15:0]              m1_touch_row_q    [0:PTOTAL-1];
    logic [15:0]              m1_touch_colgrp_q [0:PTOTAL-1];

    logic [PF-1:0]            m2_touch_v_q;
    logic [15:0]              m2_touch_bank_q    [0:PF-1];
    logic [15:0]              m2_touch_row_q     [0:PF-1];
    logic [15:0]              m2_touch_colgrp_q  [0:PF-1];

    assign ofm_dma_rd_valid = dma_valid_q;
    assign ofm_dma_rd_data  = dma_data_q;
    assign ofm_dma_rd_keep  = dma_keep_q;

    // ============================================================
    // Helper functions
    // ============================================================
    function automatic [31:0] ceil_div_u32(input [31:0] a, input [31:0] b);
        begin
            if (b == 0)
                ceil_div_u32 = 32'd0;
            else
                ceil_div_u32 = (a + b - 1) / b;
        end
    endfunction

    // Mode-1 source pack:
    // - pool_en=1 or X/Z: pooling output packs max(Pv_cur/2, 1) spatial pixels.
    // - pool_en=0       : no-pool bypass emits exactly 1 spatial pixel per write.
    // Treating X/Z as enabled preserves the legacy pooling behavior.
    function automatic [15:0] mode1_src_pack(
        input logic [7:0] pv_cur,
        input logic       pool_en
    );
        begin
            if (pool_en === 1'b0)
                mode1_src_pack = 16'd1;
            else
                mode1_src_pack = (pv_cur > 1) ? {9'd0, pv_cur[7:1]} : 16'd1;
        end
    endfunction

    function automatic [PV_MAX-1:0] calc_keep_mask(
        input [15:0] pack,
        input [15:0] col_base,
        input [15:0] w_total
    );
        integer i;
        integer valid_lanes;
        begin
            calc_keep_mask = '0;
            if (col_base >= w_total) begin
                valid_lanes = 0;
            end
            else if ((col_base + pack) <= w_total) begin
                valid_lanes = pack;
            end
            else begin
                valid_lanes = w_total - col_base;
            end

            for (i = 0; i < PV_MAX; i++)
                calc_keep_mask[i] = (i < valid_lanes);
        end
    endfunction

    function automatic logic signed [DATA_W-1:0] sat_m1(input logic signed [M1_IN_W-1:0] din);
        longint signed max_v, min_v, x;
        begin
            max_v = (1 <<< (DATA_W-1)) - 1;
            min_v = -(1 <<< (DATA_W-1));
            x = din;
            if (x > max_v)
                sat_m1 = max_v[DATA_W-1:0];
            else if (x < min_v)
                sat_m1 = min_v[DATA_W-1:0];
            else
                sat_m1 = x[DATA_W-1:0];
        end
    endfunction

    function automatic logic signed [DATA_W-1:0] sat_m2(input logic signed [M2_IN_W-1:0] din);
        longint signed max_v, min_v, x;
        begin
            max_v = (1 <<< (DATA_W-1)) - 1;
            min_v = -(1 <<< (DATA_W-1));
            x = din;
            if (x > max_v)
                sat_m2 = max_v[DATA_W-1:0];
            else if (x < min_v)
                sat_m2 = min_v[DATA_W-1:0];
            else
                sat_m2 = x[DATA_W-1:0];
        end
    endfunction

    function automatic logic [WORD_W-1:0] build_special_m1_to_m2_word(
        input logic [BANK_W-1:0] bank,
        input logic [15:0] abs_row,
        input logic [15:0] col_base,
        input logic [15:0] w_total,
        input logic [15:0] src_pack,
        input logic [15:0] src_groups,
        input logic [TAG_W-1:0] layer_tag
    );
        integer lane;
        integer abs_col;
        integer src_grp;
        integer src_lane;
        integer src_addr;
        logic [WORD_W-1:0] tmp;
        begin
            tmp = '0;
            for (lane = 0; lane < PV_MAX; lane++) begin
                if ((lane < PC) && ((col_base + lane) < w_total)) begin
                    abs_col  = col_base + lane;
                    src_grp  = abs_col / src_pack;
                    src_lane = abs_col % src_pack;
                    src_addr = abs_row * src_groups + src_grp;
                    if ((src_addr < DEPTH) && (mem_tag[bank][src_addr] == layer_tag) && mem_fill[bank][src_addr][src_lane])
                        tmp[lane*DATA_W +: DATA_W] = mem_data[bank][src_addr][src_lane*DATA_W +: DATA_W];
                end
            end
            build_special_m1_to_m2_word = tmp;
        end
    endfunction

    // ============================================================
    // Latch layer configuration / stream state / write path
    // ============================================================
    always_ff @(posedge clk or negedge rst_n) begin
        integer i_tok;
        integer pf_idx;
        integer x;
        integer x_prev;
        integer prev_col;
        integer prev_grp;
        logic   first_lane_for_word;
        integer ch;
        integer row;
        integer col;
        integer grp;
        integer lane;
        integer addr;
        integer valid_pf;
        integer slot;
        integer free_slot;
        logic found_dup;
        logic [PV_MAX-1:0] exp_keep;
        logic signed [DATA_W-1:0] px1;
        logic signed [DATA_W-1:0] px2;

        logic [PTOTAL-1:0] nxt_m1_touch_v;
        logic [15:0]       nxt_m1_touch_bank   [0:PTOTAL-1];
        logic [15:0]       nxt_m1_touch_row    [0:PTOTAL-1];
        logic [15:0]       nxt_m1_touch_colgrp [0:PTOTAL-1];

        logic [PF-1:0]     nxt_m2_touch_v;
        logic [15:0]       nxt_m2_touch_bank   [0:PF-1];
        logic [15:0]       nxt_m2_touch_row    [0:PF-1];
        logic [15:0]       nxt_m2_touch_colgrp [0:PF-1];

        if (!rst_n) begin
            src_mode_q         <= 1'b0;
            next_mode_q        <= 1'b0;
            h_out_q            <= '0;
            w_out_q            <= '0;
            f_out_q            <= '0;
            pv_cur_q           <= '0;
            pf_cur_q           <= '0;
            pv_next_q          <= '0;
            pf_next_q          <= '0;
            src_pack_q         <= 16'd1;
            store_pack_q       <= 16'd1;
            stored_groups_q    <= 16'd0;
            total_pixels_q     <= '0;
            pixels_written_q   <= '0;
            layer_write_done_q <= 1'b0;
            layer_tag_q        <= '0;
            error_q            <= 1'b0;

            strm_mode_q        <= STRM_IDLE;
            strm_active_q      <= 1'b0;
            strm_row_base_q    <= '0;
            strm_num_rows_q    <= '0;
            strm_col_base_q    <= '0;
            strm_row_q         <= '0;
            strm_ch_q          <= '0;
            strm_colgrp_q      <= '0;
            strm_m1_row_slot_q <= '0;
            strm_m1_ch_blk_q   <= '0;
            strm_m2_cgrp_q     <= '0;
            ifm_stream_done    <= 1'b0;

            m1_touch_v_q       <= '0;
            m2_touch_v_q       <= '0;
            m1_sm_ready_valid  <= '0;
            m2_sm_ready_valid  <= '0;
            for (i_tok = 0; i_tok < PTOTAL; i_tok++) begin
                m1_touch_bank_q[i_tok]   <= '0;
                m1_touch_row_q[i_tok]    <= '0;
                m1_touch_colgrp_q[i_tok] <= '0;
                m1_sm_ready_bank[i_tok]   <= '0;
                m1_sm_ready_row_g[i_tok]  <= '0;
                m1_sm_ready_colgrp_g[i_tok] <= '0;
            end
            for (i_tok = 0; i_tok < PF; i_tok++) begin
                m2_touch_bank_q[i_tok]   <= '0;
                m2_touch_row_q[i_tok]    <= '0;
                m2_touch_colgrp_q[i_tok] <= '0;
                m2_sm_ready_bank[i_tok]    <= '0;
                m2_sm_ready_row_g[i_tok]   <= '0;
                m2_sm_ready_colbase_g[i_tok] <= '0;
            end
        end
        else begin
            ifm_stream_done   <= 1'b0;
            m1_sm_ready_valid <= '0;
            m2_sm_ready_valid <= '0;

            // ----------------------------------------------------
            // Emit ready tokens for words touched in the previous cycle
            // ----------------------------------------------------
            if (!error_q) begin
                if (!src_mode_q && !next_mode_q) begin
                    integer ch_blk;
                    integer ch_base;
                    integer ch_last;
                    integer ch_chk;
                    integer ch_rel;
                    integer block_ready;
                    for (i_tok = 0; i_tok < PTOTAL; i_tok++) begin
                        if (m1_touch_v_q[i_tok]) begin
                            row      = m1_touch_row_q[i_tok];
                            grp      = m1_touch_colgrp_q[i_tok];
                            ch_blk   = m1_touch_bank_q[i_tok];
                            addr     = row * stored_groups_q + grp;
                            exp_keep = calc_keep_mask(pv_next_q, grp * pv_next_q, w_out_q);

                            block_ready = 1;
                            if ((pf_next_q == 0) || (pf_next_q > PTOTAL) || (addr >= DEPTH))
                                block_ready = 0;
                            else begin
                                ch_base = ch_blk * pf_next_q;
                                ch_last = ch_base + pf_next_q;
                                if (ch_base >= f_out_q)
                                    block_ready = 0;
                                else begin
                                    // Vivado synthesis needs a statically bounded loop.
                                    // pf_next_q/f_out_q are runtime values, so loop over
                                    // the maximum possible group size and guard inside.
                                    for (ch_rel = 0; ch_rel < PTOTAL; ch_rel++) begin
                                        ch_chk = ch_base + ch_rel;
                                        if ((ch_rel < pf_next_q) && (ch_chk < f_out_q)) begin
                                            if ((mem_tag[ch_chk][addr] != layer_tag_q) ||
                                                ((mem_fill[ch_chk][addr] & exp_keep) != exp_keep))
                                                block_ready = 0;
                                        end
                                    end
                                end
                            end

                            if (block_ready) begin
                                m1_sm_ready_valid[i_tok]      <= 1'b1;
                                m1_sm_ready_bank[i_tok]       <= ch_blk[15:0];
                                m1_sm_ready_row_g[i_tok]      <= row[15:0];
                                m1_sm_ready_colgrp_g[i_tok]   <= grp[15:0];
                            end
                        end
                    end
                end

                if (src_mode_q && next_mode_q) begin
                    integer cgrp;
                    integer c_base;
                    integer c_last;
                    integer ch_chk;
                    integer c_rel;
                    integer group_ready;
                    for (i_tok = 0; i_tok < PF; i_tok++) begin
                        if (m2_touch_v_q[i_tok]) begin
                            row      = m2_touch_row_q[i_tok];
                            grp      = m2_touch_colgrp_q[i_tok];
                            cgrp     = m2_touch_bank_q[i_tok];
                            addr     = row * stored_groups_q + grp;
                            exp_keep = calc_keep_mask(PC, grp * PC, w_out_q);

                            group_ready = 1;
                            if ((addr >= DEPTH) || (PC == 0))
                                group_ready = 0;
                            else begin
                                c_base = cgrp * PC;
                                c_last = c_base + PC;
                                if (c_base >= f_out_q)
                                    group_ready = 0;
                                else begin
                                    // PC is a parameter, so this loop has a static bound.
                                    for (c_rel = 0; c_rel < PC; c_rel++) begin
                                        ch_chk = c_base + c_rel;
                                        if (ch_chk < f_out_q) begin
                                            if ((mem_tag[ch_chk][addr] != layer_tag_q) ||
                                                ((mem_fill[ch_chk][addr] & exp_keep) != exp_keep))
                                                group_ready = 0;
                                        end
                                    end
                                end
                            end

                            if (group_ready) begin
                                m2_sm_ready_valid[i_tok]        <= 1'b1;
                                m2_sm_ready_bank[i_tok]         <= cgrp[15:0];
                                m2_sm_ready_row_g[i_tok]        <= row[15:0];
                                m2_sm_ready_colbase_g[i_tok]    <= (grp * PC);
                            end
                        end
                    end
                end
            end

            // defaults for newly collected touched-word sets
            nxt_m1_touch_v = '0;
            nxt_m2_touch_v = '0;
            for (i_tok = 0; i_tok < PTOTAL; i_tok++) begin
                nxt_m1_touch_bank[i_tok]   = '0;
                nxt_m1_touch_row[i_tok]    = '0;
                nxt_m1_touch_colgrp[i_tok] = '0;
            end
            for (i_tok = 0; i_tok < PF; i_tok++) begin
                nxt_m2_touch_bank[i_tok]   = '0;
                nxt_m2_touch_row[i_tok]    = '0;
                nxt_m2_touch_colgrp[i_tok] = '0;
            end

            if (layer_start) begin
                src_mode_q         <= cfg_src_mode;
                next_mode_q        <= cfg_next_mode;
                h_out_q            <= cfg_h_out;
                w_out_q            <= cfg_w_out;
                f_out_q            <= cfg_f_out;
                pv_cur_q           <= cfg_pv_cur;
                pf_cur_q           <= cfg_pf_cur;
                pv_next_q          <= cfg_pv_next;
                pf_next_q          <= cfg_pf_next;
                src_pack_q         <= mode1_src_pack(cfg_pv_cur, cfg_pool_en);

                if (!cfg_src_mode && !cfg_next_mode)
                    store_pack_q    <= (cfg_pv_next == 0) ? 16'd1 : cfg_pv_next;
                else if (!cfg_src_mode && cfg_next_mode)
                    store_pack_q    <= mode1_src_pack(cfg_pv_cur, cfg_pool_en);
                else
                    store_pack_q    <= PC;

                if (!cfg_src_mode && !cfg_next_mode)
                    stored_groups_q <= ceil_div_u32(cfg_w_out, (cfg_pv_next == 0) ? 1 : cfg_pv_next);
                else if (!cfg_src_mode && cfg_next_mode)
                    stored_groups_q <= ceil_div_u32(cfg_w_out, mode1_src_pack(cfg_pv_cur, cfg_pool_en));
                else
                    stored_groups_q <= ceil_div_u32(cfg_w_out, PC);
                total_pixels_q     <= cfg_f_out * cfg_h_out * cfg_w_out;
                pixels_written_q   <= '0;
                layer_write_done_q <= 1'b0;
                layer_tag_q        <= layer_tag_q + 1'b1;
                error_q            <= 1'b0;

                if ((cfg_h_out == 0) || (cfg_w_out == 0) || (cfg_f_out == 0))
                    error_q <= 1'b1;
                if ((cfg_src_mode == 1'b0) && (cfg_next_mode == 1'b0) && (cfg_pv_next == 0))
                    error_q <= 1'b1;
                if (cfg_src_mode && !cfg_next_mode)
                    error_q <= 1'b1;
                if ((cfg_h_out * ceil_div_u32(cfg_w_out,
                        (!cfg_src_mode && !cfg_next_mode) ? ((cfg_pv_next == 0) ? 1 : cfg_pv_next) :
                        ((!cfg_src_mode && cfg_next_mode) ? mode1_src_pack(cfg_pv_cur, cfg_pool_en) : PC))) > DEPTH)
                    error_q <= 1'b1;

                strm_mode_q        <= STRM_IDLE;
                strm_active_q      <= 1'b0;
                strm_row_base_q    <= '0;
                strm_num_rows_q    <= '0;
                strm_col_base_q    <= '0;
                strm_row_q         <= '0;
                strm_ch_q          <= '0;
                strm_colgrp_q      <= '0;
                strm_m1_row_slot_q <= '0;
                strm_m1_ch_blk_q   <= '0;
                strm_m2_cgrp_q     <= '0;

                m1_touch_v_q       <= '0;
                m2_touch_v_q       <= '0;
            end
            else begin
                if (ifm_stream_start && !strm_active_q) begin
                    strm_active_q   <= 1'b1;
                    strm_row_base_q <= ifm_stream_row_base;
                    strm_num_rows_q <= ifm_stream_num_rows;
                    strm_col_base_q <= ifm_stream_col_base;
                    strm_row_q      <= '0;
                    strm_ch_q       <= '0;
                    strm_colgrp_q   <= '0;
                    strm_m1_row_slot_q <= ifm_stream_m1_row_slot_l;
                    strm_m1_ch_blk_q   <= ifm_stream_m1_ch_blk_g;
                    strm_m2_cgrp_q     <= ifm_stream_m2_cgrp_g;

                    if (!src_mode_q && !next_mode_q)
                        strm_mode_q <= STRM_M1_DIRECT;
                    else if (src_mode_q && next_mode_q)
                        strm_mode_q <= STRM_M2_DIRECT;
                    else if (!src_mode_q && next_mode_q)
                        strm_mode_q <= STRM_M1_TO_M2;
                    else begin
                        strm_mode_q   <= STRM_IDLE;
                        strm_active_q <= 1'b0;
                    end
                end
                else if (strm_active_q && ifm_ofm_wr_en && ifm_ofm_wr_ready) begin
                    case (strm_mode_q)
                        STRM_M1_DIRECT: begin
                            integer m1_blk_span;
                            integer m1_ch_base;
                            integer m1_num_ch;
                            m1_blk_span = (pf_next_q == 0) ? 1 : pf_next_q;
                            m1_ch_base  = strm_m1_ch_blk_q * m1_blk_span;
                            m1_num_ch   = (m1_ch_base >= f_out_q) ? 0 : (f_out_q - m1_ch_base);
                            if (m1_num_ch > m1_blk_span)
                                m1_num_ch = m1_blk_span;

                            if (strm_ch_q + 1 < m1_num_ch[15:0]) begin
                                strm_ch_q <= strm_ch_q + 1'b1;
                            end
                            else begin
                                strm_ch_q       <= '0;
                                strm_active_q   <= 1'b0;
                                strm_mode_q     <= STRM_IDLE;
                                ifm_stream_done <= 1'b1;
                            end
                        end

                        STRM_M2_DIRECT: begin
                            integer m2_ch_base;
                            integer m2_num_ch;
                            m2_ch_base = strm_m2_cgrp_q * PC;
                            m2_num_ch  = (m2_ch_base >= f_out_q) ? 0 : (f_out_q - m2_ch_base);
                            if (m2_num_ch > PC)
                                m2_num_ch = PC;

                            if (strm_ch_q + 1 < m2_num_ch[15:0]) begin
                                strm_ch_q <= strm_ch_q + 1'b1;
                            end
                            else begin
                                strm_ch_q <= '0;
                                if (strm_row_q + 1 < strm_num_rows_q) begin
                                    strm_row_q <= strm_row_q + 1'b1;
                                end
                                else begin
                                    strm_active_q   <= 1'b0;
                                    strm_mode_q     <= STRM_IDLE;
                                    ifm_stream_done <= 1'b1;
                                end
                            end
                        end

                        STRM_M1_TO_M2: begin
                            if (strm_row_q + 1 < strm_num_rows_q) begin
                                strm_row_q <= strm_row_q + 1'b1;
                            end
                            else begin
                                strm_row_q <= '0;
                                if (strm_ch_q + 1 < f_out_q) begin
                                    strm_ch_q <= strm_ch_q + 1'b1;
                                end
                                else begin
                                    strm_active_q   <= 1'b0;
                                    strm_mode_q     <= STRM_IDLE;
                                    ifm_stream_done <= 1'b1;
                                end
                            end
                        end
                        default: begin
                            strm_active_q <= 1'b0;
                            strm_mode_q   <= STRM_IDLE;
                        end
                    endcase
                end

                // ------------------------------
                // Source mode 1 write decode
                // ------------------------------
                if (!error_q && !src_mode_q && m1_wr_en) begin
                    for (pf_idx = 0; pf_idx < PTOTAL; pf_idx++) begin
                        ch = m1_wr_filter_base + pf_idx;
                        if ((pf_idx < pf_cur_q) && (ch < f_out_q)) begin
                            for (x = 0; x < PTOTAL; x++) begin
                                if ((x < src_pack_q) && (((pf_idx * src_pack_q) + x) < m1_wr_count)) begin
                                    row     = m1_wr_row;
                                    col     = m1_wr_col_base + x;
                                    grp     = col / store_pack_q;
                                    lane    = col % store_pack_q;
                                    addr    = row * stored_groups_q + grp;
                                    px1     = sat_m1(m1_wr_data[(pf_idx * src_pack_q) + x]);

                                    if ((row < h_out_q) && (col < w_out_q) && (addr < DEPTH) && (lane < PV_MAX)) begin
                                        // A single source write can update multiple lanes of the same
                                        // stored word when src_pack_q > 1, and can also merge multiple
                                        // source writes into one wider next-layer word when store_pack_q
                                        // > src_pack_q.  Clear a newly-tagged word only once per target
                                        // word in this write transaction; otherwise later lanes would be
                                        // wiped by repeated whole-word clears scheduled in the same cycle.
                                        first_lane_for_word = 1'b1;
                                        for (x_prev = 0; x_prev < PTOTAL; x_prev++) begin
                                            if (x_prev < x) begin
                                                prev_col = m1_wr_col_base + x_prev;
                                                prev_grp = (store_pack_q == 0) ? 0 : (prev_col / store_pack_q);
                                                if ((x_prev < src_pack_q) &&
                                                    (((pf_idx * src_pack_q) + x_prev) < m1_wr_count) &&
                                                    (prev_col < w_out_q) &&
                                                    (prev_grp == grp)) begin
                                                    first_lane_for_word = 1'b0;
                                                end
                                            end
                                        end

                                        if ((mem_tag[ch][addr] !== layer_tag_q) && first_lane_for_word) begin
                                            mem_tag[ch][addr]  <= layer_tag_q;
                                            mem_data[ch][addr] <= '0;
                                            mem_fill[ch][addr] <= '0;
                                        end
                                        mem_data[ch][addr][lane*DATA_W +: DATA_W] <= px1;
                                        mem_fill[ch][addr][lane] <= 1'b1;

                                        if (!next_mode_q) begin
                                            integer ch_blk_id;
                                            if (pf_next_q == 0)
                                                ch_blk_id = 0;
                                            else
                                                ch_blk_id = ch / pf_next_q;

                                            found_dup = 1'b0;
                                            free_slot = -1;
                                            for (slot = 0; slot < PTOTAL; slot++) begin
                                                if (nxt_m1_touch_v[slot] &&
                                                    (nxt_m1_touch_bank[slot] == ch_blk_id[15:0]) &&
                                                    (nxt_m1_touch_row[slot] == row[15:0]) &&
                                                    (nxt_m1_touch_colgrp[slot] == grp[15:0]))
                                                    found_dup = 1'b1;
                                                if (!nxt_m1_touch_v[slot] && (free_slot < 0))
                                                    free_slot = slot;
                                            end
                                            if (!found_dup && (free_slot >= 0)) begin
                                                nxt_m1_touch_v[free_slot]      = 1'b1;
                                                nxt_m1_touch_bank[free_slot]   = ch_blk_id[15:0];
                                                nxt_m1_touch_row[free_slot]    = row[15:0];
                                                nxt_m1_touch_colgrp[free_slot] = grp[15:0];
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                    pixels_written_q <= (pixels_written_q + m1_wr_count >= total_pixels_q) ? total_pixels_q : (pixels_written_q + m1_wr_count);
                    if ((pixels_written_q + m1_wr_count) >= total_pixels_q)
                        layer_write_done_q <= 1'b1;
                end

                // ------------------------------
                // Source mode 2 write decode
                // ------------------------------
                if (!error_q && src_mode_q && m2_wr_en) begin
                    valid_pf = (m2_wr_f_base + PF <= f_out_q) ? PF : (f_out_q - m2_wr_f_base);
                    if (valid_pf < 0)
                        valid_pf = 0;

                    for (pf_idx = 0; pf_idx < PF; pf_idx++) begin
                        ch   = m2_wr_f_base + pf_idx;
                        row  = m2_wr_row;
                        col  = m2_wr_col;
                        grp  = col / store_pack_q;
                        lane = col % store_pack_q;
                        addr = row * stored_groups_q + grp;
                        px2  = sat_m2(m2_wr_data[pf_idx*M2_IN_W +: M2_IN_W]);

                        if ((pf_idx < valid_pf) && (row < h_out_q) && (col < w_out_q) && (addr < DEPTH) && (lane < PV_MAX)) begin
                            if (mem_tag[ch][addr] != layer_tag_q) begin
                                mem_tag[ch][addr]  <= layer_tag_q;
                                mem_data[ch][addr] <= '0;
                                mem_fill[ch][addr] <= '0;
                            end
                            mem_data[ch][addr][lane*DATA_W +: DATA_W] <= px2;
                            mem_fill[ch][addr][lane] <= 1'b1;

                            if (next_mode_q) begin
                                integer cgrp_id;
                                cgrp_id = ch / PC;

                                found_dup = 1'b0;
                                free_slot = -1;
                                for (slot = 0; slot < PF; slot++) begin
                                    if (nxt_m2_touch_v[slot] &&
                                        (nxt_m2_touch_bank[slot] == cgrp_id[15:0]) &&
                                        (nxt_m2_touch_row[slot] == row[15:0]) &&
                                        (nxt_m2_touch_colgrp[slot] == grp[15:0]))
                                        found_dup = 1'b1;
                                    if (!nxt_m2_touch_v[slot] && (free_slot < 0))
                                        free_slot = slot;
                                end
                                if (!found_dup && (free_slot >= 0)) begin
                                    nxt_m2_touch_v[free_slot]      = 1'b1;
                                    nxt_m2_touch_bank[free_slot]   = cgrp_id[15:0];
                                    nxt_m2_touch_row[free_slot]    = row[15:0];
                                    nxt_m2_touch_colgrp[free_slot] = grp[15:0];
                                end
                            end
                        end
                    end

                    pixels_written_q <= (pixels_written_q + valid_pf >= total_pixels_q) ? total_pixels_q : (pixels_written_q + valid_pf);
                    if ((pixels_written_q + valid_pf) >= total_pixels_q)
                        layer_write_done_q <= 1'b1;
                end

                // register touched-word sets for next-cycle ready-token generation
                m1_touch_v_q <= nxt_m1_touch_v;
                m2_touch_v_q <= nxt_m2_touch_v;
                for (i_tok = 0; i_tok < PTOTAL; i_tok++) begin
                    m1_touch_bank_q[i_tok]   <= nxt_m1_touch_bank[i_tok];
                    m1_touch_row_q[i_tok]    <= nxt_m1_touch_row[i_tok];
                    m1_touch_colgrp_q[i_tok] <= nxt_m1_touch_colgrp[i_tok];
                end
                for (i_tok = 0; i_tok < PF; i_tok++) begin
                    m2_touch_bank_q[i_tok]   <= nxt_m2_touch_bank[i_tok];
                    m2_touch_row_q[i_tok]    <= nxt_m2_touch_row[i_tok];
                    m2_touch_colgrp_q[i_tok] <= nxt_m2_touch_colgrp[i_tok];
                end
            end
        end
    end

// ============================================================
    // IFM stream combinational path
    // ============================================================
    logic [15:0] abs_row_v;
    logic [15:0] abs_col_base_v;
    logic [15:0] phys_grp_v;
    logic [DEPTH_W-1:0] phys_addr_v;
    logic [PV_MAX-1:0]  expected_keep_v;
    logic               word_ready_v;
    logic [WORD_W-1:0]  stream_word_v;
    logic [BANK_W-1:0]  stream_bank_v;

    always_comb begin
        integer m1_blk_span_v;
        integer m1_ch_base_v;
        integer m2_ch_base_v;

        ifm_stream_busy    = strm_active_q;
        ifm_ofm_wr_en      = 1'b0;
        ifm_ofm_wr_bank    = '0;
        ifm_ofm_wr_row_idx = '0;
        ifm_ofm_wr_col_idx = '0;
        ifm_ofm_wr_data    = '0;
        ifm_ofm_wr_keep    = '0;

        abs_row_v       = '0;
        abs_col_base_v  = '0;
        phys_grp_v      = '0;
        phys_addr_v     = '0;
        expected_keep_v = '0;
        word_ready_v    = 1'b0;
        stream_word_v   = '0;
        stream_bank_v   = '0;

        m1_blk_span_v = (pf_next_q == 0) ? 1 : pf_next_q;
        m1_ch_base_v  = strm_m1_ch_blk_q * m1_blk_span_v;
        m2_ch_base_v  = strm_m2_cgrp_q * PC;

        if (strm_active_q && !error_q) begin
            case (strm_mode_q)
                STRM_M1_DIRECT: begin
                    abs_row_v       = strm_row_base_q + strm_row_q;
                    abs_col_base_v  = strm_col_base_q;
                    phys_grp_v      = (pv_next_q == 0) ? '0 : (strm_col_base_q / pv_next_q);
                    phys_addr_v     = (abs_row_v * stored_groups_q) + phys_grp_v;
                    expected_keep_v = calc_keep_mask((pv_next_q == 0) ? 16'd1 : pv_next_q, abs_col_base_v, w_out_q);
                    stream_bank_v = m1_ch_base_v + strm_ch_q;

                    if (((m1_ch_base_v + strm_ch_q) < f_out_q) &&
                        (abs_row_v < h_out_q) &&
                        (phys_addr_v < DEPTH)) begin
                        if ((mem_tag[stream_bank_v][phys_addr_v] == layer_tag_q) &&
                            ((mem_fill[stream_bank_v][phys_addr_v] & expected_keep_v) == expected_keep_v)) begin
                            word_ready_v  = 1'b1;
                            stream_word_v = mem_data[stream_bank_v][phys_addr_v];
                        end
                    end

                    ifm_ofm_wr_en      = word_ready_v;
                    ifm_ofm_wr_bank    = stream_bank_v;
                    ifm_ofm_wr_row_idx = strm_m1_row_slot_q;
                    ifm_ofm_wr_col_idx = phys_grp_v[COLIDX_W-1:0];
                    ifm_ofm_wr_data    = stream_word_v;
                    ifm_ofm_wr_keep    = expected_keep_v;
                end

                STRM_M2_DIRECT: begin
                    abs_row_v       = strm_row_base_q + strm_row_q;
                    abs_col_base_v  = strm_col_base_q;
                    phys_grp_v      = strm_col_base_q / PC;
                    phys_addr_v     = (abs_row_v * stored_groups_q) + phys_grp_v;
                    expected_keep_v = calc_keep_mask(PC, strm_col_base_q, w_out_q);
                    stream_bank_v = m2_ch_base_v + strm_ch_q;

                    if (((m2_ch_base_v + strm_ch_q) < f_out_q) &&
                        (strm_row_q < strm_num_rows_q) &&
                        (abs_row_v < h_out_q) &&
                        (phys_addr_v < DEPTH)) begin
                        if ((mem_tag[stream_bank_v][phys_addr_v] == layer_tag_q) &&
                            ((mem_fill[stream_bank_v][phys_addr_v] & expected_keep_v) == expected_keep_v)) begin
                            word_ready_v  = 1'b1;
                            stream_word_v = mem_data[stream_bank_v][phys_addr_v];
                        end
                    end

                    ifm_ofm_wr_en      = word_ready_v;
                    ifm_ofm_wr_bank    = stream_bank_v;
                    ifm_ofm_wr_row_idx = abs_row_v[ROW_W-1:0];
                    ifm_ofm_wr_col_idx = '0;
                    ifm_ofm_wr_data    = stream_word_v;
                    ifm_ofm_wr_keep    = expected_keep_v;
                end

                STRM_M1_TO_M2: begin
                    abs_row_v       = strm_row_base_q + strm_row_q;
                    expected_keep_v = calc_keep_mask(PC, strm_col_base_q, w_out_q);
                    stream_bank_v   = strm_ch_q[BANK_W-1:0];

                    if ((strm_ch_q < f_out_q) &&
                        (strm_row_q < strm_num_rows_q) &&
                        (abs_row_v < h_out_q) &&
                        layer_write_done_q) begin
                        word_ready_v  = 1'b1;
                        stream_word_v = build_special_m1_to_m2_word(stream_bank_v, abs_row_v, strm_col_base_q, w_out_q, src_pack_q, stored_groups_q, layer_tag_q);
                    end

                    ifm_ofm_wr_en      = word_ready_v;
                    ifm_ofm_wr_bank    = stream_bank_v;
                    ifm_ofm_wr_row_idx = abs_row_v[ROW_W-1:0];
                    ifm_ofm_wr_col_idx = '0;
                    ifm_ofm_wr_data    = stream_word_v;
                    ifm_ofm_wr_keep    = expected_keep_v;
                end

                default: begin
                end
            endcase
        end
    end


    // ============================================================
    // DMA linear readback
    // ============================================================
    always_ff @(posedge clk or negedge rst_n) begin
        integer lin;
        integer words_per_ch;
        integer ch;
        integer rem;
        integer row;
        integer grp;
        integer addr;
        logic [PV_MAX-1:0] keep_v;
        if (!rst_n) begin
            dma_valid_q <= 1'b0;
            dma_data_q  <= '0;
            dma_keep_q  <= '0;
        end
        else begin
            dma_valid_q <= ofm_dma_rd_en;
            dma_data_q  <= '0;
            dma_keep_q  <= '0;

            if (ofm_dma_rd_en) begin
                lin = ofm_dma_rd_addr;
                words_per_ch = h_out_q * stored_groups_q;
                if ((lin < layer_num_words) && (words_per_ch != 0)) begin
                    ch  = lin / words_per_ch;
                    rem = lin % words_per_ch;
                    row = rem / stored_groups_q;
                    grp = rem % stored_groups_q;
                    addr = row * stored_groups_q + grp;
                    keep_v = calc_keep_mask(store_pack_q, grp * store_pack_q, w_out_q);

                    if ((ch < f_out_q) && (addr < DEPTH) && (mem_tag[ch][addr] == layer_tag_q)) begin
                        dma_data_q <= mem_data[ch][addr];
                        dma_keep_q <= keep_v & mem_fill[ch][addr];
                    end
                end
            end
        end
    end

endmodule
