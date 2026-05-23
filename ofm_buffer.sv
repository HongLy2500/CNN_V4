// -----------------------------------------------------------------------------
// OFM data bank RAM: Vivado BRAM-friendly simple dual-port template.
// All address/data/enable muxing is done in the parent ofm_buffer; this module
// only sees one write port and one registered read port, matching UG901 style.
// -----------------------------------------------------------------------------
module ofm_data_bram_sdp #(
    parameter int DATA_W = 8,
    parameter int LANES  = 8,
    parameter int DEPTH  = 128,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
)(
    input  logic clk,

    input  logic wr_en,
    input  logic [ADDR_W-1:0] wr_addr,
    input  logic [LANES-1:0] wr_keep,
    input  logic [LANES*DATA_W-1:0] wr_data,

    input  logic rd_en,
    input  logic [ADDR_W-1:0] rd_addr,
    output logic [LANES*DATA_W-1:0] rd_data
);

    (* ram_style = "block" *) logic [LANES*DATA_W-1:0] ram [0:DEPTH-1];

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

// ofm_buffer datachbank + synthesis/elaboration friendly version generated from ofm_buffer_elab.sv
module ofm_buffer #(
    parameter int DATA_W   = 8,      // stored OFM width
    parameter int M1_IN_W  = DATA_W, // input width from pooling_mode1/compute top
    parameter int M2_IN_W  = DATA_W, // input width from pooling_mode2/compute top
    parameter int PV_MAX   = 8,
    parameter int OFM_ROW_STRIDE = 4,
    parameter int PC       = 8,
    parameter int PF       = 16,
    parameter int PTOTAL   = 128,
    parameter int C_MAX    = 64,
    parameter int H_MAX    = 32,
    parameter int W_MAX    = 32,
    // OFM storage is banked, but the logical meaning of a bank depends on
    // the active layout. Do not interpret every bank as one channel in all
    // modes:
    // - Mode1/source-mode1 layout:
    //     bank = output channel/filter,
    //     addr = row*OFM_ROW_STRIDE + spatial_group,
    //     lanes = PV-style spatial pixels.
    // - Mode2/source-mode2 layout:
    //     bank = fgrp*PC + col_l, where fgrp=floor(filter/PC)
    //            and col_l=global_col%PC,
    //     addr = row*OFM_ROW_STRIDE + col_g, where col_g=global_col/PC,
    //     lanes = PC output-channel/filter lanes inside that fgrp.
    // - M1->M2 transition keeps the full Mode1 OFM first, then repacks it
    //   into the Mode2 IFM word shape through the special stream path.
    //
    // Row-aligned storage: every logical OFM row starts at a fixed physical
    // stride. For the selected small benchmark W_MAX=32 and PV_MAX=PC=8,
    // OFM_ROW_STRIDE=4 stores one full 32-pixel row per channel/fgrp-col_l
    // bank without cross-row or cross-layer aliasing.
    parameter int DEPTH    = H_MAX * OFM_ROW_STRIDE,
    parameter int TAG_W    = 8
)(
    input  logic clk,
    input  logic rst_n,

    // ============================================================
    // Layer configuration
    // cfg_src_mode  = 0: write stream comes from mode 1 pooling or no-pool bypass
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
    // For mode-1 source writes:
    //   cfg_pool_en = 1 -> input comes from pooling_mode1, source pack = max(Pv_cur/2, 1)
    //   cfg_pool_en = 0 -> input comes from no-pool bypass, source pack = 1 spatial pixel/write
    // For non-mode1 sources this signal is ignored. Treat X/Z as enabled to preserve legacy behavior.
    input  logic                        cfg_pool_en,
    input  logic [$clog2(H_MAX+1)-1:0]  cfg_h_out,
    input  logic [$clog2(W_MAX+1)-1:0]  cfg_w_out,
    input  logic [9:0]                  cfg_f_out,
    input  logic [7:0]                  cfg_pv_cur,   // current layer Pv before pooling (mode 1 only)
    input  logic [7:0]                  cfg_pf_cur,   // current layer Pf (mode 1 only)
    input  logic [7:0]                  cfg_pv_next,  // next layer Pv (mode 1 only)
    input  logic [7:0]                  cfg_pf_next,  // next layer Pf/grouping for same-mode mode1

    // ============================================================
    // Write input from mode1 pooling / compute path
    // Lane mapping:
    //   pool_en=1: lane = pf * max(Pv_cur/2, 1) + x  // pooling output
    //   pool_en=0: lane = pf                         // no-pool direct bypass, one x per write
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
    //     col_base is the exact global spatial column for one Mode2 refill entry
    // ============================================================
    input logic [$clog2(H_MAX+1)-1:0]   ifm_stream_row_base,
    input logic [$clog2(H_MAX+1)-1:0]   ifm_stream_num_rows,
    input logic [$clog2(W_MAX+1)-1:0]   ifm_stream_col_base,
    input  logic                        ifm_stream_start,
    // Explicit stream command identity from control_unit_top.
    // Do not infer stream type from the current layer cfg_src_mode/cfg_next_mode:
    // a same-mode runtime refill from the previous layer may still be active
    // while the current layer's next mode is already different.
    // Encoding matches strm_mode_t below:
    //   0: idle/invalid, 1: M1 direct, 2: M2 direct, 3: M1->M2 transition
    input  logic [1:0]                  ifm_stream_kind,
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
    output logic [$clog2(W_MAX)-1:0]    ifm_ofm_wr_col_g,
    output logic                        ifm_ofm_wr_mode2,
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
    // Maximum number of distinct Mode1 same-mode OFM words that one write beat can touch.
    // A pooled Mode1 beat carries at most ceil(PV_MAX/2) spatial outputs per PF lane;
    // a no-pool beat carries one. Keeping this lower than PTOTAL reduces ready-token
    // bookkeeping fanout without changing the external PTOTAL-wide ready interface.
    localparam int M1_TOUCH_SLOTS_RAW = PF * (((PV_MAX + 1) / 2) < 1 ? 1 : ((PV_MAX + 1) / 2));
    localparam int M1_TOUCH_SLOTS     = (M1_TOUCH_SLOTS_RAW > PTOTAL) ? PTOTAL : M1_TOUCH_SLOTS_RAW;

    typedef enum logic [1:0] {
        STRM_IDLE,
        STRM_M1_DIRECT,
        STRM_M2_DIRECT,
        STRM_M1_TO_M2
    } strm_mode_t;

    // Input stream-kind encoding. Keep these equal to strm_mode_t encodings
    // and to control_unit_top's OFM_STRM_* constants.
    localparam logic [1:0] IFM_KIND_IDLE      = 2'd0;
    localparam logic [1:0] IFM_KIND_M1_DIRECT = 2'd1;
    localparam logic [1:0] IFM_KIND_M2_DIRECT = 2'd2;
    localparam logic [1:0] IFM_KIND_M1_TO_M2  = 2'd3;

    // ============================================================
    // Physical storage
    // ============================================================
    // Metadata storage mirrors OFM data banking: one metadata RAM per
    // logical OFM bank.  This replaces the previous grouped
    // mem_fill_g*/mem_tag_g* arrays.  The grouped organization packed
    // four logical banks into one memory, so a single OFM beat could
    // create multiple writes into the same metadata array after Vivado
    // unrolled the commit loop.  Per-bank metadata gives each array at
    // most one write command per clock, matching the way mem_data is
    // banked and making distributed-RAM inference much more reliable.
    //
    // Metadata word format: {tag[TAG_W-1:0], fill[PV_MAX-1:0]}.
    localparam int META_W          = TAG_W + PV_MAX;
    localparam int META_BANKS_IMPL = 64;
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b0  [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b1  [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b2  [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b3  [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b4  [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b5  [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b6  [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b7  [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b8  [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b9  [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b10 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b11 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b12 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b13 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b14 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b15 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b16 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b17 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b18 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b19 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b20 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b21 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b22 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b23 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b24 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b25 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b26 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b27 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b28 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b29 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b30 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b31 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b32 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b33 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b34 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b35 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b36 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b37 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b38 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b39 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b40 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b41 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b42 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b43 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b44 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b45 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b46 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b47 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b48 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b49 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b50 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b51 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b52 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b53 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b54 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b55 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b56 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b57 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b58 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b59 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b60 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b61 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b62 [0:DEPTH-1];
    (* ram_style = "distributed" *) logic [META_W-1:0] mem_meta_b63 [0:DEPTH-1];

    // ============================================================
    // Synthesizable synchronous read ports for mem_data
    // ============================================================
    // The previous datachbank versions still read mem_data through
    // combinational functions.  Vivado cannot infer BRAM from that usage.
    // This bank-read fabric is the only synthesizable read path for mem_data:
    // one registered read port per logical bank.  Stream/DMA paths issue
    // requests into data_rd_* and consume data_bank_rdata_q one cycle later.
    localparam int DATA_BANKS_IMPL = 64;
    localparam int DATA_BANK_W     = 6;

    logic [DATA_BANKS_IMPL-1:0] data_rd_en_v;
    logic [DEPTH_W-1:0]         data_rd_addr_v [0:DATA_BANKS_IMPL-1];
    logic [WORD_W-1:0]          data_bank_rdata_q [0:DATA_BANKS_IMPL-1];

    // BRAM write ports are generated combinationally from the current OFM
    // write inputs.  This keeps mem_data writes aligned with the existing
    // mem_fill/mem_tag updates, while each RAM instance still sees a clean
    // UG901-style write/read port.
    logic [DATA_BANKS_IMPL-1:0] data_wr_en_c;
    logic [DEPTH_W-1:0]         data_wr_addr_c [0:DATA_BANKS_IMPL-1];
    logic [WORD_W-1:0]          data_wr_data_c [0:DATA_BANKS_IMPL-1];
    logic [PV_MAX-1:0]          data_wr_keep_c [0:DATA_BANKS_IMPL-1];

    genvar data_bank_gen;
    generate
        for (data_bank_gen = 0; data_bank_gen < DATA_BANKS_IMPL; data_bank_gen++) begin : G_OFM_DATA_BRAM
            ofm_data_bram_sdp #(
                .DATA_W(DATA_W),
                .LANES (PV_MAX),
                .DEPTH (DEPTH),
                .ADDR_W(DEPTH_W)
            ) u_data_bram (
                .clk    (clk),
                .wr_en  (data_wr_en_c[data_bank_gen]),
                .wr_addr(data_wr_addr_c[data_bank_gen]),
                .wr_keep(data_wr_keep_c[data_bank_gen]),
                .wr_data(data_wr_data_c[data_bank_gen]),
                .rd_en  (data_rd_en_v[data_bank_gen]),
                .rd_addr(data_rd_addr_v[data_bank_gen]),
                .rd_data(data_bank_rdata_q[data_bank_gen])
            );
        end
    endgenerate

    // Read one compact metadata word from a logical OFM bank.
    // Keep this as an async metadata read to preserve the simulation
    // timing of the functional baseline; only the storage organization
    // has changed from grouped arrays to per-bank arrays.
    function automatic logic [META_W-1:0] ofm_mem_meta_read(input int bank, input int addr);
        begin
            ofm_mem_meta_read = '0;
            if ((bank >= 0) && (bank < C_MAX) && (bank < META_BANKS_IMPL) &&
                (addr >= 0) && (addr < DEPTH)) begin
                case (bank)
                    0:  ofm_mem_meta_read = mem_meta_b0 [addr];
                    1:  ofm_mem_meta_read = mem_meta_b1 [addr];
                    2:  ofm_mem_meta_read = mem_meta_b2 [addr];
                    3:  ofm_mem_meta_read = mem_meta_b3 [addr];
                    4:  ofm_mem_meta_read = mem_meta_b4 [addr];
                    5:  ofm_mem_meta_read = mem_meta_b5 [addr];
                    6:  ofm_mem_meta_read = mem_meta_b6 [addr];
                    7:  ofm_mem_meta_read = mem_meta_b7 [addr];
                    8:  ofm_mem_meta_read = mem_meta_b8 [addr];
                    9:  ofm_mem_meta_read = mem_meta_b9 [addr];
                    10:  ofm_mem_meta_read = mem_meta_b10[addr];
                    11:  ofm_mem_meta_read = mem_meta_b11[addr];
                    12:  ofm_mem_meta_read = mem_meta_b12[addr];
                    13:  ofm_mem_meta_read = mem_meta_b13[addr];
                    14:  ofm_mem_meta_read = mem_meta_b14[addr];
                    15:  ofm_mem_meta_read = mem_meta_b15[addr];
                    16:  ofm_mem_meta_read = mem_meta_b16[addr];
                    17:  ofm_mem_meta_read = mem_meta_b17[addr];
                    18:  ofm_mem_meta_read = mem_meta_b18[addr];
                    19:  ofm_mem_meta_read = mem_meta_b19[addr];
                    20:  ofm_mem_meta_read = mem_meta_b20[addr];
                    21:  ofm_mem_meta_read = mem_meta_b21[addr];
                    22:  ofm_mem_meta_read = mem_meta_b22[addr];
                    23:  ofm_mem_meta_read = mem_meta_b23[addr];
                    24:  ofm_mem_meta_read = mem_meta_b24[addr];
                    25:  ofm_mem_meta_read = mem_meta_b25[addr];
                    26:  ofm_mem_meta_read = mem_meta_b26[addr];
                    27:  ofm_mem_meta_read = mem_meta_b27[addr];
                    28:  ofm_mem_meta_read = mem_meta_b28[addr];
                    29:  ofm_mem_meta_read = mem_meta_b29[addr];
                    30:  ofm_mem_meta_read = mem_meta_b30[addr];
                    31:  ofm_mem_meta_read = mem_meta_b31[addr];
                    32:  ofm_mem_meta_read = mem_meta_b32[addr];
                    33:  ofm_mem_meta_read = mem_meta_b33[addr];
                    34:  ofm_mem_meta_read = mem_meta_b34[addr];
                    35:  ofm_mem_meta_read = mem_meta_b35[addr];
                    36:  ofm_mem_meta_read = mem_meta_b36[addr];
                    37:  ofm_mem_meta_read = mem_meta_b37[addr];
                    38:  ofm_mem_meta_read = mem_meta_b38[addr];
                    39:  ofm_mem_meta_read = mem_meta_b39[addr];
                    40:  ofm_mem_meta_read = mem_meta_b40[addr];
                    41:  ofm_mem_meta_read = mem_meta_b41[addr];
                    42:  ofm_mem_meta_read = mem_meta_b42[addr];
                    43:  ofm_mem_meta_read = mem_meta_b43[addr];
                    44:  ofm_mem_meta_read = mem_meta_b44[addr];
                    45:  ofm_mem_meta_read = mem_meta_b45[addr];
                    46:  ofm_mem_meta_read = mem_meta_b46[addr];
                    47:  ofm_mem_meta_read = mem_meta_b47[addr];
                    48:  ofm_mem_meta_read = mem_meta_b48[addr];
                    49:  ofm_mem_meta_read = mem_meta_b49[addr];
                    50:  ofm_mem_meta_read = mem_meta_b50[addr];
                    51:  ofm_mem_meta_read = mem_meta_b51[addr];
                    52:  ofm_mem_meta_read = mem_meta_b52[addr];
                    53:  ofm_mem_meta_read = mem_meta_b53[addr];
                    54:  ofm_mem_meta_read = mem_meta_b54[addr];
                    55:  ofm_mem_meta_read = mem_meta_b55[addr];
                    56:  ofm_mem_meta_read = mem_meta_b56[addr];
                    57:  ofm_mem_meta_read = mem_meta_b57[addr];
                    58:  ofm_mem_meta_read = mem_meta_b58[addr];
                    59:  ofm_mem_meta_read = mem_meta_b59[addr];
                    60:  ofm_mem_meta_read = mem_meta_b60[addr];
                    61:  ofm_mem_meta_read = mem_meta_b61[addr];
                    62:  ofm_mem_meta_read = mem_meta_b62[addr];
                    63:  ofm_mem_meta_read = mem_meta_b63[addr];
                    default: ofm_mem_meta_read = '0;
                endcase
            end
        end
    endfunction

    function automatic logic [PV_MAX-1:0] ofm_mem_fill_read(input int bank, input int addr);
        logic [META_W-1:0] meta_word;
        begin
            meta_word = ofm_mem_meta_read(bank, addr);
            ofm_mem_fill_read = meta_word[PV_MAX-1:0];
        end
    endfunction

    function automatic logic [TAG_W-1:0] ofm_mem_tag_read(input int bank, input int addr);
        logic [META_W-1:0] meta_word;
        begin
            meta_word = ofm_mem_meta_read(bank, addr);
            ofm_mem_tag_read = meta_word[PV_MAX +: TAG_W];
        end
    endfunction

    function automatic logic ofm_mem_fill_lane_read(input int bank, input int addr, input int lane);
        logic [PV_MAX-1:0] f;
        begin
            f = ofm_mem_fill_read(bank, addr);
            if ((lane >= 0) && (lane < PV_MAX))
                ofm_mem_fill_lane_read = f[lane];
            else
                ofm_mem_fill_lane_read = 1'b0;
        end
    endfunction





    // Lane-write version for mem_data. This is the important synthesis cleanup:
    // the write datapath no longer has to read the old data word just to
    // preserve untouched lanes. Untouched lanes keep their stored BRAM value;
    // Data write accumulator macro. Expanded only inside the main clocked process.
    // It keeps tag/fill behavior unchanged, but accumulates mem_data into one write port per bank.
// Metadata update accumulator macros.  The previous eventready version
    // wrote mem_fill/mem_tag directly from the M1/M2 write loops.  That expands
    // a large number of dynamic write cases in Vivado.  These macros only
    // collect one metadata command per logical bank; the single commit section
    // near the end of the clocked block performs the actual mem_fill/mem_tag
    // update.
`define OFM_MEM_WRITE_LANES_ACCUM(BANK_EXPR, ADDR_EXPR, TAG_EXPR, DATA_EXPR, FILL_EXPR, KEEP_EXPR) \
    begin \
        if (((BANK_EXPR) >= 0) && ((BANK_EXPR) < C_MAX) && ((BANK_EXPR) < DATA_BANKS_IMPL) && \
            ((ADDR_EXPR) >= 0) && ((ADDR_EXPR) < DEPTH)) begin \
            meta_set_en_l[(BANK_EXPR)]   = 1'b1; \
            meta_set_addr_l[(BANK_EXPR)] = (ADDR_EXPR); \
            meta_set_tag_l[(BANK_EXPR)]  = (TAG_EXPR); \
            meta_set_fill_l[(BANK_EXPR)] = (FILL_EXPR); \
        end \
    end

`define OFM_META_CLEAR_ACCUM(BANK_EXPR, ADDR_EXPR, MASK_EXPR) \
    begin \
        if (((BANK_EXPR) >= 0) && ((BANK_EXPR) < C_MAX) && ((BANK_EXPR) < DATA_BANKS_IMPL) && \
            ((ADDR_EXPR) >= 0) && ((ADDR_EXPR) < DEPTH)) begin \
            if (!meta_set_en_l[(BANK_EXPR)] || (meta_set_addr_l[(BANK_EXPR)] != (ADDR_EXPR))) begin \
                meta_clr_en_l[(BANK_EXPR)]   = 1'b1; \
                meta_clr_addr_l[(BANK_EXPR)] = (ADDR_EXPR); \
                meta_clr_mask_l[(BANK_EXPR)] = meta_clr_mask_l[(BANK_EXPR)] | (MASK_EXPR); \
            end \
        end \
    end

    // Legacy direct metadata-clear task removed: stream consume now uses
    // OFM_META_CLEAR_ACCUM and the centralized metadata commit section.

    // ============================================================
    // Latched layer configuration
    // ============================================================
    logic         src_mode_q;
    logic         next_mode_q;
    logic [$clog2(H_MAX+1)-1:0] h_out_q;
    logic [$clog2(W_MAX+1)-1:0] w_out_q;
    logic [7:0]   f_out_q;
    logic [7:0]   pv_cur_q, pf_cur_q, pv_next_q, pf_next_q;
    // Timing helper for mode-1 OFM write path.
    // Decode cfg_pf_cur once at layer_start instead of using pf_cur_q in the
    // high-fanout write/update logic for mem_data/mem_fill. Functionally this
    // is equivalent to (pf_idx < pf_cur_q) for all PTOTAL-bounded write lanes.
    (* max_fanout = 32 *) logic [PF-1:0] pf_cur_active_q;
    logic [15:0]  src_pack_q;      // mode1 source pack: pooled Pv if pool_en=1, 1 if no-pool bypass
    logic [15:0]  store_pack_q;    // pack used by stored words for this layer
    logic [15:0]  stored_groups_q; // valid compact groups per row; physical row pitch is OFM_ROW_STRIDE
    logic [31:0]  total_pixels_q;
    logic [31:0]  pixels_written_q;
    logic         layer_write_done_q;
    logic [TAG_W-1:0] layer_tag_q;
    // Tag of the layer that was active immediately before the current layer.
    // Used by OFM->IFM runtime refill after layer advance: the OFM source
    // still belongs to the previous layer while the current layer may already
    // be writing new OFM data with layer_tag_q.
    logic [TAG_W-1:0] prev_layer_tag_q;
    // Geometry/layout of the previous layer's OFM storage. These values are
    // needed for OFM->IFM runtime refill after the scheduler advances to the
    // next layer: the current layer config may already describe the new OFM,
    // while the stream source still belongs to the previous layer.
    logic [$clog2(H_MAX+1)-1:0] prev_h_out_q;
    logic [$clog2(W_MAX+1)-1:0] prev_w_out_q;
    logic [7:0]                 prev_f_out_q;
    logic [7:0]                 prev_pv_next_q;
    logic [7:0]                 prev_pf_next_q;
    logic [15:0]                prev_stored_groups_q; // previous layer valid compact groups per row
    logic [15:0]                prev_store_pack_q;     // previous layer physical spatial pack
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

    // Shared IFM stream combinational helpers.
    // These signals are consumed by both the stream issue combinational logic
    // and the stream-accept clear logic inside the main clocked process, so
    // declare them before the clocked block to avoid Vivado's used-before-
    // declaration warnings during synthesis.
    logic [15:0] abs_row_v;
    logic [15:0] abs_col_base_v;
    logic [15:0] phys_grp_v;
    logic [DEPTH_W-1:0] phys_addr_v;
    logic [PV_MAX-1:0]  expected_keep_v;
    logic               word_ready_v;
    logic [WORD_W-1:0]  stream_word_v;
    logic [BANK_W-1:0]  stream_bank_v;
    logic [TAG_W-1:0]   stream_src_tag_v;
    logic               m1_to_m2_runtime_exact_v;

    // ============================================================
    // Mode-2 direct stream source selection
    // ============================================================
    // Match Mode 1 behavior: STRM_M2_DIRECT first tries the current layer
    // tag/geometry and, if that word is not ready, falls back to the previous
    // completed layer tag/geometry.  No extra source-select command is used.

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
    logic [M1_TOUCH_SLOTS-1:0] m1_touch_v_q;
    logic [15:0]              m1_touch_bank_q   [0:M1_TOUCH_SLOTS-1];
    logic [15:0]              m1_touch_row_q    [0:M1_TOUCH_SLOTS-1];
    logic [15:0]              m1_touch_colgrp_q [0:M1_TOUCH_SLOTS-1];

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

    // Physical address for row-aligned OFM storage.
    // valid_groups/stored_groups_q still control how many groups are meaningful
    // per row; OFM_ROW_STRIDE only controls physical row spacing.
    function automatic [31:0] ofm_phys_addr(input [31:0] row, input [31:0] grp);
        begin
            ofm_phys_addr = (row * OFM_ROW_STRIDE) + grp;
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


    // Small-pack helpers for synthesis-friendly Mode1 OFM write decode.
    // store_pack_q/src_pack_q are expected to be small powers of two in the
    // supported schedules. Defaults intentionally avoid generic div/mod hardware.
    function automatic integer pack_div_int(
        input integer val,
        input integer pack
    );
        begin
            case (pack)
                0:  pack_div_int = 0;
                1:  pack_div_int = val;
                2:  pack_div_int = val >> 1;
                4:  pack_div_int = val >> 2;
                8:  pack_div_int = val >> 3;
                16: pack_div_int = val >> 4;
                32: pack_div_int = val >> 5;
                64: pack_div_int = val >> 6;
                128: pack_div_int = val >> 7;
                default: pack_div_int = 0;
            endcase
        end
    endfunction

    function automatic integer pack_mod_int(
        input integer val,
        input integer pack
    );
        begin
            case (pack)
                0:  pack_mod_int = 0;
                1:  pack_mod_int = 0;
                2:  pack_mod_int = val & 1;
                4:  pack_mod_int = val & 3;
                8:  pack_mod_int = val & 7;
                16: pack_mod_int = val & 15;
                32: pack_mod_int = val & 31;
                64: pack_mod_int = val & 63;
                128: pack_mod_int = val & 127;
                default: pack_mod_int = 0;
            endcase
        end
    endfunction

    function automatic integer ceil_div_by_active_pf(
        input integer numer,
        input integer denom
    );
        begin
            case (denom)
                0:  ceil_div_by_active_pf = 0;
                1:  ceil_div_by_active_pf = numer;
                2:  ceil_div_by_active_pf = (numer + 1) >> 1;
                3:  ceil_div_by_active_pf = (numer + 2) / 3;
                4:  ceil_div_by_active_pf = (numer + 3) >> 2;
                5:  ceil_div_by_active_pf = (numer + 4) / 5;
                6:  ceil_div_by_active_pf = (numer + 5) / 6;
                7:  ceil_div_by_active_pf = (numer + 6) / 7;
                8:  ceil_div_by_active_pf = (numer + 7) >> 3;
                9:  ceil_div_by_active_pf = (numer + 8) / 9;
                10: ceil_div_by_active_pf = (numer + 9) / 10;
                11: ceil_div_by_active_pf = (numer + 10) / 11;
                12: ceil_div_by_active_pf = (numer + 11) / 12;
                13: ceil_div_by_active_pf = (numer + 12) / 13;
                14: ceil_div_by_active_pf = (numer + 13) / 14;
                15: ceil_div_by_active_pf = (numer + 14) / 15;
                16: ceil_div_by_active_pf = (numer + 15) >> 4;
                default: ceil_div_by_active_pf = 0;
            endcase
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

    // Simulation-only legacy helper; not called by synthesizable datapath.

    // ============================================================
    // BRAM data write port generation
    // ============================================================
    // This is intentionally separate from the large clocked control block.
    // The RAM instances see only muxed port signals: wr_en, wr_addr,
    // wr_keep, wr_data.  The sequential block below still updates mem_tag
    // and mem_fill using the original logic, so logical readiness/layer tags
    // remain unchanged.
`define OFM_DATA_WR_ACCUM(BANK_EXPR, ADDR_EXPR, DATA_EXPR, KEEP_EXPR) \
    begin \
        if (((BANK_EXPR) >= 0) && ((BANK_EXPR) < C_MAX) && ((BANK_EXPR) < DATA_BANKS_IMPL) && \
            ((ADDR_EXPR) >= 0) && ((ADDR_EXPR) < DEPTH)) begin \
            if ((!data_wr_en_c[(BANK_EXPR)]) || (data_wr_addr_c[(BANK_EXPR)] != (ADDR_EXPR))) begin \
                data_wr_en_c[(BANK_EXPR)]   = 1'b1; \
                data_wr_addr_c[(BANK_EXPR)] = (ADDR_EXPR); \
                data_wr_data_c[(BANK_EXPR)] = '0; \
                data_wr_keep_c[(BANK_EXPR)] = '0; \
            end \
            for (data_lane_i = 0; data_lane_i < PV_MAX; data_lane_i++) begin \
                if (KEEP_EXPR[data_lane_i]) begin \
                    data_wr_keep_c[(BANK_EXPR)][data_lane_i] = 1'b1; \
                    data_wr_data_c[(BANK_EXPR)][data_lane_i*DATA_W +: DATA_W] = DATA_EXPR[data_lane_i*DATA_W +: DATA_W]; \
                end \
            end \
        end \
    end

    always_comb begin : GEN_OFM_DATA_WRITE_PORTS
        integer bank_i;
        integer data_lane_i;
        integer pf_idx;
        integer slot;
        integer x;
        integer ch;
        integer row;
        integer col;
        integer grp;
        integer col_offset;
        integer lane;
        integer addr;
        integer valid_pf_m1;
        integer valid_pf;
        integer valid_x_m1;
        integer max_x_m1;
        integer src_lane_idx;
        integer compact_lane_idx;
        integer first_grp;
        integer last_grp;
        integer grp_rel;
        integer fgrp_id;
        integer m1_fgrp_id;
        integer flane_id;
        integer col_l_id;
        integer col_g_id;
        integer bank_id;
        integer bank_col_l;
        logic word_has_write;
        logic [PV_MAX-1:0] word_write_keep;
        logic [WORD_W-1:0] word_data_next;
        logic [TAG_W-1:0]  wr_meta_tag;
        logic [PV_MAX-1:0] wr_meta_fill;
        logic signed [DATA_W-1:0] px1;
        logic signed [DATA_W-1:0] px2;

        data_wr_en_c = '0;
        for (bank_i = 0; bank_i < DATA_BANKS_IMPL; bank_i++) begin
            data_wr_addr_c[bank_i] = '0;
            data_wr_data_c[bank_i] = '0;
            data_wr_keep_c[bank_i] = '0;
        end

        if (!error_q && !src_mode_q && m1_wr_en) begin
            valid_pf_m1 = 0;
            for (slot = 0; slot < PF; slot++) begin
                if (pf_cur_active_q[slot] && ((m1_wr_filter_base + slot) < f_out_q))
                    valid_pf_m1 = valid_pf_m1 + 1;
            end

            // These fields are common to all Mode1 write banks in this beat.
            // Keep them outside the bank loop so Vivado does not replicate the
            // divider/modulo cone for every OFM bank.
            valid_x_m1 = ceil_div_by_active_pf(m1_wr_count, valid_pf_m1);
            if (m1_wr_col_base >= w_out_q)
                max_x_m1 = 0;
            else begin
                max_x_m1 = w_out_q - m1_wr_col_base;
                if (max_x_m1 > src_pack_q)
                    max_x_m1 = src_pack_q;
            end
            if (valid_x_m1 > max_x_m1)
                valid_x_m1 = max_x_m1;

            grp        = pack_div_int(m1_wr_col_base, store_pack_q);
            col_offset = pack_mod_int(m1_wr_col_base, store_pack_q);
            row        = m1_wr_row;
            addr       = ofm_phys_addr(row, grp);

            // Bank-oriented Mode1 data write.
            // The previous packed version looped over PF and wrote data_wr_*[ch]
            // through a dynamic bank index.  That made Vivado build a large
            // PF-to-64-bank scatter mux.  Here each unrolled bank_i owns its
            // data_wr_* assignment directly; pf_idx is only used as the source
            // lane selector for that bank.
            // Synthesis cleanup: decode the Mode1 filter group like Mode2
            // decodes {fgrp, lane}.  The previous bank-oriented version used
            //     pf_idx = bank_i - m1_wr_filter_base
            // for every OFM bank; with integer temporaries Vivado expanded that
            // into a large number of 32-bit subtract/compare adders.  In this
            // P128 configuration PF is a power-of-two group size and
            // m1_wr_filter_base is expected to be PF-aligned, so each bank can
            // derive its local PF lane from bank_i % PF and compare group IDs.
            m1_fgrp_id = (PF == 0) ? 0 : (m1_wr_filter_base / PF);

            if ((row < h_out_q) && (grp < stored_groups_q) && (addr < DEPTH)) begin
                for (bank_i = 0; bank_i < DATA_BANKS_IMPL; bank_i++) begin
                    fgrp_id = (PF == 0) ? 0 : (bank_i / PF);
                    pf_idx  = (PF == 0) ? 0 : (bank_i % PF);
                    ch      = bank_i;

                    if ((bank_i < C_MAX) && (fgrp_id == m1_fgrp_id) &&
                        (pf_idx < PF) && (pf_idx < valid_pf_m1) && pf_cur_active_q[pf_idx] &&
                        (ch < f_out_q)) begin
                        word_has_write  = 1'b0;
                        word_write_keep = '0;
                        word_data_next  = '0;

                        for (x = 0; x < PV_MAX; x++) begin
                            src_lane_idx     = (pf_idx * src_pack_q) + x;
                            compact_lane_idx = (pf_idx * valid_x_m1) + x;
                            lane             = col_offset + x;
                            if ((x < valid_x_m1) &&
                                (compact_lane_idx < m1_wr_count) &&
                                (src_lane_idx < PTOTAL) &&
                                (lane < store_pack_q) &&
                                (lane < PV_MAX)) begin
                                col = m1_wr_col_base + x;
                                if (col < w_out_q) begin
                                    px1 = sat_m1(m1_wr_data[src_lane_idx]);
                                    word_data_next[lane*DATA_W +: DATA_W] = px1;
                                    word_write_keep[lane] = 1'b1;
                                    word_has_write = 1'b1;
                                end
                            end
                        end

                        if (word_has_write) begin
                            data_wr_en_c[bank_i]   = 1'b1;
                            data_wr_addr_c[bank_i] = addr;
                            data_wr_data_c[bank_i] = word_data_next;
                            data_wr_keep_c[bank_i] = word_write_keep;
                        end
                    end
                end
            end
        end

        if (!error_q && src_mode_q && m2_wr_en) begin
            valid_pf = (m2_wr_f_base + PF <= f_out_q) ? PF : (f_out_q - m2_wr_f_base);
            if (valid_pf < 0)
                valid_pf = 0;

            row      = m2_wr_row;
            col      = m2_wr_col;
            col_l_id = (PC == 0) ? 0 : (col % PC);
            col_g_id = (PC == 0) ? 0 : (col / PC);
            addr     = ofm_phys_addr(row, col_g_id);

            // Synthesis-friendly Mode2 data write path.
            // Old code scattered each PF scalar into data_wr_* through a dynamic
            // bank index.  That creates large 64-bank muxes during cross-boundary
            // optimization.  Here the loop is bank-oriented: each unrolled bank
            // receives at most one packed word write for the current Mode2 beat.
            if ((row < h_out_q) && (col < w_out_q) && (addr < DEPTH) && (PC > 0)) begin
                for (bank_i = 0; bank_i < DATA_BANKS_IMPL; bank_i++) begin
                    fgrp_id    = bank_i / PC;
                    bank_col_l = bank_i % PC;

                    if ((bank_i < C_MAX) && (bank_col_l == col_l_id) && ((fgrp_id * PC) < f_out_q)) begin
                        word_has_write  = 1'b0;
                        word_write_keep = '0;
                        word_data_next  = '0;

                        for (x = 0; x < PF; x++) begin
                            ch       = m2_wr_f_base + x;
                            flane_id = (PC == 0) ? 0 : (ch % PC);

                            if ((x < valid_pf) && (ch < f_out_q) &&
                                ((ch / PC) == fgrp_id) && (flane_id < PV_MAX)) begin
                                px2 = sat_m2(m2_wr_data[x*M2_IN_W +: M2_IN_W]);
                                word_data_next[flane_id*DATA_W +: DATA_W] = px2;
                                word_write_keep[flane_id] = 1'b1;
                                word_has_write = 1'b1;
                            end
                        end

                        if (word_has_write) begin
                            data_wr_en_c[bank_i]   = 1'b1;
                            data_wr_addr_c[bank_i] = addr;
                            data_wr_data_c[bank_i] = word_data_next;
                            data_wr_keep_c[bank_i] = word_write_keep;
                        end
                    end
                end
            end
        end
    end

    // ============================================================
    // Latch layer configuration / stream state / write path
    // ============================================================
    always_ff @(posedge clk) begin
        integer i_tok;
        integer pf_idx;
        integer x;
        integer ch;
        integer row;
        integer col;
        integer grp;
        integer col_offset;
        integer lane;
        integer addr;
        integer valid_pf;
        integer cfg_store_pack_v;
        integer cfg_groups_v;
        integer valid_pf_m1;
        integer valid_x_m1;
        integer max_x_m1;
        integer src_lane_idx;
        integer compact_lane_idx;
        integer slot;
        integer free_slot;
        integer first_grp;
        integer last_grp;
        integer grp_rel;
        logic found_dup;
        logic word_has_write;
        logic [PV_MAX-1:0] exp_keep;
        logic [PV_MAX-1:0] word_fill_next;
        logic [PV_MAX-1:0] word_write_keep;
        logic [WORD_W-1:0] word_data_next;
        logic [TAG_W-1:0]  wr_meta_tag;
        logic [PV_MAX-1:0] wr_meta_fill;
        logic signed [DATA_W-1:0] px1;
        logic signed [DATA_W-1:0] px2;

        integer lane_i;
        integer bank_i;

        logic [M1_TOUCH_SLOTS-1:0] nxt_m1_touch_v;
        logic [15:0]       nxt_m1_touch_bank   [0:M1_TOUCH_SLOTS-1];
        logic [15:0]       nxt_m1_touch_row    [0:M1_TOUCH_SLOTS-1];
        logic [15:0]       nxt_m1_touch_colgrp [0:M1_TOUCH_SLOTS-1];

        logic [PF-1:0]     nxt_m2_touch_v;
        logic [15:0]       nxt_m2_touch_bank   [0:PF-1];
        logic [15:0]       nxt_m2_touch_row    [0:PF-1];
        logic [15:0]       nxt_m2_touch_colgrp [0:PF-1];

        // Centralized metadata update commands. These are automatic
        // combinational temporaries inside the clocked block; they collect all
        // fill/tag writes requested by stream-consume clear and OFM write paths.
        logic [DATA_BANKS_IMPL-1:0] meta_set_en_l;
        logic [DEPTH_W-1:0]         meta_set_addr_l [0:DATA_BANKS_IMPL-1];
        logic [TAG_W-1:0]           meta_set_tag_l  [0:DATA_BANKS_IMPL-1];
        logic [PV_MAX-1:0]          meta_set_fill_l [0:DATA_BANKS_IMPL-1];
        logic [DATA_BANKS_IMPL-1:0] meta_clr_en_l;
        logic [DEPTH_W-1:0]         meta_clr_addr_l [0:DATA_BANKS_IMPL-1];
        logic [PV_MAX-1:0]          meta_clr_mask_l [0:DATA_BANKS_IMPL-1];
        integer                     meta_bank_i;
        logic                       meta_wr_do_v;
        logic [DEPTH_W-1:0]         meta_wr_addr_v;
        logic [META_W-1:0]          meta_old_word_v;
        logic [META_W-1:0]          meta_wr_word_v;

        if (!rst_n) begin
            src_mode_q         <= 1'b0;
            next_mode_q        <= 1'b0;
            h_out_q            <= '0;
            w_out_q            <= '0;
            f_out_q            <= '0;
            pv_cur_q           <= '0;
            pf_cur_q           <= '0;
            pf_cur_active_q    <= '0;
            pv_next_q          <= '0;
            pf_next_q          <= '0;
            src_pack_q         <= 16'd1;
            store_pack_q       <= 16'd1;
            stored_groups_q    <= 16'd0;
            total_pixels_q     <= '0;
            pixels_written_q   <= '0;
            layer_write_done_q <= 1'b0;
            layer_tag_q        <= '0;
            prev_layer_tag_q   <= '0;
            prev_h_out_q       <= '0;
            prev_w_out_q       <= '0;
            prev_f_out_q       <= '0;
            prev_pv_next_q     <= '0;
            prev_pf_next_q     <= '0;
            prev_stored_groups_q <= '0;
            prev_store_pack_q <= '0;
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
            for (i_tok = 0; i_tok < M1_TOUCH_SLOTS; i_tok++) begin
                m1_touch_bank_q[i_tok]   <= '0;
                m1_touch_row_q[i_tok]    <= '0;
                m1_touch_colgrp_q[i_tok] <= '0;
                m1_sm_ready_bank[i_tok]   <= '0;
                m1_sm_ready_row_g[i_tok]  <= '0;
                m1_sm_ready_colgrp_g[i_tok] <= '0;
            end
            // Preserve reset behavior for the unused upper PTOTAL-wide output slots.
            // These slots never assert valid, but clearing their payload avoids X-noise in simulation/waveforms.
            for (i_tok = M1_TOUCH_SLOTS; i_tok < PTOTAL; i_tok++) begin
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
            // mem_data BRAM read/write ports are handled by G_OFM_DATA_BRAM instances.

            ifm_stream_done   <= 1'b0;
            m1_sm_ready_valid <= '0;
            m2_sm_ready_valid <= '0;

            // Default all metadata update commands to idle.  The actual
            // mem_fill/mem_tag arrays are committed once near the end of this
            // clocked block, not from the inner M1/M2 write loops.
            meta_set_en_l = '0;
            meta_clr_en_l = '0;
            for (meta_bank_i = 0; meta_bank_i < DATA_BANKS_IMPL; meta_bank_i++) begin
                meta_set_addr_l[meta_bank_i] = '0;
                meta_set_tag_l [meta_bank_i] = '0;
                meta_set_fill_l[meta_bank_i] = '0;
                meta_clr_addr_l[meta_bank_i] = '0;
                meta_clr_mask_l[meta_bank_i] = '0;
            end

            // ----------------------------------------------------
            // Event-driven same-mode ready tokens
            // ----------------------------------------------------
            // Previous versions scanned mem_fill/mem_tag for every touched word
            // one cycle after each write. That creates a very large dynamic mux
            // cone and makes Vivado elaboration/synthesis slow. In this version,
            // ready tokens are emitted directly from the write path below, using
            // the just-computed word_fill_next value. This keeps the external
            // one-cycle pulse interface, but avoids the metadata scan block.

            // defaults for newly collected touched-word sets
            nxt_m1_touch_v = '0;
            nxt_m2_touch_v = '0;
            for (i_tok = 0; i_tok < M1_TOUCH_SLOTS; i_tok++) begin
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
                for (i_tok = 0; i_tok < PF; i_tok++) begin
                    pf_cur_active_q[i_tok] <= (i_tok < cfg_pf_cur);
                end
                pv_next_q          <= cfg_pv_next;
                pf_next_q          <= cfg_pf_next;
                // Source packing for mode-1 writes depends on whether the layer used pooling.
                // Keep legacy behavior for pool_en=1 or X/Z; use one spatial pixel per write for no-pool bypass.
                if (!cfg_src_mode && (cfg_pool_en == 1'b0))
                    src_pack_q      <= 16'd1;
                else
                    src_pack_q      <= (cfg_pv_cur > 1) ? (cfg_pv_cur >> 1) : 16'd1;

                if (!cfg_src_mode && !cfg_next_mode) begin
                    // Same-mode M1->M1 stores in the next layer's Pv layout, independent of source packing.
                    cfg_store_pack_v = (cfg_pv_next == 0) ? 1 : cfg_pv_next;
                end
                else if (!cfg_src_mode && cfg_next_mode) begin
                    // M1->M2 transition stores in source-mode layout and later repacks to PC lanes.
                    // For no-pool bypass, the source layout is one pixel per OFM write.
                    cfg_store_pack_v = (cfg_pool_en == 1'b0) ? 1 : ((cfg_pv_cur > 1) ? (cfg_pv_cur >> 1) : 1);
                end
                else begin
                    cfg_store_pack_v = PC;
                end

                cfg_groups_v = ceil_div_u32(cfg_w_out, cfg_store_pack_v);
                store_pack_q    <= cfg_store_pack_v[15:0];
                stored_groups_q <= cfg_groups_v[15:0];
                total_pixels_q     <= cfg_f_out * cfg_h_out * cfg_w_out;
                pixels_written_q   <= '0;
                layer_write_done_q <= 1'b0;
                prev_layer_tag_q   <= layer_tag_q;
                prev_h_out_q       <= h_out_q;
                prev_w_out_q       <= w_out_q;
                prev_f_out_q       <= f_out_q;
                prev_pv_next_q     <= pv_next_q;
                prev_pf_next_q     <= pf_next_q;
                prev_stored_groups_q <= stored_groups_q;
                prev_store_pack_q  <= store_pack_q;
                layer_tag_q        <= layer_tag_q + 1'b1;
                error_q            <= 1'b0;

                if ((cfg_h_out == 0) || (cfg_w_out == 0) || (cfg_f_out == 0))
                    error_q <= 1'b1;
                if ((cfg_src_mode == 1'b0) && (cfg_next_mode == 1'b0) && (cfg_pv_next == 0))
                    error_q <= 1'b1;
                if (cfg_src_mode && !cfg_next_mode)
                    error_q <= 1'b1;
                // Row-aligned layout uses a fixed physical pitch for every row.
                // stored_groups_q/cfg_groups_v remains the number of meaningful groups.
                if ((OFM_ROW_STRIDE <= 0) || (cfg_groups_v <= 0) || (cfg_groups_v > OFM_ROW_STRIDE))
                    error_q <= 1'b1;
                if ((cfg_h_out != 0) && (((cfg_h_out - 1) * OFM_ROW_STRIDE + cfg_groups_v) > DEPTH))
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

                    // Stream kind is part of the command contract.
                    // The OFM buffer must not infer it from the current layer's
                    // cfg_src_mode/cfg_next_mode because a previous->current
                    // same-mode runtime refill can still be in flight after the
                    // current layer has advanced and its next mode has changed.
                    case (ifm_stream_kind)
                        IFM_KIND_M1_DIRECT: begin
                            strm_mode_q <= STRM_M1_DIRECT;
                        end
                        IFM_KIND_M2_DIRECT: begin
                            strm_mode_q <= STRM_M2_DIRECT;
                        end
                        IFM_KIND_M1_TO_M2: begin
                            strm_mode_q <= STRM_M1_TO_M2;
                        end
                        default: begin
                            strm_mode_q   <= STRM_IDLE;
                            strm_active_q <= 1'b0;
                        end
                    endcase
                end
                else if (strm_active_q && ifm_ofm_wr_en && ifm_ofm_wr_ready) begin
                    logic [TAG_W-1:0] accept_entry_tag;
                    accept_entry_tag = ofm_mem_tag_read(stream_bank_v, phys_addr_v);
                    // Once a same-mode stream word has been accepted by IFM buffer,
                    // the source OFM word is consumed. Clear its fill bits so the
                    // storage location can be reused by the next layer without being
                    // treated as still holding unconsumed previous-layer data.
                    // M1_TO_M2 builds a word from multiple source words, so it is not
                    // cleared here by this single-address logic.
                    if ((strm_mode_q == STRM_M1_DIRECT) &&
                        (stream_bank_v < C_MAX) &&
                        (phys_addr_v < DEPTH) &&
                        (accept_entry_tag == stream_src_tag_v)) begin
                        `OFM_META_CLEAR_ACCUM(stream_bank_v, phys_addr_v, expected_keep_v)
                    end

                    // Mirror the Mode1 consume-on-stream behavior for Mode2.
                    // A Mode2 stream command transfers one logical refill entry.
                    // With the Mode2 OFM layout used here, that entry is one word:
                    //   bank = cgrp*PC + col_l, addr = row*OFM_ROW_STRIDE + col_group,
                    //   lane = output-channel lane within the cgrp.
                    // Once IFM accepts that entry, clear the consumed lanes for the
                    // selected source tag so the storage can be reused safely.
                    if ((strm_mode_q == STRM_M2_DIRECT) &&
                        (stream_bank_v < C_MAX) &&
                        (phys_addr_v < DEPTH) &&
                        (accept_entry_tag == stream_src_tag_v)) begin
                        `OFM_META_CLEAR_ACCUM(stream_bank_v, phys_addr_v, ifm_ofm_wr_keep)
                    end

                    case (strm_mode_q)
                        STRM_M1_DIRECT: begin
                            integer m1_blk_span;
                            integer m1_ch_base;
                            integer m1_num_ch;
                            if (stream_src_tag_v == prev_layer_tag_q) begin
                                m1_blk_span = (prev_pf_next_q == 0) ? 1 : prev_pf_next_q;
                                m1_ch_base  = strm_m1_ch_blk_q * m1_blk_span;
                                m1_num_ch   = (m1_ch_base >= prev_f_out_q) ? 0 : (prev_f_out_q - m1_ch_base);
                            end
                            else begin
                                m1_blk_span = (pf_next_q == 0) ? 1 : pf_next_q;
                                m1_ch_base  = strm_m1_ch_blk_q * m1_blk_span;
                                m1_num_ch   = (m1_ch_base >= f_out_q) ? 0 : (f_out_q - m1_ch_base);
                            end
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
                            // New Mode2 refill contract: one control command streams exactly
                            // one IFM entry {row_g, col_g, cgrp_g}.  Do not auto-walk a
                            // PC-wide spatial segment here; control_unit_top issues the next
                            // exact column as a separate command after free/ready match.
                            strm_ch_q       <= '0;
                            strm_row_q      <= '0;
                            strm_active_q   <= 1'b0;
                            strm_mode_q     <= STRM_IDLE;
                            ifm_stream_done <= 1'b1;
                        end

                        STRM_M1_TO_M2: begin
                            if (strm_num_rows_q == 1) begin
                                // Runtime exact refill: one command transfers
                                // exactly one IFM Mode2 entry {row_g,col_g,cgrp_g}.
                                strm_ch_q       <= '0;
                                strm_colgrp_q   <= '0;
                                strm_row_q      <= '0;
                                strm_active_q   <= 1'b0;
                                strm_mode_q     <= STRM_IDLE;
                                ifm_stream_done <= 1'b1;
                            end
                            else begin
                                // Initial M1->M2 handoff walks the first
                                // resident PC columns for all rows/cgroups.
                                integer m1m2_num_cols;
                                integer m1m2_num_cgrps;

                                if (strm_col_base_q >= w_out_q)
                                    m1m2_num_cols = 0;
                                else if ((strm_col_base_q + PC) <= w_out_q)
                                    m1m2_num_cols = PC;
                                else
                                    m1m2_num_cols = w_out_q - strm_col_base_q;

                                m1m2_num_cgrps = (PC == 0) ? 0 : ceil_div_u32(f_out_q, PC);

                                if ((m1m2_num_cols <= 0) || (m1m2_num_cgrps <= 0)) begin
                                    strm_active_q   <= 1'b0;
                                    strm_mode_q     <= STRM_IDLE;
                                    ifm_stream_done <= 1'b1;
                                end
                                else if (strm_ch_q + 1 < m1m2_num_cgrps) begin
                                    strm_ch_q <= strm_ch_q + 1'b1;
                                end
                                else begin
                                    strm_ch_q <= '0;
                                    if (strm_colgrp_q + 1 < m1m2_num_cols) begin
                                        strm_colgrp_q <= strm_colgrp_q + 1'b1;
                                    end
                                    else begin
                                        strm_colgrp_q <= '0;
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
                            end
                        end
                        default: begin
                            strm_active_q <= 1'b0;
                            strm_mode_q   <= STRM_IDLE;
                        end
                    endcase
                end

                // ------------------------------
                // Source mode write auxiliary update
                //
                // Synthesis cleanup:
                // The BRAM write-port generator above already decoded the current
                // OFM write beat into one command per logical bank:
                //   data_wr_en_c[bank], data_wr_addr_c[bank],
                //   data_wr_keep_c[bank], data_wr_data_c[bank].
                //
                // Do not re-decode the Mode1/Mode2 input stream here.  Repeating the
                // PF/PV or PF/PC scatter logic in the clocked metadata/token path was
                // the dominant synthesis cone in the OFM buffer.  Instead, reuse the
                // bank-level write commands to update {tag,fill} metadata and emit
                // same-mode ready tokens.
                // ------------------------------

                if (!error_q && !src_mode_q && m1_wr_en) begin
                    for (bank_i = 0; bank_i < DATA_BANKS_IMPL; bank_i++) begin
                        if (data_wr_en_c[bank_i] &&
                            (bank_i < C_MAX) &&
                            (data_wr_addr_c[bank_i] < DEPTH)) begin
                            ch   = bank_i;
                            addr = data_wr_addr_c[bank_i];
                            row  = pack_div_int(addr, OFM_ROW_STRIDE);
                            grp  = pack_mod_int(addr, OFM_ROW_STRIDE);

                            wr_meta_tag  = ofm_mem_tag_read(bank_i, addr);
                            wr_meta_fill = ofm_mem_fill_read(bank_i, addr);
                            if (wr_meta_tag == layer_tag_q)
                                word_fill_next = wr_meta_fill | data_wr_keep_c[bank_i];
                            else
                                word_fill_next = data_wr_keep_c[bank_i];

                            `OFM_MEM_WRITE_LANES_ACCUM(bank_i, addr, layer_tag_q, '0, word_fill_next, data_wr_keep_c[bank_i])

                            if (!next_mode_q) begin
                                integer ch_blk_id;
                                logic [PV_MAX-1:0] ready_keep_v;

                                ready_keep_v = calc_keep_mask(pv_next_q, grp * pv_next_q, w_out_q);
                                if (pf_next_q == 0)
                                    ch_blk_id = 0;
                                else
                                    ch_blk_id = ch / pf_next_q;

                                if (((word_fill_next & ready_keep_v) == ready_keep_v) &&
                                    (ready_keep_v != '0)) begin
                                    found_dup = 1'b0;
                                    free_slot = -1;
                                    for (slot = 0; slot < M1_TOUCH_SLOTS; slot++) begin
                                        if (nxt_m1_touch_v[slot] &&
                                            (nxt_m1_touch_bank[slot] == ch_blk_id[15:0]) &&
                                            (nxt_m1_touch_row[slot] == row[15:0]) &&
                                            (nxt_m1_touch_colgrp[slot] == grp[15:0])) begin
                                            found_dup = 1'b1;
                                        end
                                        if (!nxt_m1_touch_v[slot] && (free_slot < 0)) begin
                                            free_slot = slot;
                                        end
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

                    pixels_written_q <= (pixels_written_q + m1_wr_count >= total_pixels_q) ? total_pixels_q : (pixels_written_q + m1_wr_count);
                    if ((pixels_written_q + m1_wr_count) >= total_pixels_q)
                        layer_write_done_q <= 1'b1;
                end

                if (!error_q && src_mode_q && m2_wr_en) begin
                    valid_pf = (m2_wr_f_base + PF <= f_out_q) ? PF : (f_out_q - m2_wr_f_base);
                    if (valid_pf < 0)
                        valid_pf = 0;

                    for (bank_i = 0; bank_i < DATA_BANKS_IMPL; bank_i++) begin
                        if (data_wr_en_c[bank_i] &&
                            (bank_i < C_MAX) &&
                            (data_wr_addr_c[bank_i] < DEPTH)) begin
                            integer fgrp_id;
                            integer col_l_id;
                            integer col_g_id;

                            addr     = data_wr_addr_c[bank_i];
                            row      = pack_div_int(addr, OFM_ROW_STRIDE);
                            col_g_id = pack_mod_int(addr, OFM_ROW_STRIDE);
                            if (PC == 0) begin
                                fgrp_id  = 0;
                                col_l_id = 0;
                                col      = 0;
                            end
                            else begin
                                fgrp_id  = bank_i / PC;
                                col_l_id = bank_i % PC;
                                col      = (col_g_id * PC) + col_l_id;
                            end

                            wr_meta_tag  = ofm_mem_tag_read(bank_i, addr);
                            wr_meta_fill = ofm_mem_fill_read(bank_i, addr);
                            if (wr_meta_tag == layer_tag_q)
                                word_fill_next = wr_meta_fill | data_wr_keep_c[bank_i];
                            else
                                word_fill_next = data_wr_keep_c[bank_i];

                            `OFM_MEM_WRITE_LANES_ACCUM(bank_i, addr, layer_tag_q, '0, word_fill_next, data_wr_keep_c[bank_i])

                            if (next_mode_q) begin
                                integer c_base_ev;
                                integer c_rel_ev;
                                integer ch_ev;
                                logic m2_word_ready_ev;

                                c_base_ev = fgrp_id * PC;
                                m2_word_ready_ev = 1'b1;
                                if ((PC == 0) || (c_base_ev >= f_out_q)) begin
                                    m2_word_ready_ev = 1'b0;
                                end
                                else begin
                                    for (c_rel_ev = 0; c_rel_ev < PC; c_rel_ev++) begin
                                        ch_ev = c_base_ev + c_rel_ev;
                                        if (ch_ev < f_out_q) begin
                                            if (!word_fill_next[c_rel_ev]) begin
                                                m2_word_ready_ev = 1'b0;
                                            end
                                        end
                                    end
                                end

                                if (m2_word_ready_ev) begin
                                    found_dup = 1'b0;
                                    free_slot = -1;
                                    for (slot = 0; slot < PF; slot++) begin
                                        if (nxt_m2_touch_v[slot] &&
                                            (nxt_m2_touch_bank[slot] == fgrp_id[15:0]) &&
                                            (nxt_m2_touch_row[slot] == row[15:0]) &&
                                            (nxt_m2_touch_colgrp[slot] == col[15:0])) begin
                                            found_dup = 1'b1;
                                        end
                                        if (!nxt_m2_touch_v[slot] && (free_slot < 0)) begin
                                            free_slot = slot;
                                        end
                                    end
                                    if (!found_dup && (free_slot >= 0)) begin
                                        nxt_m2_touch_v[free_slot]      = 1'b1;
                                        nxt_m2_touch_bank[free_slot]   = fgrp_id[15:0];
                                        nxt_m2_touch_row[free_slot]    = row[15:0];
                                        nxt_m2_touch_colgrp[free_slot] = col[15:0];
                                    end
                                end
                            end
                        end
                    end

                    pixels_written_q <= (pixels_written_q + valid_pf >= total_pixels_q) ? total_pixels_q : (pixels_written_q + valid_pf);
                    if ((pixels_written_q + valid_pf) >= total_pixels_q)
                        layer_write_done_q <= 1'b1;
                end

                // ----------------------------------------------------
                // Centralized metadata commit
                // ----------------------------------------------------
                // Preserve the functional-baseline priority: set/write wins
                // over clear when both target the same bank in one clock.
                // Unlike the grouped version, each mem_meta_b* array is one
                // logical bank and receives at most one assignment from this
                // single write command.  This avoids the mem_*_reg
                // set/reset-priority pattern that Vivado reported for the
                // grouped mem_fill/mem_tag and grouped mem_meta versions.
                for (meta_bank_i = 0; meta_bank_i < DATA_BANKS_IMPL; meta_bank_i++) begin
                    meta_wr_do_v   = 1'b0;
                    meta_wr_addr_v = '0;
                    meta_old_word_v = '0;
                    meta_wr_word_v = '0;

                    if ((meta_bank_i < C_MAX) && (meta_bank_i < META_BANKS_IMPL) &&
                        meta_set_en_l[meta_bank_i] &&
                        (meta_set_addr_l[meta_bank_i] < DEPTH)) begin
                        meta_wr_do_v   = 1'b1;
                        meta_wr_addr_v = meta_set_addr_l[meta_bank_i];
                        meta_wr_word_v = {meta_set_tag_l[meta_bank_i], meta_set_fill_l[meta_bank_i]};
                    end
                    else if ((meta_bank_i < C_MAX) && (meta_bank_i < META_BANKS_IMPL) &&
                             meta_clr_en_l[meta_bank_i] &&
                             (meta_clr_addr_l[meta_bank_i] < DEPTH)) begin
                        meta_old_word_v = ofm_mem_meta_read(meta_bank_i, meta_clr_addr_l[meta_bank_i]);
                        meta_wr_do_v    = 1'b1;
                        meta_wr_addr_v  = meta_clr_addr_l[meta_bank_i];
                        meta_wr_word_v  = {meta_old_word_v[PV_MAX +: TAG_W],
                                           (meta_old_word_v[PV_MAX-1:0] & ~meta_clr_mask_l[meta_bank_i])};
                    end

                    if (meta_wr_do_v) begin
                        case (meta_bank_i)
                            0:  mem_meta_b0 [meta_wr_addr_v] <= meta_wr_word_v;
                            1:  mem_meta_b1 [meta_wr_addr_v] <= meta_wr_word_v;
                            2:  mem_meta_b2 [meta_wr_addr_v] <= meta_wr_word_v;
                            3:  mem_meta_b3 [meta_wr_addr_v] <= meta_wr_word_v;
                            4:  mem_meta_b4 [meta_wr_addr_v] <= meta_wr_word_v;
                            5:  mem_meta_b5 [meta_wr_addr_v] <= meta_wr_word_v;
                            6:  mem_meta_b6 [meta_wr_addr_v] <= meta_wr_word_v;
                            7:  mem_meta_b7 [meta_wr_addr_v] <= meta_wr_word_v;
                            8:  mem_meta_b8 [meta_wr_addr_v] <= meta_wr_word_v;
                            9:  mem_meta_b9 [meta_wr_addr_v] <= meta_wr_word_v;
                            10:  mem_meta_b10[meta_wr_addr_v] <= meta_wr_word_v;
                            11:  mem_meta_b11[meta_wr_addr_v] <= meta_wr_word_v;
                            12:  mem_meta_b12[meta_wr_addr_v] <= meta_wr_word_v;
                            13:  mem_meta_b13[meta_wr_addr_v] <= meta_wr_word_v;
                            14:  mem_meta_b14[meta_wr_addr_v] <= meta_wr_word_v;
                            15:  mem_meta_b15[meta_wr_addr_v] <= meta_wr_word_v;
                            16:  mem_meta_b16[meta_wr_addr_v] <= meta_wr_word_v;
                            17:  mem_meta_b17[meta_wr_addr_v] <= meta_wr_word_v;
                            18:  mem_meta_b18[meta_wr_addr_v] <= meta_wr_word_v;
                            19:  mem_meta_b19[meta_wr_addr_v] <= meta_wr_word_v;
                            20:  mem_meta_b20[meta_wr_addr_v] <= meta_wr_word_v;
                            21:  mem_meta_b21[meta_wr_addr_v] <= meta_wr_word_v;
                            22:  mem_meta_b22[meta_wr_addr_v] <= meta_wr_word_v;
                            23:  mem_meta_b23[meta_wr_addr_v] <= meta_wr_word_v;
                            24:  mem_meta_b24[meta_wr_addr_v] <= meta_wr_word_v;
                            25:  mem_meta_b25[meta_wr_addr_v] <= meta_wr_word_v;
                            26:  mem_meta_b26[meta_wr_addr_v] <= meta_wr_word_v;
                            27:  mem_meta_b27[meta_wr_addr_v] <= meta_wr_word_v;
                            28:  mem_meta_b28[meta_wr_addr_v] <= meta_wr_word_v;
                            29:  mem_meta_b29[meta_wr_addr_v] <= meta_wr_word_v;
                            30:  mem_meta_b30[meta_wr_addr_v] <= meta_wr_word_v;
                            31:  mem_meta_b31[meta_wr_addr_v] <= meta_wr_word_v;
                            32:  mem_meta_b32[meta_wr_addr_v] <= meta_wr_word_v;
                            33:  mem_meta_b33[meta_wr_addr_v] <= meta_wr_word_v;
                            34:  mem_meta_b34[meta_wr_addr_v] <= meta_wr_word_v;
                            35:  mem_meta_b35[meta_wr_addr_v] <= meta_wr_word_v;
                            36:  mem_meta_b36[meta_wr_addr_v] <= meta_wr_word_v;
                            37:  mem_meta_b37[meta_wr_addr_v] <= meta_wr_word_v;
                            38:  mem_meta_b38[meta_wr_addr_v] <= meta_wr_word_v;
                            39:  mem_meta_b39[meta_wr_addr_v] <= meta_wr_word_v;
                            40:  mem_meta_b40[meta_wr_addr_v] <= meta_wr_word_v;
                            41:  mem_meta_b41[meta_wr_addr_v] <= meta_wr_word_v;
                            42:  mem_meta_b42[meta_wr_addr_v] <= meta_wr_word_v;
                            43:  mem_meta_b43[meta_wr_addr_v] <= meta_wr_word_v;
                            44:  mem_meta_b44[meta_wr_addr_v] <= meta_wr_word_v;
                            45:  mem_meta_b45[meta_wr_addr_v] <= meta_wr_word_v;
                            46:  mem_meta_b46[meta_wr_addr_v] <= meta_wr_word_v;
                            47:  mem_meta_b47[meta_wr_addr_v] <= meta_wr_word_v;
                            48:  mem_meta_b48[meta_wr_addr_v] <= meta_wr_word_v;
                            49:  mem_meta_b49[meta_wr_addr_v] <= meta_wr_word_v;
                            50:  mem_meta_b50[meta_wr_addr_v] <= meta_wr_word_v;
                            51:  mem_meta_b51[meta_wr_addr_v] <= meta_wr_word_v;
                            52:  mem_meta_b52[meta_wr_addr_v] <= meta_wr_word_v;
                            53:  mem_meta_b53[meta_wr_addr_v] <= meta_wr_word_v;
                            54:  mem_meta_b54[meta_wr_addr_v] <= meta_wr_word_v;
                            55:  mem_meta_b55[meta_wr_addr_v] <= meta_wr_word_v;
                            56:  mem_meta_b56[meta_wr_addr_v] <= meta_wr_word_v;
                            57:  mem_meta_b57[meta_wr_addr_v] <= meta_wr_word_v;
                            58:  mem_meta_b58[meta_wr_addr_v] <= meta_wr_word_v;
                            59:  mem_meta_b59[meta_wr_addr_v] <= meta_wr_word_v;
                            60:  mem_meta_b60[meta_wr_addr_v] <= meta_wr_word_v;
                            61:  mem_meta_b61[meta_wr_addr_v] <= meta_wr_word_v;
                            62:  mem_meta_b62[meta_wr_addr_v] <= meta_wr_word_v;
                            63:  mem_meta_b63[meta_wr_addr_v] <= meta_wr_word_v;
                            default: begin end
                        endcase
                    end
                end
                // Event-driven ready-token generation.  The old implementation
                // registered touched words here and scanned mem_fill/mem_tag in the
                // next cycle.  That scan was the dominant synthesis/elaboration cone.
                // The write paths above now only insert a token when the just-updated
                // word is complete, so drive the public one-cycle ready pulses directly.
                m1_touch_v_q <= '0;
                m2_touch_v_q <= '0;
                for (i_tok = 0; i_tok < M1_TOUCH_SLOTS; i_tok++) begin
                    m1_sm_ready_valid[i_tok]    <= nxt_m1_touch_v[i_tok];
                    m1_sm_ready_bank[i_tok]     <= nxt_m1_touch_bank[i_tok];
                    m1_sm_ready_row_g[i_tok]    <= nxt_m1_touch_row[i_tok];
                    m1_sm_ready_colgrp_g[i_tok] <= nxt_m1_touch_colgrp[i_tok];
                end
                for (i_tok = 0; i_tok < PF; i_tok++) begin
                    m2_sm_ready_valid[i_tok]      <= nxt_m2_touch_v[i_tok];
                    m2_sm_ready_bank[i_tok]       <= nxt_m2_touch_bank[i_tok];
                    m2_sm_ready_row_g[i_tok]      <= nxt_m2_touch_row[i_tok];
                    m2_sm_ready_colbase_g[i_tok]  <= nxt_m2_touch_colgrp[i_tok];
                end
            end
            // mem_data writes are driven through BRAM wrapper ports from always_comb.
        end
    end


    // ============================================================
    // Registered stream/DMA read stages for synchronous mem_data BRAM
    // ============================================================
    logic                     stream_stage_valid_q;
    logic                     stream_stage_mode2_q;
    logic                     stream_stage_pack_q;
    logic [BANK_W-1:0]        stream_stage_ifm_bank_q;
    logic [ROW_W-1:0]         stream_stage_ifm_row_q;
    logic [COLIDX_W-1:0]      stream_stage_ifm_col_q;
    logic [COLIDX_W-1:0]      stream_stage_ifm_col_g_q;
    logic [PV_MAX-1:0]        stream_stage_keep_q;
    logic [BANK_W-1:0]        stream_stage_clear_bank_q;
    logic [DEPTH_W-1:0]       stream_stage_clear_addr_q;
    logic [TAG_W-1:0]         stream_stage_src_tag_q;
    logic [DATA_BANK_W-1:0]   stream_stage_src_bank_q [0:PV_MAX-1];
    logic [DEPTH_W-1:0]       stream_stage_src_addr_q [0:PV_MAX-1];
    logic [7:0]               stream_stage_src_lane_q [0:PV_MAX-1];

    logic                     stream_issue_v;
    logic                     stream_issue_mode2_v;
    logic                     stream_issue_pack_v;
    logic [BANK_W-1:0]        stream_issue_ifm_bank_v;
    logic [ROW_W-1:0]         stream_issue_ifm_row_v;
    logic [COLIDX_W-1:0]      stream_issue_ifm_col_v;
    logic [COLIDX_W-1:0]      stream_issue_ifm_col_g_v;
    logic [PV_MAX-1:0]        stream_issue_keep_v;
    logic [BANK_W-1:0]        stream_issue_clear_bank_v;
    logic [DEPTH_W-1:0]       stream_issue_clear_addr_v;
    logic [TAG_W-1:0]         stream_issue_src_tag_v;
    logic [DATA_BANK_W-1:0]   stream_issue_src_bank_v [0:PV_MAX-1];
    logic [DEPTH_W-1:0]       stream_issue_src_addr_v [0:PV_MAX-1];
    logic [7:0]               stream_issue_src_lane_v [0:PV_MAX-1];

    logic                     dma_issue_v;
    logic                     dma_issue_pack_v;
    logic [PV_MAX-1:0]        dma_issue_keep_v;
    logic [DATA_BANK_W-1:0]   dma_issue_src_bank_v [0:PV_MAX-1];
    logic [DEPTH_W-1:0]       dma_issue_src_addr_v [0:PV_MAX-1];
    logic [7:0]               dma_issue_src_lane_v [0:PV_MAX-1];

    logic                     dma_stage_pack_q;
    logic [PV_MAX-1:0]        dma_stage_keep_q;
    logic [DATA_BANK_W-1:0]   dma_stage_src_bank_q [0:PV_MAX-1];
    logic [DEPTH_W-1:0]       dma_stage_src_addr_q [0:PV_MAX-1];
    logic [7:0]               dma_stage_src_lane_q [0:PV_MAX-1];

    function automatic logic [WORD_W-1:0] data_rdata_by_bank(input int bank);
        begin
            if ((bank >= 0) && (bank < DATA_BANKS_IMPL))
                data_rdata_by_bank = data_bank_rdata_q[bank];
            else
                data_rdata_by_bank = '0;
        end
    endfunction

// ============================================================
    // IFM stream combinational path
    // ============================================================

    always_comb begin
        integer i;
        integer m1_blk_span_v;
        integer m1_ch_base_v;
        integer prev_m1_blk_span_v;
        integer prev_m1_ch_base_v;
        logic [15:0] prev_phys_grp_v;
        logic [DEPTH_W-1:0] prev_phys_addr_v;
        logic [PV_MAX-1:0] prev_expected_keep_v;
        logic [BANK_W-1:0] prev_stream_bank_v;
        logic [PV_MAX-1:0] m2_chan_keep_v;
        logic              m2_pack_ready_v;
        logic              m2_pack_have_lane_v;
        integer m2_lane_v;
        integer m2_ch_g_v;
        integer m2_src_grp_v;
        integer m2_src_lane_v;
        integer m2_src_addr_v;

        integer lin;
        integer words_per_ch;
        integer ch;
        integer rem;
        integer row;
        integer grp;
        integer addr;
        integer lane_i;
        integer col_abs;
        integer col_l;
        integer col_g;
        integer fgrp;
        integer flane;
        integer m2_bank;
        integer m2_addr;
        logic [PV_MAX-1:0] keep_v;
        logic [TAG_W-1:0]  meta_tag_v;
        logic [PV_MAX-1:0] meta_fill_v;
        logic [TAG_W-1:0]  prev_meta_tag_v;
        logic [PV_MAX-1:0] prev_meta_fill_v;
        logic [TAG_W-1:0]  lane_meta_tag_v;
        logic [PV_MAX-1:0] lane_meta_fill_v;
        logic [TAG_W-1:0]  dma_meta_tag_v;
        logic [PV_MAX-1:0] dma_meta_fill_v;

        logic [WORD_W-1:0] out_word_v;
        logic [WORD_W-1:0] dma_word_v;
        logic [PV_MAX-1:0] dma_keep_v;
        logic [WORD_W-1:0] bank_word_tmp_v;

        // Defaults for read requests
        data_rd_en_v = '0;
        for (i = 0; i < DATA_BANKS_IMPL; i++) begin
            data_rd_addr_v[i] = '0;
        end

        stream_issue_v          = 1'b0;
        stream_issue_mode2_v    = 1'b0;
        stream_issue_pack_v     = 1'b0;
        stream_issue_ifm_bank_v = '0;
        stream_issue_ifm_row_v  = '0;
        stream_issue_ifm_col_v  = '0;
        stream_issue_ifm_col_g_v = '0;
        stream_issue_keep_v     = '0;
        stream_issue_clear_bank_v = '0;
        stream_issue_clear_addr_v = '0;
        stream_issue_src_tag_v  = layer_tag_q;
        for (i = 0; i < PV_MAX; i++) begin
            stream_issue_src_bank_v[i] = '0;
            stream_issue_src_addr_v[i] = '0;
            stream_issue_src_lane_v[i] = i[7:0];
        end

        dma_issue_v      = 1'b0;
        dma_issue_pack_v = 1'b0;
        dma_issue_keep_v = '0;
        for (i = 0; i < PV_MAX; i++) begin
            dma_issue_src_bank_v[i] = '0;
            dma_issue_src_addr_v[i] = '0;
            dma_issue_src_lane_v[i] = i[7:0];
        end

        meta_tag_v       = '0;
        meta_fill_v      = '0;
        prev_meta_tag_v  = '0;
        prev_meta_fill_v = '0;
        lane_meta_tag_v  = '0;
        lane_meta_fill_v = '0;
        dma_meta_tag_v   = '0;
        dma_meta_fill_v  = '0;

        // Default metadata visible to the legacy state-advance/clear code.
        abs_row_v       = '0;
        abs_col_base_v  = '0;
        phys_grp_v      = '0;
        phys_addr_v     = stream_stage_clear_addr_q;
        expected_keep_v = stream_stage_keep_q;
        word_ready_v    = stream_stage_valid_q;
        stream_word_v   = '0;
        stream_bank_v   = stream_stage_clear_bank_q;
        stream_src_tag_v = stream_stage_src_tag_q;
        m1_to_m2_runtime_exact_v = 1'b0;

        // Registered stream output.  Data comes from the synchronous bank-read
        // results, not from a direct mem_data combinational function.
        ifm_stream_busy    = strm_active_q || stream_stage_valid_q;
        ifm_ofm_wr_en      = stream_stage_valid_q;
        ifm_ofm_wr_bank    = stream_stage_ifm_bank_q;
        ifm_ofm_wr_row_idx = stream_stage_ifm_row_q;
        ifm_ofm_wr_col_idx = stream_stage_ifm_col_q;
        ifm_ofm_wr_col_g   = stream_stage_ifm_col_g_q;
        ifm_ofm_wr_mode2   = stream_stage_mode2_q;
        ifm_ofm_wr_keep    = stream_stage_keep_q;

        out_word_v = '0;
        if (stream_stage_pack_q) begin
            for (i = 0; i < PV_MAX; i++) begin
                if (stream_stage_keep_q[i]) begin
                    bank_word_tmp_v = data_rdata_by_bank(stream_stage_src_bank_q[i]);
                    out_word_v[i*DATA_W +: DATA_W] = bank_word_tmp_v[stream_stage_src_lane_q[i]*DATA_W +: DATA_W];
                end
            end
        end
        else begin
            out_word_v = data_rdata_by_bank(stream_stage_src_bank_q[0]);
        end
        ifm_ofm_wr_data = out_word_v;
        stream_word_v   = out_word_v;

        m1_blk_span_v = (pf_next_q == 0) ? 1 : pf_next_q;
        m1_ch_base_v  = strm_m1_ch_blk_q * m1_blk_span_v;

        // Issue a new stream BRAM read only when there is no pending output word.
        if (strm_active_q && !stream_stage_valid_q && !error_q) begin
            case (strm_mode_q)
                STRM_M1_DIRECT: begin
                    abs_row_v       = strm_row_base_q + strm_row_q;
                    abs_col_base_v  = strm_col_base_q;
                    phys_grp_v      = (pv_next_q == 0) ? '0 : (strm_col_base_q / pv_next_q);
                    phys_addr_v     = ofm_phys_addr(abs_row_v, phys_grp_v);
                    expected_keep_v = calc_keep_mask((pv_next_q == 0) ? 16'd1 : pv_next_q, abs_col_base_v, w_out_q);
                    stream_bank_v   = m1_ch_base_v + strm_ch_q;

                    meta_tag_v  = ofm_mem_tag_read(stream_bank_v, phys_addr_v);
                    meta_fill_v = ofm_mem_fill_read(stream_bank_v, phys_addr_v);
                    if (((m1_ch_base_v + strm_ch_q) < f_out_q) &&
                        (abs_row_v < h_out_q) &&
                        (phys_addr_v < DEPTH) &&
                        (meta_tag_v == layer_tag_q) &&
                        ((meta_fill_v & expected_keep_v) == expected_keep_v)) begin
                        stream_issue_v           = 1'b1;
                        stream_issue_mode2_v     = 1'b0;
                        stream_issue_pack_v      = 1'b0;
                        stream_issue_ifm_bank_v  = stream_bank_v;
                        stream_issue_ifm_row_v   = strm_m1_row_slot_q;
                        stream_issue_ifm_col_v   = phys_grp_v[COLIDX_W-1:0];
                        stream_issue_ifm_col_g_v = abs_col_base_v[COLIDX_W-1:0];
                        stream_issue_keep_v      = expected_keep_v;
                        stream_issue_clear_bank_v = stream_bank_v;
                        stream_issue_clear_addr_v = phys_addr_v;
                        stream_issue_src_tag_v   = layer_tag_q;
                        stream_issue_src_bank_v[0] = DATA_BANK_W'(stream_bank_v);
                        stream_issue_src_addr_v[0] = phys_addr_v;
                        stream_issue_src_lane_v[0] = 8'd0;
                    end
                    else begin
                        prev_m1_blk_span_v   = (prev_pf_next_q == 0) ? 1 : prev_pf_next_q;
                        prev_m1_ch_base_v    = strm_m1_ch_blk_q * prev_m1_blk_span_v;
                        prev_phys_grp_v      = (prev_pv_next_q == 0) ? '0 : (strm_col_base_q / prev_pv_next_q);
                        prev_phys_addr_v     = ofm_phys_addr(abs_row_v, prev_phys_grp_v);
                        prev_expected_keep_v = calc_keep_mask((prev_pv_next_q == 0) ? 16'd1 : prev_pv_next_q,
                                                              abs_col_base_v,
                                                              prev_w_out_q);
                        prev_stream_bank_v   = prev_m1_ch_base_v + strm_ch_q;

                        prev_meta_tag_v  = ofm_mem_tag_read(prev_stream_bank_v, prev_phys_addr_v);
                        prev_meta_fill_v = ofm_mem_fill_read(prev_stream_bank_v, prev_phys_addr_v);
                        if (((prev_m1_ch_base_v + strm_ch_q) < prev_f_out_q) &&
                            (abs_row_v < prev_h_out_q) &&
                            (prev_phys_addr_v < DEPTH) &&
                            (prev_meta_tag_v == prev_layer_tag_q) &&
                            ((prev_meta_fill_v & prev_expected_keep_v) == prev_expected_keep_v)) begin
                            stream_issue_v           = 1'b1;
                            stream_issue_mode2_v     = 1'b0;
                            stream_issue_pack_v      = 1'b0;
                            stream_issue_ifm_bank_v  = prev_stream_bank_v;
                            stream_issue_ifm_row_v   = strm_m1_row_slot_q;
                            stream_issue_ifm_col_v   = prev_phys_grp_v[COLIDX_W-1:0];
                            stream_issue_ifm_col_g_v = abs_col_base_v[COLIDX_W-1:0];
                            stream_issue_keep_v      = prev_expected_keep_v;
                            stream_issue_clear_bank_v = prev_stream_bank_v;
                            stream_issue_clear_addr_v = prev_phys_addr_v;
                            stream_issue_src_tag_v   = prev_layer_tag_q;
                            stream_issue_src_bank_v[0] = DATA_BANK_W'(prev_stream_bank_v);
                            stream_issue_src_addr_v[0] = prev_phys_addr_v;
                            stream_issue_src_lane_v[0] = 8'd0;
                            stream_bank_v            = prev_stream_bank_v;
                            phys_addr_v              = prev_phys_addr_v;
                            expected_keep_v          = prev_expected_keep_v;
                            stream_src_tag_v         = prev_layer_tag_q;
                        end
                    end
                end

                STRM_M2_DIRECT: begin
                    abs_row_v      = strm_row_base_q + strm_row_q;
                    abs_col_base_v = strm_col_base_q;
                    phys_grp_v     = (PC == 0) ? '0 : (abs_col_base_v / PC);
                    stream_bank_v  = (PC == 0) ? '0 : ((strm_m2_cgrp_q * PC) + (abs_col_base_v % PC));
                    phys_addr_v    = ofm_phys_addr(abs_row_v, phys_grp_v);

                    m2_chan_keep_v      = '0;
                    m2_pack_have_lane_v = 1'b0;
                    for (m2_lane_v = 0; m2_lane_v < PV_MAX; m2_lane_v++) begin
                        if (m2_lane_v < PC) begin
                            m2_ch_g_v = (strm_m2_cgrp_q * PC) + m2_lane_v;
                            if ((m2_ch_g_v < f_out_q) &&
                                (strm_row_q < strm_num_rows_q) &&
                                (abs_row_v < h_out_q) &&
                                (abs_col_base_v < w_out_q) &&
                                (stream_bank_v < C_MAX) &&
                                (phys_addr_v < DEPTH)) begin
                                m2_chan_keep_v[m2_lane_v] = 1'b1;
                                m2_pack_have_lane_v = 1'b1;
                            end
                        end
                    end

                    meta_tag_v  = ofm_mem_tag_read(stream_bank_v, phys_addr_v);
                    meta_fill_v = ofm_mem_fill_read(stream_bank_v, phys_addr_v);
                    if (m2_pack_have_lane_v &&
                        (stream_bank_v < C_MAX) &&
                        (phys_addr_v < DEPTH) &&
                        (meta_tag_v == layer_tag_q) &&
                        ((meta_fill_v & m2_chan_keep_v) == m2_chan_keep_v)) begin
                        stream_issue_v           = 1'b1;
                        stream_issue_mode2_v     = 1'b1;
                        stream_issue_pack_v      = 1'b0;
                        stream_issue_ifm_bank_v  = (PC == 0) ? '0 : (abs_col_base_v % PC);
                        stream_issue_ifm_row_v   = abs_row_v[ROW_W-1:0];
                        stream_issue_ifm_col_v   = strm_m2_cgrp_q[COLIDX_W-1:0];
                        stream_issue_ifm_col_g_v = abs_col_base_v[COLIDX_W-1:0];
                        stream_issue_keep_v      = m2_chan_keep_v;
                        stream_issue_clear_bank_v = stream_bank_v;
                        stream_issue_clear_addr_v = phys_addr_v;
                        stream_issue_src_tag_v   = layer_tag_q;
                        stream_issue_src_bank_v[0] = DATA_BANK_W'(stream_bank_v);
                        stream_issue_src_addr_v[0] = phys_addr_v;
                        stream_issue_src_lane_v[0] = 8'd0;
                    end
                    else begin
                        prev_phys_grp_v      = (PC == 0) ? '0 : (abs_col_base_v / PC);
                        prev_stream_bank_v   = (PC == 0) ? '0 : ((strm_m2_cgrp_q * PC) + (abs_col_base_v % PC));
                        prev_phys_addr_v     = ofm_phys_addr(abs_row_v, prev_phys_grp_v);
                        prev_expected_keep_v = '0;
                        m2_pack_have_lane_v  = 1'b0;

                        for (m2_lane_v = 0; m2_lane_v < PV_MAX; m2_lane_v++) begin
                            if (m2_lane_v < PC) begin
                                m2_ch_g_v = (strm_m2_cgrp_q * PC) + m2_lane_v;
                                if ((m2_ch_g_v < prev_f_out_q) &&
                                    (strm_row_q < strm_num_rows_q) &&
                                    (abs_row_v < prev_h_out_q) &&
                                    (abs_col_base_v < prev_w_out_q) &&
                                    (prev_stream_bank_v < C_MAX) &&
                                    (prev_phys_addr_v < DEPTH)) begin
                                    prev_expected_keep_v[m2_lane_v] = 1'b1;
                                    m2_pack_have_lane_v = 1'b1;
                                end
                            end
                        end

                        prev_meta_tag_v  = ofm_mem_tag_read(prev_stream_bank_v, prev_phys_addr_v);
                        prev_meta_fill_v = ofm_mem_fill_read(prev_stream_bank_v, prev_phys_addr_v);
                        if (m2_pack_have_lane_v &&
                            (prev_stream_bank_v < C_MAX) &&
                            (prev_phys_addr_v < DEPTH) &&
                            (prev_meta_tag_v == prev_layer_tag_q) &&
                            ((prev_meta_fill_v & prev_expected_keep_v) == prev_expected_keep_v)) begin
                            stream_issue_v           = 1'b1;
                            stream_issue_mode2_v     = 1'b1;
                            stream_issue_pack_v      = 1'b0;
                            stream_issue_ifm_bank_v  = (PC == 0) ? '0 : (abs_col_base_v % PC);
                            stream_issue_ifm_row_v   = abs_row_v[ROW_W-1:0];
                            stream_issue_ifm_col_v   = strm_m2_cgrp_q[COLIDX_W-1:0];
                            stream_issue_ifm_col_g_v = abs_col_base_v[COLIDX_W-1:0];
                            stream_issue_keep_v      = prev_expected_keep_v;
                            stream_issue_clear_bank_v = prev_stream_bank_v;
                            stream_issue_clear_addr_v = prev_phys_addr_v;
                            stream_issue_src_tag_v   = prev_layer_tag_q;
                            stream_issue_src_bank_v[0] = DATA_BANK_W'(prev_stream_bank_v);
                            stream_issue_src_addr_v[0] = prev_phys_addr_v;
                            stream_issue_src_lane_v[0] = 8'd0;
                            stream_bank_v            = prev_stream_bank_v;
                            phys_addr_v              = prev_phys_addr_v;
                            expected_keep_v          = prev_expected_keep_v;
                            stream_src_tag_v         = prev_layer_tag_q;
                        end
                    end
                end

                STRM_M1_TO_M2: begin
                    abs_row_v      = strm_row_base_q + strm_row_q;
                    m1_to_m2_runtime_exact_v = (strm_num_rows_q == 1);
                    if (m1_to_m2_runtime_exact_v)
                        abs_col_base_v = strm_col_base_q;
                    else
                        abs_col_base_v = strm_col_base_q + strm_colgrp_q;

                    stream_bank_v  = (PC == 0) ? '0 : (abs_col_base_v % PC);

                    m2_chan_keep_v      = '0;
                    m2_pack_ready_v     = 1'b1;
                    m2_pack_have_lane_v = 1'b0;

                    for (m2_lane_v = 0; m2_lane_v < PV_MAX; m2_lane_v++) begin
                        if (m2_lane_v < PC) begin
                            m2_ch_g_v     = (strm_ch_q * PC) + m2_lane_v;
                            m2_src_grp_v  = (src_pack_q == 0) ? 0 : (abs_col_base_v / src_pack_q);
                            m2_src_lane_v = (src_pack_q == 0) ? 0 : (abs_col_base_v % src_pack_q);
                            m2_src_addr_v = ofm_phys_addr(abs_row_v, m2_src_grp_v);

                            if ((m2_ch_g_v < f_out_q) &&
                                (strm_row_q < strm_num_rows_q) &&
                                (abs_row_v < h_out_q) &&
                                (abs_col_base_v < w_out_q)) begin
                                m2_chan_keep_v[m2_lane_v] = 1'b1;
                                m2_pack_have_lane_v = 1'b1;

                                lane_meta_tag_v  = ofm_mem_tag_read(m2_ch_g_v, m2_src_addr_v);
                                lane_meta_fill_v = ofm_mem_fill_read(m2_ch_g_v, m2_src_addr_v);
                                if (!((m2_src_grp_v < stored_groups_q) &&
                                      (m2_src_addr_v < DEPTH) &&
                                      (m2_src_lane_v < PV_MAX) &&
                                      (lane_meta_tag_v == layer_tag_q) &&
                                      lane_meta_fill_v[m2_src_lane_v])) begin
                                    m2_pack_ready_v = 1'b0;
                                end
                                else begin
                                    stream_issue_src_bank_v[m2_lane_v] = DATA_BANK_W'(m2_ch_g_v);
                                    stream_issue_src_addr_v[m2_lane_v] = m2_src_addr_v[DEPTH_W-1:0];
                                    stream_issue_src_lane_v[m2_lane_v] = m2_src_lane_v[7:0];
                                end
                            end
                        end
                    end

                    if (layer_write_done_q && m2_pack_have_lane_v && m2_pack_ready_v) begin
                        stream_issue_v           = 1'b1;
                        stream_issue_mode2_v     = 1'b1;
                        stream_issue_pack_v      = 1'b1;
                        stream_issue_ifm_bank_v  = stream_bank_v;
                        stream_issue_ifm_row_v   = abs_row_v[ROW_W-1:0];
                        stream_issue_ifm_col_v   = COLIDX_W'(m1_to_m2_runtime_exact_v ? strm_m2_cgrp_q : strm_ch_q);
                        stream_issue_ifm_col_g_v = abs_col_base_v[COLIDX_W-1:0];
                        stream_issue_keep_v      = m2_chan_keep_v;
                        stream_issue_clear_bank_v = stream_bank_v;
                        stream_issue_clear_addr_v = '0;
                        stream_issue_src_tag_v   = layer_tag_q;
                    end
                    else begin
                        m2_chan_keep_v      = '0;
                        m2_pack_ready_v     = 1'b1;
                        m2_pack_have_lane_v = 1'b0;

                        for (m2_lane_v = 0; m2_lane_v < PV_MAX; m2_lane_v++) begin
                            if (m2_lane_v < PC) begin
                                m2_ch_g_v     = ((m1_to_m2_runtime_exact_v ? strm_m2_cgrp_q : strm_ch_q) * PC) + m2_lane_v;
                                m2_src_grp_v  = (prev_store_pack_q == 0) ? 0 : (abs_col_base_v / prev_store_pack_q);
                                m2_src_lane_v = (prev_store_pack_q == 0) ? 0 : (abs_col_base_v % prev_store_pack_q);
                                m2_src_addr_v = ofm_phys_addr(abs_row_v, m2_src_grp_v);

                                if ((m2_ch_g_v < prev_f_out_q) &&
                                    (strm_row_q < strm_num_rows_q) &&
                                    (abs_row_v < prev_h_out_q) &&
                                    (abs_col_base_v < prev_w_out_q)) begin
                                    m2_chan_keep_v[m2_lane_v] = 1'b1;
                                    m2_pack_have_lane_v = 1'b1;

                                    lane_meta_tag_v  = ofm_mem_tag_read(m2_ch_g_v, m2_src_addr_v);
                                    lane_meta_fill_v = ofm_mem_fill_read(m2_ch_g_v, m2_src_addr_v);
                                    if (!((m2_src_grp_v < prev_stored_groups_q) &&
                                          (m2_src_addr_v < DEPTH) &&
                                          (m2_src_lane_v < PV_MAX) &&
                                          (lane_meta_tag_v == prev_layer_tag_q) &&
                                          lane_meta_fill_v[m2_src_lane_v])) begin
                                        m2_pack_ready_v = 1'b0;
                                    end
                                    else begin
                                        stream_issue_src_bank_v[m2_lane_v] = DATA_BANK_W'(m2_ch_g_v);
                                        stream_issue_src_addr_v[m2_lane_v] = m2_src_addr_v[DEPTH_W-1:0];
                                        stream_issue_src_lane_v[m2_lane_v] = m2_src_lane_v[7:0];
                                    end
                                end
                            end
                        end

                        if (m2_pack_have_lane_v && m2_pack_ready_v) begin
                            stream_issue_v           = 1'b1;
                            stream_issue_mode2_v     = 1'b1;
                            stream_issue_pack_v      = 1'b1;
                            stream_issue_ifm_bank_v  = stream_bank_v;
                            stream_issue_ifm_row_v   = abs_row_v[ROW_W-1:0];
                            stream_issue_ifm_col_v   = COLIDX_W'(m1_to_m2_runtime_exact_v ? strm_m2_cgrp_q : strm_ch_q);
                            stream_issue_ifm_col_g_v = abs_col_base_v[COLIDX_W-1:0];
                            stream_issue_keep_v      = m2_chan_keep_v;
                            stream_issue_clear_bank_v = stream_bank_v;
                            stream_issue_clear_addr_v = '0;
                            stream_issue_src_tag_v   = prev_layer_tag_q;
                        end
                    end
                end

                default: begin
                end
            endcase
        end

        // Register BRAM read requests for the stream word being issued.
        if (stream_issue_v) begin
            if (stream_issue_pack_v) begin
                for (i = 0; i < PV_MAX; i++) begin
                    if (stream_issue_keep_v[i] && (stream_issue_src_bank_v[i] < DATA_BANKS_IMPL)) begin
                        data_rd_en_v[stream_issue_src_bank_v[i]] = 1'b1;
                        data_rd_addr_v[stream_issue_src_bank_v[i]] = stream_issue_src_addr_v[i];
                    end
                end
            end
            else begin
                if (stream_issue_src_bank_v[0] < DATA_BANKS_IMPL) begin
                    data_rd_en_v[stream_issue_src_bank_v[0]] = 1'b1;
                    data_rd_addr_v[stream_issue_src_bank_v[0]] = stream_issue_src_addr_v[0];
                end
            end
        end

        // DMA readback request generation.  It shares the bank read fabric with
        // stream.  Stream has priority because it can be holding IFM refill.
        if (ofm_dma_rd_en && !stream_issue_v && !stream_stage_valid_q) begin
            lin = ofm_dma_rd_addr;
            words_per_ch = h_out_q * stored_groups_q;
            if ((lin < layer_num_words) && (words_per_ch != 0)) begin
                ch  = lin / words_per_ch;
                rem = lin % words_per_ch;
                row = rem / stored_groups_q;
                grp = rem % stored_groups_q;
                keep_v = calc_keep_mask(store_pack_q, grp * store_pack_q, w_out_q);

                if (!src_mode_q) begin
                    addr = ofm_phys_addr(row, grp);
                    dma_meta_tag_v  = ofm_mem_tag_read(ch, addr);
                    dma_meta_fill_v = ofm_mem_fill_read(ch, addr);
                    if ((ch < f_out_q) && (addr < DEPTH) && (dma_meta_tag_v == layer_tag_q)) begin
                        dma_issue_v             = 1'b1;
                        dma_issue_pack_v        = 1'b0;
                        dma_issue_keep_v        = keep_v & dma_meta_fill_v;
                        dma_issue_src_bank_v[0] = DATA_BANK_W'(ch);
                        dma_issue_src_addr_v[0] = addr[DEPTH_W-1:0];
                        dma_issue_src_lane_v[0] = 8'd0;
                        if (ch < DATA_BANKS_IMPL) begin
                            data_rd_en_v[ch] = 1'b1;
                            data_rd_addr_v[ch] = addr[DEPTH_W-1:0];
                        end
                    end
                end
                else begin
                    dma_issue_pack_v = 1'b1;
                    dma_keep_v       = '0;
                    for (lane_i = 0; lane_i < PV_MAX; lane_i++) begin
                        if (keep_v[lane_i]) begin
                            col_abs = (grp * store_pack_q) + lane_i;
                            fgrp    = (PC == 0) ? 0 : (ch / PC);
                            flane   = (PC == 0) ? 0 : (ch % PC);
                            col_l   = (PC == 0) ? 0 : (col_abs % PC);
                            col_g   = (PC == 0) ? 0 : (col_abs / PC);
                            m2_bank = (fgrp * PC) + col_l;
                            m2_addr = ofm_phys_addr(row, col_g);
                            dma_meta_tag_v  = ofm_mem_tag_read(m2_bank, m2_addr);
                            dma_meta_fill_v = ofm_mem_fill_read(m2_bank, m2_addr);
                            if ((ch < f_out_q) && (col_abs < w_out_q) &&
                                (m2_bank < C_MAX) && (m2_addr < DEPTH) &&
                                (flane < PV_MAX) &&
                                (dma_meta_tag_v == layer_tag_q) &&
                                dma_meta_fill_v[flane]) begin
                                dma_keep_v[lane_i]          = 1'b1;
                                dma_issue_src_bank_v[lane_i] = DATA_BANK_W'(m2_bank);
                                dma_issue_src_addr_v[lane_i] = m2_addr[DEPTH_W-1:0];
                                dma_issue_src_lane_v[lane_i] = flane[7:0];
                                if (m2_bank < DATA_BANKS_IMPL) begin
                                    data_rd_en_v[m2_bank] = 1'b1;
                                    data_rd_addr_v[m2_bank] = m2_addr[DEPTH_W-1:0];
                                end
                            end
                        end
                    end
                    if (|dma_keep_v) begin
                        dma_issue_v      = 1'b1;
                        dma_issue_keep_v = dma_keep_v;
                    end
                end
            end
        end

        dma_word_v = '0;
        dma_keep_v = dma_stage_keep_q;
        if (dma_stage_pack_q) begin
            for (i = 0; i < PV_MAX; i++) begin
                if (dma_stage_keep_q[i]) begin
                    bank_word_tmp_v = data_rdata_by_bank(dma_stage_src_bank_q[i]);
                    dma_word_v[i*DATA_W +: DATA_W] = bank_word_tmp_v[dma_stage_src_lane_q[i]*DATA_W +: DATA_W];
                end
            end
        end
        else begin
            dma_word_v = data_rdata_by_bank(dma_stage_src_bank_q[0]);
        end
        dma_data_q = dma_word_v;
        dma_keep_q = dma_keep_v;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        integer i;
        if (!rst_n) begin
            stream_stage_valid_q <= 1'b0;
            stream_stage_mode2_q <= 1'b0;
            stream_stage_pack_q  <= 1'b0;
            stream_stage_ifm_bank_q <= '0;
            stream_stage_ifm_row_q  <= '0;
            stream_stage_ifm_col_q  <= '0;
            stream_stage_ifm_col_g_q <= '0;
            stream_stage_keep_q <= '0;
            stream_stage_clear_bank_q <= '0;
            stream_stage_clear_addr_q <= '0;
            stream_stage_src_tag_q <= '0;
            for (i = 0; i < PV_MAX; i++) begin
                stream_stage_src_bank_q[i] <= '0;
                stream_stage_src_addr_q[i] <= '0;
                stream_stage_src_lane_q[i] <= '0;
            end
        end
        else begin
            if (layer_start || !strm_active_q) begin
                stream_stage_valid_q <= 1'b0;
            end
            else if (stream_stage_valid_q && ifm_ofm_wr_ready) begin
                stream_stage_valid_q <= 1'b0;
            end

            if (!stream_stage_valid_q && stream_issue_v) begin
                stream_stage_valid_q <= 1'b1;
                stream_stage_mode2_q <= stream_issue_mode2_v;
                stream_stage_pack_q  <= stream_issue_pack_v;
                stream_stage_ifm_bank_q <= stream_issue_ifm_bank_v;
                stream_stage_ifm_row_q  <= stream_issue_ifm_row_v;
                stream_stage_ifm_col_q  <= stream_issue_ifm_col_v;
                stream_stage_ifm_col_g_q <= stream_issue_ifm_col_g_v;
                stream_stage_keep_q <= stream_issue_keep_v;
                stream_stage_clear_bank_q <= stream_issue_clear_bank_v;
                stream_stage_clear_addr_q <= stream_issue_clear_addr_v;
                stream_stage_src_tag_q <= stream_issue_src_tag_v;
                for (i = 0; i < PV_MAX; i++) begin
                    stream_stage_src_bank_q[i] <= stream_issue_src_bank_v[i];
                    stream_stage_src_addr_q[i] <= stream_issue_src_addr_v[i];
                    stream_stage_src_lane_q[i] <= stream_issue_src_lane_v[i];
                end
            end
        end
    end

    // ============================================================
    // DMA linear readback, BRAM-latency aware
    // ============================================================
    always_ff @(posedge clk or negedge rst_n) begin
        integer i;
        if (!rst_n) begin
            dma_valid_q <= 1'b0;
            dma_stage_pack_q <= 1'b0;
            dma_stage_keep_q <= '0;
            for (i = 0; i < PV_MAX; i++) begin
                dma_stage_src_bank_q[i] <= '0;
                dma_stage_src_addr_q[i] <= '0;
                dma_stage_src_lane_q[i] <= '0;
            end
        end
        else begin
            dma_valid_q <= dma_issue_v;
            if (dma_issue_v) begin
                dma_stage_pack_q <= dma_issue_pack_v;
                dma_stage_keep_q <= dma_issue_keep_v;
                for (i = 0; i < PV_MAX; i++) begin
                    dma_stage_src_bank_q[i] <= dma_issue_src_bank_v[i];
                    dma_stage_src_addr_q[i] <= dma_issue_src_addr_v[i];
                    dma_stage_src_lane_q[i] <= dma_issue_src_lane_v[i];
                end
            end
        end
    end










endmodule

`undef OFM_DATA_WR_ACCUM
`undef OFM_MEM_WRITE_LANES_ACCUM
`undef OFM_META_CLEAR_ACCUM
