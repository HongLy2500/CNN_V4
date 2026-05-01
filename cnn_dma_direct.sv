`include "cnn_ddr_defs.svh"

module cnn_dma_direct #(
    parameter int DATA_W      = 8,
    parameter int PV_MAX      = 8,
    parameter int PC          = 8,
    parameter int C_MAX       = 64,
    parameter int W_MAX       = 224,
    parameter int H_MAX       = 224,
    parameter int HT          = 8,

    // Fixed physical MAC/weight parallelism of the accelerator.
    // Mode 1: PTOTAL = Pv_mode1 * Pf_mode1(layer)
    // Mode 2: PTOTAL = Pc * Pf_mode2(constant in mode 2)
    parameter int PTOTAL      = 16,

    parameter int DDR_ADDR_W     = `CNN_DDR_ADDR_W,
    parameter int DDR_WORD_W     = PV_MAX * DATA_W,

    // Physical weight-buffer word width shared by DMA and weight_buffer.
    // One DMA write always fills one physical weight-buffer address.
    // This must be the same fixed PTOTAL used by the whole accelerator.
    parameter int WGT_WORD_LANES = PTOTAL,
    parameter int WGT_WORD_W     = WGT_WORD_LANES * DATA_W,

    parameter int WGT_DEPTH      = 4096,

    // OFM DMA addressing is linear over the entire OFM buffer storage,
    // not just one bank depth.
    parameter int OFM_BANK_DEPTH   = 4096,
    parameter int OFM_LINEAR_DEPTH = C_MAX * OFM_BANK_DEPTH
)(
    input  logic clk,
    input  logic rst_n,

    //==================================================
    // Runtime config (must match ifm_buffer cfg)
    //==================================================
    input  logic                        cfg_mode,     // 0: mode1, 1: mode2
    input  logic [$clog2(W_MAX+1)-1:0]  cfg_w_in,
    input  logic [$clog2(H_MAX+1)-1:0]  cfg_h_in,
    input  logic [$clog2(C_MAX+1)-1:0]  cfg_c_in,
    input  logic [$clog2(PV_MAX+1)-1:0] cfg_pv_cur,

    //==================================================
    // IFM command
    // mode1:
    //   DDR layout per command = [channel][row][col_group]
    //   buf_row_base is logical row base in current HT window
    //
    // mode2:
    //   DDR layout per command = [channel][row]
    //   one command loads one horizontal tile of width WT=PC
    //==================================================
    input  logic                        ifm_cmd_start,
    input  logic [DDR_ADDR_W-1:0]       ifm_cmd_ddr_base,
    input  logic [$clog2(H_MAX+1)-1:0]  ifm_cmd_num_rows,
    input  logic [$clog2(H_MAX)-1:0]    ifm_cmd_buf_row_base,

    //==================================================
    // Weight command
    // wgt_cmd_num_words counts PHYSICAL weight-buffer words.
    // Each physical word is PTOTAL lanes wide.
    //
    // Mode 1 DDR stream assumption:
    //   the DDR stream is laid out as consecutive logical Pf-mode1 bundles
    //   in flat order of (f_group, c, ky, kx). DMA packs the next Pv_cur
    //   logical bundles contiguously into one PTOTAL-lane physical word.
    //   weight_buffer + weight_read_ctrl_mode1 later select which Pf chunk
    //   inside that physical word is consumed in each cycle.
    //
    // Mode 2 DDR stream assumption:
    //   one physical word already corresponds to one PTOTAL-lane
    //   (Pf_mode2 x Pc) block.
    //==================================================
    input  logic                        wgt_cmd_start,
    input  logic                        wgt_cmd_buf_sel, // 0: current, 1: next
    input  logic [DDR_ADDR_W-1:0]       wgt_cmd_ddr_base,
    input  logic [$clog2(WGT_DEPTH+1)-1:0] wgt_cmd_num_words,

    //==================================================
    // OFM command
    // ofm_cmd_num_words / ofm_cmd_buf_base are linear over the entire
    // OFM buffer storage space.
    //==================================================
    input  logic                        ofm_cmd_start,
    input  logic [DDR_ADDR_W-1:0]       ofm_cmd_ddr_base,
    input  logic [$clog2(OFM_LINEAR_DEPTH+1)-1:0] ofm_cmd_num_words,
    input  logic [$clog2(OFM_LINEAR_DEPTH)-1:0]   ofm_cmd_buf_base,

    //==================================================
    // Status
    //==================================================
    output logic                        busy,
    output logic                        done,
    output logic                        done_ifm,
    output logic                        done_wgt,
    output logic                        done_ofm,
    output logic                        error,

    //==================================================
    // DDR direct interface
    //==================================================
    output logic                        ddr_rd_req,
    output logic [DDR_ADDR_W-1:0]       ddr_rd_addr,
    input  logic                        ddr_rd_valid,
    input  logic [DDR_WORD_W-1:0]       ddr_rd_data,

    output logic                        ddr_wr_en,
    output logic [DDR_ADDR_W-1:0]       ddr_wr_addr,
    output logic [DDR_WORD_W-1:0]       ddr_wr_data,
    output logic [(DDR_WORD_W/8)-1:0]   ddr_wr_be,

    //==================================================
    // IFM buffer DMA write port
    //==================================================
    output logic                        ifm_dma_wr_en,
    output logic [$clog2(C_MAX)-1:0]    ifm_dma_wr_bank,
    output logic [$clog2(H_MAX)-1:0]    ifm_dma_wr_row_idx,
    output logic [IFM_COL_W-1:0] ifm_dma_wr_col_idx,
    output logic [PV_MAX*DATA_W-1:0]    ifm_dma_wr_data,
    output logic [PV_MAX-1:0]           ifm_dma_wr_keep,
    input  logic                        ifm_dma_wr_ready,

    //==================================================
    // Weight buffer DMA write port
    //==================================================
    output logic                        wgt_dma_wr_en,
    output logic                        wgt_dma_wr_buf_sel,
    output logic [$clog2(WGT_DEPTH)-1:0] wgt_dma_wr_addr,
    output logic [WGT_WORD_W-1:0]       wgt_dma_wr_data,
    output logic [WGT_WORD_LANES-1:0]   wgt_dma_wr_keep,
    input  logic                        wgt_dma_wr_ready,
    output logic                        wgt_dma_load_done,
    output logic                        wgt_dma_load_buf_sel,

    //==================================================
    // OFM buffer DMA read port
    //==================================================
    output logic                        ofm_dma_rd_en,
    output logic [$clog2(OFM_LINEAR_DEPTH)-1:0] ofm_dma_rd_addr,
    input  logic                        ofm_dma_rd_valid,
    input  logic [PV_MAX*DATA_W-1:0]    ofm_dma_rd_data,
    input  logic [PV_MAX-1:0]           ofm_dma_rd_keep
);

    // Maximum runtime col-groups occurs when Pv_cur is smallest.
    // Without a PV_MIN parameter, W_MAX is the safe upper bound.
    localparam int IFM_COLS_MAX = (W_MAX <= 1) ? 1 : W_MAX;
    localparam int IFM_COL_W    = (IFM_COLS_MAX <= 1) ? 1 : $clog2(IFM_COLS_MAX);
    localparam int WGT_SUBWORDS = (WGT_WORD_W + DDR_WORD_W - 1) / DDR_WORD_W;
    localparam int WGT_PACK_CNT_W = (WGT_SUBWORDS <= 1) ? 1 : $clog2(WGT_SUBWORDS + 1);

    localparam logic [DDR_ADDR_W-1:0] DDR_IFM_END = `DDR_IFM_BASE + `DDR_IFM_SIZE - 1;
    localparam logic [DDR_ADDR_W-1:0] DDR_WGT_END = `DDR_WGT_BASE + `DDR_WGT_SIZE - 1;
    localparam logic [DDR_ADDR_W-1:0] DDR_OFM_END = `DDR_OFM_BASE + `DDR_OFM_SIZE - 1;

    typedef enum logic [3:0] {
        ST_IDLE,
        ST_IFM_REQ,
        ST_IFM_WAIT,
        ST_WGT_REQ,
        ST_WGT_WAIT,
        ST_OFM_REQ,
        ST_OFM_WAIT,
        ST_DONE
    } state_t;

    typedef enum logic [1:0] {
        CMD_NONE,
        CMD_IFM,
        CMD_WGT,
        CMD_OFM
    } cmd_t;

    state_t state, state_n;
    cmd_t   cmd_done_q, cmd_done_n;

    logic [DDR_ADDR_W-1:0] ddr_addr_base_q, ddr_addr_base_n;
    logic [DDR_ADDR_W-1:0] ddr_linear_idx_q, ddr_linear_idx_n;

    logic [$clog2(C_MAX+1)-1:0] ifm_bank_q, ifm_bank_n;
    logic [$clog2(H_MAX+1)-1:0] ifm_row_iter_q, ifm_row_iter_n;
    logic [IFM_COL_W-1:0]       ifm_col_iter_q, ifm_col_iter_n;
    logic [$clog2(H_MAX+1)-1:0] ifm_num_rows_q, ifm_num_rows_n;
    logic [$clog2(H_MAX)-1:0]   ifm_buf_row_base_q, ifm_buf_row_base_n;

    logic                       wgt_buf_sel_q, wgt_buf_sel_n;
    logic [$clog2(WGT_DEPTH+1)-1:0] wgt_num_words_q, wgt_num_words_n;
    logic [$clog2(WGT_DEPTH+1)-1:0] wgt_word_idx_q,  wgt_word_idx_n;
    logic [WGT_WORD_W-1:0]          wgt_pack_q,      wgt_pack_n;
    logic [WGT_PACK_CNT_W-1:0]      wgt_pack_cnt_q,  wgt_pack_cnt_n;
    logic                           wgt_word_ready_q, wgt_word_ready_n;

    logic [$clog2(OFM_LINEAR_DEPTH+1)-1:0] ofm_num_words_q, ofm_num_words_n;
    logic [$clog2(OFM_LINEAR_DEPTH+1)-1:0] ofm_word_idx_q,  ofm_word_idx_n;
    logic [$clog2(OFM_LINEAR_DEPTH)-1:0]   ofm_buf_base_q,  ofm_buf_base_n;

    logic [DDR_WORD_W-1:0] rd_data_hold_q, rd_data_hold_n;
    logic                  rd_hold_valid_q, rd_hold_valid_n;

    logic [PV_MAX-1:0] keep_mask_v;
    logic [31:0] mode1_groups_v;
    logic [31:0] full_words_v;
    logic [31:0] rem_pixels_v;
    logic [31:0] curr_valid_lanes_v;

    logic [31:0] ifm_total_words_in;
    logic [31:0] wgt_total_words_in;
    logic [31:0] ofm_total_words_in;

    // Helper values for mode-1 weight packing interpretation.
    // DMA itself still packs a flat stream of weight elements into PTOTAL-lane
    // physical words; the mode-1 logical slicing is handled after the buffer.
    logic [31:0] wgt_m1_pf_cur_v;
    logic [31:0] wgt_m1_bundles_per_word_v;

    logic [31:0] ifm_last_addr_in;
    logic [31:0] wgt_last_addr_in;
    logic [31:0] ofm_last_addr_in;

    // IFM mode-1 commands may load only a subset of absolute rows.
    // DDR IFM layout is [channel][row][col_group], so channel stride must use
    // the full H_in, not the command's num_rows.
    logic [31:0]          ifm_ddr_offset_v;
    logic [31:0]          ifm_last_offset_v;
    logic [DDR_ADDR_W-1:0] ifm_ddr_addr_v;

    logic        ifm_cmd_range_ok;
    logic        wgt_cmd_range_ok;
    logic        ofm_cmd_range_ok;
    logic        ifm_cmd_shape_ok;

    
    integer sub;
    integer base_bit;

    //==================================================
    // Derived values
    //==================================================
    always_comb begin
        if (cfg_pv_cur == 0)
            mode1_groups_v = 0;
        else
            mode1_groups_v = (cfg_w_in + cfg_pv_cur - 1) / cfg_pv_cur;
    end

    always_comb begin
        ifm_ddr_offset_v = 32'd0;

        if (!cfg_mode) begin
            ifm_ddr_offset_v = (ifm_bank_q * cfg_h_in * mode1_groups_v) +
                               (ifm_row_iter_q * mode1_groups_v) +
                               ifm_col_iter_q;
        end
        else begin
            // Keep the existing linear behavior for mode 2.
            ifm_ddr_offset_v = ddr_linear_idx_q;
        end

        ifm_ddr_addr_v = ddr_addr_base_q + ifm_ddr_offset_v[DDR_ADDR_W-1:0];
    end

    always_comb begin
        if (cfg_pv_cur != 0)
            wgt_m1_pf_cur_v = PTOTAL / cfg_pv_cur;
        else
            wgt_m1_pf_cur_v = 0;

        // In mode 1, one PTOTAL-lane physical word packs Pv_cur logical bundles.
        wgt_m1_bundles_per_word_v = cfg_pv_cur;
    end

    always_comb begin
        keep_mask_v = '0;

        if (!cfg_mode) begin
            full_words_v = (cfg_pv_cur == 0) ? 0 : (cfg_w_in / cfg_pv_cur);
            rem_pixels_v = (cfg_pv_cur == 0) ? 0 : (cfg_w_in % cfg_pv_cur);

            if (ifm_col_iter_q < full_words_v)
                curr_valid_lanes_v = cfg_pv_cur;
            else
                curr_valid_lanes_v = (rem_pixels_v == 0) ? cfg_pv_cur : rem_pixels_v;
        end
        else begin
            curr_valid_lanes_v = (cfg_w_in < PC) ? cfg_w_in : PC;
        end

        for (int k = 0; k < PV_MAX; k++) begin
            keep_mask_v[k] = (k < curr_valid_lanes_v);
        end
    end

    //==================================================
    // Command legality checks
    //==================================================
    always_comb begin
        if (!cfg_mode)
            ifm_total_words_in = cfg_c_in * ifm_cmd_num_rows * mode1_groups_v;
        else
            ifm_total_words_in = cfg_c_in * ifm_cmd_num_rows;

        wgt_total_words_in = wgt_cmd_num_words * WGT_SUBWORDS;
        ofm_total_words_in = ofm_cmd_num_words;

        if (!cfg_mode) begin
            ifm_last_offset_v = ((cfg_c_in == 0) ? 0 : ((cfg_c_in - 1) * cfg_h_in * mode1_groups_v)) +
                                ((ifm_cmd_num_rows == 0) ? 0 : ((ifm_cmd_num_rows - 1) * mode1_groups_v)) +
                                ((mode1_groups_v == 0) ? 0 : (mode1_groups_v - 1));
            ifm_last_addr_in  = ifm_cmd_ddr_base + ifm_last_offset_v;
        end
        else begin
            ifm_last_offset_v = (ifm_total_words_in == 0) ? 0 : (ifm_total_words_in - 1);
            ifm_last_addr_in  = ifm_cmd_ddr_base + ifm_last_offset_v;
        end

        wgt_last_addr_in = wgt_cmd_ddr_base +
                           ((wgt_total_words_in == 0) ? 0 : (wgt_total_words_in - 1));
        ofm_last_addr_in = ofm_cmd_ddr_base +
                           ((ofm_total_words_in == 0) ? 0 : (ofm_total_words_in - 1));

        if (!cfg_mode) begin
            ifm_cmd_shape_ok =
                (cfg_w_in != 0) &&
                (cfg_w_in <= W_MAX) &&
                (cfg_pv_cur != 0) &&
                (cfg_pv_cur <= PV_MAX) &&
                (mode1_groups_v != 0) &&
                (mode1_groups_v <= IFM_COLS_MAX) &&
                (ifm_cmd_num_rows != 0) &&
                (cfg_c_in != 0) &&
                (ifm_cmd_buf_row_base < HT) &&
                ((ifm_cmd_buf_row_base + ifm_cmd_num_rows) <= HT);
        end
        else begin
            ifm_cmd_shape_ok =
                (ifm_cmd_num_rows != 0) &&
                (cfg_c_in != 0) &&
                ((cfg_w_in >= 1) && (cfg_w_in <= PC)) &&
                ((ifm_cmd_buf_row_base + ifm_cmd_num_rows) <= H_MAX);
        end

        ifm_cmd_range_ok =
            (ifm_total_words_in != 0) &&
            (ifm_cmd_ddr_base >= `DDR_IFM_BASE) &&
            (ifm_last_addr_in <= DDR_IFM_END);

        wgt_cmd_range_ok =
            (wgt_cmd_num_words != 0) &&
            (wgt_cmd_num_words <= WGT_DEPTH) &&
            (wgt_cmd_ddr_base >= `DDR_WGT_BASE) &&
            (wgt_last_addr_in <= DDR_WGT_END);

        ofm_cmd_range_ok =
            (ofm_total_words_in != 0) &&
            (ofm_cmd_ddr_base >= `DDR_OFM_BASE) &&
            (ofm_last_addr_in <= DDR_OFM_END) &&
            ((ofm_cmd_buf_base + ofm_cmd_num_words) <= OFM_LINEAR_DEPTH);
    end

    //==================================================
    // Main FSM
    //==================================================
    always_comb begin
        state_n            = state;
        cmd_done_n         = cmd_done_q;

        ddr_addr_base_n    = ddr_addr_base_q;
        ddr_linear_idx_n   = ddr_linear_idx_q;

        ifm_bank_n         = ifm_bank_q;
        ifm_row_iter_n     = ifm_row_iter_q;
        ifm_col_iter_n     = ifm_col_iter_q;
        ifm_num_rows_n     = ifm_num_rows_q;
        ifm_buf_row_base_n = ifm_buf_row_base_q;

        wgt_buf_sel_n      = wgt_buf_sel_q;
        wgt_num_words_n    = wgt_num_words_q;
        wgt_word_idx_n     = wgt_word_idx_q;
        wgt_pack_n         = wgt_pack_q;
        wgt_pack_cnt_n     = wgt_pack_cnt_q;
        wgt_word_ready_n   = wgt_word_ready_q;

        ofm_num_words_n    = ofm_num_words_q;
        ofm_word_idx_n     = ofm_word_idx_q;
        ofm_buf_base_n     = ofm_buf_base_q;

        rd_data_hold_n     = rd_data_hold_q;
        rd_hold_valid_n    = rd_hold_valid_q;

        done               = 1'b0;
        done_ifm           = 1'b0;
        done_wgt           = 1'b0;
        done_ofm           = 1'b0;
        error              = 1'b0;

        ddr_rd_req         = 1'b0;
        ddr_rd_addr        = ddr_addr_base_q + ddr_linear_idx_q;

        ddr_wr_en          = 1'b0;
        ddr_wr_addr        = ddr_addr_base_q + ofm_word_idx_q;
        ddr_wr_data        = '0;
        ddr_wr_be          = '0;

        ifm_dma_wr_en      = 1'b0;
        ifm_dma_wr_bank    = ifm_bank_q[$clog2(C_MAX)-1:0];
        ifm_dma_wr_row_idx = ifm_buf_row_base_q + ifm_row_iter_q[$clog2(H_MAX)-1:0];
        ifm_dma_wr_col_idx = ifm_col_iter_q[IFM_COL_W-1:0];
        ifm_dma_wr_data    = rd_data_hold_q;
        ifm_dma_wr_keep    = keep_mask_v;

        wgt_dma_wr_en       = 1'b0;
        wgt_dma_wr_buf_sel  = wgt_buf_sel_q;
        wgt_dma_wr_addr     = wgt_word_idx_q[$clog2(WGT_DEPTH)-1:0];
        wgt_dma_wr_data     = wgt_pack_q;
        // Full physical-word write. Any mode-1 tail handling must already be
        // padded/rounded by the weight preload schedule into full PTOTAL words.
        wgt_dma_wr_keep     = {WGT_WORD_LANES{1'b1}};
        wgt_dma_load_done   = 1'b0;
        wgt_dma_load_buf_sel = wgt_buf_sel_q;

        ofm_dma_rd_en      = 1'b0;
        ofm_dma_rd_addr    = ofm_buf_base_q + ofm_word_idx_q[$clog2(OFM_LINEAR_DEPTH)-1:0];

        case (state)
            ST_IDLE: begin
                cmd_done_n = CMD_NONE;

                if (ifm_cmd_start) begin
                    if (!ifm_cmd_shape_ok || !ifm_cmd_range_ok) begin
                        error   = 1'b1;
                        state_n = ST_DONE;
                    end
                    else begin
                        ddr_addr_base_n    = ifm_cmd_ddr_base;
                        ddr_linear_idx_n   = '0;

                        ifm_bank_n         = '0;
                        ifm_row_iter_n     = '0;
                        ifm_col_iter_n     = '0;
                        ifm_num_rows_n     = ifm_cmd_num_rows;
                        ifm_buf_row_base_n = ifm_cmd_buf_row_base;

                        rd_hold_valid_n    = 1'b0;
                        state_n            = ST_IFM_REQ;
                    end
                end
                else if (wgt_cmd_start) begin
                    if (!wgt_cmd_range_ok) begin
                        error   = 1'b1;
                        state_n = ST_DONE;
                    end
                    else begin
                        ddr_addr_base_n    = wgt_cmd_ddr_base;
                        ddr_linear_idx_n   = '0;

                        wgt_buf_sel_n      = wgt_cmd_buf_sel;
                        wgt_num_words_n    = wgt_cmd_num_words;
                        wgt_word_idx_n     = '0;
                        wgt_pack_n         = '0;
                        wgt_pack_cnt_n     = '0;
                        wgt_word_ready_n   = 1'b0;

                        rd_hold_valid_n    = 1'b0;
                        state_n            = ST_WGT_REQ;
                    end
                end
                else if (ofm_cmd_start) begin
                    if (!ofm_cmd_range_ok) begin
                        error   = 1'b1;
                        state_n = ST_DONE;
                    end
                    else begin
                        ddr_addr_base_n    = ofm_cmd_ddr_base;
                        ofm_num_words_n    = ofm_cmd_num_words;
                        ofm_word_idx_n     = '0;
                        ofm_buf_base_n     = ofm_cmd_buf_base;
                        state_n            = ST_OFM_REQ;
                    end
                end
            end

            //==========================================
            // IFM: DDR -> ifm_buffer
            //==========================================
            ST_IFM_REQ: begin
                ddr_rd_req  = 1'b1;
                ddr_rd_addr = ifm_ddr_addr_v;
                state_n     = ST_IFM_WAIT;
            end

            ST_IFM_WAIT: begin
                if (ddr_rd_valid) begin
                    rd_data_hold_n  = ddr_rd_data;
                    rd_hold_valid_n = 1'b1;
                end

                if (rd_hold_valid_q && ifm_dma_wr_ready) begin
                    ifm_dma_wr_en    = 1'b1;
                    rd_hold_valid_n  = 1'b0;
                    ddr_linear_idx_n = ddr_linear_idx_q + 1'b1;

                    if (!cfg_mode) begin
                        if (ifm_col_iter_q + 1 < mode1_groups_v) begin
                            ifm_col_iter_n = ifm_col_iter_q + 1'b1;
                            state_n        = ST_IFM_REQ;
                        end
                        else begin
                            ifm_col_iter_n = '0;
                            if (ifm_row_iter_q + 1 < ifm_num_rows_q) begin
                                ifm_row_iter_n = ifm_row_iter_q + 1'b1;
                                state_n        = ST_IFM_REQ;
                            end
                            else begin
                                ifm_row_iter_n = '0;
                                if (ifm_bank_q + 1 < cfg_c_in) begin
                                    ifm_bank_n = ifm_bank_q + 1'b1;
                                    state_n    = ST_IFM_REQ;
                                end
                                else begin
                                    cmd_done_n = CMD_IFM;
                                    state_n    = ST_DONE;
                                end
                            end
                        end
                    end
                    else begin
                        ifm_dma_wr_col_idx = '0;

                        if (ifm_row_iter_q + 1 < ifm_num_rows_q) begin
                            ifm_row_iter_n = ifm_row_iter_q + 1'b1;
                            state_n        = ST_IFM_REQ;
                        end
                        else begin
                            ifm_row_iter_n = '0;
                            if (ifm_bank_q + 1 < cfg_c_in) begin
                                ifm_bank_n = ifm_bank_q + 1'b1;
                                state_n    = ST_IFM_REQ;
                            end
                            else begin
                                cmd_done_n = CMD_IFM;
                                state_n    = ST_DONE;
                            end
                        end
                    end
                end
            end

            //==========================================
            // WGT: DDR -> weight_buffer
            //==========================================
            ST_WGT_REQ: begin
                if (wgt_word_idx_q < wgt_num_words_q) begin
                    ddr_rd_req  = 1'b1;
                    ddr_rd_addr = ddr_addr_base_q + ddr_linear_idx_q;
                    state_n     = ST_WGT_WAIT;
                end
                else begin
                    cmd_done_n = CMD_WGT;
                    state_n    = ST_DONE;
                end
            end

            ST_WGT_WAIT: begin
                // Assemble one PTOTAL-lane physical weight word from one or more
                // DDR reads. The bit/lane order is preserved exactly as streamed
                // from DDR; mode-specific logical interpretation happens later.
                if (ddr_rd_valid && !wgt_word_ready_q) begin
                    wgt_pack_n = wgt_pack_q;
                    base_bit   = wgt_pack_cnt_q * DDR_WORD_W;
                    for (sub = 0; sub < DDR_WORD_W; sub++) begin
                        if ((base_bit + sub) < WGT_WORD_W)
                            wgt_pack_n[base_bit + sub] = ddr_rd_data[sub];
                    end

                    ddr_linear_idx_n = ddr_linear_idx_q + 1'b1;

                    if (wgt_pack_cnt_q + 1 >= WGT_SUBWORDS) begin
                        wgt_pack_cnt_n   = '0;
                        wgt_word_ready_n = 1'b1;
                    end
                    else begin
                        wgt_pack_cnt_n = wgt_pack_cnt_q + 1'b1;
                        state_n        = ST_WGT_REQ;
                    end
                end

                if (wgt_word_ready_q && wgt_dma_wr_ready) begin
                    wgt_dma_wr_en      = 1'b1;
                    wgt_word_idx_n     = wgt_word_idx_q + 1'b1;
                    wgt_word_ready_n   = 1'b0;
                    wgt_pack_cnt_n     = '0;
                    wgt_pack_n         = '0;

                    if (wgt_word_idx_q + 1 < wgt_num_words_q)
                        state_n = ST_WGT_REQ;
                    else begin
                        cmd_done_n = CMD_WGT;
                        state_n    = ST_DONE;
                    end
                end
            end

            //==========================================
            // OFM: ofm_buffer -> DDR
            //==========================================
            ST_OFM_REQ: begin
                if (ofm_word_idx_q < ofm_num_words_q) begin
                    ofm_dma_rd_en   = 1'b1;
                    ofm_dma_rd_addr = ofm_buf_base_q + ofm_word_idx_q[$clog2(OFM_LINEAR_DEPTH)-1:0];
                    state_n         = ST_OFM_WAIT;
                end
                else begin
                    cmd_done_n = CMD_OFM;
                    state_n    = ST_DONE;
                end
            end

            ST_OFM_WAIT: begin
                if (ofm_dma_rd_valid) begin
                    ddr_wr_en   = 1'b1;
                    ddr_wr_addr = ddr_addr_base_q + ofm_word_idx_q;
                    ddr_wr_data = ofm_dma_rd_data;

                    for (int k = 0; k < PV_MAX; k++) begin
                        ddr_wr_be[k*(DATA_W/8) +: (DATA_W/8)] =
                            {(DATA_W/8){ofm_dma_rd_keep[k]}};
                    end

                    ofm_word_idx_n = ofm_word_idx_q + 1'b1;

                    if (ofm_word_idx_q + 1 < ofm_num_words_q)
                        state_n = ST_OFM_REQ;
                    else begin
                        cmd_done_n = CMD_OFM;
                        state_n    = ST_DONE;
                    end
                end
            end

            ST_DONE: begin
                done = 1'b1;
                case (cmd_done_q)
                    CMD_IFM: done_ifm = 1'b1;
                    CMD_WGT: begin
                        done_wgt            = 1'b1;
                        wgt_dma_load_done   = 1'b1;
                        wgt_dma_load_buf_sel = wgt_buf_sel_q;
                    end
                    CMD_OFM: done_ofm = 1'b1;
                    default: ;
                endcase
                state_n = ST_IDLE;
            end

            default: state_n = ST_IDLE;
        endcase
    end

    //==================================================
    // Registers
    //==================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= ST_IDLE;
            cmd_done_q         <= CMD_NONE;

            ddr_addr_base_q    <= '0;
            ddr_linear_idx_q   <= '0;

            ifm_bank_q         <= '0;
            ifm_row_iter_q     <= '0;
            ifm_col_iter_q     <= '0;
            ifm_num_rows_q     <= '0;
            ifm_buf_row_base_q <= '0;

            wgt_buf_sel_q      <= 1'b0;
            wgt_num_words_q    <= '0;
            wgt_word_idx_q     <= '0;
            wgt_pack_q         <= '0;
            wgt_pack_cnt_q     <= '0;
            wgt_word_ready_q   <= 1'b0;

            ofm_num_words_q    <= '0;
            ofm_word_idx_q     <= '0;
            ofm_buf_base_q     <= '0;

            rd_data_hold_q     <= '0;
            rd_hold_valid_q    <= 1'b0;
        end
        else begin
            state              <= state_n;
            cmd_done_q         <= cmd_done_n;

            ddr_addr_base_q    <= ddr_addr_base_n;
            ddr_linear_idx_q   <= ddr_linear_idx_n;

            ifm_bank_q         <= ifm_bank_n;
            ifm_row_iter_q     <= ifm_row_iter_n;
            ifm_col_iter_q     <= ifm_col_iter_n;
            ifm_num_rows_q     <= ifm_num_rows_n;
            ifm_buf_row_base_q <= ifm_buf_row_base_n;

            wgt_buf_sel_q      <= wgt_buf_sel_n;
            wgt_num_words_q    <= wgt_num_words_n;
            wgt_word_idx_q     <= wgt_word_idx_n;
            wgt_pack_q         <= wgt_pack_n;
            wgt_pack_cnt_q     <= wgt_pack_cnt_n;
            wgt_word_ready_q   <= wgt_word_ready_n;

            ofm_num_words_q    <= ofm_num_words_n;
            ofm_word_idx_q     <= ofm_word_idx_n;
            ofm_buf_base_q     <= ofm_buf_base_n;

            rd_data_hold_q     <= rd_data_hold_n;
            rd_hold_valid_q    <= rd_hold_valid_n;
        end
    end

    assign busy = (state != ST_IDLE);

endmodule
