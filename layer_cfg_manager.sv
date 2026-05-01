module layer_cfg_manager
  import cnn_layer_desc_pkg::*;
#(
  parameter int CFG_DEPTH = 64,
  parameter int CFG_AW    = (CFG_DEPTH > 1) ? $clog2(CFG_DEPTH) : 1,
  parameter int CFG_NW    = $clog2(CFG_DEPTH + 1)
)(
  input  logic clk,
  input  logic rst_n,

  // Write interface for loading the layer table
  input  logic              cfg_wr_en,
  input  logic [CFG_AW-1:0] cfg_wr_addr,
  input  layer_desc_t       cfg_wr_data,
  input  logic [CFG_NW-1:0] cfg_num_layers,

  // Control from scheduler
  input  logic load_first,
  input  logic advance_layer,

  // Current / next layer view
  output logic              cur_valid,
  output logic              next_valid,
  output logic [CFG_AW-1:0] cur_layer_idx,

  output layer_desc_t       cur_cfg,
  output layer_desc_t       next_cfg,

  output logic              cur_first_layer,
  output logic              cur_last_layer
);

  // --------------------------------------------------------------------------
  // Layer table storage
  // --------------------------------------------------------------------------
  layer_desc_t cfg_mem [0:CFG_DEPTH-1];

  // Pointer to current layer
  logic [CFG_AW-1:0] cur_idx_q;

  // Becomes 1 after load_first when there is at least one valid layer
  logic active_q;

  // Registered layer descriptor outputs.
  //
  // Timing note:
  // The old version read cfg_mem directly in always_comb and drove cur_cfg/next_cfg
  // combinationally into control_unit_top/cnn_dma_direct. Vivado reported a long
  // path from cfg_mem RAMA_D1 to cnn_dma_direct registers. Registering the layer
  // descriptors here cuts that path at the layer_cfg_manager boundary.
  layer_desc_t cur_cfg_q;
  layer_desc_t next_cfg_q;

  logic cur_valid_q;
  logic next_valid_q;
  logic cur_first_layer_q;
  logic cur_last_layer_q;

  // Safe index helpers.  CFG_NW is used for comparisons against cfg_num_layers
  // to avoid wrap-around problems near the end of the table.  CFG_AW versions
  // are used only for indexing cfg_mem after the corresponding CFG_NW comparison
  // has proven the index is valid.
  logic [CFG_AW-1:0] cur_idx_plus1_aw;
  logic [CFG_AW-1:0] cur_idx_plus2_aw;
  logic [CFG_NW-1:0] cur_idx_nw;
  logic [CFG_NW-1:0] cur_idx_plus1_nw;
  logic [CFG_NW-1:0] cur_idx_plus2_nw;

  assign cur_idx_plus1_aw = cur_idx_q + CFG_AW'(1);
  assign cur_idx_plus2_aw = cur_idx_q + CFG_AW'(2);

  assign cur_idx_nw       = CFG_NW'(cur_idx_q);
  assign cur_idx_plus1_nw = cur_idx_nw + CFG_NW'(1);
  assign cur_idx_plus2_nw = cur_idx_nw + CFG_NW'(2);

  // --------------------------------------------------------------------------
  // Write layer descriptors
  // --------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (cfg_wr_en) begin
      cfg_mem[cfg_wr_addr] <= cfg_wr_data;
    end
  end

  // --------------------------------------------------------------------------
  // Current pointer / active flag / registered descriptor outputs
  // Priority:
  //   1) load_first
  //   2) advance_layer
  //
  // cur_cfg/next_cfg are updated only when the current layer pointer is loaded
  // or advanced.  This is intentional: layer config is a slow control value and
  // does not need an asynchronous/combinational read from cfg_mem every cycle.
  // --------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cur_idx_q          <= '0;
      active_q           <= 1'b0;
      cur_cfg_q          <= '0;
      next_cfg_q         <= '0;
      cur_valid_q        <= 1'b0;
      next_valid_q       <= 1'b0;
      cur_first_layer_q  <= 1'b0;
      cur_last_layer_q   <= 1'b0;
    end
    else begin
      if (load_first) begin
        cur_idx_q <= '0;
        active_q  <= (cfg_num_layers != '0);

        cur_valid_q       <= (cfg_num_layers != '0);
        cur_first_layer_q <= (cfg_num_layers != '0);
        cur_last_layer_q  <= (cfg_num_layers == CFG_NW'(1));
        next_valid_q      <= (cfg_num_layers >  CFG_NW'(1));

        if (cfg_num_layers != '0)
          cur_cfg_q <= cfg_mem[CFG_AW'(0)];
        else
          cur_cfg_q <= '0;

        if (cfg_num_layers > CFG_NW'(1))
          next_cfg_q <= cfg_mem[CFG_AW'(1)];
        else
          next_cfg_q <= '0;
      end
      else if (advance_layer) begin
        if (active_q && cur_valid_q && !cur_last_layer_q) begin
          cur_idx_q <= cur_idx_plus1_aw;

          cur_valid_q       <= 1'b1;
          cur_first_layer_q <= 1'b0;
          cur_last_layer_q  <= (cur_idx_plus1_nw == (cfg_num_layers - CFG_NW'(1)));
          next_valid_q      <= (cur_idx_plus2_nw < cfg_num_layers);

          cur_cfg_q <= cfg_mem[cur_idx_plus1_aw];

          if (cur_idx_plus2_nw < cfg_num_layers)
            next_cfg_q <= cfg_mem[cur_idx_plus2_aw];
          else
            next_cfg_q <= '0;
        end
      end
    end
  end

  // --------------------------------------------------------------------------
  // Registered outputs
  // --------------------------------------------------------------------------
  assign cur_layer_idx   = cur_idx_q;

  assign cur_valid       = cur_valid_q;
  assign next_valid      = next_valid_q;

  assign cur_cfg         = cur_cfg_q;
  assign next_cfg        = next_cfg_q;

  assign cur_first_layer = cur_first_layer_q;
  assign cur_last_layer  = cur_last_layer_q;

endmodule
