module mac_array_mode1 #(
  parameter int DATA_W = 8,
  parameter int PSUM_W = 8,
  parameter int PV_MAX = 8,
  parameter int PTOTAL = 16
)(
  input  logic clk,
  input  logic rst_n,

  // =====================================================
  // Runtime config
  // Valid mode-1 operation assumes:
  //   Pv_cur * Pf_cur = PTOTAL
  // =====================================================
  input  logic [7:0] Pv_cur,
  input  logic [7:0] Pf_cur,

  // =====================================================
  // Control
  // =====================================================
  input  logic mac_en,
  input  logic clear_psum,

  // =====================================================
  // Data input from data_register_mode1
  // - fixed width = PV_MAX * DATA_W
  // - only first Pv_cur lanes are active
  // =====================================================
  input  logic signed [PV_MAX*DATA_W-1:0] data_in_logic,

  // =====================================================
  // Weight input from weight_register_mode1
  // - fixed width = PTOTAL lanes
  // - runtime lane mapping is already handled by
  //   weight_register_mode1:
  //     lane = pf * Pv_cur + pv
  // =====================================================
  input  logic signed [DATA_W-1:0] weight_in_lane [0:PTOTAL-1],

  // =====================================================
  // Output psum
  // - one psum per physical MAC lane
  // - fixed width = PTOTAL lanes
  // =====================================================
  output logic signed [PSUM_W-1:0] psum_out_lane [0:PTOTAL-1]
);

  logic signed [DATA_W-1:0]     data_logic_lane [0:PV_MAX-1];
  logic signed [DATA_W-1:0]     data_lane       [0:PTOTAL-1];
  logic signed [(2*DATA_W)-1:0] prod_lane       [0:PTOTAL-1];


  integer pv;
  integer pf;
  integer lane_idx;

  // =====================================================
  // Unpack logic data lanes from packed input bus
  // =====================================================
  generate
    genvar gv;
    for (gv = 0; gv < PV_MAX; gv++) begin : GEN_UNPACK_DATA
      always_comb begin
        data_logic_lane[gv] = data_in_logic[gv*DATA_W +: DATA_W];
      end
    end
  endgenerate

  // =====================================================
  // Expand logic data lanes to physical MAC lanes
  // Runtime mapping:
  //   lane = pf * Pv_cur + pv
  //
  // This duplicates each data lane across all active pf groups.
  // =====================================================
  always_comb begin
    for (int i = 0; i < PTOTAL; i++) begin
      data_lane[i] = '0;
    end

    for (pf = 0; pf < PTOTAL; pf++) begin
      if (pf < Pf_cur) begin
        for (pv = 0; pv < PV_MAX; pv++) begin
          if (pv < Pv_cur) begin
            lane_idx = pf * Pv_cur + pv;
            if (lane_idx < PTOTAL) begin
              data_lane[lane_idx] = data_logic_lane[pv];
            end
          end
        end
      end
    end
  end
  // =====================================================
  // One multiplier per physical MAC lane
  // =====================================================
  generate
    genvar gl;
    for (gl = 0; gl < PTOTAL; gl++) begin : GEN_MULT
      always_comb begin
        prod_lane[gl] = data_lane[gl] * weight_in_lane[gl];
      end
    end
  endgenerate

  // =====================================================
  // One accumulator per physical MAC lane
  // No adder tree in mode 1
  // =====================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int i = 0; i < PTOTAL; i++) begin
        psum_out_lane[i] <= '0;
      end
    end
    else begin
      if (clear_psum) begin
        for (int i = 0; i < PTOTAL; i++) begin
          psum_out_lane[i] <= '0;
        end
      end
      else if (mac_en) begin
        for (int i = 0; i < PTOTAL; i++) begin
          psum_out_lane[i] <= psum_out_lane[i] + $signed(prod_lane[i]);
        end
      end
    end
  end

endmodule