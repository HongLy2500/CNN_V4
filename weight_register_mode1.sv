module weight_register_mode1 #(
  parameter int DATA_W = 8,
  parameter int PF_MAX = 16,
  parameter int PTOTAL = 128,
  // Synthesis-friendly fast mapping for the current P128 profile.
  // Physical lane layout:
  //   lane = pf * PV_MAX_LOCAL + pv
  //
  // This removes the runtime scatter:
  //   lane = pf * Pv_cur + pv
  // which is very expensive for Vivado when PTOTAL=128.
  parameter int PV_MAX_LOCAL = (PF_MAX > 0) ? (PTOTAL / PF_MAX) : 1
)(
  input  logic clk,
  input  logic rst_n,

  // =====================================================
  // Runtime config
  // =====================================================
  input  logic [7:0] Pv_cur,
  input  logic [7:0] Pf_cur,

  // =====================================================
  // Control
  // =====================================================
  input  logic load_en,
  input  logic clear,

  // =====================================================
  // Input logic weights
  // Only the first Pf_cur entries are active.
  // =====================================================
  input  logic signed [DATA_W-1:0] weight_in_logic [0:PF_MAX-1],

  // =====================================================
  // Output physical MAC lanes
  // Fast/static mode-1 mapping:
  //   lane = pf * PV_MAX_LOCAL + pv
  // Invalid pf/pv lanes are masked to zero by Pf_cur/Pv_cur.
  // =====================================================
  output logic signed [DATA_W-1:0] weight_out_lane [0:PTOTAL-1],
  output logic                     weight_ready
);

  // =====================================================
  // Internal storage
  // Keep only the logic weights. Replication to PTOTAL
  // happens combinationally at the output.
  // =====================================================
  logic signed [DATA_W-1:0] weight_logic_reg [0:PF_MAX-1];
  logic                     weight_valid_q;

  // =====================================================
  // Latch logic weights
  // =====================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      weight_valid_q <= 1'b0;
      for (int i = 0; i < PF_MAX; i++) begin
        weight_logic_reg[i] <= '0;
      end
    end
    else begin
      if (clear) begin
        weight_valid_q <= 1'b0;
        for (int i = 0; i < PF_MAX; i++) begin
          weight_logic_reg[i] <= '0;
        end
      end
      else if (load_en) begin
        weight_valid_q <= 1'b1;
        for (int i = 0; i < PF_MAX; i++) begin
          weight_logic_reg[i] <= weight_in_logic[i];
        end
      end
    end
  end

  assign weight_ready = weight_valid_q;

  // =====================================================
  // Duplicate logic weights along the Pv dimension
  //
  // IMPORTANT:
  //   This is intentionally lane-oriented rather than scatter-oriented.
  //   Each output lane is driven by constants derived from the genvar index,
  //   so Vivado does not need to build a 128-lane dynamic demux.
  // =====================================================
  genvar gl;
  generate
    for (gl = 0; gl < PTOTAL; gl++) begin : GEN_WEIGHT_OUT_LANE
      localparam int PF_IDX = gl / PV_MAX_LOCAL;
      localparam int PV_IDX = gl % PV_MAX_LOCAL;

      always_comb begin
        weight_out_lane[gl] = '0;

        if ((PF_IDX < PF_MAX) &&
            (PF_IDX < int'(Pf_cur)) &&
            (PV_IDX < int'(Pv_cur)) &&
            weight_valid_q) begin
          weight_out_lane[gl] = weight_logic_reg[PF_IDX];
        end
      end
    end
  endgenerate

`ifndef SYNTHESIS
  // This fast implementation assumes the physical mode-1 lane contract
  // lane = pf * PV_MAX_LOCAL + pv. It still masks partial Pf/Pv, but it does
  // not compact lanes using lane = pf * Pv_cur + pv.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // no-op
    end
    else if (load_en) begin
      if (PV_MAX_LOCAL <= 0) begin
        $display("WR_M1_ERR_BAD_PV_MAX_LOCAL t=%0t PTOTAL=%0d PF_MAX=%0d PV_MAX_LOCAL=%0d",
                 $time, PTOTAL, PF_MAX, PV_MAX_LOCAL);
      end

      if (Pv_cur > PV_MAX_LOCAL) begin
        $display("WR_M1_WARN_PV_GT_STATIC t=%0t Pv_cur=%0d PV_MAX_LOCAL=%0d",
                 $time, Pv_cur, PV_MAX_LOCAL);
      end
    end
  end
`endif

endmodule
