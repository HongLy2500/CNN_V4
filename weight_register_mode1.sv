module weight_register_mode1 #(
  parameter int DATA_W = 8,
  parameter int PF_MAX = 8,
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
  input  logic load_en,
  input  logic clear,

  // =====================================================
  // Input logic weights
  // Only the first Pf_cur entries are active.
  // =====================================================
  input  logic signed [DATA_W-1:0] weight_in_logic [0:PF_MAX-1],

  // =====================================================
  // Output physical MAC lanes
  // Runtime lane mapping:
  //   lane = pf * Pv_cur + pv
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


  integer pf;
  integer pv;
  integer lane_idx;

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

  // =====================================================
  // Duplicate logic weights along the Pv dimension
  // =====================================================
  always_comb begin
    // Default all lanes to zero
    for (int i = 0; i < PTOTAL; i++) begin
      weight_out_lane[i] = '0;
    end

    // Fill active lanes only
    for (pf = 0; pf < PF_MAX; pf++) begin
      if (pf < Pf_cur) begin
        for (pv = 0; pv < PTOTAL; pv++) begin
          if (pv < Pv_cur) begin
            lane_idx = pf * Pv_cur + pv;
            if (lane_idx < PTOTAL) begin
              weight_out_lane[lane_idx] = weight_logic_reg[pf];
            end
          end
        end
      end
    end
  end

endmodule