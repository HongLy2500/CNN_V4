(* use_dsp = "yes" *)
module mac_array_mode1 #(
  parameter int DATA_W = 8,
  parameter int PSUM_W = 8,
  parameter int PV_MAX = 16,
  parameter int PTOTAL = 256
)(
  input  logic clk,
  input  logic rst_n,

  // =====================================================
  // Runtime config
  // Synth-friendly P128 path assumes a static physical
  // Mode-1 lane layout:
  //   lane = pf * PV_MAX + pv
  // with Pf_static = PTOTAL / PV_MAX.
  // Pv_cur/Pf_cur are retained for lane-valid masking and
  // simulation checks, but are no longer used to build a
  // dynamic lane scatter network.
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
  // - physical lane pv is data_in_logic[pv]
  // =====================================================
  input  logic signed [PV_MAX*DATA_W-1:0] data_in_logic,

  // =====================================================
  // Weight input from weight_register_mode1
  // - fixed width = PTOTAL lanes
  // - expected physical lane mapping:
  //     lane = pf * PV_MAX + pv
  // =====================================================
  input  logic signed [DATA_W-1:0] weight_in_lane [0:PTOTAL-1],

  // =====================================================
  // Output psum
  // - one psum per physical MAC lane
  // - fixed width = PTOTAL lanes
  // =====================================================
  output logic signed [PSUM_W-1:0] psum_out_lane [0:PTOTAL-1]
);

  localparam int PF_STATIC = (PV_MAX > 0) ? (PTOTAL / PV_MAX) : 0;
  localparam logic [7:0] PV_MAX_CFG    = PV_MAX;
  localparam logic [7:0] PF_STATIC_CFG = PF_STATIC;

  logic signed [DATA_W-1:0] data_logic_lane [0:PV_MAX-1];

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
  // One multiply-accumulate lane per physical MAC lane.
  //
  // Previous DSP-hint attempt put use_dsp on the unpacked
  // prod_lane array, which Vivado did not map to DSPs.  This
  // version places the DSP hint on each per-lane product signal
  // inside the generate block and keeps the multiply-add pattern
  // local to the lane.  Functionality is unchanged:
  //   clear_psum clears all psums
  //   mac_en accumulates data*weight for valid lanes
  //   invalid lanes add zero / hold their psum
  // =====================================================
  generate
    genvar gl;
    for (gl = 0; gl < PTOTAL; gl++) begin : GEN_MAC_LANE
      localparam int PV_IDX = gl % PV_MAX;
      localparam int PF_IDX = gl / PV_MAX;

      logic lane_active;
      (* use_dsp = "yes" *) logic signed [(2*DATA_W)-1:0] prod_dsp;

      always_comb begin
        lane_active = (PV_IDX < Pv_cur) && (PF_IDX < Pf_cur);
        prod_dsp    = $signed(data_logic_lane[PV_IDX]) * $signed(weight_in_lane[gl]);
      end

      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          psum_out_lane[gl] <= '0;
        end
        else if (clear_psum) begin
          psum_out_lane[gl] <= '0;
        end
        else if (mac_en) begin
          if (lane_active) begin
            psum_out_lane[gl] <= psum_out_lane[gl] + $signed(prod_dsp);
          end
          else begin
            psum_out_lane[gl] <= psum_out_lane[gl];
          end
        end
      end
    end
  endgenerate

`ifndef SYNTHESIS
  // This module is intentionally optimized for the fixed P128
  // mapping used by the current bitstream target.  If a future
  // test drives compact/dynamic Pv/Pf lane packing, simulation
  // should flag it early instead of silently producing misleading
  // results.
  always_ff @(posedge clk or negedge rst_n) begin
    if (rst_n && mac_en && !clear_psum) begin
      if ((Pv_cur != PV_MAX_CFG) || (Pf_cur != PF_STATIC_CFG)) begin
        $display("MAC_M1_WARN_STATIC_MAP t=%0t Pv_cur=%0d expected=%0d Pf_cur=%0d expected=%0d",
                 $time, Pv_cur, PV_MAX, Pf_cur, PF_STATIC);
      end
    end
  end
`endif

endmodule
