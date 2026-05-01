module relu_mode2 #(
  parameter int PSUM_W = 32,
  parameter int PF     = 4
)(
  input  logic clk,
  input  logic rst_n,

  // Input from MAC array mode 2
  input  logic [PF*PSUM_W-1:0] data_in,
  input  logic                 data_in_valid,

  // Output to next stage (pooling / ofm buffer)
  output logic [PF*PSUM_W-1:0] data_out,
  output logic                 data_out_valid
);

  logic signed [PSUM_W-1:0] in_lane  [0:PF-1];
  logic signed [PSUM_W-1:0] out_lane [0:PF-1];


  // --------------------------------------------------
  // Unpack input
  // --------------------------------------------------
  always_comb begin
    for (int pf = 0; pf < PF; pf++) begin
      in_lane[pf] = signed'(data_in[pf*PSUM_W +: PSUM_W]);
    end
  end

  // --------------------------------------------------
  // ReLU function
  // --------------------------------------------------
  always_comb begin
    for (int pf = 0; pf < PF; pf++) begin
      if (in_lane[pf] > 0)
        out_lane[pf] = in_lane[pf];
      else
        out_lane[pf] = '0;
    end
  end

  // --------------------------------------------------
  // Output register: 1-cycle latency, aligned with mode2 wrapper
  // --------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      data_out       <= '0;
      data_out_valid <= 1'b0;
    end
    else begin
      data_out_valid <= data_in_valid;

      if (data_in_valid) begin
        for (int pf = 0; pf < PF; pf++) begin
          data_out[pf*PSUM_W +: PSUM_W] <= out_lane[pf];
        end
      end
      else begin
        data_out <= '0;
      end
    end
  end

endmodule
