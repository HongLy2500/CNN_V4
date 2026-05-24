module relu_mode1 #(
  parameter int PSUM_W = 32,
  parameter int PTOTAL = 256,
  parameter int OUT_W  = 8
)(
  input  logic signed [PSUM_W-1:0] in_data  [0:PTOTAL-1],
  output logic signed [PSUM_W-1:0] out_data [0:PTOTAL-1]
);

  // Resource-oriented ReLU stage for Mode1:
  // keep the same PSUM_W output interface/timing, but clamp each lane to the
  // final OUT_W signed positive range here. Downstream direct/pooling paths then
  // only need the low OUT_W bits instead of re-saturating PSUM_W values.
  localparam int signed OUT_MAX_I = (1 <<< (OUT_W-1)) - 1;

  integer i;
  integer signed din_i;

  always_comb begin
    for (i = 0; i < PTOTAL; i++) begin
      din_i = in_data[i];

      out_data[i] = '0;
      if (din_i <= 0) begin
        out_data[i] = '0;
      end else if (din_i > OUT_MAX_I) begin
        out_data[i][OUT_W-1:0] = OUT_MAX_I;
      end else begin
        out_data[i][OUT_W-1:0] = in_data[i][OUT_W-1:0];
      end
    end
  end

endmodule
