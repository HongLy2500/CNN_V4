module relu_mode1 #(
  parameter int PSUM_W = 32,
  parameter int PTOTAL = 16
)(
  input  logic signed [PSUM_W-1:0] in_data  [0:PTOTAL-1],
  output logic signed [PSUM_W-1:0] out_data [0:PTOTAL-1]
);

  integer i;

  always_comb begin
    for (i = 0; i < PTOTAL; i++) begin
      if (in_data[i] > 0)
        out_data[i] = in_data[i];
      else
        out_data[i] = '0;
    end
  end

endmodule
