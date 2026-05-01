module weight_register_mode2 #(
  parameter int DATA_W = 8,
  parameter int PC     = 8,
  parameter int PF     = 4
)(
  input  logic clk,
  input  logic rst_n,

  input  logic                     write_en,
  input  logic [PF*PC*DATA_W-1:0]  write_data,

  output logic [PF*PC*DATA_W-1:0]  weight_out
);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      weight_out <= '0;
    end
    else if (write_en) begin
      weight_out <= write_data;
    end
  end

endmodule