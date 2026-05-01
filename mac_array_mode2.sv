module mac_array_mode2 #(
  parameter int DATA_W = 8,
  parameter int PSUM_W = 32,
  parameter int PC     = 8,
  parameter int PF     = 4
)(
  input  logic clk,
  input  logic rst_n,

  // Tich luy partial sum trong tung MAC
  input  logic mac_en,

  // Khi =1:
  // 1) cong cac psum cung filter de tao output
  // 2) chot output vao data_out
  // 3) clear toan bo psum noi bo
  input  logic out_valid,

  // data_in[pc*DATA_W +: DATA_W]
  input  logic [PC*DATA_W-1:0]        data_in,

  // weight_in[(pf*PC + pc)*DATA_W +: DATA_W]
  input  logic [PF*PC*DATA_W-1:0]     weight_in,

  // data_out[pf*PSUM_W +: PSUM_W]
  output logic [PF*PSUM_W-1:0]        data_out,
  output logic                        data_out_valid
);

  localparam int PROD_W = 2 * DATA_W;

  // --------------------------------------------------
  // Internal signals
  // --------------------------------------------------
  logic signed [DATA_W-1:0] data_lane   [0:PC-1];
  logic signed [DATA_W-1:0] weight_lane [0:PF-1][0:PC-1];
  logic signed [PROD_W-1:0] mult_res    [0:PF-1][0:PC-1];

  // psum luu noi bo trong tung MAC
  logic signed [PSUM_W-1:0] psum_reg    [0:PF-1][0:PC-1];

  // tong cuoi theo tung filter khi flush
  logic signed [PSUM_W-1:0] filt_sum    [0:PF-1];


  // --------------------------------------------------
  // Unpack data input
  // --------------------------------------------------
  always_comb begin
    for (int pc = 0; pc < PC; pc++) begin
      data_lane[pc] = signed'(data_in[pc*DATA_W +: DATA_W]);
    end
  end

  // --------------------------------------------------
  // Unpack weight input
  // --------------------------------------------------
  always_comb begin
    for (int pf = 0; pf < PF; pf++) begin
      for (int pc = 0; pc < PC; pc++) begin
        weight_lane[pf][pc] = signed'(weight_in[(pf*PC + pc)*DATA_W +: DATA_W]);
      end
    end
  end

  // --------------------------------------------------
  // Multiplier array
  // Moi MAC tinh 1 phep nhan
  // --------------------------------------------------
  always_comb begin
    for (int pf = 0; pf < PF; pf++) begin
      for (int pc = 0; pc < PC; pc++) begin
        mult_res[pf][pc] = data_lane[pc] * weight_lane[pf][pc];
      end
    end
  end

  // --------------------------------------------------
  // Final reduction across MACs of the same filter
  // Chi dung de tao output khi out_valid = 1
  // --------------------------------------------------
  always_comb begin
    for (int pf = 0; pf < PF; pf++) begin
      filt_sum[pf] = '0;
      for (int pc = 0; pc < PC; pc++) begin
        filt_sum[pf] = filt_sum[pf] + psum_reg[pf][pc];
      end
    end
  end

  // --------------------------------------------------
  // Sequential behavior
  //
  // Priority:
  // 1) out_valid: chot output va clear psum
  // 2) mac_en   : tich luy vao psum
  // --------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      data_out       <= '0;
      data_out_valid <= 1'b0;

      for (int pf = 0; pf < PF; pf++) begin
        for (int pc = 0; pc < PC; pc++) begin
          psum_reg[pf][pc] <= '0;
        end
      end
    end
    else begin
      data_out_valid <= 1'b0;

      if (out_valid) begin
        // Chot ket qua output cho tung filter
        for (int pf = 0; pf < PF; pf++) begin
          data_out[pf*PSUM_W +: PSUM_W] <= filt_sum[pf];
        end
        data_out_valid <= 1'b1;

        // Clear psum de tinh output tiep theo
        for (int pf = 0; pf < PF; pf++) begin
          for (int pc = 0; pc < PC; pc++) begin
            psum_reg[pf][pc] <= '0;
          end
        end
      end
      else if (mac_en) begin
        // Tich luy partial sum trong tung MAC
        for (int pf = 0; pf < PF; pf++) begin
          for (int pc = 0; pc < PC; pc++) begin
            psum_reg[pf][pc] <= psum_reg[pf][pc]
                              + PSUM_W'(mult_res[pf][pc]);
          end
        end
      end
    end
  end

endmodule