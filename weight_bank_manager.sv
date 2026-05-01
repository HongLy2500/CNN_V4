module weight_bank_manager (
  input  logic clk,
  input  logic rst_n,

  // Bắt đầu một phiên chạy mới: preload đầu tiên sẽ vào bank 0
  input  logic start_first_preload,

  // DMA đã preload xong bank hiện đang là preload target
  input  logic preload_done,

  // Layer hiện tại đã compute xong
  input  logic layer_done,

  // Scheduler yêu cầu đổi sang bank preload cho layer kế tiếp
  input  logic swap_req,

  // Từ weight_buffer
  input  logic bank0_ready,
  input  logic bank1_ready,

  // Bank đang được compute sử dụng
  output logic compute_bank_sel,

  // Bank mục tiêu để preload layer kế tiếp
  output logic preload_bank_sel,

  // Ready của bank đang compute dùng
  output logic compute_bank_ready,

  // Pulse 1 chu kỳ để release bank cũ sau khi swap
  output logic bank0_release,
  output logic bank1_release
);

  // --------------------------------------------------------------------------
  // State
  // --------------------------------------------------------------------------
  logic compute_bank_sel_q, compute_bank_sel_d;
  logic preload_bank_sel_q, preload_bank_sel_d;
  logic compute_active_q,   compute_active_d;

  logic target_bank_ready;
  logic swap_fire;

  // --------------------------------------------------------------------------
  // Helper
  // --------------------------------------------------------------------------
  always_comb begin
    target_bank_ready =
      (preload_bank_sel_q == 1'b0) ? bank0_ready : bank1_ready;
  end

  // Chỉ swap khi:
  // - scheduler yêu cầu swap
  // - layer hiện tại đã xong
  // - đã có compute bank hợp lệ
  // - bank preload target đã ready
  assign swap_fire =
      swap_req
    && layer_done
    && compute_active_q
    && target_bank_ready;

  // --------------------------------------------------------------------------
  // Next-state
  // --------------------------------------------------------------------------
  always_comb begin
    compute_bank_sel_d = compute_bank_sel_q;
    preload_bank_sel_d = preload_bank_sel_q;
    compute_active_d   = compute_active_q;

    // Bắt đầu job mới:
    // - chưa có compute bank active
    // - preload đầu tiên vào bank 0
    if (start_first_preload) begin
      compute_bank_sel_d = 1'b0;
      preload_bank_sel_d = 1'b0;
      compute_active_d   = 1'b0;
    end
    else begin
      // Sau preload đầu tiên, bank vừa preload xong trở thành compute bank
      if (preload_done && !compute_active_q) begin
        compute_bank_sel_d = preload_bank_sel_q;
        preload_bank_sel_d = ~preload_bank_sel_q;
        compute_active_d   = 1'b1;
      end

      // Các preload tiếp theo chỉ làm bank target trở thành ready ở weight_buffer.
      // Việc đổi compute bank chỉ xảy ra khi scheduler yêu cầu swap hợp lệ.
      if (swap_fire) begin
        // Bank preload hiện tại trở thành compute bank mới
        compute_bank_sel_d = preload_bank_sel_q;

        // Bank compute cũ trở thành preload target mới
        preload_bank_sel_d = compute_bank_sel_q;

        compute_active_d   = 1'b1;
      end
    end
  end

  // --------------------------------------------------------------------------
  // Registers
  // --------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      compute_bank_sel_q <= 1'b0;
      preload_bank_sel_q <= 1'b0;
      compute_active_q   <= 1'b0;
    end
    else begin
      compute_bank_sel_q <= compute_bank_sel_d;
      preload_bank_sel_q <= preload_bank_sel_d;
      compute_active_q   <= compute_active_d;
    end
  end

  // --------------------------------------------------------------------------
  // Outputs
  // --------------------------------------------------------------------------
  assign compute_bank_sel = compute_bank_sel_q;
  assign preload_bank_sel = preload_bank_sel_q;

  always_comb begin
    if (!compute_active_q) begin
      compute_bank_ready = 1'b0;
    end
    else begin
      compute_bank_ready =
        (compute_bank_sel_q == 1'b0) ? bank0_ready : bank1_ready;
    end
  end

  // Release pulse đúng 1 chu kỳ khi swap thành công.
  // Release bank compute cũ để bank đó có thể dùng cho preload layer tiếp theo.
  assign bank0_release = swap_fire && (compute_bank_sel_q == 1'b0);
  assign bank1_release = swap_fire && (compute_bank_sel_q == 1'b1);

endmodule