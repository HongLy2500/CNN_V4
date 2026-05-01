`timescale 1ns/1ps
`include "cnn_ddr_defs.svh"

module tb_cnn_top_2layer_m1_to_m1;
  import cnn_layer_desc_pkg::*;

  localparam int DATA_W=8, PSUM_W=32;
  localparam int PTOTAL=8, PV_MAX=4, PF_MAX=4, PV_M1=2, PF_M1=4;
  localparam int PC_MODE2=4, PF_MODE2=2;
  localparam int C_MAX=8, F_MAX=8, W_MAX=8, H_MAX=8, HT=8, K_MAX=3;
  localparam int WGT_DEPTH=128, OFM_BANK_DEPTH=H_MAX*W_MAX;
  localparam int OFM_LINEAR_DEPTH=C_MAX*OFM_BANK_DEPTH, CFG_DEPTH=4;
  localparam int DDR_ADDR_W=`CNN_DDR_ADDR_W, DDR_WORD_W=PV_MAX*DATA_W;
  localparam int MEM_DEPTH=(`DDR_RSVD_BASE + `DDR_RSVD_SIZE);
  localparam int CLK_PERIOD_NS=10, MAX_CYCLES=400000;

  // Layer 0: 6x6x3 -> conv3x3 4x4x8 -> pool 2x2x8
  localparam int L0_H_IN=6, L0_W_IN=6, L0_C_IN=3, L0_F_OUT=8, L0_K=3;
  localparam int L0_H_CONV_OUT=L0_H_IN-L0_K+1, L0_W_CONV_OUT=L0_W_IN-L0_K+1;
  localparam int L0_H_POOL_OUT=L0_H_CONV_OUT/2, L0_W_POOL_OUT=L0_W_CONV_OUT/2;

  // Layer 1: 2x2x8 -> conv1x1 2x2x8 -> pool 1x1x8
  localparam int L1_H_IN=L0_H_POOL_OUT, L1_W_IN=L0_W_POOL_OUT, L1_C_IN=L0_F_OUT;
  localparam int L1_F_OUT=8, L1_K=1;
  localparam int L1_H_CONV_OUT=L1_H_IN-L1_K+1, L1_W_CONV_OUT=L1_W_IN-L1_K+1;
  localparam int L1_H_POOL_OUT=L1_H_CONV_OUT/2, L1_W_POOL_OUT=L1_W_CONV_OUT/2;

  localparam int L0_IFM_WORDS_PER_ROW=(L0_W_IN+PV_M1-1)/PV_M1;
  localparam int L0_NUM_FGROUP=(L0_F_OUT+PF_M1-1)/PF_M1;
  localparam int L1_NUM_FGROUP=(L1_F_OUT+PF_M1-1)/PF_M1;
  localparam int L0_M1_LOGICAL_BUNDLES=L0_NUM_FGROUP*L0_C_IN*L0_K*L0_K;
  localparam int L1_M1_LOGICAL_BUNDLES=L1_NUM_FGROUP*L1_C_IN*L1_K*L1_K;
  localparam int L0_M1_PHYS_WORDS=(L0_M1_LOGICAL_BUNDLES+PV_M1-1)/PV_M1;
  localparam int L1_M1_PHYS_WORDS=(L1_M1_LOGICAL_BUNDLES+PV_M1-1)/PV_M1;
  localparam int WGT_SUBWORDS=(PTOTAL+PV_MAX-1)/PV_MAX;
  localparam int L0_M1_WGT_DDR_WORDS=L0_M1_PHYS_WORDS*WGT_SUBWORDS;
  localparam int L1_M1_WGT_DDR_WORDS=L1_M1_PHYS_WORDS*WGT_SUBWORDS;
  localparam int L0_WGT_DDR_BASE=`DDR_WGT_BASE;
  localparam int L1_WGT_DDR_BASE=`DDR_WGT_BASE+L0_M1_WGT_DDR_WORDS;
  localparam int FINAL_STORE_PACK=1;
  localparam int FINAL_GROUPS=(L1_W_POOL_OUT+FINAL_STORE_PACK-1)/FINAL_STORE_PACK;
  localparam int EXP_OFM_WORDS=L1_F_OUT*L1_H_POOL_OUT*FINAL_GROUPS;

  logic clk,rst_n,start,abort;
  logic cfg_wr_en;
  logic [$clog2(CFG_DEPTH)-1:0] cfg_wr_addr;
  layer_desc_t cfg_wr_data;
  logic [$clog2(CFG_DEPTH+1)-1:0] cfg_num_layers;

  logic ddr_rd_req, ddr_rd_valid, ddr_wr_en;
  logic [DDR_ADDR_W-1:0] ddr_rd_addr, ddr_wr_addr;
  logic [DDR_WORD_W-1:0] ddr_rd_data, ddr_wr_data;
  logic [(DDR_WORD_W/8)-1:0] ddr_wr_be;

  logic [15:0] m1_free_col_blk_g, m1_free_ch_blk_g;
  logic m1_sm_refill_req_ready, m1_sm_refill_req_valid;
  logic [$clog2(HT)-1:0] m1_sm_refill_row_slot_l;
  logic [15:0] m1_sm_refill_row_g, m1_sm_refill_col_blk_g, m1_sm_refill_ch_blk_g;
  logic m2_sm_refill_req_ready, m2_sm_refill_req_valid;
  logic [15:0] m2_sm_refill_row_g, m2_sm_refill_col_g, m2_sm_refill_col_l, m2_sm_refill_cgrp_g;
  logic ifm_m1_free_valid;
  logic [$clog2(HT)-1:0] ifm_m1_free_row_slot_l;
  logic [15:0] ifm_m1_free_row_g;
  logic busy,done,error;
  logic [$clog2(CFG_DEPTH)-1:0] dbg_layer_idx;
  logic dbg_mode, dbg_weight_bank;
  logic [3:0] dbg_error_vec;

  logic [DDR_WORD_W-1:0] ddr_mem [0:MEM_DEPTH-1];
  logic rd_pending_q;
  logic [DDR_ADDR_W-1:0] rd_addr_q;
  integer ddr_ofm_write_count, cycle_count, b;

  logic signed [DATA_W-1:0] l0_ifm [0:L0_C_IN-1][0:L0_H_IN-1][0:L0_W_IN-1];
  logic signed [DATA_W-1:0] l0_wgt [0:L0_F_OUT-1][0:L0_C_IN-1][0:L0_K-1][0:L0_K-1];
  integer signed l0_relu [0:L0_F_OUT-1][0:L0_H_CONV_OUT-1][0:L0_W_CONV_OUT-1];
  logic signed [DATA_W-1:0] l0_pool [0:L0_F_OUT-1][0:L0_H_POOL_OUT-1][0:L0_W_POOL_OUT-1];
  logic signed [DATA_W-1:0] l1_wgt [0:L1_F_OUT-1][0:L1_C_IN-1][0:L1_K-1][0:L1_K-1];
  integer signed l1_relu [0:L1_F_OUT-1][0:L1_H_CONV_OUT-1][0:L1_W_CONV_OUT-1];
  logic signed [DATA_W-1:0] l1_pool [0:L1_F_OUT-1][0:L1_H_POOL_OUT-1][0:L1_W_POOL_OUT-1];

  cnn_top #(
    .DATA_W(DATA_W), .PSUM_W(PSUM_W), .PTOTAL(PTOTAL), .PV_MAX(PV_MAX), .PF_MAX(PF_MAX),
    .PC_MODE2(PC_MODE2), .PF_MODE2(PF_MODE2), .C_MAX(C_MAX), .F_MAX(F_MAX),
    .W_MAX(W_MAX), .H_MAX(H_MAX), .HT(HT), .K_MAX(K_MAX), .WGT_DEPTH(WGT_DEPTH),
    .OFM_BANK_DEPTH(OFM_BANK_DEPTH), .OFM_LINEAR_DEPTH(OFM_LINEAR_DEPTH),
    .CFG_DEPTH(CFG_DEPTH), .DDR_ADDR_W(DDR_ADDR_W), .DDR_WORD_W(DDR_WORD_W)
  ) dut (
    .clk(clk), .rst_n(rst_n), .start(start), .abort(abort),
    .cfg_wr_en(cfg_wr_en), .cfg_wr_addr(cfg_wr_addr), .cfg_wr_data(cfg_wr_data), .cfg_num_layers(cfg_num_layers),
    .ddr_rd_req(ddr_rd_req), .ddr_rd_addr(ddr_rd_addr), .ddr_rd_valid(ddr_rd_valid), .ddr_rd_data(ddr_rd_data),
    .ddr_wr_en(ddr_wr_en), .ddr_wr_addr(ddr_wr_addr), .ddr_wr_data(ddr_wr_data), .ddr_wr_be(ddr_wr_be),
    .m1_free_col_blk_g(m1_free_col_blk_g), .m1_free_ch_blk_g(m1_free_ch_blk_g),
    .m1_sm_refill_req_ready(m1_sm_refill_req_ready), .m1_sm_refill_req_valid(m1_sm_refill_req_valid),
    .m1_sm_refill_row_slot_l(m1_sm_refill_row_slot_l), .m1_sm_refill_row_g(m1_sm_refill_row_g),
    .m1_sm_refill_col_blk_g(m1_sm_refill_col_blk_g), .m1_sm_refill_ch_blk_g(m1_sm_refill_ch_blk_g),
    .m2_sm_refill_req_ready(m2_sm_refill_req_ready), .m2_sm_refill_req_valid(m2_sm_refill_req_valid),
    .m2_sm_refill_row_g(m2_sm_refill_row_g), .m2_sm_refill_col_g(m2_sm_refill_col_g),
    .m2_sm_refill_col_l(m2_sm_refill_col_l), .m2_sm_refill_cgrp_g(m2_sm_refill_cgrp_g),
    .ifm_m1_free_valid(ifm_m1_free_valid), .ifm_m1_free_row_slot_l(ifm_m1_free_row_slot_l),
    .ifm_m1_free_row_g(ifm_m1_free_row_g),
    .busy(busy), .done(done), .error(error),
    .dbg_layer_idx(dbg_layer_idx), .dbg_mode(dbg_mode), .dbg_weight_bank(dbg_weight_bank),
    .dbg_error_vec(dbg_error_vec)
  );

  initial clk=1'b0;
  always #(CLK_PERIOD_NS/2) clk=~clk;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      ddr_rd_valid <= 1'b0; ddr_rd_data <= '0; rd_pending_q <= 1'b0; rd_addr_q <= '0;
      ddr_ofm_write_count <= 0; cycle_count <= 0;
    end else begin
      cycle_count <= cycle_count + 1;
      ddr_rd_valid <= 1'b0;
      if (rd_pending_q) begin
        ddr_rd_valid <= 1'b1;
        ddr_rd_data <= (rd_addr_q < MEM_DEPTH) ? ddr_mem[rd_addr_q] : '0;
      end
      if (ddr_wr_en) begin
        if (ddr_wr_addr < MEM_DEPTH) begin
          for (b=0; b<(DDR_WORD_W/8); b=b+1)
            if (ddr_wr_be[b]) ddr_mem[ddr_wr_addr][8*b +: 8] <= ddr_wr_data[8*b +: 8];
        end
        if ((ddr_wr_addr >= `DDR_OFM_BASE) && (ddr_wr_addr < (`DDR_OFM_BASE + `DDR_OFM_SIZE)))
          ddr_ofm_write_count <= ddr_ofm_write_count + 1;
      end
      if (ddr_rd_req) begin
        rd_pending_q <= 1'b1; rd_addr_q <= ddr_rd_addr;
      end else if (rd_pending_q) begin
        rd_pending_q <= 1'b0;
      end
    end
  end

  function automatic logic signed [DATA_W-1:0] sat_to_i8(input integer signed x);
    begin
      if (x > 127) sat_to_i8 = 8'sd127;
      else if (x < -128) sat_to_i8 = -8'sd128;
      else sat_to_i8 = $signed(x);
    end
  endfunction

  task automatic build_golden;
    integer c,r,x,f,ky,kx,pr,pc,dy,dx,sum,max_v;
    begin
      for (c=0;c<L0_C_IN;c=c+1)
        for (r=0;r<L0_H_IN;r=r+1)
          for (x=0;x<L0_W_IN;x=x+1)
            l0_ifm[c][r][x] = $signed(((c*17 + r*5 + x*3) % 9) - 4);

      for (f=0;f<L0_F_OUT;f=f+1)
        for (c=0;c<L0_C_IN;c=c+1)
          for (ky=0;ky<L0_K;ky=ky+1)
            for (kx=0;kx<L0_K;kx=kx+1)
              l0_wgt[f][c][ky][kx] = $signed(((f*13 + c*7 + ky*3 + kx*5) % 5) - 2);

      for (f=0;f<L0_F_OUT;f=f+1)
        for (r=0;r<L0_H_CONV_OUT;r=r+1)
          for (x=0;x<L0_W_CONV_OUT;x=x+1) begin
            sum=0;
            for (c=0;c<L0_C_IN;c=c+1)
              for (ky=0;ky<L0_K;ky=ky+1)
                for (kx=0;kx<L0_K;kx=kx+1)
                  sum += $signed(l0_ifm[c][r+ky][x+kx]) * $signed(l0_wgt[f][c][ky][kx]);
            l0_relu[f][r][x] = (sum > 0) ? sum : 0;
          end

      for (f=0;f<L0_F_OUT;f=f+1)
        for (pr=0;pr<L0_H_POOL_OUT;pr=pr+1)
          for (pc=0;pc<L0_W_POOL_OUT;pc=pc+1) begin
            max_v = l0_relu[f][2*pr][2*pc];
            for (dy=0;dy<2;dy=dy+1)
              for (dx=0;dx<2;dx=dx+1)
                if (l0_relu[f][2*pr+dy][2*pc+dx] > max_v) max_v = l0_relu[f][2*pr+dy][2*pc+dx];
            l0_pool[f][pr][pc] = sat_to_i8(max_v);
          end

      for (f=0;f<L1_F_OUT;f=f+1)
        for (c=0;c<L1_C_IN;c=c+1)
          for (ky=0;ky<L1_K;ky=ky+1)
            for (kx=0;kx<L1_K;kx=kx+1)
              l1_wgt[f][c][ky][kx] = $signed(((f*11 + c*5 + ky*7 + kx*3 + 1) % 5) - 2);

      for (f=0;f<L1_F_OUT;f=f+1)
        for (r=0;r<L1_H_CONV_OUT;r=r+1)
          for (x=0;x<L1_W_CONV_OUT;x=x+1) begin
            sum=0;
            for (c=0;c<L1_C_IN;c=c+1)
              for (ky=0;ky<L1_K;ky=ky+1)
                for (kx=0;kx<L1_K;kx=kx+1)
                  sum += $signed(l0_pool[c][r+ky][x+kx]) * $signed(l1_wgt[f][c][ky][kx]);
            l1_relu[f][r][x] = (sum > 0) ? sum : 0;
          end

      for (f=0;f<L1_F_OUT;f=f+1)
        for (pr=0;pr<L1_H_POOL_OUT;pr=pr+1)
          for (pc=0;pc<L1_W_POOL_OUT;pc=pc+1) begin
            max_v = l1_relu[f][2*pr][2*pc];
            for (dy=0;dy<2;dy=dy+1)
              for (dx=0;dx<2;dx=dx+1)
                if (l1_relu[f][2*pr+dy][2*pc+dx] > max_v) max_v = l1_relu[f][2*pr+dy][2*pc+dx];
            l1_pool[f][pr][pc] = sat_to_i8(max_v);
          end
    end
  endtask

  task automatic clear_ddr;
    integer i;
    begin
      for (i=0;i<MEM_DEPTH;i=i+1) ddr_mem[i]='0;
    end
  endtask

  task automatic load_l0_ifm_to_ddr;
    integer c,r,colg,lane,abs_col,addr;
    begin
      for (c=0;c<L0_C_IN;c=c+1)
        for (r=0;r<L0_H_IN;r=r+1)
          for (colg=0;colg<L0_IFM_WORDS_PER_ROW;colg=colg+1) begin
            addr = `DDR_IFM_BASE + ((c*L0_H_IN + r)*L0_IFM_WORDS_PER_ROW) + colg;
            ddr_mem[addr]='0;
            for (lane=0;lane<PV_MAX;lane=lane+1) begin
              abs_col = colg*PV_M1 + lane;
              if ((lane<PV_M1) && (abs_col<L0_W_IN))
                ddr_mem[addr][lane*DATA_W +: DATA_W] = l0_ifm[c][r][abs_col];
            end
          end
    end
  endtask

  task automatic write_wgt_lane_to_ddr(input integer base, input integer phys_word, input integer lane_abs, input logic signed [DATA_W-1:0] value);
    integer ddr_subword, ddr_lane, addr;
    begin
      ddr_subword = lane_abs / PV_MAX;
      ddr_lane    = lane_abs % PV_MAX;
      addr        = base + phys_word*WGT_SUBWORDS + ddr_subword;
      ddr_mem[addr][ddr_lane*DATA_W +: DATA_W] = value;
    end
  endtask

  task automatic load_l0_weights_to_ddr;
    integer p, logical_idx, phys_word, subword_idx, base_lane, fg,c,ky,kx,pf,f;
    begin
      for (p=0;p<L0_M1_WGT_DDR_WORDS;p=p+1) ddr_mem[L0_WGT_DDR_BASE+p]='0;
      logical_idx=0;
      for (fg=0;fg<L0_NUM_FGROUP;fg=fg+1)
        for (c=0;c<L0_C_IN;c=c+1)
          for (ky=0;ky<L0_K;ky=ky+1)
            for (kx=0;kx<L0_K;kx=kx+1) begin
              phys_word=logical_idx/PV_M1; subword_idx=logical_idx%PV_M1; base_lane=subword_idx*PF_M1;
              for (pf=0;pf<PF_M1;pf=pf+1) begin
                f=fg*PF_M1+pf;
                write_wgt_lane_to_ddr(L0_WGT_DDR_BASE, phys_word, base_lane+pf, (f<L0_F_OUT)?l0_wgt[f][c][ky][kx]:'0);
              end
              logical_idx++;
            end
    end
  endtask

  task automatic load_l1_weights_to_ddr;
    integer p, logical_idx, phys_word, subword_idx, base_lane, fg,c,ky,kx,pf,f;
    begin
      for (p=0;p<L1_M1_WGT_DDR_WORDS;p=p+1) ddr_mem[L1_WGT_DDR_BASE+p]='0;
      logical_idx=0;
      for (fg=0;fg<L1_NUM_FGROUP;fg=fg+1)
        for (c=0;c<L1_C_IN;c=c+1)
          for (ky=0;ky<L1_K;ky=ky+1)
            for (kx=0;kx<L1_K;kx=kx+1) begin
              phys_word=logical_idx/PV_M1; subword_idx=logical_idx%PV_M1; base_lane=subword_idx*PF_M1;
              for (pf=0;pf<PF_M1;pf=pf+1) begin
                f=fg*PF_M1+pf;
                write_wgt_lane_to_ddr(L1_WGT_DDR_BASE, phys_word, base_lane+pf, (f<L1_F_OUT)?l1_wgt[f][c][ky][kx]:'0);
              end
              logical_idx++;
            end
    end
  endtask

  task automatic init_mem;
    begin
      clear_ddr();
      build_golden();
      load_l0_ifm_to_ddr();
      load_l0_weights_to_ddr();
      load_l1_weights_to_ddr();
      $display("TB_INFO: 2-layer Mode1->Mode1 test");
      $display("TB_INFO: L0 6x6x3 -> conv 4x4x8 K=3 -> pool 2x2x8");
      $display("TB_INFO: L1 2x2x8 -> conv 2x2x8 K=1 -> pool 1x1x8");
      $display("TB_INFO: Pv=%0d Pf=%0d PTOTAL=%0d", PV_M1, PF_M1, PTOTAL);
      $display("TB_INFO: L0 weight bundles=%0d phys=%0d DDR words=%0d base=0x%05h", L0_M1_LOGICAL_BUNDLES, L0_M1_PHYS_WORDS, L0_M1_WGT_DDR_WORDS, L0_WGT_DDR_BASE);
      $display("TB_INFO: L1 weight bundles=%0d phys=%0d DDR words=%0d base=0x%05h", L1_M1_LOGICAL_BUNDLES, L1_M1_PHYS_WORDS, L1_M1_WGT_DDR_WORDS, L1_WGT_DDR_BASE);
      $display("TB_INFO: expected final OFM DDR words=%0d", EXP_OFM_WORDS);
    end
  endtask

  task automatic write_cfg(input logic [$clog2(CFG_DEPTH)-1:0] addr, input layer_desc_t cfg);
    begin
      @(posedge clk); cfg_wr_en<=1'b1; cfg_wr_addr<=addr; cfg_wr_data<=cfg;
      @(posedge clk); cfg_wr_en<=1'b0; cfg_wr_addr<='0; cfg_wr_data<='0;
    end
  endtask

  task automatic program_layers;
    layer_desc_t cfg0,cfg1;
    begin
      cfg0='0;
      cfg0.layer_id=0; cfg0.mode=MODE1; cfg0.h_in=L0_H_IN; cfg0.w_in=L0_W_IN; cfg0.c_in=L0_C_IN; cfg0.f_out=L0_F_OUT; cfg0.k=L0_K;
      cfg0.h_out=L0_H_CONV_OUT; cfg0.w_out=L0_W_CONV_OUT;
      cfg0.pv_m1=PV_M1; cfg0.pf_m1=PF_M1; cfg0.pc_m2=PC_MODE2; cfg0.pf_m2=PF_MODE2;
      cfg0.conv_stride=1; cfg0.relu_en=1'b1; cfg0.pool_en=1'b1; cfg0.pool_k=2; cfg0.pool_stride=2;
      cfg0.ifm_ddr_base=`DDR_IFM_BASE; cfg0.wgt_ddr_base=L0_WGT_DDR_BASE; cfg0.ofm_ddr_base=`DDR_OFM_BASE;
      cfg0.first_layer=1'b1; cfg0.last_layer=1'b0;

      cfg1='0;
      cfg1.layer_id=1; cfg1.mode=MODE1; cfg1.h_in=L1_H_IN; cfg1.w_in=L1_W_IN; cfg1.c_in=L1_C_IN; cfg1.f_out=L1_F_OUT; cfg1.k=L1_K;
      cfg1.h_out=L1_H_CONV_OUT; cfg1.w_out=L1_W_CONV_OUT;
      cfg1.pv_m1=PV_M1; cfg1.pf_m1=PF_M1; cfg1.pc_m2=PC_MODE2; cfg1.pf_m2=PF_MODE2;
      cfg1.conv_stride=1; cfg1.relu_en=1'b1; cfg1.pool_en=1'b1; cfg1.pool_k=2; cfg1.pool_stride=2;
      cfg1.ifm_ddr_base=`DDR_IFM_BASE; cfg1.wgt_ddr_base=L1_WGT_DDR_BASE; cfg1.ofm_ddr_base=`DDR_OFM_BASE;
      cfg1.first_layer=1'b0; cfg1.last_layer=1'b1;

      write_cfg('0,cfg0);
      write_cfg(1,cfg1);
    end
  endtask

  task automatic pulse_start;
    begin
      @(posedge clk); start<=1'b1;
      @(posedge clk); start<=1'b0;
    end
  endtask

  task automatic dump_ofm_region;
    integer j;
    begin
      $display("--- OFM DDR region dump, first %0d expected words ---", EXP_OFM_WORDS);
      for (j=0;j<EXP_OFM_WORDS;j=j+1)
        $display("OFM[%0d] @0x%05h = 0x%08h", j, (`DDR_OFM_BASE+j), ddr_mem[`DDR_OFM_BASE+j]);
    end
  endtask

  task automatic check_final_ofm;
    integer lin,ch,rem,row,grp,col,mismatch;
    logic signed [DATA_W-1:0] got,exp;
    begin
      mismatch=0;
      if (ddr_ofm_write_count != EXP_OFM_WORDS) begin
        $display("TB_FAIL: OFM DDR write count mismatch. got=%0d expected=%0d", ddr_ofm_write_count, EXP_OFM_WORDS);
        mismatch++;
      end
      for (lin=0;lin<EXP_OFM_WORDS;lin=lin+1) begin
        ch=lin/(L1_H_POOL_OUT*FINAL_GROUPS);
        rem=lin%(L1_H_POOL_OUT*FINAL_GROUPS);
        row=rem/FINAL_GROUPS;
        grp=rem%FINAL_GROUPS;
        col=grp*FINAL_STORE_PACK;
        got=ddr_mem[`DDR_OFM_BASE+lin][0 +: DATA_W];
        exp=l1_pool[ch][row][col];
        if (got !== exp) begin
          $display("TB_FAIL: final OFM mismatch lin=%0d ch=%0d row=%0d col=%0d got=%0d/0x%02h expected=%0d/0x%02h word=0x%08h",
                   lin,ch,row,col,got,got,exp,exp,ddr_mem[`DDR_OFM_BASE+lin]);
          mismatch++;
        end
      end
      if (mismatch==0) $display("TB_PASS: exact final OFM matches golden for 2-layer mode1->mode1 test");
      else begin
        $display("TB_FAIL: total final OFM mismatches = %0d", mismatch);
        dump_ofm_region();
        $finish;
      end
    end
  endtask

  initial begin
    rst_n=1'b0; start=1'b0; abort=1'b0;
    cfg_wr_en=1'b0; cfg_wr_addr='0; cfg_wr_data='0; cfg_num_layers=2;
    m1_free_col_blk_g='0; m1_free_ch_blk_g='0;
    m1_sm_refill_req_ready=1'b1; m2_sm_refill_req_ready=1'b1;

    init_mem();

    repeat (5) @(posedge clk);
    rst_n=1'b1;

    program_layers();
    repeat (5) @(posedge clk);
    pulse_start();

    wait (done || error || (cycle_count > MAX_CYCLES));
    repeat (5) @(posedge clk);

    if (error) begin
      $display("TB_FAIL: DUT asserted error. dbg_error_vec=0x%0h layer=%0d mode=%0d", dbg_error_vec, dbg_layer_idx, dbg_mode);
      dump_ofm_region();
      $finish;
    end
    if (cycle_count > MAX_CYCLES) begin
      $display("TB_FAIL: timeout after %0d cycles. busy=%0b done=%0b error=%0b layer=%0d mode=%0d err=0x%0h",
               cycle_count,busy,done,error,dbg_layer_idx,dbg_mode,dbg_error_vec);
      dump_ofm_region();
      $finish;
    end

    $display("TB_INFO: done observed after %0d cycles", cycle_count);
    $display("TB_INFO: OFM DDR writes counted = %0d", ddr_ofm_write_count);
    check_final_ofm();
    $finish;
  end

endmodule
