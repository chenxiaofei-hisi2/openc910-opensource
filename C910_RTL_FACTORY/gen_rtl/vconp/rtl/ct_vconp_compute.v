/* ct_vconp_compute — 计算引擎: 权重窗 + 16×dot16 并行 (组合读)
 * wbuf 布局: [16oc][KK][CC][16B]; 尾组 lane 补零 ABI 由 esp-ppq 保证
 */
module ct_vconp_compute(
  input  wire         clk,
  input  wire [127:0] x_rdata,          // 16 × int8
  input  wire [7:0]   kk, cg, KK, CC,
  input  wire         wld_we,
  input  wire [15:0]  wld_cnt,
  input  wire [127:0] w_rdata,
  output wire signed [31:0] mac_dot [0:15]
);
  reg signed [7:0] wbuf [0:16*64*16*16-1];
  integer wi;
  always @(posedge clk) begin
    if (wld_we)
      for (wi=0; wi<16; wi=wi+1) wbuf[wld_cnt*16 + wi] <= w_rdata[wi*8 +: 8];
  end

  wire signed [7:0] xv [0:15];
  genvar gi;
  generate for (gi=0; gi<16; gi=gi+1) begin: XE
    assign xv[gi] = x_rdata[gi*8 +: 8];
  end endgenerate

  wire signed [15:0] prod [0:15][0:15];
  genvar oc, ln;
  generate
  for (oc=0; oc<16; oc=oc+1) begin: MACOC
    for (ln=0; ln<16; ln=ln+1) begin: PGEN
      assign prod[oc][ln] = xv[ln] * wbuf[((oc * KK + kk) * CC + cg) * 16 + ln];
    end
    assign mac_dot[oc] = (prod[oc][0]+prod[oc][1]+prod[oc][2]+prod[oc][3])
                       + (prod[oc][4]+prod[oc][5]+prod[oc][6]+prod[oc][7])
                       + (prod[oc][8]+prod[oc][9]+prod[oc][10]+prod[oc][11])
                       + (prod[oc][12]+prod[oc][13]+prod[oc][14]+prod[oc][15]);
  end
  endgenerate
endmodule
