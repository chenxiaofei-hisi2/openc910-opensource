/* ct_vgemm_top — vgemm 指令封装: 把 K 装进 vconp CSR 几何的退化配置
 *   csr0_in: k_len[7:0] (GEMM reduction 维)
 *   映射: cin=K, kh=kw=1, csh=csw=dh=dw=1, ph=pw=1, pool_max=0
 *   ⇒ 统一内核自动退化为 Conv1×1 无池化 = GEMM 行 × 16 列 */
module ct_vgemm_top(
  input  wire clk, rst_n, start,
  output wire busy, done,
  input  wire [31:0] x_base, w_base, y_base,
  input  wire [31:0] csr0, csr1, csr2, csr3,
  input  wire [31:0] bias [0:15],
  input  wire [31:0] bn_s [0:15],
  input  wire [31:0] bn_b [0:15],
  output wire x_re, w_re, y_we,
  output wire [31:0] x_addr, w_addr, y_addr,
  input  wire [127:0] x_rdata, w_rdata,
  output wire [127:0] y_wdata
);
  wire [7:0] k_len = csr0[7:0];
  wire relu_en = csr1[21];
  wire bn_en   = csr1[22];
  // 统一 CSR 打包: kh=kw=csh=csw=dh=dw=ph=pw=1, pool_max=0, in_w=1(未用)
  wire [31:0] ucsr0 = {4'd1,4'd1,4'd1,4'd1,4'd1,4'd1, k_len};
  wire [31:0] ucsr1 = {9'b0, bn_en, relu_en, 1'b1, 2'b0, 10'd1, 4'd1, 4'd1};   // pool_max=1: 单元素 max=恒等直通(避免 avg 双重 rq)
  ct_vcop_top u_core(.clk(clk), .rst_n(rst_n), .start(start),
                     .busy(busy), .done(done),
                     .x_base(x_base), .w_base(w_base), .y_base(y_base),
                     .csr0(ucsr0), .csr1(ucsr1), .csr2(csr2), .csr3(csr3),
                     .bias(bias), .bn_s(bn_s), .bn_b(bn_b),
                     .x_re(x_re), .w_re(w_re), .y_we(y_we),
                     .x_addr(x_addr), .w_addr(w_addr), .y_addr(y_addr),
                     .x_rdata(x_rdata), .w_rdata(w_rdata), .y_wdata(y_wdata));
endmodule
