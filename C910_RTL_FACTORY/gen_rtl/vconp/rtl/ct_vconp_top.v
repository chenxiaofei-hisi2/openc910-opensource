/* ct_vconp_top — vconp 指令封装: CSR 原样直通统一内核 ct_vcop_top
 * (Conv→[ReLU]→Pool→[BN]; NHWC 滑窗模式) */
module ct_vconp_top(
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
  ct_vcop_top u_core(.*);
endmodule
