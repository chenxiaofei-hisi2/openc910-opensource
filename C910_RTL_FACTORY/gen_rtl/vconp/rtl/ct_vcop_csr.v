/* ct_vcop_csr — CSR 位域解包 (纯组合, 0 时钟)
 * CSR0: cin[7:0] kh[11:8] kw[15:12] csh[19:16] csw[23:20] dh[27:24] dw[31:28]
 * CSR1: ph[3:0] pw[7:4] in_w[17:8] pool_max[20] relu_en[21] bn_en[22]
 * CSR2: rq_mult(signed)  CSR3: rq_shift[4:0] bn_shift[9:5]
 */
module ct_vcop_csr(
  input  wire [31:0] csr0, csr1, csr2, csr3,
  output wire [7:0]  cin,
  output wire [3:0]  kh, kw, csh, csw, dh, dw, ph, pw,
  output wire [9:0]  in_w,
  output wire        pool_max, relu_en, bn_en,
  output wire signed [31:0] rq_mult,
  output wire [4:0]  rq_shift, bn_shift
);
  assign cin   = csr0[7:0];
  assign kh    = csr0[11:8];   assign kw  = csr0[15:12];
  assign csh   = csr0[19:16];  assign csw = csr0[23:20];
  assign dh    = csr0[27:24];  assign dw = csr0[31:28];
  assign ph    = csr1[3:0];    assign pw  = csr1[7:4];
  assign in_w  = csr1[17:8];
  assign pool_max = csr1[20];  assign relu_en = csr1[21];  assign bn_en = csr1[22];
  assign rq_mult  = csr2;      // signed 视图 (混合符号乘法退化教训)
  assign rq_shift = csr3[4:0]; assign bn_shift = csr3[9:5];
endmodule
