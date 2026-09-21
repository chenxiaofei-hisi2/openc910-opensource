/* ct_vconp_agu — 滑窗地址生成 (纯组合)
 * 通用公式: row = ppy·csh + ky·dh ; col = ppx·csw + kx·dw
 *           xoff = (row·in_w + col)·cin + cg·16   (NHWC)
 */
module ct_vconp_agu(
  input  wire [7:0]  pp, kk, cg,
  input  wire [3:0]  pw, kw, csh, csw, dh, dw,
  input  wire [7:0]  cin,
  input  wire [9:0]  in_w,
  input  wire [31:0] x_base,
  output wire [31:0] x_addr
);
  wire [3:0]  ppy = pp / pw;
  wire [3:0]  ppx = pp % pw;
  wire [3:0]  ky  = kk / kw;
  wire [3:0]  kx  = kk % kw;
  wire [31:0] row = ppy * csh + ky * dh;
  wire [31:0] col = ppx * csw + kx * dw;
  assign x_addr = x_base + (row * in_w + col) * cin + {cg, 4'b0};
endmodule
