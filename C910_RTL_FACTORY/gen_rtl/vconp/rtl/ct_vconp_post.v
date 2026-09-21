/* ct_vconp_post — 后处理 (纯组合, 两相复用)
 * POOL 相: max = int32 域比较(requant 后移) / avg = 先 rq(sat8) 再累加
 * FIN  相: rq → ReLU → BN(64b 乘) → sat8 → 128b 打包
 */
module ct_vconp_post(
  input  wire        phase_fin,           // 0=POOL 相, 1=FIN 相
  input  wire        pool_max,
  input  wire signed [31:0] pacc_arr [0:15],
  input  wire signed [31:0] pool_arr [0:15],
  input  wire        pool_init,
  input  wire        relu_en, bn_en,
  input  wire signed [31:0] rq_mult,
  input  wire [4:0]  rq_shift, bn_shift,
  input  wire signed [31:0] bn_s  [0:15],
  input  wire signed [31:0] bn_b  [0:15],
  output wire signed [31:0] pool_next [0:15],   // POOL 相输出 → pool_acc D 端
  output wire [127:0] y_pack                    // FIN 相输出
);
  function signed [31:0] rq(input signed [31:0] v);
    reg signed [63:0] p;
    begin p = v * rq_mult; rq = p >>> rq_shift; end
  endfunction
  function signed [7:0] sat8(input signed [31:0] v);
    begin
      if (v > 127)  sat8 = 8'sd127;
      else if (v < -128) sat8 = -8'sd128;
      else sat8 = v[7:0];
    end
  endfunction
  function signed [31:0] sext8(input signed [7:0] v);
    begin sext8 = {{24{v[7]}}, v}; end
  endfunction

  genvar j;
  generate for (j=0; j<16; j=j+1) begin: PO
    assign pool_next[j] = pool_max
        ? ((!pool_init) ? pacc_arr[j] : ((pacc_arr[j] > pool_arr[j]) ? pacc_arr[j] : pool_arr[j]))
        : ((pool_init ? pool_arr[j] : 0) + sext8(sat8(rq(pacc_arr[j]))));
  end endgenerate

  // FIN 相
  wire signed [31:0] fv [0:15];
  genvar q;
  generate for (q=0; q<16; q=q+1) begin: FINV
    assign fv[q] = relu_en ? ((rq(pool_arr[q]) < 0) ? 0 : rq(pool_arr[q])) : rq(pool_arr[q]);
  end endgenerate

  wire signed [31:0] fvn [0:15];
  generate for (q=0; q<16; q=q+1) begin: FINBN
    wire signed [63:0] bt;                   // BN 中间积 64-bit (问题①)
    assign bt  = fv[q] * bn_s[q];
    assign fvn[q] = bn_en ? ((bt >>> bn_shift) + bn_b[q]) : fv[q];
  end endgenerate

  genvar r;
  generate for (r=0; r<16; r=r+1) begin: PACK
    assign y_pack[r*8 +: 8] = sat8(fvn[r]);
  end endgenerate
endmodule
