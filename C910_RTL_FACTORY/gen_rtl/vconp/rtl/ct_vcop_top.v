/*Copyright 2026. Licensed under Apache-2.0 (demo extension).
 *
 * ct_vcop_top — 统一融合协处理器内核 (vconp 与 vgemm 的共同实现)
 *
 * 统一性: vgemm ≡ vconp 的退化模式
 *   vgemm(行m×16列) = vconp 配置 {kh=kw=1, ph=pw=1, Cin=K}  ⇒ Conv1×1 无池化
 *   x 寻址: row=col=0 → xoff = cg·16 (行内 k 偏移, 与 vgemm 语义一致)
 *   节拍: PP=KK=1 → S_RUN 仅 CC 拍; S_POOL 单次直通 (max/1 = 自身)
 * 顶层: ct_vconp_top / ct_vgemm_top 是本内核的 CSR 打包薄封装
 */
module ct_vcop_top(
  input  wire         clk, rst_n, start,
  output reg          busy, done,
  input  wire [31:0]  x_base, w_base, y_base,
  input  wire [31:0]  csr0, csr1, csr2, csr3,
  input  wire [31:0]  bias [0:15],
  input  wire [31:0]  bn_s [0:15],
  input  wire [31:0]  bn_b [0:15],
  output wire         x_re,
  output wire  [31:0] x_addr,
  input  wire  [127:0] x_rdata,
  output wire         w_re,
  output wire  [31:0] w_addr,
  input  wire  [127:0] w_rdata,
  output reg          y_we,
  output wire  [31:0] y_addr,
  output reg   [127:0] y_wdata
);
  // ① CSR 解包
  wire [7:0]  cin;  wire [3:0] kh, kw, csh, csw, dh, dw, ph, pw;
  wire [9:0]  in_w; wire pool_max, relu_en, bn_en;
  wire signed [31:0] rq_mult; wire [4:0] rq_shift, bn_shift;
  ct_vcop_csr u_csr(.csr0(csr0), .csr1(csr1), .csr2(csr2), .csr3(csr3),
                     .cin(cin), .kh(kh), .kw(kw), .csh(csh), .csw(csw),
                     .dh(dh), .dw(dw), .ph(ph), .pw(pw), .in_w(in_w),
                     .pool_max(pool_max), .relu_en(relu_en), .bn_en(bn_en),
                     .rq_mult(rq_mult), .rq_shift(rq_shift), .bn_shift(bn_shift));
  wire [7:0] KK = kh * kw;
  wire [7:0] PP = ph * pw;
  wire [7:0] CC = (cin + 15) >> 4;
  wire [15:0] wwords = 16 * KK * CC;

  // ② AGU
  localparam S_IDLE=3'd0, S_WLOAD=3'd1, S_RUN=3'd2, S_POOL=3'd3, S_FIN=3'd4;
  reg [2:0] state;
  reg [7:0] pp, kk, cg;
  ct_vcop_agu u_agu(.pp(pp), .kk(kk), .cg(cg),
                    .pw(pw), .kw(kw), .csh(csh), .csw(csw), .dh(dh), .dw(dw),
                    .cin(cin), .in_w(in_w), .x_base(x_base), .x_addr(x_addr));
  assign x_re = (state == S_RUN);

  // ③ COMPUTE
  reg [15:0] wld_cnt;
  wire wld_we = (state == S_WLOAD) && (wld_cnt < wwords);
  wire signed [31:0] mac_dot [0:15];
  ct_vcop_compute u_cmp(.clk(clk), .x_rdata(x_rdata),
                        .kk(kk), .cg(cg), .KK(KK), .CC(CC),
                        .wld_we(wld_we), .wld_cnt(wld_cnt), .w_rdata(w_rdata),
                        .mac_dot(mac_dot));
  assign w_re   = wld_we;
  assign w_addr = w_base + wld_cnt * 16;

  // ④ POST
  wire signed [31:0] pool_next [0:15];
  wire [127:0] y_pack;
  wire signed [31:0] bn_sx [0:15];
  wire signed [31:0] bn_bx [0:15];
  wire signed [31:0] biasx [0:15];
  genvar sk;
  generate for (sk=0; sk<16; sk=sk+1) begin: SGN
    assign bn_sx[sk] = bn_s[sk]; assign bn_bx[sk] = bn_b[sk]; assign biasx[sk] = bias[sk];
  end endgenerate
  ct_vcop_post u_post(.phase_fin(state == S_FIN), .pool_max(pool_max),
                      .pacc_arr(pacc), .pool_arr(pool_acc), .pool_init(pool_init),
                      .relu_en(relu_en), .bn_en(bn_en),
                      .rq_mult(rq_mult), .rq_shift(rq_shift), .bn_shift(bn_shift),
                      .bn_s(bn_sx), .bn_b(bn_bx),
                      .pool_next(pool_next), .y_pack(y_pack));

  // ⑤ FSM + PACC
  reg signed [31:0] pacc [0:15];
  reg signed [31:0] pool_acc [0:15];
  reg pool_init;
  assign y_addr = y_base;

  integer i;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE; busy <= 1'b0; done <= 1'b0; y_we <= 1'b0; pool_init <= 1'b0;
    end else begin
      done <= 1'b0; y_we <= 1'b0;
      case (state)
      S_IDLE: if (start) begin
        busy <= 1'b1;
        for (i=0;i<16;i=i+1) pacc[i] <= biasx[i];
        pp <= 0; kk <= 0; cg <= 0; wld_cnt <= 0; pool_init <= 1'b0;
        state <= S_WLOAD;
      end
      S_WLOAD: begin
        if (wld_cnt < wwords) wld_cnt <= wld_cnt + 1;
        else state <= S_RUN;
      end
      S_RUN: begin
        for (i=0;i<16;i=i+1) pacc[i] <= pacc[i] + mac_dot[i];
        if (cg < CC-1) cg <= cg + 1;
        else begin
          cg <= 0;
          if (kk < KK-1) kk <= kk + 1;
          else begin kk <= 0; state <= S_POOL; end   // 账齐门控
        end
      end
      S_POOL: begin
        for (i=0;i<16;i=i+1) pool_acc[i] <= pool_next[i];
        pool_init <= 1'b1;
        for (i=0;i<16;i=i+1) pacc[i] <= biasx[i];
        if (pp < PP-1) begin pp <= pp + 1; state <= S_RUN; end
        else state <= S_FIN;
      end
      S_FIN: begin
        y_wdata <= y_pack;
        y_we <= 1'b1;
        busy <= 1'b0; done <= 1'b1;
        state <= S_IDLE;
      end
      default: state <= S_IDLE;
      endcase
    end
  end
endmodule
