/* vconp_ref.c — ct_vconp_top.v 累加/重定量数据类型的 C 镜像 (供人工审查)
 * 每个变量注释标注 RTL 来源行. 编译: gcc -o vconp_ref vconp_ref.c && ./vconp_ref
 */
#include <stdio.h>
#include <stdint.h>

/* ============ RTL 数据类型对照表 ============
 * RTL 声明                              | C 等价              | 说明
 * -------------------------------------+--------------------+---------------------------
 * reg signed [7:0]  wspad[]   (L53)     | int8_t             | 权重 scratchpad
 * wire signed [7:0] xv[]      (L66-69)  | int8_t             | 输入 lane
 * wire signed [15:0] prod[][] (L75)     | int16_t            | 单乘积: int8×int8 恰好 ±16256, 16-bit 够
 * wire signed [31:0] mac_dot[] (L76)    | int32_t            | 16 积之和: |max| = 16×16256 = 260096 < 2^31 ✓
 * reg signed [31:0] pacc[]    (L48)     | int32_t            | conv 累加器 (PACC 槽)
 * reg signed [31:0] pool_acc[](L49)     | int32_t            | 池化累加: max 存原始 pacc / avg 存 sat8 之和
 * rq(): reg signed [63:0] p   (L94)     | int64_t            | 32×32 乘积必须 64-bit
 * sat8(): [7:0]               (L97)     | int8_t (饱和)      | 输出域
 * sext8(): [31:0]             (L105)    | int32_t (符号扩展) | avg 累加恢复符号
 */

/* ---- 溢出界审查 ----
 * pacc 上界: bias + PP..(不含) 每拍 mac_dot ≤ 260096, 拍数 = KK*CC
 *   KK*CC ≤ 64*16 = 1024 拍 → pacc ≤ 260096×1024 + bias ≈ 2.66×10^8 < 2^31-1 ≈ 2.15×10^9  ✓ 余量 ~8×
 * pool_acc(avg): ≤ 4 × 128 = 512, int32 绰绰有余 ✓
 * rq: pacc(≤2^28) × rq_mult(≤2^31) → 必须 int64 (RTL 63:0 一致) ✓
 * BN: wb_v(≤127) × bn_s(≤2^31) → 也应 int64! RTL L177: wb_v * bn_s[i] 是 32×32 → Verilog
 *   在 32-bit 上下文里截断, 这是 RTL 潜在问题点① (见 main 末尾审查结论)
 */

static int32_t rq(int32_t v, int32_t mult, uint8_t shift) {
    int64_t p = (int64_t)v * mult;          /* RTL: p = v * rq_mult; p signed[63:0] */
    return (int32_t)(p >> shift);            /* RTL: p >>> rq_shift, 截回 32-bit */
}

static int8_t sat8(int32_t v) {
    if (v > 127)  return 127;
    if (v < -128) return -128;
    return (int8_t)v;                        /* RTL: v[7:0] */
}

/* 单拍点积 (RTL mac_dot, 加法树版) */
static int32_t dot16(const int8_t x[16], const int8_t w[16]) {
    int32_t s = 0;                           /* mac_dot: signed[31:0] */
    int16_t prod[16];                        /* prod: signed[15:0] */
    for (int l = 0; l < 16; l++) {
        prod[l] = (int16_t)(x[l] * w[l]);    /* int8×int8 → int16 精确 */
        s += prod[l];                        /* 16 项累加入 int32 */
    }
    return s;
}

/* 完整一条 vconp (单 pool 输出 ×16 oc) — 与 FSM 逐步对应 */
void vconp(const int8_t *X, const int8_t *W, const int32_t bias[16],
           int cin, int kh, int kw, int csh, int csw, int dh, int dw,
           int ph, int pw, int in_w, int pool_max, int relu_en, int bn_en,
           int32_t rq_mult, int rq_shift,
           const int32_t bn_s[16], int bn_shift, const int32_t bn_b[16],
           int8_t Y[16])
{
    int cc  = (cin + 15) / 16;               /* RTL: CC = (cin+15)>>4 */
    int32_t pacc[16], pool_acc[16];          /* 均为 signed[31:0] */
    int pool_init = 0;                       /* RTL: pool_init */

    for (int oc = 0; oc < 16; oc++) pacc[oc] = bias[oc];   /* S_IDLE */

    for (int pp = 0; pp < ph*pw; pp++) {     /* S_POOL 外层 */
        for (int k = 0; k < kh*kw; k++) {    /* S_RUN: kk 节拍 */
            int ky = k / kw, kx = k % kw;
            for (int cg = 0; cg < cc; cg++) {/* S_RUN: cg 节拍 */
                int row = (pp/pw)*csh + ky*dh;
                int col = (pp%pw)*csw + kx*dw;
                int8_t xv[16];
                for (int l = 0; l < 16; l++)
                    xv[l] = X[(row*in_w + col)*cin + cg*16 + l];  /* 注: 尾组 l>=cin%16 的 x 是垃圾,
                                                                      但权重为 0 (esp-ppq ABI 补零) */
                for (int oc = 0; oc < 16; oc++) {
                    int8_t wv[16];
                    int base = ((oc*kh*kw + k)*cc + cg)*16;        /* 权重 scratchpad 索引 */
                    for (int l = 0; l < 16; l++) wv[l] = W[base + l];
                    pacc[oc] += dot16(xv, wv);                     /* S_RUN: pacc += mac_dot */
#ifdef DBG
                    if (oc==0) printf("pp=%d k=%d cg=%d qacc0=%d\n", pp,k,cg,pacc[0]);
#endif
                }
            }
        }
        /* S_POOL: 池化 (注意 max/avg 数据域分叉) */
        for (int oc = 0; oc < 16; oc++) {
            if (pool_max) {                                        /* max: int32 原始域比较 */
                if (!pool_init || pacc[oc] > pool_acc[oc])         /* RTL: !pool_init 分支防止初值 X */
                    pool_acc[oc] = pacc[oc];
            } else {                                               /* avg: 先 requant 到 int8 再累加 */
                int8_t q = sat8(rq(pacc[oc], rq_mult, rq_shift));
                pool_acc[oc] = (pool_init ? pool_acc[oc] : 0) + (int32_t)q;  /* RTL: sext8 符号扩展 */
            }
        }
        pool_init = 1;
        for (int oc = 0; oc < 16; oc++) pacc[oc] = bias[oc];       /* PACC 重开账本 */
    }

    #ifdef DBG
    for(int oc=0;oc<2;oc++) printf("pool_acc[%d]=%d\n", oc, pool_acc[oc]);
#endif
    /* S_FIN: requant → relu → BN → 饱和打包 */
    for (int oc = 0; oc < 16; oc++) {
        int32_t v = rq(pool_acc[oc], rq_mult, rq_shift);           /* max 路径: 胜者才 requant (省 3 次/点) */
        if (relu_en && v < 0) v = 0;
        if (bn_en)  v = (int32_t)(((int64_t)v * bn_s[oc]) >> bn_shift) + bn_b[oc];  /* ★见审查① */
        Y[oc] = sat8(v);
    }
}

#ifdef MAIN
int main(void) {
    /* 冒烟: 与黄金模型一致的语义自检 (值可手工核算) */
    int8_t X[4096] = {0}, W[65536] = {0}, Y[16];
    int32_t bias[16] = {0}, bn_s[16], bn_b[16] = {0};
    for (int i = 0; i < 16; i++) bn_s[i] = 1;
    X[0] = 1; W[0] = 2;                     /* oc0,k0,c0: 1×2 */
    X[9] = 3;  W[17] = 4;                     /* k=1 (kx=1 → col=1): NHWC 偏移 (0*12+1)*8+1=9 */
    vconp(X, W, bias, 8, 3,3, 2,2, 1,1, 2,2, 12, /*conv3x3 s2 pool2x2*/
          1, 0, 0, 1000, 10, bn_s, 0, bn_b, Y);
    /* qacc0 = 2 + 12 = 14; max(4 个 pp 中仅 pp=00 非零) = 14; rq(14)=14*1000>>10=13 */
    printf("Y[0]=%d (期望 13)\n", Y[0]);
    return Y[0] == 13 ? 0 : 1;
}
#endif

/* ============ 审查结论 ============
 * ① RTL L177 (S_FIN): `wb_v * bn_s[i]` 为 32-bit 上下文乘法, 127×bn_s 在 bn_s>2^24 时溢出;
 *    本 C 镜像已改为 int64 中间量. RTL 应改: 借用 rq 的 64-bit 乘路径或限制 bn_s 范围 (软件契约).
 * ② pacc 32-bit 在最坏几何 (KK*CC=1024) 余量 8×, 安全; 若未来支持 Cin>256 需换 40-bit (XACC 式).
 * ③ prod 用 int16 精确 (±16256); mac_dot/pacc/pool_acc 必须 int32; rq/BN 乘积必须 int64.
 */
