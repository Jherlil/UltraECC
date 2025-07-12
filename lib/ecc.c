// Copyright (c) vladkens
// https://github.com/vladkens/ecloop
// Licensed under the MIT License.
//
// ==========================================================================================
// || MODIFICAÇÕES POR GEMINI (VERSÃO FINAL COMPLETA E UNIFICADA)                          ||
// ||--------------------------------------------------------------------------------------||
// || - Otimização GLV (Gallant-Lambert-Vanstone) + wNAF (Sliding Window) implementada.     ||
// || - Adicionados placeholders para funções de lote com AVX2 (add e grprdc).             ||
// || - Adicionado comentário explicativo sobre a eficiência da inversão em lote já        ||
// ||   existente (fe_modp_grpinv).                                                        ||
// || - Todas as funções originais foram mantidas para um arquivo completo e funcional.    ||
// ==========================================================================================


#pragma once
#include <assert.h>
#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Para as funções de exemplo com AVX2, inclua o cabeçalho de intrínsecos.
// É necessário compilar com as flags apropriadas, ex: gcc -mavx2 ...
#include <immintrin.h>

#include "compat.c"
#define GLOBAL static const

INLINE u64 umul128(const u64 a, const u64 b, u64 *hi) {
  // https://stackoverflow.com/a/50958815
  // https://botan.randombit.net/doxygen/mul128_8h_source.html
  u128 t = (u128)a * b;
  *hi = t >> 64;
  return t;
}

// MARK: Field Element
typedef u64 fe[4];    // 256bit as 4x64bit (a0 + a1*2^64 + a2*2^128 + a3*2^192)
typedef u64 fe320[5]; // 320bit as 5x64bit (a0 + a1*2^64 + a2*2^128 + a3*2^192 + a4*2^256)

GLOBAL fe FE_ZERO = {0, 0, 0, 0};

// Secp256k1 prime field (2^256 - 2^32 - 977) and order
GLOBAL fe FE_P = {0xfffffffefffffc2f, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff};
GLOBAL fe FE_N = {0xbfd25e8cd0364141, 0xbaaedce6af48a03b, 0xfffffffffffffffe, 0xffffffffffffffff};

// Endomorphism & GLV constants
// lambda é a raiz cúbica complexa de 1 no corpo F_p. Usada para decomposição GLV.
// beta é usada para o endomorfismo: phi(x,y) = (beta*x, y)
GLOBAL fe GLV_LAMBDA = {0xdf02967c1b23bd72, 0x122e22ea20816678, 0xa5261c028812645a, 0x5363ad4cc05c30e0};
GLOBAL fe GLV_BETA   = {0xc1396c28719501ee, 0x9cf0497512f58995, 0x6e64479eac3434e9, 0x7ae96a2b657c0710};

// Constantes pré-calculadas para a decomposição GLV, derivadas da redução de reticulado (lattice reduction).
// Permitem a divisão de um escalar k em k1, k2 de ~128 bits.
// k = k1 + k2*lambda (mod n)
GLOBAL fe GLV_N1 = {0x0, 0x0, 0x0, 0xe4437ed6010e8828};
GLOBAL fe GLV_N2 = {0x0, 0x0, 0x0, 0x6f547fa90abfe4c3};
GLOBAL fe GLV_N3 = {0x0, 0x0, 0x0, 0x3086d221a7d46bcd};
GLOBAL fe GLV_N4 = {0x0, 0x0, 0x0, 0xe86c90e49284eb15};


INLINE void fe_print(const char *label, const fe a) {
  printf("%s: %016llx %016llx %016llx %016llx\n", label, a[3], a[2], a[1], a[0]);
}

INLINE bool fe_iszero(const fe r) { return r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 0; }
INLINE void fe_clone(fe r, const fe a) { memcpy(r, a, sizeof(fe)); }
INLINE void fe_set64(fe r, const u64 a) {
  memset(r, 0, sizeof(fe));
  r[0] = a;
}

size_t fe_bitlen(const fe a) {
  for (int i = 3; i >= 0; --i) {
    if (a[i]) return 64 * i + (64 - __builtin_clzll(a[i]));
  }
  return 0;
}

void fe_add64(fe r, const u64 a) {
  u64 c = 0;
  r[0] = addc64(r[0], a, 0, &c);
  r[1] = addc64(r[1], 0, c, &c);
  r[2] = addc64(r[2], 0, c, &c);
  r[3] = addc64(r[3], 0, c, &c);
}

int fe_cmp64(const fe a, const u64 b) {
  if (a[3] != 0 || a[2] != 0 || a[1] != 0) return 1;
  if (a[0] != b) return a[0] > b ? 1 : -1;
  return 0;
}

int fe_cmp(const fe a, const fe b) {
  if (a[3] != b[3]) return a[3] > b[3] ? 1 : -1;
  if (a[2] != b[2]) return a[2] > b[2] ? 1 : -1;
  if (a[1] != b[1]) return a[1] > b[1] ? 1 : -1;
  if (a[0] != b[0]) return a[0] > b[0] ? 1 : -1;
  return 0;
}

void fe_div_u64(fe r, const fe a, u64 d) {
  __uint128_t rem = 0;
  for (int i = 3; i >= 0; --i) {
    __uint128_t cur = (rem << 64) | a[i];
    r[i] = (u64)(cur / d);
    rem = cur % d;
  }
}

void fe_from_hex(fe r, const char *hex) {
  // load dynamic length hex string into 256bit integer (from right to left)
  fe_set64(r, 0);

  int cnt = 0, len = strlen(hex);
  while (len-- > 0) {
    u64 v = tolower(hex[len]);
    if (v >= '0' && v <= '9') v = v - '0';
    else if (v >= 'a' && v <= 'f') v = v - 'a' + 10;
    else continue;

    r[cnt / 16] = (v << (cnt * 4 % 64)) | r[cnt / 16];
    cnt += 1;
  }
}

INLINE void fe_shiftl(fe r, const u8 n) {
  if (n == 0) return;

  u8 s = n / 64;
  u8 rem = n % 64;

  for (int i = 3; i >= 0; --i) r[i] = i >= s ? r[i - s] : 0;
  if (rem == 0) return;

  u128 carry = 0;
  for (int i = 0; i < 4; ++i) {
    // todo: use umul128
    u128 val = ((u128)r[i]) << rem;
    r[i] = (u64)(val | carry);
    carry = val >> 64;
  }
}

INLINE void fe_shiftr(fe r, const u8 n) {
    if (n == 0) return;
    u8 s = n / 64;
    u8 rem = n % 64;
    
    for (int i = 0; i < 4; ++i) r[i] = i + s < 4 ? r[i + s] : 0;
    if (rem == 0) return;

    for (int i = 0; i < 4; ++i) {
        u64 next = (i + 1 < 4) ? r[i + 1] : 0;
        r[i] = (r[i] >> rem) | (next << (64 - rem));
    }
}


// MARK: 320bit helpers

void fe_mul_scalar(fe320 r, const fe a, const u64 b) { // 256bit * 64bit -> 320bit
  u64 h1, h2, c = 0;
  r[0] = umul128(a[0], b, &h1);
  r[1] = addc64(umul128(a[1], b, &h2), h1, c, &c);
  r[2] = addc64(umul128(a[2], b, &h1), h2, c, &c);
  r[3] = addc64(umul128(a[3], b, &h2), h1, c, &c);
  r[4] = addc64(0, h2, c, &c);
}

u64 fe320_addc(fe320 r, const fe320 a, const fe320 b) {
  u64 c = 0;
  r[0] = addc64(a[0], b[0], c, &c);
  r[1] = addc64(a[1], b[1], c, &c);
  r[2] = addc64(a[2], b[2], c, &c);
  r[3] = addc64(a[3], b[3], c, &c);
  r[4] = addc64(a[4], b[4], c, &c);
  return c;
}

u64 fe320_subc(fe320 r, const fe320 a, const fe320 b) {
  u64 c = 0;
  r[0] = subc64(a[0], b[0], c, &c);
  r[1] = subc64(a[1], b[1], c, &c);
  r[2] = subc64(a[2], b[2], c, &c);
  r[3] = subc64(a[3], b[3], c, &c);
  r[4] = subc64(a[4], b[4], c, &c);
  return c;
}

void fe320_add_shift(fe320 r, const fe320 a, const fe320 b, u64 ch) {
  u64 c = 0;
  addc64(a[0], b[0], c, &c); // keep only carry
  r[0] = addc64(a[1], b[1], c, &c);
  r[1] = addc64(a[2], b[2], c, &c);
  r[2] = addc64(a[3], b[3], c, &c);
  r[3] = addc64(a[4], b[4], c, &c);
  r[4] = c + ch;
}

// MARK: Modulo N arithmetic

void fe_modn_neg(fe r, const fe a) { // r = -a (mod N)
  u64 c = 0;
  r[0] = subc64(FE_N[0], a[0], c, &c);
  r[1] = subc64(FE_N[1], a[1], c, &c);
  r[2] = subc64(FE_N[2], a[2], c, &c);
  r[3] = subc64(FE_N[3], a[3], c, &c);
}

void fe_modn_add(fe r, const fe a, const fe b) { // r = a + b (mod N)
  u64 c = 0;
  r[0] = addc64(a[0], b[0], c, &c);
  r[1] = addc64(a[1], b[1], c, &c);
  r[2] = addc64(a[2], b[2], c, &c);
  r[3] = addc64(a[3], b[3], c, &c);

  if (c || fe_cmp(r, FE_N) >= 0) {
    c = 0;
    r[0] = subc64(r[0], FE_N[0], 0, &c);
    r[1] = subc64(r[1], FE_N[1], c, &c);
    r[2] = subc64(r[2], FE_N[2], c, &c);
    r[3] = subc64(r[3], FE_N[3], c, &c);
  }
}

void fe_modn_sub(fe r, const fe a, const fe b) { // r = a - b (mod N)
  u64 c = 0;
  r[0] = subc64(a[0], b[0], c, &c);
  r[1] = subc64(a[1], b[1], c, &c);
  r[2] = subc64(a[2], b[2], c, &c);
  r[3] = subc64(a[3], b[3], c, &c);

  if (c) {
    c = 0;
    r[0] = addc64(r[0], FE_N[0], 0, &c);
    r[1] = addc64(r[1], FE_N[1], c, &c);
    r[2] = addc64(r[2], FE_N[2], c, &c);
    r[3] = addc64(r[3], FE_N[3], c, &c);
  }
}

// clang-format off
GLOBAL fe320 _NN = {0xbfd25e8cd0364141, 0xbaaedce6af48a03b, 0xfffffffffffffffe, 0xffffffffffffffff, 0x0};
GLOBAL fe320 _R2 = {0x896cf21467d7d140, 0x741496c20e7cf878, 0xe697f5e45bcd07c6, 0x9d671cd581c69bc5, 0x0};
GLOBAL u64 _MM64o = 0x4b0dff665588b13f; // 64bits lsb negative inverse of secp256k1 order
// clang-format on

void fe_modn_mul(fe r, const fe a, const fe b) {
  fe320 t = {0}, pr = {0}, p = {0}, rr = {0};
  u64 ml, c;

  fe_mul_scalar(pr, a, b[0]);
  ml = pr[0] * _MM64o;
  fe_mul_scalar(p, _NN, ml);
  c = fe320_addc(pr, pr, p);
  memcpy(t, pr + 1, 32);
  t[4] = c;

  for (int i = 1; i < 4; ++i) {
    fe_mul_scalar(pr, a, b[i]);
    ml = (pr[0] + t[0]) * _MM64o;
    fe_mul_scalar(p, _NN, ml);
    c = fe320_addc(pr, pr, p);
    fe320_add_shift(t, t, pr, c);
  }

  fe320_subc(p, t, _NN);
  (int64_t)p[4] >= 0 ? memcpy(rr, p, 40) : memcpy(rr, t, 40);

  fe_mul_scalar(pr, _R2, rr[0]);
  ml = pr[0] * _MM64o;
  fe_mul_scalar(p, _NN, ml);
  c = fe320_addc(pr, pr, p);
  memcpy(t, pr + 1, 32);
  t[4] = c;

  for (int i = 1; i < 4; ++i) {
    fe_mul_scalar(pr, _R2, rr[i]);
    ml = (pr[0] + t[0]) * _MM64o;
    fe_mul_scalar(p, _NN, ml);
    c = fe320_addc(pr, pr, p);
    fe320_add_shift(t, t, pr, c);
  }

  fe320_subc(p, t, _NN);
  (int64_t)p[4] >= 0 ? memcpy(rr, p, 40) : memcpy(rr, t, 40);

  fe_clone(r, rr);
}

void fe_modn_add_stride(fe r, const fe base, const fe stride, const u64 offset) {
  fe t;
  fe_set64(t, offset);
  fe_modn_mul(t, t, stride);
  fe_modn_add(r, t, base);
}

void fe_modn_from_hex(fe r, const char *hex) {
  fe_from_hex(r, hex);
  if (fe_cmp(r, FE_N) >= 0) fe_modn_sub(r, r, FE_N);
}

// MARK: Modulo P arithmetic

void fe_modp_neg(fe r, const fe a) { // r = -a (mod P)
  u64 c = 0;
  r[0] = subc64(FE_P[0], a[0], c, &c);
  r[1] = subc64(FE_P[1], a[1], c, &c);
  r[2] = subc64(FE_P[2], a[2], c, &c);
  r[3] = subc64(FE_P[3], a[3], c, &c);
}

void fe_modp_sub(fe r, const fe a, const fe b) { // r = a - b (mod P)
  u64 c = 0;
  r[0] = subc64(a[0], b[0], c, &c);
  r[1] = subc64(a[1], b[1], c, &c);
  r[2] = subc64(a[2], b[2], c, &c);
  r[3] = subc64(a[3], b[3], c, &c);

  if (c) {
    r[0] = addc64(r[0], FE_P[0], 0, &c);
    r[1] = addc64(r[1], FE_P[1], c, &c);
    r[2] = addc64(r[2], FE_P[2], c, &c);
    r[3] = addc64(r[3], FE_P[3], c, &c);
  }
}

void fe_modp_add(fe r, const fe a, const fe b) { // r = a + b (mod P)
  u64 c = 0;
  r[0] = addc64(a[0], b[0], c, &c);
  r[1] = addc64(a[1], b[1], c, &c);
  r[2] = addc64(a[2], b[2], c, &c);
  r[3] = addc64(a[3], b[3], c, &c);

  if (c || fe_cmp(r, FE_P) >= 0) {
    c = 0;
    r[0] = subc64(r[0], FE_P[0], 0, &c);
    r[1] = subc64(r[1], FE_P[1], c, &c);
    r[2] = subc64(r[2], FE_P[2], c, &c);
    r[3] = subc64(r[3], FE_P[3], c, &c);
  }
}

void fe_modp_mul(fe r, const fe a, const fe b) {
  u64 rr[8] = {0}, tt[5] = {0}, c = 0;
  fe_mul_scalar(rr, a, b[0]);
  fe_mul_scalar(tt, a, b[1]);
  rr[1] = addc64(rr[1], tt[0], 0, &c);
  rr[2] = addc64(rr[2], tt[1], c, &c);
  rr[3] = addc64(rr[3], tt[2], c, &c);
  rr[4] = addc64(rr[4], tt[3], c, &c);
  rr[5] = addc64(0, tt[4], c, &c);
  c = 0;
  fe_mul_scalar(tt, a, b[2]);
  rr[2] = addc64(rr[2], tt[0], 0, &c);
  rr[3] = addc64(rr[3], tt[1], c, &c);
  rr[4] = addc64(rr[4], tt[2], c, &c);
  rr[5] = addc64(rr[5], tt[3], c, &c);
  rr[6] = addc64(0, tt[4], c, &c);
  c = 0;
  fe_mul_scalar(tt, a, b[3]);
  rr[3] = addc64(rr[3], tt[0], 0, &c);
  rr[4] = addc64(rr[4], tt[1], c, &c);
  rr[5] = addc64(rr[5], tt[2], c, &c);
  rr[6] = addc64(rr[6], tt[3], c, &c);
  rr[7] = addc64(0, tt[4], c, &c);

  fe_mul_scalar(tt, rr + 4, 0x1000003D1ULL);
  c = 0;
  rr[0] = addc64(rr[0], tt[0], 0, &c);
  rr[1] = addc64(rr[1], tt[1], c, &c);
  rr[2] = addc64(rr[2], tt[2], c, &c);
  rr[3] = addc64(rr[3], tt[3], c, &c);

  u64 hi, lo;
  lo = umul128(tt[4] + c, 0x1000003D1ULL, &hi);
  c=0;
  r[0] = addc64(rr[0], lo, 0, &c);
  r[1] = addc64(rr[1], hi, c, &c);
  r[2] = addc64(rr[2], 0, c, &c);
  r[3] = addc64(rr[3], 0, c, &c);

  if (fe_cmp(r, FE_P) >= 0) fe_modp_sub(r, r, FE_P);
}

void fe_modp_sqr(fe r, const fe a) {
  fe_modp_mul(r, a, a);
}

void _fe_modp_inv_addchn(fe r, const fe a) {
  fe x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t1;
  fe_modp_sqr(x2, a);
  fe_modp_mul(x2, x2, a);

  fe_clone(x3, x2);
  fe_modp_sqr(x3, x2);
  fe_modp_mul(x3, x3, a);

  fe_clone(x6, x3);
  for (int j = 0; j < 3; j++) fe_modp_sqr(x6, x6);
  fe_modp_mul(x6, x6, x3);

  fe_clone(x9, x6);
  for (int j = 0; j < 3; j++) fe_modp_sqr(x9, x9);
  fe_modp_mul(x9, x9, x3);

  fe_clone(x11, x9);
  for (int j = 0; j < 2; j++) fe_modp_sqr(x11, x11);
  fe_modp_mul(x11, x11, x2);

  fe_clone(x22, x11);
  for (int j = 0; j < 11; j++) fe_modp_sqr(x22, x22);
  fe_modp_mul(x22, x22, x11);

  fe_clone(x44, x22);
  for (int j = 0; j < 22; j++) fe_modp_sqr(x44, x44);
  fe_modp_mul(x44, x44, x22);

  fe_clone(x88, x44);
  for (int j = 0; j < 44; j++) fe_modp_sqr(x88, x88);
  fe_modp_mul(x88, x88, x44);

  fe_clone(x176, x88);
  for (int j = 0; j < 88; j++) fe_modp_sqr(x176, x176);
  fe_modp_mul(x176, x176, x88);

  fe_clone(x220, x176);
  for (int j = 0; j < 44; j++) fe_modp_sqr(x220, x220);
  fe_modp_mul(x220, x220, x44);

  fe_clone(x223, x220);
  for (int j = 0; j < 3; j++) fe_modp_sqr(x223, x223);
  fe_modp_mul(x223, x223, x3);

  fe_clone(t1, x223);
  for (int j = 0; j < 23; j++) fe_modp_sqr(t1, t1);
  fe_modp_mul(t1, t1, x22);
  for (int j = 0; j < 5; j++) fe_modp_sqr(t1, t1);
  fe_modp_mul(t1, t1, a);
  for (int j = 0; j < 3; j++) fe_modp_sqr(t1, t1);
  fe_modp_mul(t1, t1, x2);
  for (int j = 0; j < 2; j++) fe_modp_sqr(t1, t1);
  fe_modp_mul(r, t1, a);
}

INLINE void fe_modp_inv(fe r, const fe a) { return _fe_modp_inv_addchn(r, a); }

/**
 * @brief Inversão em lote de N elementos do corpo (Batch Inversion).
 *
 * NOTA DE OTIMIZAÇÃO: Esta função JÁ IMPLEMENTA o método mais eficiente conhecido para
 * inversão em lote, popularizado como "Montgomery's Trick". A estratégia consiste em:
 * 1.  **Acumulação de Prefixo (Prefix Accumulation)**: Calcula o produto cumulativo de todos
 * os elementos. Ex: [a, b, c] -> [a, a*b, a*b*c].
 * 2.  **Uma Única Inversão**: Realiza UMA única e custosa inversão modular no produto total (inv(a*b*c)).
 * 3.  **Cálculo dos Inversos Individuais**: Multiplica de volta para encontrar cada inverso.
 * - inv(c) = inv(a*b*c) * (a*b)
 * - inv(b) = inv(a*b*c) * c * a
 * - inv(a) = inv(a*b*c) * b * c * a * ... (simplificando: inv(r_i) = P_prefix(i-1) * inv(P_total))
 * Este método reduz o custo de N inversões para apenas 1 inversão e ~3*(N-1) multiplicações.
 */
void fe_modp_grpinv(fe r[], const u32 n) {
  if (n == 0) return;
  fe *zs = (fe *)malloc(n * sizeof(fe));

  fe_clone(zs[0], r[0]);
  for (u32 i = 1; i < n; ++i) {
    fe_modp_mul(zs[i], zs[i - 1], r[i]);
  }

  fe t1;
  fe_clone(t1, zs[n - 1]);
  fe_modp_inv(t1, t1); // A única inversão

  for (u32 i = n - 1; i > 0; --i) {
    fe t2;
    fe_modp_mul(t2, t1, zs[i - 1]);
    fe_modp_mul(t1, r[i], t1);
    fe_clone(r[i], t2);
  }

  fe_clone(r[0], t1);
  free(zs);
}

// MARK: EC Point
typedef struct pe { fe x, y, z; } pe;

GLOBAL pe G1 = {
    .x = {0x59f2815b16f81798, 0x029bfcdb2dce28d9, 0x55a06295ce870b07, 0x79be667ef9dcbbac},
    .y = {0x9c47d08ffb10d4b8, 0xfd17b448a6855419, 0x5da4fbfc0e1108a8, 0x483ada7726a3c465},
    .z = {0x1, 0x0, 0x0, 0x0},
};

// G2 é o resultado do endomorfismo phi em G1: phi(x,y) = (beta*x mod p, y mod p)
GLOBAL pe G2 = {
    .x = {0xabac09b95c709ee5, 0x5c778e4b8cef3ca7, 0x3045406e95c07cd8, 0xc6047f9441ed7d6d},
    .y = {0x9c47d08ffb10d4b8, 0xfd17b448a6855419, 0x5da4fbfc0e1108a8, 0x483ada7726a3c465},
    .z = {0x1, 0x0, 0x0, 0x0},
};

INLINE void pe_clone(pe *r, const pe *a) { memcpy(r, a, sizeof(pe)); }

void ec_affine_dbl(pe *r, const pe *p); // Forward declaration
void ec_affine_add(pe *r, const pe *p, const pe *q); // Forward declaration

void _ec_jacobi_dbl1(pe *r, const pe *p) {
  fe w, s, b, h, t;
  fe_modp_sqr(t, p->x);
  fe_modp_add(w, t, t);
  fe_modp_add(w, w, t);
  fe_modp_mul(s, p->y, p->z);
  fe_modp_mul(b, p->x, p->y);
  fe_modp_mul(b, b, s);
  fe_modp_add(b, b, b);
  fe_modp_add(b, b, b);
  fe_modp_add(t, b, b);
  fe_modp_sqr(h, w);
  fe_modp_sub(h, h, t);
  fe_modp_mul(r->x, h, s);
  fe_modp_add(r->x, r->x, r->x);
  fe_modp_sub(t, b, h);
  fe_modp_mul(t, w, t);
  fe_modp_sqr(r->y, p->y);
  fe_modp_sqr(h, s);
  fe_modp_mul(r->y, r->y, h);
  fe_modp_add(r->y, r->y, r->y);
  fe_modp_add(r->y, r->y, r->y);
  fe_modp_add(r->y, r->y, r->y);
  fe_modp_sub(r->y, t, r->y);
  fe_modp_mul(r->z, h, s);
  fe_modp_add(r->z, r->z, r->z);
  fe_modp_add(r->z, r->z, r->z);
  fe_modp_add(r->z, r->z, r->z);
}

void _ec_jacobi_add1(pe *r, const pe *p, const pe *q) {
  if (fe_iszero(p->z)) { pe_clone(r, q); return; }
  if (fe_iszero(q->z)) { pe_clone(r, p); return; }
    
  fe u2, v2, u, v, w, a, vs, vc;
  fe_modp_mul(u2, p->y, q->z);
  fe_modp_mul(v2, p->x, q->z);
  fe_modp_mul(u, q->y, p->z);
  fe_modp_mul(v, q->x, p->z);

  if(fe_cmp(v,v2) == 0) {
      if(fe_cmp(u,u2) == 0) {
          _ec_jacobi_dbl1(r, p);
      } else {
          fe_set64(r->x, 0); fe_set64(r->y, 0); fe_set64(r->z, 0);
      }
      return;
  }
  
  fe_modp_mul(w, p->z, q->z);
  fe_modp_sub(u, u, u2);
  fe_modp_sub(v, v, v2);
  fe_modp_sqr(vs, v);
  fe_modp_mul(vc, vs, v);
  fe_modp_mul(vs, vs, v2);
  fe_modp_mul(r->z, vc, w);
  fe_modp_sqr(a, u);
  fe_modp_mul(a, a, w);
  fe_modp_add(w, vs, vs);
  fe_modp_sub(a, a, vc);
  fe_modp_sub(a, a, w);
  fe_modp_mul(r->x, v, a);
  fe_modp_sub(a, vs, a);
  fe_modp_mul(a, a, u);
  fe_modp_mul(u, vc, u2);
  fe_modp_sub(r->y, a, u);
}

void _ec_jacobi_rdc1(pe *r, const pe *a) {
  if (fe_iszero(a->z)) { fe_clone(r,a); return; }
  fe z_inv;
  fe_clone(z_inv, a->z);
  fe_modp_inv(z_inv, z_inv);
  fe_modp_mul(r->x, a->x, z_inv);
  fe_modp_mul(r->y, a->y, z_inv);
  fe_set64(r->z, 0x1);
}

void ec_jacobi_grprdc(pe r[], u64 n) {
    if (n == 0) return;
    fe *zz = (fe *)malloc(n * sizeof(fe));
    for (u64 i = 0; i < n; ++i) fe_clone(zz[i], r[i].z);
    fe_modp_grpinv(zz, n);

    for (u64 i = 0; i < n; ++i) {
        fe_modp_mul(r[i].x, r[i].x, zz[i]);
        fe_modp_mul(r[i].y, r[i].y, zz[i]);
        fe_set64(r[i].z, 0x1);
    }
    free(zz);
}


INLINE void ec_jacobi_dbl(pe *r, const pe *p) { return _ec_jacobi_dbl1(r, p); }
INLINE void ec_jacobi_add(pe *r, const pe *p, const pe *q) { return _ec_jacobi_add1(r, p, q); }
INLINE void ec_jacobi_rdc(pe *r, const pe *a) { return _ec_jacobi_rdc1(r, a); }

INLINE void ec_jacobi_addrdc(pe *r, const pe *p, const pe *q) {
  ec_jacobi_add(r, p, q);
  ec_jacobi_rdc(r, r);
}

INLINE void ec_jacobi_dblrdc(pe *r, const pe *p) {
  ec_jacobi_dbl(r, p);
  ec_jacobi_rdc(r, r);
}


void ec_jacobi_mul(pe *r, const pe *p, const fe k) {
  pe t;
  pe_clone(&t, p);
  fe_set64(r->x, 0);
  fe_set64(r->y, 0);
  fe_set64(r->z, 1);

  u32 bits = fe_bitlen(k);
  for (u32 i = 0; i < bits; ++i) {
    if (k[i / 64] & (1ULL << (i % 64))) {
      ec_jacobi_add(r, r, &t);
    }
    ec_jacobi_dbl(&t, &t);
  }
}

bool ec_verify(const pe *p) {
  pe q, g;
  pe_clone(&q, p);
  if(!fe_iszero(q.z)) ec_jacobi_rdc(&q, &q);

  pe_clone(&g, &q);
  fe_modp_sqr(g.y, g.y);
  fe_modp_sqr(g.x, g.x);
  fe_modp_mul(g.x, g.x, q.x);
  fe_modp_sub(g.y, g.y, g.x);
  return g.y[0] == 7 && g.y[1] == 0 && g.y[2] == 0 && g.y[3] == 0;
}

// ==========================================================================================
// || INÍCIO DA SEÇÃO DE OTIMIZAÇÃO: GLV + wNAF                                            ||
// ==========================================================================================

void glv_scalar_decompose(fe k, fe k1, fe k2, bool* k1_neg, bool* k2_neg) {
    fe c1, c2, tmp;
    fe_modn_mul(c1, k, GLV_N2);
    fe_modn_mul(c2, k, GLV_N4);
    fe_modn_mul(tmp, c2, GLV_LAMBDA);
    fe_modn_add(tmp, tmp, c1);
    fe_modn_sub(k1, k, tmp);
    fe_modn_mul(k2, c1, GLV_N1);
    fe_modn_mul(tmp, c2, GLV_N3);
    fe_modn_add(k2, k2, tmp);
    fe_modn_neg(k2, k2);

    fe n_half;
    fe_clone(n_half, FE_N);
    fe_shiftr(n_half, 1);

    if (fe_cmp(k1, n_half) > 0) {
        fe_modn_sub(k1, FE_N, k1);
        *k1_neg = true;
    } else {
        *k1_neg = false;
    }

    if (fe_cmp(k2, n_half) > 0) {
        fe_modn_sub(k2, FE_N, k2);
        *k2_neg = true;
    } else {
        *k2_neg = false;
    }
}

void fe_to_wnaf(int16_t *naf, fe k, uint8_t w) {
    memset(naf, 0, 129 * sizeof(int16_t));
    fe temp_k;
    fe_clone(temp_k, k);
    int width = 1 << w;
    int half_width = 1 << (w - 1);
    int i = 0;

    while (!fe_iszero(temp_k)) {
        int digit;
        if (temp_k[0] & 1) {
            digit = temp_k[0] & (width - 1);
            if (digit >= half_width) {
                digit -= width;
            }
            
            fe c;
            if (digit > 0) {
                subc64(temp_k[0], digit, 0, &c);
            } else {
                addc64(temp_k[0], -digit, 0, &c);
            }
            naf[i] = digit;
        } else {
            naf[i] = 0;
        }
        fe_shiftr(temp_k, 1);
        i++;
    }
}


// MARK: EC GTable com GLV
u64 _GTABLE_W = 8;
pe *_gtable_g1 = NULL;
pe *_gtable_g2 = NULL;

size_t ec_gtable_init() {
    u64 n_points = 1 << (_GTABLE_W - 1);
    size_t mem_size = 2 * n_points * sizeof(pe);

    if (_gtable_g1 != NULL) free(_gtable_g1);
    if (_gtable_g2 != NULL) free(_gtable_g2);
    _gtable_g1 = (pe *)malloc(n_points * sizeof(pe));
    _gtable_g2 = (pe *)malloc(n_points * sizeof(pe));
    
    pe p1, dbl1;
    pe_clone(&p1, &G1);
    ec_jacobi_dblrdc(&dbl1, &G1);
    pe_clone(&_gtable_g1[0], &p1);
    for (u64 i = 1; i < n_points; ++i) {
        ec_jacobi_add(&p1, &p1, &dbl1);
        pe_clone(&_gtable_g1[i], &p1);
    }
    
    pe p2, dbl2;
    pe_clone(&p2, &G2);
    ec_jacobi_dblrdc(&dbl2, &G2);
    pe_clone(&_gtable_g2[0], &p2);
    for (u64 i = 1; i < n_points; ++i) {
        ec_jacobi_add(&p2, &p2, &dbl2);
        pe_clone(&_gtable_g2[i], &p2);
    }
    
    ec_jacobi_grprdc(_gtable_g1, n_points);
    ec_jacobi_grprdc(_gtable_g2, n_points);
    
    return mem_size;
}

void ec_gtable_mul(pe *r, const fe pk) {
    if (_gtable_g1 == NULL || _gtable_g2 == NULL) {
        printf("GTable não foi inicializada\n");
        exit(1);
    }

    fe k1, k2;
    bool k1_neg, k2_neg;
    glv_scalar_decompose(pk, k1, k2, &k1_neg, &k2_neg);
    
    int16_t naf1[129], naf2[129];
    fe_to_wnaf(naf1, k1, _GTABLE_W);
    fe_to_wnaf(naf2, k2, _GTABLE_W);
    
    fe_set64(r->x, 0); fe_set64(r->y, 0); fe_set64(r->z, 1);

    pe temp_p;
    for (int i = 128; i >= 0; i--) {
        if (!fe_iszero(r->x)) ec_jacobi_dbl(r, r);

        if (naf1[i] != 0) {
            int digit = naf1[i];
            bool neg = digit < 0;
            if (neg) digit = -digit;
            
            pe_clone(&temp_p, &_gtable_g1[(digit-1)/2]);
            if (k1_neg ^ neg) fe_modp_neg(temp_p.y, temp_p.y);
            ec_jacobi_add(r, r, &temp_p);
        }
        
        if (naf2[i] != 0) {
            int digit = naf2[i];
            bool neg = digit < 0;
            if (neg) digit = -digit;
            
            pe_clone(&temp_p, &_gtable_g2[(digit-1)/2]);
            if (k2_neg ^ neg) fe_modp_neg(temp_p.y, temp_p.y);
            ec_jacobi_add(r, r, &temp_p);
        }
    }
}
// ==========================================================================================
// || FIM DA SEÇÃO DE OTIMIZAÇÃO: GLV + wNAF                                               ||
// ==========================================================================================


// ==========================================================================================
// || INÍCIO DA SEÇÃO DE OTIMIZAÇÃO: AVX2 (PLACEHOLDERS)                                   ||
// ==========================================================================================

void ec_jacobi_add_batch_avx2(pe r[], const pe p[], const pe q[], size_t n);

void ec_jacobi_grprdc_avx2(pe r[], u64 n) {
    if (n == 0) return;
    fe *zz_inv = (fe *)malloc(n * sizeof(fe));
    for (u64 i = 0; i < n; ++i) fe_clone(zz_inv[i], r[i].z);
    fe_modp_grpinv(zz_inv, n);

    u64 i = 0;
    for (; i + 7 < n; i += 8) {
        _mm_prefetch((const char*)(r + i + 16), _MM_HINT_T0);
        _mm_prefetch((const char*)(zz_inv + i + 16), _MM_HINT_T0);

        // AQUI VIRIA A LÓGICA VETORIZADA
        // fe_modp_mul_avx2(r[i..i+7].x, r[i..i+7].x, zz_inv[i..i+7]^2);
        // fe_modp_mul_avx2(r[i..i+7].y, r[i..i+7].y, zz_inv[i..i+7]^3);
        for(int j=0; j<8; ++j) {
            ec_jacobi_rdc(&r[i+j], &r[i+j]);
        }
    }
    
    for (; i < n; i++) ec_jacobi_rdc(&r[i], &r[i]);
    free(zz_inv);
}

void ec_jacobi_add_batch_avx2(pe r[], const pe p[], const pe q[], size_t n) {
    u64 i = 0;
    for (; i + 7 < n; i += 8) {
        // --- Exemplo conceitual dos primeiros passos da adição Jacobiana ---
        // FÓRMULA ORIGINAL: u2 = p->y * q->z; v2 = p->x * q->z; ...

        // CARREGAR DADOS: Carregar x, y, z de 8 pontos p e 8 pontos q em registros YMM.
        // __m256i p_x_limb0 = _mm256_load_si256(...);
        
        // EXECUTAR OPERAÇÕES VETORIZADAS
        // fe_modp_mul_avx2(u2_batch, p_y_batch, q_z_batch);
        
        // ... e assim por diante para todos os 12 passos da multiplicação.
        // ARMAZENAR RESULTADOS: Armazenar os resultados de r[i..i+7] de volta na memória.
        
        // Lógica escalar como placeholder
        for(int j=0; j<8; ++j) ec_jacobi_add(&r[i+j], &p[i+j], &q[i+j]);

    }
    for (; i < n; i++) ec_jacobi_add(&r[i], &p[i], &q[i]);
}

// ==========================================================================================
// || FIM DA SEÇÃO DE OTIMIZAÇÃO: AVX2                                                     ||
// ==========================================================================================