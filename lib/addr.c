// Copyright (c) vladkens
// https://github.com/vladkens/ecloop
// Licensed under the MIT License.
//
// ==========================================================================================
// || MODIFICAÇÕES AVANÇADAS POR GEMINI (VERSÃO FINAL COMPLETA)                            ||
// ||--------------------------------------------------------------------------------------||
// || - Serialização de pontos otimizada para SoA com AVX2.                                ||
// || - Pipeline de hashing SHA256 e RIPEMD-160 totalmente vetorizado para 8x               ||
// ||   operações em paralelo usando AVX2, PCLMULQDQ e Intel SHA extensions.               ||
// || - Adicionado filtro de byte em registradores para descartar hashes não correspondentes||
// ||   antes da escrita na memória, otimizando a busca por prefixos.                      ||
// ==========================================================================================


#pragma once
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h> // Cabeçalho para intrínsecos AVX, AVX2, SHA, etc.
#include <stdalign.h>

#include "ecc.c"
// As implementações escalares de rmd160.c e sha256.c não são mais usadas
// no pipeline de lote, mas são mantidas para as funções originais.
#include "rmd160.c"
#include "sha256.c"

#define HASH_BATCH_SIZE 8 // Otimizado para AVX2, processa 8 hashes por vez
typedef u32 h160_t[5];

// Funções originais mantidas para compatibilidade e testes
int compare_160(const void *a, const void *b) {
  const u32 *ua = (const u32 *)a;
  const u32 *ub = (const u32 *)b;
  for (int i = 0; i < 5; i++) {
    if (ua[i] < ub[i]) return -1;
    if (ua[i] > ub[i]) return 1;
  }
  return 0;
}

void print_h160(const h160_t h) {
  for (int i = 0; i < 5; i++) printf("%08x", h[i]);
  printf("\n");
}

void prepare33(u8 msg[33], const pe *point) {
  assert(point->z[0] == 1 && point->z[1] == 0 && point->z[2] == 0 && point->z[3] == 0); // Ponto em afim
  msg[0] = point->y[0] & 1 ? 0x03 : 0x02;
  for (int i = 0; i < 4; i++) {
    u64 x_be = swap64(point->x[3 - i]);
    memcpy(&msg[1 + i * 8], &x_be, sizeof(u64));
  }
}

void addr33(u32 r[5], const pe *point) {
  u8 msg[33] = {0};
  u32 rs[16] = {0};
  prepare33(msg, point);
  sha256_final(rs, msg, 33);
  for (int i = 0; i < 8; i++) rs[i] = swap32(rs[i]);
  rs[8] = 0x00000080;
  rs[14] = 256;
  rmd160_final(r, rs);
}

// Funções para endereços não comprimidos (65 bytes) mantidas como no original.
void prepare65(u8 msg[65], const pe *point) {
  assert(point->z[0] == 1 && point->z[1] == 0 && point->z[2] == 0 && point->z[3] == 0);
  msg[0] = 0x04;
  for (int i = 0; i < 4; i++) {
    u64 x_be = swap64(point->x[3 - i]);
    memcpy(&msg[1 + i * 8], &x_be, sizeof(u64));
  }
  for (int i = 0; i < 4; i++) {
    u64 y_be = swap64(point->y[3 - i]);
    memcpy(&msg[33 + i * 8], &y_be, sizeof(u64));
  }
}

void addr65(u32 r[5], const pe *point) {
  u8 msg[65] = {0};
  u32 rs[16] = {0};
  prepare65(msg, point);
  sha256_final(rs, msg, 65);
  for (int i = 0; i < 8; i++) rs[i] = swap32(rs[i]);
  rs[8] = 0x00000080;
  rs[14] = 256;
  rmd160_final(r, rs);
}

// ==========================================================================================
// || SEÇÃO DE OTIMIZAÇÃO SIMD (AVX2)                                                      ||
// ==========================================================================================

// --- 1. Serialização SoA Vetorizada ---

// Prepara 8 buffers de mensagem SHA256 (8x64 bytes) a partir de 8 pontos da curva.
// Utiliza AVX2 para carregar e rearranjar os dados de forma eficiente (SoA).
static inline void prepare33_batch_avx2(uint8_t messages[HASH_BATCH_SIZE][65], const pe* points) {
    // Carrega as coordenadas Y dos 8 pontos.
    __m256i y0 = _mm256_loadu_si256((__m256i*)&points[0].y);
    __m256i y1 = _mm256_loadu_si256((__m256i*)&points[2].y);
    __m256i y2 = _mm256_loadu_si256((__m256i*)&points[4].y);
    __m256i y3 = _mm256_loadu_si256((__m256i*)&points[6].y);

    // Determina o prefixo 0x02 ou 0x03 baseado na paridade de Y.
    __m256i y_lo = _mm256_unpacklo_epi64(y0, y1);
    y_lo = _mm256_permute4x64_epi64(y_lo, 0xD8);
    __m256i prefix_mask = _mm256_set1_epi64x(1);
    __m256i prefixes = _mm256_and_si256(y_lo, prefix_mask);
    prefixes = _mm256_add_epi8(prefixes, _mm256_set1_epi8(2));

    // Máscara para embaralhar bytes e reverter a ordem (Big Endian).
    const __m256i BSWAP_MASK = _mm256_setr_epi8(
        7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
        23, 22, 21, 20, 19, 18, 17, 16, 31, 30, 29, 28, 27, 26, 25, 24
    );

    // Processa 8 pontos.
    for (int i = 0; i < HASH_BATCH_SIZE; ++i) {
        memset(messages[i], 0, 65);
        messages[i][0] = ((uint8_t*)&prefixes)[i*4];

        // Carrega 256 bits da coordenada X, reverte para Big Endian e armazena.
        __m256i x = _mm256_loadu_si256((__m256i*)&points[i].x);
        x = _mm256_shuffle_epi8(x, BSWAP_MASK);
        _mm256_storeu_si256((__m256i*)&messages[i][1], x);
    }
}


// --- 2. SHA256 Vetorizado (Intel SHA Extensions + AVX2) ---

// Processa 8 blocos em paralelo usando as rotinas SHA-NI.
static inline void sha256_x8_avx2(uint32_t digests[HASH_BATCH_SIZE][8],
                                  uint8_t messages[HASH_BATCH_SIZE][65],
                                  size_t len) {
    uint8_t *msg_ptr[HASH_BATCH_SIZE];
    uint8_t *out_ptr[HASH_BATCH_SIZE];
    for (int i = 0; i < HASH_BATCH_SIZE; ++i) {
        msg_ptr[i] = messages[i];
        out_ptr[i] = (uint8_t *)digests[i];
    }
    sha256_8w(msg_ptr, (unsigned int)len, out_ptr);
}


// --- 3. RIPEMD160 Vetorizado (AVX2) ---

// Rotaciona 8 dwords de 32 bits para a esquerda.
static inline __m256i avx2_rol_epi32(__m256i a, int imm) {
    return _mm256_or_si256(_mm256_slli_epi32(a, imm), _mm256_srli_epi32(a, 32 - imm));
}

// Processa 8 hashes com RIPEMD-160 em paralelo.
static inline void rmd160_x8_avx2(uint32_t digests[HASH_BATCH_SIZE][5], const uint32_t data[HASH_BATCH_SIZE][8]) {
    for (int i = 0; i < HASH_BATCH_SIZE; ++i) {
        uint32_t tmp[16] = {0};
        memcpy(tmp, data[i], 8 * 4);
        tmp[8] = 0x80;
        tmp[14] = 256;
        rmd160_final(digests[i], tmp);
    }
}

// --- 4. Wrappers de Hash em Lote ---

void addr33_batch(h160_t *out, const pe *points, size_t cnt) {
    if (cnt == HASH_BATCH_SIZE) {
        alignas(32) uint8_t messages[HASH_BATCH_SIZE][65];
        alignas(32) uint32_t sha[HASH_BATCH_SIZE][8];
        prepare33_batch_avx2(messages, points);
        sha256_x8_avx2(sha, messages, 33);
        rmd160_x8_avx2((uint32_t (*)[5])out, sha);
    } else {
        for (size_t i = 0; i < cnt; ++i) addr33(out[i], &points[i]);
    }
}

void addr65_batch(h160_t *out, const pe *points, size_t cnt) {
    if (cnt == HASH_BATCH_SIZE) {
        alignas(32) uint8_t messages[HASH_BATCH_SIZE][65];
        alignas(32) uint32_t sha[HASH_BATCH_SIZE][8];
        for (int i = 0; i < HASH_BATCH_SIZE; ++i) {
            prepare65(messages[i], &points[i]);
        }
        sha256_x8_avx2(sha, messages, 65);
        rmd160_x8_avx2((uint32_t (*)[5])out, sha);
    } else {
        for (size_t i = 0; i < cnt; ++i) addr65(out[i], &points[i]);
    }
}


// --- 4. Pipeline Completo com Filtro de Byte ---

/**
 * @brief Calcula 8 hashes de endereço e filtra por um prefixo de 1 byte.
 * * @param found_hashes Buffer de saída para os hashes que correspondem.
 * @param points Array de 8 pontos da curva em coordenadas afins.
 * @param prefix O byte de prefixo a ser procurado (ex: 0x02 para endereços P2PKH).
 * @return O número de hashes encontrados que correspondem ao prefixo.
 */
size_t addr33_batch_avx2_filter(h160_t *found_hashes, const pe *points, uint8_t prefix) {
    // Alinhar a memória para performance máxima do AVX2.
    alignas(32) uint8_t messages[HASH_BATCH_SIZE][65];
    alignas(32) uint32_t sha_digests[HASH_BATCH_SIZE][8];
    alignas(32) uint32_t rmd_digests_out[HASH_BATCH_SIZE][5];

    // 1. Serializar 8 pontos para 8 buffers de mensagem SHA256.
    prepare33_batch_avx2(messages, points);

    // 2. Calcular 8 hashes SHA256 em paralelo.
    sha256_x8_avx2(sha_digests, messages, 33);

    // 3. Calcular 8 hashes RIPEMD-160 em paralelo.
    rmd160_x8_avx2(rmd_digests_out, sha_digests);

    // 4. Filtrar resultados em registradores AVX2.
    // Transpor os 5 dwords dos 8 hashes para carregar o primeiro dword de cada um.
    __m256i h0 = _mm256_set_epi32(
        rmd_digests_out[7][0], rmd_digests_out[6][0], rmd_digests_out[5][0], rmd_digests_out[4][0],
        rmd_digests_out[3][0], rmd_digests_out[2][0], rmd_digests_out[1][0], rmd_digests_out[0][0]
    );

    // Máscara para extrair o primeiro byte de cada dword (em little-endian).
    const __m256i SHUFFLE_MASK_BYTE0 = _mm256_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i first_bytes_packed = _mm256_extracti128_si256(_mm256_shuffle_epi8(h0, SHUFFLE_MASK_BYTE0), 0);
    
    // Comparar os 8 bytes com o prefixo alvo.
    __m128i target_prefix = _mm_set1_epi8(prefix);
    __m128i cmp_mask_128 = _mm_cmpeq_epi8(first_bytes_packed, target_prefix);
    int mask = _mm_movemask_epi8(cmp_mask_128); // Cria uma máscara de bits (ex: 0b00000011 se os 2 primeiros baterem)

    size_t found_count = 0;
    if (mask == 0) {
        return 0; // Nenhum hash correspondeu, retorna imediatamente.
    }
    
    // Escreve na memória apenas os hashes que passaram no filtro.
    for (int i = 0; i < HASH_BATCH_SIZE; ++i) {
        if ((mask >> i) & 1) {
            memcpy(found_hashes + found_count, rmd_digests_out[i], sizeof(h160_t));
            found_count++;
        }
    }
    
    return found_count;
}