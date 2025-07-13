#pragma once

#define USE_BUILTIN 1
#define HAS_BUILTIN(fn) (USE_BUILTIN && __has_builtin(fn))
#if defined(__ADX__) || defined(__BMI2__) || defined(__BMI__)
#include <immintrin.h>
#endif

typedef __uint128_t u128;
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;
#define INLINE static inline __attribute__((always_inline))

// https://stackoverflow.com/a/32107675/3664464

#define MIN(x, y)                                                                                  \
  ({                                                                                               \
    __auto_type _x = (x);                                                                          \
    __auto_type _y = (y);                                                                          \
    _x < _y ? _x : _y;                                                                             \
  })

#define MAX(x, y)                                                                                  \
  ({                                                                                               \
    __auto_type _x = (x);                                                                          \
    __auto_type _y = (y);                                                                          \
    _x > _y ? _x : _y;                                                                             \
  })

// https://clang.llvm.org/docs/LanguageExtensions.html#:~:text=__builtin_addcll
// https://gcc.gnu.org/onlinedocs/gcc/Integer-Overflow-Builtins.html#:~:text=__builtin_uaddll_overflow

INLINE u64 addc64(u64 x, u64 y, u64 carryin, u64 *carryout) {
#if defined(__ADX__)
  unsigned long long out;
  unsigned char c = _addcarry_u64((unsigned char)carryin, x, y, &out);
  *carryout = c;
  return out;
#elif HAS_BUILTIN(__builtin_addcll)
  return __builtin_addcll(x, y, carryin, carryout);
#else
  u64 rs;
  bool overflow1 = __builtin_uaddll_overflow(x, y, &rs);
  bool overflow2 = __builtin_uaddll_overflow(rs, carryin, &rs);
  *carryout = (overflow1 || overflow2) ? 1 : 0;
  return rs;
#endif
}

// https://clang.llvm.org/docs/LanguageExtensions.html#:~:text=__builtin_subcll
// https://gcc.gnu.org/onlinedocs/gcc/Integer-Overflow-Builtins.html#:~:text=__builtin_usubll_overflow

INLINE u64 subc64(u64 x, u64 y, u64 carryin, u64 *carryout) {
#if defined(__ADX__)
  unsigned long long out;
  unsigned char c = _subborrow_u64((unsigned char)carryin, x, y, &out);
  *carryout = c;
  return out;
#elif HAS_BUILTIN(__builtin_subcll)
  return __builtin_subcll(x, y, carryin, carryout);
#else
  u64 rs;
  bool underflow1 = __builtin_usubll_overflow(x, y, &rs);
  bool underflow2 = __builtin_usubll_overflow(rs, carryin, &rs);
  *carryout = (underflow1 || underflow2) ? 1 : 0;
  return rs;
#endif
}

// Other builtins

#if HAS_BUILTIN(__builtin_rotateleft32)
  #define rotl32(x, n) __builtin_rotateleft32(x, n)
#else
  #define rotl32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#endif

#if HAS_BUILTIN(__builtin_bswap32)
  #define swap32(x) __builtin_bswap32(x)
#else
  #define swap32(x) ((x) << 24) | ((x) << 8 & 0x00ff0000) | ((x) >> 8 & 0x0000ff00) | ((x) >> 24)
#endif

#if HAS_BUILTIN(__builtin_bswap64)
  #define swap64(x) __builtin_bswap64(x)
#else
  #define swap64(x)                                                                                \
    ((x) << 56) | ((x) << 40 & 0x00ff000000000000) | ((x) << 24 & 0x0000ff0000000000) |            \
        ((x) << 8 & 0x000000ff00000000) | ((x) >> 8 & 0x00000000ff000000) |                        \
        ((x) >> 24 & 0x0000000000ff0000) | ((x) >> 40 & 0x000000000000ff00) | ((x) >> 56)
#endif
