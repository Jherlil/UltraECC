#include <stddef.h>
#include <stdint.h>

#include "xoshiro256ss.h"

uint64_t
xoshiro256ss_splitmix64(uint64_t *st)
{
	uint64_t rt;

	rt = (*st += UINT64_C(0x9E3779B97f4A7C15));
	rt = (rt ^ (rt >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
	rt = (rt ^ (rt >> 27)) * UINT64_C(0x94D049BB133111EB);

	return rt ^ (rt >> 31);
}

static inline uint64_t
rotl64(const uint64_t x, int k)
{
	return (x << k) | (x >> (64 - k));
}

static uint64_t
scalar_next(uint64_t *s)
{
	const uint64_t rt = rotl64(s[1] * 5, 7) * 9;

	const uint64_t t = s[1] << 17;
	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];
	s[2] ^= t;
	s[3] = rotl64(s[3], 45);

	return rt;
}

static void
scalar_jump128(uint64_t *s)
{
	static const uint64_t JUMP[4] = { 0x180ec6d33cfd0aba,
		0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };

	uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
	for (size_t i = 0; i < 4; i++)
		for (size_t b = 0; b < 64; b++) {
			if (JUMP[i] & UINT64_C(1) << b) {
				s0 ^= s[0];
				s1 ^= s[1];
				s2 ^= s[2];
				s3 ^= s[3];
			}
			scalar_next(s);
		}
	s[0] = s0;
	s[1] = s1;
	s[2] = s2;
	s[3] = s3;
}

int
xoshiro256ss_init(struct xoshiro256ss *rng, uint64_t seed)
{
	uint64_t smx = seed;
	uint64_t st[4];

	for (size_t j = 0; j < 4; j++)
		st[j] = xoshiro256ss_splitmix64(&smx);
	for (size_t i = 0; i < XOSHIRO256SS_WIDTH; i++) {
		scalar_jump128(st);
		for (size_t j = 0; j < 4; j++)
			rng->s[i + XOSHIRO256SS_WIDTH * j] = st[j];
	}

	return 1;
}

extern void
xoshiro256ss_filln_avx2(
	struct xoshiro256ss *rng, void *buf, size_t n, int f64n_conv);

extern void
xoshiro256ss_filln_avx512(
	struct xoshiro256ss *rng, void *buf, size_t n, int f64n_conv);

void
xoshiro256ss_filln(struct xoshiro256ss *rng, uint64_t *buf, size_t n)
{
#if XOSHIRO256SS_TECH == 1

	xoshiro256ss_filln_avx2(rng, buf, n, 0);

#elif XOSHIRO256SS_TECH == 2

	xoshiro256ss_filln_avx512(rng, buf, n, 0);

#else

#error "Wrong technology specifier"

#endif
}

void
xoshiro256ss_filln_f64n(struct xoshiro256ss *rng, double *buf, size_t n)
{
#if XOSHIRO256SS_TECH == 1

	xoshiro256ss_filln_avx2(rng, buf, n, 1);

#elif XOSHIRO256SS_TECH == 2

	xoshiro256ss_filln_avx512(rng, buf, n, 1);

#else

#error "Wrong technology specifier"

#endif
}
