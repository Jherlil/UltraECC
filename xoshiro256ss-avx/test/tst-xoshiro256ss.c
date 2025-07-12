#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <threads.h>
#include <unistd.h> /* sysconf */

#include "xoshiro256ss.h"
#include "xoshiro256starstar_orig.h"

#define TEST_FAIL(...)                                                         \
	do {                                                                   \
		fprintf(stderr, "FAIL %s:%d ", __FILE__, __LINE__);            \
		fprintf(stderr, __VA_ARGS__);                                  \
		fprintf(stderr, "\n");                                         \
	} while (0)

static uint64_t
splitmix64(uint64_t *st)
{
	uint64_t rt;

	rt = (*st += UINT64_C(0x9E3779B97f4A7C15));
	rt = (rt ^ (rt >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
	rt = (rt ^ (rt >> 27)) * UINT64_C(0x94D049BB133111EB);

	return rt ^ (rt >> 31);
}

/*
 * Verify if the intialization is performed expliticly like this:
 *
 * 1. Split the seed with splitmix64
 * 2. Jump 8 times, to split the state into paralel PRNGs (as colums)
 *
 */
static void
seed_orig_rng(uint64_t *s, uint64_t seed)
{
	uint64_t smx = seed;
	for (size_t i = 0; i < 4; i++)
		s[i] = splitmix64(&smx);
}

static int
test_init(void *p)
{
	(void)p;
	_Alignas(0x40) struct xoshiro256ss rng;

	const uint64_t seed = UINT64_C(0x100e881);
	uint64_t       s[4];
	uint64_t       expct[XOSHIRO256SS_WIDTH][4];
	seed_orig_rng(s, seed);
	for (size_t i = 0; i < XOSHIRO256SS_WIDTH; i++) {
		xoshiro256starstar_orig_jump(s);
		memcpy(expct[i], s, 8 * 4);
	}

	int rt = xoshiro256ss_init(&rng, seed);
	if (rt < 0) {
		TEST_FAIL("init() reported error: %d", rt);
		return -1;
	}
	if (rt != XOSHIRO256SS_TECH) {
		TEST_FAIL("init() reported wrong technology: %d (should be %d)",
			rt, XOSHIRO256SS_TECH);
		return -1;
	}

	/* Verify */
	for (size_t i = 0; i < XOSHIRO256SS_WIDTH; i++) {
		for (size_t j = 0; j < 4; j++) {
			if (expct[i][j] != rng.s[XOSHIRO256SS_WIDTH * j + i]) {
				TEST_FAIL(
					"PRGN init, wrong state at i=%zu, j=%zu",
					i, j);
				return -1;
			}
		}
	}

	return 0;
}

static int
test_zeroinit(void *p)
{
	(void)p;

	_Alignas(0x40) struct xoshiro256ss rng;
	xoshiro256ss_init(&rng, UINT64_C(0x00));
	for (size_t i = 0; i < XOSHIRO256SS_WIDTH * 4; i++) {
		if (rng.s[i] == 0) {
			TEST_FAIL("state contains 0x00 at %zu", i);
			return -1;
		}
	}

	return 0;
}

static int
test_filln_aligned01(void *p)
{
	(void)p;
	int rt = 0;

	uint64_t seed = UINT64_C(0x834333c);
	uint64_t s[4];
#define SIZE UINT64_C(3 * 1024 * 1024)
	uint64_t *expct = malloc(sizeof *expct * XOSHIRO256SS_WIDTH * SIZE);
	uint64_t *buf =
		aligned_alloc(0x40, sizeof *buf * XOSHIRO256SS_WIDTH * SIZE);
	if (!(expct && buf)) {
		TEST_FAIL("memory allocation");
		rt = -1;
		goto exit;
	}

	for (size_t b = 0; b < XOSHIRO256SS_WIDTH; b++) {
		seed_orig_rng(s, seed);
		for (size_t _ = 0; _ < b + 1; _++)
			xoshiro256starstar_orig_jump(s);
		for (size_t i = 0; i < SIZE; i++)
			expct[b * SIZE + i] = xoshiro256starstar_orig_next(s);
	}

	_Alignas(0x40) struct xoshiro256ss rng;
	xoshiro256ss_init(&rng, seed);
	xoshiro256ss_filln(&rng, buf, SIZE);

	for (size_t b = 0; b < XOSHIRO256SS_WIDTH; b++) {
		for (size_t i = 0; i < SIZE; i++) {
			if (buf[i * XOSHIRO256SS_WIDTH + b] !=
				expct[b * SIZE + i]) {
				TEST_FAIL(
					"result differs at block %zu, index %zu",
					b, i);
				rt = -1;
				goto exit;
			}
		}
	}
#undef SIZE
exit:
	free(expct);
	free(buf);

	return rt;
}

static inline double
to_double(uint64_t x)
{
	const union {
		uint64_t i;
		double	 d;
	} u = { .i = UINT64_C(0x3FF) << 52 | x >> 12 };
	return u.d - 1.0;
}

static int
test_filln_aligned02_f64n(void *p)
{
	(void)p;
	int rt = 0;

	uint64_t seed = UINT64_C(0x1e42fffc);
	uint64_t s[4];
#define SIZE UINT64_C(7 * 1024 * 1024)
	double *expct = malloc(sizeof *expct * XOSHIRO256SS_WIDTH * SIZE);
	double *buf =
		aligned_alloc(0x40, sizeof *buf * XOSHIRO256SS_WIDTH * SIZE);
	if (!(expct && buf)) {
		TEST_FAIL("memory allocation");
		rt = -1;
		goto exit;
	}

	for (size_t b = 0; b < XOSHIRO256SS_WIDTH; b++) {
		seed_orig_rng(s, seed);
		for (size_t _ = 0; _ < b + 1; _++)
			xoshiro256starstar_orig_jump(s);
		for (size_t i = 0; i < SIZE; i++)
			expct[b * SIZE + i] =
				to_double(xoshiro256starstar_orig_next(s));
	}

	_Alignas(0x40) struct xoshiro256ss rng;
	xoshiro256ss_init(&rng, seed);
	xoshiro256ss_filln_f64n(&rng, buf, SIZE);

	for (size_t b = 0; b < XOSHIRO256SS_WIDTH; b++) {
		for (size_t i = 0; i < SIZE; i++) {
			if (buf[i * XOSHIRO256SS_WIDTH + b] !=
				expct[b * SIZE + i]) {
				TEST_FAIL(
					"result differs at block %zu, index %zu",
					b, i);
				rt = -1;
				goto exit;
			}
		}
	}
#undef SIZE
exit:
	free(expct);
	free(buf);

	return rt;
}

/* Test harness */
#define TEST_THRDS_MAX (64)
static thrd_t test_thrd[TEST_THRDS_MAX];
static int    test_thrd_res[TEST_THRDS_MAX];

static int (*test_tests[])(void *) = { test_init, test_zeroinit,
	test_filln_aligned01, test_filln_aligned02_f64n };
static _Atomic(int) test_top	   = sizeof test_tests / sizeof *test_tests - 1;

int
test_thrd_worker(void *p)
{
	(void)p;

	int rt = 0;

	while (1) {
		int top = atomic_fetch_sub(&test_top, 1);
		if (top < 0)
			return rt;
		if (test_tests[top](NULL) < 0)
			rt = -1;
	}

	__builtin_unreachable();
}

int
main(int argc, char **argv)
{
	(void)argc;
	(void)argv;

	int  rt		   = 0;
	long num_cpus	   = sysconf(_SC_NPROCESSORS_ONLN);
	long test_thrd_num = TEST_THRDS_MAX > num_cpus - 1 ? num_cpus - 1 :
							     TEST_THRDS_MAX;
	for (int i = 0; i < test_thrd_num; i++)
		if (thrd_create(&test_thrd[i], test_thrd_worker, NULL) !=
			thrd_success) {
			TEST_FAIL("cannot create thread %d", i);
			rt = -1;
		}
	for (int i = 0; i < test_thrd_num; i++) {
		thrd_join(test_thrd[i], &test_thrd_res[i]);
		if (test_thrd_res[i] < 0)
			rt = -1;
	}
	if (rt == 0)
		printf("OK\n");

	return rt;
}
