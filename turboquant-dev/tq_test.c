#include "ggml_turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Defined manually (not via _USE_MATH_DEFINES / -D_GNU_SOURCE) so this
// builds cleanly under strict C11 on both MSVC and glibc without relying
// on compiler-specific feature macros.
#ifndef TQ_PI
#define TQ_PI 3.14159265358979323846f
#endif

static float randn(void) {
    // Box-Muller
    float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * TQ_PI * u2);
}

static int test_fwht_self_inverse(void) {
    float x[TQ_BLOCK_SIZE], orig[TQ_BLOCK_SIZE];
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) x[i] = orig[i] = randn() * (i % 5 + 1); // non-uniform scale on purpose
    tq_fwht_inplace(x, TQ_BLOCK_SIZE);
    tq_fwht_inplace(x, TQ_BLOCK_SIZE);
    float max_err = 0;
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) {
        float e = fabsf(x[i] - orig[i]);
        if (e > max_err) max_err = e;
    }
    printf("[test] FWHT self-inverse max error: %.8f  %s\n", max_err, max_err < 1e-4 ? "PASS" : "FAIL");
    return max_err < 1e-4;
}

static int test_sign_flip_self_inverse(void) {
    float x[TQ_BLOCK_SIZE], orig[TQ_BLOCK_SIZE];
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) x[i] = orig[i] = randn();
    tq_apply_sign_flip(x, TQ_BLOCK_SIZE, 0x5eed1234u);
    tq_apply_sign_flip(x, TQ_BLOCK_SIZE, 0x5eed1234u);
    int ok = memcmp(x, orig, sizeof(x)) == 0;
    printf("[test] sign-flip self-inverse: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// Round trip N blocks of random Gaussian data (worst case for the codebook,
// since it's tuned exactly for this - but real KV data won't be Gaussian
// pre-rotation, that's the whole point, so this validates the quantizer's
// ceiling performance) and report MSE.
static void test_roundtrip_gaussian(int n_blocks, int bits) {
    int64_t n = (int64_t) n_blocks * TQ_BLOCK_SIZE;
    float * x   = malloc(n * sizeof(float));
    float * y   = malloc(n * sizeof(float));

    for (int64_t i = 0; i < n; i++) x[i] = randn();

    size_t packed_bytes;
    if (bits == 3) {
        block_tq3_0 * blocks = malloc((size_t) n_blocks * sizeof(block_tq3_0));
        quantize_row_tq3_0(x, blocks, n);
        dequantize_row_tq3_0(blocks, y, n);
        packed_bytes = (size_t) n_blocks * sizeof(block_tq3_0);
        free(blocks);
    } else {
        block_tq4_0 * blocks = malloc((size_t) n_blocks * sizeof(block_tq4_0));
        quantize_row_tq4_0(x, blocks, n);
        dequantize_row_tq4_0(blocks, y, n);
        packed_bytes = (size_t) n_blocks * sizeof(block_tq4_0);
        free(blocks);
    }

    double mse = 0.0;
    for (int64_t i = 0; i < n; i++) {
        double e = (double) x[i] - (double) y[i];
        mse += e * e;
    }
    mse /= n;

    size_t fp16_bytes = n * 2;
    double compression = (double) fp16_bytes / (double) packed_bytes;
    double bpw = (double) packed_bytes * 8.0 / (double) n;

    printf("[test] TQ%d roundtrip: n=%lld  MSE=%.5f  bpw=%.2f  compression=%.2fx vs fp16\n",
           bits, (long long) n, mse, bpw, compression);

    free(x); free(y);
}

int main(void) {
    srand(42);
    int ok = 1;
    ok &= test_fwht_self_inverse();
    ok &= test_sign_flip_self_inverse();

    printf("\n-- round-trip MSE (paper reports: TQ3=0.034, TQ4=0.009) --\n");
    test_roundtrip_gaussian(2000, 3);
    test_roundtrip_gaussian(2000, 4);

    printf("\n-- struct sizes --\n");
    printf("block_tq3_0 = %zu bytes (expect 52)\n", sizeof(block_tq3_0));
    printf("block_tq4_0 = %zu bytes (expect 68)\n", sizeof(block_tq4_0));

    printf("\n%s\n", ok ? "ALL SANITY TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
