#pragma once
// TurboQuant CPU reference implementation
// Algorithm 1 ("TurboQuant_mse") from Zandieh et al., ICLR 2026 (arXiv:2504.19874)
//   1. Randomized Hadamard rotation (sign-flip + Fast Walsh-Hadamard Transform)
//   2. Per-block scale (RMS norm of rotated coords)
//   3. Lloyd-Max optimal scalar quantization vs. N(0,1) codebook
//
// QJL residual correction (Algorithm 2) is deliberately NOT implemented here.
// Multiple independent reproductions (see llama.cpp discussion #20969) found
// it increases variance more than it removes bias, and hurts top-1 token
// accuracy under softmax. MSE-only quantization is the community-converged
// choice as of this writing.

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TQ_BLOCK_SIZE 128   // matches typical attention head_dim

// --- TQ3: 3 bits/value, 128 values -> 48 bytes packed + 4 byte scale = 52 bytes/block ---
// bits-per-weight = 52*8/128 = 3.25 bpw  (4.9x compression vs fp16)
typedef struct {
    float    d;                      // block scale (RMS of rotated coords)
    uint8_t  qs[(TQ_BLOCK_SIZE * 3 + 7) / 8]; // packed 3-bit codebook indices
} block_tq3_0;

// --- TQ4: 4 bits/value, 128 values -> 64 bytes packed + 4 byte scale = 68 bytes/block ---
// bits-per-weight = 68*8/128 = 4.25 bpw  (3.8x compression vs fp16)
typedef struct {
    float    d;
    uint8_t  qs[TQ_BLOCK_SIZE / 2];  // packed 4-bit codebook indices (nibbles)
} block_tq4_0;

// Portable compile-time size check (works even without /std:c11 on MSVC,
// unlike _Static_assert which MSVC only recognizes in explicit C11/C17 mode).
typedef char tq_assert_tq3_0_size[(sizeof(block_tq3_0) == 52) ? 1 : -1];
typedef char tq_assert_tq4_0_size[(sizeof(block_tq4_0) == 68) ? 1 : -1];

// Quantize `n` floats (n must be a multiple of TQ_BLOCK_SIZE) into blocks.
void quantize_row_tq3_0(const float * restrict x, block_tq3_0 * restrict y, int64_t n);
void quantize_row_tq4_0(const float * restrict x, block_tq4_0 * restrict y, int64_t n);

// Dequantize back to float.
void dequantize_row_tq3_0(const block_tq3_0 * restrict x, float * restrict y, int64_t n);
void dequantize_row_tq4_0(const block_tq4_0 * restrict x, float * restrict y, int64_t n);

// Raw building blocks, exposed for testing / future fused-kernel work.
void tq_fwht_inplace(float * x, int n);           // in-place fast Walsh-Hadamard transform (orthonormal)
void tq_apply_sign_flip(float * x, int n, uint32_t seed); // deterministic pseudo-random sign flip, self-inverse

#ifdef __cplusplus
}
#endif
