#include "ggml_turboquant.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// ---------------------------------------------------------------------
// Lloyd-Max codebooks for N(0,1), independently computed (see
// generate_codebook.py) via classic Lloyd's algorithm on 20M samples,
// converged to <1e-7 centroid shift. MSE matches the TurboQuant paper's
// reported values (3-bit: 0.0346 vs paper's 0.034, 4-bit: 0.0095 vs 0.009).
// ---------------------------------------------------------------------
static const float TQ3_CODEBOOK[8] = {
    -2.15378061f, -1.34594303f, -0.75760113f, -0.24651119f,
     0.24373149f,  0.75461671f,  1.34234174f,  2.15101808f
};

static const float TQ4_CODEBOOK[16] = {
    -2.72910483f, -2.06576973f, -1.61525775f, -1.25372221f,
    -0.94019814f, -0.65480632f, -0.38604090f, -0.12646789f,
     0.13024319f,  0.38968998f,  0.65846276f,  0.94434183f,
     1.25809455f,  1.62007773f,  2.07204708f,  2.73592722f
};

// Precomputed midpoints between adjacent codebook centroids, used for
// O(log levels) nearest-neighbor search via binary search.
static const float TQ3_MID[7] = {
    -1.74986182f, -1.05177208f, -0.50205616f, -0.00138985f,
     0.49917410f,  1.04847923f,  1.74667991f
};
static const float TQ4_MID[15] = {
    -2.39743728f, -1.84051374f, -1.43448998f, -1.09696017f,
    -0.79750223f, -0.52042361f, -0.25625440f,  0.00188765f,
     0.25996659f,  0.52407637f,  0.80140230f,  1.10121819f,
     1.43908614f,  1.84606241f,  2.40398715f
};

static inline int nearest_idx(float v, const float * mid, int n_mid) {
    // binary search over sorted midpoints -> returns codebook index [0, n_mid]
    int lo = 0, hi = n_mid;
    while (lo < hi) {
        int mid_i = (lo + hi) / 2;
        if (v < mid[mid_i]) hi = mid_i; else lo = mid_i + 1;
    }
    return lo;
}

// ---------------------------------------------------------------------
// Fast Walsh-Hadamard Transform, in-place, orthonormal (scaled by 1/sqrt(n))
// so that it is its own inverse: FWHT(FWHT(x)) == x.
// Requires n to be a power of two. O(n log n).
// ---------------------------------------------------------------------
void tq_fwht_inplace(float * x, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += (len << 1)) {
            for (int j = i; j < i + len; j++) {
                float a = x[j];
                float b = x[j + len];
                x[j]       = a + b;
                x[j + len] = a - b;
            }
        }
    }
    const float norm = 1.0f / sqrtf((float) n);
    for (int i = 0; i < n; i++) x[i] *= norm;
}

// ---------------------------------------------------------------------
// Deterministic pseudo-random sign flip (+1/-1 diagonal matrix).
// Combined with FWHT this approximates a random orthogonal rotation in
// O(n log n) instead of the O(n^2) cost of a dense random rotation matrix
// - this is the "randomized Hadamard transform" trick. Self-inverse
// (applying the same signs twice cancels out) since it's a diagonal
// +-1 matrix, D * D = I.
// ---------------------------------------------------------------------
void tq_apply_sign_flip(float * x, int n, uint32_t seed) {
    uint32_t s = seed;
    for (int i = 0; i < n; i++) {
        // xorshift32
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        if (s & 1) x[i] = -x[i];
    }
}

#define TQ_SIGN_SEED 0x5eed1234u

// ---------------------------------------------------------------------
// Bit packing helpers (MSB-first within each byte, sequential across values)
// ---------------------------------------------------------------------
static void pack_bits(const uint8_t * idx, int n, int bits, uint8_t * out) {
    memset(out, 0, (n * bits + 7) / 8);
    uint64_t acc = 0;
    int acc_bits = 0;
    int out_pos = 0;
    for (int i = 0; i < n; i++) {
        acc = (acc << bits) | (idx[i] & ((1u << bits) - 1));
        acc_bits += bits;
        while (acc_bits >= 8) {
            acc_bits -= 8;
            out[out_pos++] = (uint8_t)((acc >> acc_bits) & 0xFF);
        }
    }
    if (acc_bits > 0) {
        out[out_pos++] = (uint8_t)((acc << (8 - acc_bits)) & 0xFF);
    }
}

static void unpack_bits(const uint8_t * in, int n, int bits, uint8_t * idx) {
    uint64_t acc = 0;
    int acc_bits = 0;
    int in_pos = 0;
    for (int i = 0; i < n; i++) {
        while (acc_bits < bits) {
            acc = (acc << 8) | in[in_pos++];
            acc_bits += 8;
        }
        acc_bits -= bits;
        idx[i] = (uint8_t)((acc >> acc_bits) & ((1u << bits) - 1));
    }
}

// ---------------------------------------------------------------------
// TQ3_0: 3-bit quantize / dequantize
// ---------------------------------------------------------------------
void quantize_row_tq3_0(const float * restrict x, block_tq3_0 * restrict y, int64_t n) {
    float buf[TQ_BLOCK_SIZE];
    uint8_t idx[TQ_BLOCK_SIZE];

    for (int64_t b = 0; b < n / TQ_BLOCK_SIZE; b++) {
        memcpy(buf, x + b * TQ_BLOCK_SIZE, sizeof(buf));

        tq_apply_sign_flip(buf, TQ_BLOCK_SIZE, TQ_SIGN_SEED);
        tq_fwht_inplace(buf, TQ_BLOCK_SIZE);

        // RMS scale so the codebook (trained for unit variance) applies
        double ss = 0.0;
        for (int i = 0; i < TQ_BLOCK_SIZE; i++) ss += (double) buf[i] * buf[i];
        float scale = (float) sqrt(ss / TQ_BLOCK_SIZE);
        if (scale < 1e-8f) scale = 1e-8f;
        float inv_scale = 1.0f / scale;

        for (int i = 0; i < TQ_BLOCK_SIZE; i++) {
            idx[i] = (uint8_t) nearest_idx(buf[i] * inv_scale, TQ3_MID, 7);
        }

        y[b].d = scale;
        pack_bits(idx, TQ_BLOCK_SIZE, 3, y[b].qs);
    }
}

void dequantize_row_tq3_0(const block_tq3_0 * restrict x, float * restrict y, int64_t n) {
    uint8_t idx[TQ_BLOCK_SIZE];

    for (int64_t b = 0; b < n / TQ_BLOCK_SIZE; b++) {
        unpack_bits(x[b].qs, TQ_BLOCK_SIZE, 3, idx);

        float * out = y + b * TQ_BLOCK_SIZE;
        for (int i = 0; i < TQ_BLOCK_SIZE; i++) {
            out[i] = TQ3_CODEBOOK[idx[i]] * x[b].d;
        }

        tq_fwht_inplace(out, TQ_BLOCK_SIZE);       // self-inverse
        tq_apply_sign_flip(out, TQ_BLOCK_SIZE, TQ_SIGN_SEED); // self-inverse
    }
}

// ---------------------------------------------------------------------
// TQ4_0: 4-bit quantize / dequantize
// ---------------------------------------------------------------------
void quantize_row_tq4_0(const float * restrict x, block_tq4_0 * restrict y, int64_t n) {
    float buf[TQ_BLOCK_SIZE];

    for (int64_t b = 0; b < n / TQ_BLOCK_SIZE; b++) {
        memcpy(buf, x + b * TQ_BLOCK_SIZE, sizeof(buf));

        tq_apply_sign_flip(buf, TQ_BLOCK_SIZE, TQ_SIGN_SEED);
        tq_fwht_inplace(buf, TQ_BLOCK_SIZE);

        double ss = 0.0;
        for (int i = 0; i < TQ_BLOCK_SIZE; i++) ss += (double) buf[i] * buf[i];
        float scale = (float) sqrt(ss / TQ_BLOCK_SIZE);
        if (scale < 1e-8f) scale = 1e-8f;
        float inv_scale = 1.0f / scale;

        y[b].d = scale;
        for (int i = 0; i < TQ_BLOCK_SIZE; i += 2) {
            uint8_t i0 = (uint8_t) nearest_idx(buf[i]     * inv_scale, TQ4_MID, 15);
            uint8_t i1 = (uint8_t) nearest_idx(buf[i + 1] * inv_scale, TQ4_MID, 15);
            y[b].qs[i / 2] = (uint8_t)(i0 | (i1 << 4));
        }
    }
}

void dequantize_row_tq4_0(const block_tq4_0 * restrict x, float * restrict y, int64_t n) {
    for (int64_t b = 0; b < n / TQ_BLOCK_SIZE; b++) {
        float * out = y + b * TQ_BLOCK_SIZE;
        for (int i = 0; i < TQ_BLOCK_SIZE; i += 2) {
            uint8_t byte = x[b].qs[i / 2];
            out[i]     = TQ4_CODEBOOK[byte & 0x0F]  * x[b].d;
            out[i + 1] = TQ4_CODEBOOK[byte >> 4]     * x[b].d;
        }
        tq_fwht_inplace(out, TQ_BLOCK_SIZE);
        tq_apply_sign_flip(out, TQ_BLOCK_SIZE, TQ_SIGN_SEED);
    }
}
