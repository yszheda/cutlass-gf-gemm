/**
 * @file gf_ops.h
 * @brief Galois Field arithmetic operations
 *
 * This file defines the GF(2^8) element type and basic operations
 * for use in CUDA kernels.
 */

#pragma once

#include <stdint.h>

#ifdef __CUDACC__
#define GF_HOST_DEVICE __host__ __device__
#else
#define GF_HOST_DEVICE
#endif

namespace cutlass {

// ============================================================================
//                     GF(2^8) Element Type Definition
// ============================================================================

/// Galois Field element - stored as uint8_t
struct gf28_t {
    uint8_t storage;

    GF_HOST_DEVICE
    gf28_t() : storage(0) {}

    GF_HOST_DEVICE
    gf28_t(uint8_t value) : storage(value) {}

    GF_HOST_DEVICE
    gf28_t(int value) : storage(static_cast<uint8_t>(value)) {}

    GF_HOST_DEVICE
    operator uint8_t() const { return storage; }

    GF_HOST_DEVICE
    operator int() const { return static_cast<int>(storage); }

    GF_HOST_DEVICE
    gf28_t& operator=(uint8_t value) {
        storage = value;
        return *this;
    }

    GF_HOST_DEVICE
    gf28_t& operator=(int value) {
        storage = static_cast<uint8_t>(value);
        return *this;
    }
};

// ============================================================================
//                     GF(2^8) Arithmetic Traits
// ============================================================================

/// Traits for GF(2^8) arithmetic
struct GF28Arithmetic {
    using Element = gf28_t;

    /// Prime field characteristic (for GF(2^8), this is 2)
    static constexpr int kFieldCharacteristic = 2;

    /// Field extension degree
    static constexpr int kExtensionDegree = 8;

    /// Field size = 2^8 = 256
    static constexpr int kFieldSize = 256;

    /// Field max element = 255
    static constexpr int kFieldMax = 255;

    /// Primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 = 0x11D
    static constexpr uint8_t kPrimitivePolynomial = 0x1D;

    /// Lookup table sizes
    static constexpr int kLogTableSize = 256;
    static constexpr int kExpTableSize = 768;  // 255 * 3 for overflow handling
};

} // namespace cutlass

// ============================================================================
//                     Global Namespace GF Operations
// ============================================================================

/**
 * @brief Initialize GF(2^8) lookup tables
 *
 * @param gf_exp Device pointer to exponentiation table (size >= 768)
 * @param gf_log Device pointer to logarithm table (size >= 256)
 */
extern "C" void init_gf_tables(uint8_t* gf_exp, uint8_t* gf_log);

/**
 * @brief GF(2^8) multiplication using lookup tables
 *
 * @param a First operand
 * @param b Second operand
 * @param gf_exp Exponentiation table
 * @param gf_log Logarithm table
 * @return Product in GF(2^8)
 */
GF_HOST_DEVICE inline uint8_t gf_mul(uint8_t a, uint8_t b,
                                      const uint8_t* gf_exp,
                                      const uint8_t* gf_log) {
    if (a == 0 || b == 0) return 0;
    return gf_exp[gf_log[a] + gf_log[b]];
}

/**
 * @brief GF(2^8) multiplication using constant memory tables
 */
GF_HOST_DEVICE inline uint8_t gf_mul_const(uint8_t a, uint8_t b);

/**
 * @brief GF(2^8) addition (XOR)
 */
GF_HOST_DEVICE inline uint8_t gf_add(uint8_t a, uint8_t b) {
    return a ^ b;
}

/**
 * @brief GF(2^8) bit-level multiplication (no tables required)
 */
GF_HOST_DEVICE inline uint8_t gf_mul_bitwise(uint8_t a, uint8_t b) {
    uint8_t result = 0;
    uint8_t prim_poly = 0x1D;  // x^8 + x^4 + x^3 + x^2 + 1

    while (b) {
        if (b & 1) {
            result ^= a;
        }
        a = (a << 1) ^ ((a & 0x80) ? prim_poly : 0);
        b >>= 1;
    }
    return result;
}

#undef GF_HOST_DEVICE
