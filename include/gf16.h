/**
 * @file gf16.h
 * @brief Galois Field GF(2^8) lookup tables and utilities
 *
 * This file contains precomputed logarithm and exponentiation tables
 * for efficient multiplication in GF(2^8) with primitive polynomial 0x11D (0435 octal).
 */

#pragma once

#include <stdint.h>

#ifdef __CUDA_ARCH__
#define GF_HOST_DEVICE __host__ __device__
#else
#define GF_HOST_DEVICE
#endif

namespace gf {

// Primitive polynomial for GF(2^8): x^8 + x^4 + x^3 + x^2 + 1 = 0x11D
constexpr uint8_t PRIM_POLY = 0x1D;
constexpr int W = 8;
constexpr int FIELD_SIZE = 256;
constexpr int FIELD_MAX = 255;

// Extended tables for efficient multiplication without bounds checking
// gfexp has size 768 = 255 * 3 to handle overflow in index calculations
constexpr int GFEXP_TABLE_SIZE = 768;
constexpr int GFLOG_TABLE_SIZE = 256;

/**
 * @brief Precomputed GF(2^8) exponentiation table
 *
 * gfexp[i] = alpha^i where alpha is the primitive element (0x03)
 * Table is extended: gfexp[i + 255] = gfexp[i] for i in [0, 511]
 * This allows multiplication without modulo operation on indices
 */
GF_HOST_DEVICE constexpr uint8_t get_gfexp(int index) {
    // Generate exp table at compile time using primitive polynomial
    // This is a simplified version - actual table will be generated at runtime
    // or stored in constant memory
    uint8_t exp = 1;
    uint8_t prim_poly = PRIM_POLY;

    // Handle extended indices
    index = index % FIELD_MAX;
    if (index < 0) index += FIELD_MAX;

    for (int i = 0; i < index; ++i) {
        exp = (exp << 1) ^ ((exp & 0x80) ? prim_poly : 0);
    }
    return exp;
}

/**
 * @brief Precomputed GF(2^8) logarithm table
 *
 * gflog[alpha^i] = i, gflog[0] = 0 (undefined, but set to 0 for safety)
 */
GF_HOST_DEVICE constexpr uint8_t get_gflog(uint8_t value) {
    if (value == 0) return 0;

    uint8_t exp = 1;
    uint8_t prim_poly = PRIM_POLY;

    for (int log = 0; log < FIELD_MAX; ++log) {
        if (exp == value) return (uint8_t)log;
        exp = (exp << 1) ^ ((exp & 0x80) ? prim_poly : 0);
    }
    return 0; // Should never reach here
}

/**
 * @brief Initialize GF lookup tables in device memory
 *
 * @param gfexp Device pointer to exponentiation table (size >= GFEXP_TABLE_SIZE)
 * @param gflog Device pointer to logarithm table (size >= GFLOG_TABLE_SIZE)
 */
void init_gf_tables_device(uint8_t* gfexp, uint8_t* gflog);

/**
 * @brief GF(2^8) addition (XOR)
 */
GF_HOST_DEVICE inline uint8_t add(uint8_t a, uint8_t b) {
    return a ^ b;
}

/**
 * @brief GF(2^8) subtraction (same as addition)
 */
GF_HOST_DEVICE inline uint8_t sub(uint8_t a, uint8_t b) {
    return a ^ b;
}

/**
 * @brief GF(2^8) multiplication using log/exp tables
 *
 * @param a First operand
 * @param b Second operand
 * @param gfexp Exponentiation table (can be nullptr for slow path)
 * @param gflog Logarithm table (can be nullptr for slow path)
 * @return a * b in GF(2^8)
 */
GF_HOST_DEVICE inline uint8_t mul(uint8_t a, uint8_t b,
                                   const uint8_t* gfexp = nullptr,
                                   const uint8_t* gflog = nullptr) {
    if (a == 0 || b == 0) return 0;

    if (gfexp != nullptr && gflog != nullptr) {
        // Fast path using lookup tables
        int sum_log = gflog[a] + gflog[b];
        return gfexp[sum_log];
    } else {
        // Slow path using bit manipulation (Russian peasant algorithm)
        uint8_t result = 0;
        uint8_t prim_poly = PRIM_POLY;

        while (b) {
            if (b & 1) {
                result ^= a;
            }
            a = (a << 1) ^ ((a & 0x80) ? prim_poly : 0);
            b >>= 1;
        }
        return result;
    }
}

/**
 * @brief GF(2^8) division using log/exp tables
 *
 * @param a Dividend
 * @param b Divisor (must not be 0)
 * @param gfexp Exponentiation table
 * @param gflog Logarithm table
 * @return a / b in GF(2^8), or 0 if b == 0
 */
GF_HOST_DEVICE inline uint8_t div(uint8_t a, uint8_t b,
                                   const uint8_t* gfexp,
                                   const uint8_t* gflog) {
    if (a == 0) return 0;
    if (b == 0) return 0; // Undefined, return 0 for safety

    int diff_log = gflog[a] + FIELD_MAX - gflog[b];
    return gfexp[diff_log];
}

/**
 * @brief GF(2^8) exponentiation
 *
 * @param a Base
 * @param power Exponent
 * @param gfexp Exponentiation table
 * @param gflog Logarithm table
 * @return a^power in GF(2^8)
 */
GF_HOST_DEVICE inline uint8_t pow(uint8_t a, uint8_t power,
                                   const uint8_t* gfexp,
                                   const uint8_t* gflog) {
    if (power == 0) return 1;
    if (a == 0) return 0;

    int pow_log = (gflog[a] * power) % FIELD_MAX;
    return gfexp[pow_log];
}

/**
 * @brief GF(2^8) multiplicative inverse
 *
 * @param a Element to invert (must not be 0)
 * @param gfexp Exponentiation table
 * @param gflog Logarithm table
 * @return a^(-1) in GF(2^8), or 0 if a == 0
 */
GF_HOST_DEVICE inline uint8_t inverse(uint8_t a,
                                       const uint8_t* gfexp,
                                       const uint8_t* gflog) {
    if (a == 0) return 0;
    return gfexp[FIELD_MAX - gflog[a]];
}

} // namespace gf

#undef GF_HOST_DEVICE
