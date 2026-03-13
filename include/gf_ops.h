/**
 * @file gf_ops.h
 * @brief Galois Field arithmetic operations for CUTLASS
 *
 * This file defines the numeric type traits and operations required
 * by CUTLASS for Galois Field arithmetic in GF(2^8).
 */

#pragma once

#include <stdint.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/array.h>

namespace cutlass {

// ============================================================================
//                     GF(2^8) Element Type Definition
// ============================================================================

/// Galois Field element - stored as uint8_t
struct gf28_t {
    uint8_t storage;

    CUTLASS_HOST_DEVICE
    gf28_t() : storage(0) {}

    CUTLASS_HOST_DEVICE
    gf28_t(uint8_t value) : storage(value) {}

    CUTLASS_HOST_DEVICE
    gf28_t(int value) : storage(static_cast<uint8_t>(value)) {}

    CUTLASS_HOST_DEVICE
    operator uint8_t() const { return storage; }

    CUTLASS_HOST_DEVICE
    operator int() const { return static_cast<int>(storage); }

    CUTLASS_HOST_DEVICE
    gf28_t& operator=(uint8_t value) {
        storage = value;
        return *this;
    }

    CUTLASS_HOST_DEVICE
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

// ============================================================================
//                     NumericCast Specialization
// ============================================================================

template <>
struct NumericCast<gf28_t, gf28_t> {
    CUTLASS_HOST_DEVICE
    static gf28_t transform(gf28_t source) {
        return source;
    }
};

template <>
struct NumericCast<gf28_t, uint8_t> {
    CUTLASS_HOST_DEVICE
    static gf28_t transform(uint8_t source) {
        return gf28_t(source);
    }
};

template <>
struct NumericCast<uint8_t, gf28_t> {
    CUTLASS_HOST_DEVICE
    static uint8_t transform(gf28_t source) {
        return source.storage;
    }
};

template <>
struct NumericCast<gf28_t, int> {
    CUTLASS_HOST_DEVICE
    static gf28_t transform(int source) {
        return gf28_t(static_cast<uint8_t>(source));
    }
};

template <>
struct NumericCast<int, gf28_t> {
    CUTLASS_HOST_DEVICE
    static int transform(gf28_t source) {
        return static_cast<int>(source.storage);
    }
};

// ============================================================================
//                     NumericArrayConverter Specializations
// ============================================================================

/// Converter for arrays of GF(2^8) elements
template <int N>
struct NumericArrayConverter<gf28_t, gf28_t, N> {
    CUTLASS_HOST_DEVICE
    Array<gf28_t, N> operator()(Array<gf28_t, N> const& source) {
        return source;
    }
};

template <int N>
struct NumericArrayConverter<gf28_t, uint8_t, N> {
    CUTLASS_HOST_DEVICE
    Array<gf28_t, N> operator()(Array<uint8_t, N> const& source) {
        Array<gf28_t, N> result;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            result[i] = gf28_t(source[i]);
        }
        return result;
    }
};

template <int N>
struct NumericArrayConverter<uint8_t, gf28_t, N> {
    CUTLASS_HOST_DEVICE
    Array<uint8_t, N> operator()(Array<gf28_t, N> const& source) {
        Array<uint8_t, N> result;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            result[i] = source[i].storage;
        }
        return result;
    }
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
CUTLASS_HOST_DEVICE inline uint8_t gf_mul(uint8_t a, uint8_t b,
                                           const uint8_t* gf_exp,
                                           const uint8_t* gf_log) {
    if (a == 0 || b == 0) return 0;
    return gf_exp[gf_log[a] + gf_log[b]];
}

/**
 * @brief GF(2^8) addition (XOR)
 */
CUTLASS_HOST_DEVICE inline uint8_t gf_add(uint8_t a, uint8_t b) {
    return a ^ b;
}
