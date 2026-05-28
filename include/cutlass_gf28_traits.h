/**
 * @file cutlass_gf28_traits.h
 * @brief CUTLASS numeric traits and custom MMA operator for GF(2^8)
 *
 * Bridges cutlass::gf28_t to CUTLASS's template system by defining:
 * - NumericTraits<gf28_t>: size, alignment, compute class
 * - Gf28MmaOperator: custom multiply-accumulate (gf_mul ^ acc)
 */

#pragma once

#include "gf_ops.h"
#include <cutlass/numeric_types.h>
#include <cutlass/array.h>
#include <cutlass/gemm/gemm.h>
#include <stdint.h>

namespace cutlass {

// ============================================================================
//                     GF(2^8) Numeric Traits
// ============================================================================

/// Numeric traits for GF(2^8) element type
template <>
struct NumericTraits<gf28_t> {
    using Element = gf28_t;

    /// Storage type is uint8_t
    using StorageType = uint8_t;

    /// Breakdown into component types (GF(2^8) is a single element)
    using BreakdownType = Element;

    /// Number of breakdown elements
    static constexpr int kCount = 1;

    /// Alignment in bytes
    static constexpr int kAlignment = 1;

    /// Element size in bytes
    static constexpr int kElementSize = sizeof(Element);

    /// Is integer type (GF elements are stored as integers)
    static constexpr bool kIsInteger = true;

    /// Is real number (not complex)
    static constexpr bool kIsReal = false;

    /// Is floating point
    static constexpr bool kIsFloat = false;

    /// Is signed
    static constexpr bool kIsSigned = false;

    /// Compute class — treated as integer for CUTLASS
    static constexpr cutlass::FloatRoundStyle kRoundStyle = cutlass::FloatRoundStyle::round_to_zero;

    /// Minimum value
    static GF_HOST_DEVICE Element min() { return Element(0); }

    /// Maximum value
    static GF_HOST_DEVICE Element max() { return Element(255); }

    /// Zero
    static GF_HOST_DEVICE Element zero() { return Element(0); }

    /// One (multiplicative identity)
    static GF_HOST_DEVICE Element one() { return Element(1); }
};

// ============================================================================
//                     GF(2^8) Custom MMA Operator
// ============================================================================

/// Custom Matrix Multiply-Accumulate operator for GF(2^8)
///
/// Computes: result = a * b + c
/// Where * is GF(2^8) multiplication and + is XOR
struct Gf28MmaOperator {
    using ElementA = gf28_t;
    using ElementB = gf28_t;
    using ElementC = gf28_t;
    using ElementD = gf28_t;

    /// Shape of the MMA operation (1x1x1 element-wise)
    struct Shape {
        static constexpr int kM = 1;
        static constexpr int kN = 1;
        static constexpr int kK = 1;
    };

    GF_HOST_DEVICE
    ElementD operator()(
        ElementA const& a,
        ElementB const& b,
        ElementC const& c,
        const uint8_t* gf_exp,
        const uint8_t* gf_log
    ) const {
        uint8_t va = static_cast<uint8_t>(a);
        uint8_t vb = static_cast<uint8_t>(b);
        uint8_t vc = static_cast<uint8_t>(c);

        // GF multiplication: a * b
        uint8_t product;
        if (va == 0 || vb == 0) {
            product = 0;
        } else {
            int log_sum = gf_log[va] + gf_log[vb];
            product = gf_exp[log_sum];
        }

        // GF addition (XOR) with accumulator
        return ElementD(product ^ vc);
    }
};

} // namespace cutlass
