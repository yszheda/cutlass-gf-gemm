/**
 * @file test_cutlass_traits.cu
 * @brief Verify CUTLASS numeric traits for gf28_t work correctly
 */

#include "cutlass_gf28_traits.h"
#include <stdio.h>
#include <assert.h>

// Test 1: Numeric traits compile and have correct constants
int test_numeric_traits_constants() {
    printf("  Test: numeric traits constants... ");
    using Traits = cutlass::NumericTraits<cutlass::gf28_t>;

    // GF(2^8) elements are 1 byte
    static_assert(sizeof(cutlass::gf28_t) == 1, "gf28_t must be 1 byte");
    static_assert(Traits::kCount == 1, "kCount should be 1");

    // Element should be trivially copyable
    static_assert(std::is_trivially_copyable<cutlass::gf28_t>::value,
                  "gf28_t must be trivially copyable");

    printf("PASS\n");
    return 0;
}

// Test 2: Custom MMA operator produces correct results
int test_mma_operator() {
    printf("  Test: MMA operator correctness... ");

    // Build GF tables
    uint8_t gf_exp[768], gf_log[256];
    const uint8_t prim_poly = 0x1D;
    uint8_t exp = 1;
    for (int log = 0; log < 255; ++log) {
        gf_log[exp] = static_cast<uint8_t>(log);
        gf_exp[log] = exp;
        gf_exp[log + 255] = exp;
        gf_exp[log + 510] = exp;
        exp = (exp << 1) ^ ((exp & 0x80) ? prim_poly : 0);
    }
    gf_log[0] = 0;
    gf_exp[0] = 1;

    using MmaOp = cutlass::Gf28MmaOperator;

    // Test: 100 * 50 + 30 in GF(2^8)
    cutlass::gf28_t a(100), b(50), c(30);
    auto result = MmaOp{}(a, b, c, gf_exp, gf_log);

    // Manual: 100 * 50 = gf_exp[gf_log[100] + gf_log[50]]
    uint8_t prod = gf_exp[gf_log[100] + gf_log[50]];
    uint8_t expected = prod ^ 30; // XOR for addition

    printf("result=%d, expected=%d ", (int)result, (int)expected);
    assert(result == expected);
    printf("PASS\n");
    return 0;
}

// Test 3: MMA operator with zero values
int test_mma_operator_zeros() {
    printf("  Test: MMA operator with zeros... ");

    uint8_t gf_exp[768], gf_log[256];
    const uint8_t prim_poly = 0x1D;
    uint8_t exp = 1;
    for (int log = 0; log < 255; ++log) {
        gf_log[exp] = static_cast<uint8_t>(log);
        gf_exp[log] = exp;
        gf_exp[log + 255] = exp;
        gf_exp[log + 510] = exp;
        exp = (exp << 1) ^ ((exp & 0x80) ? prim_poly : 0);
    }
    gf_log[0] = 0;
    gf_exp[0] = 1;

    using MmaOp = cutlass::Gf28MmaOperator;

    // 0 * anything = 0, so result = 0 ^ c = c
    auto r1 = MmaOp{}(cutlass::gf28_t(0), cutlass::gf28_t(50), cutlass::gf28_t(30), gf_exp, gf_log);
    assert(r1 == 30);

    auto r2 = MmaOp{}(cutlass::gf28_t(100), cutlass::gf28_t(0), cutlass::gf28_t(30), gf_exp, gf_log);
    assert(r2 == 30);

    // 0 * 0 + 0 = 0
    auto r3 = MmaOp{}(cutlass::gf28_t(0), cutlass::gf28_t(0), cutlass::gf28_t(0), gf_exp, gf_log);
    assert(r3 == 0);

    printf("PASS\n");
    return 0;
}

int main() {
    printf("=== CUTLASS GF(2^8) Traits Tests ===\n");
    int failed = 0;
    failed += test_numeric_traits_constants();
    failed += test_mma_operator();
    failed += test_mma_operator_zeros();
    if (failed == 0) printf("\nAll traits tests PASSED!\n");
    else printf("\n%d test(s) FAILED!\n", failed);
    return failed;
}
