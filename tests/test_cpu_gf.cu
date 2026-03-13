#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int main() {
    // Build GF tables
    uint8_t gf_exp[768];
    uint8_t gf_log[256];
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

    // Test matrices (same as example)
    int m = 4, k = 4, n = 4;
    uint8_t A[16] = {70, 100, 49, 41, 100, 134, 237, 156, 215, 31, 194, 7, 37, 72, 32, 162};
    uint8_t B[16] = {196, 168, 90, 235, 11, 32, 65, 73, 79, 139, 241, 248, 205, 48, 241, 19};
    uint8_t C[16];

    memset(C, 0, 16);

    // Compute C = A * B (row-major)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            uint8_t sum = 0;
            for (int l = 0; l < k; ++l) {
                uint8_t a = A[i * k + l];
                uint8_t b = B[l * n + j];
                if (a != 0 && b != 0) {
                    uint8_t product = gf_exp[gf_log[a] + gf_log[b]];
                    sum ^= product;
                }
            }
            C[i * n + j] = sum;
        }
    }

    printf("CPU Result C = A * B:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%3d ", C[i * n + j]);
        }
        printf("\n");
    }

    // Manual test for C[0,0]
    printf("\nManual calculation for C[0,0]:\n");
    printf("A[0,:] = [%d, %d, %d, %d]\n", A[0], A[1], A[2], A[3]);
    printf("B[:,0] = [%d, %d, %d, %d]\n", B[0], B[4], B[8], B[12]);

    uint8_t c00 = 0;
    for (int l = 0; l < 4; ++l) {
        uint8_t a = A[0 * 4 + l];
        uint8_t b = B[l * 4 + 0];
        uint8_t p = gf_exp[gf_log[a] + gf_log[b]];
        printf("  %d * %d = %d (log: %d + %d = %d)\n", a, b, p, gf_log[a], gf_log[b], gf_log[a] + gf_log[b]);
        c00 ^= p;
    }
    printf("C[0,0] = %d\n", c00);

    return 0;
}
