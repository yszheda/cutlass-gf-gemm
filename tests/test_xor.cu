#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

__constant__ uint8_t d_gfexp[768];
__constant__ uint8_t d_gflog[256];

__global__ void init_kernel(uint8_t* exp, uint8_t* log) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint8_t prim_poly = 0x1D;
        uint8_t e = 1;
        for (int l = 0; l < 255; ++l) {
            log[e] = l;
            exp[l] = e;
            exp[l + 255] = e;
            exp[l + 510] = e;
            e = (e << 1) ^ ((e & 0x80) ? prim_poly : 0);
        }
        log[0] = 0;
        exp[0] = 1;
    }
}

__global__ void test_xor_kernel(uint8_t* result) {
    // Test 1: Direct XOR
    uint8_t test1 = 245 ^ 171 ^ 222 ^ 115;

    // Test 2: Step by step with XOR
    uint8_t accum = 0;
    accum ^= 245;
    accum ^= 171;
    accum ^= 222;
    accum ^= 115;

    // Test 3: Using GF tables (product from log/exp)
    uint8_t test3 = 0;
    test3 ^= d_gfexp[d_gflog[70] + d_gflog[196]];  // 245
    test3 ^= d_gfexp[d_gflog[100] + d_gflog[11]];  // 171
    test3 ^= d_gfexp[d_gflog[49] + d_gflog[79]];   // 222
    test3 ^= d_gfexp[d_gflog[41] + d_gflog[205]];  // 115

    // Test 4: In a loop
    uint8_t products[4] = {
        d_gfexp[d_gflog[70] + d_gflog[196]],
        d_gfexp[d_gflog[100] + d_gflog[11]],
        d_gfexp[d_gflog[49] + d_gflog[79]],
        d_gfexp[d_gflog[41] + d_gflog[205]]
    };
    uint8_t test4 = 0;
    for (int i = 0; i < 4; ++i) {
        test4 ^= products[i];
    }

    result[0] = test1;
    result[1] = accum;
    result[2] = test3;
    result[3] = test4;

    printf("Direct XOR: %d\n", test1);
    printf("Step by step: %d\n", accum);
    printf("Using GF tables: %d\n", test3);
    printf("In a loop: %d\n", test4);
}

int main() {
    uint8_t *d_exp, *d_log, *d_result;
    cudaMalloc(&d_exp, 768);
    cudaMalloc(&d_log, 256);
    cudaMalloc(&d_result, 4);

    init_kernel<<<1,1>>>(d_exp, d_log);
    cudaDeviceSynchronize();

    cudaMemcpyToSymbol(d_gfexp, d_exp, 768);
    cudaMemcpyToSymbol(d_gflog, d_log, 256);
    cudaDeviceSynchronize();

    test_xor_kernel<<<1,1>>>(d_result);
    cudaDeviceSynchronize();

    uint8_t h_result[4];
    cudaMemcpy(h_result, d_result, 4, cudaMemcpyDeviceToHost);

    printf("\nResults:\n");
    printf("  Direct XOR: %d (expected: 243)\n", h_result[0]);
    printf("  Step by step: %d (expected: 243)\n", h_result[1]);
    printf("  Using GF tables: %d (expected: 243)\n", h_result[2]);
    printf("  In a loop: %d (expected: 243)\n", h_result[3]);

    cudaFree(d_exp); cudaFree(d_log); cudaFree(d_result);
    return 0;
}
