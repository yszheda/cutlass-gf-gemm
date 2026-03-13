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

__global__ void gf_mul_kernel(uint8_t* result) {
    // Test: compute 70 * 196 in GF(2^8) using constant memory tables
    uint8_t a = 70;
    uint8_t b = 196;
    uint8_t log_sum = d_gflog[a] + d_gflog[b];
    uint8_t product = d_gfexp[log_sum];
    result[0] = product;

    // Also test the accumulation for C[0,0]
    uint8_t A[4] = {70, 100, 49, 41};
    uint8_t B[4] = {196, 11, 79, 205};
    uint8_t accum = 0;
    for (int i = 0; i < 4; ++i) {
        if (A[i] != 0 && B[i] != 0) {
            log_sum = d_gflog[A[i]] + d_gflog[B[i]];
            accum ^= d_gfexp[log_sum];
        }
    }
    result[1] = accum;
}

int main() {
    uint8_t *h_exp, *h_log;
    h_exp = (uint8_t*)malloc(768);
    h_log = (uint8_t*)malloc(256);

    uint8_t *d_exp, *d_log, *d_result;
    cudaMalloc(&d_exp, 768);
    cudaMalloc(&d_log, 256);
    cudaMalloc(&d_result, 2);

    // Initialize tables
    init_kernel<<<1,1>>>(d_exp, d_log);
    cudaDeviceSynchronize();

    // Copy to constant memory
    cudaMemcpyToSymbol(d_gfexp, d_exp, 768);
    cudaMemcpyToSymbol(d_gflog, d_log, 256);
    cudaDeviceSynchronize();

    // Run multiplication kernel
    gf_mul_kernel<<<1,1>>>(d_result);
    cudaDeviceSynchronize();

    // Get results
    uint8_t h_result[2];
    cudaMemcpy(h_result, d_result, 2, cudaMemcpyDeviceToHost);

    printf("GPU GF multiplication test:\n");
    printf("70 * 196 = %d (expected: 245)\n", h_result[0]);
    printf("C[0,0] accumulation = %d (expected: 243)\n", h_result[1]);

    // Verify tables were copied correctly
    cudaMemcpy(h_exp, d_exp, 768, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_log, d_log, 256, cudaMemcpyDeviceToHost);

    printf("\nVerification of tables:\n");
    printf("gflog[70] = %d, gflog[196] = %d\n", h_log[70], h_log[196]);
    printf("gfexp[231] = %d\n", h_exp[231]);

    free(h_exp); free(h_log);
    cudaFree(d_exp); cudaFree(d_log); cudaFree(d_result);
    return 0;
}
