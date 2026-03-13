#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

__constant__ uint8_t d_gfexp[768];
__constant__ uint8_t d_gflog[256];

// Initialize tables in global memory
__global__ void init_tables_global(uint8_t* gf_exp, uint8_t* gf_log) {
    uint8_t prim_poly = 0x1D;
    uint8_t e = 1;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int log = 0; log < 255; ++log) {
            gf_log[e] = (uint8_t)log;
            gf_exp[log] = e;
            gf_exp[log + 255] = e;
            gf_exp[log + 510] = e;
            e = (e << 1) ^ ((e & 0x80) ? prim_poly : 0);
        }
        gf_log[0] = 0;
        gf_exp[0] = 1;
    }
}

__global__ void test_accum() {
    uint8_t A[4] = {70, 100, 49, 41};
    uint8_t B[4] = {196, 11, 79, 205};
    uint8_t accum = 0;

    for (int i = 0; i < 4; ++i) {
        uint8_t a = A[i];
        uint8_t b = B[i];
        uint8_t log_sum = d_gflog[a] + d_gflog[b];
        uint8_t product = d_gfexp[log_sum];
        printf("Step %d: %d * %d = %d (log: %d+%d=%d), accum: %d -> %d\n",
               i, a, b, product, d_gflog[a], d_gflog[b], log_sum, accum, accum ^ product);
        accum ^= product;
    }
    printf("Final accum = %d (expected: 243)\n", accum);
}

int main() {
    uint8_t *h_exp, *h_log, *d_exp, *d_log;
    h_exp = (uint8_t*)malloc(768);
    h_log = (uint8_t*)malloc(256);
    cudaMalloc(&d_exp, 768);
    cudaMalloc(&d_log, 256);

    // Initialize in global memory
    init_tables_global<<<1, 1>>>(d_exp, d_log);
    cudaDeviceSynchronize();

    // Copy to constant memory
    cudaMemcpyToSymbol(d_gfexp, d_exp, 768);
    cudaMemcpyToSymbol(d_gflog, d_log, 256);
    cudaDeviceSynchronize();

    // Verify constant memory
    cudaMemcpy(h_exp, d_exp, 768, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_log, d_log, 256, cudaMemcpyDeviceToHost);

    printf("Verification from global memory:\n");
    printf("gflog[70]=%d, gflog[196]=%d, gfexp[231]=%d\n",
           h_log[70], h_log[196], h_exp[231]);

    // Test accumulation using constant memory
    printf("\nTesting accumulation with constant memory:\n");
    test_accum<<<1, 1>>>();
    cudaDeviceSynchronize();

    free(h_exp); free(h_log);
    cudaFree(d_exp); cudaFree(d_log);
    return 0;
}
