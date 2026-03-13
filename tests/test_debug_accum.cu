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
    uint8_t A[4] = {70, 100, 49, 41};
    uint8_t B[4] = {196, 11, 79, 205};
    uint8_t accum = 0;

    for (int i = 0; i < 4; ++i) {
        uint8_t a = A[i];
        uint8_t b = B[i];
        if (a != 0 && b != 0) {
            uint8_t log_a = d_gflog[a];
            uint8_t log_b = d_gflog[b];
            uint16_t log_sum = (uint16_t)log_a + (uint16_t)log_b;
            uint8_t product = d_gfexp[log_sum];
            printf("Step %d: %d * %d = %d (log: %d + %d = %d), accum: %d -> %d\n",
                   i, a, b, product, log_a, log_b, log_sum, accum, accum ^ product);
            accum ^= product;
        }
    }
    result[0] = accum;
}

int main() {
    uint8_t *h_exp, *h_log;
    h_exp = (uint8_t*)malloc(768);
    h_log = (uint8_t*)malloc(256);

    uint8_t *d_exp, *d_log, *d_result;
    cudaMalloc(&d_exp, 768);
    cudaMalloc(&d_log, 256);
    cudaMalloc(&d_result, 1);

    init_kernel<<<1,1>>>(d_exp, d_log);
    cudaDeviceSynchronize();

    cudaMemcpyToSymbol(d_gfexp, d_exp, 768);
    cudaMemcpyToSymbol(d_gflog, d_log, 256);
    cudaDeviceSynchronize();

    gf_mul_kernel<<<1,1>>>(d_result);
    cudaDeviceSynchronize();

    uint8_t h_result;
    cudaMemcpy(&h_result, d_result, 1, cudaMemcpyDeviceToHost);
    printf("\nGPU result: %d\n", h_result);

    // CPU verification
    printf("\nCPU verification:\n");
    uint8_t A[4] = {70, 100, 49, 41};
    uint8_t B[4] = {196, 11, 79, 205};
    uint8_t cpu_accum = 0;
    for (int i = 0; i < 4; ++i) {
        uint8_t product = h_exp[h_log[A[i]] + h_log[B[i]]];
        printf("Step %d: %d * %d = %d, accum: %d -> %d\n",
               i, A[i], B[i], product, cpu_accum, cpu_accum ^ product);
        cpu_accum ^= product;
    }
    printf("\nCPU result: %d\n", cpu_accum);

    free(h_exp); free(h_log);
    cudaFree(d_exp); cudaFree(d_log); cudaFree(d_result);
    return 0;
}
