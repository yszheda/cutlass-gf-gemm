#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

__constant__ uint8_t d_gfexp[768];
__constant__ uint8_t d_gflog[256];

__global__ void init_and_verify() {
    uint8_t prim_poly = 0x1D;
    uint8_t e = 1;

    // Only one thread does initialization
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int log = 0; log < 255; ++log) {
            d_gflog[e] = log;
            d_gfexp[log] = e;
            d_gfexp[log + 255] = e;
            d_gfexp[log + 510] = e;
            e = (e << 1) ^ ((e & 0x80) ? prim_poly : 0);
        }
        d_gflog[0] = 0;
        d_gfexp[0] = 1;
    }

    __syncthreads();

    // All threads verify
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 16) {
        printf("Thread %d: gflog[%d]=%d, gfexp[%d]=%d\n",
               idx, idx, d_gflog[idx], idx, d_gfexp[idx]);
    }

    // Test multiplication
    if (idx == 0) {
        uint8_t a = 70, b = 196;
        uint8_t log_sum = d_gflog[a] + d_gflog[b];
        uint8_t product = d_gfexp[log_sum];
        printf("\n70 * 196 = %d (log: %d + %d = %d)\n",
               product, d_gflog[a], d_gflog[b], log_sum);

        // Test accumulation
        uint8_t A[4] = {70, 100, 49, 41};
        uint8_t B[4] = {196, 11, 79, 205};
        uint8_t accum = 0;
        for (int i = 0; i < 4; ++i) {
            uint8_t p = d_gfexp[d_gflog[A[i]] + d_gflog[B[i]]];
            printf("  Step %d: %d * %d = %d, accum = %d ^ %d = %d\n",
                   i, A[i], B[i], p, accum ^ p, p, accum);
            accum ^= p;
        }
        printf("Final accum = %d (expected: 243)\n", accum);
    }
}

int main() {
    dim3 block(256);
    dim3 grid(1);

    init_and_verify<<<grid, block>>>();
    cudaDeviceSynchronize();

    return 0;
}
