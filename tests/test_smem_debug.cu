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

// Debug version - check shared memory layout
__global__ void gf_gemm_debug2(const uint8_t* A, const uint8_t* B, uint8_t* C,
                               int m, int n, int k) {
    constexpr int TILE_SIZE = 4;

    __shared__ uint8_t As[TILE_SIZE][TILE_SIZE];
    __shared__ uint8_t Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // For block (0,0), all threads print their loaded values
    if (blockIdx.x == 0 && blockIdx.y == 0) {
        printf("Thread (%d,%d): row=%d, col=%d\n", threadIdx.x, threadIdx.y, row, col);
    }

    uint8_t accum = 0;

    // Load phase
    As[threadIdx.y][threadIdx.x] = (row < m && threadIdx.x < k) ? A[row * k + threadIdx.x] : 0;
    Bs[threadIdx.y][threadIdx.x] = (threadIdx.y < k && col < n) ? B[threadIdx.y * n + col] : 0;

    __syncthreads();

    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("\nShared memory As:\n");
        for (int ty = 0; ty < 4; ++ty) {
            printf("  ");
            for (int tx = 0; tx < 4; ++tx) {
                printf("%3d ", As[ty][tx]);
            }
            printf("\n");
        }
        printf("\nShared memory Bs:\n");
        for (int ty = 0; ty < 4; ++ty) {
            printf("  ");
            for (int tx = 0; tx < 4; ++tx) {
                printf("%3d ", Bs[ty][tx]);
            }
            printf("\n");
        }
    }

    __syncthreads();

    // Compute phase for C[0,0]
    if (row == 0 && col == 0) {
        printf("\nThread (0,0) computing C[0,0]:\n");
        for (int i = 0; i < TILE_SIZE; ++i) {
            uint8_t a = As[0][i];  // Row 0 of As
            uint8_t b = Bs[i][0];  // Col 0 of Bs
            printf("  i=%d: As[0][%d]=%d, Bs[%d][0]=%d", i, i, a, i, b);
            if (a != 0 && b != 0) {
                uint8_t p = d_gfexp[d_gflog[a] + d_gflog[b]];
                printf(" -> product=%d, accum=%d -> %d\n", p, accum, accum ^ p);
                accum ^= p;
            } else {
                printf(" -> skip (zero)\n");
            }
        }
        C[0] = accum;
        printf("Final C[0,0] = %d (expected: 243)\n", accum);
    }
}

int main() {
    int m = 4, k = 4, n = 4;

    uint8_t A[16] = {70, 100, 49, 41, 100, 134, 237, 156, 215, 31, 194, 7, 37, 72, 32, 162};
    uint8_t B[16] = {196, 168, 90, 235, 11, 32, 65, 73, 79, 139, 241, 248, 205, 48, 241, 19};

    uint8_t *d_exp, *d_log, *d_A, *d_B, *d_C;
    cudaMalloc(&d_exp, 768);
    cudaMalloc(&d_log, 256);
    cudaMalloc(&d_A, 16);
    cudaMalloc(&d_B, 16);
    cudaMalloc(&d_C, 16);

    init_kernel<<<1,1>>>(d_exp, d_log);
    cudaDeviceSynchronize();

    cudaMemcpyToSymbol(d_gfexp, d_exp, 768);
    cudaMemcpyToSymbol(d_gflog, d_log, 256);
    cudaDeviceSynchronize();

    cudaMemcpy(d_A, A, 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, 16, cudaMemcpyHostToDevice);

    dim3 block(4, 4);
    dim3 grid(1, 1);

    gf_gemm_debug2<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();

    uint8_t h_C[16];
    cudaMemcpy(h_C, d_C, 16, cudaMemcpyDeviceToHost);

    printf("\n\nFinal GPU Result:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%3d ", h_C[i * n + j]);
        }
        printf("\n");
    }

    cudaFree(d_exp); cudaFree(d_log);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
