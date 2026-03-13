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

// Debug version with printf
__global__ void gf_gemm_debug(const uint8_t* A, const uint8_t* B, uint8_t* C,
                              int m, int n, int k) {
    constexpr int TILE_SIZE = 4;

    __shared__ uint8_t As[TILE_SIZE][TILE_SIZE];
    __shared__ uint8_t Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Only for thread (0,0) in block (0,0) - computing C[0,0]
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("Computing C[0,0]: A[0,:] dot B[:,0]\n");
        printf("A[0,:] = [%d, %d, %d, %d]\n", A[0], A[1], A[2], A[3]);
        printf("B[:,0] = [%d, %d, %d, %d]\n", B[0], B[4], B[8], B[12]);
    }

    uint8_t accum = 0;
    int num_tiles = 1;  // k=4, TILE_SIZE=4

    for (int t = 0; t < num_tiles; ++t) {
        int tiled_col_a = t * TILE_SIZE + threadIdx.x;
        int tiled_row_b = t * TILE_SIZE + threadIdx.y;

        // Load tile of A: each thread loads one element
        if (row < m && tiled_col_a < k) {
            As[threadIdx.y][threadIdx.x] = A[row * k + tiled_col_a];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        // Load tile of B: each thread loads one element
        if (tiled_row_b < k && col < n) {
            Bs[threadIdx.y][threadIdx.x] = B[tiled_row_b * n + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
            printf("After load: As[0,:] = [%d, %d, %d, %d]\n",
                   As[0][0], As[0][1], As[0][2], As[0][3]);
            printf("After load: Bs[:,0] = [%d, %d, %d, %d]\n",
                   Bs[0][0], Bs[1][0], Bs[2][0], Bs[3][0]);
        }

        // Compute partial dot product: row of As dot col of Bs
        for (int i = 0; i < TILE_SIZE; ++i) {
            uint8_t a = As[threadIdx.y][i];
            uint8_t b = Bs[i][threadIdx.x];

            if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
                printf("  i=%d: a=%d, b=%d, product=%d\n",
                       i, a, b, (a && b) ? d_gfexp[d_gflog[a] + d_gflog[b]] : 0);
            }

            if (a != 0 && b != 0) {
                uint8_t log_sum = d_gflog[a] + d_gflog[b];
                accum ^= d_gfexp[log_sum];
            }
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = accum;
        if (row == 0 && col == 0) {
            printf("C[0,0] = %d (expected: 243)\n", accum);
        }
    }
}

int main() {
    int m = 4, k = 4, n = 4;

    uint8_t A[16] = {70, 100, 49, 41, 100, 134, 237, 156, 215, 31, 194, 7, 37, 72, 32, 162};
    uint8_t B[16] = {196, 168, 90, 235, 11, 32, 65, 73, 79, 139, 241, 248, 205, 48, 241, 19};

    // GPU
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

    gf_gemm_debug<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();

    uint8_t h_C[16];
    cudaMemcpy(h_C, d_C, 16, cudaMemcpyDeviceToHost);

    printf("\nFinal GPU Result:\n");
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
