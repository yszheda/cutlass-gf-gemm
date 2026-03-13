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

// Full tiled GEMM kernel for 4x4 matrix
__global__ void gf_gemm_4x4(const uint8_t* A, const uint8_t* B, uint8_t* C,
                            int m, int n, int k, int lda, int ldb, int ldc) {
    constexpr int TILE_SIZE = 4;

    __shared__ uint8_t As[TILE_SIZE][TILE_SIZE];
    __shared__ uint8_t Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    uint8_t accum = 0;

    int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        int tiled_col_a = t * TILE_SIZE + threadIdx.x;
        int tiled_row_b = t * TILE_SIZE + threadIdx.y;

        // Load tile of A
        if (row < m && tiled_col_a < k) {
            As[threadIdx.y][threadIdx.x] = A[row * lda + tiled_col_a];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        // Load tile of B
        if (tiled_row_b < k && col < n) {
            Bs[threadIdx.y][threadIdx.x] = B[tiled_row_b * ldb + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute partial dot product
        for (int i = 0; i < TILE_SIZE; ++i) {
            uint8_t a = As[threadIdx.y][i];
            uint8_t b = Bs[i][threadIdx.x];

            if (a != 0 && b != 0) {
                uint8_t log_sum = d_gflog[a] + d_gflog[b];
                accum ^= d_gfexp[log_sum];
            }
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * ldc + col] = accum;
    }
}

int main() {
    int m = 4, k = 4, n = 4;

    uint8_t A[16] = {70, 100, 49, 41, 100, 134, 237, 156, 215, 31, 194, 7, 37, 72, 32, 162};
    uint8_t B[16] = {196, 168, 90, 235, 11, 32, 65, 73, 79, 139, 241, 248, 205, 48, 241, 19};

    // CPU reference
    uint8_t gf_exp[768];
    uint8_t gf_log[256];
    uint8_t prim_poly = 0x1D;
    uint8_t e = 1;
    for (int l = 0; l < 255; ++l) {
        gf_log[e] = l;
        gf_exp[l] = e;
        gf_exp[l + 255] = e;
        gf_exp[l + 510] = e;
        e = (e << 1) ^ ((e & 0x80) ? prim_poly : 0);
    }
    gf_log[0] = 0;
    gf_exp[0] = 1;

    uint8_t C_cpu[16] = {0};
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            uint8_t sum = 0;
            for (int l = 0; l < k; ++l) {
                uint8_t a = A[i * k + l];
                uint8_t b = B[l * n + j];
                if (a != 0 && b != 0) {
                    sum ^= gf_exp[gf_log[a] + gf_log[b]];
                }
            }
            C_cpu[i * n + j] = sum;
        }
    }

    printf("CPU Reference Result:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%3d ", C_cpu[i * n + j]);
        }
        printf("\n");
    }

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
    dim3 grid((n + 3) / 4, (m + 3) / 4);

    gf_gemm_4x4<<<grid, block>>>(d_A, d_B, d_C, m, n, k, k, n, n);
    cudaDeviceSynchronize();

    uint8_t h_C[16];
    cudaMemcpy(h_C, d_C, 16, cudaMemcpyDeviceToHost);

    printf("\nGPU Result:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%3d ", h_C[i * n + j]);
        }
        printf("\n");
    }

    // Verify
    int errors = 0;
    for (int i = 0; i < 16; ++i) {
        if (h_C[i] != C_cpu[i]) {
            errors++;
        }
    }
    printf("\nErrors: %d / 16\n", errors);

    cudaFree(d_exp); cudaFree(d_log);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return errors > 0 ? 1 : 0;
}
