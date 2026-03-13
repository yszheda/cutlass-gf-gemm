# CUTLASS GF Gemm 实现技术文档

## 任务概述

实现基于 CUTLASS 的 GF(2^8) 伽罗瓦域矩阵乘法，用于 Reed-Solomon 纠错编码加速。

## 技术背景

### GF(2^8) 域算术

GF(2^8) 是包含 256 个元素的有限域，在 Reed-Solomon 编码中广泛应用。核心运算：

- **加法**: XOR 运算 (`a ^ b`)
- **乘法**: 通过对数表转换 `a * b = exp(log(a) + log(b)) mod primitive_poly`
- **本原多项式**: `x^8 + x^4 + x^3 + x^2 + 1 = 0x1D`

### 对数/指数表优化

直接计算 GF 乘法需要逐位运算，效率低。使用查找表优化：

```
gf_log[x]:  x -> log_a(x)    其中 a 是本原元
gf_exp[x]: log_a(x) -> x

乘法: a * b = gf_exp[gf_log[a] + gf_log[b]]
```

**关键问题**: 两个 log 值相加范围为 [0, 508] (254+254)，超过 uint8_t 的 255。

**解决方案**: 扩展 gf_exp 表至 768 字节 (255*3)，避免模运算开销。

## 实现方法

### 1. 数据结构定义

```cpp
// gf_ops.h
struct gf28_t {
    uint8_t storage;
    // 构造函数和类型转换
};

struct GF28Arithmetic {
    static constexpr uint8_t kPrimitivePolynomial = 0x1D;
    static constexpr int kLogTableSize = 256;
    static constexpr int kExpTableSize = 768;  // 255 * 3
};
```

### 2. 查找表初始化

```cpp
__global__ void init_gf_tables_kernel(uint8_t* gf_exp, uint8_t* gf_log) {
    const uint8_t prim_poly = 0x1D;
    uint8_t exp = 1;
    for (int log = 0; log < 255; ++log) {
        gf_log[exp] = static_cast<uint8_t>(log);
        gf_exp[log] = exp;
        gf_exp[log + 255] = exp;      // 溢出区
        gf_exp[log + 510] = exp;      // 二次溢出区
        exp = (exp << 1) ^ ((exp & 0x80) ? prim_poly : 0);
    }
    gf_log[0] = 0;
    gf_exp[0] = 1;
}
```

### 3. GEMM 内核实现

```cpp
__global__ void gf_gemm_kernel_simple(const uint8_t* A, const uint8_t* B, uint8_t* C,
                                       int m, int n, int k, int lda, int ldb, int ldc) {
    constexpr int TILE_SIZE = 16;
    __shared__ uint8_t As[TILE_SIZE][TILE_SIZE];
    __shared__ uint8_t Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    uint8_t accum = 0;

    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 加载 tile 到共享内存
        if (row < m && tiled_col_a < k)
            As[threadIdx.y][threadIdx.x] = A[row * lda + tiled_col_a];
        if (tiled_row_b < k && col < n)
            Bs[threadIdx.y][threadIdx.x] = B[tiled_row_b * ldb + col];

        __syncthreads();

        // 计算点积
        for (int i = 0; i < TILE_SIZE; ++i) {
            uint8_t a = As[threadIdx.y][i];
            uint8_t b = Bs[i][threadIdx.x];
            if (a != 0 && b != 0) {
                // 关键：使用 int 避免溢出
                int log_sum = d_gflog_const[a] + d_gflog_const[b];
                accum ^= d_gfexp_const[log_sum];
            }
        }
        __syncthreads();
    }

    if (row < m && col < n)
        C[row * ldc + col] = accum;
}
```

### 4. Constant Memory 优化

```cpp
__constant__ uint8_t d_gfexp_const[768];
__constant__ uint8_t d_gflog_const[256];

// 初始化后拷贝到常量内存
cudaMemcpyToSymbol(d_gfexp_const, d_gf_exp, exp_size);
cudaMemcpyToSymbol(d_gflog_const, d_gf_log, log_size);
```

## 关键 Bug 调试过程

### 问题现象

初始测试显示 100% 错误率：
```
Matrix size: 32x32 * 32x32 = 32x32
Errors: 1024 / 1024
```

### 调试步骤

1. **验证表初始化**: 创建 `test_gf_tables.cu` 确认 gf_log/gf_exp 正确
2. **验证单点乘法**: 创建 `test_kernel_gf.cu` 发现单个乘法正确但累加错误
3. **隔离问题**: 创建 `test_smem_debug.cu` 确认共享内存加载正确
4. **发现溢出**: 创建 `test_overflow.cu` 证明 uint8_t 溢出：
   ```
   gf_log[100] = 195, gf_log[11] = 238
   log_sum = 195 + 238 = 433
   uint8_t: 433 % 256 = 177 → gf_exp[177] = 219 (错误)
   int:     gf_exp[433] = 171 (正确)
   ```

### 根本原因

```cpp
// 错误代码
uint8_t log_sum = d_gflog_const[a] + d_gflog_const[b];  // 溢出

// 正确代码
int log_sum = d_gflog_const[a] + d_gflog_const[b];      // 无溢出
```

log 值范围 [0, 254]，两数之和最大 508，超出 uint8_t 范围 [0, 255]。

### 修复方案

扩展表类型定义，同时确保中间计算使用 int：
- `gf_exp` 表大小 768 字节，支持索引到 508
- 中间变量 `log_sum` 使用 int 类型

## 验证方法

### CPU 参考实现

```cpp
void gf_gemm_reference(const uint8_t* A, const uint8_t* B, uint8_t* C,
                       int m, int n, int k) {
    // 初始化 GF 表
    uint8_t gf_exp[768], gf_log[256];
    // ... 初始化代码

    memset(C, 0, m * n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            for (int l = 0; l < k; ++l) {
                uint8_t a = A[i * k + l];
                uint8_t b = B[l * n + j];
                if (a != 0 && b != 0)
                    C[i * n + j] ^= gf_exp[gf_log[a] + gf_log[b]];
            }
}
```

### 测试用例

| 测试 | 描述 | 验证点 |
|------|------|--------|
| basic_gemm | 32x32 矩阵乘法 | 与 CPU 结果完全一致 |
| identity | 单位矩阵 | I * A = A |
| zero_matrix | 零矩阵 | 0 * A = 0 |
| various_sizes | 7 种尺寸 | 4x4 到 256x512 |
| performance | 性能基准 | 64³ 到 1024³ |

### 性能结果 (NVIDIA GPU)

| 尺寸 | 时间 (ms) | 吞吐量 (GMACS) |
|------|-----------|----------------|
| 64³  | 0.199     | 2.64           |
| 128³ | 0.558     | 7.52           |
| 256³ | 2.984     | 11.25          |
| 512³ | 21.758    | 12.34          |
| 1024³| 168.425   | 12.75          |

## 编译和部署

### CMake 配置

```cmake
cmake_minimum_required(VERSION 3.18)
project(cutlass-gf-gemm LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 90)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

add_library(cutlass_gf_gemm STATIC src/cutlass_gf_gemm.cu)
set_target_properties(cutlass_gf_gemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
```

### 构建命令

```bash
mkdir build && cd build
cmake .. -DCUTLASS_PATH=/path/to/cutlass
cmake --build . --config Release
```

### 部署到远程 GPU 服务器

```bash
# SSH 克隆到服务器
git clone ssh://user@server/home/user/Code/cutlass-gf-gemm
cd cutlass-gf-gemm/build
cmake .. -DCUTLASS_PATH=../cutlass
make -j
./test_gf_gemm
```

## 方法论总结

### 1. 增量式开发

1. 基础 GF 表初始化 → 验证
2. 简单 kernel 实现 → 验证
3. 优化 (共享内存/常量内存) → 验证
4. 完整 API 封装 → 验证

### 2. 系统性调试

- 从最简单的 single block 测试开始
- 逐步增加复杂度 (multi-block → 完整矩阵)
- 使用多个独立测试程序隔离问题
- CPU-GPU 逐元素对比

### 3. 性能优化层次

1. **正确性优先**: 先保证结果正确
2. **内存优化**: shared memory tiling, constant memory
3. **计算优化**: 查找表预计算，避免运行时模运算
4. **架构适配**: 针对不同 GPU compute capability 编译

### 4. 验证策略

- 多个测试用例覆盖边界条件
- 单位矩阵、零矩阵等数学性质验证
- 多种尺寸压力测试
- 性能基准测试

## 参考资料

- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [GF(2^8) 多项式表](https://en.wikipedia.org/wiki/Finite_field_arithmetic)
- [Reed-Solomon 编码原理](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction)
