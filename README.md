# CUTLASS GF Gemm

基于 CUTLASS 库实现的 Galois Field 矩阵乘法 CUDA 加速库。

## 项目背景

本项目利用 NVIDIA CUTLASS 库的模板架构，实现 GF(2^8) 有限域上的矩阵乘法运算，主要用于 Reed-Solomon 编码等存储系统纠错场景的 GPU 加速。

## 技术特点

- 基于 CUTLASS 2.x/3.x API 实现
- 支持 GF(2^8) 有限域算术
- 使用对数/指数表优化乘法运算
- 共享内存优化的矩阵分块算法
- 支持多种矩阵布局（行优先/列优先）

## 目录结构

```
cutlass-gf-gemm/
├── include/           # 头文件
│   ├── gf_ops.h      # Galois Field 运算定义
│   ├── cutlass_gf_gemm.h  # CUTLASS GF Gemm 主接口
│   └── gf16.h        # GF(2^8) 查找表
├── src/              # 源文件
│   └── cutlass_gf_gemm.cu
├── examples/         # 示例程序
│   └── example_gf_gemm.cu
├── tests/            # 测试程序
│   └── test_gf_gemm.cu
├── CMakeLists.txt    # CMake 构建配置
└── README.md
```

## 构建要求

- CUDA Toolkit 11.0+
- CMake 3.18+
- CUTLASS (作为 submodule 包含)
- GPU Compute Capability 7.0+

## 构建方法

```bash
# 克隆项目
git clone --recursive https://github.com/yszheda/cutlass-gf-gemm.git
cd cutlass-gf-gemm

# 创建构建目录
mkdir build && cd build

# 配置 CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
make -j8

# 运行测试
./tests/test_gf_gemm
```

## 使用示例

```cpp
#include "cutlass_gf_gemm.h"

// 创建矩阵乘法实例
GFGemm gemm;

// 初始化矩阵
std::vector<uint8_t> A(n * k);
std::vector<uint8_t> B(k * m);
std::vector<uint8_t> C(n * m);

// 填充数据...

// 执行 GF(2^8) 矩阵乘法: C = A * B
gemm.compute(n, k, m, A.data(), B.data(), C.data());
```

## 性能优化

1. **共享内存分块**: 使用 tile-based 算法减少全局内存访问
2. **常量内存查找表**: GF 对数/指数表存储在常量内存
3. **向量化内存访问**: 支持 word-aligned 访问模式
4. **多流并发**: 支持 CUDA Stream 并行执行

## 参考资料

- [CUTLASS Documentation](https://nvidia.github.io/cutlass/)
- [Galois Field Arithmetic](https://en.wikipedia.org/wiki/Finite_field_arithmetic)
- [Reed-Solomon Error Correction](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction)

## 许可证

BSD 3-Clause License
