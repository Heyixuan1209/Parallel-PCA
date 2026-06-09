# PCA并行计算 课程展示项目

本仓库实现了一个适合并行计算课程展示的 PCA（Principal Component Analysis）项目。代码使用 `C` 语言编写，并结合 `MPI`、`OpenMP`、`BLAS/LAPACK/LAPACKE` 对 PCA 的两条经典数值路线进行实现与比较：

- 基于协方差矩阵特征分解的 PCA
- 基于奇异值分解（SVD）的 PCA

项目的核心展示重点是：**同一条特征分解路线，在不同并行层次下会呈现怎样的性能差异与工程权衡**。当前仓库包含三种并行特征分解方案：

1. `root_serial_jacobi`
2. `root_omp_jacobi`
3. `mpi_jacobi`

此外还包含：

- `eig_ser`：串行特征分解 PCA
- `svd_ser`：串行 SVD PCA
- `svd_par`：并行 SVD PCA

---

## 1. 仓库结构

```text
PCA/
├─ include/                # 头文件：矩阵、PCA、Jacobi、工具函数接口
├─ src/
│  ├─ core/                # 核心实现：矩阵运算、PCA、Jacobi、MPI辅助函数
│  └─ benchmark/           # benchmark 入口程序
├─ data/                   # 数据生成脚本与示例数据
├─ scripts/                # 批量测试与绘图脚本
├─ results/                # 实验结果输出目录
├─ bin/                    # 可执行文件输出目录
└─ CMakeLists.txt          # CMake 构建配置
```

推荐重点阅读：

- `src/core/pca.c`
- `src/core/jacobi_ser.c`
- `src/core/jacobi_par.c`
- `src/benchmark/eig_par_pro.c`
- `src/benchmark/eig_par_thr.c`
- `scripts/run_benchmarks.sh`
- `scripts/plot.py`

---

## 2. PCA 原理

设中心化后的数据矩阵为：

$$
X \in \mathbb{R}^{n \times d}
$$

其中：

- `n` 为样本数
- `d` 为原始特征维数

PCA 的目标是找到一个低维线性子空间，使得数据投影到该子空间后仍保留尽可能多的方差信息。若取前 `k` 个主方向，则投影结果为：

$$
Y = X V_k
$$

其中 `V_k` 由前 `k` 个主成分方向组成。

### 2.1 基于特征分解的 PCA

先构造协方差矩阵：

$$
C = \frac{1}{n-1} X^T X
$$

再做特征分解：

$$
C = V \Lambda V^T
$$

其中：

- $`\Lambda`$ 的对角元是特征值
- `V` 的列向量是特征向量

按特征值从大到小排序后，前 `k` 个特征向量即为 PCA 的主方向。

### 2.2 基于 SVD 的 PCA

对中心化矩阵直接做奇异值分解：

$$
X = U \Sigma V^T
$$

则：

- `V` 的列向量就是 PCA 主方向
- $`\Sigma^2 / (n - 1)`$ 与协方差矩阵特征值对应

---

## 3. 特征分解路线的三种并行方案

这是本项目最核心的展示内容。

### 3.1 `root_serial_jacobi`

流程如下：

1. `root` 读取完整数据
2. 用 `MPI` 按行分发样本
3. 各进程并行计算全局均值并完成中心化
4. 各进程计算本地协方差贡献 `C_local`
5. 使用 `MPI_Reduce` 将协方差矩阵汇总到 `root`
6. `root` 串行执行 Jacobi 特征分解
7. 广播特征值和特征向量
8. 各进程本地完成投影 `X_local V_k`

这是最清晰、最稳定的并行基线。

### 3.2 `root_omp_jacobi`

上一版本的算法中，核心部分Jacobi算法仍然是串行的，这个版本考虑将其并行化。出于最小化改动的原则，我们保留前后 `MPI` 流程不变，仅把第 6 步的 `root` Jacobi 求解替换为 `OpenMP` 版本。

#### 设计框架

我采用了“**串行控制流 + 计算密集段并行化**”的思路：

1. 外层 Jacobi sweep 仍由串行控制
2. 每一轮先并行扫描，寻找当前最大非对角元
3. 旋转参数 `(c, s)` 在串行部分计算
4. 用 `OpenMP parallel for` 并行更新矩阵中除 `p, q` 外的其余元素
5. 再用 `OpenMP parallel for` 并行更新特征向量矩阵

也就是说，并行化并不是包住整个 Jacobi 算法，而是针对其中最耗时的数组更新部分。

#### 实现要点

在 `src/core/jacobi_par.c` 中，`root_omp_jacobi` 的核心结构可以概括为：

```c
for (int iter = 0; iter < maxiter; ++iter) {
    #pragma omp parallel for reduction(max:max_off)
    for (...) {
        // 扫描最大非对角元
    }

    for (p) {
        for (q) {
            // 串行计算旋转参数

            #pragma omp parallel for
            for (...) {
                // 并行更新 A 的其他行/列
            }

            #pragma omp parallel for
            for (...) {
                // 并行更新特征向量
            }
        }
    }
}
```

相比在一个大 `parallel` 区域中频繁穿插 `single`、`for`、`barrier` 的写法，这样的结构更容易理解，也更不容易出现同步死锁。

#### 这一版本的意义

它适合回答：

- `root` 成为瓶颈时，线程级并行能否缓解？
- 共享内存线程并行是否足以胜过纯 `MPI` 的协同分解？

### 3.3 `mpi_jacobi`

这个版本进一步把 Jacobi 本身推进到了进程级并行。

#### 总体思路

前半部分和 `root_serial_jacobi` 相同：

1. 数据分发
2. 全局均值计算
3. 中心化
4. 本地协方差
5. `MPI_Reduce` 汇总完整协方差矩阵到 `root`

不同点在于：

- `root` 不再独占特征分解
- `root` 拿到完整协方差矩阵后，会将其重新按行块分发给所有进程
- 每个进程维护自己负责的一段行块
- Jacobi 旋转由所有进程共同完成

#### 行块分布

当前实现采用**按行块切分矩阵**：

- 每个进程维护若干完整的矩阵行
- 使用 `row_counts` / `row_displs` 描述分布
- 对应的局部矩阵存储为按行组织的连续缓冲区

这样做的好处是：

- 容易实现 `Scatterv / Gatherv`
- 每个进程更新自己负责的行时逻辑清晰

#### 每个旋转对 `(p, q)` 的处理

对于每个旋转对 `(p, q)`：

1. 找到第 `p` 行和第 `q` 行的所有者进程
2. 拥有者进程取出：
   - `app`
   - `aqq`
   - `apq`
3. 用 `MPI_Bcast` 将这些标量广播给全部进程
4. 各进程并行更新自己负责的局部行块中的第 `p`、`q` 列
5. 用 `MPI_Allgatherv` 汇总旋转后整列数据
6. 由拥有 `p`、`q` 行的进程恢复对应完整行
7. 各进程同步更新自己局部的特征向量行块

核心结构可概括为：

```c
for (p = 0; p < n - 1; ++p) {
    for (q = p + 1; q < n; ++q) {
        // owner_p / owner_q 广播 app, aqq, apq
        MPI_Bcast(...);

        // 各进程更新本地行块中的第 p, q 列
        for (local_row ...) {
            ...
        }

        // 汇总旋转后的整列
        MPI_Allgatherv(... col_p ...);
        MPI_Allgatherv(... col_q ...);

        // 更新局部特征向量
        for (local_row ...) {
            ...
        }
    }
}
```

#### 这一版本的意义

它适合展示一个很重要的课程事实：

> 核心数值算法即使理论上可以并行，也不代表工程上就一定更快。

原因在于：

- Jacobi 每一步都强依赖最新矩阵状态
- 通信与同步非常频繁
- 当特征维数 `d` 不大时，通信成本可能超过计算收益

因此，`mpi_jacobi` 更适合用来解释“为什么真正并行化特征分解很难”，而不一定总是最快版本。


### 3.4 三种方案之间的关系

这三种方案分别代表三种并行层次：

- `root_serial_jacobi`：只并行化数据流，不并行化核心 Jacobi
- `root_omp_jacobi`：Jacobi 仍在 `root`，但用线程并行优化
- `mpi_jacobi`：Jacobi 主过程进入 `MPI` 协同求解

因此展示时可以自然比较：

1. 只做数据流并行能得到多少收益？
2. `OpenMP` 是否能有效缓解 `root` 瓶颈？
3. 把 Jacobi 推进到纯 `MPI` 后，通信代价是否抵消了理论加速？


---

## 4. 仓库使用说明

### 4.1构建环境

推荐在 **Linux / WSL** 环境中构建和运行，依赖包括：

- `gcc` / `mpicc`
- `cmake`
- `MPI`
- `BLAS`
- `LAPACK`
- `LAPACKE`
- `OpenMP`
- `python3`
- `matplotlib`

构建方法为

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j
```

生成的主要可执行文件：

```text
bin/pca_eig_ser
bin/pca_eig_par_pro
bin/pca_eig_par_thr
bin/pca_svd_ser
bin/pca_svd_par
```

### 4.2 单独运行方法

### 4.2.1 生成测试数据

```bash
python3 data/generate_data.py \
  --samples 2000 \
  --features 200 \
  --rank 20 \
  --noise 0.01 \
  --seed 2026 \
  --out data/synthetic.csv
```

### 4.2.2 串行特征分解 PCA

```bash
./bin/pca_eig_ser data/synthetic.csv 10 results/eig_ser_k10.csv
```

### 4.2.3 并行特征分解 PCA：`root_serial_jacobi`

```bash
mpirun -np 4 ./bin/pca_eig_par_pro \
  data/synthetic.csv \
  10 \
  results/eig_par_root_serial_k10 \
  root_serial_jacobi
```

### 4.2.4 并行特征分解 PCA：`root_omp_jacobi`

```bash
OMP_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=1 \
mpirun -np 4 ./bin/pca_eig_par_thr \
  data/synthetic.csv \
  10 \
  results/eig_par_root_omp_k10
```

### 4.2.5 并行特征分解 PCA：`mpi_jacobi`

```bash
mpirun -np 4 ./bin/pca_eig_par_pro \
  data/synthetic.csv \
  10 \
  results/eig_par_mpi_k10 \
  mpi_jacobi
```

### 4.2.6 串行 / 并行 SVD

```bash
./bin/pca_svd_ser data/synthetic.csv 10 results/svd_ser_k10.csv
mpirun -np 4 ./bin/pca_svd_par data/synthetic.csv 10 results/svd_par_k10
```

---

### 4.3批量运行方法

 一键运行 benchmark

```bash
bash scripts/run_benchmarks.sh
```

默认脚本会：

1. 构建项目
2. 生成合成数据
3. 运行 `eig_ser`、`svd_ser`
4. 运行 `eig_par_pro` 的两种模式：
   - `root_serial_jacobi`
   - `mpi_jacobi`
5. 运行 `eig_par_thr`（`root_omp_jacobi`）
6. 运行 `svd_par`
7. 自动生成 `CSV` 与绘图

默认关键环境变量：

```bash
PCA_EIG_PRO_MODES="root_serial_jacobi mpi_jacobi"
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
GOTO_NUM_THREADS=1
```

示例：

```bash
FORCE_REBUILD=1 \
K_LIST="5 10 20 40" \
PROCS="2 4" \
PCA_EIG_PRO_MODES="root_serial_jacobi mpi_jacobi" \
OMP_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=1 \
bash scripts/run_benchmarks.sh
```

输出结果：

- `results/benchmarks.csv`
- `results/benchmark_summary.png`
- `results/benchmark_speedup.png`

---

## 5. 绘图与结果解释

`scripts/plot.py` 会根据 `results/benchmarks.csv` 自动生成：

- `results/benchmark_summary.png`
- `results/benchmark_speedup.png`

图中会区分：

- `eig_par[root_serial_jacobi]`
- `eig_par[root_omp_jacobi]`
- `eig_par[mpi_jacobi]`
- `svd_par`

每张图适合回答的问题包括：

1. `explained variance ratio` 是否一致
2. `k` 增大后投影开销是否明显上升
3. 哪种并行方案随进程数增长更稳定
4. `root_omp_jacobi` 与 `mpi_jacobi` 谁更适合当前机器规模

---

## 6. 当前实现的局限性

当前仓库非常适合课程展示，但仍有明确边界：

- `root_serial_jacobi` 本质上仍是 `root` 求解
- `root_omp_jacobi` 只是把 `root` 求解器替换为线程并行
- `mpi_jacobi` 虽然把 Jacobi 主过程扩展到了多个进程，但仍从 `root` 聚合完整协方差矩阵开始，且进程间通信开销过大
- `svd_par` 仍然依赖 `root` 完成最终 SVD

因此，这个项目更准确地说是：

> 一个面向课程展示的、逐步增强并行程度的 PCA 实验框架。

---

## 7. 后续扩展方向

如果希望继续深入，可以考虑：

1. 实现完全分布式的对称特征分解，即不考虑汇总全局协方差矩阵，直接由多个进程执行Jacobi迭代
2. 引入 `ScaLAPACK` / `Elemental` 实现分布式 SVD
3. 扩展实验分析部分，例如增加并行效率（efficiency）指标、增加重构误差与数值稳定性分析、比较不同 `n`、`d` 下瓶颈如何转移


