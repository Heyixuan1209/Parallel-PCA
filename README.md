# PCA并行计算 课程展示项目

本仓库实现了一个面向并行计算课程展示的 PCA（Principal Component Analysis）项目，使用 `C` 语言，结合 `MPI`、`OpenMP` 与 `BLAS/LAPACK/LAPACKE`，对两条经典数值路线进行实现与性能比较：

- 基于协方差矩阵特征分解的 PCA
- 基于奇异值分解（SVD）的 PCA

项目当前包含以下可执行版本：

- `eig_ser`：串行特征分解 PCA
- `eig_par_pro`：进程级并行 PCA，支持 `root_serial_jacobi` 与 `mpi_jacobi`
- `eig_par_thr`：线程级并行 PCA，对应 `root_omp_jacobi`
- `svd_ser`：串行 SVD PCA
- `svd_par`：并行 SVD PCA

---

## 1. 仓库结构

```text
PCA/
├─ include/                # 头文件：矩阵、PCA、Jacobi、工具函数接口
├─ src/
│  ├─ core/                # 核心实现：矩阵运算、PCA、Jacobi、MPI辅助函数
│  └─ benchmark/           # 各可执行程序入口
├─ data/                   # 数据生成脚本与示例数据
├─ scripts/                # 批量测试与绘图脚本
├─ results/                # 实验结果输出目录
├─ bin/                    # 可执行文件输出目录
└─ CMakeLists.txt
```

建议优先阅读：

- `src/core/pca.c`
- `src/core/jacobi_ser.c`
- `src/core/jacobi_par.c`
- `src/benchmark/eig_par_pro.c`
- `src/benchmark/eig_par_thr.c`
- `scripts/run_benchmarks.sh`
- `scripts/plot.py`

---

## 2. PCA 原理

设中心化后的数据矩阵为

$$
X \in \mathbb{R}^{n \times d}
$$

其中 `n` 为样本数，`d` 为特征维数。PCA 的目标是找到一个低维正交子空间，使投影后的数据保留尽可能多的方差信息。若取前 `k` 个主方向，则投影结果为

$$
Y = X V_k
$$

其中 `V_k` 由前 `k` 个主成分方向组成。

### 2.1 基于特征分解的 PCA

先构造协方差矩阵

$$
C = \frac{1}{n-1} X^T X
$$

再做特征分解

$$
C = V \Lambda V^T
$$

其中：

- `\Lambda` 的对角元是特征值
- `V` 的列向量是特征向量

按特征值从大到小排序后，前 `k` 个特征向量就是 PCA 的主方向。

### 2.2 基于 SVD 的 PCA

对中心化矩阵直接做奇异值分解

$$
X = U \Sigma V^T
$$

此时：

- `V` 的列向量就是 PCA 主方向
- `\Sigma^2 / (n - 1)` 对应协方差矩阵特征值

---

## 3. 特征分解路线的三种并行方案

本项目围绕 Jacobi 特征分解设计了三种并行策略，用于展示“只并行化数据流”到“并行化特征分解核心阶段”的逐步增强过程。当前 Jacobi 实现采用固定配对顺序的循环扫描方式，每轮 sweep 按既定 `(p, q)` 配对推进。

### 3.1 `root_serial_jacobi`

这是最直接、最稳定的并行基线：`MPI` 负责数据分发、全局均值、中心化、本地协方差汇总与最终投影，特征分解本身仍由 `root` 串行完成。

流程如下：

1. `root` 读取完整数据矩阵
2. 使用 `MPI` 按行分发样本到各进程
3. 各进程并行计算全局均值并完成中心化
4. 各进程计算本地协方差贡献 `C_local`
5. 使用 `MPI_Reduce` 将协方差矩阵汇总到 `root`
6. `root` 对全局协方差矩阵执行 Jacobi 特征分解
7. 广播特征值和特征向量
8. 各进程在本地完成投影 `X_local V_k`

关键调用如下：

```c
rc = jacobi_eigen_parallel_gather_root_serial(
    C_local->data, d_cov, eigvals, Vbuf,
    MPI_COMM_WORLD, 0, jacobi_tol, jacobi_maxiter
);
```

这一版本适合作为对照组，用来观察：即便特征分解仍是串行的，仅通过 `MPI` 并行化数据流处理，能获得多少性能收益。

### 3.2 `root_omp_jacobi`

这一版本保留前后 `MPI` 数据流不变，只把 `root` 上的 Jacobi 求解替换成 `OpenMP` 线程级实现。

整体思路是：

- `MPI` 负责数据分发、均值计算、协方差汇总和投影
- `OpenMP` 只用于 `root` 进程内部的 Jacobi 迭代更新

在实现上，外层 sweep 由 `root` 串行控制；每轮先按固定顺序生成当前批次的 `(p, q)` 配对，再由多个线程并行更新矩阵相关行列和特征向量。由于同一轮中的配对互不冲突，这种并行方式能够在不改变整体算法框架的前提下，减少 `root` 上 Jacobi 阶段的耗时。

关键调用如下：

```c
rc = jacobi_eigen_parallel_gather_root_omp(
    C_local->data, d_cov, eigvals, Vbuf,
    MPI_COMM_WORLD, 0, jacobi_tol, jacobi_maxiter
);
```

这一版本适合回答：当瓶颈集中在 `root` 特征分解阶段时，线程级并行是否足以缓解单点计算压力。

### 3.3 `mpi_jacobi`

这一版本进一步把 Jacobi 本身推进到进程级并行。

前半段与前两种方案一致：

1. `root` 读取数据并按行分发
2. 各进程并行完成全局均值、中心化与本地协方差计算
3. 通过 `MPI_Reduce` 汇总全局协方差矩阵

随后进入 `MPI` 协同的 Jacobi 阶段：

1. `root` 将完整协方差矩阵按行重新分发给各进程
2. 所有进程按相同的循环配对顺序推进 sweep
3. 每个进程只更新自己负责的局部矩阵行和局部特征向量行
4. 每轮结束后通过集合通信同步所需数据
5. 最终由 `root` 汇总、排序并广播特征值和特征向量

关键调用如下：

```c
rc = jacobi_eigen_parallel_gather(
    C_local->data, d_cov, eigvals, Vbuf,
    MPI_COMM_WORLD, 0, jacobi_tol, jacobi_maxiter
);
```

这一版本最适合展示：当特征分解主过程也进入 `MPI` 协作后，计算加速和通信开销之间会形成怎样的权衡。

---

## 4. 构建与运行

### 4.1 依赖环境

推荐在 `Linux / WSL` 环境中构建和运行，依赖包括：

- `gcc` / `mpicc`
- `cmake`
- `MPI`
- `BLAS`
- `LAPACK`
- `LAPACKE`
- `OpenMP`
- `python3`
- `matplotlib`

### 4.2 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j
```

生成的主要可执行文件：

- `bin/pca_eig_ser`
- `bin/pca_eig_par_pro`
- `bin/pca_eig_par_thr`
- `bin/pca_svd_ser`
- `bin/pca_svd_par`

### 4.3 单一脚本运行示例

生成测试数据：

```bash
python3 data/generate_data.py \
  --samples 2000 \
  --features 200 \
  --rank 20 \
  --noise 0.01 \
  --seed 2026 \
  --out data/synthetic.csv
```

串行特征分解 PCA

```bash
./bin/pca_eig_ser data/synthetic.csv 10 results/eig_ser_k10.csv
```

并行特征分解 PCA：`root_serial_jacobi`

```bash
mpirun -np 4 ./bin/pca_eig_par_pro \
  data/synthetic.csv \
  10 \
  results/eig_par_root_serial_k10 \
  root_serial_jacobi
```

并行特征分解 PCA：`root_omp_jacobi`

```bash
OMP_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=1 \
mpirun -np 4 ./bin/pca_eig_par_thr \
  data/synthetic.csv \
  10 \
  results/eig_par_root_omp_k10
```

并行特征分解 PCA：`mpi_jacobi`

```bash
mpirun -np 4 ./bin/pca_eig_par_pro \
  data/synthetic.csv \
  10 \
  results/eig_par_mpi_k10 \
  mpi_jacobi
```

串行 / 并行 SVD

```bash
./bin/pca_svd_ser data/synthetic.csv 10 results/svd_ser_k10.csv
mpirun -np 4 ./bin/pca_svd_par data/synthetic.csv 10 results/svd_par_k10
```

---

### 4.4批量运行方法

 一键运行 benchmark

```bash
bash scripts/run_benchmarks.sh
```


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


## 5. 局限与后续扩展

当前实现适合课程展示与性能对比，但仍有一些明确边界：

- `root_serial_jacobi` 的特征分解仍受 `root` 单点限制
- `root_omp_jacobi` 只缓解了 `root` 内部计算，不能消除单点瓶颈
- `mpi_jacobi` 虽然把 Jacobi 推进到进程级并行，但同步与通信成本较高
- `svd_par` 仍然依赖 `root` 完成最终 SVD

如果需要进一步扩展，可以考虑：

- 引入更成熟的分布式特征分解 / SVD 库
- 增加效率、加速比、重构误差等分析指标
- 扩展不同 `n / d / k` 组合下的实验矩阵与结果对比
