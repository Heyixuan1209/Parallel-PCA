# PCA 并行计算课程项目

本仓库实现了一个面向并行计算课程展示的 PCA（Principal Component Analysis）实验框架，使用 `C` 语言、`MPI`、`OpenMP` 以及 `BLAS/LAPACK/LAPACKE` 完成数值问题的并行化比较。

项目的核心目标是：在同一份数据、同一条 PCA 流水线下，对比不同特征分解策略和不同并行层次的性能表现。

当前实现包含以下路线：

- 基于特征分解的 PCA
  - `eig_ser`
  - `eig_par_pro`：进程级并行，支持 `root_serial_jacobi` 与 `mpi_jacobi`
  - `eig_par_thr`：线程级并行，`root_omp_jacobi`
- 基于 SVD 的 PCA
  - `svd_ser`
  - `svd_par`

---

## 仓库结构

```text
PCA/
├─ include/               # 头文件
├─ src/core/              # 矩阵、PCA、Jacobi、MPI 辅助实现
├─ src/benchmark/         # 各可执行程序入口
├─ data/                  # 数据生成脚本与示例数据
├─ scripts/               # 批量测试与绘图脚本
├─ results/               # 结果输出目录
├─ bin/                   # 可执行文件输出目录
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

## PCA 原理

设中心化后的数据矩阵为

$$
X \in \mathbb{R}^{n \times d}
$$

其中 `n` 是样本数，`d` 是特征维数。PCA 的目标是找到一个低维正交子空间，使得投影后尽可能保留原始数据方差。

若取前 `k` 个主方向，则投影结果为

$$
Y = X V_k
$$

其中 `V_k` 由前 `k` 个主成分方向组成。

### 1. 基于特征分解的 PCA

先构造协方差矩阵：

$$
C = \frac{1}{n-1} X^T X
$$

再做特征分解：

$$
C = V \Lambda V^T
$$

其中：

- `\Lambda` 的对角元是特征值
- `V` 的列向量是特征向量

按特征值从大到小排序后，前 `k` 个特征向量就是 PCA 的主方向。

### 2. 基于 SVD 的 PCA

对中心化矩阵直接做奇异值分解：

$$
X = U \Sigma V^T
$$

此时：

- `V` 的列向量就是主方向
- `\Sigma^2 / (n-1)` 对应协方差矩阵特征值

---

## Jacobi 特征分解

本项目的 Jacobi 求解器采用**循环 Jacobi（cyclic Jacobi）**。与“古典 Jacobi”不断全局搜索最大非对角元不同，循环 Jacobi 会按照固定的行列扫描顺序遍历配对，例如通过轮转日程构造每一轮的 `(p, q)` 配对。

### 为什么改成循环 Jacobi

- 扫描顺序固定，便于并行调度与实验复现
- 每一轮可自然划分为若干互不冲突的旋转对
- 更适合做线程级或进程级并行对比

### 一轮扫描示意

设当前轮的配对序列为

$$
(0,1), (2,3), \dots
$$

下一轮再通过轮转规则生成新的配对，直到收敛或达到最大迭代次数。

---

## 三种特征分解实现

### 1. `eig_ser`

串行 PCA 的基线版本。

流程：

1. 读取并中心化数据
2. 计算协方差矩阵
3. 在单线程中执行循环 Jacobi
4. 排序特征值与特征向量
5. 计算低维投影

### 2. `eig_par_pro`

进程级并行版本，支持两种 Jacobi 策略：

- `root_serial_jacobi`
- `mpi_jacobi`

#### `root_serial_jacobi`

这是最稳妥的对照组：只有协方差构造和投影是并行的，Jacobi 仍然只在 `root` 进程串行执行。

关键实现思路：

```c
rc = jacobi_eigen_parallel_gather_root_serial(
    C_local->data, d_cov, eigvals, Vbuf,
    MPI_COMM_WORLD, 0, jacobi_tol, jacobi_maxiter
);
```

这条路径适合证明“仅并行外层数据处理”能带来多少收益。

#### `mpi_jacobi`

这是真正的全局进程级 Jacobi：

1. 各进程先完成本地协方差贡献
2. `MPI_Reduce` 汇总到 `root`
3. `root` 把完整协方差矩阵分发为行块
4. 所有进程按循环 Jacobi 的固定配对顺序共同迭代
5. `MPI_Allgatherv` 同步被更新的列
6. `root` 收集对角元并排序特征对

这个版本的重点是：**每次旋转后都保持全局一致性**，从而让所有进程在同一轮扫描中协作推进。

### 3. `eig_par_thr`

线程级并行版本，只在 `root` 上并行 Jacobi。

实现方式是：

- `MPI` 负责数据分发、均值、协方差和投影
- `OpenMP` 负责 `root` 内部的循环 Jacobi 旋转对更新

核心调用：

```c
rc = jacobi_eigen_parallel_gather_root_omp(
    C_local->data, d_cov, eigvals, Vbuf,
    MPI_COMM_WORLD, 0, jacobi_tol, jacobi_maxiter
);
```

这种设计的优点是实现简单，适合和 `root_serial_jacobi` 直接对比；缺点是加速上限仍受 `root` 单点约束。

---

## 4. 循环 Jacobi 的代码组织

循环 Jacobi 的核心不是“找最大元素”，而是“按固定调度顺序扫描所有配对”。代码中主要分为三步：

1. 初始化配对轮转序列
2. 按当前轮序列生成 `(p, q)` 对
3. 执行旋转并更新矩阵/特征向量

简化后的更新逻辑如下：

```c
for (phase = 0; phase < phase_count; ++phase) {
    build_pairs(order, pairs_p, pairs_q);
    for (idx = 0; idx < pair_count; ++idx) {
        rotate_pair(A, eigvecs, pairs_p[idx], pairs_q[idx]);
    }
    rotate_order(order);
}
```

在 `root_omp_jacobi` 中，`pair_count` 这一层还可以进一步用 `OpenMP` 并行分发，因为同一轮中的配对是互不冲突的。

---

## 构建与运行

### 依赖环境

推荐在 `Linux/WSL` 下构建，依赖包括：

- `gcc` / `mpicc`
- `cmake`
- `MPI`
- `BLAS`
- `LAPACK`
- `LAPACKE`
- `OpenMP`
- `python3`
- `matplotlib`

### 构建

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

### 运行示例

生成数据：

```bash
python3 data/generate_data.py \
  --samples 2000 \
  --features 200 \
  --rank 20 \
  --noise 0.01 \
  --seed 2026 \
  --out data/synthetic.csv
```

串行 Jacobi PCA：

```bash
./bin/pca_eig_ser data/synthetic.csv 10 results/eig_ser_k10.csv
```

进程级 PCA（串行 Jacobi）：

```bash
mpirun -np 4 ./bin/pca_eig_par_pro \
  data/synthetic.csv 10 results/eig_par_root_serial_k10 root_serial_jacobi
```

进程级 PCA（MPI Jacobi）：

```bash
mpirun -np 4 ./bin/pca_eig_par_pro \
  data/synthetic.csv 10 results/eig_par_mpi_k10 mpi_jacobi
```

线程级 PCA（OpenMP Jacobi）：

```bash
OMP_NUM_THREADS=4 \
mpirun -np 4 ./bin/pca_eig_par_thr \
  data/synthetic.csv 10 results/eig_par_root_omp_k10
```

SVD 版本：

```bash
./bin/pca_svd_ser data/synthetic.csv 10 results/svd_ser_k10.csv
mpirun -np 4 ./bin/pca_svd_par data/synthetic.csv 10 results/svd_par_k10
```

---

## 批量测试与绘图

一键运行基准并生成图表：

```bash
bash scripts/run_benchmarks.sh
```

常见输出：

- `results/benchmarks.csv`
- `results/benchmark_summary.png`
- `results/benchmark_speedup.png`

你也可以通过环境变量控制规模，例如：

```bash
K_LIST="5 10 20 40" PROCS="2 4" OMP_NUM_THREADS=4 bash scripts/run_benchmarks.sh
```

---

## 结果解读建议

比较时建议重点看：

- 不同 `k` 下的总耗时变化
- `root_serial_jacobi`、`root_omp_jacobi`、`mpi_jacobi` 的相对加速比
- 随进程数增加，`MPI` 通信开销是否开始主导
- `k` 增大后，投影阶段与特征分解阶段的比例变化

---

## 局限与后续扩展

当前实现适合课程展示与性能对比，仍有进一步优化空间：

- 将 Jacobi 做成更细粒度的块并行版本
- 引入分布式 SVD 作为对照
- 增加效率、加速比与可扩展性分析
- 扩展不同 `n/d/k` 组合下的实验矩阵

