/* utils.h
 * 通用工具与辅助函数声明（I/O、计时、验证函数等）。
 */

#ifndef PCA_UTILS_H
#define PCA_UTILS_H

#include <stddef.h>
#include <mpi.h>
#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 时间工具：如果在 MPI 环境下，使用 MPI_Wtime，否则使用 gettimeofday 实现。
 * 返回秒数（double）。
 */
double pca_wtime(void);

/* 简易错误检查：在 MPI 中广播错误并在 root 打印消息（实现文件中） */
void pca_mpi_abort_if_error(int errcode, const char *msg, MPI_Comm comm);

/* 验证函数：计算矩阵 Frobenius 范数，重构误差等 */
double pca_frobenius_norm(const double *A, size_t rows, size_t cols);

/* 按全局样本数计算中心化均值；串行时可传 MPI_COMM_SELF。 */
int pca_compute_global_mean(const matrix_t *X_local, double *mu, MPI_Comm comm);

/* 由特征值 / 奇异值计算前 k 个主成分的累计解释方差占比。 */
double pca_explained_ratio_from_eigvals(const double *eigvals, size_t d, int k);
double pca_explained_ratio_from_singvals(const double *singvals, size_t d, int k);

/* 打印矩阵的前小块（调试用） */
void pca_print_block(const double *A, size_t rows, size_t cols, size_t maxr, size_t maxc, const char *name);

#ifdef __cplusplus
}
#endif

#endif /* PCA_UTILS_H */
