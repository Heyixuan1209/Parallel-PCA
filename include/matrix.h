/* matrix.h
 * 基础矩阵类型与常用操作的声明。
 * 约定：矩阵以列优先（column-major / Fortran order）存储，便于与 BLAS/LAPACK 互操作。
 */

#ifndef PCA_MATRIX_H
#define PCA_MATRIX_H

#include <stddef.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct matrix_t {
	size_t rows;    /* 行数 n */
	size_t cols;    /* 列数 d */
	double *data;   /* 指向列优先存放的连续内存，长度 rows*cols */
} matrix_t;

/* 内存管理 */
matrix_t *matrix_create(size_t rows, size_t cols);
void matrix_free(matrix_t *m);

/* 直接访问数据指针（列优先） */
double *matrix_data(matrix_t *m);

/* 基础操作：拷贝、置零 */
int matrix_clear(matrix_t *m); /* set all entries to zero */
int matrix_copy(matrix_t *dst, const matrix_t *src);

/* 矩阵乘法（包装 BLAS dgemm），接口采用 column-major */
int matrix_dgemm(double alpha, const matrix_t *A, const matrix_t *B, double beta, matrix_t *C);

/* 计算局部协方差 C_local = X_local^T * X_local（X_local: n_local x d，结果 d x d）
 * 要求 C_local 已分配并大小为 d x d。返回 0 表示成功。 */
int matrix_local_covariance(const matrix_t *X_local, matrix_t *C_local);

/* 列中心化：给定长度为 cols 的均值向量 mu，执行 X[:,j] -= mu[j] */
void matrix_center_columns(matrix_t *X, const double *mu);

/* 简易 I/O：CSV 读写（实现可在 utils 中使用更高效的二进制载入）
 * 返回 0 表示成功，非 0 表示失败。 */
int matrix_read_csv(const char *path, matrix_t **out); /* allocates matrix */
int matrix_write_csv(const char *path, const matrix_t *m);

/* MPI 辅助：按行块划分矩阵 */
void matrix_row_block(size_t rows, int nranks, int rank, size_t *start, size_t *count);

/* 由 root 读取完整矩阵后，按行块分发到所有进程。非 root 进程传入 global == NULL。 */
int matrix_scatter_rows_from_root(const matrix_t *global, matrix_t **local_out, int root, MPI_Comm comm);

/* 将各进程的行块矩阵按原顺序收集回 root。仅 root 上的 global_out 有效。 */
int matrix_gather_rows_to_root(const matrix_t *local, matrix_t **global_out, int root, MPI_Comm comm);

#ifdef __cplusplus
}
#endif

#endif /* PCA_MATRIX_H */
