/* pca.h
 * 高层 PCA 接口：提供基于特征分解（Jacobi）和基于 SVD（LAPACK）的函数原型。
 */

#ifndef PCA_PCA_H
#define PCA_PCA_H

#include <stddef.h>
#include <mpi.h>
#include "matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/* PCA via eigendecomposition (serial)
 * X: n x d matrix (column-major)
 * k: number of principal components to compute
 * eigvals: output array length d (allocated by caller) - top-k are relevant
 * eigvecs: output pointer (allocated inside) to d x d matrix of eigenvectors (column-major)
 * projection: output pointer (allocated inside) to n x k projected data (column-major)
 * Returns 0 on success.
 */
int pca_by_eig_serial(const matrix_t *X, int k, double *eigvals, matrix_t **eigvecs, matrix_t **projection);

/* PCA via eigendecomposition (parallel simplified)
 * X_local: local rows of X (n_local x d)
 * k: number of components
 * comm: MPI communicator
 * root: rank which will collect full C and run eig (common simplified approach)
 * Returns 0 on success; eigvecs and projection are allocated per-rank: eigvecs valid on all ranks after call.
 */
int pca_by_eig_parallel_gather(const matrix_t *X_local, int k, MPI_Comm comm, int root, double *eigvals, matrix_t **eigvecs, matrix_t **projection_local);

/* PCA via SVD (serial) - use LAPACK/LAPACKE in implementation
 * Computes top-k singular vectors/values. Similar allocation conventions as above.
 */
int pca_by_svd_serial(const matrix_t *X, int k, double *singvals, matrix_t **V, matrix_t **projection);

/* PCA via SVD (parallel) - higher level. If ScaLAPACK available, implementation can call it.
 * Here we provide a simplified interface which may fall back to gather+LAPACK when distributed libraries are not available.
 */
int pca_by_svd_parallel(const matrix_t *X_local, int k, MPI_Comm comm, int root, double *singvals, matrix_t **V_local, matrix_t **projection_local);

#ifdef __cplusplus
}
#endif

#endif /* PCA_PCA_H */
