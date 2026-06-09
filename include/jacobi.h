/* jacobi.h
 * Interfaces for serial and MPI-based Jacobi eigendecomposition.
 */

#ifndef PCA_JACOBI_H
#define PCA_JACOBI_H

#include <stddef.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

int jacobi_eigen_serial(double *A, size_t n, double *eigvals, double *eigvecs, double tol, int maxiter);

/* MPI Jacobi:
 * 1) reduce local covariance contributions to root
 * 2) root forms the full covariance matrix
 * 3) root scatters row blocks to all ranks
 * 4) all ranks cooperate to perform Jacobi rotations with MPI
 * 5) root gathers and sorts eigenpairs, then broadcasts them back
 */
int jacobi_eigen_parallel_gather(const double *C_local, size_t d, double *eigvals, double *eigvecs, MPI_Comm comm, int root, double tol, int maxiter);

/* Baseline mode: root gathers the full covariance matrix and runs the
 * original serial Jacobi kernel locally.
 */
int jacobi_eigen_parallel_gather_root_serial(const double *C_local, size_t d, double *eigvals, double *eigvecs, MPI_Comm comm, int root, double tol, int maxiter);

/* Hybrid mode: root gathers the full covariance matrix and runs an
 * OpenMP-accelerated Jacobi kernel locally, then broadcasts eigenpairs.
 */
int jacobi_eigen_parallel_gather_root_omp(const double *C_local, size_t d, double *eigvals, double *eigvecs, MPI_Comm comm, int root, double tol, int maxiter);

void jacobi_parallel_set_debug(int debug_level);

#ifdef __cplusplus
}
#endif

#endif /* PCA_JACOBI_H */
