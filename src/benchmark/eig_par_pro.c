/* eig_par.c
 * Parallel PCA using eigendecomposition.
 * Supports two eigensolver modes:
 *   - root_serial_jacobi
 *   - mpi_jacobi
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "pca.h"
#include "jacobi.h"
#include "matrix.h"
#include "utils.h"

static int is_supported_eig_mode(const char *mode) {
	return mode &&
		(strcmp(mode, "root_serial_jacobi") == 0 ||
		 strcmp(mode, "mpi_jacobi") == 0);
}

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (argc < 3) {
		if (rank == 0) fprintf(stderr, "Usage: %s input.csv k [out_prefix] [mode]\n", argv[0]);
		MPI_Finalize();
		return 1;
	}

	const char *input = argv[1];
	int k = atoi(argv[2]);
	const char *outp_prefix = (argc >= 4) ? argv[3] : "results/eig_par";
	size_t global_n = 0;
	double jacobi_tol = 1e-8;
	int jacobi_maxiter = 2000;
	int jacobi_debug = 0;
	int gather_projection = 0;
	int eig_debug = 0;
	const char *eig_mode = "root_serial_jacobi";

	const char *tol_env = getenv("PCA_JACOBI_TOL");
	const char *iter_env = getenv("PCA_JACOBI_MAXITER");
	const char *debug_env = getenv("PCA_JACOBI_DEBUG");
	const char *gather_env = getenv("PCA_GATHER_PROJECTION");
	const char *eig_debug_env = getenv("PCA_EIG_DEBUG");
	const char *eig_mode_env = getenv("PCA_EIG_PAR_MODE");
	if (tol_env) jacobi_tol = atof(tol_env);
	if (iter_env) jacobi_maxiter = atoi(iter_env);
	if (debug_env) jacobi_debug = atoi(debug_env);
	if (gather_env) gather_projection = atoi(gather_env);
	if (eig_debug_env) eig_debug = atoi(eig_debug_env);
	if (eig_mode_env && *eig_mode_env) eig_mode = eig_mode_env;
	if (argc >= 5 && argv[4] && argv[4][0]) eig_mode = argv[4];
	if (!is_supported_eig_mode(eig_mode)) {
		if (rank == 0) {
			fprintf(stderr, "Unknown eig mode: %s\n", eig_mode);
			fprintf(stderr, "Supported modes: root_serial_jacobi, mpi_jacobi\n");
		}
		MPI_Finalize();
		return 1;
	}
	jacobi_parallel_set_debug(jacobi_debug);

	MPI_Barrier(MPI_COMM_WORLD);
	double t0 = pca_wtime();
	double t_read_start = t0;
	matrix_t *X = NULL;
	if (rank == 0 && matrix_read_csv(input, &X) != 0) {
		fprintf(stderr, "Failed to read %s\n", input);
		MPI_Abort(MPI_COMM_WORLD, 2);
	}
	if (rank == 0) {
		global_n = X->rows;
	}

	matrix_t *X_local = NULL;
	if (matrix_scatter_rows_from_root(X, &X_local, 0, MPI_COMM_WORLD) != 0) {
		if (rank == 0) fprintf(stderr, "Failed to scatter data\n");
		if (X) matrix_free(X);
		MPI_Abort(MPI_COMM_WORLD, 2);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	double t_read = pca_wtime() - t_read_start;

	size_t d = X_local->cols;
	unsigned long long global_n_ull = (unsigned long long)global_n;
	MPI_Bcast(&global_n_ull, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
	global_n = (size_t)global_n_ull;
	double *mu = (double *)malloc(d * sizeof(double));
	if (!mu || pca_compute_global_mean(X_local, mu, MPI_COMM_WORLD) != 0) {
		if (rank == 0) fprintf(stderr, "Failed to compute mean\n");
		free(mu);
		matrix_free(X_local);
		if (X) matrix_free(X);
		MPI_Abort(MPI_COMM_WORLD, 2);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double t_center = pca_wtime();
	matrix_center_columns(X_local, mu);
	MPI_Barrier(MPI_COMM_WORLD);
	t_center = pca_wtime() - t_center;

	double *eigvals = (double *)malloc(d * sizeof(double));
	matrix_t *eigvecs = NULL;
	matrix_t *proj_local = NULL;
	if (!eigvals) {
		free(mu);
		matrix_free(X_local);
		if (X) matrix_free(X);
		MPI_Abort(MPI_COMM_WORLD, 2);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double t_pca_start = pca_wtime();
	int rc = 0;
	if (!X_local || !eigvals) {
		rc = -1;
	} else {
		size_t d_cov = X_local->cols;
		matrix_t *C_local = matrix_create(d_cov, d_cov);
		if (!C_local) {
			rc = -1;
		} else {
			matrix_local_covariance(X_local, C_local);
			long long local_n = (long long)X_local->rows;
			long long total_n = 0;
			MPI_Allreduce(&local_n, &total_n, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
			if (total_n < 2) {
				rc = -1;
			} else {
				double scale = 1.0 / (double)(total_n - 1);
				for (size_t i = 0; i < d_cov * d_cov; ++i) C_local->data[i] *= scale;
				double *Vbuf = (double *)malloc(d_cov * d_cov * sizeof(double));
				if (!Vbuf) {
					rc = -1;
				} else {
					if (strcmp(eig_mode, "mpi_jacobi") == 0) {
						rc = jacobi_eigen_parallel_gather(C_local->data, d_cov, eigvals, Vbuf, MPI_COMM_WORLD, 0, jacobi_tol, jacobi_maxiter);
					} else {
						rc = jacobi_eigen_parallel_gather_root_serial(C_local->data, d_cov, eigvals, Vbuf, MPI_COMM_WORLD, 0, jacobi_tol, jacobi_maxiter);
					}
					if (rc == 0) {
						eigvecs = matrix_create(d_cov, d_cov);
						if (!eigvecs) {
							rc = -1;
						} else {
							memcpy(eigvecs->data, Vbuf, d_cov * d_cov * sizeof(double));
							int kk = (k <= 0 || k > (int)d_cov) ? (int)d_cov : k;
							matrix_t *Vmat = matrix_create(d_cov, (size_t)kk);
							if (!Vmat) {
								rc = -1;
							} else {
								for (int col = 0; col < kk; ++col) {
									memcpy(Vmat->data + (size_t)col * d_cov, Vbuf + (size_t)col * d_cov, d_cov * sizeof(double));
								}
								proj_local = matrix_create(X_local->rows, (size_t)kk);
								if (!proj_local || matrix_dgemm(1.0, X_local, Vmat, 0.0, proj_local) != 0) {
									rc = -1;
								}
								matrix_free(Vmat);
							}
						}
					}
					free(Vbuf);
				}
			}
			matrix_free(C_local);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	double t_pca = pca_wtime() - t_pca_start;
	if (rc != 0) {
		if (rank == 0) fprintf(stderr, "pca_by_eig_parallel_gather failed: %d\n", rc);
		free(mu);
		free(eigvals);
		matrix_free(X_local);
		matrix_free(eigvecs);
		matrix_free(proj_local);
		if (X) matrix_free(X);
		MPI_Finalize();
		return 3;
	}

	char outpath[512];
	snprintf(outpath, sizeof(outpath), "%s_rank%02d.csv", outp_prefix, rank);
	matrix_write_csv(outpath, proj_local);

	MPI_Barrier(MPI_COMM_WORLD);
	double total_time = pca_wtime() - t0;

	if (rank == 0) {
		double explained = pca_explained_ratio_from_eigvals(eigvals, d, k);
		printf("eig_par[%s],%zu,%zu,%d,%d,%.6f,%.6f,%.6f,%.6f,%.8f\n",
			   eig_mode, global_n, d, k, size, t_read, t_center, t_pca, total_time, explained);
	}

	if (gather_projection) {
		if (eig_debug && rank == 0) {
			fprintf(stderr, "[eig_par] gathering projected data to root\n");
		}
		matrix_t *projection_global = NULL;
		if (rank == 0) {
			matrix_gather_rows_to_root(proj_local, &projection_global, 0, MPI_COMM_WORLD);
			if (projection_global) {
				matrix_write_csv("results/eig_par_projection.csv", projection_global);
				matrix_free(projection_global);
			}
		} else {
			matrix_gather_rows_to_root(proj_local, NULL, 0, MPI_COMM_WORLD);
		}
	}

	free(mu);
	free(eigvals);
	matrix_free(X_local);
	matrix_free(eigvecs);
	matrix_free(proj_local);
	if (X) matrix_free(X);

	MPI_Finalize();
	return 0;
}
