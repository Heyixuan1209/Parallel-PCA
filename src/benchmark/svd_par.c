/* svd_par.c
 * Parallel PCA using row-distributed data + root LAPACK SVD
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "pca.h"
#include "matrix.h"
#include "utils.h"

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (argc < 3) {
		if (rank == 0) fprintf(stderr, "Usage: %s input.csv k [out_prefix]\n", argv[0]);
		MPI_Finalize();
		return 1;
	}

	const char *input = argv[1];
	int k = atoi(argv[2]);
	const char *outp_prefix = (argc >= 4) ? argv[3] : "results/svd_par";
	size_t global_n = 0;

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

	double *singvals = (double *)malloc(d * sizeof(double));
	matrix_t *V_local = NULL;
	matrix_t *proj_local = NULL;
	if (!singvals) {
		free(mu);
		matrix_free(X_local);
		if (X) matrix_free(X);
		MPI_Abort(MPI_COMM_WORLD, 2);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double t_pca_start = pca_wtime();
	int rc = pca_by_svd_parallel(X_local, k, MPI_COMM_WORLD, 0, singvals, &V_local, &proj_local);
	MPI_Barrier(MPI_COMM_WORLD);
	double t_pca = pca_wtime() - t_pca_start;
	if (rc != 0) {
		if (rank == 0) fprintf(stderr, "pca_by_svd_parallel failed: %d\n", rc);
		free(mu);
		free(singvals);
		matrix_free(X_local);
		matrix_free(V_local);
		matrix_free(proj_local);
		if (X) matrix_free(X);
		MPI_Finalize();
		return 3;
	}

	char outpath[512];
	snprintf(outpath, sizeof(outpath), "%s_rank%02d.csv", outp_prefix, rank);
	matrix_write_csv(outpath, proj_local);

	matrix_t *projection_global = NULL;
	if (rank == 0) {
		matrix_gather_rows_to_root(proj_local, &projection_global, 0, MPI_COMM_WORLD);
		if (projection_global) {
			matrix_write_csv("results/svd_par_projection.csv", projection_global);
			matrix_free(projection_global);
		}
	} else {
		matrix_gather_rows_to_root(proj_local, NULL, 0, MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double total_time = pca_wtime() - t0;

	if (rank == 0) {
		double explained = pca_explained_ratio_from_singvals(singvals, d, k);
		printf("svd_par,%zu,%zu,%d,%d,%.6f,%.6f,%.6f,%.6f,%.8f\n",
			   global_n, d, k, size, t_read, t_center, t_pca, total_time, explained);
	}

	free(mu);
	free(singvals);
	matrix_free(X_local);
	matrix_free(V_local);
	matrix_free(proj_local);
	if (X) matrix_free(X);

	MPI_Finalize();
	return 0;
}
